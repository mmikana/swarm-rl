"""
Runtime compatibility patch for Sample Factory GPU training.

Sample Factory sends the learner state_dict to inference workers through the
signal_slot event loop during initialization. When that payload contains CUDA
storages, some multi-process runs fail with:

    RuntimeError: CUDA error: invalid resource handle

To avoid CUDA IPC handles in control-plane messages, this module keeps a shared
CPU copy of the learner weights for cross-process initialization and async
weight refreshes. Inference workers still run their local model on GPU and load
from the shared CPU snapshot when the policy version changes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import MutableMapping

import torch

from sample_factory.utils.utils import log

_PATCHED = False
_PROCESS_WRAPPERS_PATCHED = False
_ORIGINAL_INFERENCE_INIT = None
_ORIGINAL_LEARNER_INIT = None


def _as_torch_device(device) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    try:
        return torch.device(device)
    except (TypeError, RuntimeError):
        return None


def _needs_cpu_weight_bridge(cfg, device) -> bool:
    torch_device = _as_torch_device(device)
    return (
        torch_device is not None
        and torch_device.type == "cuda"
        and getattr(cfg, "device", None) == "gpu"
        and not getattr(cfg, "serial_mode", False)
    )


def _clone_tensor_to_shared_cpu(tensor: torch.Tensor) -> torch.Tensor:
    cpu_tensor = tensor.detach().to(device="cpu", copy=True)
    cpu_tensor.share_memory_()
    return cpu_tensor


def _copy_state_dict_to_shared_cpu(state_dict) -> OrderedDict:
    shared_state = OrderedDict()
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            shared_state[name] = _clone_tensor_to_shared_cpu(value)
        else:
            shared_state[name] = value
    return shared_state


def _sync_state_dict_to_shared_cpu(module, shared_state: MutableMapping) -> MutableMapping:
    current_state = module.state_dict()
    stale_keys = [name for name in shared_state.keys() if name not in current_state]
    for name in stale_keys:
        del shared_state[name]

    for name, value in current_state.items():
        if not torch.is_tensor(value):
            shared_state[name] = value
            continue

        shared_value = shared_state.get(name)
        if (
            shared_value is None
            or not torch.is_tensor(shared_value)
            or shared_value.shape != value.shape
            or shared_value.dtype != value.dtype
        ):
            shared_state[name] = _clone_tensor_to_shared_cpu(value)
        else:
            shared_value.copy_(value.detach(), non_blocking=False)

    return shared_state


def _get_or_create_shared_cpu_state_dict(actor_critic) -> MutableMapping:
    shared_state = getattr(actor_critic, "_sf_shared_cpu_state_dict", None)
    if shared_state is None:
        shared_state = _copy_state_dict_to_shared_cpu(actor_critic.state_dict())
        actor_critic._sf_shared_cpu_state_dict = shared_state
    else:
        _sync_state_dict_to_shared_cpu(actor_critic, shared_state)
    return shared_state


def init_inference_process_with_gpu_compat(sf_context, worker):
    enable_sample_factory_gpu_compat()
    _ORIGINAL_INFERENCE_INIT(sf_context, worker)


def init_learner_process_with_gpu_compat(sf_context, learner_worker):
    enable_sample_factory_gpu_compat()
    _ORIGINAL_LEARNER_INIT(sf_context, learner_worker)


def _patch_process_entrypoints() -> None:
    global _PROCESS_WRAPPERS_PATCHED, _ORIGINAL_INFERENCE_INIT, _ORIGINAL_LEARNER_INIT

    if _PROCESS_WRAPPERS_PATCHED:
        return

    import sample_factory.algo.learning.learner_worker as sf_learner_worker
    import sample_factory.algo.runners.runner_parallel as sf_runner_parallel
    import sample_factory.algo.sampling.inference_worker as sf_inference_worker
    import sample_factory.algo.sampling.sampler as sf_sampler

    _ORIGINAL_INFERENCE_INIT = sf_inference_worker.init_inference_process
    _ORIGINAL_LEARNER_INIT = sf_learner_worker.init_learner_process

    sf_inference_worker.init_inference_process = init_inference_process_with_gpu_compat
    sf_sampler.init_inference_process = init_inference_process_with_gpu_compat

    sf_learner_worker.init_learner_process = init_learner_process_with_gpu_compat
    sf_runner_parallel.init_learner_process = init_learner_process_with_gpu_compat

    _PROCESS_WRAPPERS_PATCHED = True


def enable_sample_factory_gpu_compat() -> None:
    global _PATCHED

    _patch_process_entrypoints()

    if _PATCHED:
        return

    from sample_factory.algo.learning import learner as sf_learner

    original_model_initialization_data = sf_learner.model_initialization_data
    original_after_optimizer_step = sf_learner.Learner._after_optimizer_step

    def patched_model_initialization_data(cfg, policy_id, actor_critic, policy_version, device):
        if not _needs_cpu_weight_bridge(cfg, device):
            return original_model_initialization_data(cfg, policy_id, actor_critic, policy_version, device)

        shared_state = _get_or_create_shared_cpu_state_dict(actor_critic)
        if not getattr(actor_critic, "_sf_logged_shared_cpu_bridge", False):
            actor_critic._sf_logged_shared_cpu_bridge = True
            log.info("Using shared CPU model snapshot for GPU policy initialization")
        return policy_id, shared_state, device, policy_version

    def patched_after_optimizer_step(self):
        original_after_optimizer_step(self)

        if not _needs_cpu_weight_bridge(self.cfg, self.device):
            return

        shared_state = getattr(self.actor_critic, "_sf_shared_cpu_state_dict", None)
        if shared_state is None:
            return

        with torch.no_grad(), self.param_server.policy_lock:
            _sync_state_dict_to_shared_cpu(self.actor_critic, shared_state)

    sf_learner.model_initialization_data = patched_model_initialization_data
    sf_learner.Learner._after_optimizer_step = patched_after_optimizer_step
    _PATCHED = True
    log.info("Enabled Sample Factory GPU multiprocessing compatibility patch")
