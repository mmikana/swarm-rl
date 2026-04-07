"""
Quad actor-critic variants with a differentiable CBF layer.
"""

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCriticSeparateWeights, ActorCriticSharedWeights
from swarm_rl.cbf.cbf_layer import DistanceAwareCBFLayer


def _get_flat_obs_dim(obs_space) -> int:
    obs_subspace = obs_space.spaces["obs"] if hasattr(obs_space, "spaces") else obs_space
    return int(np.prod(obs_subspace.shape))


def _extract_nominal_action(action_distribution, sample_actions: bool) -> tuple[Tensor, Tensor | None]:
    if sample_actions:
        raw_actions = action_distribution.sample()
        log_prob_actions = action_distribution.log_prob(raw_actions)
    else:
        raw_actions = getattr(action_distribution, "means", action_distribution.mean)
        log_prob_actions = None

    return raw_actions, log_prob_actions


class _CBFModelMixin:
    def _validate_cbf_requirements(self, cfg) -> None:
        use_cbf = getattr(cfg, "quads_use_cbf", False)
        use_obstacles = getattr(cfg, "quads_use_obstacles", False)

        if use_cbf and not use_obstacles:
            raise ValueError(
                "CBF requires obstacles to be enabled. Use --quads_use_obstacles=True when --quads_use_cbf=True."
            )
        if use_cbf and getattr(cfg, "normalize_input", False):
            raise ValueError(
                "CBF requires raw physical observations, but normalize_input=True would feed normalized "
                "velocities, rotations, and SDF values into the constraint. Use --normalize_input=False."
            )

    def _init_cbf_support(self, obs_space, cfg) -> None:
        self.use_cbf = getattr(cfg, "quads_use_cbf", False)

        self.last_cbf_projection_loss = None
        self.last_cbf_violation_loss = None
        self.last_cbf_action_delta = None
        self.last_cbf_intervention_rate = None
        self.last_cbf_margin = None
        self.last_cbf_safe_violation = None
        self.last_cbf_qp_failure_rate = None

        if not self.use_cbf:
            return

        from gym_art.quadrotor_multi.quad_utils import QUADS_NEIGHBOR_OBS_TYPE, QUADS_OBS_REPR

        self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        if cfg.quads_neighbor_visible_num == -1:
            self.num_neighbors = cfg.quads_num_agents - 1
        else:
            self.num_neighbors = cfg.quads_neighbor_visible_num

        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_neighbors

        obs_dim = _get_flat_obs_dim(obs_space)
        expected_obs_dim = self.self_obs_dim + self.all_neighbor_obs_dim + 9
        if obs_dim < expected_obs_dim:
            raise ValueError(
                f"Observation dimension ({obs_dim}) is too small for CBF. "
                f"CBF requires at least {expected_obs_dim} dimensions for self/neighbor/SDF observations."
            )

        self.cbf_layer = DistanceAwareCBFLayer(
            alpha_cbf=getattr(cfg, "quads_cbf_alpha", 1.0),
            k=getattr(cfg, "quads_cbf_k", 2.0),
            sigma=getattr(cfg, "quads_cbf_sigma", 0.1),
        )

        expected_total_dim = self.self_obs_dim + self.all_neighbor_obs_dim + 9
        if obs_dim != expected_total_dim:
            print(
                "Warning: Observation dimension mismatch. "
                f"Expected {expected_total_dim} (self:{self.self_obs_dim} + "
                f"neighbors:{self.all_neighbor_obs_dim} + sdf:9), got {obs_dim}. "
                "CBF may not work correctly."
            )

    @staticmethod
    def _extract_state_from_obs(obs: Tensor) -> Dict[str, Tensor]:
        vel = obs[:, 3:6]
        rot_flat = obs[:, 6:15]
        R = rot_flat.reshape(-1, 3, 3)
        return {"vel": vel, "R": R}

    def _extract_sdf_from_obs(self, obs: Tensor) -> Tensor:
        sdf_start = self.self_obs_dim + self.all_neighbor_obs_dim
        return obs[:, sdf_start : sdf_start + 9]

    def _resolve_cbf_obs(self, obs: Tensor | None) -> Tensor | None:
        if obs is not None:
            return obs
        return getattr(self, "_cbf_forward_obs", None)

    def _reset_cbf_cache(self, reference: Tensor) -> None:
        zero = reference.new_zeros(())
        self.last_cbf_projection_loss = zero
        self.last_cbf_violation_loss = zero
        self.last_cbf_action_delta = zero
        self.last_cbf_intervention_rate = zero
        self.last_cbf_margin = zero
        self.last_cbf_safe_violation = zero
        self.last_cbf_qp_failure_rate = zero

    def _apply_cbf(self, nominal_actions: Tensor, obs: Tensor | None) -> tuple[Tensor, Tensor]:
        cbf_input_actions = torch.clamp(nominal_actions, min=-1.0, max=1.0)
        self._reset_cbf_cache(cbf_input_actions)

        if not self.use_cbf:
            return nominal_actions, cbf_input_actions

        obs = self._resolve_cbf_obs(obs)
        if obs is None:
            return nominal_actions, cbf_input_actions

        state = self._extract_state_from_obs(obs)
        sdf_obs = self._extract_sdf_from_obs(obs)
        safe_actions, cbf_info = self.cbf_layer.project_with_info(cbf_input_actions, state, sdf_obs)

        omega_delta = safe_actions[:, 1:4] - cbf_input_actions[:, 1:4]
        self.last_cbf_projection_loss = omega_delta.square().sum(dim=1).mean()
        self.last_cbf_violation_loss = cbf_info["violation_nominal"].square().mean()
        self.last_cbf_action_delta = omega_delta.abs().mean()
        self.last_cbf_intervention_rate = cbf_info["active_mask"].float().mean()
        self.last_cbf_margin = cbf_info["margin_nominal"].mean()
        self.last_cbf_safe_violation = cbf_info["violation_safe"].mean()
        self.last_cbf_qp_failure_rate = cbf_info["qp_failed"].to(dtype=cbf_input_actions.dtype)

        return nominal_actions, torch.clamp(safe_actions, min=-1.0, max=1.0)

    def summaries(self) -> Dict:
        stats = super().summaries()
        if not self.use_cbf or self.last_cbf_projection_loss is None:
            return stats

        stats["cbf/projection_loss"] = self.last_cbf_projection_loss
        stats["cbf/violation_loss"] = self.last_cbf_violation_loss
        stats["cbf/action_delta_mean"] = self.last_cbf_action_delta
        stats["cbf/intervention_rate"] = self.last_cbf_intervention_rate
        stats["cbf/nominal_margin_mean"] = self.last_cbf_margin
        stats["cbf/safe_violation_mean"] = self.last_cbf_safe_violation
        stats["cbf/qp_failure_rate"] = self.last_cbf_qp_failure_rate
        return stats


class QuadActorCriticWithCBF(_CBFModelMixin, ActorCriticSharedWeights):
    def __init__(self, model_factory, obs_space, action_space, cfg):
        self._validate_cbf_requirements(cfg)
        super().__init__(model_factory, obs_space, action_space, cfg)
        self._init_cbf_support(obs_space, cfg)

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool, obs=None) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output)
        result["action_logits"] = action_distribution_params

        nominal_actions, log_prob_actions = _extract_nominal_action(self.last_action_distribution, sample_actions)
        nominal_actions, env_actions = self._apply_cbf(nominal_actions, obs)

        result["actions"] = nominal_actions
        result["env_actions"] = env_actions
        if log_prob_actions is not None:
            result["log_prob_actions"] = log_prob_actions

        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)

        obs = normalized_obs_dict.get("obs", None)
        result = self.forward_tail(x, values_only, sample_actions=True, obs=obs)
        result["new_rnn_states"] = new_rnn_states
        return result


class QuadActorCriticWithCBFSeparate(_CBFModelMixin, ActorCriticSeparateWeights):
    def __init__(self, model_factory, obs_space, action_space, cfg):
        self._validate_cbf_requirements(cfg)
        super().__init__(model_factory, obs_space, action_space, cfg)
        self._init_cbf_support(obs_space, cfg)

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool, obs=None) -> TensorDict:
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        critic_decoder_output = self.critic_decoder(core_outputs[1])
        values = self.critic_linear(critic_decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        actor_decoder_output = self.actor_decoder(core_outputs[0])
        action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_decoder_output)
        result["action_logits"] = action_distribution_params

        nominal_actions, log_prob_actions = _extract_nominal_action(self.last_action_distribution, sample_actions)
        nominal_actions, env_actions = self._apply_cbf(nominal_actions, obs)

        result["actions"] = nominal_actions
        result["env_actions"] = env_actions
        if log_prob_actions is not None:
            result["log_prob_actions"] = log_prob_actions

        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)

        obs = normalized_obs_dict.get("obs", None)
        result = self.forward_tail(x, values_only, sample_actions=True, obs=obs)
        result["new_rnn_states"] = new_rnn_states
        return result
