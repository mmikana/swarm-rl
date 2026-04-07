"""
Runtime patches that keep CBF-specific action handling isolated to CBF runs.
"""

from __future__ import annotations

import numpy as np

from sample_factory.algo.sampling import batched_sampling, non_batched_sampling
from sample_factory.algo.utils import shared_buffers

_PATCHED = False
_ORIGINAL_POLICY_OUTPUT_SHAPES = shared_buffers.policy_output_shapes


def _policy_output_shapes_with_env_actions(num_actions, num_action_distribution_parameters):
    outputs = list(_ORIGINAL_POLICY_OUTPUT_SHAPES(num_actions, num_action_distribution_parameters))
    if not any(name == "env_actions" for name, _ in outputs):
        outputs.append(("env_actions", [num_actions]))
    return outputs


def _batched_advance_rollouts_with_env_actions(self, policy_id, timing):
    with timing.add_time("process_policy_outputs"):
        self.curr_step[:] = self.policy_output_tensors
        action_key = "env_actions" if "env_actions" in self.policy_output_tensors else "actions"
        actions = batched_sampling.preprocess_actions(self.env_info, self.policy_output_tensors[action_key])

    complete_rollouts, episodic_stats = [], []

    with timing.add_time("env_step"):
        self.last_obs, rewards, terminated, truncated, infos = self.vec_env.step(actions)
        dones = terminated | truncated

    with timing.add_time("post_env_step"):
        self.policy_id_buffer[:] = self.policy_id

        rewards_cpu = rewards.cpu()
        processed_rewards = self._process_rewards(rewards, rewards_cpu)
        self.curr_step[:] = dict(
            rewards=processed_rewards,
            dones=dones,
            time_outs=truncated,
            policy_id=self.policy_id_buffer,
        )

        not_done = (1.0 - self.curr_step["dones"].float()).unsqueeze(-1)
        self.last_rnn_state = self.policy_output_tensors["new_rnn_states"] * not_done

        with timing.add_time("process_env_step"):
            stats = self._process_env_step(rewards_cpu, dones, infos)
            episodic_stats.extend(stats)

        self.rollout_step += 1
        if self.rollout_step == self.cfg.rollout:
            complete_rollouts = self._finalize_trajectories()
            self.rollout_step = 0

    return complete_rollouts, episodic_stats


def _non_batched_process_policy_outputs_with_env_actions(self, policy_id, timing):
    all_actors_ready = True

    for env_i in range(self.num_envs):
        for agent_i in range(self.num_agents):
            actor_state = self.actor_states[env_i][agent_i]
            if not actor_state.is_active:
                continue

            actor_policy = actor_state.curr_policy_id
            assert actor_policy != -1

            if actor_policy == policy_id:
                with timing.add_time("split_output_tensors"):
                    policy_outputs = np.split(
                        actor_state.policy_output_tensors,
                        indices_or_sections=actor_state.policy_output_indices,
                        axis=0,
                    )

                policy_outputs_dict = {}
                for tensor_idx, name in enumerate(actor_state.policy_output_names):
                    policy_outputs_dict[name] = policy_outputs[tensor_idx]

                actor_state.set_trajectory_data(policy_outputs_dict, self.rollout_step)
                action_key = "env_actions" if "env_actions" in policy_outputs_dict else "actions"
                actor_state.last_actions = policy_outputs_dict[action_key].squeeze()

                actor_state.last_rnn_state = policy_outputs_dict["new_rnn_states"]
                actor_state.last_value = policy_outputs_dict["values"].item()
                actor_state.ready = True
            elif not actor_state.ready:
                all_actors_ready = False

    return all_actors_ready


def enable_cbf_sample_factory_integration() -> None:
    global _PATCHED

    if _PATCHED:
        return

    shared_buffers.policy_output_shapes = _policy_output_shapes_with_env_actions
    batched_sampling.BatchedVectorEnvRunner.advance_rollouts = _batched_advance_rollouts_with_env_actions
    non_batched_sampling.NonBatchedVectorEnvRunner._process_policy_outputs = (
        _non_batched_process_policy_outputs_with_env_actions
    )
    _PATCHED = True
