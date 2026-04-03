"""
Custom Learner for Adaptive Skill RL.

Adds learner-side diversity regularization without touching rollout storage.
"""

import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.torch_utils import to_scalar
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log


def get_adaptive_skill_learner_class():
    """Return the AdaptiveSkillLearner class for registration."""
    return AdaptiveSkillLearner


class AdaptiveSkillLearner(Learner):
    """
    Minimal Learner extension for Adaptive Skill.

    - model computes and caches the current minibatch diversity loss
    - learner injects it into actor loss by adjusting policy_loss
    - summaries are recorded in TensorBoard through the normal SF path
    """

    def __init__(
        self,
        cfg: Config,
        env_info,
        policy_versions_tensor: Tensor,
        policy_id,
        param_server,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)

        self.use_diversity_loss = getattr(cfg, "quads_use_diversity_loss", False)
        self.diversity_loss_weight = getattr(
            cfg,
            "diversity_loss_weight",
            getattr(cfg, "quads_diversity_loss_weight", 0.0),
        )
        self._warned_missing_diversity_cache = False

    def _get_cached_diversity_loss(self, reference_tensor: Tensor) -> Tensor:
        diversity_loss = getattr(self.actor_critic, "last_diversity_loss", None)
        if diversity_loss is not None:
            return diversity_loss

        if self.use_diversity_loss and not self._warned_missing_diversity_cache:
            log.warning(
                "AdaptiveSkillPolicy did not expose last_diversity_loss. "
                "Diversity regularization will be 0 for this run."
            )
            self._warned_missing_diversity_cache = True

        return reference_tensor.new_zeros(())

    def _calculate_losses(self, mb, num_invalids):
        (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
        ) = super()._calculate_losses(mb, num_invalids)

        policy_loss_raw = policy_loss
        diversity_loss = policy_loss.new_zeros(())
        diversity_loss_weighted = policy_loss.new_zeros(())

        if self.use_diversity_loss:
            diversity_loss = self._get_cached_diversity_loss(policy_loss)
            diversity_loss_weighted = self.diversity_loss_weight * diversity_loss
            policy_loss = policy_loss + diversity_loss_weighted

        loss_summaries["policy_loss_raw"] = policy_loss_raw
        loss_summaries["diversity_loss"] = diversity_loss
        loss_summaries["diversity_loss_weighted"] = diversity_loss_weighted

        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
        )

    def _record_summaries(self, train_loop_vars):
        stats = super()._record_summaries(train_loop_vars)

        policy_loss_raw = getattr(train_loop_vars, "policy_loss_raw", train_loop_vars.policy_loss)
        diversity_loss = getattr(train_loop_vars, "diversity_loss", 0.0)
        diversity_loss_weighted = getattr(train_loop_vars, "diversity_loss_weighted", 0.0)

        # Preserve the standard SF policy_loss metric as raw PPO policy loss.
        stats.policy_loss = to_scalar(policy_loss_raw)
        stats["loss/policy_loss_total"] = to_scalar(train_loop_vars.policy_loss)
        stats["loss/diversity_loss"] = to_scalar(diversity_loss)
        stats["loss/diversity_loss_weighted"] = to_scalar(diversity_loss_weighted)

        return stats
