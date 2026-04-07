"""
Learner extensions that let the CBF layer shape policy updates.
"""

from __future__ import annotations

from typing import Type

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.torch_utils import to_scalar
from sample_factory.utils.utils import log

_CBF_LEARNER_CACHE: dict[Type[Learner], Type[Learner]] = {}


class CBFLearnerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cbf = getattr(self.cfg, "quads_use_cbf", False)
        self.cbf_loss_weight = getattr(self.cfg, "quads_cbf_loss_weight", 0.1)
        self.cbf_projection_loss_weight = getattr(self.cfg, "quads_cbf_projection_loss_weight", 0.25)
        self._warned_missing_cbf_cache = False

    def _get_cached_cbf_metric(self, attr_name, reference_tensor):
        metric = getattr(self.actor_critic, attr_name, None)
        if metric is not None:
            return metric

        if self.use_cbf and not self._warned_missing_cbf_cache:
            log.warning(
                "CBF actor-critic did not expose cached CBF metrics. "
                "CBF learner regularization will be disabled for this run."
            )
            self._warned_missing_cbf_cache = True

        return reference_tensor.new_zeros(())

    def _calculate_losses(self, mb, num_invalids):
        if self.use_cbf:
            setattr(self.actor_critic, "_cbf_forward_obs", mb.normalized_obs.get("obs", None))

        try:
            (
                action_distribution,
                policy_loss,
                exploration_loss,
                kl_old,
                kl_loss,
                value_loss,
                loss_summaries,
            ) = super()._calculate_losses(mb, num_invalids)
        finally:
            if hasattr(self.actor_critic, "_cbf_forward_obs"):
                delattr(self.actor_critic, "_cbf_forward_obs")

        policy_loss_pre_cbf = policy_loss
        cbf_projection_loss = policy_loss.new_zeros(())
        cbf_violation_loss = policy_loss.new_zeros(())
        cbf_loss = policy_loss.new_zeros(())

        if self.use_cbf:
            cbf_projection_loss = self._get_cached_cbf_metric("last_cbf_projection_loss", policy_loss)
            cbf_violation_loss = self._get_cached_cbf_metric("last_cbf_violation_loss", policy_loss)
            cbf_loss = self.cbf_loss_weight * (
                cbf_violation_loss + self.cbf_projection_loss_weight * cbf_projection_loss
            )
            policy_loss = policy_loss + cbf_loss

        loss_summaries["policy_loss_pre_cbf"] = policy_loss_pre_cbf
        loss_summaries["cbf_projection_loss"] = cbf_projection_loss
        loss_summaries["cbf_violation_loss"] = cbf_violation_loss
        loss_summaries["cbf_loss"] = cbf_loss

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

        stats["loss/policy_loss_pre_cbf"] = to_scalar(getattr(train_loop_vars, "policy_loss_pre_cbf", 0.0))
        stats["loss/cbf_projection_loss"] = to_scalar(getattr(train_loop_vars, "cbf_projection_loss", 0.0))
        stats["loss/cbf_violation_loss"] = to_scalar(getattr(train_loop_vars, "cbf_violation_loss", 0.0))
        stats["loss/cbf_loss"] = to_scalar(getattr(train_loop_vars, "cbf_loss", 0.0))
        stats["loss/policy_loss_total"] = to_scalar(train_loop_vars.policy_loss)
        return stats


def get_cbf_learner_class(base_cls: Type[Learner] = Learner) -> Type[Learner]:
    if issubclass(base_cls, CBFLearnerMixin):
        return base_cls

    if base_cls in _CBF_LEARNER_CACHE:
        return _CBF_LEARNER_CACHE[base_cls]

    class CBFLearner(CBFLearnerMixin, base_cls):
        pass

    CBFLearner.__name__ = f"CBF{base_cls.__name__}"
    _CBF_LEARNER_CACHE[base_cls] = CBFLearner
    return CBFLearner
