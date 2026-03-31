"""
Custom Learner for Adaptive Skill RL

只添加多样性损失，保留 Sample Factory 所有原有功能
"""

import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.utils.typing import Config


def get_adaptive_skill_learner_class():
    """返回 AdaptiveSkillLearner 类，用于 Sample Factory 注册"""
    return AdaptiveSkillLearner


class AdaptiveSkillLearner(Learner):
    """
    自定义 Learner，仅在原有损失基础上添加多样性损失
    """

    def __init__(
        self,
        cfg: Config,
        env_info,
        policy_versions_tensor: Tensor,
        policy_id,
        param_server,
    ):
        # 调用父类初始化，保留所有原有功能
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)

        # 多样性损失配置
        self.use_diversity_loss = getattr(cfg, 'quads_use_diversity_loss', False)
        self.diversity_loss_weight = getattr(cfg, 'quads_diversity_loss_weight', 0.1)
        self.num_skills = getattr(cfg, 'quads_num_skills', 3)

    def _calculate_losses(self, mb, num_invalids):
        """
        调用父类计算所有基础损失，然后添加多样性损失
        
        注意：diversity_loss 需要在模型的 forward_tail 中计算并保存到 buffer
        如果 mb 中没有 diversity_loss，多样性损失将为 0
        """
        from sample_factory.utils.utils import log
        
        # 调用父类计算 PPO 损失、探索损失、KL 损失、价值损失
        (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
        ) = super()._calculate_losses(mb, num_invalids)

        # 计算多样性损失
        diversity_loss = torch.tensor(0.0, device=self.device)

        if self.use_diversity_loss:
            # 从 minibatch 中获取预计算的多样性损失
            # 注意：这需要模型在 forward_tail 中计算并保存 diversity_loss
            mb_diversity_loss = mb.get('diversity_loss', None)

            if mb_diversity_loss is not None and mb_diversity_loss.numel() > 0:
                diversity_loss = mb_diversity_loss.mean()  # 取 batch 平均

                # 更新 loss_summaries 用于记录
                loss_summaries['diversity_loss'] = diversity_loss
            else:
                # 如果没有 diversity_loss，记录警告（仅一次）
                if not hasattr(self, '_warned_no_diversity_loss'):
                    log.warning(
                        "diversity_loss not found in minibatch. "
                        "Diversity loss will be 0. "
                        "Make sure the model computes and saves diversity_loss in forward_tail."
                    )
                    self._warned_no_diversity_loss = True

        # 返回时添加 diversity_loss 到 loss_summaries
        # 注意：不在这里修改 policy_loss，而是在 _train 中添加
        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
        )

    def _compute_diversity_loss(self, skill_action_means: Tensor) -> Tensor:
        """
        计算技能多样性损失

        通过余弦相似度鼓励技能差异化

        Args:
            skill_action_means: [batch, num_skills, action_dim]

        Returns:
            diversity_loss: scalar
        """
        num_skills = skill_action_means.shape[1]

        if num_skills < 2:
            return torch.tensor(0.0, device=skill_action_means.device)

        # 计算技能对之间的余弦相似度
        similarities = []
        for i in range(num_skills):
            for j in range(i + 1, num_skills):
                sim = torch.nn.functional.cosine_similarity(
                    skill_action_means[:, i],
                    skill_action_means[:, j],
                    dim=-1
                )
                similarities.append(sim)

        # 平均相似度
        if len(similarities) > 0:
            avg_similarity = torch.mean(torch.stack(similarities))
        else:
            avg_similarity = torch.tensor(0.0, device=skill_action_means.device)

        # 多样性损失 = -平均相似度（鼓励低相似度）
        diversity_loss = -avg_similarity

        return diversity_loss

    def _train(self, gpu_buffer, batch_size: int, experience_size: int, num_invalids: int):
        """
        覆写 _train 方法，在 actor_loss 中添加多样性损失
        保留所有原有逻辑
        """
        import numpy as np
        from sample_factory.algo.utils.tensor_dict import TensorDict
        from sample_factory.utils.attr_dict import AttrDict
        from sample_factory.utils.utils import log
        from sample_factory.algo.utils.torch_utils import to_scalar

        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    # ========== 添加多样性损失到 actor_loss ==========
                    diversity_loss = loss_summaries.get('diversity_loss', torch.tensor(0.0, device=self.device))
                    
                    # noinspection PyTypeChecker
                    actor_loss: Tensor = policy_loss + exploration_loss + kl_loss
                    
                    # 添加多样性损失（如果启用）
                    if self.use_diversity_loss and diversity_loss.numel() > 0:
                        actor_loss = actor_loss + self.diversity_loss_weight * diversity_loss
                    
                    critic_loss = value_loss
                    loss: Tensor = actor_loss + critic_loss

                    epoch_actor_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f div_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(loss),
                            to_scalar(policy_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                            to_scalar(kl_loss),
                            to_scalar(diversity_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # if kl_old is not None it is already calculated above
                    if kl_old is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = action_distribution.__class__(
                            self.actor_critic.action_space,
                            mb.action_logits,
                        )
                        kl_old = action_distribution.kl_divergence(old_action_distribution)
                        kl_old = kl_old  # masked_select already done in parent

                    kl_old_mean = float(kl_old.mean().item())
                    recent_kls.append(kl_old_mean)
                    if kl_old.numel() > 0 and kl_old.max().item() > 100:
                        log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

                # update the weights
                with timing.add_time("update"):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None

                    loss.backward()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time("clip"):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                    self._apply_lr(actual_lr)

                    with self.param_server.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        del summary_vars
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    from sample_factory.algo.utils.torch_utils import synchronize
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta = prev_epoch_actor_loss - new_epoch_actor_loss
            prev_epoch_actor_loss = new_epoch_actor_loss

            if loss_delta < 0:
                # actor loss increased, which is bad. Could happen due to the randomness of the training process
                # but also could be a sign of a serious problem.
                # We'll just log it and hope for the best.
                log.debug(
                    "Actor loss increased: %.4f -> %.4f. This could be a sign of a serious problem...",
                    prev_epoch_actor_loss,
                    new_epoch_actor_loss,
                )

            # check for early stopping
            if self.cfg.early_stopping:
                if abs(loss_delta) < early_stopping_tolerance:
                    log.info("Early stopping condition met. Stopping training.")
                    early_stop = True

        # end of all epochs
        return stats_and_summaries
