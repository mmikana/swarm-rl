"""
Adaptive Skill Policy - 多技能自适应策略网络

支持两种模式：
1. 纯 PPO（quads_num_skills=1）
2. Adaptive Skill（quads_num_skills=3）

架构：
- 共享编码器（复用 Sample Factory 基类）
- 多技能头（每个输出完整的 action_distribution_params: [mean, log_std]）
- 技能选择器（加权融合多个高斯分布）
- 标准差：完全兼容 Sample Factory 的 adaptive_stddev 选项

高斯分布融合：
- 每个技能头输出一个高斯分布 N(μ_k, σ_k)
- 使用 gating 权重 w_k 进行融合
- 融合后分布：μ_final = Σ w_k × μ_k
-              σ²_final = Σ w_k × (σ²_k + μ²_k) - μ²_final
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.action_distributions import get_action_distribution


class SkillHead(nn.Module):
    """
    技能头 - 输出完整的 action_distribution_params [mean, log_std]

    与 Sample Factory 的 ActionParameterizationContinuousNonAdaptiveStddev 类似，
    但每个技能头有自己的独立参数
    """

    def __init__(self, hidden_dim: int, action_dim: int, cfg):
        super().__init__()
        self.action_dim = action_dim
        self.cfg = cfg

        # 均值网络（输出 mean）
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # 标准差参数（与 Sample Factory 一致）
        # adaptive_stddev=False: 使用独立学习的 log_std 参数
        # adaptive_stddev=True: 使用网络输出 log_std
        if not cfg.adaptive_stddev:
            # 非自适应模式：独立参数（存储 log_std）
            initial_log_std = torch.empty([action_dim])
            initial_log_std.fill_(math.log(cfg.initial_stddev))
            self.learned_log_std = nn.Parameter(initial_log_std, requires_grad=True)
            self.use_adaptive_log_std = False
        else:
            # 自适应模式：网络输出
            self.log_std_net = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )
            self.use_adaptive_log_std = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出 action_distribution_params [mean, log_std]

        返回形状：[batch, action_dim * 2]
        """
        # 1. 输出 mean
        action_means = self.mean_net(h)  # [batch, action_dim]

        # 2. 获取 log_std
        if self.use_adaptive_log_std:
            action_log_std = self.log_std_net(h)  # [batch, action_dim]
        else:
            # 独立参数 repeat 到 batch
            batch_size = h.shape[0]
            action_log_std = self.learned_log_std.repeat(batch_size, 1)

        # 3. 拼接 [mean, log_std]
        action_distribution_params = torch.cat([action_means, action_log_std], dim=-1)
        return action_distribution_params


class GatingWithTemperature(nn.Module):
    """技能选择器（带温度系数）"""

    def __init__(self, hidden_dim: int, num_skills: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_skills),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.net(h)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return weights


class AdaptiveSkillPolicy(ActorCriticSharedWeights):
    """
    自定义 Actor-Critic 模型，集成 Adaptive Skill

    设计要点：
    1. 继承 ActorCriticSharedWeights，复用所有基础组件
       - 复用 forward()
       - 复用 forward_head()
       - 复用 forward_core()
       - 复用 _maybe_sample_actions()
       - 复用 initialize_weights()
    2. 每个技能头输出完整的 [mean, log_std]
    3. 使用高斯分布融合公式融合多个技能分布
    4. 只覆写 forward_tail() 添加技能融合逻辑
    """

    def __init__(self, model_factory, obs_space, action_space, cfg):
        # 技能配置
        self.num_skills = getattr(cfg, 'quads_num_skills', 3)
        self.use_adaptive_skill = getattr(cfg, 'quads_use_adaptive_skill', True)
        self.use_diversity_loss = getattr(cfg, 'quads_use_diversity_loss', False)

        # 调用基类初始化（创建 Encoder, Core, Decoder, Critic）
        # 注意：我们覆写了 forward_tail，所以基类的 action_parameterization 不会被使用
        super().__init__(model_factory, obs_space, action_space, cfg)

        # 获取 decoder 输出维度
        decoder_out_size = self.decoder.get_out_size()
        action_dim = action_space.shape[0]

        # 技能头 - 每个输出完整的 [mean, log_std]
        self.skill_heads = nn.ModuleList([
            SkillHead(decoder_out_size, action_dim, cfg)
            for _ in range(self.num_skills)
        ])

        # 选择器（带温度）
        self.gating = GatingWithTemperature(decoder_out_size, self.num_skills)

        # 使用基类的 apply(self.initialize_weights) 进行初始化
        # 但我们需要单独初始化 skill_heads 和 gating
        self._init_custom_modules()

        # Learner 侧多样性损失和监控缓存
        self.last_diversity_loss = None
        self.last_skill_weight_mean = None
        self.last_selector_temperature = None

    def _init_custom_modules(self):
        """初始化自定义模块（skill_heads 和 gating）"""
        def init_fn(m):
            if isinstance(m, (SkillHead, GatingWithTemperature)):
                self.initialize_weights(m)
        
        self.skill_heads.apply(init_fn)
        self.gating.apply(init_fn)

    def _compute_diversity_loss(self, skill_means: torch.Tensor) -> torch.Tensor:
        """
        计算技能多样性损失。

        按当前方案，对每对 skill 的 cosine similarity 做 ReLU，
        只惩罚正相似度，避免把相反方向动作当成负奖励。
        """
        if skill_means.shape[1] < 2:
            return skill_means.new_zeros(())

        similarities = []
        for i in range(skill_means.shape[1]):
            for j in range(i + 1, skill_means.shape[1]):
                sim = F.cosine_similarity(skill_means[:, i], skill_means[:, j], dim=-1)
                similarities.append(F.relu(sim))

        if not similarities:
            return skill_means.new_zeros(())

        return torch.stack(similarities, dim=0).mean()

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        """
        覆写 forward_tail()，添加技能融合

        流程：
        1. 每个技能头输出完整的 action_distribution_params [mean_k, log_std_k]
        2. gating 输出权重 w_k
        3. 高斯分布融合：
           - μ_final = Σ w_k × μ_k
           - σ²_final = Σ w_k × (σ²_k + μ²_k) - μ²_final
        4. 创建最终分布并采样（与基类一致）
        """
        # 1. Decoder 和 Critic（完全复用基类）
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        # 2. 每个技能头输出 action_distribution_params [mean_k, log_std_k]
        skill_distribution_params = torch.stack([
            head(decoder_output) for head in self.skill_heads
        ], dim=1)  # [batch, num_skills, action_dim * 2]

        # 3. 选择器权重
        weights = self.gating(decoder_output)  # [batch, num_skills]

        # 4. 分离 mean 和 log_std
        action_dim = self.action_space.shape[0]

        # skill_distribution_params: [batch, num_skills, action_dim * 2]
        # 分离成：
        skill_means = skill_distribution_params[..., :action_dim]  # [batch, num_skills, action_dim]
        skill_log_stds = skill_distribution_params[..., action_dim:]  # [batch, num_skills, action_dim]

        # Learner 会直接读取这些缓存，不通过 rollout buffer 传递。
        self.last_skill_weight_mean = weights.detach().mean(dim=0)
        self.last_selector_temperature = self.gating.temperature.detach().mean()
        if self.training and self.use_diversity_loss and self.num_skills > 1:
            self.last_diversity_loss = self._compute_diversity_loss(skill_means)
        else:
            self.last_diversity_loss = skill_means.new_zeros(())

        # 5. 简单加权融合（保证非负）
        # 融合均值：μ_final = Σ w_k × μ_k
        mu_final = torch.sum(
            weights.unsqueeze(-1) * skill_means,
            dim=1
        )  # [batch, action_dim]

        # 融合方差：σ²_final = Σ (w_k)² × σ²_k
        skill_stds = skill_log_stds.exp()  # [batch, num_skills, action_dim]
        skill_variances = skill_stds ** 2  # σ²_k

        final_variance = torch.sum(
            (weights ** 2).unsqueeze(-1) * skill_variances,
            dim=1
        )  # [batch, action_dim]

        # 确保方差足够大（防止 log_std 过小）
        final_variance = torch.clamp(final_variance, min=0.01)  # 最小 std = 0.1

        # 转换为 log_std
        final_std = torch.sqrt(final_variance)
        final_log_std = torch.log(final_std)

        # 6. 拼接最终的 [mean, log_std]
        final_action_params = torch.cat([mu_final, final_log_std], dim=-1)

        # 7. 创建高斯分布
        final_dist = get_action_distribution(self.action_space, raw_logits=final_action_params)
        self.last_action_distribution = final_dist

        # 8. 采样动作（复用基类的 _maybe_sample_actions）
        # 基类会自动设置 result["actions"] 和 result["log_prob_actions"]
        self._maybe_sample_actions(sample_actions, result)

        # 9. 记录 action_logits（覆盖基类的）
        result["action_logits"] = final_action_params

        # 10. 记录技能信息（用于监控）
        result["skill_weights"] = weights.detach()
        result["skill_action_means"] = skill_means.detach()
        result["skill_action_log_stds"] = skill_log_stds.detach()

        return result

    def summaries(self) -> Dict:
        stats = super().summaries()

        if self.last_skill_weight_mean is not None:
            for idx, weight in enumerate(self.last_skill_weight_mean):
                stats[f"skill/weight_{idx}"] = weight

        if self.last_selector_temperature is not None:
            stats["skill/selector_temperature"] = self.last_selector_temperature

        return stats


def register_adaptive_skill_model():
    """注册到 Sample Factory 模型工厂"""
    from sample_factory.algo.utils.context import global_model_factory

    def make_adaptive_skill_model(model_factory, obs_space, action_space, cfg):
        return AdaptiveSkillPolicy(model_factory, obs_space, action_space, cfg)

    global_model_factory().make_actor_critic_func = make_adaptive_skill_model
