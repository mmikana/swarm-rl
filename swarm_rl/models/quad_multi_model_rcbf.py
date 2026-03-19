"""
QuadActorCriticWithCBF - 集成CBF安全层的Actor-Critic模型

支持两种架构:
1. QuadActorCriticWithCBF: 继承 ActorCriticSharedWeights (共享权重)
2. QuadActorCriticWithCBFSeparate: 继承 ActorCriticSeparateWeights (独立权重)
"""

import torch
import numpy as np
from typing import Dict
from torch import Tensor

from sample_factory.model.actor_critic import ActorCriticSharedWeights, ActorCriticSeparateWeights
from sample_factory.algo.utils.tensor_dict import TensorDict
from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer


class QuadActorCriticWithCBF(ActorCriticSharedWeights):
    """
    自定义 Actor-Critic 模型，集成 CBF-QP 安全层

    设计要点：
    1. 继承 ActorCriticSharedWeights，复用所有基础组件
    2. 只覆写 forward() 和 forward_tail()，添加 CBF-QP 层
    3. 通过 cfg.quads_use_cbf 控制是否启用CBF
    4. 训练时：log_prob 基于 u_rl，但环境执行 u_safe
    5. 推理时：直接使用 u_safe
    """

    def __init__(self, model_factory, obs_space, action_space, cfg):
        # CBF 配置检查（在调用基类之前）
        use_cbf = getattr(cfg, 'quads_use_cbf', False)
        use_obstacles = getattr(cfg, 'quads_use_obstacles', False)
        
        if use_cbf and not use_obstacles:
            raise ValueError(
                "CBF requires obstacles to be enabled! "
                "Please add --quads_use_obstacles=True to your training command.\n"
                "\nExample:\n"
                "  python -m swarm_rl.train --algo=APPO --env=quadrotor_multi \\\n"
                "      --quads_use_cbf=True --quads_use_obstacles=True \\\n"
                "      --train_for_env_steps=1000"
            )
        
        # 调用基类初始化
        super().__init__(model_factory, obs_space, action_space, cfg)

        # CBF 开关和参数
        self.use_cbf = use_cbf

        if self.use_cbf:
            # 验证观测空间包含 SDF（9 维）
            # 从cfg计算观测维度,而不是从obs_space
            from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
            self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

            if cfg.quads_neighbor_visible_num == -1:
                self.num_neighbors = cfg.quads_num_agents - 1
            else:
                self.num_neighbors = cfg.quads_neighbor_visible_num

            self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_neighbors

            # 计算总观测维度
            obs_dim = self.self_obs_dim + self.all_neighbor_obs_dim + 9  # +9 for SDF

            if obs_dim < self.self_obs_dim + 9:
                raise ValueError(
                    f"Observation dimension ({obs_dim}) is too small for CBF. "
                    f"CBF requires at least {self.self_obs_dim + 9} dimensions for SDF observations."
                )

            # 创建 CBF-QP 层
            self.cbf_layer = QuadCBFQPLayer(
                mass=getattr(cfg, 'quads_mass', 0.028),
                thrust_to_weight=getattr(cfg, 'quads_thrust_to_weight', 3.0),
                alpha_1=getattr(cfg, 'quads_cbf_alpha_1', 1.0),
                alpha_2=getattr(cfg, 'quads_cbf_alpha_2', 1.0),
                k_omega=getattr(cfg, 'quads_cbf_k_omega', 0.1),
                R_obs=getattr(cfg, 'quads_cbf_R_obs', 0.5),
                epsilon=getattr(cfg, 'quads_cbf_epsilon', 0.1),
                sdf_resolution=getattr(cfg, 'quads_cbf_sdf_resolution', 0.1),
            )

            # 验证 SDF 观测位置
            expected_sdf_start = self.self_obs_dim + self.all_neighbor_obs_dim
            expected_total_dim = expected_sdf_start + 9
            if obs_dim != expected_total_dim:
                print(f"Warning: Observation dimension mismatch. "
                      f"Expected {expected_total_dim} (self:{self.self_obs_dim} + "
                      f"neighbors:{self.all_neighbor_obs_dim} + sdf:9), got {obs_dim}. "
                      f"CBF may not work correctly.")

    def _extract_state_from_obs(self, obs):
        """
        从观测中提取状态信息（用于CBF约束计算）

        观测结构：[pos_rel(3), vel(3), rot(9), omega(3), ...]
        注意：pos_rel 是相对于目标的位置，CBF 不需要全局位置（SDF 已包含空间信息）

        Args:
            obs: (batch_size, obs_dim) tensor

        Returns:
            dict with 'vel', 'rot', 'omega'
        """
        # 提取速度 vel (index 3:6)
        vel = obs[:, 3:6]

        # 提取旋转矩阵 rot (index 6:15)
        rot_flat = obs[:, 6:15]
        rot = rot_flat.reshape(-1, 3, 3)

        # 提取角速度 omega (index 15:18)
        omega = obs[:, 15:18]

        return {'vel': vel, 'rot': rot, 'omega': omega}

    def _extract_sdf_from_obs(self, obs):
        """
        从观测中提取 SDF 信息

        观测结构：[self_obs, neighbor_obs, obstacle_obs(SDF 9维)]

        Args:
            obs: (batch_size, obs_dim) tensor

        Returns:
            sdf_obs: (batch_size, 9) tensor
        """
        # SDF 在观测的最后 9 维
        sdf_start = self.self_obs_dim + self.all_neighbor_obs_dim
        sdf_obs = obs[:, sdf_start:sdf_start + 9]
        return sdf_obs

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool, obs=None) -> TensorDict:
        """
        覆写 forward_tail()，在采样动作后添加 CBF-QP 层

        Args:
            core_output: encoder + core 的输出
            values_only: 是否只计算 value
            sample_actions: 是否采样动作
            obs: 原始观测（用于提取状态和SDF信息）

        Returns:
            TensorDict with 'values', 'actions', 'action_logits', 'log_prob_actions'
        """
        # 1. Decoder 和 Critic（复用基类）
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        # 2. Policy 输出 action_logits（复用基类）
        action_distribution_params, self.last_action_distribution = \
            self.action_parameterization(decoder_output)
        result["action_logits"] = action_distribution_params

        # 3. 采样动作 u_rl（标称控制）
        if sample_actions:
            # 使用基类方法采样
            actions = self.last_action_distribution.sample()
            log_prob_actions = self.last_action_distribution.log_prob(actions)
            u_rl = actions
        else:
            # 推理时使用 mean
            u_rl = action_distribution_params
            log_prob_actions = None
            actions = action_distribution_params

        # 4. CBF-QP 层（如果启用）
        if self.use_cbf and obs is not None:
            # 提取状态和 SDF
            state = self._extract_state_from_obs(obs)
            sdf_obs = self._extract_sdf_from_obs(obs)

            # 调用 CBF-QP 层计算安全动作
            try:
                u_final = self.cbf_layer(state, u_rl, sdf_obs)
            except Exception as e:
                # 如果 CBF 失败，回退到原始动作
                print(f"Warning: CBF-QP failed: {e}, using u_rl")
                u_final = u_rl

            # 记录 CBF 信息（用于调试和分析）
            result["u_rl"] = u_rl
            result["u_safe"] = u_final
        else:
            u_final = u_rl

        # 5. 输出最终动作
        # 注意：log_prob 是基于 u_rl 计算的（APPO-CBF 的关键）
        result["actions"] = u_final
        if log_prob_actions is not None:
            result["log_prob_actions"] = log_prob_actions

        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        """
        覆写 forward()，传递原始观测给 forward_tail()

        Args:
            normalized_obs_dict: dict with 'obs' key
            rnn_states: RNN 状态
            values_only: 是否只计算 value

        Returns:
            TensorDict
        """
        # 1. Encoder
        x = self.forward_head(normalized_obs_dict)

        # 2. Core (RNN)
        x, new_rnn_states = self.forward_core(x, rnn_states)

        # 3. Decoder + Policy + CBF (传递原始观测)
        obs = normalized_obs_dict.get('obs', None)
        result = self.forward_tail(x, values_only, sample_actions=True, obs=obs)
        result["new_rnn_states"] = new_rnn_states

        return result


class QuadActorCriticWithCBFSeparate(ActorCriticSeparateWeights):
    """
    自定义 Actor-Critic 模型 (Separate Weights)，集成 CBF-QP 安全层

    设计要点：
    1. 继承 ActorCriticSeparateWeights，actor和critic使用独立的encoder/core
    2. 只覆写 forward_tail()，在actor输出后添加 CBF-QP 层
    3. CBF层的初始化和状态提取逻辑与 QuadActorCriticWithCBF 相同
    """

    def __init__(self, model_factory, obs_space, action_space, cfg):
        # CBF 配置检查（在调用基类之前）
        use_cbf = getattr(cfg, 'quads_use_cbf', False)
        use_obstacles = getattr(cfg, 'quads_use_obstacles', False)

        if use_cbf and not use_obstacles:
            raise ValueError(
                "CBF requires obstacles to be enabled! "
                "Please add --quads_use_obstacles=True to your training command."
            )

        # 调用基类初始化
        super().__init__(model_factory, obs_space, action_space, cfg)

        # CBF 开关和参数
        self.use_cbf = use_cbf

        if self.use_cbf:
            # 验证观测空间包含 SDF（9 维）
            # 从cfg计算观测维度,而不是从obs_space
            from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
            self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

            if cfg.quads_neighbor_visible_num == -1:
                self.num_neighbors = cfg.quads_num_agents - 1
            else:
                self.num_neighbors = cfg.quads_neighbor_visible_num

            self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_neighbors

            # 计算总观测维度
            obs_dim = self.self_obs_dim + self.all_neighbor_obs_dim + 9  # +9 for SDF

            if obs_dim < self.self_obs_dim + 9:
                raise ValueError(
                    f"Observation dimension ({obs_dim}) is too small for CBF. "
                    f"CBF requires at least {self.self_obs_dim + 9} dimensions for SDF observations."
                )

            # 创建 CBF-QP 层
            self.cbf_layer = QuadCBFQPLayer(
                mass=getattr(cfg, 'quads_mass', 0.028),
                thrust_to_weight=getattr(cfg, 'quads_thrust_to_weight', 3.0),
                alpha_1=getattr(cfg, 'quads_cbf_alpha_1', 1.0),
                alpha_2=getattr(cfg, 'quads_cbf_alpha_2', 1.0),
                k_omega=getattr(cfg, 'quads_cbf_k_omega', 0.1),
                R_obs=getattr(cfg, 'quads_cbf_R_obs', 0.5),
                epsilon=getattr(cfg, 'quads_cbf_epsilon', 0.1),
                sdf_resolution=getattr(cfg, 'quads_cbf_sdf_resolution', 0.1),
            )

            # 验证 SDF 观测位置
            expected_sdf_start = self.self_obs_dim + self.all_neighbor_obs_dim
            expected_total_dim = expected_sdf_start + 9
            if obs_dim != expected_total_dim:
                print(f"Warning: Observation dimension mismatch. "
                      f"Expected {expected_total_dim} (self:{self.self_obs_dim} + "
                      f"neighbors:{self.all_neighbor_obs_dim} + sdf:9), got {obs_dim}. "
                      f"CBF may not work correctly.")

    def _extract_state_from_obs(self, obs):
        """从观测中提取状态信息（用于CBF约束计算）"""
        vel = obs[:, 3:6]
        rot_flat = obs[:, 6:15]
        rot = rot_flat.reshape(-1, 3, 3)
        omega = obs[:, 15:18]
        return {'vel': vel, 'rot': rot, 'omega': omega}

    def _extract_sdf_from_obs(self, obs):
        """从观测中提取 SDF 信息"""
        sdf_start = self.self_obs_dim + self.all_neighbor_obs_dim
        sdf_obs = obs[:, sdf_start:sdf_start + 9]
        return sdf_obs

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool, obs=None) -> TensorDict:
        """
        覆写 forward_tail()，在actor采样动作后添加 CBF-QP 层

        注意：ActorCriticSeparateWeights 的 core_output 包含 [actor_core_output, critic_core_output]
        """
        # 1. 分离 actor 和 critic 的 core 输出
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        # 2. Critic 分支（与基类相同）
        critic_decoder_output = self.critic_decoder(core_outputs[1])
        values = self.critic_linear(critic_decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        # 3. Actor 分支：Policy 输出 action_logits
        actor_decoder_output = self.actor_decoder(core_outputs[0])
        action_distribution_params, self.last_action_distribution = \
            self.action_parameterization(actor_decoder_output)
        result["action_logits"] = action_distribution_params

        # 4. 采样动作 u_rl（标称控制）
        if sample_actions:
            actions = self.last_action_distribution.sample()
            log_prob_actions = self.last_action_distribution.log_prob(actions)
            u_rl = actions
        else:
            # 推理时使用 mean
            u_rl = action_distribution_params
            log_prob_actions = None
            actions = action_distribution_params

        # 5. CBF-QP 层（如果启用）
        if self.use_cbf and obs is not None:
            # 提取状态和 SDF
            state = self._extract_state_from_obs(obs)
            sdf_obs = self._extract_sdf_from_obs(obs)

            # 调用 CBF-QP 层计算安全动作
            try:
                u_final = self.cbf_layer(state, u_rl, sdf_obs)
            except Exception as e:
                # 如果 CBF 失败，回退到原始动作
                print(f"Warning: CBF-QP failed: {e}, using u_rl")
                u_final = u_rl

            # 记录 CBF 信息
            result["u_rl"] = u_rl
            result["u_safe"] = u_final
        else:
            u_final = u_rl

        # 6. 输出最终动作
        # 注意：log_prob 是基于 u_rl 计算的
        result["actions"] = u_final
        if log_prob_actions is not None:
            result["log_prob_actions"] = log_prob_actions

        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        """
        覆写 forward()，传递原始观测给 forward_tail()
        """
        # 1. Encoder (actor + critic)
        x = self.forward_head(normalized_obs_dict)

        # 2. Core (RNN)
        x, new_rnn_states = self.forward_core(x, rnn_states)

        # 3. Decoder + Policy + CBF (传递原始观测)
        obs = normalized_obs_dict.get('obs', None)
        result = self.forward_tail(x, values_only, sample_actions=True, obs=obs)
        result["new_rnn_states"] = new_rnn_states

        return result
