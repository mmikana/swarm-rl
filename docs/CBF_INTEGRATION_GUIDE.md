"""
CBF 集成到 Sample Factory 训练指南

两种集成方式：
1. 使用现有的 QuadActorCriticWithCBF（基于 QuadCBFQPLayer）
2. 使用新的 QuadActorCriticWithDistanceCBF（基于 DistanceAwareCBFLayer）
"""

# ============ 方式 1：使用现有 CBF 模型 ============

# 训练命令：
# python -m sample_factory.launcher.run \
#     --run=swarm_rl.runs.obstacles.quads_multi_obstacles \
#     --quads_use_cbf=True \
#     --quads_use_obstacles=True \
#     --quads_cbf_alpha_1=1.0 \
#     --quads_cbf_alpha_2=1.0 \
#     --quads_cbf_safety_allowance=0.1 \
#     --num_gpus=1

# 特点：
# - 使用 QuadCBFQPLayer（基于加速度约束）
# - 需要 cvxpy 求解器
# - 支持训练和推理


# ============ 方式 2：使用新的距离感知 CBF 模型 ============

# 1. 在 train.py 中添加新的工厂函数：

def make_actor_critic_with_distance_cbf(cfg, obs_space, action_space):
    """使用 DistanceAwareCBFLayer 的 Actor-Critic"""
    from sample_factory.algo.utils.context import global_model_factory
    from swarm_rl.models.quad_multi_model_distance_cbf import QuadActorCriticWithDistanceCBF

    model_factory = global_model_factory()
    return QuadActorCriticWithDistanceCBF(model_factory, obs_space, action_space, cfg)


# 2. 在 register_swarm_components 中添加选项：

def register_swarm_components(use_cbf=False, cbf_type='rcbf'):
    """
    Args:
        use_cbf: 是否启用 CBF
        cbf_type: 'rcbf' 或 'distance_cbf'
    """
    register_env("quadrotor_multi", make_quadrotor_env)
    register_models()

    if use_cbf:
        from sample_factory.algo.utils.context import global_model_factory

        if cbf_type == 'distance_cbf':
            global_model_factory().make_actor_critic_func = make_actor_critic_with_distance_cbf
        else:
            global_model_factory().make_actor_critic_func = make_actor_critic_with_cbf


# 3. 训练命令：

# python -m sample_factory.launcher.run \
#     --run=swarm_rl.runs.obstacles.quads_multi_obstacles \
#     --quads_use_cbf=True \
#     --quads_use_obstacles=True \
#     --quads_cbf_alpha=1.0 \
#     --quads_cbf_k=2.0 \
#     --quads_cbf_sigma=0.1 \
#     --num_gpus=1

# 特点：
# - 使用 DistanceAwareCBFLayer（基于姿态屏障函数）
# - 需要 qpth 求解器
# - 支持梯度回传
# - 距离加权约束


# ============ 参数说明 ============

# RCBF 模型参数：
# --quads_cbf_alpha_1: CBF 增益 1（默认 1.0）
# --quads_cbf_alpha_2: CBF 增益 2（默认 1.0）
# --quads_cbf_safety_allowance: 安全余量（默认 0.1）
# --quads_cbf_sdf_resolution: SDF 分辨率（默认 0.1）

# DistanceCBF 模型参数：
# --quads_cbf_alpha: CBF 增益（默认 1.0）
# --quads_cbf_k: 距离衰减率（默认 2.0）
# --quads_cbf_sigma: 安全缓冲（默认 0.1）


# ============ 关键实现细节 ============

# 1. 观测结构（必须包含 SDF）：
#    [self_obs(13), neighbors_obs(...), sdf_obs(9)]
#    其中 self_obs = [pos(3), vel(3), rot(9)]

# 2. 动作空间：
#    [a_thrust(1), omega_x(1), omega_y(1), omega_z(1)]
#    归一化到 [-1, 1]

# 3. CBF 修正流程：
#    RL 输出 u_rl → CBF 层 → 安全动作 u_safe → 环境执行

# 4. 训练时：
#    - log_prob 基于 u_rl（原始 RL 输出）
#    - 环境执行 u_safe（安全修正后）
#    - 梯度通过 CBF 层回传到 RL 策略

# 5. 推理时：
#    - 直接使用 u_safe（安全动作）
