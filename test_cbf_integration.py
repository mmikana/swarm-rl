"""
测试 CBF-QP 层的基本功能

用于验证：
1. SDF 梯度计算
2. CBF 约束计算
3. QP 求解
"""

import numpy as np


def test_cbf_qp_layer():
    """测试 CBF-QP 层"""
    print("=" * 60)
    print("测试 QuadCBFQPLayer")
    print("=" * 60)

    try:
        from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer
        print("✓ 成功导入 QuadCBFQPLayer\n")
    except ImportError as e:
        print(f"✗ 导入失败: {e}\n")
        return False

    # 1. 创建 CBF层
    print("1. 创建 CBF-QP 层...")
    cbf_layer = QuadCBFQPLayer(
        mass=0.028,
        thrust_to_weight=3.0,
        alpha_1=1.0,
        alpha_2=1.0,
        k_omega=0.1,
        R_obs=0.5,
        epsilon=0.1,
    )
    print(f"   - 质量: {cbf_layer.m} kg")
    print(f"   - 最大推力: {cbf_layer.T_max:.4f} N")
    print(f"   - CBF 增益: α₁={cbf_layer.alpha_1}, α₂={cbf_layer.alpha_2}")
    print()

    # 2. 测试 SDF 梯度计算
    print("2. 测试 SDF 梯度计算...")
    # 创建一个简单的 SDF 网格（无人机在安全区域）
    sdf_obs = np.array([
        0.5, 0.6, 0.7,  # 上排
        0.4, 0.5, 0.6,  # 中排（中心点是 0.5m）
        0.3, 0.4, 0.5   # 下排
    ])
    n, h = cbf_layer.compute_sdf_gradient(sdf_obs)
    print(f"   - SDF 中心值: h = {h:.3f} m")
    print(f"   - 梯度向量: n = {n}")
    print(f"   - 梯度模长: ||n|| = {np.linalg.norm(n):.3f}")
    print()

    # 3. 测试 CBF 约束计算
    print("3. 测试 CBF 约束计算...")
    state = {
        'vel': np.array([0.5, 0.3, 0.0]),  # 速度 (m/s)
        'rot': np.eye(3),  # 单位旋转矩阵（无倾斜）
        'omega': np.array([0.1, 0.2, 0.0])  # 角速度 (rad/s)
    }
    u_rl = np.array([0.2, 0.2, 0.2, 0.2])  # RL 策略输出

    A, b = cbf_layer.compute_cbf_constraints(state, sdf_obs)
    print(f"   - 控制矩阵 A: {A}")
    print(f"   - 约束标量 b: {b[0]:.4f}")
    print()

    # 4. 测试 QP 求解（如果 cvxpy 可用）
    print("4. 测试 QP 求解...")
    if cbf_layer.cvxpy_available:
        u_safe = cbf_layer.solve_qp_cvxpy(u_rl, A, b)
        print(f"   - RL 动作: {u_rl}")
        print(f"   - 安全动作: {u_safe}")
        print(f"   - 修正量: {np.linalg.norm(u_safe - u_rl):.4f}")
        print("   ✓ cvxpy 可用，QP 求解成功")
    else:
        print("   ✗ cvxpy 未安装，跳过 QP 测试")
    print()

    # 5. 测试完整流程
    print("5. 测试完整的 get_safe_action...")
    u_safe = cbf_layer.get_safe_action(state, u_rl, sdf_obs)
    print(f"   - 输入动作: {u_rl}")
    print(f"   - 安全动作: {u_safe}")
    print()

    print("=" * 60)
    print("✓ CBF-QP 层测试完成")
    print("=" * 60)
    return True


def test_actor_critic_import():
    """测试 Actor-Critic 模型导入"""
    print("\n" + "=" * 60)
    print("测试 QuadActorCriticWithCBF")
    print("=" * 60)

    try:
        from swarm_rl.models.quad_multi_model_rcbf import QuadActorCriticWithCBF
        print("✓ 成功导入 QuadActorCriticWithCBF")
        print()
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print()
        return False


if __name__ == '__main__':
    print("\nCBF 集成测试\n")

    # 测试 CBF-QP 层
    success1 = test_cbf_qp_layer()

    # 测试 Actor-Critic 导入
    success2 = test_actor_critic_import()

    if success1 and success2:
        print("\n✓ 所有测试通过！")
        print("\n下一步：")
        print("1. 安装依赖: pip install cvxpy qpth（如果还没安装）")
        print("2. 运行训练测试（小规模）：")
        print("   python -m swarm_rl.train --algo=APPO --env=quadrotor_multi \\")
        print("       --quads_use_cbf=True --quads_use_obstacles=True \\")
        print("       --quads_obstacle_obs_type=octomap --train_for_env_steps=1000")
    else:
        print("\n✗ 部分测试失败，请检查错误信息")
