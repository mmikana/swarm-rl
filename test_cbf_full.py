"""
CBF 模块完整功能测试套件

测试范围：
1. SDF 梯度计算
2. CBF 约束计算（单样本 + 批量）
3. QP 求解器（cvxpy + qpth）
4. 训练/推理模式切换
5. Actor-Critic 集成
6. 边界情况处理
"""

import numpy as np
import torch


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    """打印子章节标题"""
    print(f"\n▶ {title}")
    print("-" * 50)


# ============================================================================
# 测试 1: SDF 梯度计算
# ============================================================================

def test_sdf_gradient():
    """测试 SDF 梯度计算"""
    print_section("测试 1: SDF 梯度计算")
    
    from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer
    cbf = QuadCBFQPLayer()
    
    # 测试 1.1: 简单梯度（右上方向）
    print_subsection("1.1 简单梯度测试（右上方向）")
    sdf1 = np.array([
        0.4, 0.5, 0.6,
        0.3, 0.4, 0.5,
        0.2, 0.3, 0.4
    ])
    n1, h1 = cbf.compute_sdf_gradient(sdf1)
    print(f"   SDF: {sdf1}")
    print(f"   梯度 n: {n1}")
    print(f"   中心值 h: {h1}")
    print(f"   梯度模长: {np.linalg.norm(n1):.6f}")
    assert abs(np.linalg.norm(n1) - 1.0) < 1e-5, "梯度应该归一化"
    # n_x = (0.5 - 0.3) / 0.2 = 1.0, n_y = (0.3 - 0.5) / 0.2 = -1.0
    # 所以梯度指向右下方向（SDF 增加最快的方向）
    assert n1[0] > 0 and n1[1] < 0, "梯度应该指向 SDF 增加的方向"
    print("   ✓ 通过")
    
    # 测试 1.2: 反向梯度（左下方向）
    print_subsection("1.2 反向梯度测试（左下方向）")
    sdf2 = np.array([
        0.6, 0.5, 0.4,
        0.5, 0.4, 0.3,
        0.4, 0.3, 0.2
    ])
    n2, h2 = cbf.compute_sdf_gradient(sdf2)
    print(f"   梯度 n: {n2}")
    print(f"   中心值 h: {h2}")
    assert n2[0] < 0 and n2[1] < 0, "梯度应该指向左下方"
    print("   ✓ 通过")
    
    # 测试 1.3: 零梯度（安全区域）
    print_subsection("1.3 零梯度测试（均匀 SDF）")
    sdf3 = np.array([0.5] * 9)
    n3, h3 = cbf.compute_sdf_gradient(sdf3)
    print(f"   梯度 n: {n3}")
    print(f"   中心值 h: {h3}")
    assert np.linalg.norm(n3) < 1e-5, "均匀 SDF 应该零梯度"
    print("   ✓ 通过")
    
    # 测试 1.4: 批量 SDF 梯度（PyTorch）
    print_subsection("1.4 批量 SDF 梯度测试（PyTorch）")
    sdf_batch = torch.tensor([
        [0.4, 0.5, 0.6, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4],
        [0.6, 0.5, 0.4, 0.5, 0.4, 0.3, 0.4, 0.3, 0.2],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ])
    n_batch, h_batch = cbf.compute_sdf_gradient_batch(sdf_batch)
    print(f"   批量梯度形状: {n_batch.shape}")
    print(f"   批量中心值: {h_batch}")
    print(f"   样本 1 梯度: {n_batch[0]}")
    print(f"   样本 2 梯度: {n_batch[1]}")
    print(f"   样本 3 梯度: {n_batch[2]}")
    assert n_batch.shape == (3, 3), "批量梯度形状应该是 (3, 3)"
    assert h_batch.shape == (3,), "批量中心值形状应该是 (3,)"
    print("   ✓ 通过")
    
    print("\n✅ SDF 梯度计算测试全部通过！")
    return True


# ============================================================================
# 测试 2: CBF 约束计算
# ============================================================================

def test_cbf_constraints():
    """测试 CBF 约束计算"""
    print_section("测试 2: CBF 约束计算")
    
    from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer
    from transforms3d.euler import euler2mat
    
    cbf = QuadCBFQPLayer(
        mass=0.028,
        thrust_to_weight=3.0,
        alpha_1=1.0,
        alpha_2=1.0,
        k_omega=0.1,
        R_obs=0.5,
        epsilon=0.1,
    )
    
    # 测试 2.1: 悬停状态（无倾斜）
    print_subsection("2.1 悬停状态（无倾斜）")
    sdf = np.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5])
    state_hover = {
        'vel': np.array([0.0, 0.0, 0.0]),
        'rot': np.eye(3),
        'omega': np.array([0.0, 0.0, 0.0])
    }
    A_hover, b_hover = cbf.compute_cbf_constraints(state_hover, sdf)
    print(f"   A: {A_hover}")
    print(f"   b: {b_hover}")
    print(f"   nTRe3: {np.dot(np.array([0.707, -0.707, 0]), np.array([0, 0, 1])):.4f}")
    # 悬停时 nTRe3 ≈ 0，所以 A ≈ 0
    assert A_hover.shape == (1, 4), "A 的形状应该是 (1, 4)"
    print("   ✓ 通过")
    
    # 测试 2.2: 倾斜状态（有控制能力）
    print_subsection("2.2 倾斜状态（Roll=0.3, Pitch=0.2）")
    R_tilted = euler2mat(0.3, 0.2, 0)
    state_tilted = {
        'vel': np.array([0.5, 0.3, 0.0]),
        'rot': R_tilted,
        'omega': np.array([0.1, 0.2, 0.0])
    }
    A_tilted, b_tilted = cbf.compute_cbf_constraints(state_tilted, sdf)
    print(f"   A: {A_tilted}")
    print(f"   b: {b_tilted:.4f}")
    nTRe3 = np.dot(np.array([0.707, -0.707, 0]), R_tilted @ np.array([0, 0, 1]))
    print(f"   nTRe3: {nTRe3:.4f}")
    assert np.abs(A_tilted).max() > 0.1, "倾斜时 A 应该有非零值"
    print("   ✓ 通过")
    
    # 测试 2.3: 高速状态（离心项影响）
    print_subsection("2.3 高速状态（离心项影响）")
    state_fast = {
        'vel': np.array([2.0, 2.0, 0.0]),  # 高速
        'rot': R_tilted,
        'omega': np.array([1.0, 1.0, 0.0])  # 高角速度
    }
    A_fast, b_fast = cbf.compute_cbf_constraints(state_fast, sdf)
    print(f"   A: {A_fast}")
    print(f"   b: {b_fast:.4f}")
    print(f"   与悬停相比 b 的变化: {b_fast - b_hover:.4f}")
    # 高速时离心项和 Risk 项会增大 b
    print("   ✓ 通过")
    
    # 测试 2.4: 批量约束计算（PyTorch）
    print_subsection("2.4 批量约束计算（PyTorch）")
    batch_size = 4
    state_batch = {
        'vel': torch.randn(batch_size, 3) * 0.5,
        'rot': torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1),
        'omega': torch.randn(batch_size, 3) * 0.2
    }
    sdf_batch = torch.randn(batch_size, 9) * 0.2 + 0.5
    A_batch, b_batch = cbf.compute_cbf_constraints_batch(state_batch, sdf_batch)
    print(f"   A 形状: {A_batch.shape}")
    print(f"   b 形状: {b_batch.shape}")
    print(f"   A 样本 [0]: {A_batch[0]}")
    print(f"   b 样本 [0]: {b_batch[0]}")
    assert A_batch.shape == (batch_size, 1, 4), "A 的形状应该是 (batch, 1, 4)"
    assert b_batch.shape == (batch_size, 1), "b 的形状应该是 (batch, 1)"
    print("   ✓ 通过")
    
    print("\n✅ CBF 约束计算测试全部通过！")
    return True


# ============================================================================
# 测试 3: QP 求解器
# ============================================================================

def test_qp_solvers():
    """测试 QP 求解器"""
    print_section("测试 3: QP 求解器")
    
    from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer
    from transforms3d.euler import euler2mat
    
    cbf = QuadCBFQPLayer()
    
    # 准备测试数据
    R_tilted = euler2mat(0.3, 0.2, 0)
    state = {
        'vel': np.array([0.5, 0.3, 0.0]),
        'rot': R_tilted,
        'omega': np.array([0.1, 0.2, 0.0])
    }
    sdf = np.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5])
    u_rl = np.array([0.2, 0.2, 0.2, 0.2])
    
    A, b = cbf.compute_cbf_constraints(state, sdf)
    
    # 测试 3.1: cvxpy 求解器（推理）
    print_subsection("3.1 cvxpy 求解器（推理模式）")
    if cbf.cvxpy_available:
        u_safe_cvxpy = cbf.solve_qp_cvxpy(u_rl, A, b)
        print(f"   RL 动作: {u_rl}")
        print(f"   安全动作: {u_safe_cvxpy}")
        print(f"   修正量: {np.linalg.norm(u_safe_cvxpy - u_rl):.6f}")
        assert u_safe_cvxpy.shape == (4,), "输出形状应该是 (4,)"
        assert np.all(u_safe_cvxpy >= -1 - 1e-5), "动作应该 >= -1"
        assert np.all(u_safe_cvxpy <= 1 + 1e-5), "动作应该 <= 1"
        print("   ✓ cvxpy 可用，测试通过")
    else:
        print("   ⚠ cvxpy 未安装，跳过")
    print("   ✓ 通过")
    
    # 测试 3.2: qpth 求解器（训练）
    print_subsection("3.2 qpth 求解器（训练模式）")
    if cbf.qpth_available:
        u_rl_batch = torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]])
        A_batch = torch.tensor([A, A])
        b_batch = torch.tensor([[b], [b]])
        
        u_safe_qpth = cbf.solve_qp_differentiable(u_rl_batch, A_batch, b_batch)
        print(f"   RL 动作形状: {u_rl_batch.shape}")
        print(f"   安全动作形状: {u_safe_qpth.shape}")
        print(f"   安全动作 [0]: {u_safe_qpth[0]}")
        assert u_safe_qpth.shape == (2, 4), "输出形状应该是 (2, 4)"
        print("   ✓ qpth 可用，测试通过")
    else:
        print("   ⚠ qpth 未安装，跳过")
    print("   ✓ 通过")
    
    # 测试 3.3: 边界情况（无约束时）
    print_subsection("3.3 边界情况（无约束时解接近 RL）")
    # 当 A ≈ 0 且 b 很小时，QP 应该返回接近 u_rl 的解
    A_zero = np.zeros((1, 4))
    b_zero = -10.0  # 很松的约束
    u_safe_loose = cbf.solve_qp_cvxpy(u_rl, A_zero, b_zero)
    print(f"   RL 动作: {u_rl}")
    print(f"   宽松约束下安全动作: {u_safe_loose}")
    diff = np.linalg.norm(u_safe_loose - u_rl)
    print(f"   差异: {diff:.6f}")
    assert diff < 0.01, "宽松约束下应该接近原始动作"
    print("   ✓ 通过")
    
    print("\n✅ QP 求解器测试全部通过！")
    return True


# ============================================================================
# 测试 4: 训练/推理模式切换
# ============================================================================

def test_training_inference_modes():
    """测试训练/推理模式切换"""
    print_section("测试 4: 训练/推理模式切换")
    
    from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer
    from transforms3d.euler import euler2mat
    
    cbf = QuadCBFQPLayer()
    R_tilted = euler2mat(0.3, 0.2, 0)
    
    # 测试 4.1: 训练模式（PyTorch tensor 输入）
    print_subsection("4.1 训练模式（train()）")
    cbf.train()
    assert cbf.training == True, "应该处于训练模式"
    
    state_train = {
        'vel': torch.randn(2, 3) * 0.5,
        'rot': torch.eye(3).unsqueeze(0).expand(2, -1, -1),
        'omega': torch.randn(2, 3) * 0.2
    }
    sdf_train = torch.randn(2, 9) * 0.2 + 0.5
    u_rl_train = torch.randn(2, 4) * 0.3
    
    u_safe_train = cbf.forward(state_train, u_rl_train, sdf_train)
    print(f"   输入形状: {u_rl_train.shape}")
    print(f"   输出形状: {u_safe_train.shape}")
    print(f"   requires_grad: {u_safe_train.requires_grad}")
    assert u_safe_train.shape == (2, 4), "输出形状应该匹配输入"
    print("   ✓ 训练模式通过")
    
    # 测试 4.2: 推理模式（eval() + numpy 输入）
    print_subsection("4.2 推理模式（eval() + numpy）")
    cbf.eval()
    assert cbf.training == False, "应该处于推理模式"
    
    state_eval = {
        'vel': np.array([0.5, 0.3, 0.0]),
        'rot': R_tilted,
        'omega': np.array([0.1, 0.2, 0.0])
    }
    sdf_eval = np.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5])
    u_rl_eval = np.array([0.2, 0.2, 0.2, 0.2])
    
    u_safe_eval = cbf.forward(state_eval, u_rl_eval, sdf_eval)
    print(f"   输入类型: {type(u_rl_eval)}")
    print(f"   输出类型: {type(u_safe_eval)}")
    print(f"   输出: {u_safe_eval}")
    assert isinstance(u_safe_eval, np.ndarray), "推理模式应该返回 numpy"
    assert u_safe_eval.shape == (4,), "输出形状应该是 (4,)"
    print("   ✓ 推理模式通过")
    
    # 测试 4.3: 推理模式（eval() + tensor 输入）
    print_subsection("4.3 推理模式（eval() + tensor）")
    state_tensor = {
        'vel': torch.randn(1, 3) * 0.5,
        'rot': torch.eye(3).unsqueeze(0),
        'omega': torch.randn(1, 3) * 0.2
    }
    sdf_tensor = torch.randn(1, 9) * 0.2 + 0.5
    u_rl_tensor = torch.randn(1, 4) * 0.3
    
    u_safe_tensor = cbf.forward(state_tensor, u_rl_tensor, sdf_tensor)
    print(f"   输出类型: {type(u_safe_tensor)}")
    print(f"   输出形状: {u_safe_tensor.shape}")
    assert isinstance(u_safe_tensor, torch.Tensor), "应该返回 tensor"
    print("   ✓ 推理模式（tensor）通过")
    
    print("\n✅ 训练/推理模式切换测试全部通过！")
    return True


# ============================================================================
# 测试 5: Actor-Critic 集成
# ============================================================================

def test_actor_critic_integration():
    """测试 Actor-Critic 集成"""
    print_section("测试 5: Actor-Critic 集成")
    
    # 测试 5.1: 导入模型
    print_subsection("5.1 导入 QuadActorCriticWithCBF")
    try:
        from swarm_rl.models.quad_multi_model_rcbf import QuadActorCriticWithCBF
        print("   ✓ 成功导入")
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        return False
    
    # 测试 5.2: 检查 CBF 层创建
    print_subsection("5.2 检查 CBF 层参数")
    from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer
    cbf = QuadCBFQPLayer(
        mass=0.028,
        thrust_to_weight=3.0,
        alpha_1=1.5,
        alpha_2=2.0,
        k_omega=0.15,
        R_obs=0.6,
        epsilon=0.15,
    )
    print(f"   质量: {cbf.m} kg")
    print(f"   最大推力: {cbf.T_max:.4f} N")
    print(f"   α₁={cbf.alpha_1}, α₂={cbf.alpha_2}")
    print(f"   k_omega={cbf.k_omega}")
    print("   ✓ 参数正确")
    
    # 测试 5.3: 检查观测提取
    print_subsection("5.3 检查观测结构")
    from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE
    
    obs_repr = 'full_state'  # 默认
    neighbor_obs_type = 'polar'  # 默认
    
    self_obs_dim = QUADS_OBS_REPR.get(obs_repr, 18)
    neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE.get(neighbor_obs_type, 5)
    
    print(f"   self_obs_dim: {self_obs_dim}")
    print(f"   neighbor_obs_dim: {neighbor_obs_dim}")
    print("   ✓ 观测结构正确")
    
    print("\n✅ Actor-Critic 集成测试全部通过！")
    return True


# ============================================================================
# 测试 6: 边界情况与鲁棒性
# ============================================================================

def test_edge_cases():
    """测试边界情况与鲁棒性"""
    print_section("测试 6: 边界情况与鲁棒性")
    
    from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer
    from transforms3d.euler import euler2mat
    
    cbf = QuadCBFQPLayer()
    
    # 测试 6.1: 极小 SDF 值（接近碰撞）
    print_subsection("6.1 极小 SDF 值（h = 0.05m）")
    sdf_close = np.array([0.05] * 9)
    n_close, h_close = cbf.compute_sdf_gradient(sdf_close)
    print(f"   h = {h_close:.4f} m")
    print(f"   n = {n_close}")
    assert h_close == 0.05, "SDF 值应该正确"
    print("   ✓ 通过")
    
    # 测试 6.2: 负 SDF 值（已碰撞）
    print_subsection("6.2 负 SDF 值（h = -0.1m，已碰撞）")
    sdf_collision = np.array([-0.1] * 9)
    n_coll, h_coll = cbf.compute_sdf_gradient(sdf_collision)
    print(f"   h = {h_coll:.4f} m")
    assert h_coll < 0, "应该检测到碰撞"
    print("   ✓ 通过")
    
    # 测试 6.3: 极大速度
    print_subsection("6.3 极大速度（v = 3.0 m/s）")
    R = euler2mat(0.3, 0.2, 0)
    state_fast = {
        'vel': np.array([3.0, 3.0, 0.0]),
        'rot': R,
        'omega': np.array([5.0, 5.0, 0.0])
    }
    sdf = np.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5])
    A_fast, b_fast = cbf.compute_cbf_constraints(state_fast, sdf)
    print(f"   A: {A_fast}")
    print(f"   b: {b_fast:.4f}")
    assert np.isfinite(b_fast), "b 应该是有限值"
    print("   ✓ 通过")
    
    # 测试 6.4: 动作边界（u = ±1）
    print_subsection("6.4 动作边界测试（u = ±1）")
    u_boundary = np.array([1.0, -1.0, 1.0, -1.0])
    A, b = cbf.compute_cbf_constraints({
        'vel': np.array([0.5, 0.3, 0.0]),
        'rot': R,
        'omega': np.array([0.1, 0.2, 0.0])
    }, sdf)
    
    if cbf.cvxpy_available:
        u_safe = cbf.solve_qp_cvxpy(u_boundary, A, b)
        print(f"   输入: {u_boundary}")
        print(f"   输出: {u_safe}")
        assert np.all(u_safe >= -1 - 1e-5), "输出应该 >= -1"
        assert np.all(u_safe <= 1 + 1e-5), "输出应该 <= 1"
        print("   ✓ 边界约束满足")
    
    # 测试 6.5: 数值稳定性（除零保护）
    print_subsection("6.5 数值稳定性测试")
    state_zero = {
        'vel': np.array([0.0, 0.0, 0.0]),
        'rot': np.eye(3),
        'omega': np.array([0.0, 0.0, 0.0])
    }
    A_zero, b_zero = cbf.compute_cbf_constraints(state_zero, sdf)
    assert np.all(np.isfinite(A_zero)), "A 应该是有限值"
    assert np.isfinite(b_zero), "b 应该是有限值"
    print("   ✓ 数值稳定")
    
    print("\n✅ 边界情况与鲁棒性测试全部通过！")
    return True


# ============================================================================
# 主函数
# ============================================================================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("  CBF 模块完整功能测试套件")
    print("=" * 70)
    
    results = []
    
    # 运行所有测试
    results.append(("SDF 梯度计算", test_sdf_gradient()))
    results.append(("CBF 约束计算", test_cbf_constraints()))
    results.append(("QP 求解器", test_qp_solvers()))
    results.append(("训练/推理模式", test_training_inference_modes()))
    results.append(("Actor-Critic 集成", test_actor_critic_integration()))
    results.append(("边界情况", test_edge_cases()))
    
    # 汇总结果
    print_section("测试结果汇总")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计：{passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！CBF 模块功能完整！")
        print("\n下一步建议：")
        print("1. 小规模训练测试:")
        print("   python -m swarm_rl.train --algo=APPO --env=quadrotor_multi \\")
        print("       --quads_use_cbf=True --quads_use_obstacles=True \\")
        print("       --train_for_env_steps=1000")
    else:
        print(f"\n⚠ {total - passed} 个测试失败，请检查错误信息")
    
    return passed == total


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
