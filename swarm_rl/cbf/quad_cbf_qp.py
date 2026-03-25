"""
QuadCBF-QP Layer for Safe Control

基于 SDF (Signed Distance Field) 的 RCBF 实现
理论推导见: docs/RCBF_理论推导.md
"""

import numpy as np
import torch
import torch.nn as nn

# 导入动力学参数
from gym_art.quadrotor_multi.quad_models import crazyflie_params
from gym_art.quadrotor_multi.inertia import QuadLink


def get_crazyflie_physics():
    """从动力学模型获取物理参数"""
    params = crazyflie_params()

    # 计算质量 (与 QuadrotorDynamics 一致)
    quad_model = QuadLink(params=params["geom"], verbose=False)
    mass = quad_model.m

    # 推力重量比
    thrust_to_weight = params["motor"]["thrust_to_weight"]

    return mass, thrust_to_weight


class QuadCBFQPLayer(nn.Module):
    """
    四旋翼 CBF-QP 层

    功能：
    - 从 SDF 观测计算 CBF 约束
    - 使用 QP 求解安全动作
    - 支持训练（可微分）和推理（精确求解）两种模式

    Args:
        mass: 无人机质量 (kg), 默认从 crazyflie_params 读取
        thrust_to_weight: 推力重量比, 默认从 crazyflie_params 读取
        alpha_1: CBF 增益参数 1 (s^-1)
        alpha_2: CBF 增益参数 2 (s^-1)
        k_omega: 角速度风险权重 (m/rad^2)
        R_obs: 障碍物半径 (m)
        epsilon: 数值稳定性常数 (m/s^2)
        sdf_resolution: SDF 网格分辨率 (m)
    """

    def __init__(
        self,
        mass=None,
        thrust_to_weight=None,
        alpha_1=1.0,
        alpha_2=1.0,
        R_obs=0.5,
        sdf_resolution=0.1,
    ):
        super().__init__()

        # 物理参数: 优先使用传入值,否则从动力学模型读取
        if mass is None or thrust_to_weight is None:
            dyn_mass, dyn_ttw = get_crazyflie_physics()
            mass = mass if mass is not None else dyn_mass
            thrust_to_weight = thrust_to_weight if thrust_to_weight is not None else dyn_ttw
            print(f"[CBF] 从动力学模型读取参数: mass={mass:.4f}kg, thrust_to_weight={thrust_to_weight:.1f}")

        self.m = mass
        self.g = 9.81
        self.T_max = mass * self.g * thrust_to_weight / 4.0  # 单电机最大推力

        # CBF 参数
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.R_obs = R_obs
        self.delta = sdf_resolution

        # 注册常数为 buffer（自动跟随 device）
        self.register_buffer('e3', torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))

        # 尝试导入 QP 求解器（不保存模块引用，以支持 pickle）
        self.cvxpy_available = False
        self.qpth_available = False

        try:
            import cvxpy as cp
            self.cvxpy_available = True
        except ImportError:
            pass

        try:
            from qpth.qp import QPFunction
            self.qpth_available = True
        except ImportError:
            pass

    def compute_sdf_gradient(self, sdf_obs):
        """
        从 3×3 SDF 网格计算梯度和安全距离

        SDF 网格布局:
            [0]  [1]  [2]
            [3]  [4]  [5]  (中心点是 [4])
            [6]  [7]  [8]

        Args:
            sdf_obs: (9,) numpy array

        Returns:
            n: (3,) numpy array, 归一化梯度向量
            h: float, 中心点的安全距离
        """
        h = sdf_obs[4]  # 中心点的 SDF 值

        # 中心差分计算梯度
        n_x = (sdf_obs[5] - sdf_obs[3]) / (2 * self.delta)
        n_y = (sdf_obs[7] - sdf_obs[1]) / (2 * self.delta)
        n = np.array([n_x, n_y, 0.0])

        # 归一化
        norm = np.linalg.norm(n)
        if norm > 1e-6:
            n = n / norm

        return n, h

    def compute_cbf_constraints(self, state, sdf_obs):
        """
        计算 CBF 约束: A @ u >= b

        改进的二阶 CBF：不直接约束加速度，而是约束速度
        这样避免了当推力无法直接产生加速度时的问题

        h_ddot + (α₁ + α₂) * h_dot + α₁ * α₂ * h >= 0

        其中：
        - h = 距离障碍的安全距离
        - h_dot = n^T @ v （速度沿梯度方向的分量）
        - h_ddot = 通过约束速度来间接控制（不直接用加速度）

        Args:
            state: dict with 'vel' (3,), 'rot' (3,3), 'omega' (3,)
            sdf_obs: (9,) numpy array

        Returns:
            A: (1, 4) numpy array
            b: float
        """
        v = state['vel']
        R = state['rot']
        omega = state['omega']

        # 1. SDF 梯度和安全距离
        n, h = self.compute_sdf_gradient(sdf_obs)

        # 2. h_dot = n^T @ v （速度沿梯度方向的分量）
        h_dot = np.dot(n, v)

        # 3. 梯度变化率（由于旋转）
        # dn/dt = [ω]× @ n
        omega_cross_n = np.array([
            omega[1] * n[2] - omega[2] * n[1],
            omega[2] * n[0] - omega[0] * n[2],
            omega[0] * n[1] - omega[1] * n[0]
        ])
        # 但这项实际上很小，可以忽略

        # 4. 关键修改：不约束加速度，而是约束速度
        #
        # 原约束试图做：h_ddot >= -...
        # 但 h_ddot = d(n^T @ v)/dt 包含角动力学项，难以直接控制
        #
        # 新方法：转化为对速度的约束
        # 目标：当接近障碍时，限制向障碍方向的速度
        #
        # 重新参数化：定义新的"虚拟控制"为期望的 h_dot（安全速度）
        # u_desired_h_dot = -α₁ * h  （使 h 指数收敛到 0）
        #
        # 实际约束：n^T @ v >= u_desired_h_dot
        # 即：n^T @ v >= -α₁ * h
        #
        # 转化为 u 的约束：
        # h_dot = n^T @ v
        # v_dot = (T/m) * R @ e3 - g*e3
        # 我们希望约束下一步的 h_dot

        # 重力加速度项
        g_vec = np.array([0.0, 0.0, self.g])
        a_gravity = -g_vec  # 重力作用的加速度

        # 推力产生的加速度方向（单位向量）
        Re3 = R @ np.array([0.0, 0.0, 1.0])

        # 下一步的 h_dot 变化：
        # h_dot_next = h_dot + dt * (dn/dt^T @ v + n^T @ a)
        #            ≈ h_dot + n^T @ a_gravity + n^T @ (T/m) * Re3
        #            = h_dot + n^T @ g + (T/m) * n^T @ Re3
        #
        # 为了满足 CBF 约束 h_ddot + α₁ * h_dot + α₁ * α₂ * h >= 0：
        # n^T @ v_dot >= -α₁ * h_dot - α₁ * α₂ * h
        # n^T @ a_gravity + (T/m) * n^T @ Re3 >= -α₁ * h_dot - α₁ * α₂ * h

        nTRe3 = np.dot(n, Re3)
        n_g = np.dot(n, a_gravity)

        # A 矩阵：推力对约束的影响系数
        A = (self.T_max / (2 * self.m)) * nTRe3 * np.ones(4)

        # b 标量：约束右侧
        b = (
            - n_g  # 重力的负贡献
            - (self.alpha_1 + self.alpha_2) * h_dot  # CBF 阻尼
            - self.alpha_1 * self.alpha_2 * h  # CBF 恢复
        )

        # 当 nTRe3 ≈ 0 时的处理（推力无法直接约束加速度）
        # 此时推力必须通过改变姿态来间接影响加速度
        # 但这需要时间！所以我们添加一个安全松弛：
        # 降低约束要求，给系统更多时间反应
        if abs(nTRe3) < 0.01:
            # 当推力不能直接产生约束加速度时
            # 松弛约束标量 b，允许更多的速度
            # 这相当于降低安全要求，但给了系统时间改变姿态
            b = b + 0.5  # 松弛 0.5 m/s²

        return A.reshape(1, -1), b

    def solve_qp_cvxpy(self, u_rl, A, b):
        """
        使用 cvxpy 求解 QP（推理用，精确但不可微）

        Args:
            u_rl: (4,) numpy array
            A: (1, 4) numpy array
            b: float

        Returns:
            u_safe: (4,) numpy array
        """
        if not self.cvxpy_available:
            return u_rl

        # 在方法内部导入 cvxpy（避免 pickle 问题）
        import cvxpy as cp

        # 检查 A 是否接近 0（推力方向与避障方向垂直）
        A_norm = np.linalg.norm(A)
        if A_norm < 1e-6:
            # 推力无法在避障方向产生加速度
            # 如果约束不满足，直接返回最小推力（减速）
            if b > 0:
                # 需要减速，返回最小推力
                return np.array([-1.0, -1.0, -1.0, -1.0])
            else:
                # 约束已满足，返回原始动作
                return u_rl

        u = cp.Variable(4)
        slack = cp.Variable(1, nonneg=True)

        objective = cp.Minimize(
            cp.sum_squares(u - u_rl) + 1000 * cp.sum_squares(slack)
        )

        constraints = [
            A @ u >= b - slack,
            u >= -1,
            u <= 1,
            slack >= 0
        ]

        problem = cp.Problem(objective, constraints)

        try:
            # 使用 ECOS 求解器（更稳定，适合小规模 QP）
            problem.solve(solver=cp.ECOS, verbose=False, abstol=1e-6, reltol=1e-6, max_iters=200)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                return np.clip(u.value, -1, 1)
        except:
            pass

        return u_rl

    def get_safe_action(self, state, u_rl, sdf_obs):
        """
        计算安全动作（推理模式）

        Args:
            state: dict
            u_rl: (4,) numpy array, RL 策略输出
            sdf_obs: (9,) numpy array

        Returns:
            u_safe: (4,) numpy array
        """
        # 计算约束
        A, b = self.compute_cbf_constraints(state, sdf_obs)

        # 求解 QP
        u_safe = self.solve_qp_cvxpy(u_rl, A, b)

        return u_safe

    def compute_sdf_gradient_batch(self, sdf_obs):
        """
        批量计算 SDF 梯度（PyTorch 版本）

        Args:
            sdf_obs: (batch_size, 9) tensor

        Returns:
            n: (batch_size, 3) tensor, 归一化梯度向量
            h: (batch_size,) tensor, 中心点的安全距离
        """
        batch_size = sdf_obs.shape[0]
        device = sdf_obs.device

        # 中心点的 SDF 值
        h = sdf_obs[:, 4]  # (batch_size,)

        # 中心差分计算梯度
        n_x = (sdf_obs[:, 5] - sdf_obs[:, 3]) / (2 * self.delta)
        n_y = (sdf_obs[:, 7] - sdf_obs[:, 1]) / (2 * self.delta)
        n_z = torch.zeros(batch_size, device=device)

        n = torch.stack([n_x, n_y, n_z], dim=1)  # (batch_size, 3)

        # 归一化
        norm = torch.norm(n, dim=1, keepdim=True)  # (batch_size, 1)
        n = n / (norm + 1e-6)  # 防止除零

        return n, h

    def compute_cbf_constraints_batch(self, state, sdf_obs):
        """
        批量计算 CBF 约束: A @ u >= b（PyTorch 版本，包含角动力学）

        Args:
            state: dict with tensors
                - 'vel': (batch_size, 3)
                - 'rot': (batch_size, 3, 3)
                - 'omega': (batch_size, 3)
            sdf_obs: (batch_size, 9) tensor

        Returns:
            A: (batch_size, 1, 4) tensor
            b: (batch_size, 1) tensor
        """
        batch_size = sdf_obs.shape[0]
        device = sdf_obs.device

        v = state['vel']        # (batch_size, 3)
        R = state['rot']        # (batch_size, 3, 3)
        omega = state['omega']  # (batch_size, 3)

        # 1. SDF 梯度和安全距离
        n, h = self.compute_sdf_gradient_batch(sdf_obs)  # n: (batch_size, 3), h: (batch_size,)

        # 2. h_dot = n^T @ v
        h_dot = torch.sum(n * v, dim=1)  # (batch_size,)

        # 3. 角速度产生的梯度变化率（关键：角动力学项）
        # dn/dt ≈ [ω]× @ n （叉乘）
        omega_cross_n = torch.stack([
            omega[:, 1] * n[:, 2] - omega[:, 2] * n[:, 1],
            omega[:, 2] * n[:, 0] - omega[:, 0] * n[:, 2],
            omega[:, 0] * n[:, 1] - omega[:, 1] * n[:, 0]
        ], dim=1)  # (batch_size, 3)
        n_dot_v = torch.sum(omega_cross_n * v, dim=1)  # (batch_size,)

        # 4. 离心项
        v_squared = torch.sum(v * v, dim=1)  # (batch_size,)
        denom = torch.clamp(h + self.R_obs, min=1e-6)
        centrifugal = (v_squared - h_dot**2) / denom  # (batch_size,)

        # 5. 控制矩阵 A（包含角动力学的影响）
        e3 = self.e3.to(device)  # (3,)
        Re3 = torch.matmul(R, e3)  # (batch_size, 3)
        nTRe3 = torch.sum(n * Re3, dim=1)  # (batch_size,)

        # 直接项：(T_max / (2m)) * nTRe3
        A_direct = (self.T_max / (2 * self.m)) * nTRe3  # (batch_size,)

        # 当直接项很小时，添加通过角动力学的二阶项
        A_coeff = A_direct.clone()

        mask = torch.abs(nTRe3) < 0.1  # (batch_size,)
        if mask.any():
            # 计算水平梯度分量
            n_xy_norm = torch.norm(n[:, :2], dim=1)  # (batch_size,)

            # 推力通过角动力学的影响（简化估计）
            arm_length = 0.046  # 四旋翼臂长
            max_torque = arm_length * self.T_max / 2
            max_angular_accel = max_torque / 1e-5  # I ≈ 1e-5 kg*m^2

            A_angular = (1.0 / self.m) * n_xy_norm * max_angular_accel

            # 只在 nTRe3 接近 0 的地方添加角动力学项
            A_coeff = torch.where(mask, A_direct + A_angular, A_direct)

        A = A_coeff.unsqueeze(1).unsqueeze(2) * torch.ones(batch_size, 1, 4, device=device)

        # 6. 约束标量 b
        e3_tensor = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        gravity_term = torch.sum(n * (self.g * e3_tensor), dim=1)  # (batch_size,)
        bias_term = (2 * self.T_max / self.m) * nTRe3  # (batch_size,)

        b = (
            gravity_term
            - n_dot_v - centrifugal
            - (self.alpha_1 + self.alpha_2) * h_dot
            - self.alpha_1 * self.alpha_2 * h
            - bias_term
        )  # (batch_size,)

        return A, b.unsqueeze(1)  # (batch_size, 1, 4), (batch_size, 1)

    def solve_qp_differentiable(self, u_rl, A, b):
        """
        可微分 QP 求解（训练用）

        使用 qpth 库求解 QP，支持梯度回传

        Args:
            u_rl: (batch_size, 4) tensor
            A: (batch_size, 1, 4) tensor
            b: (batch_size, 1) tensor

        Returns:
            u_safe: (batch_size, 4) tensor
        """
        if not self.qpth_available:
            # 如果 qpth 不可用，返回原始动作
            print("Warning: qpth not available, returning u_rl")
            return u_rl

        # 在方法内部导入 qpth（避免 pickle 问题）
        from qpth.qp import QPFunction
        # 提高求解精度：eps=1e-5（默认1e-4太松），maxIter=20000
        # 这对梯度流的质量至关重要
        qp_solver = QPFunction(verbose=False, maxIter=20000, eps=1e-5)

        device = u_rl.device
        batch_size = u_rl.shape[0]
        n_u, n_slack = 4, 1
        n_x = n_u + n_slack

        # ============ 数值稳定化处理 ============
        # 检查约束的可行性和数值条件数
        A_norm = torch.norm(A, dim=-1, keepdim=True)  # (batch_size, 1, 1)
        A_norm = torch.clamp(A_norm, min=1e-6)  # 防止除以零

        # 当 A 范数很小时（无人机无法在某方向产生加速度）
        # 添加一个小的正则项改善数值稳定性
        regularization = 1e-6
        A = A + regularization * torch.randn_like(A) * 0.1  # 添加微小随机噪声以避免奇异性

        # ============ 构造 QP 问题 ============
        # 1. 构造目标函数 P 和 q
        # 目标: minimize ||u - u_rl||^2 + 1000*slack^2
        # P = [[2*I_4,  0  ],
        #      [ 0 , 2000]]
        P = torch.zeros(batch_size, n_x, n_x, device=device, dtype=u_rl.dtype)
        P[:, :n_u, :n_u] = 2.0 * torch.eye(n_u, device=device, dtype=u_rl.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        P[:, n_u:, n_u:] = 2000.0  # slack 的权重

        # q = [-2*u_rl, 0]
        q = torch.zeros(batch_size, n_x, device=device, dtype=u_rl.dtype)
        q[:, :n_u] = -2.0 * u_rl

        # 2. 构造不等式约束 G x <= h
        # CBF 约束: A @ u - slack >= b  =>  -A @ u + slack <= -b
        G_cbf = torch.zeros(batch_size, 1, n_x, device=device, dtype=u_rl.dtype)
        G_cbf[:, :, :n_u] = -A  # -A @ u
        G_cbf[:, :, n_u] = 1.0  # + slack
        h_cbf = -b

        # 上界约束: u_i <= 1
        G_upper = torch.zeros(batch_size, n_u, n_x, device=device, dtype=u_rl.dtype)
        G_upper[:, :, :n_u] = torch.eye(n_u, device=device, dtype=u_rl.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        h_upper = torch.ones(batch_size, n_u, device=device, dtype=u_rl.dtype)

        # 下界约束: -u_i <= 1  =>  u_i >= -1
        G_lower = torch.zeros(batch_size, n_u, n_x, device=device, dtype=u_rl.dtype)
        G_lower[:, :, :n_u] = -torch.eye(n_u, device=device, dtype=u_rl.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        h_lower = torch.ones(batch_size, n_u, device=device, dtype=u_rl.dtype)

        # slack >= 0  =>  -slack <= 0
        G_slack = torch.zeros(batch_size, 1, n_x, device=device, dtype=u_rl.dtype)
        G_slack[:, :, n_u] = -1.0
        h_slack = torch.zeros(batch_size, 1, device=device, dtype=u_rl.dtype)

        # 合并所有约束
        G = torch.cat([G_cbf, G_upper, G_lower, G_slack], dim=1)
        h = torch.cat([h_cbf, h_upper, h_lower, h_slack], dim=1)

        # 3. 求解 QP
        try:
            x = qp_solver(P, q, G, h, torch.Tensor().to(device), torch.Tensor().to(device))
            u_safe = x[:, :n_u]

            # ============ 数值验证 ============
            # 检查结果是否有效（防止 NaN）
            if torch.isnan(u_safe).any() or torch.isinf(u_safe).any():
                print(f"Warning: QP solution contains NaN/Inf, returning u_rl")
                return u_rl

            return u_safe
        except Exception as e:
            # 如果 QP 求解失败，回退到原始动作
            print(f"Warning: QP solving failed: {e}, returning u_rl")
            return u_rl

    def forward(self, state, u_rl, sdf_obs):
        """
        前向传播（支持批量处理和训练/推理两种模式）

        重要改变：使用基于距离/速度的直接安全策略
        而不是基于加速度的 CBF（对四旋翼不适用）

        Args:
            state: dict with tensors (batch mode) or numpy arrays (single mode)
            u_rl: (batch_size, 4) tensor or (4,) numpy array
            sdf_obs: (batch_size, 9) tensor or (9,) numpy array

        Returns:
            u_safe: same type and shape as u_rl
        """
        if self.training:
            # 训练模式：使用可微分的速度约束
            if not torch.is_tensor(u_rl):
                raise ValueError("In training mode, u_rl must be a tensor")

            u_safe = self._apply_velocity_based_safety(state, u_rl, sdf_obs)
            return u_safe
        else:
            # 推理模式
            if torch.is_tensor(u_rl):
                is_batched = u_rl.dim() == 2
                if is_batched:
                    batch_size = u_rl.shape[0]
                    u_safe_list = []
                    for i in range(batch_size):
                        state_i = {k: v[i].cpu().numpy() if torch.is_tensor(v) else v
                                   for k, v in state.items()}
                        u_rl_i = u_rl[i].cpu().numpy()
                        sdf_obs_i = sdf_obs[i].cpu().numpy()
                        u_safe_i = self._apply_velocity_safety_numpy(state_i, u_rl_i, sdf_obs_i)
                        u_safe_list.append(torch.from_numpy(u_safe_i))
                    return torch.stack(u_safe_list).to(u_rl.device)
                else:
                    state_np = {k: v.cpu().numpy() if torch.is_tensor(v) else v
                               for k, v in state.items()}
                    u_rl_np = u_rl.cpu().numpy()
                    sdf_obs_np = sdf_obs.cpu().numpy()
                    u_safe_np = self._apply_velocity_safety_numpy(state_np, u_rl_np, sdf_obs_np)
                    return torch.from_numpy(u_safe_np).to(u_rl.device)
            else:
                return self._apply_velocity_safety_numpy(state, u_rl, sdf_obs)

    def _apply_velocity_based_safety(self, state, u_rl, sdf_obs):
        """
        基于速度和距离的安全策略（PyTorch 版本，支持梯度）

        当接近障碍且向障碍运动时，逐步降低推力
        这给了系统时间来改变姿态以避障

        Args:
            state: dict with batch tensors
            u_rl: (batch_size, 4) tensor
            sdf_obs: (batch_size, 9) tensor

        Returns:
            u_safe: (batch_size, 4) tensor，支持梯度回传
        """
        batch_size = u_rl.shape[0]
        device = u_rl.device
        u_safe = u_rl.clone()

        # 提取中心点 SDF
        sdf_h = sdf_obs[:, 4]  # (batch_size,)

        # 计算 SDF 梯度（指向安全方向）
        n, h = self.compute_sdf_gradient_batch(sdf_obs)  # n: (batch_size, 3)

        # 提取速度
        v = state['vel']  # (batch_size, 3)

        # 计算径向速度（向障碍方向）
        h_dot = torch.sum(n * v, dim=1)  # (batch_size,)

        # ========== 安全策略（使用 torch 操作，支持梯度） ==========
        SAFE_DISTANCE = 0.3  # m
        CRITICAL_DISTANCE = 0.08  # m

        # 创建安全系数张量（初始为 1.0，表示无修改）
        safety_factor = torch.ones(batch_size, device=device, dtype=u_rl.dtype)

        # 条件 1：距离接近且向障碍运动
        condition = (sdf_h < SAFE_DISTANCE) & (h_dot > 0)  # (batch_size,)

        # 对满足条件的样本，计算安全系数
        # 分两种情况：
        # - 非常接近（< CRITICAL_DISTANCE）：safety_factor = 0
        # - 接近（CRITICAL_DISTANCE 到 SAFE_DISTANCE）：线性插值

        # 情况 1：非常接近
        very_close = sdf_h < CRITICAL_DISTANCE  # (batch_size,)
        safety_factor = torch.where(very_close, torch.zeros_like(safety_factor), safety_factor)

        # 情况 2：接近但不是非常接近
        close = (sdf_h >= CRITICAL_DISTANCE) & (sdf_h < SAFE_DISTANCE)  # (batch_size,)
        linear_factor = (sdf_h - CRITICAL_DISTANCE) / (SAFE_DISTANCE - CRITICAL_DISTANCE)
        safety_factor = torch.where(close, linear_factor, safety_factor)

        # 最后，只对"向障碍运动"的样本应用安全系数
        # 对于离开障碍的（h_dot <= 0），保持 safety_factor = 1.0
        safety_factor = torch.where(h_dot > 0, safety_factor, torch.ones_like(safety_factor))

        # 应用安全系数到推力
        # 注意：这里使用广播，确保梯度正确传播
        u_safe = u_rl * safety_factor.unsqueeze(1)  # (batch_size, 4)

        return u_safe

    def _apply_velocity_safety_numpy(self, state, u_rl, sdf_obs):
        """
        基于速度和距离的安全策略（NumPy 版本，用于推理）

        Args:
            state: dict with numpy arrays
            u_rl: (4,) numpy array
            sdf_obs: (9,) numpy array

        Returns:
            u_safe: (4,) numpy array
        """
        u_safe = u_rl.copy()

        # 提取中心点 SDF
        sdf_h = sdf_obs[4]

        # 计算梯度
        n, h = self.compute_sdf_gradient(sdf_obs)

        # 提取速度
        v = state['vel']

        # 计算径向速度
        h_dot = np.dot(n, v)

        # 安全策略
        SAFE_DISTANCE = 0.3
        CRITICAL_DISTANCE = 0.08

        if sdf_h < SAFE_DISTANCE and h_dot > 0:
            if sdf_h < CRITICAL_DISTANCE:
                safety_factor = 0.0
            else:
                safety_factor = (sdf_h - CRITICAL_DISTANCE) / (SAFE_DISTANCE - CRITICAL_DISTANCE)
                safety_factor = np.clip(safety_factor, 0.0, 1.0)

            u_safe = u_rl * safety_factor

        return u_safe
