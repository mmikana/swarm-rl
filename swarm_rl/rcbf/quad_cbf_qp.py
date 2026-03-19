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
        k_omega=0.1,
        R_obs=0.5,
        epsilon=0.1,
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
        self.k_omega = k_omega
        self.R_obs = R_obs
        self.epsilon = epsilon
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

        # 2. h_dot = n^T @ v
        h_dot = np.dot(n, v)

        # 3. n_dot^T @ v (离心项)
        denom = max(h + self.R_obs, 1e-6)
        n_dot_v = (np.dot(v, v) - h_dot**2) / denom

        # 4. Risk 项（角速度补偿）
        Risk = self.k_omega * np.dot(omega[:2], omega[:2]) + self.epsilon

        # 5. 控制矩阵 A
        Re3 = R @ np.array([0.0, 0.0, 1.0])
        nTRe3 = np.dot(n, Re3)
        A = (self.T_max / (2 * self.m)) * nTRe3 * np.ones(4)

        # 6. 约束标量 b
        gravity_term = np.dot(n, self.g * np.array([0.0, 0.0, 1.0]))
        bias_term = (2 * self.T_max / self.m) * nTRe3

        b = (
            Risk
            + gravity_term
            - n_dot_v
            - (self.alpha_1 + self.alpha_2) * h_dot
            - self.alpha_1 * self.alpha_2 * h
            - bias_term
        )

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
        批量计算 CBF 约束: A @ u >= b（PyTorch 版本）

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

        # 3. n_dot^T @ v (离心项)
        v_squared = torch.sum(v * v, dim=1)  # (batch_size,)
        denom = torch.clamp(h + self.R_obs, min=1e-6)
        n_dot_v = (v_squared - h_dot**2) / denom  # (batch_size,)

        # 4. Risk 项（角速度补偿）
        omega_xy_squared = torch.sum(omega[:, :2]**2, dim=1)  # (batch_size,)
        Risk = self.k_omega * omega_xy_squared + self.epsilon  # (batch_size,)

        # 5. 控制矩阵 A
        # Re3 = R @ e3, e3 = [0, 0, 1]^T
        e3 = self.e3.to(device)  # (3,)
        Re3 = torch.matmul(R, e3)  # (batch_size, 3)
        nTRe3 = torch.sum(n * Re3, dim=1)  # (batch_size,)

        # A = (T_max / (2m)) * nTRe3 * [1, 1, 1, 1]
        A_coeff = (self.T_max / (2 * self.m)) * nTRe3  # (batch_size,)
        A = A_coeff.unsqueeze(1).unsqueeze(2) * torch.ones(batch_size, 1, 4, device=device)  # (batch_size, 1, 4)

        # 6. 约束标量 b
        e3_np = torch.tensor([0.0, 0.0, 1.0], device=device)
        gravity_term = torch.sum(n * (self.g * e3_np), dim=1)  # (batch_size,)
        bias_term = (2 * self.T_max / self.m) * nTRe3  # (batch_size,)

        b = (
            Risk
            + gravity_term
            - n_dot_v
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
        qp_solver = QPFunction(verbose=False, maxIter=10000, eps=1e-4)

        device = u_rl.device
        batch_size = u_rl.shape[0]
        n_u, n_slack = 4, 1
        n_x = n_u + n_slack

        # 1. 构造目标函数 P 和 q
        # P = [[I_4,  0  ],
        #      [ 0 , 1000]]
        P = torch.zeros(batch_size, n_x, n_x, device=device, dtype=u_rl.dtype)
        P[:, :n_u, :n_u] = torch.eye(n_u, device=device, dtype=u_rl.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        P[:, n_u:, n_u:] = 1000.0

        # q = [-u_rl, 0]
        q = torch.zeros(batch_size, n_x, device=device, dtype=u_rl.dtype)
        q[:, :n_u] = -u_rl

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
            return u_safe
        except Exception as e:
            # 如果 QP 求解失败，返回原始动作
            print(f"Warning: QP solving failed: {e}, returning u_rl")
            return u_rl

    def forward(self, state, u_rl, sdf_obs):
        """
        前向传播（支持批量处理和训练/推理两种模式）

        Args:
            state: dict with tensors (batch mode) or numpy arrays (single mode)
            u_rl: (batch_size, 4) tensor or (4,) numpy array
            sdf_obs: (batch_size, 9) tensor or (9,) numpy array

        Returns:
            u_safe: same type and shape as u_rl
        """
        if self.training:
            # 训练模式：使用可微分 QP
            if not torch.is_tensor(u_rl):
                raise ValueError("In training mode, u_rl must be a tensor")

            # 批量计算约束
            A, b = self.compute_cbf_constraints_batch(state, sdf_obs)

            # 可微分 QP 求解
            u_safe = self.solve_qp_differentiable(u_rl, A, b)
            return u_safe
        else:
            # 推理模式：使用 cvxpy
            if torch.is_tensor(u_rl):
                # 如果输入是 tensor，转为 numpy 处理后再转回
                is_batched = u_rl.dim() == 2
                if is_batched:
                    # 批量推理（循环处理）
                    batch_size = u_rl.shape[0]
                    u_safe_list = []
                    for i in range(batch_size):
                        state_i = {k: v[i].cpu().numpy() for k, v in state.items()}
                        u_rl_i = u_rl[i].cpu().numpy()
                        sdf_obs_i = sdf_obs[i].cpu().numpy()
                        u_safe_i = self.get_safe_action(state_i, u_rl_i, sdf_obs_i)
                        u_safe_list.append(torch.from_numpy(u_safe_i))
                    return torch.stack(u_safe_list).to(u_rl.device)
                else:
                    # 单样本推理
                    state_np = {k: v.cpu().numpy() for k, v in state.items()}
                    u_rl_np = u_rl.cpu().numpy()
                    sdf_obs_np = sdf_obs.cpu().numpy()
                    u_safe_np = self.get_safe_action(state_np, u_rl_np, sdf_obs_np)
                    return torch.from_numpy(u_safe_np).to(u_rl.device)
            else:
                # numpy 输入（单样本推理）
                return self.get_safe_action(state, u_rl, sdf_obs)
