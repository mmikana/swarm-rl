"""
距离感知姿态 CBF-QP 层

屏障函数：h(p, R) = α(p) · (nᵀRe₃) + σ
优化空间：归一化 [-1, 1]
约束计算：物理参数转换
优化目标：min ||ω_safe - ω_rl||²（归一化空间）
"""

import torch
import torch.nn as nn

try:
    from qpth.qp import QPFunction
except ImportError:
    print("Warning: qpth not installed. CBF layer will not work.")
    QPFunction = None


class DistanceAwareCBFLayer(nn.Module):
    """
    距离感知的姿态 CBF-QP 层
    
    屏障函数：h(p, R) = α(p) · (nᵀRe₃) + σ
    优化空间：归一化 [-1, 1]
    约束计算：物理参数转换
    优化目标：min ||ω_safe - ω_rl||²（归一化空间）
    约束：A_physical @ (ω_safe ⊙ ω_scale) >= b
    """
    
    def __init__(self, alpha_cbf=1.0, k=2.0, sigma=0.1):
        super().__init__()
        self.alpha_cbf = alpha_cbf
        self.k = k
        self.sigma = sigma
        
        if QPFunction is not None:
            self.qp = QPFunction()
        else:
            self.qp = None
        
        # 物理范围缩放（归一化 → 物理）
        # ω_physical = ω_normalized * ω_scale
        self.register_buffer('omega_scale', torch.tensor([31.42, 31.42, 6.28]))
        
    def compute_alpha(self, sdf_obs):
        """
        计算距离权重 α(p) = exp(-k * p)
        
        Args:
            sdf_obs: SDF 观测 [batch, 9]
            
        Returns:
            alpha: [batch, 1]
        """
        # 使用中心点的 SDF 值作为距离
        p = sdf_obs[:, 4:5]  # [batch, 1]
        alpha = torch.exp(-self.k * p)
        return alpha
    
    def compute_sdf_gradient(self, sdf_obs, resolution=0.1):
        """
        从 3x3 SDF 网格计算梯度

        SDF 网格布局 (g_id = g_i * 3 + g_j):
            g_i=0 (x-δ=后):  [0]=(后，右)  [1]=(后，中)  [2]=(后，左)
            g_i=1 (x=中):    [3]=(中，右)  [4]=(中，中)  [5]=(中，左)
            g_i=2 (x+δ=前):  [6]=(前，右)  [7]=(前，中)  [8]=(前，左)
                             g_j=0(右)     g_j=1(中)     g_j=2(左)
        
        其中：x=前，y=左（世界系定义）
        
        梯度计算（标准命名）:
            n_x = ∂SDF/∂x = (前 - 后) / (2δ) = (SDF[7] - SDF[1]) / (2δ)
            n_y = ∂SDF/∂y = (左 - 右) / (2δ) = (SDF[5] - SDF[3]) / (2δ)

        Args:
            sdf_obs: [batch, 9]
            resolution: SDF 网格分辨率

        Returns:
            n: 归一化梯度 [batch, 3]
            p: SDF 值 [batch, 1]
        """
        batch = sdf_obs.shape[0]

        # 中心点 SDF 值
        p = sdf_obs[:, 4:5]  # [batch, 1]

        # 有限差分计算梯度（标准命名：n_x 对应∂SDF/∂x（前后），n_y 对应∂SDF/∂y（左右））
        n_y = (sdf_obs[:, 5] - sdf_obs[:, 3]) / (2 * resolution)  # ∂SDF/∂y = (左) - (右)
        n_x = (sdf_obs[:, 7] - sdf_obs[:, 1]) / (2 * resolution)  # ∂SDF/∂x = (前) - (后)
        n_z = torch.zeros(batch, device=sdf_obs.device)  # 圆柱形障碍物，z 方向无变化

        n = torch.stack([n_x, n_y, n_z], dim=1)  # [batch, 3]

        # 归一化
        norm = torch.norm(n, dim=1, keepdim=True) + 1e-6
        n = n / norm

        return n, p
    
    def compute_n_dot(self, n, Re3, state, dt=0.01):
        """
        计算 ṅᵀRe₃
        
        简化：假设 ṅ ≈ 0（梯度变化慢）
        """
        return torch.zeros(n.shape[0], 1, device=n.device)
    
    def forward(self, rl_output, state, sdf_obs):
        """
        CBF-QP 前向传播
        
        Args:
            rl_output: RL 输出 [batch, 4] = [a_thrust, wx, wy, wz]（归一化 [-1, 1]）
            state: 状态字典
                - 'R': 旋转矩阵 [batch, 3, 3]
                - 'vel': 速度 [batch, 3]
            sdf_obs: SDF 观测 [batch, 9]
            
        Returns:
            safe_action: 安全动作 [batch, 4]
                - a_thrust: [batch, 1] 不变
                - omega: [batch, 3] 物理空间（rad/s）
        """
        if self.qp is None:
            raise RuntimeError("qpth not installed. Cannot run CBF layer.")
        
        batch = rl_output.shape[0]
        device = rl_output.device
        
        # ========== 1. 计算 SDF 梯度 ==========
        n, p = self.compute_sdf_gradient(sdf_obs)  # n: [batch, 3], p: [batch, 1]
        
        # ========== 2. 提取状态 ==========
        R = state['R']  # [batch, 3, 3]
        Re3 = R[:, 2, :]  # [batch, 3] 推力方向（第3行）
        v = state['vel']  # [batch, 3] 速度
        
        # ========== 3. 计算距离权重 α(p) ==========
        alpha = self.compute_alpha(sdf_obs)  # [batch, 1]
        
        # ========== 4. 计算中间量 ==========
        n_dot_Re3 = torch.sum(n * Re3, dim=1, keepdim=True)  # nᵀRe₃ [batch, 1]
        n_dot_v = torch.sum(n * v, dim=1, keepdim=True)      # nᵀv [batch, 1]
        
        # ṅᵀRe₃（简化为 0）
        n_dot_Re3_dot = self.compute_n_dot(n, Re3, state)
        
        # ========== 5. 计算 CBF 约束（物理空间） ==========
        # A_physical @ ω_physical >= b
        n_cross_Re3 = torch.cross(n, Re3, dim=1)  # n × Re₃ [batch, 3]
        A_physical = -n_cross_Re3  # [batch, 3] 物理空间约束矩阵
        
        # b = k · (nᵀv) · (nᵀRe₃) - ṅᵀRe₃ - α_cbf · (nᵀRe₃) - (α_cbf · σ) / α(p)
        # 注意：当 α(p) → 0 时，最后一项会很大，这是期望的行为
        # 为了防止数值不稳定，设置最小值
        alpha_min = torch.clamp(alpha, min=1e-6)
        
        b = (
            self.k * n_dot_v * n_dot_Re3
            - n_dot_Re3_dot
            - self.alpha_cbf * n_dot_Re3
            - (self.alpha_cbf * self.sigma) / alpha_min
        )  # [batch, 1]
        
        # ========== 6. QP 求解（归一化空间） ==========
        # 分解动作：a_thrust 直接通过，ω 被约束
        a_thrust = rl_output[:, 0:1]  # [batch, 1]
        omega_rl = rl_output[:, 1:4]  # [batch, 3] 归一化 [-1, 1]
        
        # QP: min ||ω_safe - ω_rl||²（归一化空间）
        # 标准形式：min 0.5 * ω'Qω + p'ω
        # 展开：||ω - ω_rl||² = ω'ω - 2ω_rl'ω + ω_rl'ω_rl
        # 忽略常数项：Q = 2I, p = -2ω_rl
        Q = 2.0 * torch.eye(3, device=device).unsqueeze(0).expand(batch, 3, 3)
        p = -2.0 * omega_rl  # [batch, 3]
        
        # 约束转换：A_physical @ (ω_safe ⊙ ω_scale) >= b
        # 即：(A_physical ⊙ ω_scale) @ ω_safe >= b
        A_normalized = A_physical * self.omega_scale  # [batch, 3] 归一化空间

        # 约束：A_normalized @ ω_safe >= b
        # 加上动作限制：-1 <= ω_safe <= 1
        G_cbf = -A_normalized.unsqueeze(1)  # [batch, 1, 3]
        G_lower = -torch.eye(3, device=device).unsqueeze(0).expand(batch, 3, 3)  # [batch, 3, 3]
        G_upper = torch.eye(3, device=device).unsqueeze(0).expand(batch, 3, 3)   # [batch, 3, 3]

        G = torch.cat([G_cbf, G_lower, G_upper], dim=1)  # [batch, 7, 3]

        h_qp = torch.cat([
            -b,                                      # CBF 约束右侧 [batch, 1]
            -torch.ones(batch, 3, device=device),    # 下界：-1 [batch, 3]
            torch.ones(batch, 3, device=device)      # 上界：+1 [batch, 3]
        ], dim=1)  # [batch, 7]
        
        # 求解 QP（输出在归一化空间 [-1, 1]）
        # qpth QPFunction 需要等式约束矩阵 A_ 和 b_
        A_eq = torch.Tensor().to(device)  # 无等式约束
        b_eq = torch.Tensor().to(device)

        try:
            # 数值稳定化：规范化约束矩阵
            G_norm = torch.norm(G, dim=-1, keepdim=True).clamp(min=1e-6)
            G_normalized = G / G_norm
            h_qp_normalized = h_qp / G_norm.squeeze(-1)

            omega_safe_normalized = self.qp(Q, p, G_normalized, h_qp_normalized, A_eq, b_eq)
        except Exception as e:
            print(f"QP solving failed: {e}, returning RL output")
            omega_safe_normalized = omega_rl
        
        # 缩放回物理空间（rad/s）
        omega_safe = omega_safe_normalized * self.omega_scale  # [batch, 3]
        
        # ========== 7. 合并输出 ==========
        safe_action = torch.cat([a_thrust, omega_safe], dim=1)  # [batch, 4]
        
        return safe_action
