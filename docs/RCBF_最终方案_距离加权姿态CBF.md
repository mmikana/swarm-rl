# RCBF 最终方案：距离加权姿态屏障函数

**版本**: 2026-03-26  
**核心创新**: 距离加权的姿态屏障函数 + 常数安全缓冲 $\sigma$

---

## 一、屏障函数

### 定义

$$h(p, R) = \alpha(p) \cdot (n^\top Re_3) + \sigma$$

其中：
- $\alpha(p) = \exp(-k \cdot p)$：距离权重
- $p = \text{SDF}[4]$：中心点 SDF 值（距离障碍物的距离）
- $n = \nabla \text{SDF}$：SDF 梯度（指向安全方向）
- $Re_3$：推力方向（机体系 Z 轴）
- $\sigma \ge 0$：常数安全缓冲

### 物理意义

| 项 | 含义 | 作用 |
|----|------|------|
| $n^\top Re_3$ | 推力方向与安全方向的夹角 | 姿态安全 |
| $\alpha(p)$ | 距离权重 | 近距强，远距弱 |
| $\sigma$ | 安全缓冲 | 防止 $n^\top Re_3 = 0$ 时临界 |

### 行为表

| 距离 $p$ | $\alpha(p)$ | 屏障函数 $h$ | CBF 效果 |
|---------|-------------|-------------|---------|
| 0m（接触） | 1.0 | $n^\top Re_3 + \sigma$ | ✅ 强约束 |
| 0.5m | 0.37（k=2） | $0.37 \cdot n^\top Re_3 + \sigma$ | ✅ 强约束 |
| 1.0m | 0.14 | $0.14 \cdot n^\top Re_3 + \sigma$ | ⚠️ 中约束 |
| 2.0m | 0.02 | $0.02 \cdot n^\top Re_3 + \sigma$ | ❌ 弱约束 |
| >3.0m | ≈ 0 | $\sigma$ | ❌ 不约束 |

### $\sigma$ 的作用

| 场景 | $n^\top Re_3$ | 没有 $\sigma$ | 有 $\sigma$ |
|------|--------------|-------------|-----------|
| 推力安全 | > 0 | $h > 0$ | $h > 0$ |
| 推力平行 | = 0 | $h = 0$（临界） | $h = \sigma > 0$（安全） |
| 推力危险 | < 0 | $h < 0$ | $h$ 可能仍 > 0 |

**$\sigma$ 提供安全缓冲区**！✅

---

## 二、CBF 推导

### 一阶导数

$$\dot{h} = \frac{d}{dt}\left[\alpha(p) \cdot (n^\top Re_3) + \sigma\right]$$

$$= \dot{\alpha}(p) \cdot (n^\top Re_3) + \alpha(p) \cdot \frac{d}{dt}(n^\top Re_3) + \underbrace{\frac{d\sigma}{dt}}_{0}$$

其中：
$$\dot{\alpha}(p) = \frac{d}{dt}\exp(-k \cdot p) = -k \cdot \exp(-k \cdot p) \cdot \dot{p}$$
$$\dot{p} = n^\top v$$
$$\frac{d}{dt}(n^\top Re_3) = \dot{n}^\top Re_3 - (n \times Re_3)^\top \boldsymbol{\omega}$$

### 完整 $\dot{h}$

$$\dot{h} = -k \cdot \alpha(p) \cdot (n^\top v) \cdot (n^\top Re_3) + \alpha(p) \cdot \left[\dot{n}^\top Re_3 - (n \times Re_3)^\top \boldsymbol{\omega}\right]$$

$$= \alpha(p) \cdot \left[-k \cdot (n^\top v) \cdot (n^\top Re_3) + \dot{n}^\top Re_3 - (n \times Re_3)^\top \boldsymbol{\omega}\right]$$

### CBF 条件

$$\dot{h} + \alpha_{cbf} h \ge 0$$

$$\alpha(p) \cdot \left[-k \cdot (n^\top v) \cdot (n^\top Re_3) + \dot{n}^\top Re_3 - (n \times Re_3)^\top \boldsymbol{\omega}\right] + \alpha_{cbf} \cdot \left[\alpha(p) \cdot (n^\top Re_3) + \sigma\right] \ge 0$$

---

## 三、约束矩阵形式

### 整理为 $A \boldsymbol{\omega} \ge b$

$$A \boldsymbol{\omega} \ge b$$

其中：

**$A$ 矩阵** ($1 \times 3$)：
$$A = -\alpha(p) \cdot (n \times Re_3)^\top$$

**$b$ 向量** ($1 \times 1$)：
$$b = -\alpha(p) \cdot \left[-k \cdot (n^\top v) \cdot (n^\top Re_3) + \dot{n}^\top Re_3\right] - \alpha_{cbf} \cdot \left[\alpha(p) \cdot (n^\top Re_3) + \sigma\right]$$

### 展开

$$b = \alpha(p) \cdot k \cdot (n^\top v) \cdot (n^\top Re_3) - \alpha(p) \cdot \dot{n}^\top Re_3 - \alpha_{cbf} \cdot \alpha(p) \cdot (n^\top Re_3) - \alpha_{cbf} \cdot \sigma$$

---

## 四、关键特性

### 1. 距离加权

| 距离 | $\alpha(p)$ | $A$ | $b$ | 约束强度 |
|------|-------------|-----|-----|---------|
| 近 | ≈ 1 | 大 | 大 | 强 |
| 远 | ≈ 0 | ≈ 0 | ≈ $-\alpha_{cbf} \cdot \sigma$ | 弱 |

### 2. 远距离行为

$$p \to \infty \Rightarrow \alpha(p) \to 0$$

$$A \to 0, \quad b \to -\alpha_{cbf} \cdot \sigma$$

约束变为：
$$0 \cdot \boldsymbol{\omega} \ge -\alpha_{cbf} \cdot \sigma$$

$$0 \ge -\alpha_{cbf} \cdot \sigma$$

**只要 $\sigma \ge 0$，约束自动满足**！✅

### 3. $\sigma$ 的必要性

当 $n^\top Re_3 = 0$（推力平行障碍物）：

**没有 $\sigma$**：
$$h = 0, \quad b = 0$$
$$0 \cdot \boldsymbol{\omega} \ge 0 \quad \text{（临界，无约束）}$$

**有 $\sigma$**：
$$h = \sigma > 0, \quad b = -\alpha_{cbf} \cdot \sigma < 0$$
$$0 \cdot \boldsymbol{\omega} \ge -\alpha_{cbf} \cdot \sigma \quad \text{（成立，安全）}$$

---

## 五、完整实现

```python
import torch
import torch.nn as nn
from qpth.qp import QPFunction

class DistanceWeightedCBFLayer(nn.Module):
    """
    距离加权姿态 CBF-QP 层
    h(p, R) = α(p) * (nᵀRe₃) + σ
    """
    
    def __init__(self, alpha_cbf=1.0, k=2.0, sigma=0.1):
        super().__init__()
        self.alpha_cbf = alpha_cbf
        self.k = k
        self.sigma = sigma
        self.qp = QPFunction()
        
        # 动作空间限制
        self.omega_min = torch.tensor([-31.42, -31.42, -6.28])  # [wx, wy, wz]
        self.omega_max = torch.tensor([31.42, 31.42, 6.28])
        
    def compute_alpha(self, sdf_obs):
        """
        计算距离权重 α(p) = exp(-k * p)
        """
        p = sdf_obs[:, 4:5]  # 中心点 SDF 值 [batch, 1]
        alpha = torch.exp(-self.k * p)
        return alpha
    
    def compute_sdf_gradient(self, sdf_obs, resolution=0.1):
        """
        从 3x3 SDF 网格计算梯度
        """
        batch = sdf_obs.shape[0]
        h = sdf_obs[:, 4:5]  # [batch, 1]
        
        # 有限差分
        grad_x = (sdf_obs[:, 5] - sdf_obs[:, 3]) / (2 * resolution)
        grad_y = (sdf_obs[:, 7] - sdf_obs[:, 1]) / (2 * resolution)
        grad_z = torch.zeros(batch, device=sdf_obs.device)
        
        grad = torch.stack([grad_x, grad_y, grad_z], dim=1)
        norm = torch.norm(grad, dim=1, keepdim=True) + 1e-6
        n = grad / norm
        
        return n, h
    
    def compute_n_dot(self, n, v, h_sdf, R_obs=0.5):
        """
        计算 ṅᵀv（离心项）
        ṅᵀv = (‖v‖² - (nᵀv)²) / (h + R_obs)
        """
        v_squared = torch.sum(v * v, dim=1, keepdim=True)
        n_dot_v = torch.sum(n * v, dim=1, keepdim=True)
        n_dot_v = (v_squared - n_dot_v ** 2) / (h_sdf + R_obs + 1e-6)
        return n_dot_v
    
    def forward(self, rl_output, state, sdf_obs):
        """
        CBF-QP 前向传播
        
        Args:
            rl_output: [batch, 4] = [a_thrust, wx, wy, wz]
            state: {'R': [batch,3,3], 'vel': [batch,3]}
            sdf_obs: [batch, 9]
        """
        batch = rl_output.shape[0]
        device = rl_output.device
        
        # ========== 1. 计算 SDF 梯度 ==========
        n, h_sdf = self.compute_sdf_gradient(sdf_obs)
        
        # ========== 2. 提取状态 ==========
        R = state['R']
        Re3 = R[:, :, 2]  # 推力方向
        v = state['vel']
        
        # ========== 3. 计算距离权重 α(p) ==========
        alpha = self.compute_alpha(sdf_obs)  # [batch, 1]
        
        # ========== 4. 计算中间量 ==========
        n_dot_Re3 = torch.sum(n * Re3, dim=1, keepdim=True)  # nᵀRe₃
        n_dot_v = torch.sum(n * v, dim=1, keepdim=True)      # nᵀv
        
        # ṅᵀRe₃（简化：假设 ṅ 主要来自位置变化）
        # 完整计算需要 Hessian，这里用近似
        n_dot_Re3_dot = torch.zeros(batch, 1, device=device)
        
        # ========== 5. 计算 CBF 约束 A @ omega >= b ==========
        n_cross_Re3 = torch.cross(n, Re3, dim=1)  # n × Re₃
        
        # A = -α * (n × Re₃)ᵀ
        A = -alpha * n_cross_Re3  # [batch, 3]
        
        # b = α * k * (nᵀv) * (nᵀRe₃) - α * ṅᵀRe₃ - α_cbf * α * (nᵀRe₃) - α_cbf * σ
        b = (
            alpha * self.k * n_dot_v * n_dot_Re3
            - alpha * n_dot_Re3_dot
            - self.alpha_cbf * alpha * n_dot_Re3
            - self.alpha_cbf * self.sigma
        )  # [batch, 1]
        
        # ========== 6. QP 求解 ==========
        a_thrust = rl_output[:, 0:1]  # 直接通过
        omega = rl_output[:, 1:4]     # 被约束
        
        Q = 2.0 * torch.eye(3, device=device).unsqueeze(0).expand(batch, 3, 3)
        p = -2.0 * omega
        
        # 约束：A @ omega >= b  →  -A @ omega <= -b
        G = torch.cat([
            -A,  # CBF 约束
            -torch.eye(3, device=device).unsqueeze(0).expand(batch, 3, 3),  # 下界
            torch.eye(3, device=device).unsqueeze(0).expand(batch, 3, 3)    # 上界
        ], dim=1)
        
        h_qp = torch.cat([
            -b,
            -self.omega_min.unsqueeze(0).expand(batch, -1),
            self.omega_max.unsqueeze(0).expand(batch, -1)
        ], dim=1)
        
        omega_safe = self.qp(Q, p, G, h_qp)
        
        # ========== 7. 合并输出 ==========
        safe_action = torch.cat([a_thrust, omega_safe], dim=1)
        
        return safe_action
```

---

## 六、参数推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| $\alpha_{cbf}$ | 1.0-2.0 | CBF 增益 |
| $k$ | 2.0 | 距离衰减率 |
| $\sigma$ | 0.1-0.5 | 安全缓冲 |

### 调参指南

| 现象 | 调整 |
|------|------|
| 避障太激进 | 减小 $\alpha_{cbf}$ 或 $\sigma$ |
| 避障太保守 | 增大 $\alpha_{cbf}$ 或 $\sigma$ |
| 远距也约束 | 增大 $k$ |
| 近距约束弱 | 减小 $k$ |

---

## 七、关键公式汇总

### 屏障函数

$$\boxed{h(p, R) = \exp(-k \cdot p) \cdot (n^\top Re_3) + \sigma}$$

### CBF 约束

$$\boxed{-(n \times Re_3)^\top \boldsymbol{\omega} \ge k \cdot (n^\top v) \cdot (n^\top Re_3) - \dot{n}^\top Re_3 - \alpha_{cbf} \cdot (n^\top Re_3) - \frac{\alpha_{cbf} \cdot \sigma}{\exp(-k \cdot p)}}$$

### 简化实现（不除）

$$A = -\exp(-k \cdot p) \cdot (n \times Re_3)^\top$$

$$b = \exp(-k \cdot p) \cdot \left[k \cdot (n^\top v) \cdot (n^\top Re_3) - \dot{n}^\top Re_3 - \alpha_{cbf} \cdot (n^\top Re_3)\right] - \alpha_{cbf} \cdot \sigma$$

---

## 八、总结

### 核心创新

| 特性 | 说明 |
|------|------|
| ✅ 距离加权 | $\alpha(p) = \exp(-k \cdot p)$ |
| ✅ 姿态屏障 | $n^\top Re_3$ 约束转向 |
| ✅ 安全缓冲 | $\sigma$ 防止临界 |
| ✅ 数值稳定 | 远距离自然衰减 |

### 与之前方案对比

| 方案 | 屏障函数 | 远距离 | $\sigma=0$ 时 |
|------|---------|--------|-------------|
| 分段 $\beta(d)$ | $n^\top Re_3 - \beta(d)$ | 不连续 | 可用 |
| 指数衰减 | $\exp(-k \cdot d) \cdot (n^\top Re_3)$ | 连续 | 可用 |
| **本方案** | $\exp(-k \cdot p) \cdot (n^\top Re_3) + \sigma$ | **连续 + 缓冲** | **临界** |

### 适用场景

| 场景 | 效果 |
|------|------|
| 侧边障碍物 | ✅ 优秀（强制转向） |
| 前后障碍物 | ✅ 良好 |
| 密集障碍 | ✅ 距离加权生效 |
| 开放空间 | ✅ 不约束（自由飞行） |

---

**最终结论**：距离加权姿态屏障函数 + 安全缓冲 $\sigma$

- 距离近：强约束，强制转向避障
- 距离远：弱约束，自由飞行
- $\sigma > 0$：提供安全缓冲区
