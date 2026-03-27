# RCBF 最终方案：距离感知姿态屏障函数

**版本**: 2026-03-27  
**核心创新**: 距离加权的姿态屏障函数 + 常数安全缓冲 $\sigma$

---

## 一、核心问题

### 位置 CBF 的局限

$$h(p) = \text{SDF}(p)$$

$$g(n^\top Re_3) \cdot a_{thrust} \ge b$$

| 问题 | 说明 |
|------|------|
| ❌ 只能约束 $a_{thrust}$ | $\boldsymbol{\omega}$ 不在约束中 |
| ❌ 侧边失效 | $n^\top Re_3 = 0$ 时约束为 0 |
| ❌ 只是"刹车" | 不能强制转向避障 |

### 物理本质

```
位置 CBF: "不要往前撞！"（限制推力）
          ↓
       无人机减速/悬停
          ↓
       ⚠️ 但不会转向
```

---

## 二、解决方案：姿态屏障函数

### 屏障函数定义

$$h(p, R) = \alpha(p) \cdot (n^\top Re_3) + \sigma$$

| 符号 | 含义 | 取值范围 |
|------|------|---------|
| $p$ | 到障碍物的距离（SDF 中心点值） | $[0, \infty)$ |
| $n$ | SDF 梯度（指向安全方向） | 单位向量 |
| $Re_3$ | 推力方向（机体系 Z 轴） | 单位向量 |
| $n^\top Re_3$ | 推力方向与安全方向的夹角余弦 | $[-1, 1]$ |
| $\alpha(p)$ | 距离权重 | $(0, 1]$ |
| $\sigma$ | 常数安全缓冲 | $[0.1, 0.5]$ |

### 距离权重函数

$$\alpha(p) = \exp(-k \cdot p)$$

| 距离 $p$ | $\alpha(p)$ (k=2) | 约束强度 |
|---------|------------------|---------|
| 0m（接触） | 1.0 | 🔴 最强 |
| 0.5m | 0.37 | 🔴 强 |
| 1.0m | 0.14 | 🟡 中等 |
| 2.0m | 0.02 | 🟢 弱 |
| >3.0m | ≈ 0 | ⚪ 无约束 |

### 物理意义

| $n^\top Re_3$ | 推力方向 | 含义 |
|--------------|---------|------|
| 1.0 | 完全指向安全方向 | ✅ 最安全 |
| 0.5 | 60° 夹角 | ✅ 安全 |
| 0.0 | 平行于障碍物（悬停） | ✅ 悬停安全 |
| -0.5 | 斜向障碍物 | ⚠️ 危险 |
| -1.0 | 直指障碍物 | ❌ 最危险 |

### $\sigma$ 的作用

| 场景 | $n^\top Re_3$ | 没有 $\sigma$ | 有 $\sigma=0.1$ |
|------|--------------|-------------|---------------|
| 推力安全 | > 0 | $h > 0$ | $h > 0$ |
| 悬停 | = 0 | $h = 0$（临界） | $h = 0.1 > 0$（安全）✅ |
| 推力危险 | < 0 | $h < 0$ | $h$ 可能仍 < 0 |

**$\sigma$ 提供安全缓冲区，悬停时不干预**！✅

---

## 三、CBF 推导

### 一阶导数

$$\dot{h} = \frac{d}{dt}\left[\alpha(p) \cdot (n^\top Re_3) + \sigma\right]$$

$$= \dot{\alpha}(p) \cdot (n^\top Re_3) + \alpha(p) \cdot \frac{d}{dt}(n^\top Re_3) + \underbrace{\frac{d\sigma}{dt}}_{0}$$

### 计算各项

#### 1. $\dot{\alpha}(p)$

$$\alpha(p) = \exp(-k \cdot p)$$

$$\dot{\alpha}(p) = \frac{d}{dt}\exp(-k \cdot p) = -k \cdot \exp(-k \cdot p) \cdot \dot{p} = -k \cdot \alpha(p) \cdot \dot{p}$$

#### 2. $\dot{p}$（距离变化率）

$$\dot{p} = \frac{d}{dt}\text{SDF}(p) = n^\top v$$

其中 $v$ 是无人机速度（世界系）。

#### 3. $\frac{d}{dt}(n^\top Re_3)$

$$\frac{d}{dt}(n^\top Re_3) = \dot{n}^\top Re_3 + n^\top \frac{d}{dt}(Re_3)$$

其中：
$$\frac{d}{dt}(Re_3) = \boldsymbol{\omega} \times Re_3$$

使用恒等式：
$$n^\top(\boldsymbol{\omega} \times Re_3) = -(n \times Re_3)^\top \boldsymbol{\omega}$$

所以：
$$\frac{d}{dt}(n^\top Re_3) = \dot{n}^\top Re_3 - (n \times Re_3)^\top \boldsymbol{\omega}$$

### 完整 $\dot{h}$

$$\dot{h} = -k \cdot \alpha(p) \cdot (n^\top v) \cdot (n^\top Re_3) + \alpha(p) \cdot \left[\dot{n}^\top Re_3 - (n \times Re_3)^\top \boldsymbol{\omega}\right]$$

$$= \alpha(p) \cdot \left[-k \cdot (n^\top v) \cdot (n^\top Re_3) + \dot{n}^\top Re_3 - (n \times Re_3)^\top \boldsymbol{\omega}\right]$$

---

## 四、CBF 条件

### 标准形式

$$\dot{h} + \alpha_{cbf} h \ge 0$$

代入：

$$\alpha(p) \cdot \left[-k \cdot (n^\top v) \cdot (n^\top Re_3) + \dot{n}^\top Re_3 - (n \times Re_3)^\top \boldsymbol{\omega}\right] + \alpha_{cbf} \cdot \left[\alpha(p) \cdot (n^\top Re_3) + \sigma\right] \ge 0$$

### 整理为约束形式

$$-(n \times Re_3)^\top \boldsymbol{\omega} \ge k \cdot (n^\top v) \cdot (n^\top Re_3) - \dot{n}^\top Re_3 - \alpha_{cbf} \cdot (n^\top Re_3) - \frac{\alpha_{cbf} \cdot \sigma}{\alpha(p)}$$

### 简化版本（推荐）

**忽略 $\dot{n}^\top Re_3$ 项**（梯度变化慢）：

$$-(n \times Re_3)^\top \boldsymbol{\omega} \ge k \cdot (n^\top v) \cdot (n^\top Re_3) - \alpha_{cbf} \cdot (n^\top Re_3) - \frac{\alpha_{cbf} \cdot \sigma}{\alpha(p)}$$

---

## 五、约束矩阵形式

### 物理空间约束

$$A_{physical} \boldsymbol{\omega}_{physical} \ge b$$

其中：

**$A_{physical}$ 矩阵** ($1 \times 3$)：
$$A_{physical} = -(n \times Re_3)^\top$$

**$b$ 向量** ($1 \times 1$)：
$$b = k \cdot (n^\top v) \cdot (n^\top Re_3) - \alpha_{cbf} \cdot (n^\top Re_3) - \frac{\alpha_{cbf} \cdot \sigma}{\alpha(p)}$$

### 归一化空间转换

为了在归一化空间求解 QP，需要转换约束：

$$\boldsymbol{\omega}_{physical} = \boldsymbol{\omega}_{safe} \odot \boldsymbol{\omega}_{scale}$$

代入约束：

$$A_{physical} (\boldsymbol{\omega}_{safe} \odot \boldsymbol{\omega}_{scale}) \ge b$$

$$(A_{physical} \odot \boldsymbol{\omega}_{scale}) \boldsymbol{\omega}_{safe} \ge b$$

所以归一化空间的约束矩阵：

$$A_{normalized} = A_{physical} \odot \boldsymbol{\omega}_{scale}$$

---

## 六、完整实现

```python
import torch
import torch.nn as nn
from qpth.qp import QPFunction

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
        self.qp = QPFunction()
        
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
            h: SDF 值 [batch, 1]
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
        batch = rl_output.shape[0]
        device = rl_output.device
        
        # ========== 1. 计算 SDF 梯度 ==========
        n, p = self.compute_sdf_gradient(sdf_obs)  # n: [batch, 3], p: [batch, 1]
        
        # ========== 2. 提取状态 ==========
        R = state['R']  # [batch, 3, 3]
        Re3 = R[:, :, 2]  # [batch, 3] 推力方向
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
        G = torch.cat([
            -A_normalized,                           # CBF 约束（归一化空间）[batch, 3]
            -torch.eye(3, device=device).unsqueeze(0).expand(batch, 3, 3),  # 下界
            torch.eye(3, device=device).unsqueeze(0).expand(batch, 3, 3)    # 上界
        ], dim=1)  # [batch, 9]
        
        h_qp = torch.cat([
            -b,                                      # CBF 约束右侧
            -torch.ones(batch, 3, device=device),    # 下界：-1
            torch.ones(batch, 3, device=device)      # 上界：+1
        ], dim=1)  # [batch, 9]
        
        # 求解 QP（输出在归一化空间 [-1, 1]）
        omega_safe_normalized = self.qp(Q, p, G, h_qp)  # [batch, 3]
        
        # 缩放回物理空间（rad/s）
        omega_safe = omega_safe_normalized * self.omega_scale  # [batch, 3]
        
        # ========== 7. 合并输出 ==========
        safe_action = torch.cat([a_thrust, omega_safe], dim=1)  # [batch, 4]
        
        return safe_action
```

---

## 七、QP 优化项详解

### 优化目标

$$\min_{\boldsymbol{\omega}_{safe}} \quad \|\boldsymbol{\omega}_{safe} - \boldsymbol{\omega}_{rl}\|^2$$

**含义**：安全动作尽可能接近 RL 输出，**最小化干预**。

### 推导

$$\|\boldsymbol{\omega}_{safe} - \boldsymbol{\omega}_{rl}\|^2 = (\boldsymbol{\omega}_{safe} - \boldsymbol{\omega}_{rl})^\top (\boldsymbol{\omega}_{safe} - \boldsymbol{\omega}_{rl})$$

展开：
$$= \boldsymbol{\omega}_{safe}^\top \boldsymbol{\omega}_{safe} - 2 \boldsymbol{\omega}_{rl}^\top \boldsymbol{\omega}_{safe} + \boldsymbol{\omega}_{rl}^\top \boldsymbol{\omega}_{rl}$$

忽略常数项 $\boldsymbol{\omega}_{rl}^\top \boldsymbol{\omega}_{rl}$：

$$= \frac{1}{2} \boldsymbol{\omega}_{safe}^\top (2I) \boldsymbol{\omega}_{safe} + (-2 \boldsymbol{\omega}_{rl})^\top \boldsymbol{\omega}_{safe}$$

### 标准 QP 形式

$$\min \quad \frac{1}{2} \boldsymbol{\omega}^\top Q \boldsymbol{\omega} + p^\top \boldsymbol{\omega}$$

其中：
$$Q = 2I$$
$$p = -2 \boldsymbol{\omega}_{rl}$$

### 为什么这样设计？

| 设计 | 理由 |
|------|------|
| **最小干预** | RL 已学会最优策略，CBF 只保安全 |
| **简单** | 无需额外超参数 |
| **高效** | Q = 2I 是常数，计算快 |

---

## 八、训练集成

```python
# 初始化
policy = PolicyNetwork()
cbf_layer = DistanceAwareCBFLayer(alpha_cbf=1.0, k=2.0, sigma=0.1)
optimizer = torch.optim.Adam(policy.parameters())

# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    for t in range(max_steps):
        # 1. RL 输出
        rl_output = policy(state)  # [4]
        
        # 2. CBF 修正（可微）
        safe_action = cbf_layer(rl_output, state, sdf_obs)
        
        # 3. 环境交互
        next_state, reward, done = env.step(safe_action)
        
        # 4. 训练（PPO）
        loss = compute_ppo_loss(policy, state, safe_action, reward)
        loss.backward()  # 梯度通过 CBF-QP 层传回！
        optimizer.step()
        
        state = next_state
```

---

## 九、行为演示

### 场景 1：悬停（$n^\top Re_3 = 0$）

```
距离：p = 0.5m
α = exp(-2 × 0.5) = 0.37
n·Re3 = 0（推力水平，悬停）
n·v = 0（静止）
σ = 0.1

h = 0.37 × 0 + 0.1 = 0.1 > 0
```

**CBF 不干预** ✅ **正确**（悬停，安全）

---

### 场景 2：近距离强制避障

```
距离：p = 0.5m
α = 0.37
n·Re3 = -0.5（推力指向障碍）
n·v = -1.0（靠近）
σ = 0.1

h = 0.37 × (-0.5) + 0.1 = -0.085 < 0
b = 2 × (-1.0) × (-0.5) - 1 × (-0.5) - (1 × 0.1) / 0.37
  = 1.0 + 0.5 - 0.27 = 1.23

约束：-(n×Re3)ᵀω >= 1.23
```

**CBF 强制转向** ✅

---

### 场景 3：远距离自由飞行

```
距离：p = 3.0m
α = exp(-2 × 3.0) ≈ 0.002
n·Re3 = 0.5（推力安全）
n·v = 1.0（远离）
σ = 0.1

h = 0.002 × 0.5 + 0.1 ≈ 0.1 > 0
b ≈ 0 - 0 - (1 × 0.1) / 0.002 = -50

约束：-(n×Re3)ᵀω >= -50
```

**约束自动满足** ✅ **CBF 不干预**

---

### 场景 4：中距离引导

```
距离：p = 1.0m
α = exp(-2 × 1.0) = 0.14
n·Re3 = 0.0（推力平行）
n·v = -0.5（靠近）
σ = 0.1

h = 0.14 × 0 + 0.1 = 0.1 > 0
b = 2 × (-0.5) × 0 - 0 - (1 × 0.1) / 0.14 = -0.71

约束：-(n×Re3)ᵀω >= -0.71
```

**约束较弱，温和引导** ✅

---

## 十、参数调优指南

### $\alpha_{cbf}$（CBF 增益）

| 值 | 效果 |
|----|------|
| 0.5 | 温和，允许短暂违规 |
| 1.0 | 平衡（推荐） |
| 2.0 | 激进，严格约束 |

### $k$（距离衰减率）

| 值 | 效果 |
|----|------|
| 1.0 | 缓慢衰减，约束范围大 |
| 2.0 | 平衡（推荐） |
| 3.0 | 快速衰减，约束范围小 |

### $\sigma$（安全缓冲）

| 值 | 效果 |
|----|------|
| 0.1 | 小缓冲，悬停时轻微约束 |
| 0.2 | 平衡（推荐） |
| 0.5 | 大缓冲，悬停时不约束 |

---

## 十一、总结

### 核心创新

| 特性 | 说明 |
|------|------|
| ✅ 距离加权 | $\alpha(p) = \exp(-k \cdot p)$ |
| ✅ 姿态屏障 | $n^\top Re_3$ 约束转向 |
| ✅ 安全缓冲 | $\sigma$ 防止临界 |
| ✅ 悬停不干预 | $h = \sigma > 0$ 时自动满足 |
| ✅ 可微 QP 层 | 梯度传回 RL，引导学习 |

### 与之前方案对比

| 方案 | 屏障函数 | 悬停 ($n^\top Re_3=0$) | 远距离 |
|------|---------|---------------------|--------|
| 位置 CBF | SDF(p) | ❌ 不适用 | ❌ 不连续 |
| 距离加权 | $\alpha(p) \cdot (n^\top Re_3)$ | $h=0$（临界） | ✅ 连续 |
| **本方案** | $\alpha(p) \cdot (n^\top Re_3) + \sigma$ | **$h=\sigma > 0$** ✅ | ✅ 连续 |

### 适用场景

| 场景 | 效果 |
|------|------|
| 侧边障碍物 | ✅ 优秀（强制转向） |
| 前后障碍物 | ✅ 良好 |
| 悬停 | ✅ 不干预 |
| 密集障碍 | ✅ 距离加权生效 |
| 开放空间 | ✅ 不约束（自由飞行） |

---

## 十二、关键公式汇总

### 屏障函数

$$\boxed{h(p, R) = \exp(-k \cdot p) \cdot (n^\top Re_3) + \sigma}$$

### CBF 约束

$$\boxed{-(n \times Re_3)^\top \boldsymbol{\omega} \ge k \cdot (n^\top v) \cdot (n^\top Re_3) - \dot{n}^\top Re_3 - \alpha_{cbf} \cdot (n^\top Re_3) - \frac{\alpha_{cbf} \cdot \sigma}{\exp(-k \cdot p)}}$$

### 简化实现

$$A = -(n \times Re_3)^\top$$

$$b = k \cdot (n^\top v) \cdot (n^\top Re_3) - \alpha_{cbf} \cdot (n^\top Re_3) - \frac{\alpha_{cbf} \cdot \sigma}{\exp(-k \cdot p)}$$

---

**最终结论**：距离感知姿态屏障函数 + 安全缓冲 $\sigma$

- 距离近：强约束，强制转向避障
- 距离远：弱约束，自由飞行
- 悬停时：$h = \sigma > 0$，不干预 ✅
- $\sigma > 0$：提供安全缓冲区
