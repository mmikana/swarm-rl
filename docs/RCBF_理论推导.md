# RCBF 理论推导：推力空间（球体碰撞模型）

> **版本**: 2026-03-05（RL 环境参数版）
> **假设**: 无人机碰撞模型为球体（半径 $R_{drone}$ 为常数），姿态不影响碰撞边界
> **控制输入**: $u \in [0, 1]^4$ (归一化电机推力)
> **动力学模型**: Crazyflie（`gym_art/quadrotor_multi/quad_models.py`）

---

## 符号表

| 符号 | 含义 | 单位 | RL 环境实际值 |
|------|------|------|------|
| $p \in \mathbb{R}^3$ | 无人机位置（质心） | m | - |
| $v \in \mathbb{R}^3$ | 无人机速度 | m/s | $\|v\| \le 3.0$ |
| $R \in SO(3)$ | 旋转矩阵（机体系→惯性系） | - | - |
| $u \in [0,1]^4$ | 归一化电机推力 | - | - |
| $T_{max}$ | 单电机最大推力 | N | 0.130 N |
| $m$ | 无人机质量 | kg | 0.028 kg |
| $g$ | 重力加速度 | m/s² | 9.81 |
| $e_3$ | 惯性系 z 轴单位向量 $[0,0,1]^\top$ | - | - |
| $p_{obs}$ | 障碍物位置 | m | - |
| $R_{obs}$ | 障碍物半径 | m | 0.15 ~ 0.5 (`obst_size/2`) |
| $R_{drone}$ | 无人机球体半径 | m | 0.046 (碰撞模型参数) |
| $d_{safe}$ | 安全裕度 | m | 0.2 ~ 0.3 |
| $n$ | 避障方向单位法向量 | - | - |
| $\alpha_1, \alpha_2$ | CBF 增益参数 | s⁻¹ | 1.0-3.0 |
| $d$ | 外部扰动加速度 | m/s² | $\|d\| \le d_{max}$ |
| $\omega$ | 机身角速度 | rad/s | $\|\omega\| \le 40$ |

**坐标系约定**:
- **惯性系** $\{I\}$: $e_3 = [0, 0, 1]^\top$ 指向上方
- **机体系** $\{B\}$: $z_b$ 轴垂直于机身平面向下
- **推力方向**: $R e_3$ 表示机体系 $z_b$ 轴在惯性系中的方向（即推力矢量方向，指向上方为正）

**RL 环境参数说明**（基于 `gym_art/quadrotor_multi/quad_models.py` 和 `gym_art/quadrotor_multi/obstacles/`）:
- 质量 $m = 0.028 \, \text{kg}$（body + payload + arms + motors + propellers）
- 推力重量比 `thrust_to_weight` = 1.9
- 单电机最大推力：$T_{max} = \frac{m \cdot g \cdot \text{thrust\_to\_weight}}{4} = \frac{0.028 \cdot 9.81 \cdot 1.9}{4} \approx 0.130 \, \text{N}$
- 电机臂长：$\text{arm} = \|\text{motor}_{xy}\| = 0.046 \, \text{m}$
- **无人机碰撞半径**：$R_{drone} = 0.046 \, \text{m}$（`obstacles.py` 中的 `quad_radius`）
- 房间尺寸：默认 $10 \times 10 \times 10$ m（`quads_room_dims`）
- 最大速度：$v_{max} = 3.0 \, \text{m/s}$（`vxyz_max`）
- 最大角速度：$\omega_{max} = 40 \, \text{rad/s}$（`omega_max`）
- **障碍物半径**：$R_{obs} = \frac{\text{obst\_size}}{2}$，默认 0.5 m（`obstacles.py` 中的 `obstacle_radius = obstacle_size / 2.0`）
- 障碍物尺寸配置：`quads_obst_size` 默认 1.0 m（直径），随机化时 0.3 ~ 0.6 m
- 碰撞半径：$2.0 \times \text{arm} \approx 0.092 \, \text{m}$（`quads_collision_hitbox_radius`）

---

## 一、RCBF 二阶级联推导

由于控制输入 $u$（电机推力）在位置 $h$ 的二阶导数中才显式出现（相对度为 2），我们需要构造级联的屏障函数。

### 1. 定义原始屏障函数 $h$

**障碍物屏障**:
$$h(p) = \|p - p_{obs}\| - (R_{obs} + R_{drone} + d_{safe})$$

其中:
- $R_{drone}$: 无人机球体半径（常数，如 0.2m）
- $R_{obs}$: 障碍物半径
- $d_{safe}$: 安全裕度（如 0.3m）
- $n = \frac{p - p_{obs}}{\|p - p_{obs}\|}$: 避障方向单位法向量（从障碍物指向无人机）

**梯度**: $\nabla_p h = n^\top$

**量纲**: $[h] = \text{m}$（长度）

### 2. 第一层级联：定义 $\psi$

为了让安全性包含速度信息，定义：
$$\psi = \dot{h} + \alpha_1 h, \quad \alpha_1 > 0$$

其中:
$$\dot{h} = \nabla_p h^\top \dot{p} = n^\top v$$

**量纲**: $[\dot{h}] = \text{m/s}$, $[\psi] = \text{m/s}$

**物理意义**: 只要 $\psi \ge 0$，就能保证 $h \ge 0$（安全性）。这是因为 $\dot{h} = -\alpha_1 h + \psi$，当 $\psi \ge 0$ 时，$h$ 按指数衰减但不会穿过零点。

### 3. 第二层级联：应用鲁棒 RCBF 条件

对 $\psi$ 应用鲁棒约束公式：
$$\dot{\psi} \ge -\alpha_2 \psi + \text{Risk}, \quad \alpha_2 > 0$$

**展开左侧**:
$$\dot{\psi} = \ddot{h} + \alpha_1 \dot{h}$$

**代入 RCBF 条件**:
$$\ddot{h} + \alpha_1 \dot{h} \ge -\alpha_2 (\dot{h} + \alpha_1 h) + \text{Risk}$$

**移项整理**（关键步骤）:
$$\ddot{h} + \alpha_1 \dot{h} + \alpha_2 \dot{h} + \alpha_1 \alpha_2 h \ge \text{Risk}$$

$$\boxed{\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge \text{Risk}}$$

**量纲**: $[\ddot{h}] = \text{m/s}^2$, $[\alpha_i \dot{h}] = \text{s}^{-1} \cdot \text{m/s} = \text{m/s}^2$, $[\alpha_1 \alpha_2 h] = \text{s}^{-2} \cdot \text{m} = \text{m/s}^2$

---

## 二、提取控制矩阵 $A$ 与向量 $b$

我们需要将 RCBF 安全方程转化为 $A u \ge b$ 的形式，以便送入 QP 求解器。

### 1. 展开 $\ddot{h}$

$$\ddot{h} = \frac{d}{dt}(n^\top v) = n^\top \dot{v} + \dot{n}^\top v$$

**维度分析**:
- $n^\top \dot{v}$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $\dot{n}^\top v$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）

**关键：$\dot{n}$ 的显式计算**

由于 $n = \frac{p - p_{obs}}{\|p - p_{obs}\|}$，对时间求导：

$$\dot{n} = \frac{d}{dt}\left(\frac{p - p_{obs}}{\|p - p_{obs}\|}\right) = \frac{v}{\|p - p_{obs}\|} - \frac{(p - p_{obs})}{\|p - p_{obs}\|^3} (p - p_{obs})^\top v$$

$$\dot{n} = \frac{v}{\|p - p_{obs}\|} - \frac{n (n^\top v)}{\|p - p_{obs}\|} = \frac{1}{\|p - p_{obs}\|} \left( v - n (n^\top v) \right)$$

$$\boxed{\dot{n} = \frac{1}{\|p - p_{obs}\|} \left( v - n (n^\top v) \right)}$$

**维度分析**:
- $v$: $3 \times 1$
- $n^\top v$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $n (n^\top v)$: $(3 \times 1) \cdot 1 \to 3 \times 1$
- $v - n (n^\top v)$: $3 \times 1 - 3 \times 1 \to 3 \times 1$
- $\frac{1}{\|p - p_{obs}\|}$: 标量
- **结果**: $\dot{n} \in \mathbb{R}^{3 \times 1}$

**物理意义**: $\dot{n}$ 是速度在**垂直于 $n$ 方向**的分量除以距离。这是向心加速度项，表示当无人机沿切向运动时，法向量 $n$ 的旋转速率。

**$\dot{n}^\top v$ 的物理解释**:
$$\dot{n}^\top v = \frac{1}{\|p - p_{obs}\|} \left( \|v\|^2 - (n^\top v)^2 \right) = \frac{\|v_\perp\|^2}{\|p - p_{obs}\|}$$

**维度分析**:
- $\dot{n}^\top$: $1 \times 3$
- $v$: $3 \times 1$
- **结果**: $\dot{n}^\top v \to 1$（标量）

其中 $v_\perp = v - (n^\top v)n$ 是速度在垂直于 $n$ 方向的分量。这是**离心加速度项**，表示切向运动导致的"远离"效应。

**量纲**: $[\dot{n}] = \text{s}^{-1}$, $[\dot{n}^\top v] = \text{m/s}^2$

### 2. 代入无人机平移动力学

无人机平移动力学：
$$\dot{v} = \frac{T_{max}}{m} R e_3 (\mathbf{1}^\top u) - g e_3 + d$$

**维度分析**:
- $\frac{T_{max}}{m}$: $\text{N}/\text{kg} = \text{m/s}^2$（标量）
- $R e_3$: $(3 \times 3) \cdot (3 \times 1) \to 3 \times 1$
- $\mathbf{1}^\top u$: $(1 \times 4) \cdot (4 \times 1) \to 1$（标量，总推力比例）
- $\frac{T_{max}}{m} R e_3 (\mathbf{1}^\top u)$: $\text{m/s}^2 \cdot (3 \times 1) \cdot 1 \to 3 \times 1$
- $g e_3$: $\text{m/s}^2 \cdot (3 \times 1) \to 3 \times 1$
- $d$: $3 \times 1$
- **结果**: $\dot{v} \in \mathbb{R}^{3 \times 1}$

其中:
- $R e_3 \in \mathbb{R}^3$: 推力方向单位向量（机体系 z 轴在惯性系中的方向）
- $\mathbf{1}^\top = [1, 1, 1, 1] \in \mathbb{R}^{1 \times 4}$
- $\mathbf{1}^\top u = \sum_{i=1}^4 u_i$: 总推力比例
- $d \in \mathbb{R}^3$: 外部扰动加速度，$\|d\| \le d_{max}$

代入 $\ddot{h}$：

$$\ddot{h} = n^\top \left( \frac{T_{max}}{m} R e_3 (\mathbf{1}^\top u) - g e_3 + d \right) + \dot{n}^\top v$$

**维度分析**:
- $n^\top$: $1 \times 3$
- $\frac{T_{max}}{m} R e_3 (\mathbf{1}^\top u)$: $3 \times 1$
- $n^\top \cdot \frac{T_{max}}{m} R e_3 (\mathbf{1}^\top u)$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- 提取 $u$ 后：$n^\top \frac{T_{max}}{m} R e_3 \mathbf{1}^\top u = \left[\frac{T_{max}}{m} (n^\top R e_3) \mathbf{1}^\top\right] u$

$$\ddot{h} = \underbrace{\left[ \frac{T_{max}}{m} (n^\top R e_3) \mathbf{1}^\top \right]}_{A \in \mathbb{R}^{1 \times 4}} u - n^\top g e_3 + \dot{n}^\top v + n^\top d$$

**$A$ 矩阵维度推导**:
$$
\begin{aligned}
A &= \frac{T_{max}}{m} (n^\top R e_3) \mathbf{1}^\top \\
&= \underbrace{\frac{T_{max}}{m}}_{\text{标量}} \cdot \underbrace{(n^\top R e_3)}_{1 \times 1} \cdot \underbrace{\mathbf{1}^\top}_{1 \times 4} \\
&\to 1 \times 4
\end{aligned}
$$

### 3. 确定约束项

**$A$ 矩阵** ($1 \times 4$): 表征了总推力在避障法线 $n$ 上的投影效率。

$$A = \frac{T_{max}}{m} (n^\top R e_3) \mathbf{1}^\top = \frac{T_{max}}{m} (n^\top R e_3) \begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix}$$

**维度推导**:
$$
\begin{bmatrix}
A_1 & A_2 & A_3 & A_4
\end{bmatrix}
= \frac{T_{max}}{m} (n^\top R e_3)
\begin{bmatrix}
1 & 1 & 1 & 1
\end{bmatrix}
$$

**物理解释**: 
- $n^\top R e_3$: 推力方向与避障法线的夹角的余弦（标量）
- $\mathbf{1}^\top u$: 四个电机的总推力比例（标量）
- 因为球体模型下，四个电机对位置加速度的影响相同，所以 $A$ 的四个元素相同

**量纲**: $[A] = \frac{\text{N}}{\text{kg}} \cdot 1 = \text{m/s}^2$（每单位 $u$ 产生的加速度）

**$A u$ 维度验证**:
$$(1 \times 4) \cdot (4 \times 1) \to 1 \times 1 \quad \text{（标量，与 } \ddot{h} \text{ 一致）}$$

**$b$ 标量**:

将 $\ddot{h}$ 代入 RCBF 条件 $\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge \text{Risk}$：

$$A u - n^\top g e_3 + \dot{n}^\top v + n^\top d + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge \text{Risk}$$

**各项维度检查**:
- $A u$: $(1 \times 4) \cdot (4 \times 1) \to 1$（标量）
- $n^\top g e_3$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $\dot{n}^\top v$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $n^\top d$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $(\alpha_1 + \alpha_2)\dot{h}$: $\text{s}^{-1} \cdot \text{m/s} \to \text{m/s}^2$（标量）
- $\alpha_1 \alpha_2 h$: $\text{s}^{-2} \cdot \text{m} \to \text{m/s}^2$（标量）
- $\text{Risk}$: $\text{m/s}^2$（标量）

移项得到 $A u \ge b$ 的形式：

$$A u \ge \text{Risk} - n^\top d + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)\dot{h} - \alpha_1 \alpha_2 h$$

考虑最坏情况扰动 $n^\top d \ge -\|d\| \ge -d_{max}$，得到：

$$\boxed{b = \text{Risk} + d_{max} + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)\dot{h} - \alpha_1 \alpha_2 h}$$

**量纲检查**: $[b] = \text{m/s}^2$ ✅

**$A u \ge b$ 维度验证**:
$$(1 \times 4) \cdot (4 \times 1) \ge 1 \quad \Rightarrow \quad 1 \ge 1 \quad \text{✓}$$

---

## 三、Risk 项的具体设计（论文创新点）

由于球体模型忽略了姿态，我们将论文中的干扰项 $\min \nabla h^\top \Psi$ 具象化为针对欠驱动特性的风险补偿。

### 1. Risk 项的说明

**重要**: $d_{max}$ 是从扰动项 $n^\top d$ 推导出的最坏情况边界（见第二节 $b$ 的推导），**不属于** Risk 的设计部分。

Risk 项专门针对**模型特定的风险**进行补偿：

### 2. 姿态波动风险项 $\text{Risk}_{att}$

无人机产生水平加速度必须先进行 Roll/Pitch 翻转，这会导致瞬时升力损失。

$$\text{Risk}_{att} = k_\omega \|\omega_{xy}\|^2$$

其中:
- $\omega_{xy} = [\omega_x, \omega_y]^\top$: 机身角速度的水平分量（Roll/Pitch 速率）
- $k_\omega$: 角速度增益系数

**量纲分析**:
- $[\omega] = \text{rad/s}$
- $[\|\omega_{xy}\|^2] = \text{rad}^2/\text{s}^2$
- $[\text{Risk}_{att}] = \text{m/s}^2$

因此 $k_\omega$ 的量纲为：
$$[k_\omega] = \frac{\text{m/s}^2}{\text{rad}^2/\text{s}^2} = \text{m/rad}^2$$

**物理意义**: $k_\omega$ 表示单位角速度平方对应的加速度风险补偿。典型值 $k_\omega = 0.05 \sim 0.2 \, \text{m/rad}^2$。

**设计逻辑**: 角速度 $\omega$ 越大，说明推力矢量偏转越剧烈，位置控制的不确定性越高。

**效果**: 当无人机剧烈翻转时，该项会拉高 $b$ 值，强制 RCBF 预留出更多空间来抵消"姿态调整"期间的高度掉落。

### 3. 干扰估计项 $\text{Risk}_{dist}$

利用高斯过程或历史数据估计的扰动标准差 $\sigma(x)$：

$$\text{Risk}_{dist} = k_c \cdot \sigma(x)$$

其中:
- $\sigma(x)$: 扰动加速度的估计标准差（单位：$\text{m/s}^2$）
- $k_c$: 置信度参数（如 $k_c=2$ 代表 95.5% 置信度，无量纲）

**量纲**: $[\text{Risk}_{dist}] = 1 \cdot \text{m/s}^2 = \text{m/s}^2$ ✅

**设计逻辑**: $k_c$ 是置信度参数（如 $k_c=2$ 代表 95.5% 置信度）。

**效果**: 确保在存在风场或电机噪声时，安全边界依然有效。

### 4. 最终 Risk 表达式

$$\boxed{\text{Risk} = k_\omega \|\omega_{xy}\|^2 + k_c \sigma(x) + \epsilon}$$

其中 $\epsilon$ 是小常数（如 $\epsilon = 0.1 \, \text{m/s}^2$）用于数值稳定性。

**量纲检查**: 所有项均为 $\text{m/s}^2$ ✅

**注意**: $d_{max}$ 已经包含在 $b$ 的表达式中（来自 $n^\top d$ 的最坏情况估计），不在 Risk 中重复出现。

---

## 四、完整 RCBF-QP 公式

### 1. 优化问题

给定 RL 策略输出 $u_{nom} \in [0, 1]^4$，求解安全控制：

$$\boxed{
\begin{aligned}
u^* = \arg\min_{u \in \mathbb{R}^4} \quad & \|u - u_{nom}\|^2 \\
\text{s.t.} \quad & A_{obs,i} u \ge b_{obs,i}, \quad i = 1,\dots,n_{obs} \quad (\text{障碍物}) \\
& 0 \le u_i \le 1, \quad i = 1,2,3,4
\end{aligned}
}$$

### 2. 约束矩阵汇总

将所有约束组装成矩阵形式 $G u \le h$（标准 QP 格式）：

$$G = \begin{bmatrix}
-A_{obs} \\
I_4 \\
-I_4
\end{bmatrix} \in \mathbb{R}^{(n_{obs}+8) \times 4}, \quad
h = \begin{bmatrix}
-b_{obs} \\
\mathbf{1}_4 \\
\mathbf{0}_4
\end{bmatrix} \in \mathbb{R}^{n_{obs}+8}$$

其中 $A_{obs} \in \mathbb{R}^{n_{obs} \times 4}$，$b_{obs} \in \mathbb{R}^{n_{obs}}$。

### 3. 参数选择指南

| 参数 | 符号 | 推荐值 | 量纲 | 影响 |
|------|------|--------|------|------|
| CBF 增益 1 | $\alpha_1$ | 1.0 - 3.0 | s⁻¹ | 大→响应快，可能震荡 |
| CBF 增益 2 | $\alpha_2$ | 1.0 - 3.0 | s⁻¹ | 大→收敛快，更保守 |
| 安全距离 | $d_{safe}$ | 0.3 - 0.5m | m | 大→更安全，更难通过 |
| 扰动界 | $d_{max}$ | 2.0 - 3.0 m/s² | m/s² | 大→更保守，小→风险高 |
| 角速度增益 | $k_\omega$ | 0.05 - 0.2 | m/rad² | 大→抑制剧烈机动 |
| 置信度 | $k_c$ | 1.5 - 2.5 | 无量纲 | 大→更保守 |

---

## 五、可行性分析（基于 Crazyflie 实际参数）

### 1. 可行性条件推导

RCBF-QP 可行的充分条件是：存在 $u \in [0, 1]^4$ 使得 $A u \ge b$。

**最大可用控制权限**:
$$\max_{u \in [0,1]^4} A u = \frac{T_{max}}{m} (n^\top R e_3) \cdot 4$$

最坏情况下（推力方向与避障法线反向），$n^\top R e_3 = -1$，此时无法产生远离障碍物的加速度。

**可行条件**（假设 $n^\top R e_3 > 0$，即推力可以指向远离障碍物方向）:

$$\frac{4 T_{max}}{m} (n^\top R e_3) \ge b_{max}$$

其中 $b_{max}$ 是 $b$ 的最大可能值。根据第二节推导：

$$b = \text{Risk} + d_{max} + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)\dot{h} - \alpha_1 \alpha_2 h$$

**最坏情况分析**:
- $\dot{n}^\top v = \frac{\|v_\perp\|^2}{\|p - p_{obs}\|} \ge 0$（离心项，有助于安全）
- $\dot{h} = n^\top v$ 可正可负（负值表示接近障碍物）
- $h \ge 0$（在安全集内）
- $n^\top g e_3 \le g$（重力项最大为 $g$）

当 $h \to 0$ 且 $\dot{h} < 0$（快速接近障碍物）时，$b$ 达到最大值：

$$b_{max} \approx \text{Risk} + d_{max} + g + (\alpha_1 + \alpha_2)|\dot{h}_{max}|$$

代入 $\text{Risk} = k_\omega \|\omega_{xy}\|^2 + k_c \sigma(x) + \epsilon$：

$$b_{max} \approx d_{max} + g + k_\omega \|\omega_{xy}\|^2 + k_c \sigma(x) + \epsilon + (\alpha_1 + \alpha_2)|v_{max}|$$

**可行性条件**:

$$\boxed{\frac{4 T_{max}}{m} (n^\top R e_3)_{min} > g + d_{max} + \max(\text{Risk}) + (\alpha_1 + \alpha_2)|v_{max}|}$$

**物理解读**: 最大推力必须能够克服重力、扰动、姿态波动风险和紧急制动需求。

### 2. Crazyflie 参数代入

根据 RL 环境中的实际参数（`gym_art/quadrotor_multi/quad_models.py` 和 `gym_art/quadrotor_multi/obstacles/`）：

| 参数 | 符号 | 值 | 来源 |
|------|------|-----|------|
| 单电机最大推力 | $T_{max}$ | 0.206 N | `thrust_to_weight=3.0` 计算得出 |
| 无人机质量 | $m$ | 0.028 kg | `QuadLink.m` |
| 重力加速度 | $g$ | 9.81 m/s² | 常数 |
| 最大速度 | $v_{max}$ | 3.0 m/s | `vxyz_max` |
| 最大角速度 | $\omega_{max}$ | 40 rad/s | `omega_max` |
| 无人机碰撞半径 | $R_{drone}$ | 0.046 m | `quad_radius` |
| 障碍物半径 | $R_{obs}$ | 0.15 ~ 0.5 m | `obst_size/2` |

**计算最大加速度**:
$$\frac{4 T_{max}}{m} = \frac{4 \times 0.206}{0.028} = 29.43 \, \text{m/s}^2$$

**完整形式**（考虑 $\text{Risk}$ 项，假设 $(n^\top R e_3)_{min} \approx 1$）:

$$29.43 > 9.81 + d_{max} + \underbrace{k_\omega \|\omega_{xy}\|^2 + k_c \sigma(x) + \epsilon}_{\text{Risk}} + (\alpha_1 + \alpha_2) \times 3.0$$

**典型 Risk 值估计**（假设 $k_\omega = 0.1$，$k_c = 2.0$，$\sigma = 1.0$，$\epsilon = 0.1$）:
- 当 $\|\omega_{xy}\| = 10 \, \text{rad/s}$（中等角速度）时：$\text{Risk} \approx 0.1 \times 100 + 2.0 \times 1.0 + 0.1 = 12.1 \, \text{m/s}^2$
- 当 $\|\omega_{xy}\| = 20 \, \text{rad/s}$（较高角速度）时：$\text{Risk} \approx 0.1 \times 400 + 2.0 \times 1.0 + 0.1 = 42.1 \, \text{m/s}^2$（**已超出最大推力！**）
- 当 $\|\omega_{xy}\| = 15 \, \text{rad/s}$（高角速度）时：$\text{Risk} \approx 0.1 \times 225 + 2.0 \times 1.0 + 0.1 = 24.6 \, \text{m/s}^2$

**不同参数组合下的可行性检查**（考虑 $\text{Risk}$ 项，`thrust_to_weight=3.0`）:

| $d_{max}$ (m/s²) | $\alpha_1 = \alpha_2$ | $\|\omega_{xy}\|$ (rad/s) | Risk (m/s²) | 右侧 (m/s²) | 可行性 | 裕度 |
|-----------------|---------------------|-------------------------|-------------|------------|--------|------|
| 1.0 | 0.5 | 0 | 2.1 | 15.91 | ✓ | 13.52 |
| 1.0 | 0.5 | 5 | 4.6 | 18.41 | ✓ | 11.02 |
| 1.0 | 0.5 | 10 | 12.1 | 25.91 | ✓ | 3.52 |
| 1.0 | 0.5 | 15 | 24.6 | 38.41 | ✗ | -8.98 |
| 1.0 | 1.0 | 0 | 2.1 | 18.91 | ✓ | 10.52 |
| 1.0 | 1.0 | 5 | 4.6 | 21.41 | ✓ | 8.02 |
| 1.0 | 1.0 | 10 | 12.1 | 28.91 | ✓ | 0.52 |
| 1.0 | 1.0 | 15 | 24.6 | 41.41 | ✗ | -11.98 |
| 2.0 | 0.5 | 0 | 2.1 | 16.91 | ✓ | 12.52 |
| 2.0 | 0.5 | 5 | 4.6 | 19.41 | ✓ | 10.02 |
| 2.0 | 0.5 | 10 | 12.1 | 26.91 | ✓ | 2.52 |
| 2.0 | 1.0 | 0 | 2.1 | 19.91 | ✓ | 9.52 |
| 2.0 | 1.0 | 5 | 4.6 | 22.41 | ✓ | 7.02 |
| 2.0 | 1.0 | 10 | 12.1 | 29.91 | ✗ | -0.48 |
| 3.0 | 1.0 | 0 | 2.1 | 20.91 | ✓ | 8.52 |
| 3.0 | 1.0 | 5 | 4.6 | 23.41 | ✓ | 6.02 |
| 3.0 | 1.0 | 10 | 12.1 | 30.91 | ✗ | -1.48 |

**结论**: 
1. 当 `thrust_to_weight=3.0` 时，**可行性条件显著改善**，最大加速度从 18.57 m/s² 提升至 29.43 m/s²
2. 在中等角速度（$\|\omega_{xy}\| \le 10 \, \text{rad/s}$）下，大多数参数组合可行
3. 当 $\|\omega_{xy}\| > 15 \, \text{rad/s}$ 时，几乎所有参数组合都不可行（Risk 项过大）
4. 建议参数选择：
   - $d_{max} \le 2.0 \, \text{m/s}^2$（扰动上界）
   - $\alpha_1 = \alpha_2 \le 1.0 \, \text{s}^{-1}$（CBF 增益）
   - $\|\omega_{xy}\| \le 10 \, \text{rad/s}$（限制角速度）
   - 或者限制最大速度 $v_{max} \le 2.0 \, \text{m/s}$
5. 当 $\alpha_1 = \alpha_2 = 1.0$，$d_{max} = 2.0$，$\|\omega_{xy}\| = 10 \, \text{rad/s}$ 时，系统有约 **0.52 m/s²** 的裕度（勉强可行）

**改进方案**:
1. **使用更强力的无人机**（`thrust_to_weight=3.0` 已显著改善可行性）
2. 降低 CBF 增益 $\alpha_1, \alpha_2$ 至 1.0 以下
3. 限制最大飞行速度 $v_{max}$ 至 2.0 m/s 以下
4. **限制最大角速度 $\|\omega_{xy}\|$ 至 10 rad/s 以下**（关键！）
5. 减小 Risk 增益 $k_\omega$ 至 0.05 以下（可降低 Risk 项影响）
6. 减小扰动估计 $d_{max}$（通过更好的扰动观测器）

### 3. 障碍物参数对可行性的影响

障碍物半径 $R_{obs}$ 和无人机半径 $R_{drone}$ 不直接影响可行性条件，但影响屏障函数 $h$ 的初始值：

$$h_0 = \|p - p_{obs}\| - (R_{obs} + R_{drone} + d_{safe})$$

- 当 $R_{obs} = 0.5 \, \text{m}$（默认），$R_{drone} = 0.046 \, \text{m}$，$d_{safe} = 0.3 \, \text{m}$ 时
- 要求初始距离 $\|p - p_{obs}\| > 0.846 \, \text{m}$ 才能保证 $h_0 > 0$

---

## 六、理论性质

### 1. 安全性定理

**定理**: 如果 RCBF-QP 始终可行，且初始状态满足 $\psi(x_0) \ge 0$，则对于所有 $\|d\| \le d_{max}$，安全集 $\mathcal{C} = \{p : h(p) \ge 0\}$ 是鲁棒前向不变集。

**证明思路**:

1. 由 RCBF 条件 $\dot{\psi} + \alpha_2 \psi \ge \text{Risk} \ge 0$
2. 当 $h \to 0$ 时，$\psi = \dot{h} + \alpha_1 h \to \dot{h}$
3. 若 $\psi \ge 0$，则 $\dot{h} \ge 0$，阻止 $h$ 继续减小
4. 因此 $h(t) \ge 0$ 对所有 $t \ge 0$ 成立 ∎

### 2. 前向不变性

**引理**: 集合 $\mathcal{C}_\psi = \{(p, v) : \psi(p, v) \ge 0\}$ 是前向不变集。

**证明**: 由 $\dot{\psi} \ge -\alpha_2 \psi$，应用比较引理：
$$\psi(t) \ge \psi(0) e^{-\alpha_2 t}$$
若 $\psi(0) \ge 0$，则 $\psi(t) \ge 0$ 对所有 $t \ge 0$ 成立 ∎

---

## 七、论文创新点总结（针对球体模型）

1. **端到端的可微安全约束层**: 在 Sample Factory 框架下，将基于球形简化模型的二阶 RCBF 转化为可微 QP 层，实现了梯度从避障边界向策略网络的直接回传。

2. **欠驱动动力学补偿机制**: 提出了一种基于机身角速度 $\omega$ 的动态 Risk 补偿项，解决了球形碰撞实体在高速机动中由于忽略姿态变化而导致的安全性下降问题。

3. **模块化安全迁移学习**: 借鉴并优化了 Modular Task Learning 架构，在训练中解耦了导航奖励与避障约束，使得基于球形 RCBF 训练的策略具备极强的环境泛化能力。

4. **完整的推力空间公式**: 直接在 4 维电机推力空间求解 QP，输出与 RL 策略完全对齐，无需逆动力学转换。

5. **完整的量纲分析**: 本推导提供了所有公式的量纲检查，确保物理一致性。

---

## 附录：推导检查清单

| 公式 | 量纲 | 备注 |
|------|------|------|
| $h = \|p - p_{obs}\| - (R_{obs} + R_{drone} + d_{safe})$ | m | 位置屏障 |
| $\dot{h} = n^\top v$ | m/s | 屏障一阶导 |
| $\psi = \dot{h} + \alpha_1 h$ | m/s | 级联屏障 |
| $\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge \text{Risk}$ | m/s² | RCBF 条件 |
| $A = \frac{T_{max}}{m} (n^\top R e_3) \mathbf{1}^\top$ | m/s² | 控制矩阵 |
| $b = \text{Risk} + d_{max} + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)\dot{h} - \alpha_1 \alpha_2 h$ | m/s² | 约束向量 |
| $\text{Risk} = k_\omega \|\omega_{xy}\|^2 + k_c \sigma(x) + \epsilon$ | m/s² | **不含** $d_{max}$ |

---

## 关键公式汇总

### RCBF-QP 完整形式

$$
\begin{aligned}
u^* = \arg\min_{u} \quad & \|u - u_{nom}\|^2 \\
\text{s.t.} \quad & A u \ge b \\
& 0 \le u_i \le 1, \quad i=1,2,3,4
\end{aligned}
$$

其中：
$$
\begin{aligned}
A &= \frac{T_{max}}{m} (n^\top R e_3) \begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix} \\
b &= \underbrace{k_\omega \|\omega_{xy}\|^2 + k_c \sigma(x) + \epsilon}_{\text{Risk}} + \underbrace{d_{max}}_{\text{扰动界}} + \underbrace{n^\top g e_3}_{\text{重力}} - \underbrace{\dot{n}^\top v}_{\text{离心}} - \underbrace{(\alpha_1 + \alpha_2)\dot{h}}_{\text{速度}} - \underbrace{\alpha_1 \alpha_2 h}_{\text{位置}}
\end{aligned}
$$
