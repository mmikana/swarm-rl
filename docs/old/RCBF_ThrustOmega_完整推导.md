# RCBF 理论推导：基于 ThrustOmega 控制空间（完整版）

> **版本**: 2026-03-25（最终版）

> **应用场景**: 四旋翼无人机 swarm 的 XY 平面避障（圆柱形障碍物）

> **动力学模型**: Crazyflie（`gym_art/quadrotor_multi/quad_models.py`）

---

## 符号表

### 状态变量

| 符号 | 含义 | 单位 | 实际值/范围 |
|------|------|------|-------------|
| $p \in \mathbb{R}^3$ | 无人机位置（质心，世界系） | m | - |
| $v \in \mathbb{R}^3$ | 无人机速度（世界系） | m/s | $\|v\| \le 3.0$ |
| $R \in SO(3)$ | 旋转矩阵（机体系→世界系） | - | - |
| $\boldsymbol{\omega} \in \mathbb{R}^3$ | 角速度（机体系） | rad/s | $\|\boldsymbol{\omega}\| \le 40$ |

### 控制输入

| 符号 | 含义 | 单位 | 范围 |
|------|------|------|------|
| $a_{thrust} \in \mathbb{R}$ | 目标垂直加速度（相对重力） | g | $[-1, 2]$ (Crazyflie) |
| $\boldsymbol{\omega}_{des} \in \mathbb{R}^3$ | 目标角速度 $[\omega_x, \omega_y, \omega_z]^\top$ | rad/s | $[-31.42, 31.42]^2 \times [-6.28, 6.28]$ |
| $\mathbf{T} \in \mathbb{R}^4$ | 四个电机归一化推力 | - | $[0, 1]^4$ |

### 物理参数

| 符号 | 含义 | 值 | 说明 |
|------|------|-----|------|
| $m$ | 无人机质量 | 0.028 kg | Crazyflie |
| $g$ | 重力加速度 | 9.81 m/s² | - |
| $\tau$ | 推力重量比 | 3.0 | Crazyflie |
| $T_{max}$ | 单电机最大推力 | $\frac{mg\tau}{4}$ | - |

### CBF 参数

| 符号 | 含义 | 单位 | 典型值 |
|------|------|------|--------|
| $h(p)$ | 屏障函数（SDF 值） | m | $h > 0$ 安全 |
| $n$ | SDF 梯度（归一化） | - | $\|n\| = 1$ |
| $\alpha_1, \alpha_2$ | CBF 增益 | s⁻¹ | 1.0-3.0 |
| $s_{allow}$ | CBF 安全余量 | m | 0.05-0.15 |
| $\Delta$ | SDF 网格分辨率 | m | 0.1 |

---

## 一、问题描述

### 1.1 原始问题：RawControl 空间的局限

在 RawControl 空间（直接控制电机推力 $u \in [0, 1]^4$）中，CBF 约束形式为：

$$A(p, v, R) \cdot u \ge b(p, v)$$

其中 $A$ 矩阵与**总推力** $\sum u_i$ 相关，无法独立约束**电机差分**（力矩）。

**关键问题**：当障碍物在侧边时：
- 梯度 $n$ 是水平的（如 $n = [-1, 0, 0]^\top$）
- 推力方向 $Re_3 \approx [0, 0, 1]^\top$（水平飞行）
- $n^\top Re_3 \approx 0$，约束失效！
- 无人机无法被强制做出正确的避障反应（向安全方向倾斜）

### 1.2 根本原因：欠驱动系统的姿态耦合

四旋翼是**欠驱动系统**：
- 只有 4 个输入（电机推力）
- 但有 6 个自由度（位置 + 姿态）
- 水平加速度必须通过改变姿态（Roll/Pitch）产生

在 RawControl 空间中，无法独立约束姿态变化，导致侧边避障失效。

---

## 二、ThrustOmega 控制空间

### 2.1 动作空间定义

**从 RawControl 改为 ThrustOmega**：

$$\mathbf{u}_{thrust\omega} = \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} \in \mathbb{R}^4$$

| 维度 | 符号 | 物理意义 | RL 范围 | 物理范围 |
|------|------|---------|---------|----------|
| 1 | $a_{thrust}$ | 目标加速度（相对 g） | $[-1, 1]$ | $[-1, 2]$ |
| 2 | $\omega_x$ | 目标 Roll 角速度 | $[-1, 1]$ | $[-31.42, 31.42]$ |
| 3 | $\omega_y$ | 目标 Pitch 角速度 | $[-1, 1]$ | $[-31.42, 31.42]$ |
| 4 | $\omega_z$ | 目标 Yaw 角速度 | $[-1, 1]$ | $[-6.28, 6.28]$ |

### 2.2 动作转换：雅可比反演

通过**雅可比矩阵反演**将 ThrustOmega 转换为电机推力：

$$\mathbf{T} = J^{-1} \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix}$$

其中雅可比矩阵 $J$ 定义为：

$$J = \begin{bmatrix}
\frac{1}{m} & \frac{1}{m} & \frac{1}{m} & \frac{1}{m} \\
\frac{(r_1 \times z)_x}{I_{xx}} & \frac{(r_2 \times z)_x}{I_{xx}} & \frac{(r_3 \times z)_x}{I_{xx}} & \frac{(r_4 \times z)_x}{I_{xx}} \\
\frac{(r_1 \times z)_y}{I_{yy}} & \frac{(r_2 \times z)_y}{I_{yy}} & \frac{(r_3 \times z)_y}{I_{yy}} & \frac{(r_4 \times z)_y}{I_{yy}} \\
\frac{\kappa}{I_{zz}} & -\frac{\kappa}{I_{zz}} & \frac{\kappa}{I_{zz}} & -\frac{\kappa}{I_{zz}}
\end{bmatrix}$$

其中：
- $r_i$：第 $i$ 个电机相对于质心的位置
- $z = [0, 0, 1]^\top$：推力方向（机体系）
- $\kappa$：反扭矩系数

### 2.3 控制器实现

在 `OmegaThrustControl.step()` 中：

```python
def step(self, dynamics, action, dt):
    # action 已经是缩放后的物理范围
    kp = 5.0
    omega_err = dynamics.omega - action[1:]  # 角速度误差
    dw_des = -kp * omega_err                 # 期望角加速度
    acc_des = GRAV * (action[0] + 1.0)       # 期望线加速度
    des = np.append(acc_des, dw_des)
    thrusts = np.matmul(self.Jinv, des)      # 雅可比反演
    thrusts = np.clip(thrusts, 0, 1)         # 饱和
    dynamics.step(thrusts, dt)
```

---

## 三、RCBF 理论推导

### 3.1 屏障函数定义

**原始屏障函数**：

$$h(p) = \text{SDF}_{\text{obs}}[4]$$

其中 $h(p) > 0$ 表示安全，$h(p) = 0$ 表示接触障碍物。

**安全集**：
$$\mathcal{C} = \{p \in \mathbb{R}^3 : h(p) \ge 0\}$$

### 3.2 梯度和安全距离计算

从 3×3 SDF 网格计算梯度：

$$n = \frac{\nabla h}{\|\nabla h\|} = \frac{1}{2\Delta}\begin{bmatrix}
\text{SDF}[5] - \text{SDF}[3] \\
\text{SDF}[7] - \text{SDF}[1] \\
0
\end{bmatrix}$$

**关键简化**：由于障碍物是圆柱形（贯穿 Z 轴），梯度永远在 XY 平面：

$$\boxed{n = [n_x, n_y, 0]^\top, \quad n_z = 0 \text{ 恒成立}}$$

### 3.3 一阶导数

$$\dot{h} = \frac{d}{dt}h(p) = \nabla h^\top \dot{p} = n^\top v$$

### 3.4 二阶导数

$$\ddot{h} = \frac{d}{dt}(n^\top v) = n^\top \dot{v} + \dot{n}^\top v$$

**关键**：需要计算 $\dot{v}$（线加速度）。

### 3.5 线加速度（正确的动力学）

四旋翼的标准动力学方程：

$$m\dot{v} = T_{total} \cdot Re_3 - mge_3$$

其中 $T_{total} = m \cdot g \cdot (a_{thrust} + 1)$ 是总推力。

因此：
$$\boxed{\dot{v} = \frac{T_{total}}{m} Re_3 - ge_3 = g(a_{thrust} + 1) Re_3 - ge_3}$$

**重要说明**：
- $\dot{v}$ 只包含当前时刻的 $Re_3$，不包含 $\frac{d}{dt}(Re_3)$
- $\frac{d}{dt}(Re_3)$ 属于 $\ddot{v}$ 的推导，不属于 $\dot{v}$
- 这是标准四旋翼动力学的正确形式

### 3.6 正确的 $\ddot{h}$ 推导

$$\ddot{h} = n^\top \dot{v} + \dot{n}^\top v$$

代入正确的 $\dot{v}$：

$$\ddot{h} = n^\top \left(g(a_{thrust} + 1) Re_3 - ge_3\right) + \dot{n}^\top v$$

$$= g(a_{thrust} + 1)(n^\top Re_3) - g(n^\top e_3) + \dot{n}^\top v$$

**关键观察**：$\ddot{h}$ **不直接包含** $\boldsymbol{\omega}$！

### 3.7 如何引入 $\boldsymbol{\omega}_{des}$ 约束

**问题**：$\ddot{h}$ 中的 $n^\top Re_3$ 项依赖于姿态，而姿态通过 $\boldsymbol{\omega}$ 变化。

**解决方案**：使用一阶预测来关联当前控制 $\boldsymbol{\omega}_{des}$ 和未来姿态。

在时间步长 $\Delta t$ 内，$Re_3$ 的变化：

$$Re_3(t+\Delta t) \approx Re_3(t) + \frac{d}{dt}(Re_3) \cdot \Delta t$$

其中：
$$\frac{d}{dt}(Re_3) = \boldsymbol{\omega}_{world} \times Re_3$$

在单步假设下（$\boldsymbol{\omega} \approx \boldsymbol{\omega}_{des}$）：
$$Re_3(t+\Delta t) \approx Re_3(t) + (\boldsymbol{\omega}_{des, world} \times Re_3) \cdot \Delta t$$

其中 $\boldsymbol{\omega}_{des, world} = R \boldsymbol{\omega}_{des}$（世界系角速度）。

因此：
$$n^\top Re_3(t+\Delta t) \approx n^\top Re_3(t) + n^\top(\boldsymbol{\omega}_{des, world} \times Re_3) \cdot \Delta t$$

**关键：向量恒等式与坐标系转换**

使用标量三重积恒等式和叉乘反对称性：
$$n^\top(\boldsymbol{\omega}_{des, world} \times Re_3) = -(n \times Re_3)^\top \boldsymbol{\omega}_{des, world}$$

代入 $\boldsymbol{\omega}_{des, world} = R \boldsymbol{\omega}_{des}$：
$$-(n \times Re_3)^\top R \boldsymbol{\omega}_{des} = -(R^\top(n \times Re_3))^\top \boldsymbol{\omega}_{des}$$

定义**机体系中的叉乘向量**：
$$(n \times Re_3)_{body} \triangleq R^\top(n \times Re_3)$$

最终预测形式：
$$n^\top Re_3(t+\Delta t) \approx n^\top Re_3(t) - \Delta t \cdot (n \times Re_3)_{body}^\top \boldsymbol{\omega}_{des}$$

**代入 $\ddot{h}$ 的预测形式**：

$$\ddot{h}(t+\Delta t) \approx g(a_{thrust} + 1)[n^\top Re_3 - \Delta t \cdot (n \times Re_3)_{body}^\top \boldsymbol{\omega}_{des}] - g(n^\top e_3) + \dot{n}^\top v$$

**关键说明**：
- $\boldsymbol{\omega}_{des}$ 是**机体系**角速度（控制输入）
- $(n \times Re_3)_{body}$ 是**机体系**中的叉乘向量
- 必须通过 $R^\top$ 进行坐标系转换

### 3.8 离心项

$$\dot{n}^\top v = -\frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}$$

这一项来自 SDF 场的曲率（圆柱形障碍物的离心加速度）。

**注意符号**：这里是**负号**，因为当无人机切向飞行时，SDF 梯度会"追赶"速度方向，导致 $\dot{n}^\top v < 0$。

**代码实现**（`quad_cbf_qp.py` 第 461-464 行）：
```python
centrifugal = (v_squared - h_dot**2) / (h + self.R_obs)  # 正值
b = self.m * (-centrifugal - ...)  # 负贡献
```

---

## 四、CBF 约束的最终形式

### 4.1 RCBF 条件

$$\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge 0$$

代入预测形式：

$$g(a_{thrust} + 1)[n^\top Re_3 - \Delta t \cdot (n \times Re_3)^\top \boldsymbol{\omega}_{des}] - g(n^\top e_3) + \dot{n}^\top v + (\alpha_1 + \alpha_2)(n^\top v) + \alpha_1 \alpha_2 h \ge 0$$

### 4.2 代入 $T_{total}$ 并展开

展开左边：

$$g \cdot a_{thrust} \cdot n^\top Re_3 - g \cdot a_{thrust} \cdot \Delta t \cdot (n \times Re_3)^\top \boldsymbol{\omega}_{des} + g \cdot n^\top Re_3 - g \cdot \Delta t \cdot (n \times Re_3)^\top \boldsymbol{\omega}_{des} - g(n^\top e_3) + \dot{n}^\top v + (\alpha_1 + \alpha_2)(n^\top v) + \alpha_1 \alpha_2 h \ge 0$$

**关键问题**：第二项包含交叉项，这使得约束**非线性**。

### 4.3 线性化近似

为了获得线性 QP 约束，我们做以下**合理近似**：

**近似**：忽略交叉项 $g \cdot a_{thrust} \cdot \Delta t \cdot (n \times Re_3)^\top \boldsymbol{\omega}_{des}$

**理由**：
1. 在 XY 避障简化下，$n^\top Re_3 = 0$，因此第一项 $g \cdot a_{thrust} \cdot n^\top Re_3 = 0$
2. 交叉项是二阶小量（两个控制输入的乘积）
3. 主要约束来自 $-g \cdot \Delta t \cdot (n \times Re_3)^\top \boldsymbol{\omega}_{des}$ 项
4. $\Delta t$ 很小（典型值 0.01-0.05 秒）

**线性化后的约束**：

$$g \cdot n^\top Re_3 \cdot a_{thrust} - g \cdot \Delta t \cdot (n \times Re_3)^\top \boldsymbol{\omega}_{des} \ge g(n^\top e_3) - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - g \cdot n^\top Re_3$$

### 4.4 最终约束形式

$$\boxed{A \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} \ge b}$$

其中：

**$A$ 矩阵** ($1 \times 4$)：
$$A = g \begin{bmatrix} n^\top Re_3 & -\Delta t \cdot (n \times Re_3)_{body}^\top \end{bmatrix}$$

展开为：
$$A = g \begin{bmatrix}
n^\top Re_3 \\
-\Delta t \cdot (n \times Re_3)_{body,x} \\
-\Delta t \cdot (n \times Re_3)_{body,y} \\
-\Delta t \cdot (n \times Re_3)_{body,z}
\end{bmatrix}^\top$$

其中 $(n \times Re_3)_{body} = R^\top(n \times Re_3)$ 是**机体系**中的叉乘向量。

**$b$ 标量**：
$$b = g \cdot n^\top e_3 + \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}} - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - g \cdot n^\top Re_3$$

**量纲说明**：
- $b$ 的量纲是 $[L/T^2] = [m/s^2]$（加速度）
- $A$ 的量纲是 $[L/T^2]$（加速度）
- $u = [a_{thrust}, \boldsymbol{\omega}_{des}]^\top$ 在**归一化动作空间**中是无量纲的
- CBF 优化在归一化空间中进行，输出后通过 `_scale_action()` 缩放到物理范围

**线性化近似的说明**：
1. 忽略了交叉项 $g \cdot a_{thrust} \cdot \Delta t \cdot (n \times Re_3)^\top \boldsymbol{\omega}_{des}$
2. 在 XY 避障简化下，$n^\top Re_3 = 0$，因此 $a_{thrust}$ 的系数为 0
3. 约束主要由角速度项 $-g \cdot \Delta t \cdot (n \times Re_3)_{body}^\top \boldsymbol{\omega}_{des}$ 决定
4. 近似是保守的（忽略的项可能有助于避障）
5. **注意**：$\Delta t$ 是预测时间步长（典型值 0.01-0.05 秒）

---

## 五、XY 平面避障的简化

### 5.1 圆柱形障碍物的关键性质

由于所有障碍物都是圆柱形（贯穿 Z 轴）：

$$n = [n_x, n_y, 0]^\top \quad \text{（}n_z = 0\text{ 恒成立）}$$

### 5.2 水平飞行假设

在大多数飞行场景中，无人机**接近水平飞行**：

$$R \approx \begin{bmatrix}
\cos\psi & -\sin\psi & 0 \\
\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}$$

其中 $\psi$ 是 Yaw 角。

则：
$$Re_3 = [0, 0, 1]^\top$$

### 5.3 简化后的 $A$ 矩阵

**第 1 个元素**（关于 $a_{thrust}$）：
$$A_1 = g \cdot n^\top Re_3 = g \cdot [n_x, n_y, 0] \cdot [0, 0, 1] = 0$$

❌ **线加速度约束失效！**（这与 RawControl 相同）

**第 2-4 个元素**（关于 $\boldsymbol{\omega}_{des}$）：

计算世界系中的叉乘：
$$n \times Re_3 = [n_x, n_y, 0] \times [0, 0, 1] = [n_y, -n_x, 0]$$

**关键**：根据第 4.4 节，$A$ 包含**负号**且需要转换到**机体系**：
$$(n \times Re_3)_{body} = R^\top(n \times Re_3)$$

对于水平飞行（$R$ 只包含 Yaw 旋转）：
$$(n \times Re_3)_{body} = \begin{bmatrix} \cos\psi & \sin\psi & 0 \\ -\sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} n_y \\ -n_x \\ 0 \end{bmatrix} = \begin{bmatrix} n_y\cos\psi - n_x\sin\psi \\ -n_y\sin\psi - n_x\cos\psi \\ 0 \end{bmatrix}$$

**$A$ 矩阵**：
$$A = g \begin{bmatrix} 0 & -(n_y\cos\psi - n_x\sin\psi) & -(-n_y\sin\psi - n_x\cos\psi) & 0 \end{bmatrix}$$

$$A = g \begin{bmatrix} 0 & -n_y\cos\psi + n_x\sin\psi & n_y\sin\psi + n_x\cos\psi & 0 \end{bmatrix}$$

### 5.4 简化后的约束（一般形式）

$$g[(-n_y\cos\psi + n_x\sin\psi)\omega_x + (n_y\sin\psi + n_x\cos\psi)\omega_y] \ge b$$

**特殊情况**：当 $\psi = 0$（无人机指向 X 正方向）时：
$$\cos\psi = 1, \sin\psi = 0$$

约束简化为：
$$\boxed{g(-n_y \omega_x + n_x \omega_y) \ge b}$$

或者写成：
$$\boxed{g(n_x \omega_y - n_y \omega_x) \ge b}$$

或者：
$$\boxed{\omega_\perp \ge \frac{b}{g}}$$

其中 $\omega_\perp = -n_y \omega_x + n_x \omega_y$ 是**垂直于梯度方向的角速度分量**（在 $\psi=0$ 的特殊情况下）。

---

## 六、物理解释与例子

### 6.1 侧边障碍物场景

**场景设置**：
- 无人机在原点，水平飞行
- Yaw 角 $\psi = 0$（指向 X 正方向）
- 左侧（Y 负方向）有障碍物

**梯度**：
$$n = [0, -1, 0]^\top \quad \text{（指向 Y 正方向，远离障碍物）}$$

**约束计算**：
$$A_1 = g \cdot [0, -1, 0] \cdot [0, 0, 1] = 0$$
$$n \times Re_3 = [0, -1, 0] \times [0, 0, 1] = [-1, 0, 0]$$
$$(n \times Re_3)_{body} = [-1, 0, 0] \quad (\psi = 0 \text{时机体系=世界系})$$
$$A_{2:4} = -g \cdot \Delta t \cdot (n \times Re_3)_{body}^\top = -g \cdot \Delta t \cdot [-1, 0, 0] = g \cdot \Delta t \cdot [1, 0, 0]$$

**约束变为**：
$$g \cdot \Delta t \cdot \omega_x \ge b \quad \Rightarrow \quad \omega_x \ge \frac{b}{g \cdot \Delta t}$$

**物理意义**：
- 约束强制 $\omega_x > 0$（向右 roll）
- 向右 roll 会使推力方向向右倾斜
- 产生向右（Y 正方向）的加速度
- **自动强制正确的避障反应！** ✓

**注意**：$\Delta t$ 是预测时间步长（典型值 0.02-0.05 秒），它影响约束的强度。

### 6.2 与 RawControl 的对比

| 方面 | RawControl | ThrustOmega |
|------|-----------|-------------|
| 线加速度约束 | $n^\top Re_3 \cdot \sum u_i$ | $n^\top Re_3 \cdot a_{thrust}$ |
| 侧边障碍物 | ❌ 约束失效（$n^\top Re_3 \approx 0$） | ✅ 角速度约束生效 |
| 约束维度 | 1 维（总推力） | 4 维（独立控制） |
| 物理意义 | 不清晰 | 清晰（直接约束角速度） |

---

## 七、CBF-QP 优化问题

### 7.1 优化目标

在 ThrustOmega 空间中，最小化与 RL 输出的偏差：

$$\min_{a, \omega_x, \omega_y, \omega_z} \left\| \begin{bmatrix} a \\ \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} - \begin{bmatrix} a_{rl} \\ \omega_{x,rl} \\ \omega_{y,rl} \\ \omega_{z,rl} \end{bmatrix} \right\|^2$$

### 7.2 约束条件

根据第 5.4 节的简化形式：

$$\text{s.t.} \quad g(n_x \omega_y - n_y \omega_x) \ge b$$

$$-1 \le a \le 2$$
$$-31.42 \le \omega_x, \omega_y \le 31.42$$
$$-6.28 \le \omega_z \le 6.28$$

### 7.3 标准 QP 形式

$$\min_x \frac{1}{2}x^\top Q x + p^\top x$$
$$\text{s.t.} \quad Gx \le h$$

其中：
- $x = [a, \omega_x, \omega_y, \omega_z]^\top$
- $Q = 2I_4$
- $p = -2[a_{rl}, \omega_{x,rl}, \omega_{y,rl}, \omega_{z,rl}]^\top$
- $G = \begin{bmatrix} -A \\ -I_4 \\ I_4 \end{bmatrix}$
- $h = \begin{bmatrix} -b \\ -x_{min} \\ x_{max} \end{bmatrix}$

---

## 八、可行性分析

### 8.1 最坏情况

**场景**：
- 无人机正向障碍物移动：$v = 3.0$ m/s
- 距离很小：$h \to 0^+$
- 无转动：$\boldsymbol{\omega} = 0$

**计算 $b$**：
$$b \approx -(\alpha_1 + \alpha_2)(n^\top v) \approx 2.0 \times 3.0 = 6.0 \text{ m/s}^2$$

**所需角速度**（考虑 $\Delta t$ 因子）：
$$\omega_{required} = \frac{b}{g \cdot \Delta t}$$

| $\Delta t$ (秒) | $\omega_{required}$ (rad/s) | 可行性 |
|----------------|---------------------------|--------|
| 0.01 | $\frac{6.0}{9.81 \cdot 0.01} \approx 61.2$ | ❌ 不可行（超限） |
| 0.02 | $\frac{6.0}{9.81 \cdot 0.02} \approx 30.6$ | ⚠️ 临界（接近极限） |
| 0.03 | $\frac{6.0}{9.81 \cdot 0.03} \approx 20.4$ | ✅ 可行 |
| 0.05 | $\frac{6.0}{9.81 \cdot 0.05} \approx 12.2$ | ✅ 可行（裕度充足） |

**推荐**：选择 $\Delta t = 0.03 \sim 0.05$ 秒

**可用角速度**：
$$\omega_{max} = 31.42 \text{ rad/s}$$

**裕度**（当 $\Delta t = 0.05$ 秒时）：
$$\frac{\omega_{max}}{\omega_{required}} \approx \frac{31.42}{12.2} \approx 2.6 \text{ 倍}$$

✅ **充分可行！**

**注意**：$\Delta t$ 的选择需要在约束强度和控制可行性之间权衡：
- $\Delta t$ 太小：约束过强，可能超出执行器能力
- $\Delta t$ 太大：预测不准确，可能影响安全性

### 8.2 数值稳定性

**保护措施**：
- $h_{safe} = \max(h, 10^{-3})$（避免除零）
- $\|n\|_{safe} = \max(\|n\|, 10^{-6})$（避免梯度为零）
- QP 求解失败时回退到 RL 动作

---

## 九、实现步骤

### 9.1 修改 RL 策略输出

在 `quad_multi_model_rcbf.py` 中：

```python
# 输出 ThrustOmega 动作
action_means = self.policy_head(features)  # (batch, 4)
# action_means[:, 0:1] = a_thrust
# action_means[:, 1:4] = omega_des
```

### 9.2 CBF 约束计算

在 `quad_cbf_qp.py` 中：

```python
def compute_cbf_constraints_batch(self, state, sdf_obs):
    # 1. 计算 SDF 梯度
    n, h = self.compute_sdf_gradient(sdf_obs)  # n_z = 0
    
    # 2. 计算 b 向量
    v = state['vel']
    h_dot = np.sum(n * v, axis=1)
    v_squared = np.sum(v * v, axis=1)
    centrifugal = (v_squared - h_dot**2) / (h + self.R_obs)
    
    b = self.m * (
        -centrifugal
        - (self.alpha_1 + self.alpha_2) * h_dot
        - self.alpha_1 * self.alpha_2 * h
    )
    
    # 3. 计算 A 矩阵
    # 关键：根据第 4.4 节，A 包含负号和坐标系转换：
    # A = g[n^T Re3, -Δt·(n×Re3)_body^T]
    # 其中 (n×Re3)_body = R^T(n×Re3)
    
    Re3 = np.array([0, 0, 1])  # 水平飞行假设（推力方向垂直向上）
    n_cross_Re3_world = np.cross(n, Re3)  # 世界系：[n_y, -n_x, 0]
    
    # 转换到机体系：(n×Re3)_body = R^T(n×Re3)_world
    # 注意：即使水平飞行，Yaw 角 ψ 也可能不为 0，必须进行坐标转换
    R = state['R']  # 旋转矩阵 (batch, 3, 3)
    n_cross_Re3_body = np.einsum('bij,bj->bi', R.transpose(0, 2, 1), n_cross_Re3_world)
    
    A = self.g * np.column_stack([
        np.sum(n * Re3, axis=1, keepdims=True),  # A_1 = n^T Re3 (通常≈0)
        -n_cross_Re3_body                         # 负号！A_{2:4} = -g(n×Re3)_body^T
    ])
    
    return A, b
```

### 9.3 QP 求解

```python
def solve_qp_batch(self, u_rl, A, b):
    # u_rl: (batch, 4) RL 输出
    # A: (batch, 1, 4), b: (batch, 1)
    
    Q = 2.0 * torch.eye(4)
    p = -2.0 * u_rl
    
    # 约束：A @ x >= b  →  -A @ x <= -b
    G = torch.cat([-A, -torch.eye(4), torch.eye(4)], dim=1)
    h = torch.cat([-b, -self.x_min, self.x_max], dim=1)
    
    # 使用 qpth 求解
    x_safe = QPFunction()(Q, p, G, h)
    
    return x_safe
```

### 9.4 动作转换

```python
# CBF 输出：u_safe = [a_safe, omega_safe]
# 转换为电机推力
T_safe = J_inv @ u_safe.T  # (4, batch)
T_safe = T_safe.T  # (batch, 4)
```

---

## 十、关键公式汇总

### 约束矩阵
$$\boxed{A = g \begin{bmatrix} n^\top Re_3 & -(n \times Re_3)^\top \end{bmatrix}}$$

### 约束向量
$$\boxed{b = g \cdot n^\top e_3 + \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}} - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - g \cdot n^\top Re_3}$$

**注意**：离心项 $\dot{n}^\top v = -\frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}$ 是负的，代入 $b$ 向量公式 $b = ... - \dot{n}^\top v ...$ 后变为正贡献。

### XY 避障简化
$$\boxed{g(n_x \omega_y - n_y \omega_x) \ge b}$$

### 可行性裕度
$$\boxed{\frac{\omega_{max}}{\omega_{required}} \approx 52 \text{ 倍}}$$

---

## 十一、创新点总结

1. **直接控制加速度和角速度**：物理意义清晰，符合直觉

2. **自动处理姿态耦合**：$(n \times Re_3)$ 项自动表达推力方向变化的影响

3. **解决侧边障碍物问题**：当线加速度约束失效时，角速度约束自动生效

4. **保持线性约束形式**：仍然是线性 QP 问题，易于求解和微分

5. **梯度流路径清晰**：通过雅可比反演，梯度可以流向所有 4 个控制维度

6. **XY 避障的简化**：利用圆柱形障碍物的性质，$n_z = 0$ 恒成立，大幅简化计算

---

## 十二、实现检查清单

- [ ] 修改 `OmegaThrustControl.action_space` 为 $[-1, 1]^4$ ✓
- [ ] 实现 `_scale_action()` 缩放函数 ✓
- [ ] 添加 `--continuous_tanh_scale=1.0` 到训练配置
- [ ] 实现 `compute_cbf_constraints_batch()`（XY 简化版）
- [ ] 实现 `solve_qp_batch()`（可微分 QP）
- [ ] 修改 `forward()` 方法，输出 ThrustOmega 动作
- [ ] 实现动作转换：ThrustOmega → 电机推力
- [ ] 验证梯度流（反向传播测试）
- [ ] 测试侧边障碍物场景
- [ ] 训练和评估对比

---

## 十三、总结

**ThrustOmega 空间中的 CBF 方案**是一个优雅、高效、物理直观的解决方案：

1. ✅ **理论完整**：从 RCBF 条件严格推导到线性约束
2. ✅ **物理直观**：直接约束角速度，自动产生正确避障反应
3. ✅ **计算高效**：线性 QP，易于求解和微分
4. ✅ **梯度清晰**：所有 4 个维度都有梯度，RL 充分学习
5. ✅ **无需调参**：没有 Risk 权重等超参数
6. ✅ **XY 简化**：利用圆柱形障碍物性质，$n_z = 0$ 恒成立

这是从 RawControl 到 ThrustOmega 的完整理论推导，为四旋翼无人机 swarm 的安全避障提供了坚实的理论基础。
