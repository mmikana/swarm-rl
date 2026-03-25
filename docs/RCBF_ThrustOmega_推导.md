# RCBF 理论推导：基于ThrustOmega控制空间（quad-swarm-rl）

> **版本**: 2026-03-23（ThrustOmega改进版）
> **核心创新**: 将控制空间从RawControl改为ThrustOmega，直接约束线加速度和角速度，自动处理欠驱动系统的姿态耦合
> **假设**: 无人机碰撞模型为球体，使用雅可比反演将(thrust, omega)映射到电机推力
> **动力学模型**: Crazyflie（`gym_art/quadrotor_multi/quad_models.py`）
> **障碍物表示**: 3×3 局部 SDF 网格
> **简化**: 移除 Risk 项，CBF 约束仅依赖位置和速度信息

---

## 符号表

| 符号 | 含义 | 单位 | 实际值 |
|------|------|------|------|
| $p \in \mathbb{R}^3$ | 无人机位置（质心） | m | - |
| $v \in \mathbb{R}^3$ | 无人机速度 | m/s | $\|v\| \le 3.0$ |
| $R \in SO(3)$ | 旋转矩阵 | - | - |
| $\boldsymbol{\omega} \in \mathbb{R}^3$ | 角速度（机体系） | rad/s | $\|\boldsymbol{\omega}\| \le 40$ |
| $a_{thrust} \in [-1, \tau-1]$ | 目标加速度（相对重力） | g | - |
| $\boldsymbol{\omega}_{des} \in \mathbb{R}^3$ | 目标角速度 | rad/s | - |
| $T_{total} = (a_{thrust} + 1) \cdot m \cdot g$ | 总推力 | N | - |
| $\mathbf{T} \in \mathbb{R}^4$ | 四个电机推力 | N | - |
| $J \in \mathbb{R}^{4 \times 4}$ | 雅可比矩阵（T到加速度的映射） | - | - |
| $h$ | 障碍物安全距离 | m | - |
| $n$ | 梯度法向量 | - | - |
| $\alpha_1, \alpha_2$ | CBF 增益 | s⁻¹ | 1.0-3.0 |

---

## 一、RCBF 二阶级联推导

### 1. 定义原始屏障函数 $h$

$$\boxed{h(p) = \text{SDF}_{\text{obs}}[4]}$$

安全集：$\mathcal{C} = \{p : h(p) \ge 0\}$

### 2. 第一层级联：定义 $\psi$

$$\psi = \dot{h} + \alpha_1 h, \quad \alpha_1 > 0$$

其中 $\dot{h} = n^\top v$

### 3. 第二层级联：RCBF 条件

$$\dot{\psi} \ge -\alpha_2 \psi$$

展开整理得：

$$\boxed{\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge 0}$$

---

## 二、在ThrustOmega空间中提取约束

### 1. 展开 $\ddot{h}$

$$\ddot{h} = \frac{d}{dt}(n^\top v) = n^\top \dot{v} + \dot{n}^\top v$$

**关键**：$\dot{v}$ 由线加速度决定，线加速度由**推力方向和大小**决定：

$$\dot{v} = \frac{1}{m}(Re_3) T_{total} - g e_3 + \text{其他项}$$

其中 $T_{total}$ 是总推力，$(Re_3)$ 是推力方向。

### 2. 推力与加速度的关系（通过雅可比）

根据 `quadrotor_control.py` 的雅可比矩阵（第158-169行），电机推力与加速度/角速度的关系为：

$$\begin{bmatrix} a_z \\ \boldsymbol{\alpha} \end{bmatrix} = J \cdot \mathbf{T}$$

其中：
- $a_z$：垂直加速度（由总推力决定）
- $\boldsymbol{\alpha}$：角加速度（由电机差分决定）
- $J$：雅可比矩阵，可逆

**反演**：
$$\mathbf{T} = J^{-1} \begin{bmatrix} a_z \\ \boldsymbol{\alpha} \end{bmatrix}$$

但在ThrustOmega控制空间中，我们不直接控制 $a_z$ 和 $\boldsymbol{\alpha}$，而是控制：
- $a_{thrust}$：目标加速度（相对重力）
- $\boldsymbol{\omega}_{des}$：目标角速度

控制器使用反馈来实现这些目标（见 `OmegaThrustControl.step()`，第191-200行）。

### 3. 关键洞察：线加速度的两个来源

线加速度有两个来源：

**来源1：推力大小**
$$\dot{v}|_{\text{thrust}} = \frac{1}{m}(Re_3) T_{total}$$

**来源2：姿态变化**（当 $\boldsymbol{\omega} \neq 0$ 时）
$$\dot{v}|_{\text{attitude}} = \frac{d(Re_3)}{dt} \frac{T_{total}}{m} = [\boldsymbol{\omega} \times](Re_3) \frac{T_{total}}{m}$$

其中使用了 $\dot{R} = R[\boldsymbol{\omega}]_\times$，所以 $\frac{d(Re_3)}{dt} = \boldsymbol{\omega} \times (Re_3)$

总线加速度：
$$\dot{v} = \frac{T_{total}}{m}\left[(Re_3) + [\boldsymbol{\omega} \times](Re_3)\right] - ge_3$$

简化为：
$$\dot{v} = \frac{T_{total}}{m}(I + [\boldsymbol{\omega}]_\times)Re_3 - ge_3$$

### 4. 代入 $\ddot{h}$

$$\ddot{h} = n^\top \dot{v} + \dot{n}^\top v$$

$$= \frac{T_{total}}{m} n^\top(I + [\boldsymbol{\omega}]_\times)Re_3 - n^\top g e_3 + \dot{n}^\top v$$

使用 $a^\top([b]_\times c) = (a \times b)^\top c$：

$$\ddot{h} = \frac{T_{total}}{m}\left[n^\top Re_3 + (n \times \boldsymbol{\omega})^\top Re_3\right] - n^\top g e_3 + \dot{n}^\top v$$

设：
$$\Gamma = n^\top Re_3 + (n \times \boldsymbol{\omega})^\top Re_3 = (n + n \times \boldsymbol{\omega})^\top Re_3$$

则：
$$\ddot{h} = \frac{T_{total}}{m}\Gamma - n^\top g e_3 + \dot{n}^\top v$$

### 5. 离心项

$$\dot{n}^\top v = \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}$$

这一项在原推导中已经有，保持不变。

---

## 三、约束矩阵 $A$ 与向量 $b$

### 1. 将RCBF条件改写为关于 $(a_{thrust}, \boldsymbol{\omega}_{des})$ 的约束

RCBF条件：
$$\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge 0$$

代入 $\ddot{h}$：

$$\frac{T_{total}}{m}\Gamma - n^\top g e_3 + \dot{n}^\top v + (\alpha_1 + \alpha_2)(n^\top v) + \alpha_1 \alpha_2 h \ge 0$$

其中 $T_{total} = m \cdot g \cdot (a_{thrust} + 1)$，代入：

$$g(a_{thrust} + 1)\Gamma - n^\top g e_3 + \dot{n}^\top v + (\alpha_1 + \alpha_2)(n^\top v) + \alpha_1 \alpha_2 h \ge 0$$

整理得：

$$g \cdot a_{thrust} \cdot \Gamma \ge n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - g\Gamma$$

### 2. 约束形式

$$\boxed{A \cdot \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} \ge b}$$

其中：

**$A$ 矩阵** ($1 \times 4$)：

$$A = \begin{bmatrix}
g \cdot (n^\top Re_3) & g \cdot (Re_3)^\top(n \times \cdot)
\end{bmatrix}$$

第二项是 $1 \times 3$ 的行向量，其第 $j$ 个元素为 $g \cdot (Re_3)^\top(n \times e_j)$

更精确地：
$$A_\omega = g \cdot (n \times \boldsymbol{\omega}_{des})^\top Re_3$$

但这是双线性的（含 $\boldsymbol{\omega}_{des}$），需要处理...

**等等，我意识到一个问题**：$(n \times \boldsymbol{\omega})$ 项中的 $\boldsymbol{\omega}$ 是当前状态，不是控制输入。

让我重新整理。

---

## 四、重新整理：区分状态和控制

### 问题的关键

在表达式 $\Gamma = n^\top Re_3 + (n \times \boldsymbol{\omega})^\top Re_3$ 中：

- $n$：当前状态（从SDF梯度计算）
- $\boldsymbol{\omega}$：当前状态（机体系角速度）
- $R$：当前状态（旋转矩阵）
- $a_{thrust}$：**控制输入**
- $\boldsymbol{\omega}_{des}$：**控制目标**（controller会调节电机推力来追踪）

### 正确的做法

在QP问题中：

**状态项**（已知）：
$$\Gamma_{\text{state}} = n^\top Re_3 + (n \times \boldsymbol{\omega})^\top Re_3$$

这一项与 $a_{thrust}$ 相乘，形成约束左侧。

**控制项**：
- $a_{thrust}$ 直接影响 $\ddot{h}$ 中的 $T_{total}$ 项
- $\boldsymbol{\omega}_{des}$ 通过 controller 的反馈环节（第193-194行）间接影响推力

### 简化策略

在**单步控制**中，我们假设controller快速追踪 $\boldsymbol{\omega}_{des}$，即 $\boldsymbol{\omega} \approx \boldsymbol{\omega}_{des}$。

那么约束变为：

$$g \cdot a_{thrust} \cdot (n^\top Re_3) + g \cdot (n \times \boldsymbol{\omega}_{des})^\top Re_3 \ge b$$

这里 $(n \times \boldsymbol{\omega}_{des})^\top Re_3$ 仍然是**关于 $\boldsymbol{\omega}_{des}$ 的线性项**！

使用向量三重积恒等式 $(a \times b) \cdot c = a \cdot (b \times c)$：

$$(n \times \boldsymbol{\omega}_{des})^\top Re_3 = n^\top(\boldsymbol{\omega}_{des} \times (Re_3))$$

这是关于 $\boldsymbol{\omega}_{des}$ 的**线性形式**：
$$n^\top[\boldsymbol{\omega}_{des} \times (Re_3)] = (n \times (Re_3))^\top \boldsymbol{\omega}_{des}$$

### 最终约束形式

$$\boxed{A \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} \ge b}$$

其中：

$$A = \begin{bmatrix} g \cdot n^\top Re_3 & g \cdot (n \times (Re_3))^\top \end{bmatrix} \in \mathbb{R}^{1 \times 4}$$

$$b = n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - g \cdot n^\top Re_3$$

---

## 五、具体计算

### $A$ 矩阵的计算

**第1个元素**（关于 $a_{thrust}$）：
$$A_1 = g \cdot n^\top Re_3$$

**第2-4个元素**（关于 $\boldsymbol{\omega}_{des} = [\omega_x, \omega_y, \omega_z]^\top$）：
$$A_{2:4} = g \cdot (n \times (Re_3))^\top$$

计算 $n \times (Re_3)$：
$$n \times (Re_3) = \begin{bmatrix} n_y(Re_3)_z - n_z(Re_3)_y \\ n_z(Re_3)_x - n_x(Re_3)_z \\ n_x(Re_3)_y - n_y(Re_3)_x \end{bmatrix}$$

由于 $n = [n_x, n_y, 0]^\top$（从SDF梯度，$n_z = 0$）：
$$n \times (Re_3) = \begin{bmatrix} 0 - 0 \\ 0 - n_x(Re_3)_z \\ n_x(Re_3)_y - n_y(Re_3)_x \end{bmatrix} = \begin{bmatrix} 0 \\ -n_x(Re_3)_z \\ n_x(Re_3)_y - n_y(Re_3)_x \end{bmatrix}$$

所以：
$$A_{2:4} = g \begin{bmatrix} 0 \\ -n_x(Re_3)_z \\ n_x(Re_3)_y - n_y(Re_3)_x \end{bmatrix}^\top$$

### $b$ 向量的计算

$$b = n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - g \cdot n^\top Re_3$$

各项同原推导，保持不变。

---

## 六、物理解释

### 为什么能处理侧边障碍物

**场景**：左侧有障碍物，$n = [-1, 0, 0]^\top$，无人机水平飞行 $Re_3 \approx [0, 0, 1]^\top$

- $A_1 = g \cdot (-1) \cdot 0 = 0$ ❌（线加速度约束无效）
- 但 $n \times (Re_3) = [-1, 0, 0] \times [0, 0, 1] = [0, 1, 0]$
- $A_{2:4} = g[0, 1, 0]$（关于 $\omega_y$ 非零）✓

**约束变为**：
$$g \cdot 0 \cdot a_{thrust} + g \cdot \omega_y \ge b$$

即：
$$\omega_y \ge \frac{b}{g}$$

**物理意义**：约束强制 $\omega_y > 0$（向右roll），这正是避开左侧障碍物的正确反应！

---

## 七、约束的可行性

### 可行条件

由于 $a_{thrust}$ 和 $\boldsymbol{\omega}_{des}$ 有独立的范围：
- $a_{thrust} \in [-1, \tau - 1]$
- $\boldsymbol{\omega}_{des} \in [-\omega_{max}, \omega_{max}]^3$

QP 的可行性由以下条件决定：

$$\max_{a_{thrust}, \boldsymbol{\omega}_{des}} \left(A_1 a_{thrust} + A_{2:4} \cdot \boldsymbol{\omega}_{des}\right) \ge b$$

---

## 八、实现步骤

1. **修改RL策略输出**：从 `u \in [-1,1]^4` 改为 `(a_{thrust}, \boldsymbol{\omega}_{des}) \in \mathbb{R}^4`

2. **CBF约束计算**：
   ```python
   A = g * np.array([
       n @ Re3,
       *(n_cross_Re3)  # 3个元素
   ])
   b = gravity_term - centrifugal - velocity_damping - position_stiffness - bias_term
   ```

3. **QP求解**：在 `(a_{thrust}, \boldsymbol{\omega}_{des})` 空间中求解

4. **动作转换**：
   ```python
   a_safe, omega_safe = qp_solver(...)
   T_safe = J_inv @ [a_safe, omega_safe]  # 转换回电机推力
   ```

---

## 九、关键公式汇总

$$\boxed{
\begin{aligned}
&\text{RCBF约束：} \\
&A \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} \ge b \\
&-1 \le a_{thrust} \le \tau - 1 \\
&-\omega_{max} \le \omega_{des,i} \le \omega_{max}
\end{aligned}
}$$

其中：
$$A = g \begin{bmatrix} n^\top Re_3 & (n \times (Re_3))^\top \end{bmatrix}$$

$$b = g \cdot n^\top e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - g \cdot n^\top Re_3$$

---

## 十、创新点总结

1. **直接控制加速度和角速度**：而不是原始电机推力，物理意义更清晰

2. **自动处理姿态耦合**：$(n \times (Re_3))$ 项自动表达推力方向变化的影响

3. **解决侧边障碍物问题**：当线加速度约束失效时，角速度约束自动生效

4. **保持线性约束形式**：仍然是线性QP问题，易于求解

5. **梯度流路径清晰**：通过雅可比反演，梯度可以流向两个独立的控制维度

 