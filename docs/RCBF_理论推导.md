# RCBF 理论推导：基于 SDF 的障碍物约束（quad-swarm-rl）

> **版本**: 2026-03-09（修订版）
> **核心创新**: 使用 Signed Distance Field (SDF) 替代显式障碍物列表，实现固定约束数量
> **假设**: 无人机碰撞模型为球体（半径 $R_{drone}$ 为常数），姿态不影响碰撞边界
> **控制输入**: $u \in [-1, 1]^4$（与 RL 策略输出对齐）
> **动力学模型**: Crazyflie（`gym_art/quadrotor_multi/quad_models.py`）
> **障碍物表示**: 3×3 局部 SDF 网格（`gym_art/quadrotor_multi/obstacles/utils.py`）

---

## 符号表

| 符号 | 含义 | 单位 | RL 环境实际值 |
|------|------|------|------|
| $p \in \mathbb{R}^3$ | 无人机位置（质心） | m | - |
| $v \in \mathbb{R}^3$ | 无人机速度 | m/s | $\|v\| \le 3.0$ |
| $R \in SO(3)$ | 旋转矩阵（机体系→惯性系） | - | - |
| $u_{rl} \in [-1,1]^4$ | RL 策略输出（标称控制） | - | `raw_control_zero_middle=True` |
| $u \in [-1,1]^4$ | RCBF 修正后的安全控制 | - | 与 $u_{rl}$ 同空间 |
| $u_{thrust} \in [0,1]^4$ | 实际电机推力（归一化） | - | $u_{thrust} = 0.5 \times (u + 1)$ |
| $T_{max}$ | 单电机最大推力 | N | 0.130 N |
| $m$ | 无人机质量 | kg | 0.028 kg |
| $g$ | 重力加速度 | m/s² | 9.81 |
| $e_3$ | 惯性系 z 轴单位向量 $[0,0,1]^\top$ | - | - |
| $\text{SDF}_{\text{obs}} \in \mathbb{R}^9$ | 3×3 局部 SDF 网格 | m | 从环境观测获取 |
| $\Delta$ | SDF 网格分辨率 | m | 0.1 (`resolution`) |
| $R_{obs}$ | 障碍物半径 | m | 0.15 ~ 0.5 (`obst_size/2`) |
| $R_{drone}$ | 无人机球体半径 | m | 0.046 (碰撞模型参数) |
| $n \in \mathbb{R}^3$ | 避障方向法向量（从 SDF 梯度估计） | - | $\|n\| \approx 1$ |
| $\alpha_1, \alpha_2$ | CBF 增益参数 | s⁻¹ | 1.0-3.0 |
| $\omega$ | 机身角速度 | rad/s | $\|\omega\| \le 40$ |

**坐标系约定**:
- **惯性系** $\{I\}$: $e_3 = [0, 0, 1]^\top$ 指向上方
- **机体系** $\{B\}$: $z_b$ 轴垂直于机身平面向下
- **推力方向**: $R e_3$ 表示机体系 $z_b$ 轴在惯性系中的方向（即推力矢量方向，指向上方为正）

**RL 环境参数说明**（基于 `gym_art/quadrotor_multi/quad_models.py` 和 `gym_art/quadrotor_multi/obstacles/`）:
- 质量 $m = 0.028 \, \text{kg}$（body + payload + arms + motors + propellers）
- 推力重量比 `thrust_to_weight` = 3.0
- 单电机最大推力：$T_{max} = \frac{m \cdot g \cdot \text{thrust\_to\_weight}}{4} = \frac{0.028 \cdot 9.81 \cdot 3.0}{4} \approx 0.206 \, \text{N}$
- 电机臂长：$\text{arm} = \|\text{motor}_{xy}\| = 0.046 \, \text{m}$
- **无人机碰撞半径**：$R_{drone} = 0.046 \, \text{m}$（`obstacles.py` 中的 `quad_radius`）
- 房间尺寸：默认 $10 \times 10 \times 10$ m（`quads_room_dims`）
- 最大速度：$v_{max} = 3.0 \, \text{m/s}$（`vxyz_max`）
- 最大角速度：$\omega_{max} = 40 \, \text{rad/s}$（`omega_max`）
- **障碍物半径**：$R_{obs} = \frac{\text{obst\_size}}{2}$，范围 0.15 ~ 0.3 m（`obstacles.py` 中的 `obstacle_radius = obstacle_size / 2.0`）
- 障碍物尺寸配置：`quads_obst_size` 默认 1.0 m（直径），随机化时 0.3 ~ 0.6 m（直径）
- 碰撞半径：$2.0 \times \text{arm} \approx 0.092 \, \text{m}$（`quads_collision_hitbox_radius`）
- **SDF 网格**：3×3 局部采样，分辨率 $\Delta = 0.1$ m（`obstacles/utils.py:get_surround_sdfs`）
- **SDF 计算**：$\text{SDF}[i] = \min_j \|p_{\text{grid}}[i] - p_{\text{obst}}^{(j)}\| - R_{\text{obs}}$（自动跟踪最近障碍物）

**动作空间说明**（`gym_art/quadrotor_multi/quadrotor_control.py:RawControl`）:
- **RL 输出**：$u_{rl} \in [-1, 1]^4$（`raw_control_zero_middle=True`）
- **RCBF 输出**：$u \in [-1, 1]^4$（与 RL 对齐，最小干预）
- **实际推力**：$u_{thrust} = 0.5 \times (u + 1) \in [0, 1]^4$
- **物理意义**：$u=-1 \Rightarrow u_{thrust}=0$（最低转速），$u=1 \Rightarrow u_{thrust}=1$（最高转速）
- **最小干预原则**：如果 RL 策略安全，则 $u = u_{rl}$；否则 RCBF 最小修正

---

## 一、RCBF 二阶级联推导

由于控制输入 $u$（电机推力）在位置 $h$ 的二阶导数中才显式出现（相对度为 2），我们需要构造级联的屏障函数。

### 1. 定义原始屏障函数 $h$（基于 SDF）

**关键创新：使用 SDF 替代显式障碍物**

在 quad-swarm-rl 中，环境提供 3×3 局部 SDF 网格：
$$\text{SDF}_{\text{obs}} \in \mathbb{R}^9, \quad \text{SDF}_{\text{obs}}[i] = \min_{j} \|p_{\text{grid}}[i] - p_{\text{obst}}^{(j)}\| - R_{\text{obs}}$$

其中网格点布局（相对无人机位置 $p = [x, y, z]$，分辨率 $\Delta = 0.1$ m）：
```
索引排列（XY 平面）:
  [0]  [1]  [2]     对应位置:
  [3]  [4]  [5]  =  (x±Δ, y±Δ, z)
  [6]  [7]  [8]
```

**障碍物屏障函数**:
$$\boxed{h(p) = \text{SDF}_{\text{obs}}[4]}$$

即：**直接使用中心点的 SDF 值作为 CBF**。

**物理意义**:
- $h > 0$: 无人机距离最近障碍物表面的距离（安全）
- $h = 0$: 无人机恰好接触障碍物表面
- $h < 0$: 无人机进入障碍物内部（碰撞）

**梯度估计**（从 3×3 网格数值计算）:
$$\nabla_p h \approx \begin{bmatrix}
\frac{\text{SDF}[5] - \text{SDF}[3]}{2\Delta} \\
\frac{\text{SDF}[7] - \text{SDF}[1]}{2\Delta} \\
0
\end{bmatrix} = \begin{bmatrix} n_x \\ n_y \\ 0 \end{bmatrix}$$

其中 $n = [n_x, n_y, 0]^\top$ 近似为避障方向单位法向量（从最近障碍物指向无人机）。

**⚠️ 梯度计算的关键假设与限制**：

1. **局部一致性假设**：中心差分要求 $\text{SDF}[3]$、$\text{SDF}[4]$、$\text{SDF}[5]$ 对应**同一个障碍物**。

2. **Voronoi 边界问题**：当无人机位于两个障碍物的 Voronoi 边界附近时，不同网格点可能对应不同障碍物，导致梯度不连续。

   ```
   示例：
       Obst A    |    Obst B
          ●      |       ●
       [3] [4]   |   [5]
                 ↑ Voronoi 边界

   此时：grad_x = (sdf[5] - sdf[3]) / 0.2 混合了两个障碍物的信息
   ```

3. **梯度有效性检查**：对于圆形障碍物，理论上 $\|\nabla h\| = 1$。如果 $\|\nabla h\| > 2$，说明梯度可能不可靠。

4. **实际影响**：
   - ✅ **好消息**：Voronoi 边界通常位于两障碍物等距处，此时 $h$ 较大（安全裕度充足）
   - ⚠️ **坏消息**：梯度误差可能导致控制抖动或次优轨迹
   - 🔧 **缓解措施**：使用前向差分 $\nabla h \approx (\text{SDF}[5] - \text{SDF}[4], \text{SDF}[7] - \text{SDF}[4])^\top / \Delta$ 并归一化

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

**关键**：控制输入 $u \in [-1, 1]^4$ 是 RCBF 的输出，与 RL 策略输出 $u_{rl}$ 在同一空间。

### 1. 展开 $\ddot{h}$（基于 SDF 梯度）

$$\ddot{h} = \frac{d}{dt}(n^\top v) = n^\top \dot{v} + \dot{n}^\top v$$

**维度分析**:
- $n^\top \dot{v}$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $\dot{n}^\top v$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）

**关键：$\dot{n}$ 的近似计算（基于 SDF）**

在基于 SDF 的方法中，$n$ 是从 3×3 网格数值估计的梯度：
$$n = \nabla_p h \approx \begin{bmatrix}
\frac{\text{SDF}[5] - \text{SDF}[3]}{2\Delta} \\
\frac{\text{SDF}[7] - \text{SDF}[1]}{2\Delta} \\
0
\end{bmatrix}$$

**$\dot{n}$ 的近似**：

由于 $n$ 指向远离最近障碍物的方向，对于圆形障碍物有 $n = \frac{p - p_{\text{nearest}}}{\|p - p_{\text{nearest}}\|}$。

使用标准公式：
$$\dot{n} = \frac{v}{\|p - p_{\text{nearest}}\|} - \frac{(p - p_{\text{nearest}})}{\|p - p_{\text{nearest}}\|^3} (p - p_{\text{nearest}})^\top v$$

简化为：
$$\dot{n} = \frac{1}{h + R_{\text{obs}}} (I - n n^\top) v$$

其中 $h + R_{\text{obs}} = \|p - p_{\text{nearest}}\|$ 是到障碍物中心的距离。

**进一步简化**：
$$\dot{n}^\top v = \frac{1}{h + R_{\text{obs}}} v^\top (I - n n^\top) v = \frac{1}{h + R_{\text{obs}}} (\|v\|^2 - (n^\top v)^2)$$

**注意**：
1. 当最近障碍物切换时，$n$ 会跳变，但 $\dot{n}$ 仍可计算
2. 如果 $h + R_{\text{obs}} \to 0$（接近碰撞），$\dot{n}$ 会变大（保守）
3. 对于 2D 障碍物（$n_z = 0$），只需考虑 XY 平面的速度分量

$$\dot{n} = \frac{v}{\|p - p_{obs}\|} - \frac{n (n^\top v)}{\|p - p_{obs}\|} = \frac{1}{\|p - p_{obs}\|} \left( v - n (n^\top v) \right)$$

**但在 SDF 方法中，我们用 $h + R_{\text{obs}}$ 近似 $\|p - p_{\text{nearest}}\|$**：

$$\boxed{\dot{n} = \frac{1}{h + R_{\text{obs}}} \left( v - n (n^\top v) \right)}$$

**维度分析**:
- $v$: $3 \times 1$
- $n^\top v$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $n (n^\top v)$: $(3 \times 1) \cdot 1 \to 3 \times 1$
- $v - n (n^\top v)$: $3 \times 1 - 3 \times 1 \to 3 \times 1$
- $\frac{1}{h + R_{\text{obs}}}$: 标量
- **结果**: $\dot{n} \in \mathbb{R}^{3 \times 1}$

**物理意义**: $\dot{n}$ 是速度在**垂直于 $n$ 方向**的分量除以距离。这是向心加速度项，表示当无人机沿切向运动时，法向量 $n$ 的旋转速率。

**$\dot{n}^\top v$ 的物理解释**:
$$\dot{n}^\top v = \frac{1}{h + R_{\text{obs}}} \left( \|v\|^2 - (n^\top v)^2 \right) = \frac{\|v_\perp\|^2}{h + R_{\text{obs}} + \epsilon}$$

**数值保护**（重要！）:
- **问题**：当网格探测误差导致 $h + R_{\text{obs}}$ 极小时，会产生数值爆炸
- **解决方案**：添加 $\epsilon = 10^{-6}$ 防止除零
- **代码实现**：`denom = max(h + R_obs, 1e-6)`

**维度分析**:
- $\dot{n}^\top$: $1 \times 3$
- $v$: $3 \times 1$
- **结果**: $\dot{n}^\top v \to 1$（标量）

其中 $v_\perp = v - (n^\top v)n$ 是速度在垂直于 $n$ 方向的分量。这是**离心加速度项**，表示切向运动导致的"远离"效应。

**量纲**: $[\dot{n}] = \text{s}^{-1}$, $[\dot{n}^\top v] = \text{m/s}^2$

**SDF 方法的实现注意事项**：
1. $R_{\text{obs}}$ 可从环境配置获取（默认 0.5m）
2. 当 $h \to 0$ 时，$\dot{n}^\top v$ 会变大（保守，有利于安全）
3. 对于 2D 障碍物，$n_z = 0$，只需计算 XY 平面分量

### 2. 代入无人机平移动力学

**实际推力与控制输入的关系**：
$$u_{thrust} = 0.5 \times (u + 1), \quad u \in [-1, 1]^4$$

无人机平移动力学（基于实际推力 $u_{thrust}$）：
$$\dot{v} = \frac{T_{max}}{m} R e_3 (\mathbf{1}^\top u_{thrust}) - g e_3 + d$$

代入 $u_{thrust} = 0.5(u + 1)$：
$$\begin{aligned}
\dot{v} &= \frac{T_{max}}{m} R e_3 \left(\mathbf{1}^\top \cdot 0.5(u + 1)\right) - g e_3 + d \\
&= \frac{T_{max}}{2m} R e_3 (\mathbf{1}^\top u) + \frac{2 T_{max}}{m} R e_3 - g e_3 + d
\end{aligned}$$

**维度分析**:
- $\frac{T_{max}}{2m}$: $\text{N}/\text{kg} = \text{m/s}^2$（标量）
- $R e_3$: $(3 \times 3) \cdot (3 \times 1) \to 3 \times 1$
- $\mathbf{1}^\top u$: $(1 \times 4) \cdot (4 \times 1) \to 1$（标量，总推力比例）
- $\frac{T_{max}}{2m} R e_3 (\mathbf{1}^\top u)$: $\text{m/s}^2 \cdot (3 \times 1) \cdot 1 \to 3 \times 1$
- $\frac{2 T_{max}}{m} R e_3$: $\text{m/s}^2 \cdot (3 \times 1) \to 3 \times 1$（偏置项）
- $g e_3$: $\text{m/s}^2 \cdot (3 \times 1) \to 3 \times 1$
- $d$: $3 \times 1$
- **结果**: $\dot{v} \in \mathbb{R}^{3 \times 1}$

其中:
- $R e_3 \in \mathbb{R}^3$: 推力方向单位向量（机体系 z 轴在惯性系中的方向）
- $\mathbf{1}^\top = [1, 1, 1, 1] \in \mathbb{R}^{1 \times 4}$
- $\mathbf{1}^\top u = \sum_{i=1}^4 u_i$: 总推力比例（在 $[-4, 4]$ 范围）
- $d \in \mathbb{R}^3$: 外部扰动加速度，$\|d\| \le d_{max}$

代入 $\ddot{h}$：

$$\begin{aligned}
\ddot{h} &= n^\top \dot{v} + \dot{n}^\top v \\
&= n^\top \left[ \frac{T_{max}}{2m} R e_3 (\mathbf{1}^\top u) + \frac{2 T_{max}}{m} R e_3 - g e_3 + d \right] + \dot{n}^\top v \\
&= \underbrace{\left[ \frac{T_{max}}{2m} (n^\top R e_3) \mathbf{1}^\top \right]}_{A \in \mathbb{R}^{1 \times 4}} u + \frac{2 T_{max}}{m} (n^\top R e_3) - n^\top g e_3 + \dot{n}^\top v + n^\top d
\end{aligned}$$

**$A$ 矩阵维度推导**:
$$
\begin{aligned}
A &= \frac{T_{max}}{2m} (n^\top R e_3) \mathbf{1}^\top \\
&= \underbrace{\frac{T_{max}}{2m}}_{\text{标量}} \cdot \underbrace{(n^\top R e_3)}_{1 \times 1} \cdot \underbrace{\mathbf{1}^\top}_{1 \times 4} \\
&\to 1 \times 4
\end{aligned}
$$

### 3. 确定约束项

**$A$ 矩阵** ($1 \times 4$): 表征了总推力在避障法线 $n$ 上的投影效率。

$$A = \frac{T_{max}}{2m} (n^\top R e_3) \mathbf{1}^\top = \frac{T_{max}}{2m} (n^\top R e_3) \begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix}$$

**维度推导**:
$$
\begin{bmatrix}
A_1 & A_2 & A_3 & A_4
\end{bmatrix}
= \frac{T_{max}}{2m} (n^\top R e_3)
\begin{bmatrix}
1 & 1 & 1 & 1
\end{bmatrix}
$$

**物理解释**: 
- $n^\top R e_3$: 推力方向与避障法线的夹角的余弦（标量）
- $\mathbf{1}^\top u$: 四个电机的总推力比例（标量，范围 $[-4, 4]$）
- 因为球体模型下，四个电机对位置加速度的影响相同，所以 $A$ 的四个元素相同

**量纲**: $[A] = \frac{\text{N}}{\text{kg}} \cdot 1 = \text{m/s}^2$（每单位 $u$ 产生的加速度）

**$A u$ 维度验证**:
$$(1 \times 4) \cdot (4 \times 1) \to 1 \times 1 \quad \text{（标量，与 } \ddot{h} \text{ 一致）}$$

**$b$ 标量**:

将 $\ddot{h}$ 代入 RCBF 条件 $\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge \text{Risk}$：

$$A u + \frac{2 T_{max}}{m} (n^\top R e_3) - n^\top g e_3 + \dot{n}^\top v + n^\top d + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge \text{Risk}$$

**各项维度检查**:
- $A u$: $(1 \times 4) \cdot (4 \times 1) \to 1$（标量）
- $\frac{2 T_{max}}{m} (n^\top R e_3)$: $\text{m/s}^2 \cdot 1 \to 1$（标量，偏置项）
- $n^\top g e_3$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $\dot{n}^\top v$: $(1 \times 3) \cdot (3 \times 1) \to 1$（标量）
- $(\alpha_1 + \alpha_2)\dot{h}$: $\text{s}^{-1} \cdot \text{m/s} \to \text{m/s}^2$（标量）
- $\alpha_1 \alpha_2 h$: $\text{s}^{-2} \cdot \text{m} \to \text{m/s}^2$（标量）
- $\text{Risk}$: $\text{m/s}^2$（标量）

移项得到 $A u \ge b$ 的形式：

$$A u \ge \text{Risk} + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)\dot{h} - \alpha_1 \alpha_2 h - \frac{2 T_{max}}{m} (n^\top R e_3)$$

代入 $\dot{h} = n^\top v$ 和 $h = \text{SDF}_{\text{obs}}[4]$，得到：

$$\boxed{b = \text{Risk} + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 \cdot \text{SDF}_{\text{obs}}[4] - \frac{2 T_{max}}{m} (n^\top R e_3)}$$

**量纲检查**: $[b] = \text{m/s}^2$ ✅

**$A u \ge b$ 维度验证**:
$$(1 \times 4) \cdot (4 \times 1) \ge 1 \quad \Rightarrow \quad 1 \ge 1 \quad \text{✓}$$

---

## 三、Risk 项的具体设计（论文创新点）

由于球体模型忽略了姿态，我们将论文中的干扰项 $\min \nabla h^\top \Psi$ 具象化为针对欠驱动特性的风险补偿。

### 1. Risk 项的说明

Risk 项专门针对**欠驱动系统的姿态波动风险**进行补偿，这是本文的核心创新点。

传统 RCBF 方法通常假设系统为全驱动或忽略姿态动力学，但四旋翼无人机是典型的欠驱动系统：水平加速度必须通过 Roll/Pitch 翻转产生，这会导致瞬时升力损失和位置控制的不确定性。

### 2. 姿态波动风险项 $\text{Risk}_{att}$

无人机产生水平加速度必须先进行 Roll/Pitch 翻转，这会导致瞬时升力损失。

$$\boxed{\text{Risk}_{att} = k_\omega \|\omega_{xy}\|^2}$$

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

### 3. 最终 Risk 表达式

$$\boxed{\text{Risk} = k_\omega \|\omega_{xy}\|^2 + \epsilon}$$

其中 $\epsilon$ 是小常数（如 $\epsilon = 0.1 \, \text{m/s}^2$）用于数值稳定性。

**量纲检查**: 所有项均为 $\text{m/s}^2$ ✅

---

## 四、完整 RCBF-QP 公式

### 1. 优化问题（在 $u \in [-1, 1]^4$ 空间）

给定 RL 策略输出 $u_{rl} \in [-1, 1]^4$，求解安全控制 $u \in [-1, 1]^4$：

$$\boxed{
\begin{aligned}
u^* = \arg\min_{u \in \mathbb{R}^4} \quad & \|u - u_{rl}\|^2 \\
\text{s.t.} \quad & A u \ge b \\
& -1 \le u_i \le 1, \quad i = 1,2,3,4
\end{aligned}
}$$

**关键说明**：
- **优化变量**：$u \in [-1, 1]^4$ 是 RCBF 修正后的安全控制
- **标称控制**：$u_{rl} \in [-1, 1]^4$ 是 RL 策略当前输出
- **最小干预**：$\|u - u_{rl}\|^2$ 最小化 RCBF 对 RL 策略的修正
- **边界约束**：$-1 \le u_i \le 1$ 与 RL 动作空间完全对齐
- **输出**：$u^* \in [-1, 1]^4$ 直接输出，无需转换

其中：
$$
\begin{aligned}
A &= \frac{T_{max}}{2m} (n^\top R e_3) \begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix} \\
b &= \underbrace{k_\omega \|\omega_{xy}\|^2 + \epsilon}_{\text{Risk}} + \underbrace{n^\top g e_3}_{\text{重力}} - \underbrace{\dot{n}^\top v}_{\text{离心}} - \underbrace{(\alpha_1 + \alpha_2)(n^\top v)}_{\text{速度}} - \underbrace{\alpha_1 \alpha_2 \cdot \text{SDF}_{\text{obs}}[4]}_{\text{位置}} - \underbrace{\frac{2 T_{max}}{m} (n^\top R e_3)}_{\text{偏置项}}
\end{aligned}
$$

**关键简化**：由于 $h = \text{SDF}_{\text{obs}}[4]$ 已经是到**最近障碍物**的距离，我们只需要**一个障碍物约束**，而不是 $n_{obs}$ 个！

其中：
- $A \in \mathbb{R}^{1 \times 4}$：单个约束的控制矩阵（标量行向量）
- $b \in \mathbb{R}$：单个约束的右端项（标量）

### 2. 约束矩阵汇总（基于 SDF）

将所有约束组装成矩阵形式 $G u \le h$（标准 QP 格式）：

$$G = \begin{bmatrix}
-A \\
I_4 \\
-I_4
\end{bmatrix} \in \mathbb{R}^{9 \times 4}, \quad
h = \begin{bmatrix}
-b \\
\mathbf{1}_4 \\
\mathbf{1}_4
\end{bmatrix} \in \mathbb{R}^{9}$$

**维度分析**：
- 障碍物约束：1 个（固定！）
- 上界约束：4 个（$u_i \le 1$）
- 下界约束：4 个（$u_i \ge -1$）
- **总约束数**：$1 + 4 + 4 = 9$ 个（与障碍物数量无关）

**对比传统方法**：
- 传统 RCBF：$n_{obs} + 8$ 个约束（随障碍物数量线性增长）
- SDF-RCBF：$9$ 个约束（固定）
- **优势**：计算复杂度从 $O(n_{obs})$ 降为 $O(1)$

### 3. 参数选择指南

| 参数 | 符号 | 推荐值 | 量纲 | 影响 |
|------|------|--------|------|------|
| CBF 增益 1 | $\alpha_1$ | 1.0 - 3.0 | s⁻¹ | 大→响应快，可能震荡 |
| CBF 增益 2 | $\alpha_2$ | 1.0 - 3.0 | s⁻¹ | 大→收敛快，更保守 |
| 障碍物半径 | $R_{obs}$ | 0.5 m | m | 从环境配置获取 |
| SDF 分辨率 | $\Delta$ | 0.1 m | m | 影响梯度精度 |
| 角速度增益 | $k_\omega$ | 0.05 - 0.2 | m/rad² | 大→抑制剧烈机动 |

---

## 五、可行性分析（基于 Crazyflie 实际参数）

### 1. 可行性条件推导

RCBF-QP 可行的充分条件是：存在 $u \in [-1, 1]^4$ 使得 $A u \ge b$。

**最大可用控制权限**:
$$\max_{u \in [-1,1]^4} A u = \frac{T_{max}}{2m} (n^\top R e_3) \cdot 4 = \frac{2 T_{max}}{m} (n^\top R e_3)$$

**注意**：当 $u = [1, 1, 1, 1]^\top$ 时，实际推力最大；当 $u = [-1, -1, -1, -1]^\top$ 时，实际推力为 0。

最坏情况下（推力方向与避障法线反向），$n^\top R e_3 = -1$，此时无法产生远离障碍物的加速度。

**可行条件**（假设 $n^\top R e_3 > 0$，即推力可以指向远离障碍物方向）:

$$\frac{2 T_{max}}{m} (n^\top R e_3) \ge b_{max}$$

其中 $b_{max}$ 是 $b$ 的最大可能值。根据第二节推导：

$$b = \text{Risk} + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)\dot{h} - \alpha_1 \alpha_2 h - \frac{2 T_{max}}{m} (n^\top R e_3)$$

**最坏情况分析**:
- $\dot{n}^\top v = \frac{\|v_\perp\|^2}{\|p - p_{obs}\|} \ge 0$（离心项，有助于安全）
- $\dot{h} = n^\top v$ 可正可负（负值表示接近障碍物）
- $h \ge 0$（在安全集内）
- $n^\top g e_3 \le g$（重力项最大为 $g$）

当 $h \to 0$ 且 $\dot{h} < 0$（快速接近障碍物）时，$b$ 达到最大值：

$$b_{max} \approx \text{Risk} + g - \frac{2 T_{max}}{m} (n^\top R e_3) + (\alpha_1 + \alpha_2)|\dot{h}_{max}|$$

代入 $\text{Risk} = k_\omega \|\omega_{xy}\|^2 + \epsilon$，并假设 $(n^\top R e_3)_{min} \approx 1$：

$$b_{max} \approx \text{Risk} + g - \frac{2 T_{max}}{m} + (\alpha_1 + \alpha_2)|v_{max}|$$

**可行性条件**:

$$\frac{2 T_{max}}{m} > \text{Risk} + g - \frac{2 T_{max}}{m} + (\alpha_1 + \alpha_2)|v_{max}|$$

整理得到：

$$\boxed{\frac{4 T_{max}}{m} > g + \max(\text{Risk}) + (\alpha_1 + \alpha_2)|v_{max}|}$$

**物理解读**: 最大推力必须能够克服重力、姿态波动风险和紧急制动需求。（与原推导一致！）

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

$$29.43 > 9.81 + \underbrace{k_\omega \|\omega_{xy}\|^2 + \epsilon}_{\text{Risk}} + (\alpha_1 + \alpha_2) \times 3.0$$

**典型 Risk 值估计**（假设 $k_\omega = 0.1$，$\epsilon = 0.1$）:
- 当 $\|\omega_{xy}\| = 0 \, \text{rad/s}$（悬停）时：$\text{Risk} \approx 0.1 \, \text{m/s}^2$
- 当 $\|\omega_{xy}\| = 10 \, \text{rad/s}$（中等角速度）时：$\text{Risk} \approx 0.1 \times 100 + 0.1 = 10.1 \, \text{m/s}^2$
- 当 $\|\omega_{xy}\| = 20 \, \text{rad/s}$（较高角速度）时：$\text{Risk} \approx 0.1 \times 400 + 0.1 = 40.1 \, \text{m/s}^2$（**已超出最大推力！**）
- 当 $\|\omega_{xy}\| = 15 \, \text{rad/s}$（高角速度）时：$\text{Risk} \approx 0.1 \times 225 + 0.1 = 22.6 \, \text{m/s}^2$

**不同参数组合下的可行性检查**（考虑 $\text{Risk}$ 项，`thrust_to_weight=3.0`）:

| $\alpha_1 = \alpha_2$ | $\|\omega_{xy}\|$ (rad/s) | Risk (m/s²) | 右侧 (m/s²) | 可行性 | 裕度 |
|---------------------|-------------------------|-------------|-------------|--------|------|
| 0.5 | 0 | 0.1 | 12.91 | ✓ | 16.52 |
| 0.5 | 5 | 2.6 | 15.41 | ✓ | 14.02 |
| 0.5 | 10 | 10.1 | 22.91 | ✓ | 6.52 |
| 0.5 | 15 | 22.6 | 35.41 | ✗ | -5.98 |
| 1.0 | 0 | 0.1 | 15.91 | ✓ | 13.52 |
| 1.0 | 5 | 2.6 | 18.41 | ✓ | 11.02 |
| 1.0 | 10 | 10.1 | 25.91 | ✓ | 3.52 |
| 1.0 | 15 | 22.6 | 38.41 | ✗ | -8.98 |

### 2.5 最大可抵抗角速度计算

根据可行性条件，可以反推出系统能够抵抗的最大角速度 $\|\omega_{xy}\|_{max}$：

由可行性条件：
$$\frac{4 T_{max}}{m} > g + k_\omega \|\omega_{xy}\|^2 + \epsilon + (\alpha_1 + \alpha_2)|v_{max}|$$

整理得到：
$$k_\omega \|\omega_{xy}\|^2 < \frac{4 T_{max}}{m} - g - \epsilon - (\alpha_1 + \alpha_2)|v_{max}|$$

定义**可用推力裕度**：
$$\Delta a = \frac{4 T_{max}}{m} - g - \epsilon - (\alpha_1 + \alpha_2)|v_{max}|$$

则最大可抵抗角速度为：
$$\boxed{\|\omega_{xy}\|_{max} = \sqrt{\frac{\Delta a}{k_\omega}}}$$

**不同参数组合下的最大可抵抗角速度**（假设 $k_\omega = 0.1$，$\epsilon = 0.1$）:

| $\alpha_1 = \alpha_2$ | $v_{max}$ (m/s) | $\Delta a$ (m/s²) | $\|\omega_{xy}\|_{max}$ (rad/s) | 说明 |
|---------------------|-----------------|-------------------|--------------------------------|------|
| 0.5 | 3.0 | 16.52 | **12.85** | 宽松增益 |
| 1.0 | 3.0 | 13.52 | **11.63** | 中等增益 |
| 1.0 | 2.0 | 16.52 | **12.85** | 降速飞行 |
| 1.5 | 3.0 | 10.52 | **10.26** | 激进增益 |
| 2.0 | 3.0 | 7.52 | **8.67** | 高增益（受限） |

**物理解读**:
- $\|\omega_{xy}\|_{max}$ 表示 RCBF-QP 能够安全处理的最大角速度
- 当实际角速度 $\|\omega_{xy}\| > \|\omega_{xy}\|_{max}$ 时，QP 可能无解（不可行）
- 降低 CBF 增益 $\alpha_1, \alpha_2$ 或限制飞行速度可以提升 $\|\omega_{xy}\|_{max}$

**设计建议**:
- 对于 $k_\omega = 0.1$，$\alpha_1 = \alpha_2 = 1.0$ 的典型配置，$\|\omega_{xy}\|_{max} \approx 11.6 \, \text{rad/s}$
- 建议在策略层限制角速度 $\|\omega_{xy}\| < 0.8 \times \|\omega_{xy}\|_{max} \approx 9 \, \text{rad/s}$（保留 20% 安全裕度）

**结论**: 
1. 当 `thrust_to_weight=3.0` 时，**可行性条件显著改善**，最大加速度从 18.57 m/s² 提升至 29.43 m/s²
2. 在中等角速度（$\|\omega_{xy}\| \le 10 \, \text{rad/s}$）下，大多数参数组合可行
3. 当 $\|\omega_{xy}\| > \|\omega_{xy}\|_{max}$ 时，QP 不可行（Risk 项过大）
4. 建议参数选择：
   - $\alpha_1 = \alpha_2 \le 1.0 \, \text{s}^{-1}$（CBF 增益）
   - $\|\omega_{xy}\| \le 10 \, \text{rad/s}$（限制角速度）
   - 或者限制最大速度 $v_{max} \le 2.0 \, \text{m/s}$
5. 当 $\alpha_1 = \alpha_2 = 1.0$，$\|\omega_{xy}\| = 10 \, \text{rad/s}$ 时，系统有约 **3.52 m/s²** 的裕度（可行）

**改进方案**:
1. **使用更强力的无人机**（`thrust_to_weight=3.0` 已显著改善可行性）
2. 降低 CBF 增益 $\alpha_1, \alpha_2$ 至 1.0 以下
3. 限制最大飞行速度 $v_{max}$ 至 2.0 m/s 以下
4. **限制最大角速度 $\|\omega_{xy}\|$ 至 10 rad/s 以下**（关键！）
5. 减小 Risk 增益 $k_\omega$ 至 0.05 以下（可降低 Risk 项影响）

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

## 七、基于 SDF 的 RCBF 创新点总结

1. **固定约束数量的 RCBF**: 使用 Signed Distance Field (SDF) 替代显式障碍物列表，实现了与障碍物数量无关的固定约束数量（始终 1 个障碍物约束），解决了传统 RCBF 在非结构化环境中约束数量爆炸的问题。

2. **数值梯度估计**: 从环境提供的 3×3 局部 SDF 网格直接计算 CBF 梯度，无需显式障碍物位置信息，实现了完全隐式的障碍物表示。

3. **自动障碍物切换**: SDF 自动跟踪最近障碍物，当最近障碍物切换时，RCBF-QP 仍可正常求解（每步独立优化），无需额外的切换逻辑。

4. **端到端的可微安全约束层**: 在 Sample Factory 框架下，将基于 SDF 的二阶 RCBF 转化为可微 QP 层，实现了梯度从避障边界向策略网络的直接回传。

5. **欠驱动动力学补偿机制**: 提出了一种基于机身角速度 $\omega$ 的动态 Risk 补偿项，解决了球形碰撞实体在高速机动中由于忽略姿态变化而导致的安全性下降问题。

6. **动作空间对齐设计**: RCBF 输出 $u \in [-1, 1]^4$ 与 RL 策略输出 $u_{rl}$ 完全对齐，实现最小干预原则。

7. **完整的量纲分析**: 本推导提供了所有公式的量纲检查，确保物理一致性。

---

## 附录：推导检查清单

| 公式 | 量纲 | 备注 |
|------|------|------|
| $h = \text{SDF}_{\text{obs}}[4]$ | m | **位置屏障（基于 SDF）** |
| $n = \nabla_p h \approx [\frac{\text{SDF}[5]-\text{SDF}[3]}{2\Delta}, \frac{\text{SDF}[7]-\text{SDF}[1]}{2\Delta}, 0]^\top$ | - | **梯度（从 3×3 网格估计）** |
| $\dot{h} = n^\top v$ | m/s | 屏障一阶导 |
| $\psi = \dot{h} + \alpha_1 h$ | m/s | 级联屏障 |
| $\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge \text{Risk}$ | m/s² | RCBF 条件 |
| $\dot{n}^\top v = \frac{\|v\|^2 - (n^\top v)^2}{h + R_{\text{obs}}}$ | m/s² | **离心项（SDF 版本）** |
| $A = \frac{T_{max}}{2m} (n^\top R e_3) \mathbf{1}^\top$ | m/s² | 控制矩阵 |
| $b = \text{Risk} + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 \cdot \text{SDF}_{\text{obs}}[4] - \frac{2 T_{max}}{m} (n^\top R e_3)$ | m/s² | 约束向量 |
| $\text{Risk} = k_\omega \|\omega_{xy}\|^2 + \epsilon$ | m/s² | **姿态波动风险项** |

---

## 关键公式汇总（基于 SDF）

### RCBF-QP 完整形式（在 $u \in [-1, 1]^4$ 空间）

$$
\begin{aligned}
u^* = \arg\min_{u} \quad & \|u - u_{rl\_nom}\|^2 \\
\text{s.t.} \quad & A u \ge b \\
& -1 \le u_i \le 1, \quad i=1,2,3,4
\end{aligned}
$$

其中：
$$
\begin{aligned}
A &= \frac{T_{max}}{2m} (n^\top R e_3) \begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix} \\
b &= \underbrace{k_\omega \|\omega_{xy}\|^2 + \epsilon}_{\text{Risk}} + \underbrace{n^\top g e_3}_{\text{重力}} - \underbrace{\dot{n}^\top v}_{\text{离心}} - \underbrace{(\alpha_1 + \alpha_2)(n^\top v)}_{\text{速度}} - \underbrace{\alpha_1 \alpha_2 \cdot \text{SDF}_{\text{obs}}[4]}_{\text{位置}} - \underbrace{\frac{2 T_{max}}{m} (n^\top R e_3)}_{\text{偏置项}}
\end{aligned}
$$

### 动作空间与物理推力

**RL/RCBF 动作空间**：$u, u_{rl} \in [-1, 1]^4$

**物理意义**：
- $u = -1 \Rightarrow$ 最低转速（无推力）
- $u = 0 \Rightarrow$ 中间转速
- $u = 1 \Rightarrow$ 最高转速（最大推力）

**实际推力转换**：
$$u_{thrust} = 0.5 \times (u + 1) \in [0, 1]^4$$

### SDF 相关计算

**从环境观测提取 CBF 信息**：
$$
\begin{aligned}
h &= \text{SDF}_{\text{obs}}[4] \quad \text{（中心点 SDF 值）} \\
n_x &= \frac{\text{SDF}_{\text{obs}}[5] - \text{SDF}_{\text{obs}}[3]}{2\Delta} \\
n_y &= \frac{\text{SDF}_{\text{obs}}[7] - \text{SDF}_{\text{obs}}[1]}{2\Delta} \\
n &= [n_x, n_y, 0]^\top \quad \text{（梯度向量）} \\
\dot{h} &= n^\top v \\
\dot{n}^\top v &= \frac{\|v\|^2 - (n^\top v)^2}{h + R_{\text{obs}}}
\end{aligned}
$$

其中 $\Delta = 0.1$ m（SDF 网格分辨率），$R_{\text{obs}}$ 从环境配置获取（默认 0.5 m）。

### 实现伪代码

```python
def rcbf_qp(u_rl_nom, state, params):
    """
    RCBF-QP 在 [-1, 1]^4 空间直接求解
    
    Args:
        u_rl_nom: (4,) RL 策略输出，范围 [-1, 1]^4
        state: 环境状态（包含 sdf_obs, velocity, R, omega_xy 等）
        params: 参数（T_max, m, alpha1, alpha2, k_omega 等）
    
    Returns:
        u_star: (4,) 安全控制输出，范围 [-1, 1]^4
    """
    # 1. 提取状态
    n, v, R, omega_xy = extract_state(state)
    
    # 2. 计算控制矩阵 A
    A = (params.T_max / (2 * params.m)) * (n.T @ R @ e3) * np.ones(4)
    
    # 3. 计算约束向量 b
    Risk = params.k_omega * np.linalg.norm(omega_xy)**2 + params.epsilon
    h_dot = n.T @ v
    h = state.sdf_obs[4]
    n_dot_v = (np.dot(v, v) - h_dot**2) / (h + params.R_obs + 1e-6)
    
    b = (Risk + n.T @ g * e3 - n_dot_v 
         - (params.alpha1 + params.alpha2) * h_dot 
         - params.alpha1 * params.alpha2 * h
         - (2 * params.T_max / params.m) * (n.T @ R @ e3))
    
    # 4. 求解 QP（在 u ∈ [-1, 1]^4 空间）
    u_star = qp_solve(
        min ||u - u_rl_nom||^2,
        s.t. A @ u >= b,
             -1 <= u <= 1  # ← 边界约束与 RL 对齐
    )
    
    return u_star  # 直接输出，无需转换！
```
