# RCBF 理论推导：基于 SDF 的障碍物约束（quad-swarm-rl）

> **版本**: 2026-03-20（无 Risk 项简化版）
> **核心创新**: 使用 Signed Distance Field (SDF) 替代显式障碍物列表，实现固定约束数量
> **假设**: 无人机碰撞模型为球体（半径 $R_{drone}$ 为常数），姿态不影响碰撞边界
> **控制输入**: $u \in [-1, 1]^4$（与 RL 策略输出对齐）
> **动力学模型**: Crazyflie（`gym_art/quadrotor_multi/quad_models.py`）
> **障碍物表示**: 3×3 局部 SDF 网格（`gym_art/quadrotor_multi/obstacles/utils.py`）
> **简化**: 移除 Risk 项，CBF 约束仅依赖位置和速度信息

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
| $T_{max}$ | 单电机最大推力 | N | 0.206 N | `thrust_to_weight=3.0` 计算得出 |
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

**RL 环境参数说明**:
- 质量 $m = 0.028 \, \text{kg}$
- 推力重量比 `thrust_to_weight` = 3.0
- 单电机最大推力：$T_{max} = \frac{m \cdot g \cdot \text{thrust\_to\_weight}}{4} \approx 0.206 \, \text{N}$
- 最大速度：$v_{max} = 3.0 \, \text{m/s}$
- 最大角速度：$\omega_{max} = 40 \, \text{rad/s}$
- 障碍物半径：$R_{obs} = 0.15 \sim 0.5 \, \text{m}$

---

## 一、RCBF 二阶级联推导

### 1. 定义原始屏障函数 $h$（基于 SDF）

$$\boxed{h(p) = \text{SDF}_{\text{obs}}[4]}$$

**物理意义**:
- $h > 0$: 无人机距离最近障碍物表面的距离（安全）
- $h = 0$: 无人机恰好接触障碍物表面
- $h < 0$: 无人机进入障碍物内部（碰撞）

**梯度估计**:
$$\nabla_p h \approx \begin{bmatrix}
\frac{\text{SDF}[5] - \text{SDF}[3]}{2\Delta} \\
\frac{\text{SDF}[7] - \text{SDF}[1]}{2\Delta} \\
0
\end{bmatrix} = \begin{bmatrix} n_x \\ n_y \\ 0 \end{bmatrix}$$

### 2. 第一层级联：定义 $\psi$

$$\psi = \dot{h} + \alpha_1 h, \quad \alpha_1 > 0$$

其中 $\dot{h} = n^\top v$

### 3. 第二层级联：RCBF 条件

$$\dot{\psi} \ge -\alpha_2 \psi$$

展开整理得：

$$\boxed{\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge 0}$$

---

## 二、提取控制矩阵 $A$ 与向量 $b$

### 1. 展开 $\ddot{h}$

$$\ddot{h} = n^\top \dot{v} + \dot{n}^\top v$$

其中：
$$\dot{n}^\top v = \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}$$

### 2. 代入无人机动力学

$$\dot{v} = \frac{T_{max}}{2m} R e_3 (\mathbf{1}^\top u) + \frac{2 T_{max}}{m} R e_3 - g e_3$$

代入 $\ddot{h}$：

$$\ddot{h} = \underbrace{\left[ \frac{T_{max}}{2m} (n^\top R e_3) \mathbf{1}^\top \right]}_{A} u + \frac{2 T_{max}}{m} (n^\top R e_3) - n^\top g e_3 + \dot{n}^\top v$$

### 3. 控制矩阵和约束向量

$$A = \frac{T_{max}}{2m} (n^\top R e_3) \begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix}$$

$$b = n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - \frac{2 T_{max}}{m} (n^\top R e_3)$$

---

## 三、完整 RCBF-QP 公式

### 优化问题

$$
\begin{aligned}
u^* = \arg\min_{u} \quad & \|u - u_{rl}\|^2 \\
\text{s.t.} \quad & A u \ge b \\
& -1 \le u_i \le 1, \quad i=1,2,3,4
\end{aligned}
$$

### 约束矩阵汇总

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

**总约束数**: 9 个（1 个避障 + 4 个上界 + 4 个下界）

---

## 四、可行性分析

### 可行性条件

$$\frac{4 T_{max}}{m} > g + (\alpha_1 + \alpha_2)|v_{max}|$$

### Crazyflie 参数验证

$$\frac{4 T_{max}}{m} = \frac{4 \times 0.206}{0.028} = 29.43 \, \text{m/s}^2$$

$$29.43 > 9.81 + (\alpha_1 + \alpha_2) \times 3.0$$

| $\alpha_1 = \alpha_2$ | 右侧 (m/s²) | 可行性 | 裕度 |
|---------------------|-------------|--------|------|
| 0.5 | 12.81 | ✓ | 16.62 |
| 1.0 | 15.81 | ✓ | 13.62 |
| 1.5 | 18.81 | ✓ | 10.62 |
| 2.0 | 21.81 | ✓ | 7.62 |
| 2.5 | 24.81 | ✓ | 4.62 |
| 3.0 | 27.81 | ✓ | 1.62 |

**结论**: 无 Risk 项时，可行性条件大幅改善。即使 $\alpha_1 = \alpha_2 = 3.0$，仍有 1.62 m/s² 的裕度。

---

## 五、推力方向不利时的处理

### 问题

当 $n^\top R e_3 < 0$ 时（推力方向与避障法线夹角 > 90°），RCBF-QP 可能不可行。

### 解决方案

```python
nTRe3 = np.dot(n, Re3)
if nTRe3 < 0.1:
    # 推力方向不利，返回宽松约束
    # 让 RL 策略自由调整姿态
    return np.zeros((1, 4)), -1e6
```

---

## 六、关键公式汇总

### RCBF-QP 完整形式

$$
\begin{aligned}
u^* = \arg\min_{u} \quad & \|u - u_{rl}\|^2 \\
\text{s.t.} \quad & A u \ge b \\
& -1 \le u_i \le 1
\end{aligned}
$$

其中：
$$
\begin{aligned}
A &= \frac{T_{max}}{2m} (n^\top R e_3) \begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix} \\
b &= n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - \frac{2 T_{max}}{m} (n^\top R e_3)
\end{aligned}
$$

### SDF 相关计算

$$
\begin{aligned}
h &= \text{SDF}_{\text{obs}}[4] \\
n_x &= \frac{\text{SDF}[5] - \text{SDF}[3]}{2\Delta} \\
n_y &= \frac{\text{SDF}[7] - \text{SDF}[1]}{2\Delta} \\
\dot{h} &= n^\top v \\
\dot{n}^\top v &= \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}
\end{aligned}
$$

---

## 七、创新点总结

1. **固定约束数量的 RCBF**: 使用 SDF 替代显式障碍物列表，实现固定约束数量（始终 1 个障碍物约束）

2. **数值梯度估计**: 从 3×3 局部 SDF 网格直接计算 CBF 梯度

3. **自动障碍物切换**: SDF 自动跟踪最近障碍物

4. **端到端的可微安全约束层**: 在 Sample Factory 框架下，将 RCBF 转化为可微 QP 层

5. **动作空间对齐设计**: RCBF 输出 $u \in [-1, 1]^4$ 与 RL 策略输出完全对齐

---

## 附录：推导检查清单

| 公式 | 量纲 | 备注 |
|------|------|------|
| $h = \text{SDF}_{\text{obs}}[4]$ | m | 位置屏障 |
| $n = \nabla_p h$ | - | 梯度（从 3×3 网格估计） |
| $\dot{h} = n^\top v$ | m/s | 屏障一阶导 |
| $\psi = \dot{h} + \alpha_1 h$ | m/s | 级联屏障 |
| $\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge 0$ | m/s² | RCBF 条件 |
| $\dot{n}^\top v = \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}$ | m/s² | 离心项 |
| $A = \frac{T_{max}}{2m} (n^\top R e_3) \mathbf{1}^\top$ | m/s² | 控制矩阵 |
| $b = n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - \frac{2 T_{max}}{m} (n^\top R e_3)$ | m/s² | 约束向量 |
