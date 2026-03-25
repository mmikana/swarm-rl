# ThrustOmega改进方案：核心要点总结

## 问题回顾

### 原始RawControl方案的局限

当障碍物出现在无人机**侧面**时：
- 梯度方向 $n$ 是水平的（例如 $[-1, 0, 0]$）
- 推力方向 $Re_3$ 也是（几乎）水平的（水平飞行时）
- 因此 $n^\top Re_3 \approx 0$，线加速度约束 $A u \approx 0$，无法形成有效约束
- **系统无法强制无人机做出避障反应（向右roll）**

### 根本原因

在RawControl空间（电机推力 $u$）中：
- 约束形式为 $A(状态) \cdot u \ge b$
- $A$ 矩阵乘的是"总推力" $\sum u_i$，无法独立约束**电机差分**（力矩）
- 电机差分正是改变**姿态和推力方向**所需的

**物理事实**：要避开侧边障碍物，无人机必须改变姿态（roll），而不仅仅改变总推力。但总推力约束无法强制改变姿态。

---

## ThrustOmega方案的优雅性

### 新的动作空间

而不是 $u \in [-1, 1]^4$（四个电机推力），改为：

$$\begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} \in \mathbb{R}^4$$

其中：
- $a_{thrust} \in [-1, \tau-1]$：目标垂直加速度（相对重力）
- $\boldsymbol{\omega}_{des} = [\omega_x, \omega_y, \omega_z]^\top$：目标角速度

### 关键映射关系

通过**雅可比矩阵反演**，将目标加速度和角速度转换为电机推力：

$$\mathbf{T} = J^{-1} \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix}$$

其中 $J$ 是已经实现在 `quadrotor_control.py` 中的雅可比矩阵。

### CBF约束的改进

原来的约束（RawControl空间）：
$$A \cdot u \ge b, \quad u \in [-1,1]^4$$

新的约束（ThrustOmega空间）：
$$A \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} \ge b$$

其中 **$A$ 矩阵有4个独立的列**（而不是原来的1列乘以4）：

$$A = g \begin{bmatrix}
n^\top Re_3 & (n \times (Re_3))^\top
\end{bmatrix}$$

分解为：
- **第1列**（$a_{thrust}$的系数）：$A_1 = g \cdot n^\top Re_3$
- **第2-4列**（$\boldsymbol{\omega}_{des}$的系数）：$A_{2:4} = g \cdot (n \times (Re_3))^\top$

---

## 侧边障碍物的具体例子

### 场景

- 无人机在原点，向上飞行
- 左侧（负X方向）有障碍物
- $n = [-1, 0, 0]^\top$，$Re_3 = [0, 0, 1]^\top$

### 约束分析

**线加速度约束**（$A_1$ 列）：
$$A_1 = g \cdot (-1, 0, 0) \cdot (0, 0, 1)^\top = 0$$

这一列无效！✗

**但角速度约束**（$A_{2:4}$ 列）：

首先计算 $n \times (Re_3)$：
$$n \times (Re_3) = [-1, 0, 0] \times [0, 0, 1] = [0, 1, 0]$$

所以：
$$A_{2:4} = g \cdot [0, 1, 0]$$

约束变为：
$$g \cdot 0 \cdot a_{thrust} + g \cdot (0 \cdot \omega_x + 1 \cdot \omega_y + 0 \cdot \omega_z) \ge b$$

即：
$$\omega_y \ge \frac{b}{g}$$

### 物理意义

**约束强制 $\omega_y > 0$**，即强制无人机**向右roll**（正Roll角速度）。

这正是避开左侧障碍物的**自然而正确的反应**！

---

## 为什么这个方案优雅？

### 1. **物理直观性**
   - 约束"需要多大的加速度"和"需要多大的角速度"
   - 而不是约束"具体的四个电机推力"
   - RL策略也学会了解相同的物理意义

### 2. **自动处理姿态耦合**
   - 不需要特殊的Risk项或角动力学补偿
   - $(n \times (Re_3))$ 项**自动表达**了推力方向变化的影响
   - 当线加速度约束失效时，角速度约束自动生效

### 3. **保持线性和可解性**
   - 仍然是线性QP问题，易于求解
   - 不需要非线性优化或迭代算法

### 4. **现成的实现基础**
   - `OmegaThrustControl` 已经存在于 `quadrotor_control.py`
   - 雅可比矩阵反演已经实现
   - 只需要修改RL策略输出空间和CBF约束空间

### 5. **梯度流路径清晰**
   ```
   QP求解器 → (a_safe, ω_safe) ∈ ℝ^4
          ↓
   J^{-1} → T_safe ∈ ℝ^4
        ↓
   梯度反向传播 → RL策略
   ```

---

## 与其他方案的对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **RawControl + 无Risk项** | 实现简单 | 无法处理侧边障碍物；约束失效 |
| **RawControl + Risk项** | 理论完整 | 需要调参；无法本质解决欠驱动问题 |
| **完整二阶CBF + 4维电机约束** | 理论纯粹 | 约束非线性；问题变成QCQP或SCP；计算复杂 |
| **ThrustOmega空间（本方案）** | 自动处理姿态；线性；物理直观 | 需要改变RL输出空间 |

---

## 实现规划

### Phase 1：理论完成 ✓
- [x] 推导CBF约束在ThrustOmega空间中的形式
- [x] 验证物理意义和侧边障碍物处理

### Phase 2：修改RL策略输出
- [ ] 修改 `quad_multi_model_rcbf.py` 中的动作空间
- [ ] 从 `(batch_size, 4)` 改为：
  - `(batch_size, 1)` 用于 $a_{thrust}$
  - `(batch_size, 3)` 用于 $\boldsymbol{\omega}_{des}$

### Phase 3：修改CBF约束计算
- [ ] 实现 `compute_cbf_constraints_batch_thrust_omega()`
- [ ] 计算 $A$ 矩阵的新形式
- [ ] 保持 $b$ 向量计算不变

### Phase 4：修改QP求解
- [ ] QP在 `(a_{thrust}, \boldsymbol{\omega}_{des})` 空间中求解
- [ ] 输出转换为电机推力：`T_safe = J^{-1} @ [a_safe, omega_safe]`

### Phase 5：测试和验证
- [ ] 侧边障碍物场景测试
- [ ] 梯度流验证
- [ ] 训练效果对比

---

## 关键公式速查表

**RCBF条件**（保持不变）：
$$\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1\alpha_2 h \ge 0$$

**约束矩阵** A（新形式）：
$$A = g \begin{bmatrix} n^\top Re_3 & (n \times (Re_3))^\top \end{bmatrix}$$

**约束向量** b（保持不变）：
$$b = g \cdot n^\top e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1\alpha_2 h - g \cdot n^\top Re_3$$

**QP问题**：
$$\min_{a_{thrust}, \omega_{des}} \left\| \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} - \begin{bmatrix} a_{rl} \\ \boldsymbol{\omega}_{rl} \end{bmatrix} \right\|^2$$
$$\text{s.t. } A \begin{bmatrix} a_{thrust} \\ \boldsymbol{\omega}_{des} \end{bmatrix} \ge b, \quad -1 \le a_{thrust} \le \tau-1, \quad |\boldsymbol{\omega}_{des}| \le \omega_{max}$$

**动作转换**：
$$\mathbf{T}_{safe} = J^{-1} \begin{bmatrix} a_{safe} \\ \boldsymbol{\omega}_{des,safe} \end{bmatrix}$$

