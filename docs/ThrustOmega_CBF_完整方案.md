# ThrustOmega空间中的CBF完整方案

> **版本**: 2026-03-23（最终方案）
> **核心思想**: 将RL动作空间从RawControl改为ThrustOmega，在归一化动作空间中进行CBF优化
> **关键创新**: 直接约束Roll/Pitch角速度，自动处理欠驱动系统的姿态耦合

---

## 一、问题背景

### 原始问题
- RawControl空间中的CBF约束只能约束总推力 $\sum T_i$
- 当障碍物在侧边时，$n^\top Re_3 \approx 0$，约束失效
- 无法强制无人机做出正确的避障反应（向安全方向倾斜）

### 根本原因
- 四旋翼是欠驱动系统
- 水平加速度需要通过改变姿态（Roll/Pitch）产生
- 但RawControl空间无法独立约束电机差分（力矩）

---

## 二、解决方案：ThrustOmega空间

### 2.1 动作空间转换

**从RawControl改为ThrustOmega**：

| 维度 | RawControl | ThrustOmega | 范围 | 物理意义 |
|------|-----------|-------------|------|---------|
| 1 | $u_1$ | $a_{thrust}$ | $[-1, 2]$ | 目标加速度（相对g） |
| 2 | $u_2$ | $\omega_x$ | $[-31.42, 31.42]$ | 目标Roll角速度 |
| 3 | $u_3$ | $\omega_y$ | $[-31.42, 31.42]$ | 目标Pitch角速度 |
| 4 | $u_4$ | $\omega_z$ | $[-6.28, 6.28]$ | 目标Yaw角速度 |

### 2.2 动作转换

通过**雅可比矩阵反演**将ThrustOmega转换为电机推力：

$$\mathbf{T} = J^{-1} \begin{bmatrix} a_{thrust} \\ \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}$$

其中 $J$ 是已在 `quadrotor_control.py` 中实现的雅可比矩阵。

---

## 三、CBF约束推导

### 3.1 基于SDF的屏障函数

$$h(p) = \text{SDF}_{\text{obs}}[4]$$

其中 $h > 0$ 表示安全，$h = 0$ 表示接触障碍物。

### 3.2 二阶RCBF条件

$$\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \ge 0$$

### 3.3 关键简化：圆柱形障碍物

由于所有障碍物都是圆柱形（贯穿Z轴），梯度总是在XY平面：

$$n = [n_x, n_y, 0]^\top, \quad n_z = 0 \text{ 总是成立}$$

### 3.4 约束的最终形式

在水平飞行假设下（$Re_3 \approx [0, 0, 1]^\top$）：

$$\boxed{T \cdot (n_x \omega_y - n_y \omega_x) \ge b'}$$

其中：
$$T = m \cdot g \cdot (a_{thrust} + 1)$$

$$b' = -m[\dot{n}^\top v + (\alpha_1 + \alpha_2)(n^\top v) + \alpha_1 \alpha_2 h]$$

$$\dot{n}^\top v = \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}$$

### 3.5 物理意义

- **约束只涉及 $(\omega_x, \omega_y)$**：这两个维度直接产生水平加速度
- **$a_{thrust}$ 对约束无直接影响**：但可以改变水平加速度的大小
- **$\omega_z$ 对约束无直接影响**：但可以改变Roll/Pitch的作用方向

---

## 四、CBF优化问题

### 4.1 优化目标

在**归一化的ThrustOmega动作空间**中，最小化与RL输出的偏差：

$$\boxed{
\min_{\mathbf{a}} \left\| \begin{bmatrix} a_{thrust} \\ \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} - \begin{bmatrix} a_{thrust,rl} \\ \omega_{x,rl} \\ \omega_{y,rl} \\ \omega_{z,rl} \end{bmatrix} \right\|^2
}$$

### 4.2 约束条件

$$\boxed{
\begin{aligned}
\text{s.t.} \quad & T \cdot (n_x \omega_y - n_y \omega_x) \ge b' \\
& -1 \le a_{thrust} \le 2 \\
& -31.42 \le \omega_x, \omega_y \le 31.42 \\
& -6.28 \le \omega_z \le 6.28
\end{aligned}
}$$

### 4.3 为什么这个设计是最优的？

1. **最小干预原则**
   - 所有4个维度同等对待
   - 在满足安全约束的前提下，找最接近RL输出的控制

2. **梯度流清晰**
   - 所有维度都在目标函数中
   - 所有维度都有梯度
   - RL可以充分学习

3. **物理自然性**
   - 约束只涉及 $(\omega_x, \omega_y)$，所以通常只这两个被修改
   - $(a_{thrust}, \omega_z)$ 通常不被修改（除非有帮助）
   - QP自动做出最优决策

4. **没有超参数**
   - 不需要手动调整权重
   - RL会通过梯度学到各维度的重要性

5. **标准凸QP问题**
   - 可微分，梯度清晰
   - 易于求解

---

## 五、可行性分析

### 5.1 约束可行性

**最坏情况**（紧急避障）：
- 无人机正向障碍物移动，速度 3.0 m/s
- 距离 $h \to 0$
- 所需约束值：$b' \approx 6.0$ m/s²

**所需角速度**：
$$\omega_{xy,required} = \frac{b'}{T} \approx 0.61 \text{ rad/s}$$

**可用角速度**：
$$\omega_{xy,max} = 31.42 \text{ rad/s}$$

**裕度**：$31.42 / 0.61 \approx 52$ 倍 ✓ **充分可行**

### 5.2 梯度流可行性

即使某个维度没有被约束修改，梯度仍然会通过目标函数流回去：

```
Loss = f(a_safe, omega_safe)
         ↑
    QP输出（通过隐函数定理）
         ↑
    目标函数：||a_safe - a_rl||^2
         ↑
RL输出 (a_rl, omega_rl)
```

所有维度都有梯度，RL可以充分学习。

---

## 六、实现步骤

### 6.1 修改RL策略输出空间

**当前**：RL输出 $u \in [-1, 1]^4$（RawControl）

**改为**：RL输出 $(a_{thrust}, \omega_x, \omega_y, \omega_z)$（ThrustOmega）

在 `quad_multi_model_rcbf.py` 中修改：
```python
# 原来
action_output = self.policy_head(features)  # shape: (batch, 4)

# 改为
a_thrust = self.policy_head_a(features)  # shape: (batch, 1)
omega = self.policy_head_omega(features)  # shape: (batch, 3)
action_output = torch.cat([a_thrust, omega], dim=1)  # shape: (batch, 4)
```

### 6.2 实现CBF约束计算

在 `quad_cbf_qp.py` 中实现 `compute_cbf_constraints_batch_thrust_omega()`：

```python
def compute_cbf_constraints_batch_thrust_omega(self, state, sdf_obs):
    """
    计算CBF约束在ThrustOmega空间中的形式

    约束：T * (n_x * omega_y - n_y * omega_x) >= b'
    """
    # 1. 计算梯度和安全距离
    n, h = self.compute_sdf_gradient_batch(sdf_obs)

    # 2. 计算 b' 的各项
    v = state['vel']
    v_squared = torch.sum(v * v, dim=1)
    h_dot = torch.sum(n * v, dim=1)
    denom = torch.clamp(h + self.R_obs, min=1e-6)
    centrifugal = (v_squared - h_dot**2) / denom

    b_prime = -self.m * (
        centrifugal
        + (self.alpha_1 + self.alpha_2) * h_dot
        + self.alpha_1 * self.alpha_2 * h
    )

    # 3. 构造约束矩阵
    # 约束形式：[0, -T*n_y, T*n_x, 0] @ [a, omega_x, omega_y, omega_z] >= b'
    T = self.m * self.g * (a_rl + 1.0)

    A = torch.zeros(batch_size, 1, 4)
    A[:, 0, 1] = -T * n[:, 1]
    A[:, 0, 2] = T * n[:, 0]

    return A, b_prime.unsqueeze(1)
```

### 6.3 实现QP求解

在 `quad_cbf_qp.py` 中实现 `solve_qp_differentiable_thrust_omega()`：

```python
def solve_qp_differentiable_thrust_omega(self, a_rl, omega_rl, A, b):
    """
    在ThrustOmega空间中求解QP

    min ||[a, omega] - [a_rl, omega_rl]||^2
    s.t. A @ [a, omega] >= b
         bounds
    """
    # 使用qpth求解
    # 目标函数：min ||x - x_rl||^2
    # 约束：A @ x >= b, bounds

    x_rl = torch.cat([a_rl, omega_rl], dim=1)

    x_safe = qpth.QPFunction()(
        Q = 2.0 * torch.eye(4),
        p = -2.0 * x_rl,
        G = torch.cat([-A, -I_4, I_4]),
        h = torch.cat([-b, bounds_lower, bounds_upper]),
        ...
    )

    return x_safe[:, 0], x_safe[:, 1:]  # a_safe, omega_safe
```

### 6.4 修改forward()方法

在 `quad_cbf_qp.py` 中修改 `forward()` 方法：

```python
def forward(self, state, a_rl, omega_rl, sdf_obs):
    """
    前向传播：在ThrustOmega空间中应用CBF
    """
    if self.training:
        # 计算约束
        A, b = self.compute_cbf_constraints_batch_thrust_omega(state, sdf_obs)

        # 求解QP
        a_safe, omega_safe = self.solve_qp_differentiable_thrust_omega(
            a_rl, omega_rl, A, b
        )

        return a_safe, omega_safe
    else:
        # 推理模式：使用NumPy版本
        ...
```

### 6.5 修改模型输出

在 `quad_multi_model_rcbf.py` 中修改 `forward()` 方法：

```python
def forward(self, obs):
    # ... 特征提取 ...

    # 输出ThrustOmega动作
    a_thrust = self.policy_head_a(features)  # (batch, 1)
    omega = self.policy_head_omega(features)  # (batch, 3)

    # 应用CBF
    a_safe, omega_safe = self.cbf_layer(state, a_thrust, omega, sdf_obs)

    # 转换为电机推力（用于环境执行）
    T_safe = self.J_inv @ torch.cat([a_safe, omega_safe], dim=1)

    return T_safe  # 返回电机推力给环境
```

---

## 七、关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| $\alpha_1$ | 1.0 | CBF增益1 |
| $\alpha_2$ | 1.0 | CBF增益2 |
| $R_{obs}$ | 0.5 m | 障碍物半径 |
| $\Delta$ | 0.1 m | SDF网格分辨率 |
| $m$ | 0.028 kg | 无人机质量 |
| $g$ | 9.81 m/s² | 重力加速度 |
| $\tau$ | 3.0 | 推力重量比 |

---

## 八、优势总结

### 相比RawControl + 无Risk项CBF

| 方面 | RawControl | ThrustOmega |
|------|-----------|------------|
| 侧边障碍物 | ❌ 约束失效 | ✓ 自动处理 |
| 约束形式 | 线性但无效 | ✓ 线性且有效 |
| 物理意义 | 不清晰 | ✓ 清晰 |
| 梯度流 | 部分维度为0 | ✓ 所有维度非0 |
| RL学习 | 困难 | ✓ 充分 |
| 超参数 | 需要调权重 | ✓ 无需调参 |

### 相比RawControl + Risk项CBF

| 方面 | Risk项 | ThrustOmega |
|------|--------|------------|
| 理论复杂度 | 复杂 | ✓ 简洁 |
| 参数调整 | 需要 $k_\omega$ | ✓ 无需 |
| 计算成本 | 高 | ✓ 低 |
| 可解释性 | 低 | ✓ 高 |

---

## 九、实现检查清单

- [ ] 修改RL策略输出空间为ThrustOmega
- [ ] 实现 `compute_cbf_constraints_batch_thrust_omega()`
- [ ] 实现 `solve_qp_differentiable_thrust_omega()`
- [ ] 修改 `forward()` 方法
- [ ] 修改模型输出和动作转换
- [ ] 验证梯度流（通过反向传播测试）
- [ ] 测试侧边障碍物场景
- [ ] 训练和评估

---

## 十、总结

**ThrustOmega空间中的CBF方案**是一个优雅、高效、物理直观的解决方案：

1. **直接约束Roll/Pitch角速度**，自动处理欠驱动系统的姿态耦合
2. **在归一化动作空间中优化**，所有维度同等对待
3. **梯度流清晰**，RL可以充分学习
4. **没有超参数**，无需手动调整权重
5. **标准凸QP问题**，易于求解和微分

这是从RawControl到ThrustOmega的完整转换方案，解决了原始CBF方法的所有问题。

