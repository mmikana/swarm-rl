# ThrustOmega空间的可行性分析（仅XY避障版本）

## 一、简化后的问题设定

### 1.1 关键简化

根据 `obstacles/utils.py` 的代码：
- **所有障碍物都是圆柱形**，贯穿整个Z轴空间
- SDF计算**只在XY平面上**进行（第12-15行）
- 距离 $dist = \sqrt{(x-x_0)^2 + (y-y_0)^2}$，**与Z无关**
- 因此梯度 $\nabla h = [n_x, n_y, 0]^\top$ 永远成立

### 1.2 重要推论

$$n = [n_x, n_y, 0]^\top, \quad n_z = 0 \text{ 总是成立}$$

这意味着：
1. **永远不需要考虑竖直避障**（天花板/地板）
2. **只需考虑XY平面的避障**
3. **竖直方向的加速度完全自由**，不受CBF约束

---

## 二、简化后的CBF约束

### 2.1 约束矩阵 A 的简化

原来的形式：
$$A = g \begin{bmatrix} n^\top Re_3 & (n \times (Re_3))^\top \end{bmatrix}$$

由于 $n_z = 0$ 总是成立，考虑各项：

#### 第1项：线加速度约束 $A_1 = g \cdot n^\top Re_3$

$$n^\top Re_3 = [n_x, n_y, 0] \cdot \begin{bmatrix} (Re_3)_x \\ (Re_3)_y \\ (Re_3)_z \end{bmatrix} = n_x(Re_3)_x + n_y(Re_3)_y$$

这是 $n$ 在XY平面上的分量与推力方向在XY平面上的投影的点积。

#### 第2项：角速度约束 $A_{2:4} = g \cdot (n \times (Re_3))^\top$

$$n \times (Re_3) = \begin{bmatrix} n_x \\ n_y \\ 0 \end{bmatrix} \times \begin{bmatrix} (Re_3)_x \\ (Re_3)_y \\ (Re_3)_z \end{bmatrix} = \begin{bmatrix} n_y(Re_3)_z - 0 \\ 0 - n_x(Re_3)_z \\ n_x(Re_3)_y - n_y(Re_3)_x \end{bmatrix}$$

$$= \begin{bmatrix} n_y(Re_3)_z \\ -n_x(Re_3)_z \\ n_x(Re_3)_y - n_y(Re_3)_x \end{bmatrix}$$

**关键观察**：
- 第1、2项（对应 $\omega_x, \omega_y$）：与 $(Re_3)_z$ 相关，**水平飞行时为0**（因为 $(Re_3)_z \approx 1$ 且其他项很小）
- 第3项（对应 $\omega_z$）：与 $n_x(Re_3)_y - n_y(Re_3)_x$ 有关，**Yaw角速度**

所以约束简化为：
$$A_{2:4} \approx g \cdot [0, 0, n_x(Re_3)_y - n_y(Re_3)_x]$$

**这意味着Yaw角速度约束**（$\omega_z$）才是关键！

### 2.2 更精确的分析：水平飞行假设

当无人机**水平飞行**（最常见的情况），$R$ 接近：
$$R \approx \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

其中 $\psi$ 是Yaw角。

则 $Re_3 = [0, 0, 1]^\top$

$$A_1 = g \cdot [n_x, n_y, 0] \cdot [0, 0, 1] = 0$$  ❌（线加速度约束失效）$$

$$n \times (Re_3) = \begin{bmatrix} n_y \\ -n_x \\ 0 \end{bmatrix}$$

$$A_{2:4} = g \cdot [n_y, -n_x, 0]^\top$$

**约束变为**：
$$g \cdot (n_y \omega_x - n_x \omega_y) \ge b$$

或者等价地：
$$(n_y, -n_x) \cdot (\omega_x, \omega_y) \ge \frac{b}{g}$$

**物理意义**：约束 **Roll和Pitch角速度的加权组合**，权重为 $(n_y, -n_x)$。

---

## 三、具体例子：水平飞行，侧边障碍物

### 3.1 场景设置

- 无人机位置：$(0, 0, 1)$
- 水平飞行，Yaw角 $\psi = 0$（指向X正方向）
- 左侧（Y负方向）有障碍物
- 梯度：$n = [0, -1, 0]^\top$（指向Y正方向，远离障碍物）
- 推力方向：$Re_3 = [0, 0, 1]^\top$（竖直向上）

### 3.2 约束计算

$$A_1 = g \cdot [0, -1, 0] \cdot [0, 0, 1] = 0$$

$$n \times (Re_3) = [0, -1, 0] \times [0, 0, 1] = [-(-1) \cdot 1, 0 - 0, 0] = [1, 0, 0]$$

$$A_{2:4} = g \cdot [1, 0, 0]$$

**约束变为**：
$$g \cdot \omega_x \ge b$$

**即：约束右侧的Roll角速度 $\omega_x > 0$**

### 3.3 物理含义

当左侧有障碍物时：
- 无人机需要**向右倾斜**（正的Roll）
- Roll倾斜会使推力方向从竖直向右倾斜
- 这样就产生了向右（Y正方向）的加速度
- **约束自动强制正确的Roll反应** ✓

---

## 四、一般XY避障的可行性分析

### 4.1 任意方向的XY障碍物

设梯度 $n = [\cos\theta, \sin\theta, 0]^\top$（XY平面上的单位向量）

水平飞行假设下：
$$A_{2:4} = g \cdot [\sin\theta, -\cos\theta, 0]$$

约束：
$$g(\sin\theta \cdot \omega_x - \cos\theta \cdot \omega_y) \ge b$$

或写成：
$$\omega_\perp = (\sin\theta, -\cos\theta) \cdot (\omega_x, \omega_y) \ge \frac{b}{g}$$

这是**沿垂直于梯度方向的Roll/Pitch角速度**的约束。

### 4.2 最坏情况分析

最坏情况（最紧的约束）发生在：
- 无人机正向障碍物移动：$n^\top v < 0$，速度很大
- 距离很小：$h \to 0^+$
- 没有转动：$\boldsymbol{\omega} = 0$

此时 $b$ 值很大，需要 $|\omega_\perp|$ 很大才能满足约束。

**关键数值**：
$$\omega_\perp,required = \frac{b}{g}$$

在最坏情况下，$b$ 可能达到 $(\alpha_1 + \alpha_2)|v_{max}| \approx 2.0 \times 3.0 = 6.0$ m/s²

所以：
$$\omega_\perp,required \approx \frac{6.0}{9.81} \approx 0.61 \text{ rad/s}$$

**可用的Roll/Pitch角速度**：
$$\omega_{x,max} = \omega_{y,max} = 31.42 \text{ rad/s}$$

**裕度**：$31.42 / 0.61 \approx 52$ 倍 ✓（仍然充分）

### 4.3 重要的可行性检查

在水平飞行时，只要能约束Roll和Pitch角速度，就能产生任意方向的水平加速度。

而 $\omega_{x,max}, \omega_{y,max} = 31.42$ rad/s 远大于所需，所以**XY避障的可行性是充分的**。

---

## 五、Yaw角速度的角色

有趣的是，在约束中，**Yaw角速度** $\omega_z$ **不出现**（$A_4 = 0$）！

这是因为：
- Yaw旋转改变身体的方向，但**不改变推力矢量的竖直方向**
- 水平飞行时，推力永远竖直向上，无论怎么旋转
- 因此Yaw对XY避障没有直接贡献

**但Yaw不是完全自由的**：
- Yaw会改变Roll/Pitch的作用方向
- 如果Yaw角速度太大，可能难以精确控制Roll/Pitch

---

## 六、 与RawControl的对比

### RawControl空间的问题

在RawControl中，约束是关于电机推力 $u_i \in [-1, 1]$ 的：
$$\sum T_i \cdot (\text{某个系数}) \ge b$$

只能约束总推力，**无法单独约束任何方向的加速度**。

### ThrustOmega空间的优势

在ThrustOmega中，约束是：
$$(\sin\theta \cdot \omega_x - \cos\theta \cdot \omega_y) \ge \frac{b}{g}$$

**能直接约束特定方向的加速度**（通过Roll/Pitch角速度）。

而且约束涉及的量（Roll和Pitch角速度）有**充足的可用范围**：
- 所需：~0.6 rad/s（最坏）
- 可用：31.4 rad/s
- **裕度：>50倍**

---

## 七、最终可行性结论

### ✅ 充分可行的地方

1. **XY避障约束本质上有效**
   - 梯度总是在XY平面（$n_z = 0$）
   - Roll/Pitch角速度能产生任意方向的水平加速度
   - 可用角速度（31.4 rad/s）远大于所需（~0.6 rad/s）

2. **约束保持线性**
   - 整个约束仍是关于 $(a_{thrust}, \omega_x, \omega_y, \omega_z)$ 的线性形式
   - QP问题易解

3. **物理意义清晰**
   - 约束自动产生正确的避障反应（倾斜向安全方向）
   - 无需特殊逻辑或Risk补偿

4. **梯度流通畅**
   - 通过J^{-1}反演，梯度能清晰反向传播

### ⚠️ 需要注意的细节

1. **竖直飞行边界情况**
   - 如果无人机不是水平飞行（例如垂直上升），$Re_3$ 不再是 $[0,0,1]$
   - 但在实际应用中，无人机通常水平飞行
   - 可以添加检查：当 $(Re_3)_z < 0.8$ 时警告

2. **数值保护**
   - 当 $h \to 0$ 时，需要 `h_safe = max(h, 1e-3)`
   - 离心项的分母可能变小

3. **控制延迟**
   - Controller需要快速响应（调整 $k_p$）
   - 但31.4 rad/s的可用范围给了充足的缓冲

4. **多障碍物**
   - 当前只约束最近的一个
   - 理论上可扩展到多个约束行

---

## 八、实现建议

### 立即可以做的

```python
# 1. 在 compute_cbf_constraints_batch 中，简化约束计算
# 由于 n_z = 0 总是成立，可以简化：

def compute_cbf_constraints_batch_xy(self, state, sdf_obs):
    # 计算 n_x, n_y（只在XY平面）
    n_xy = self.compute_sdf_gradient_batch(sdf_obs)  # 已经 n_z=0
    h = sdf_obs[:, 4]

    # 简化的A矩阵（水平飞行假设）
    # 当 Re3 ≈ [0, 0, 1] 时：
    # A_1 = 0（线加速度约束总是失效）
    # A_2:4 = g * [n_y, -n_x, 0]

    Re3 = torch.matmul(R, torch.tensor([0, 0, 1]))  # 推力方向

    # 只计算Roll/Pitch的约束
    A_omega = g * torch.stack([n[:, 1], -n[:, 0], torch.zeros_like(n[:, 0])], dim=1)
    A = torch.concatenate([torch.zeros(batch_size, 1), A_omega], dim=1)

    # b向量保持不变
    b = ...

    return A, b
```

### 验证要点

1. ✓ 验证梯度总是 $n_z = 0$
2. ✓ 验证约束中 $A_1$ 接近0（水平飞行）
3. ✓ 验证Roll/Pitch约束能产生正确的加速度方向
4. ✓ 验证角速度范围充足

---

## 九、总结表

| 指标 | 值 | 评价 |
|------|-----|--------|
| 约束形式 | $(\sin\theta \omega_x - \cos\theta \omega_y) \ge b/g$ | 线性 ✓ |
| 所需角速度（最坏） | ~0.61 rad/s | 非常小 |
| 可用Roll/Pitch角速度 | 31.42 rad/s | 非常大 |
| **裕度倍数** | **>50倍** | **充分** |
| Z轴避障需求 | 无 | 圆柱形障碍物 ✓ |
| 水平飞行假设 | 合理 | 通常成立 ✓ |
| 梯度特性 | $n_z = 0$ 总成立 | 简化约束 ✓ |
| 物理直观性 | 强 | 自动正确反应 ✓ |

**结论**：基于圆柱形障碍物和XY平面避障的假设，**CBF约束条件是充分的，可以直接实现**。

