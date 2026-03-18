# CBF 理论推导与代码实现对齐检查

## ✅ 符号和参数对齐

| 理论符号 | 代码变量 | 位置 | 检查 |
|---------|---------|------|------|
| $m = 0.028$ kg | `self.m` | `quad_cbf_qp.py:47` | ✅ |
| $g = 9.81$ m/s² | `self.g` | `quad_cbf_qp.py:48` | ✅ |
| $T_{max} = \frac{m \cdot g \cdot \text{thrust\_to\_weight}}{4}$ | `self.T_max` | `quad_cbf_qp.py:49` | ✅ |
| $e_3 = [0, 0, 1]^\top$ | `self.e3` | `quad_cbf_qp.py:60` | ✅ |
| $\alpha_1$ | `self.alpha_1` | `quad_cbf_qp.py:52` | ✅ |
| $\alpha_2$ | `self.alpha_2` | `quad_cbf_qp.py:53` | ✅ |
| $k_\omega$ | `self.k_omega` | `quad_cbf_qp.py:54` | ✅ |
| $R_{obs}$ | `self.R_obs` | `quad_cbf_qp.py:55` | ✅ |
| $\epsilon$ | `self.epsilon` | `quad_cbf_qp.py:56` | ✅ |
| $\Delta$ | `self.delta` | `quad_cbf_qp.py:57` | ✅ |

---

## ✅ SDF 梯度计算对齐

### 理论公式
$$h = \text{SDF}_{\text{obs}}[4]$$

$$n_x = \frac{\text{SDF}[5] - \text{SDF}[3]}{2\Delta}$$

$$n_y = \frac{\text{SDF}[7] - \text{SDF}[1]}{2\Delta}$$

$$n = [n_x, n_y, 0]^\top$$

### 代码实现
```python
# quad_cbf_qp.py:87-99 (compute_sdf_gradient)
h = sdf_obs[4]  # ✅ 中心点 SDF 值

n_x = (sdf_obs[5] - sdf_obs[3]) / (2 * self.delta)  # ✅ 中心差分
n_y = (sdf_obs[7] - sdf_obs[1]) / (2 * self.delta)  # ✅ 中心差分
n = np.array([n_x, n_y, 0.0])  # ✅ z 分量为 0
```

**状态**: ✅ 完全一致

---

## ✅ CBF 约束 A 矩阵对齐

### 理论公式
$$A = \frac{T_{max}}{2m} (n^\top R e_3) \begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix}$$

### 代码实现
```python
# quad_cbf_qp.py:289-296 (compute_cbf_constraints_batch)
Re3 = torch.matmul(R, e3)  # ✅ R @ e3
nTRe3 = torch.sum(n * Re3, dim=1)  # ✅ n^T @ R @ e3

A_coeff = (self.T_max / (2 * self.m)) * nTRe3  # ✅ T_max / (2m) * nTRe3
A = A_coeff.unsqueeze(1).unsqueeze(2) * torch.ones(batch_size, 1, 4)  # ✅ [1,1,1,1]
```

**状态**: ✅ 完全一致

---

## ✅ CBF 约束 b 向量对齐

### 理论公式
$$b = \text{Risk} + n^\top g e_3 - \dot{n}^\top v - (\alpha_1 + \alpha_2)(n^\top v) - \alpha_1 \alpha_2 h - \frac{2 T_{max}}{m} (n^\top R e_3)$$

其中：
- $\text{Risk} = k_\omega \|\omega_{xy}\|^2 + \epsilon$
- $h\_dot = n^\top v$
- $\dot{n}^\top v = \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}$

### 代码实现
```python
# quad_cbf_qp.py:275-310 (compute_cbf_constraints_batch)

# 1. h_dot = n^T @ v
h_dot = torch.sum(n * v, dim=1)  # ✅

# 2. n_dot^T @ v (离心项)
v_squared = torch.sum(v * v, dim=1)  # ✅ ||v||^2
denom = torch.clamp(h + self.R_obs, min=1e-6)  # ✅ h + R_obs
n_dot_v = (v_squared - h_dot**2) / denom  # ✅

# 3. Risk 项
omega_xy_squared = torch.sum(omega[:, :2]**2, dim=1)  # ✅ ||omega_xy||^2
Risk = self.k_omega * omega_xy_squared + self.epsilon  # ✅ k_omega * ... + epsilon

# 4. gravity_term = n^T @ g @ e3
gravity_term = torch.sum(n * (self.g * e3_np), dim=1)  # ✅

# 5. bias_term = (2 * T_max / m) * nTRe3
bias_term = (2 * self.T_max / self.m) * nTRe3  # ✅

# 6. 组合 b
b = (
    Risk                                      # ✅ Risk 项
    + gravity_term                            # ✅ n^T g e3
    - n_dot_v                                 # ✅ - n_dot^T v
    - (self.alpha_1 + self.alpha_2) * h_dot  # ✅ - (α1+α2)(n^T v)
    - self.alpha_1 * self.alpha_2 * h        # ✅ - α1α2 h
    - bias_term                               # ✅ - (2T_max/m)(n^T R e3)
)
```

**状态**: ✅ 完全一致

---

## ✅ QP 优化问题对齐

### 理论公式
$$\min_{u, \delta} \|u - u_{rl}\|^2 + K \cdot \delta^2$$

$$\text{s.t. } A u \ge b - \delta$$

$$-1 \le u \le 1$$

$$\delta \ge 0$$

### 代码实现
```python
# quad_cbf_qp.py:338-363 (solve_qp_differentiable)

# P 矩阵 (二次项系数)
P[:, :n_u, :n_u] = 2.0 * torch.eye(n_u)  # ✅ ||u - u_rl||^2 的二次项
P[:, n_u:, n_u:] = 2.0 * self.penalty    # ✅ K * δ^2 的二次项

# q 向量 (一次项系数)
q[:, :n_u] = -2.0 * u_rl  # ✅ -2 * u_rl^T u

# CBF 约束：A @ u >= b - δ  →  -A @ u + δ <= -b
G_cbf[:, :, :n_u] = -A    # ✅ -A
G_cbf[:, :, n_u] = 1.0    # ✅ +δ
h_cbf = -b                # ✅ -b

# 边界约束
u >= -1  →  -u <= 1  ✅
u <= 1   →  +u <= 1  ✅
δ >= 0   →  -δ <= 0  ✅
```

**状态**: ✅ 完全一致

---

## ⚠️ 发现的不一致

### 问题 1：T_max 计算

**理论值**（文档第 43 行）:
$$T_{max} = \frac{0.028 \cdot 9.81 \cdot 3.0}{4} \approx 0.206 \, \text{N}$$

**代码计算**（`quad_cbf_qp.py:49`）:
```python
self.T_max = mass * self.g * thrust_to_weight / 4.0
```

**检查**:
```python
0.028 * 9.81 * 3.0 / 4.0 = 0.20601 ✅
```

**状态**: ✅ 一致

---

### 问题 2：离心项分母

**理论公式**（文档第 255 行）:
$$\dot{n}^\top v = \frac{\|v\|^2 - (n^\top v)^2}{h + R_{obs}}$$

**代码实现**（`quad_cbf_qp.py:281`）:
```python
denom = torch.clamp(h + self.R_obs, min=1e-6)
```

**状态**: ✅ 一致（添加了数值保护 `1e-6`）

---

### 问题 3：Risk 项的 epsilon

**理论公式**（文档第 356 行）:
$$\text{Risk}_{att} = k_\omega \|\omega_{xy}\|^2$$

**代码实现**（`quad_cbf_qp.py:286`）:
```python
Risk = self.k_omega * omega_xy_squared + self.epsilon
```

**差异**: 代码添加了 `+ self.epsilon`

**原因**: 数值稳定性，防止 Risk 为零

**建议**: 更新理论文档，明确说明 $\text{Risk} = k_\omega \|\omega_{xy}\|^2 + \epsilon$

---

## 📊 总结

| 检查项 | 状态 | 备注 |
|--------|------|------|
| 符号和参数 | ✅ | 完全一致 |
| SDF 梯度计算 | ✅ | 完全一致 |
| A 矩阵计算 | ✅ | 完全一致 |
| b 向量计算 | ✅ | 完全一致（除 epsilon） |
| QP 优化问题 | ✅ | 完全一致 |
| T_max 计算 | ✅ | 数值验证通过 |
| 离心项分母 | ✅ | 添加了数值保护 |
| Risk 项 | ⚠️ | 代码添加了 epsilon |

---

## 🔧 建议修复

### 1. 更新理论文档

在 `docs/RCBF_理论推导.md` 中明确 Risk 项：

```markdown
$$\boxed{\text{Risk}_{att} = k_\omega \|\omega_{xy}\|^2 + \epsilon}$$

其中 $\epsilon = 0.1$ 为数值稳定性常数。
```

### 2. 添加代码注释

在 `quad_cbf_qp.py:286` 添加注释：

```python
# Risk = k_omega * ||omega_xy||^2 + epsilon (数值稳定性)
Risk = self.k_omega * omega_xy_squared + self.epsilon
```

---

## ✅ 结论

**代码实现与理论推导完全对齐！**

唯一的小差异是 Risk 项添加了 `epsilon`，这是为了提高数值稳定性，建议更新理论文档以反映这一实现细节。
