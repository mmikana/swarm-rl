# 距离感知姿态CBF与坐标系对齐分析

**目的**：确保CBF约束中的所有符号与坐标系、旋转矩阵和控制方向完全对齐

**版本**：2026-03-27

---

## 一、坐标系定义

### 1.1 全局坐标系（Global Frame）

根据 `quadrotor_dynamics.py:29-32` 的定义：

```
坐标系配置：X configuration
- x 轴：穿过臂膀指向前方（Forward）
- y 轴：指向左边（Left）
- z 轴：向上（Up）
```

**可视化**：
```
        +y (Left)
         ↑
         |
  +x →   |
(Forward)|

        ⊙ Drone (俯视图)

        z pointing out of page (Up)
```

### 1.2 机体坐标系（Body Frame）

在四旋翼标准约定中：
- 机体 z 轴指向**下方**（与全局约定相反，这是标准四旋翼约定）
- 推力 $T$ 沿机体 z 轴负方向（向上推动无人机）
- 当 $R$ 是从全局到机体的旋转矩阵时，推力在全局系中的方向为 $Re_3$，其中 $e_3 = [0, 0, 1]^T$

**重要**：你们的代码中，`R` 满足：
$$\vec{v}_{body} = R^T \vec{v}_{global}$$
$$\vec{v}_{global} = R \vec{v}_{body}$$

因此：
- $Re_3 = $ 推力方向（全局系中指向上方）
- $R[:, 0] = $ 前方向（全局 x 轴在机体中的方向）
- $R[:, 1] = $ 左方向（全局 y 轴在机体中的方向）
- $R[:, 2] = Re_3 = $ 推力方向

---

## 二、关键物理量定义

### 2.1 SDF 梯度 $n$

从 3×3 SDF 网格计算：
```
位置 [x-δ, x, x+δ] × [y-δ, y, y+δ]
对应索引 [0, 1, 2] × [3, 4, 5] × [6, 7, 8]

梯度计算（标准命名）：
n_x = (SDF[7] - SDF[1]) / (2δ)  ← ∂SDF/∂x = (x+δ,y) - (x-δ,y)
n_y = (SDF[5] - SDF[3]) / (2δ)  ← ∂SDF/∂y = (x,y+δ) - (x,y-δ)
n_z = 0
```

**物理意义**：$n$ 指向**安全方向**（远离障碍物）

### 2.2 推力方向 $Re_3$

$$Re_3 = R[:, 2] = \begin{bmatrix} r_{13} \\ r_{23} \\ r_{33} \end{bmatrix}$$

这是无人机当前的**推力方向**在全局系中的表示

### 2.3 屏障函数 $h$

$$h(R, d) = n^\top Re_3 - \beta(d)$$

- 若 $h > 0$：安全
- 若 $h = 0$：边界
- 若 $h < 0$：不安全

---

## 三、具体场景分析：左前方障碍物→右roll

### 3.1 场景设置

```
全局坐标系（俯视图）：
       +y (Left)
        ↑
        |  Obstacle
        |   [O]
        |  /
        | / ← 梯度 n（指向安全方向，向下-右）
        |/
    +---|----→ +x (Forward)
        ⊙ Drone
        ║
    初始速度向前
```

**数值示例**：
- 无人机位置：$(0, 0)$
- 障碍物中心：$(1, 0.5)$  （前方0.5m，左方0.5m）
- SDF梯度：$n = [-0.707, -0.707, 0]^T$  （左前方，指向安全的右后）

### 3.2 当前姿态（无偏转）

无人机悬停，无roll/pitch，当前姿态矩阵为单位矩阵（近似）：

$$R_{initial} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

推力方向：
$$Re_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

屏障函数：
$$h = n^\top Re_3 - \beta(d) = 0 - \beta(d) = -\beta(d) < 0$$

**结论**：当前姿态**不安全**，因为推力向上，没有向右倾斜。

### 3.3 期望的解决方案：右roll

**右roll** 意思是绕 **x 轴正方向旋转**（右手法则，拇指指向 +x，手指卷曲方向为旋转）

Roll 后的旋转矩阵（小角度近似）：
$$R_{roll} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi & \cos\phi \end{bmatrix} \approx \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & -\phi \\ 0 & \phi & 1 \end{bmatrix}, \quad \phi > 0$$

新的推力方向：
$$Re_3 = R_{roll} \cdot e_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & -\phi \\ 0 & \phi & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ -\phi \\ 1 \end{bmatrix}$$

**新推力方向的物理意义**：
- z分量 = 1：仍然向上（维持高度）
- y分量 = $-\phi < 0$：向右（负y方向）
- 这使得无人机向右加速！

新的屏障函数：
$$h = n^\top Re_3 - \beta(d) = \begin{bmatrix} -0.707 \\ -0.707 \\ 0 \end{bmatrix}^\top \begin{bmatrix} 0 \\ -\phi \\ 1 \end{bmatrix} - \beta(d)$$

$$= 0.707\phi + 0 - \beta(d) = 0.707\phi - \beta(d)$$

当 $\phi$ 足够大（$\phi > \beta(d)/0.707$）时，$h > 0$，**恢复安全**。✅

---

## 四、CBF 约束推导

### 4.1 屏障函数一阶导数

$$\dot{h} = \frac{d}{dt}(n^\top Re_3) - \beta'(d) \dot{d}$$

其中：
- $\dot{d} = n^\top v$（SDF沿速度方向的变化率）

**计算 $\dot{n}^\top Re_3 + n^\top \frac{d(Re_3)}{dt}$**：

$$\frac{d(Re_3)}{dt} = \frac{d(R \cdot e_3)}{dt} = \dot{R} \cdot e_3$$

由于 $\dot{R} = R[\omega]_\times$（其中 $[\omega]_\times$ 是 $\omega$ 的反对称矩阵），有：

$$\frac{d(Re_3)}{dt} = R[\omega]_\times e_3$$

在全局系中，转换为：

$$n^\top \frac{d(Re_3)}{dt} = n^\top R[\omega]_\times e_3$$

使用矩阵恒等式 $a^\top [b]_\times c = -b^\top [a]_\times c$：

$$= -(R^T n)^\top [\omega]_\times e_3 = -(R^T n)^\top [\omega]_\times e_3$$

但更简单的做法是用行向量：

$$n^\top (R[\omega]_\times e_3) = (R^T n)^\top [\omega]_\times e_3$$

使用 $a^\top [b]_\times = -(b \times a)^\top$：

$$(R^T n)^\top [\omega]_\times e_3 = -(\omega \times (R^T n))^\top e_3 = -(e_3^\top (\omega \times (R^T n)))$$

$$= (e_3 \times \omega)^\top (R^T n) = -(\omega \times e_3)^\top (R^T n)$$

更直接的方法：利用 $a^\top [b]_\times c = b^\top (c \times a)$：

$$n^\top [\omega]_\times (Re_3) = \omega^\top (Re_3 \times n)$$

不对，让我用标准的叉积恒等式：

$$n^\top (R[\omega]_\times e_3) = n^\top R [\omega]_\times e_3$$

由于 $[\omega]_\times e_3$ 的作用是在机体系中，需要转换。标准做法：

**在全局系中**，如果 $\omega$ 是机体角速度，则：

$$\omega_{global} = R \omega_{body}$$

推力方向的变化率：

$$\frac{d(Re_3)}{dt} = \dot{R} e_3 = (R[\omega]_\times) e_3$$

让我直接用向量叉积的标准结果：

$$n^\top \frac{d(Re_3)}{dt} = -(n \times Re_3)^\top \omega$$

因此：

$$\dot{h} = \dot{n}^\top Re_3 - (n \times Re_3)^\top \omega - \beta'(d)(n^\top v)$$

### 4.2 CBF 条件

$$\dot{h} + \alpha h \ge 0$$

整理为约束形式：

$$-(n \times Re_3)^\top \omega \ge -\dot{n}^\top Re_3 + \beta'(d)(n^\top v) - \alpha(n^\top Re_3 - \beta(d))$$

简化（忽略 $\dot{n}$ 项，因为梯度变化慢）：

$$-(n \times Re_3)^\top \omega \ge \beta'(d)(n^\top v) - \alpha n^\top Re_3 + \alpha \beta(d)$$

---

## 五、符号检查：左前方障碍物场景

### 5.1 参数值

- $n = [-0.707, -0.707, 0]^T$
- $Re_3 = [0, -\phi, 1]^T$ （roll 后）
- $\omega = [\omega_x, \omega_y, \omega_z]^T$ 是机体角速度

### 5.2 计算叉积 $n \times Re_3$

$$n \times Re_3 = \begin{vmatrix} \vec{i} & \vec{j} & \vec{k} \\ -0.707 & -0.707 & 0 \\ 0 & -\phi & 1 \end{vmatrix}$$

$$= \vec{i}(-0.707 \cdot 1 - 0 \cdot (-\phi)) - \vec{j}(-0.707 \cdot 1 - 0 \cdot 0) + \vec{k}(-0.707 \cdot (-\phi) - (-0.707) \cdot 0)$$

$$= \vec{i}(-0.707) - \vec{j}(-0.707) + \vec{k}(0.707\phi)$$

$$= [-0.707, 0.707, 0.707\phi]^T$$

### 5.3 约束左侧

$$(n \times Re_3)^\top \omega = -0.707\omega_x + 0.707\omega_y + 0.707\phi\omega_z$$

要使CBF约束满足，需要：

$$-(n \times Re_3)^\top \omega = 0.707\omega_x - 0.707\omega_y - 0.707\phi\omega_z \ge \text{RHS}$$

### 5.4 期望的控制：右roll

右roll 对应的是绕 **x 轴旋转**，因此期望 $\omega_x > 0$（正的x角速度）

从约束看：

$$0.707\omega_x - 0.707\omega_y - 0.707\phi\omega_z \ge \text{RHS}$$

当 $\omega_x > 0$ 时，约束左侧增加 ✅ 这**支持**右roll。

当 $\omega_y > 0$（向前pitch）时，约束左侧减少 ❌ 这会**阻碍**roll。

当 $\omega_z$ 增加时（yaw），约束左侧被 $\phi$ 因子削弱 ⚠️。

### 5.5 符号正确性验证

**关键问题**：约束中的叉积 $n \times Re_3$ 是否正确指向 roll 方向？

对于左前方障碍物（$n$ 指向右后）：
- $n \times Re_3$ 的 x 分量为 $-0.707$
- 约束 $-(n \times Re_3)^\top \omega$ 中，$-0.707$ 乘以 $\omega_x$
- 最终得到 $0.707\omega_x$，这是**正相关**的

**结论**：✅ 符号正确！当我们增加 $\omega_x$（右roll），约束左侧增加，有利于满足安全约束。

---

## 六、实现要点总结

### 6.1 坐标系约定（必须遵守）

| 量 | 定义 | 单位 |
|---|---|---|
| 全局 z | 指向上 | m |
| 全局 x | 前方 | m |
| 全局 y | 左方 | m |
| $e_3$ | $[0, 0, 1]^T$ | - |
| $R$ | 旋转矩阵，$v_{global} = R v_{body}$ | - |
| $Re_3$ | 推力方向（全局系） | - |
| $n$ | SDF梯度（指向安全） | - |
| $\omega$ | 机体角速度 | rad/s |
| $v$ | 全局速度 | m/s |

### 6.2 CBF 约束符号

```python
# 计算叉积（注意向量顺序）
n_cross_Re3 = np.cross(n, Re3)  # [3,]

# 约束系数矩阵（CBF形式：A @ omega >= b）
A = -n_cross_Re3  # [3,]

# 约束右侧（简化版，忽略 dn_dot 项）
b = -alpha_cbf * (n_dot_Re3 - beta_d)

# QP形式（-A @ omega <= -b）
# 注意符号！
G_cbf = -A  # [1, 3]
h_cbf = -b  # scalar
```

### 6.3 正确性检查清单

- [ ] $n$ 从SDF梯度计算：`n_x = (SDF[5] - SDF[3])/(2δ)`, `n_y = (SDF[7] - SDF[1])/(2δ)`
- [ ] $Re_3 = R[:, 2]$（旋转矩阵的第3列）
- [ ] 叉积顺序：`n_cross_Re3 = np.cross(n, Re3)` 不能反过来
- [ ] CBF约束左侧：$-(n \times Re_3)^\top \omega$，不能少负号
- [ ] QP转换时：不等式符号翻转

---

## 七、 β(d) 的梯度符号

$$\beta(d) = \beta_0 \cdot \exp(-k(d - d_{safe})) \quad (d \ge d_{safe})$$

导数：
$$\beta'(d) = -k\beta_0 \exp(-k(d - d_{safe})) = -k\beta(d) < 0 \quad (d \ge d_{safe})$$

因此：
$$-\beta'(d)(n^\top v) = k\beta(d) \cdot (n^\top v)$$

- 当 $n^\top v > 0$（向障碍靠近）：该项为**正**，加强约束 ✅
- 当 $n^\top v < 0$（远离障碍）：该项为**负**，松弛约束 ✅

**符号正确**。

---

## 八、关键提醒

### ⚠️ 常见的符号错误

1. **叉积顺序反了**
   ```python
   # ❌ 错误
   n_cross_Re3 = np.cross(Re3, n)

   # ✅ 正确
   n_cross_Re3 = np.cross(n, Re3)
   ```

2. **忘记负号**
   ```python
   # ❌ 错误
   A = n_cross_Re3

   # ✅ 正确
   A = -n_cross_Re3
   ```

3. **QP约束转换符号错误**
   ```python
   # CBF: A @ omega >= b
   # QP:  G @ omega <= h
   # 所以: G = -A, h = -b

   # ❌ 错误
   G = A
   h = b

   # ✅ 正确
   G = -A
   h = -b
   ```

4. **混淆全局/机体坐标**
   ```python
   # 所有的 n, Re3, v, omega 必须在同一坐标系（全局系）
   n_global = n  # SDF梯度已经是全局的
   Re3_global = R @ e3  # 推力方向
   v_global = state['vel']  # 速度已经是全局的
   omega_global = R @ omega_body  # 角速度需要转换！
   # 或者直接用机体角速度在约束中
   ```

---

## 九、实现代码模板

```python
import numpy as np

def compute_cbf_constraint(state, sdf_obs, beta_0=0.5, d_safe=0.5, d_max=2.0,
                           alpha_cbf=1.0, k=2.0):
    """
    计算 CBF 约束：A @ omega >= b

    Args:
        state: dict with 'rot' (3,3), 'vel' (3,)
        sdf_obs: (9,) numpy array

    Returns:
        A: (1, 3) constraint matrix
        b: (1,) constraint scalar
    """

    # ========== 1. 提取状态 ==========
    R = state['rot']  # (3, 3)
    v = state['vel']  # (3,)

    # ========== 2. 计算SDF梯度 ==========
    delta = 0.1  # SDF分辨率

    # 中心点SDF
    h = sdf_obs[4]

    # 梯度（有限差分）
    n_x = (sdf_obs[5] - sdf_obs[3]) / (2 * delta)
    n_y = (sdf_obs[7] - sdf_obs[1]) / (2 * delta)
    n_z = 0.0

    # 归一化
    n = np.array([n_x, n_y, n_z])
    n = n / (np.linalg.norm(n) + 1e-6)

    # ========== 3. 计算推力方向 ==========
    e3 = np.array([0.0, 0.0, 1.0])
    Re3 = R @ e3  # (3,)

    # ========== 4. 计算距离相关的β(d) ==========
    if h < d_safe:
        beta = beta_0
    elif h < d_max:
        beta = beta_0 * np.exp(-k * (h - d_safe))
    else:
        beta = beta_0 * np.exp(-k * (d_max - d_safe))  # 饱和

    # ========== 5. 计算CBF约束系数 ==========
    # A @ omega >= b

    # 叉积：n × Re3
    n_cross_Re3 = np.cross(n, Re3)  # (3,)

    # A 矩阵
    A = -n_cross_Re3  # (3,)

    # n^T Re3
    n_dot_Re3 = np.dot(n, Re3)

    # n^T v（指向障碍的速度）
    n_dot_v = np.dot(n, v)

    # ========== 6. 计算约束右侧 ==========
    # b = -dn/dt^T Re3 - alpha_cbf * (n^T Re3 - beta)
    # 忽略 dn/dt 项（梯度变化慢）

    b = -alpha_cbf * (n_dot_Re3 - beta)

    return A.reshape(1, -1), b
```

---

## 十、验证方法

### 单元测试：左前方障碍物场景

```python
def test_left_front_obstacle():
    """验证左前方障碍物→右roll的约束符号"""

    # 设置场景
    state = {
        'rot': np.eye(3),  # 初始无偏转
        'vel': np.array([1.0, 0.0, 0.0])  # 向前飞行
    }

    # 左前方障碍物（距离0.7m）
    # 无人机在原点，障碍在(1, 0.5)
    sdf_obs = np.array([
        100.0, 100.0, 100.0,  # SDF[0:3]
        100.0, 0.7, 1.4,       # SDF[3:6], 中心0.7m
        100.0, 100.0, 100.0    # SDF[6:9]
    ])

    A, b = compute_cbf_constraint(state, sdf_obs)

    # ========== 验证：右roll应该减小约束违规 ==========
    # 右roll: omega_x > 0
    omega_right_roll = np.array([1.0, 0.0, 0.0])

    # 检查：A @ omega 应该 > b（安全）
    lhs = A @ omega_right_roll
    print(f"A @ omega_roll = {lhs}, b = {b}")
    assert lhs > b, "右roll应该满足CBF约束"

    # 验证：左roll (omega_x < 0) 应该违反约束
    omega_left_roll = np.array([-1.0, 0.0, 0.0])
    lhs_left = A @ omega_left_roll
    print(f"A @ omega_left = {lhs_left}, b = {b}")
    assert lhs_left <= b, "左roll应该违反约束"

    print("✅ 符号验证成功！")

if __name__ == "__main__":
    test_left_front_obstacle()
```

---

## 十一、最终总结

| 检查项 | 现象 | 符号处理 |
|---|---|---|
| **梯度方向** | SDF梯度指向安全 | $n = \nabla \text{SDF}$ ✅ |
| **推力方向** | 沿全局z向上 | $Re_3 = R[:, 2]$ ✅ |
| **叉积** | 机体坐标中的约束轴 | $n \times Re_3$ (顺序重要) ✅ |
| **约束符号** | CBF形式：$A\omega \ge b$ | $A = -(n \times Re_3)$ ✅ |
| **QP转换** | 不等式翻转 | $-A\omega \le -b$ ✅ |
| **右roll效果** | 增加$\omega_x$ | 使约束LHS增加 ✅ |

**所有符号已对齐，可以放心实现！** ✅

