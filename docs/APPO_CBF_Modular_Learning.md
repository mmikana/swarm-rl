# APPO+CBF的Modular Learning方案

## 问题回顾

SAC-RCBF论文的方法:
- SAC使用Q(s,a),动作a直接参与Q值计算
- 可以通过Q值梯度直接让策略感知CBF修正
- Modular learning: 训练时忽略未来的安全修正,避免"污染"价值函数

APPO的挑战:
- APPO使用V(s),不依赖动作
- 无法直接通过V值梯度传递CBF信息
- 需要新的方法让策略感知CBF修正

## 核心思想

虽然V(s)不依赖动作,但**advantage = Q(s,a) - V(s)**隐式地包含了动作信息!

在APPO中:
```python
advantage = Σ(γ^t * r_t) - V(s)  # GAE或n-step return
policy_loss = -log_prob(u) * advantage
```

关键洞察:**如果u_safe影响了环境交互,就会影响reward序列,进而影响advantage**

## 方案对比

### 方案1: 修正log_prob (简单但有缺陷)

```python
# 当前实现
u_rl = policy.sample()
u_safe = CBF_QP(u_rl)  # 可微分
log_prob = policy.log_prob(u_safe)  # 重新计算

policy_loss = -log_prob * advantage
```

**问题**:
- 当u_safe远离u_rl时,log_prob(u_safe)很小
- 高方差梯度,训练不稳定

### 方案2: 可微CBF + 保持原始log_prob (推荐)

```python
# 关键: 保持log_prob(u_rl),但让梯度通过u_safe回传
u_rl = policy.sample()
u_safe = CBF_QP_differentiable(u_rl)  # 可微分!

# 使用u_rl的log_prob (稳定)
log_prob = policy.log_prob(u_rl)

# 但u_safe参与环境交互,影响advantage
# 梯度路径: policy -> u_rl -> u_safe -> env -> reward -> advantage -> policy_loss
policy_loss = -log_prob * advantage
```

**优点**:
- log_prob稳定(基于u_rl)
- 梯度通过advantage隐式传递CBF信息
- 策略会学习"CBF喜欢的动作"

**关键**: 需要在**训练模式**下使用可微分QP!

### 方案3: Modular Learning (最优)

借鉴SAC-RCBF的思想,在APPO中实现模块化学习:

```python
# 训练时: 使用可微CBF,但在计算advantage时"忽略"未来的CBF修正
u_rl_t = policy.sample()
u_safe_t = CBF_QP_differentiable(u_rl_t)

# 执行u_safe_t,获得r_t, s_{t+1}
r_t, s_next = env.step(u_safe_t)

# 计算advantage时,假设未来没有CBF修正
# 这样V(s)学习的是"无障碍物"的价值函数
advantage = r_t + γ * V(s_next) - V(s_t)

# 但当前时刻的u_safe_t仍然影响r_t
# 所以策略会学习安全行为,但V不会被"污染"
policy_loss = -log_prob(u_rl_t) * advantage
```

**实现细节**:
- 在rollout时使用CBF保证安全
- 在计算advantage时,V(s)不感知CBF
- 策略通过当前reward学习安全行为

## 当前实现分析

让我检查当前代码:

```python
# swarm_rl/models/quad_multi_model_rcbf.py:401-423
def forward(self, state, u_rl, sdf_obs):
    if self.training:
        # 训练模式：使用可微分 QP
        A, b = self.compute_cbf_constraints_batch(state, sdf_obs)
        u_safe = self.solve_qp_differentiable(u_rl, A, b)
        return u_safe
    else:
        # 推理模式：使用 cvxpy
        ...
```

**好消息**: 训练时已经使用可微分QP!

**问题**: 在`quad_multi_model_rcbf.py`中,log_prob是基于u_rl计算的:

```python
# 第168行
log_prob_actions = self.last_action_distribution.log_prob(actions)  # actions = u_rl

# 第184行
u_final = self.cbf_layer(state, u_rl, sdf_obs)  # u_final = u_safe

# 第200行
result["log_prob_actions"] = log_prob_actions  # 使用u_rl的log_prob
```

这实际上是**方案2**的实现!

## 验证当前实现是否正确

当前实现应该已经能工作,因为:

1. ✓ 训练时使用可微分QP
2. ✓ log_prob基于u_rl (稳定)
3. ✓ u_safe参与环境交互,影响reward
4. ✓ 梯度可以通过advantage回传

**但需要确认**: Sample Factory是否保留了计算图?

### 检查点1: u_safe是否保留梯度?

```python
# 在forward_tail中
u_final = self.cbf_layer(state, u_rl, sdf_obs)
result["actions"] = u_final

# 问题: u_final是否保留了梯度?
# 如果Sample Factory在rollout后detach了actions,梯度就断了
```

### 检查点2: advantage计算是否包含u_safe的影响?

```python
# Sample Factory的advantage计算
# 如果使用GAE,需要确保reward序列包含了u_safe的影响
advantage = compute_gae(rewards, values, ...)
```

## 推荐实现

### 短期: 验证当前实现

创建一个测试脚本,检查梯度是否正确传播:

```python
# test_cbf_gradient.py
import torch
from swarm_rl.models.quad_multi_model_rcbf import QuadActorCriticWithCBF

# 创建模型
model = QuadActorCriticWithCBF(...)
model.train()  # 训练模式

# 前向传播
obs = torch.randn(batch_size, obs_dim, requires_grad=True)
result = model(obs, rnn_states)

u_safe = result["actions"]
log_prob = result["log_prob_actions"]

# 模拟advantage
advantage = torch.randn(batch_size)

# 计算loss
policy_loss = -(log_prob * advantage).mean()

# 反向传播
policy_loss.backward()

# 检查梯度
print("u_safe.grad:", u_safe.grad)  # 应该不为None
print("obs.grad:", obs.grad)  # 应该不为None
```

### 长期: 实现Modular Learning

如果需要更好的性能,可以实现完整的modular learning:

1. 修改Sample Factory的rollout逻辑
2. 在计算advantage时,使用"无CBF"的V值
3. 但在执行时仍然使用CBF

## 总结

**当前实现应该已经接近正确**,因为:
- ✓ 使用可微分QP (训练时)
- ✓ log_prob基于u_rl (稳定)
- ✓ u_safe影响环境交互

**需要验证**:
- Sample Factory是否保留了actions的梯度?
- 如果没有,需要修改Sample Factory的rollout逻辑

**下一步**:
1. 运行梯度测试脚本
2. 如果梯度正确传播,当前实现就是对的
3. 如果梯度断了,需要修改Sample Factory或使用方案1

你想我先创建梯度测试脚本吗?
