# APPO-CBF 模式 B 伪代码（Sample Factory 自定义模型版）

## 核心思想

**不修改 Sample Factory 源代码**，通过以下方式集成 APPO-CBF：

1. **自定义 Actor-Critic 网络**：继承 Sample Factory 的 `ActorCriticSharedWeights` 类
2. **利用 Sample Factory 标准 APPO 训练流程**：V-trace、GAE、Rollout Buffer 等保持不变
3. **在 `forward_tail()` 中集成 CBF-QP 层**：梯度自动回传
4. **通过变量区分 CBF 模式**：`u_final`、`u_rl`、`u_safe`

---

## 符号说明

| 符号 | 含义 | 代码变量 | 说明 |
|------|------|---------|------|
| $u_t^{RL}$ | RL 标称动作 | `u_rl` | Policy 网络直接输出的动作 |
| $u_t^{safe}$ | CBF 修正后的安全动作 | `u_safe` | 经过 CBF-QP 层修正的动作 |
| $u_t^{final}$ | 最终执行动作 | `u_final` | **实际输出给环境的动作** |
| $\pi_\phi$ | 策略网络 | `action_parameterization` | - |
| $V_{\theta}$ | Critic 网络 | `critic_linear` | 输出状态价值 V(s) |
| $\log \pi$ | 对数概率 | `log_probs` | 基于 $u_{RL}$ 计算 |

---

## Pseudocode

### APPO-CBF 训练流程

```
算法：APPO-CBF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入:
    - obs: 环境观测（包含状态和 SDF 信息）
    - cfg: 配置参数（包括 CBF 参数 α₁, α₂, k_ω 等）
    - π_φ: 策略网络参数
    - V_θ: Critic 网络参数

输出:
    - actions: u_final (环境执行的动作)
    - values: V(s) (Critic 评估值)
    - log_probs: log π(u_rl | s) (用于 PPO ratio 和 V-trace)
    - old_log_probs: log μ(u_rl | s) (用于 V-trace 重要性采样)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【异步 Rollout Workers】（多个 worker 并行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

每个 Worker 执行：
   ┌─────────────────────────────────────────────────────────────┐
   │ for step in range(num_steps):                               │
   │     with torch.no_grad():                                   │
   │         # 1. 前向传播（使用当前策略 π_φ）                     │
   │         obs → Encoder → encoder_out                         │
   │         encoder_out → policy_head → u_rl_mean               │
   │         u_rl_mean → action_distribution → u_rl (采样)       │
   │         log_probs = log π(u_rl | s)                         │
   │         u_rl, obs → QuadCBFQPLayer → u_safe                 │
   │         u_final = u_safe                                    │
   │         encoder_out → critic_head → V(s)                    │
   │                                                             │
   │         # 2. 环境交互                                       │
   │         env.step(u_final) → obs_next, reward, done          │
   │                                                             │
   │         # 3. 存储到 Rollout Buffer                          │
   │         buffer.store({                                      │
   │             obs, u_final, V(s), log π(u_rl|s),             │
   │             reward, done, old_log_probs                     │
   │         })                                                  │
   │                                                             │
   │         # 4. 定期从 Learner 获取最新模型参数                │
   │         if step % sync_interval == 0:                       │
   │             sync_params_from_learner()                      │
   └─────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【Learner】（GPU 训练）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ┌─────────────────────────────────────────────────────────────┐
   │ for epoch in range(num_epochs):                             │
   │     # 1. 从 Rollout Workers 收集数据                        │
   │     batch = collect_batch_from_workers()                    │
   │                                                             │
   │     # 2. V-trace 重要性采样修正（APPO 特有）               │
   │     # 计算重要性权重 ρ 和 c                                  │
   │     ρ = exp(log π(u_rl|s) - log μ(u_rl|s))                  │
   │     ρ = clip(ρ, 0, ρ_bar)                                   │
   │     c = clip(ρ, 0, c_bar)                                   │
   │                                                             │
   │     # 3. V-trace Target 计算                                │
   │     δ_t = r_t + γ * V(s_{t+1}) - V(s_t)                    │
   │     V_trace(s) = V(s) + Σ γ^t (Π c_i) ρ_t δ_t              │
   │                                                             │
   │     # 4. Advantage 计算（GAE）                              │
   │     A_t = GAE(V_trace, rewards, dones)                      │
   │                                                             │
   │     # 5. 损失计算                                           │
   │     # Actor Loss: PPO-CLIP + V-trace                        │
   │     ratio = exp(log π_new - log π_old)                      │
   │     ratio = ratio * Π ρ_i  # V-trace 修正                   │
   │     actor_loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) │
   │                                                             │
   │     # Critic Loss: V-trace Target                           │
   │     critic_loss = MSE(V(s), V_trace(s))                     │
   │                                                             │
   │     # 6. 梯度回传                                           │
   │     total_loss = actor_loss + critic_loss + entropy_loss    │
   │     total_loss.backward()                                   │
   │     optimizer.step()                                        │
   └─────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 模块输入输出对齐

| 模块 | 输入 | 输出 | 依赖 |
|------|------|------|------|
| **Encoder** | `obs` | `encoder_out` | - |
| **Policy Head** | `encoder_out` | `u_rl_mean` | Encoder |
| **Action Distribution** | `u_rl_mean` | `u_rl`, `log_probs` | Policy Head |
| **QuadCBFQPLayer** | `u_rl`, `obs` | `u_safe` | Policy Head |
| **Critic Head** | `encoder_out` | `V(s)` | Encoder |
| **V-trace** | `log π`, `log μ`, `rewards`, `V(s)` | `V_trace(s)`, `A_t` | Rollout Buffer |

---


## 1. 自定义 Actor-Critic 网络（最简实现）

**文件**: `swarm_rl/models/quad_multi_model_rcbf.py`

```python
import torch
import torch.nn as nn
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.algo.utils.tensor_dict import TensorDict
from swarm_rl.rcbf.quad_cbf_qp import QuadCBFQPLayer
from swarm_rl.models.quad_multi_model import make_quadmulti_encoder

class QuadActorCriticWithCBF(ActorCriticSharedWeights):
    """
    自定义 Actor-Critic（支持 APPO-CBF 模式 B / 纯 RL 模式）

    关键设计：
    - 继承 ActorCriticSharedWeights，复用所有组件（encoder/decoder/core 等）
    - 只覆写 forward_tail() 方法，添加 CBF-QP 层（最小修改）
    - 通过 --quads_use_cbf 参数控制开关

    注意：
    - 所有组件（encoder/decoder/core/action_parameterization/critic_linear）都来自基类
    - cfg 来自基类（self.cfg）
    - 动作分布由基类的 action_parameterization 生成
    """

    def __init__(self, model_factory, obs_space, action_space, cfg):
        # 1. 调用基类初始化（复用所有组件）
        super().__init__(model_factory, obs_space, action_space, cfg)
        
        self.use_cbf = cfg.quads_use_cbf  # CBF 开关

        # 2. CBF-QP 层（唯一新增组件）
        if self.use_cbf:
            self.cbf_layer = QuadCBFQPLayer(
                mass=cfg.quads_mass,
                thrust_to_weight=cfg.quads_thrust_to_weight,
                alpha_1=cfg.quads_cbf_alpha_1,
                alpha_2=cfg.quads_cbf_alpha_2,
                k_omega=cfg.quads_cbf_k_omega,
                device=cfg.device
            )

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        """
        覆写 forward()，传递 obs 到 forward_tail() 用于 CBF 约束计算
        """
        # forward_head: encoder
        x = self.forward_head(normalized_obs_dict)
        
        # forward_core: RNN (if use_rnn=True)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        
        # forward_tail: decoder + action + critic + CBF
        result = self.forward_tail(x, values_only, sample_actions=True, 
                                    obs=normalized_obs_dict['obs'])
        result["new_rnn_states"] = new_rnn_states
        
        return result

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool, 
                     obs=None) -> TensorDict:
        """
        覆写 forward_tail()，添加 CBF-QP 层

        基类实现：
            decoder_output = self.decoder(core_output)
            values = self.critic_linear(decoder_output).squeeze()
            action_distribution_params, self.last_action_distribution = 
                self.action_parameterization(decoder_output)
            self._maybe_sample_actions(sample_actions, result)

        我们的修改：
            在获取 action_distribution 后，采样 u_rl，然后通过 CBF-QP 得到 u_safe
        """
        # 1. Decoder（复用基类）
        decoder_output = self.decoder(core_output)
        
        # 2. Critic 输出 V(s)（复用基类）
        values = self.critic_linear(decoder_output).squeeze()
        
        result = TensorDict(values=values)
        if values_only:
            return result
        
        # 3. Policy 输出 action_logits（复用基类）
        action_distribution_params, self.last_action_distribution = \
            self.action_parameterization(decoder_output)
        result["action_logits"] = action_distribution_params
        
        # 4. 采样动作 u_rl（复用基类）
        if sample_actions:
            u_rl = self.last_action_distribution.sample()
            log_probs = self.last_action_distribution.log_prob(u_rl)
        else:
            u_rl = action_distribution_params
            log_probs = torch.zeros_like(action_distribution_params[:, 0:1])
        
        # 5. CBF-QP 层（新增）
        if self.use_cbf and obs is not None:
            state = self._extract_state(obs)
            sdf_obs = self._extract_sdf_obs(obs)
            
            if self.training:
                A, b = self.cbf_layer.compute_cbf_constraints(state, u_rl, sdf_obs)
                u_safe, _ = self.cbf_layer.solve_qp_differentiable(u_rl, A, b)
            else:
                u_safe = self.cbf_layer.get_safe_action(state, u_rl.cpu().numpy(), sdf_obs)
                u_safe = torch.tensor(u_safe, dtype=u_rl.dtype, device=u_rl.device)
            u_final = u_safe
        else:
            u_final = u_rl
        
        # 6. 输出最终动作
        result["actions"] = u_final
        result["log_probs"] = log_probs
        
        return result

    def _extract_state(self, obs):
        """
        从观测中提取状态（用于 CBF）
        
        obs 结构：[pos_rel(3), vel(3), rot(9), omega(3), sdf(9), ...]
        """
        return {
            'pos': obs[:, 0:3],
            'vel': obs[:, 3:6],
            'rot': obs[:, 6:15].reshape(-1, 3, 3),
            'omega': obs[:, 15:18],
        }

    def _extract_sdf_obs(self, obs):
        """
        从观测中提取 SDF 观测（用于 CBF）
        
        假设 SDF 观测在 18 维之后，长度为 9
        """
        return obs[:, 18:27]
```

---

## 2. CBF 核心模块

**文件**: `swarm_rl/rcbf/quad_cbf_qp.py`

```python
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
from qpth.qp import QPFunction

class QuadCBFQPLayer(nn.Module):
    """四旋翼 CBF-QP 层（训练/推理自动切换）
    
    注意：
    - 继承 nn.Module，自动跟随模型的 device
    - 不硬编码 device，在 forward() 中动态获取当前 device
    """
    
    def __init__(self, mass=0.028, thrust_to_weight=3.0,
                 alpha_1=1.0, alpha_2=1.0, k_omega=0.1):
        super().__init__()
        self.m = mass
        self.g = 9.81
        self.T_max = mass * self.g * thrust_to_weight / 4.0
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.k_omega = k_omega
        self.e3 = torch.tensor([0.0, 0.0, 1.0])  # 注册为 buffer，自动跟随 device
        
        # 注册为 buffer，自动跟随模型的 device
        self.register_buffer('e3_buffer', torch.tensor([0.0, 0.0, 1.0]))
        
        # qpth 的 QPFunction 会在 forward 中根据输入 tensor 的 device 自动处理
        self.qp_layer = QPFunction(verbose=False, maxIter=10000, eps=1e-4)
    
    def compute_sdf_gradient(self, sdf_obs, delta=0.1):
        """从 3×3 SDF 网格计算梯度（中心差分）"""
        h = sdf_obs[4]
        n_x = (sdf_obs[5] - sdf_obs[3]) / (2 * delta)
        n_y = (sdf_obs[7] - sdf_obs[1]) / (2 * delta)
        n = np.array([n_x, n_y, 0])
        norm = np.linalg.norm(n)
        if norm > 1e-6:
            n = n / norm
        return n, h
    
    def compute_cbf_constraints(self, state, u_rl, sdf_obs):
        """计算 CBF 约束 A @ u >= b"""
        v = state['vel']
        R = state['rot']
        omega = state['omega']
        
        n, h = self.compute_sdf_gradient(sdf_obs)
        h_dot = np.dot(n, v)
        denom = max(h + 0.5, 1e-6)
        n_dot_v = (np.dot(v, v) - h_dot**2) / denom
        Risk = self.k_omega * np.dot(omega[:2], omega[:2]) + 0.1
        
        Re3 = R @ self.e3
        nTRe3 = np.dot(n, Re3)
        A = (self.T_max / (2 * self.m)) * nTRe3 * np.ones(4)
        
        gravity_term = np.dot(n, self.g * self.e3)
        bias_term = (2 * self.T_max / self.m) * nTRe3
        b = (Risk + gravity_term - n_dot_v
             - (self.alpha_1 + self.alpha_2) * h_dot
             - self.alpha_1 * self.alpha_2 * h
             - bias_term)
        
        return A.reshape(1, -1), np.array([b])
    
    def solve_qp_differentiable(self, u_rl, A, b):
        """可微分 QP 求解（训练用）"""
        batch_size = u_rl.shape[0]
        n_u, n_slack = 4, 1
        n_x = n_u + n_slack
        
        P = torch.zeros(batch_size, n_x, n_x, device=self.device)
        P[:, :n_u, :n_u] = torch.eye(n_u, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        P[:, n_u:, n_u:] = torch.tensor([[1000.0]], device=self.device).expand(batch_size, -1, -1)
        
        q = torch.zeros(batch_size, n_x, device=self.device)
        q[:, :n_u] = -u_rl
        
        G_cbf = torch.zeros(batch_size, 1, n_x, device=self.device)
        G_cbf[:, :, :n_u] = -A
        G_cbf[:, :, n_u] = 1.0
        h_cbf = -b
        
        G_upper = torch.zeros(batch_size, n_u, n_x, device=self.device)
        G_upper[:, :, :n_u] = torch.eye(n_u, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        h_upper = torch.ones(batch_size, n_u, device=self.device)
        
        G_lower = torch.zeros(batch_size, n_u, n_x, device=self.device)
        G_lower[:, :, :n_u] = -torch.eye(n_u, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        h_lower = torch.ones(batch_size, n_u, device=self.device)
        
        G_slack = torch.zeros(batch_size, 1, n_x, device=self.device)
        G_slack[:, :, n_u] = -1.0
        h_slack = torch.zeros(batch_size, 1, device=self.device)
        
        G = torch.cat([G_cbf, G_upper, G_lower, G_slack], dim=1)
        h = torch.cat([h_cbf, h_upper, h_lower, h_slack], dim=1)
        
        try:
            x = self.qp_layer(P, q, G, h, torch.Tensor().to(self.device), torch.Tensor().to(self.device))
            u_safe = x[:, :n_u]
            return u_safe, {"solved": True}
        except Exception as e:
            return u_rl.clone(), {"solved": False}
    
    def get_safe_action(self, state, u_rl, sdf_obs):
        """完整求解：约束计算 + QP 求解（推理用）"""
        A, b = self.compute_cbf_constraints(state, u_rl, sdf_obs)
        u = cp.Variable(4)
        slack = cp.Variable(1, nonneg=True)
        objective = cp.Minimize(cp.sum_squares(u - u_rl) + 1000 * cp.sum_squares(slack))
        constraints = [A @ u >= b - slack, u >= -1, u <= 1, slack >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        return np.clip(u.value, -1, 1) if problem.status == 'optimal' else u_rl
```

---

## 3. 完整训练流程

```python
# swarm_rl/train.py
from sample_factory.train import run_rl
from swarm_rl.models.quad_multi_model_rcbf import QuadActorCriticWithCBF

def register_models():
    """注册自定义模型"""
    from sample_factory.algo.utils.context import global_model_factory
    
    def make_actor_critic_func(cfg, obs_space, action_space):
        return QuadActorCriticWithCBF(cfg, obs_space, action_space)
    
    global_model_factory().make_actor_critic_func = make_actor_critic_func

def main():
    register_models()
    cfg = parse_swarm_cfg()
    status = run_rl(cfg)
    return status
```

**训练命令（APPO + CBF）**：
```bash
python -m swarm_rl.train \
    --algo=APPO \
    --env=quadrotor_multi \
    --experiment=single_quad_cbf \
    --quads_use_cbf=True \
    --quads_cbf_alpha_1=1.0 \
    --quads_cbf_alpha_2=1.0 \
    --quads_cbf_k_omega=0.1 \
    --actor_critic_share_weights=False \
    --num_workers=12 \
    --num_envs_per_worker=2 \
    --learning_rate=0.0001 \
    --ppo_clip_value=5.0 \
    --with_vtrace=False \
    --max_policy_lag=100000000 \
    --rnn_size=256 \
    --rollout=128 \
    --batch_size=1024
```

**关键 APPO 参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algo` | APPO | 使用异步 PPO |
| `--num_workers` | 12 | Rollout Workers 数量 |
| `--num_envs_per_worker` | 2 | 每个 Worker 的环境数 |
| `--with_vtrace` | False | 是否使用 V-trace 重要性采样 |
| `--max_policy_lag` | 1e8 | 允许的最大策略滞后步数 |
| `--ppo_clip_value` | 5.0 | PPO-CLIP 的裁剪值 |
| `--quads_use_cbf` | False | **是否启用 CBF（新增）** |

---

## 网络结构详解（与 baseline 完全一致）

### Encoder 结构（复用 baseline）

```
┌─────────────────────────────────────────────────────────────┐
│                      Encoder                                 │
│                                                              │
│  obs_self (18 维) ──→ FC(256) ──→ Tanh ──→ FC(256) ──→ Tanh ──┐ │
│                                                            Cat ──→ FC(512) ──→ Tanh ──→ encoder_out (512)
│  obs_obstacle (9 维) ──→ FC(256) ──→ Tanh ──→ FC(256) ──→ Tanh ──┘ │
│                                                              │
│  结构：                                                       │
│  - Self Encoder: 2 层 MLP (18 → 256 → 256)                   │
│  - Obstacle Encoder: 2 层 MLP (9 → 256 → 256)                │
│  - Feed Forward: 1 层 MLP (512 → 512)                        │
│  - 输出：512 维                                               │
└─────────────────────────────────────────────────────────────┘
```

### Policy Head 结构

```
┌─────────────────────────────────────────────────────────────┐
│                    Policy Head                               │
│                                                              │
│  encoder_out (512) ──→ FC(256) ──→ Tanh ──→ FC(256) ──→ Tanh ──→ FC(4) ──→ u_rl_mean
│                                                              │
│  结构：2 层 MLP + 输出层 (512 → 256 → 256 → 4)               │
└─────────────────────────────────────────────────────────────┘
```

### Critic Head 结构

```
┌─────────────────────────────────────────────────────────────┐
│                     Critic Head                              │
│                                                              │
│  encoder_out (512) ──→ FC(256) ──→ Tanh ──→ FC(256) ──→ Tanh ──→ FC(1) ──→ V(s)
│                                                              │
│  结构：2 层 MLP + 输出层 (512 → 256 → 256 → 1)               │
└─────────────────────────────────────────────────────────────┘
```

### 网络结构总结表

| 组件 | 输入维度 | 隐藏层 | 输出维度 | 激活函数 |
|------|---------|--------|---------|---------|
| **Encoder** | 18+9 | 2 层 MLP + FF | 512 | Tanh |
| **Policy Head** | 512 | 256 → 256 | 4 | Tanh |
| **Critic Head** | 512 | 256 → 256 | 1 | Tanh |

---

## 总结

### 关键设计

| 变量 | 含义 | 用途 |
|------|------|------|
| `u_rl` | RL 标称动作 | 计算 `log_probs`（PPO ratio 和 V-trace） |
| `u_safe` | CBF 修正后的安全动作 | 环境执行 |
| `u_final` | 最终执行动作 | `u_final = u_safe`（启用 CBF 时） |

### APPO + CBF 架构优势

1. **最小修改**：只自定义 Actor-Critic 网络，不修改 Sample Factory 源码
2. **完全兼容**：使用标准 APPO 的 V-trace 和 GAE，与 Sample Factory 完全兼容
3. **网络结构一致**：Encoder 和 Head 结构与 baseline 完全一致
4. **复用 Encoder**：通过 `make_quadmulti_encoder` 复用 baseline 的 Encoder
5. **职责清晰**：`QuadCBFQPLayer` 封装所有 CBF 逻辑，训练/推理自动切换
6. **支持 share_weights**：兼容 `--actor_critic_share_weights` 参数
7. **易于部署**：推理时用 cvxpy（CPU 友好），训练时用 qpth（GPU 加速）

### APPO vs PPO 选择

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| **小规模实验/调试** | PPO | 样本效率高，配置简单 |
| **大规模训练** | APPO | 吞吐量高，适合多 GPU/多 Worker |
| **需要快速迭代** | APPO | 异步收集，减少等待时间 |
| **样本昂贵** | PPO | on-policy，样本利用率高 |
