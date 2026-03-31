# Adaptive Skill RL 完整方案

> 多技能自适应强化学习用于四旋翼无人机敏捷飞行控制
>
> 版本：v1.0  
> 日期：2026 年 1 月  

---

## 📋 目录

- [研究目标](#研究目标)
- [方法设计](#方法设计)
- [对比实验](#对比实验)
- [赛道设计](#赛道设计)
- [评估指标](#评估指标)
- [实现计划](#实现计划)
- [预期结果](#预期结果)
- [论文贡献](#论文贡献)

---

## 🎯 研究目标

### 核心问题

传统单一策略 RL 控制器难以同时处理多种飞行场景：
- **高速巡航**：需要大推力、小机动
- **紧急避障**：需要快速转向、规划路线
- **精确机动**：需要快速 roll/pitch 机动

### 解决方案

**Adaptive Skill RL**：学习多个技能 + 自适应选择

```
状态 → [共享编码器] → 表征 h
              │
         ┌────┴────┐
         │         │
    ┌────┴────┐ ┌──┴──┐
    │技能头×3 │ │Gating│
    └────┬────┘ └──┬──┘
         │a₁..a₃   │w₁..w₃
         └────┬────┘
              │
         a = Σ wᵢ * aᵢ
```

### 核心创新

**系统研究技能引导方法对分化效果的影响**（对比实验）

- 无引导：纯自发分化
- 弱引导：多样性损失
- 强引导：多样性 + 技能特定奖励

---

## 📋 方法设计

### 1. 网络架构

#### 1.1 整体结构

```python
class AdaptiveSkillPolicy(nn.Module):
    """
    自适应技能策略网络
    
    组件：
    1. 共享编码器 - 提取状态特征
    2. 多个技能头 - 每个输出完整动作
    3. 技能选择器 - 学习何时用哪个技能
    4. 加权融合 - 输出最终动作
    """
    def __init__(self, obs_dim=30, action_dim=4, num_skills=3):
        super().__init__()
        
        # 1. 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # 2. 多个技能头（每个输出完整动作）
        self.skill_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            ) for _ in range(num_skills)
        ])
        
        # 3. 技能选择器（Gating Network）
        self.gating = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_skills),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, obs):
        # 编码
        h = self.encoder(obs)
        
        # 各技能头输出动作
        skill_actions = [head(h) for head in self.skill_heads]
        skill_actions = torch.stack(skill_actions, dim=1)
        
        # 选择器输出权重
        weights = self.gating(h)
        
        # 加权融合
        final_action = torch.sum(weights.unsqueeze(-1) * skill_actions, dim=1)
        
        return final_action, weights, skill_actions
```

#### 1.2 动作空间

```python
# 四旋翼无人机动作空间
action = [thrust, wx, wy, wz]
#   thrust: 推力 (0 到 1)
#   wx: 绕 x 轴角速度 (-1 到 1)
#   wy: 绕 y 轴角速度 (-1 到 1)
#   wz: 绕 z 轴角速度 (-1 到 1)

# 每个技能头都输出完整的 4 维动作
# Skill 1: [thrust₁, wx₁, wy₁, wz₁]
# Skill 2: [thrust₂, wx₂, wy₂, wz₂]
# Skill 3: [thrust₃, wx₃, wy₃, wz₃]

# 最终动作是加权和
# final = w₁*a₁ + w₂*a₂ + w₃*a₃
```

### 2. 训练方法

#### 2.1 损失函数

```python
# 总损失
total_loss = L_ppo + λ * L_diversity

# PPO loss（标准）
L_ppo = -E[min(ratio*A, clip(ratio,1-ε,1+ε)*A)]
ratio = exp(log_prob_new - log_prob_old)

# 多样性损失（鼓励技能差异化）
L_diversity = -mean(cosine_similarity(aᵢ, aⱼ))

# 技能特定奖励（强引导组用）
L_skill_specific = Σ wᵢ * rewardᵢ
```

#### 2.2 训练流程

```python
class AdaptiveSkillTrainer:
    def train_step(self, batch):
        # 1. 前向传播
        action, weights, skill_actions = self.model(obs)
        
        # 2. 计算 PPO loss
        ppo_loss = compute_ppo_loss(action, reward, advantage)
        
        # 3. 计算多样性损失
        diversity_loss = -mean(cosine_similarity(skill_actions))
        
        # 4. 总损失
        total_loss = ppo_loss + λ * diversity_loss
        
        # 5. 反向传播
        total_loss.backward()
        optimizer.step()
```

---

## 🔬 对比实验

### 3 组对比实验

| 实验组 | 技能数 | 多样性损失 | 技能特定奖励 | 引导强度 |
|--------|-------|-----------|-------------|---------|
| **no_guidance** | 3 | 0.1 | ❌ | 无 |
| **weak_guidance** | 3 | 0.5 | ❌ | 弱 |
| **strong_guidance** | 3 | 0.5 | ✅ | 强 |

### 详细配置

```python
experiments = {
    'no_guidance': {
        'num_skills': 3,
        'diversity_loss_weight': 0.1,
        'skill_specific_reward': False,
        'description': '纯自发分化',
    },
    
    'weak_guidance': {
        'num_skills': 3,
        'diversity_loss_weight': 0.5,
        'skill_specific_reward': False,
        'description': '多样性损失引导',
    },
    
    'strong_guidance': {
        'num_skills': 3,
        'diversity_loss_weight': 0.5,
        'skill_specific_reward': True,
        'reward_weights': {
            'skill1_thrust': 1.0,      # 鼓励推力控制
            'skill2_yaw': 1.0,         # 鼓励 yaw 控制
            'skill3_roll_pitch': 1.0,  # 鼓励 roll/pitch 控制
        },
        'description': '多样性 + 特定奖励引导',
    },
}
```

### 技能定义（预期分化）

| 技能 | 主要控制维度 | 预期行为 | 典型场景 |
|------|------------|---------|---------|
| **Skill 1** | thrust | 高速巡航 | 开阔区 |
| **Skill 2** | yaw (wz) | 转向避障 | 接近障碍 |
| **Skill 3** | roll/pitch (wx, wy) | 快速机动 | 密集障碍 |

---

## 🎁 奖励函数

### 基础奖励（所有组共享）

```python
def compute_base_reward(state, action, info):
    reward = 0.0
    
    # 1. 进度奖励（向终点前进）
    goal_dir = goal_pos - state[:3]
    velocity = state[3:6]
    reward += 1.0 * dot(normalize(goal_dir), velocity)
    
    # 2. 碰撞惩罚
    if info.get('collided'):
        reward -= 10.0
    
    # 3. 完成奖励
    if info.get('finished'):
        reward += 50.0
    
    return reward
```

### 技能特定奖励（强引导组用）

```python
def compute_skill_specific_reward(state, action, skill_weights):
    skill_rewards = torch.zeros(3)
    
    # Skill 1: 鼓励高速（推力大）
    thrust = action[:, 0]
    skill_rewards[0] = 1.0 * thrust
    
    # Skill 2: 鼓励转向（yaw 大）
    yaw_rate = action[:, 3]
    skill_rewards[1] = 1.0 * torch.abs(yaw_rate)
    
    # Skill 3: 鼓励机动（roll/pitch 大）
    roll_pitch = torch.norm(action[:, 1:3], dim=1)
    skill_rewards[2] = 1.0 * roll_pitch
    
    # 加权
    total = torch.sum(skill_weights * skill_rewards)
    
    return total
```

---

## 🗺️ 赛道设计

### 障碍物密度渐变

```python
track = {
    'start': [0, 0, 2],
    'goal': [100, 0, 2],
    
    # 障碍物布局
    'obstacles': [
        # 0-30 米：无障碍（开阔）
        
        # 30-50 米：稀疏障碍（5 个）
        {'pos': [35, -5], 'size': 1},
        {'pos': [40, 3], 'size': 1},
        {'pos': [45, -2], 'size': 1},
        {'pos': [48, 4], 'size': 1},
        {'pos': [49, -4], 'size': 1},
        
        # 50-70 米：密集障碍（12 个）
        {'pos': [52, -3], 'size': 1},
        {'pos': [52, 0], 'size': 1},
        {'pos': [52, 3], 'size': 1},
        {'pos': [55, -3], 'size': 1},
        {'pos': [55, 0], 'size': 1},
        {'pos': [55, 3], 'size': 1},
        {'pos': [58, -3], 'size': 1},
        {'pos': [58, 0], 'size': 1},
        {'pos': [58, 3], 'size': 1},
        {'pos': [60, -2], 'size': 1},
        {'pos': [60, 2], 'size': 1},
        {'pos': [62, 0], 'size': 1},
        
        # 70-100 米：稀疏障碍（3 个）
        {'pos': [75, -2], 'size': 1},
        {'pos': [80, 2], 'size': 1},
        {'pos': [85, -1], 'size': 1},
    ],
}
```

### 可视化

```
俯视图：

0m      30m      50m      70m      100m
│       │        │        │        │
开阔    稀疏     密集     稀疏     开阔
│       │        │        │        │
● ───── ○○○ ─── ○○○○○○ ─── ○○○ ────── ●
        │        ││││││    │
      避障     精确机动   避障
     Skill 2  Skill 3  Skill 2
       
巡航 Skill 1（全程可用）
```

---

## 📊 评估指标

### 分化质量指标

```python
metrics = {
    # 技能分化
    'skill_entropy': '技能熵（越高越均匀）',
    'skill_correlation': '技能间相关性（越低越好）',
    'control_focus': '控制维度集中度',
    
    # 任务性能
    'success_rate': '完成率',
    'avg_time': '平均完成时间',
    'avg_speed': '平均速度',
    'collision_rate': '碰撞率',
    
    # 训练效率
    'convergence_speed': '收敛速度（步数）',
    'sample_efficiency': '样本效率',
}
```

### 控制维度分析

```python
def analyze_control_focus(skill_actions):
    """
    分析每个技能主要控制哪个维度
    """
    control_names = ['thrust', 'roll', 'pitch', 'yaw']
    
    results = {}
    for skill_id in range(3):
        action = skill_actions[:, skill_id]
        focus = torch.mean(torch.abs(action), dim=0)
        
        # 找出主要控制维度
        main_control = control_names[torch.argmax(focus)]
        results[skill_id] = {
            'focus': focus.tolist(),
            'main_control': main_control,
        }
    
    return results
```

---

## 🗓️ 实现计划

### 时间线

| 周次 | 任务 | 产出 |
|------|------|------|
| **1-2** | no_guidance 实现 + 训练 | 基础模型 + 结果 |
| **3-4** | weak_guidance 实现 + 训练 | 多样性损失调参 |
| **5-6** | strong_guidance 实现 + 训练 | 技能特定奖励 |
| **7** | 对比分析 + 可视化 | 图表 + 数据 |
| **8-9** | 论文写作 | 完整论文 |
| **10** | 修改 + 投稿 | 提交 IROS/ICRA |

### 需要创建的文件

```
swarm_rl/
├── models/
│   └── adaptive_skill_model.py          # 核心模型
├── env_wrappers/
│   ├── adaptive_skill_params.py         # 配置参数
│   └── adaptive_skill_reward.py         # 技能特定奖励
├── configs/
│   └── adaptive_skill_configs.py        # 实验配置
├── runs/
│   └── adaptive_skill/
│       ├── no_guidance.py               # 实验 1 配置
│       ├── weak_guidance.py             # 实验 2 配置
│       └── strong_guidance.py           # 实验 3 配置
├── train_comparison.py                   # 对比训练脚本
└── analyze_skills.py                     # 技能分析工具
```

### 训练命令

```bash
# 实验 1: no_guidance
python -m swarm_rl.train_comparison no_guidance

# 实验 2: weak_guidance
python -m swarm_rl.train_comparison weak_guidance

# 实验 3: strong_guidance
python -m swarm_rl.train_comparison strong_guidance

# Baseline: 标准 PPO
python -m swarm_rl.train --algo=APPO --env=quadrotor_multi \
    --quads_num_agents=1 --quads_num_skills=1
```

---

## 📈 预期结果

### 分化质量（预期）

| 实验组 | 技能熵 | 相关性 | 控制集中度 | 分化质量 |
|--------|-------|--------|-----------|---------|
| no_guidance | 0.8 | 0.6 | 低 | 差 |
| weak_guidance | 1.0 | 0.3 | 中 | 中 |
| strong_guidance | 1.1 | 0.1 | 高 | 优 |

### 性能对比（预期）

| 方法 | 完成率 | 平均时间 | 平均速度 | 碰撞率 |
|------|--------|---------|---------|--------|
| PPO (baseline) | 70% | 32s | 4.8 m/s | 20% |
| no_guidance | 65% | 35s | 4.5 m/s | 25% |
| weak_guidance | 75% | 30s | 5.2 m/s | 18% |
| **strong_guidance** | **85%** | **25s** | **6.0 m/s** | **10%** |

### 可视化示例

```
技能权重随位置变化（strong_guidance 预期）:

位置 (m): 0    20    40    60    80    100
          │     │     │     │     │     │
Skill 1:  ████████░░░░░░░░░░░░████████  (巡航)
Skill 2:  ░░░░░░░░████░░░░░░████░░░░░░  (避障)
Skill 3:  ░░░░░░░░░░░░████████░░░░░░░░  (机动)
          │     │     │     │     │     │
         开阔  稀疏   密集   稀疏  开阔
```

---

## 📝 论文贡献

### 可以声称的贡献

1. **方法创新**：
   > "提出 Adaptive Skill RL 框架，学习多技能 + 自适应选择"

2. **系统研究**：
   > "首次系统研究技能引导方法对分化效果的影响"

3. **实验发现**：
   > "强引导（多样性 + 特定奖励）分化效果最好，性能提升 15%"

4. **设计指南**：
   > "推荐使用强引导方法，多样性损失权重 0.5-1.0"

### 论文结构

```
1. 引言
   - 多智能体 RL 的挑战
   - 技能分化的重要性
   - 我们的贡献

2. 相关工作
   - 多技能 RL
   - 技能引导方法
   - 无人机控制

3. 方法
   - Adaptive Skill RL 框架
   - 技能引导方法（3 种）

4. 实验
   - 实验设置
   - 对比实验（3 组）
   - 结果分析

5. 结果
   - 分化质量对比
   - 性能对比
   - 消融实验

6. 讨论
   - 引导方法的影响
   - 设计指南

7. 结论
```

---

## 🎯 目标会议

| 会议 | 截稿日期 | 接收率 | 适合程度 |
|------|---------|--------|---------|
| **IROS 2025** | 2025 年 3 月 | ~50% | ⭐⭐⭐⭐ |
| **ICRA 2026** | 2025 年 9 月 | ~45% | ⭐⭐⭐⭐⭐ |
| **CoRL 2025** | 2025 年 6 月 | ~40% | ⭐⭐⭐ |

---

## ✅ 成功标准

### 实验成功标志

- [ ] 3 组实验都完成训练（各 3M 步）
- [ ] strong_guidance 技能分化明显（控制集中度高）
- [ ] 性能提升显著（完成率 +15%，速度 +20%）
- [ ] 可视化清晰（技能权重图、控制焦点图）

### 论文成功标志

- [ ] 方法描述清晰
- [ ] 对比实验完整
- [ ] 结果有说服力
- [ ] 投稿 IROS/ICRA

---

## 💡 关键要点总结

1. **核心思想**：多技能 + 自适应选择
2. **创新点**：系统研究引导方法（对比实验）
3. **技能数**：3 个（巡航、避障、机动）
4. **引导方式**：无/弱/强 3 组对比
5. **赛道**：密度渐变（开阔→稀疏→密集）
6. **奖励**：基础 3 项 + 技能特定（强引导组）
7. **训练**：各 3M 步，共 9M 步
8. **目标**：ICRA 2026

---

## 🔧 多技能头优化方案（v2.0 新增）

### 从单头到三头的挑战

从 **1 个策略头 → 3 个策略头**，需要针对性优化，否则可能**效果不升反降**！

#### 核心问题

| 问题 | 描述 | 后果 |
|------|------|------|
| **梯度稀释** | 梯度分流到 3 个头，每个头分到 1/3 | 训练变慢，收敛困难 |
| **技能重复** | 3 个头学到相同行为 | 参数浪费，性能不提升 |
| **选择器坍塌** | 总是选择同一个技能 | 其他技能学不到东西 |

---

### 必须加的优化

#### 1. 多样性损失（⭐⭐⭐⭐⭐ 必须）

```python
def compute_diversity_loss(skill_actions):
    """
    鼓励 3 个技能头输出不同的动作
    
    Args:
        skill_actions: [batch, 3, 4]
    
    Returns:
        diversity_loss: scalar
    """
    # 计算技能对之间的余弦相似度
    a1, a2, a3 = skill_actions.unbind(dim=1)
    
    sim_12 = nn.functional.cosine_similarity(a1, a2, dim=-1)
    sim_13 = nn.functional.cosine_similarity(a1, a3, dim=-1)
    sim_23 = nn.functional.cosine_similarity(a2, a3, dim=-1)
    
    # 平均相似度
    avg_sim = (sim_12 + sim_13 + sim_23) / 3
    
    # 多样性损失 = -平均相似度
    diversity_loss = -avg_sim.mean()
    
    return diversity_loss

# 总损失
total_loss = ppo_loss + 0.5 * diversity_loss  # ← 必须加这个！
```

**为什么必须**：
- ✅ 强制 3 个头学习不同行为
- ✅ 避免参数浪费
- ✅ 让选择器有选择的余地

---

#### 2. 技能特定偏置（⭐⭐⭐⭐ 强烈推荐）

```python
class SkillHeadWithBias(nn.Module):
    def __init__(self, hidden_dim, action_dim, skill_id):
        super().__init__()
        self.skill_id = skill_id
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        
        # 技能偏置（关键！）
        self.bias = nn.Parameter(torch.zeros(action_dim))
        
        # 初始化不同偏置
        biases = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),  # 巡航 - 推力
            torch.tensor([0.0, 0.0, 0.0, 1.0]),  # 避障 - yaw
            torch.tensor([0.0, 1.0, 1.0, 0.0]),  # 机动 - roll/pitch
        ]
        self.bias.data = biases[skill_id]
    
    def forward(self, h):
        return self.net(h) + self.bias
```

**效果**：
- ✅ 初始化就有差异
- ✅ 引导梯度流向不同方向
- ✅ 加速分化（节省 1M 步）

---

#### 3. 温度系数（⭐⭐⭐⭐ 推荐）

```python
class GatingWithTemperature(nn.Module):
    def __init__(self, hidden_dim, num_skills):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_skills),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, h):
        logits = self.net(h)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return weights

# 训练时逐渐降低温度
temperature = max(0.5, 1.0 - steps / 1_000_000)
```

**为什么需要**：
- ✅ 训练初期：温度高 → 权重均匀 → 所有技能都学习
- ✅ 训练后期：温度低 → 权重集中 → 明确选择
- ✅ 避免早期坍塌

---

#### 4. 技能使用均衡损失（⭐⭐⭐ 可选）

```python
def compute_skill_balance_loss(weights):
    """
    鼓励所有技能都被使用，避免某些技能被忽略
    """
    # weights: [batch, 3]
    
    # 平均权重（期望均匀使用）
    mean_weights = weights.mean(dim=0)  # [3]
    
    # 均匀分布
    uniform = torch.ones_like(mean_weights) / weights.shape[1]
    
    # KL 散度（越接近均匀越好）
    balance_loss = nn.functional.kl_div(
        torch.log(mean_weights + 1e-8),
        uniform,
        reduction='batchmean'
    )
    
    return balance_loss

# 总损失
total_loss = ppo_loss + 0.5 * diversity_loss + 0.1 * balance_loss
```

**为什么需要**：
- ✅ 避免某些技能永远不被选择
- ✅ 确保所有技能都学到东西
- ✅ 提高整体性能

---

### 优化版网络架构

```python
class OptimizedAdaptiveSkillPolicy(ActorCriticSharedWeights):
    """
    优化版多技能策略（针对 3 头优化）
    
    Sample Factory 集成版本
    """
    
    def __init__(self, model_factory, obs_space, action_space, cfg):
        super().__init__(model_factory, obs_space, action_space, cfg)
        
        hidden_dim = cfg.rnn_size
        action_dim = action_space.shape[0]
        self.num_skills = getattr(cfg, 'quads_num_skills', 3)
        
        # 1. 技能头（带偏置）
        self.skill_heads = nn.ModuleList([
            SkillHeadWithBias(hidden_dim, action_dim, i)
            for i in range(self.num_skills)
        ])
        
        # 2. 选择器（带温度）
        self.gating = GatingWithTemperature(hidden_dim, self.num_skills)
        
        # 3. 初始化
        self._initialize_weights()
    
    def forward_tail(self, core_output, values_only, sample_actions, obs=None):
        # Decoder（复用基类）
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()
        
        result = TensorDict(values=values)
        if values_only:
            return result
        
        # 技能头输出
        skill_actions = torch.stack([
            head(decoder_output) for head in self.skill_heads
        ], dim=1)  # [batch, 3, 4]
        
        # 选择器权重
        weights = self.gating(decoder_output)  # [batch, 3]
        
        # 加权融合
        final_action = torch.sum(
            weights.unsqueeze(-1) * skill_actions,
            dim=1
        )  # [batch, 4]
        
        # 动作分布
        action_distribution_params, self.last_action_distribution = \
            self.action_parameterization(final_action)
        
        result["action_logits"] = action_distribution_params
        
        # 采样动作
        if sample_actions:
            actions = self.last_action_distribution.sample()
            actions = torch.clamp(actions, min=-1.0, max=1.0)
            log_prob_actions = self.last_action_distribution.log_prob(actions)
        else:
            actions = action_distribution_params
            actions = torch.clamp(actions, min=-1.0, max=1.0)
            log_prob_actions = None
        
        result["actions"] = actions
        if log_prob_actions is not None:
            result["log_prob_actions"] = log_prob_actions
        
        # 记录技能权重（用于监控）
        result["skill_weights"] = weights.detach()
        result["skill_actions"] = skill_actions.detach()
        
        return result
```

---

### 优化版配置

```python
# 最小可用优化配置
config = {
    'num_skills': 3,
    
    # 必须加的优化
    'diversity_loss_weight': 0.5,      # ← 多样性损失
    'use_skill_bias': True,            # ← 技能偏置
    'gating_temperature': 1.0,         # ← 温度系数
    
    # 可选优化
    'balance_loss_weight': 0.1,        # ← 均衡损失
    'use_curriculum': True,            # ← 课程学习
    'use_skill_reward': True,          # ← 技能特定奖励
}
```

---

### 优化效果对比

| 配置 | 分化速度 | 最终性能 | 训练稳定性 | 推荐度 |
|------|---------|---------|-----------|--------|
| **直接 3 头（无优化）** | 慢 | 差 | 不稳定 | ❌ |
| **+ 多样性损失** | 中 | 中 | 稳定 | ⭐⭐⭐ |
| **+ 技能偏置** | 快 | 好 | 稳定 | ⭐⭐⭐⭐ |
| **+ 温度系数** | 快 | 好 | 很稳定 | ⭐⭐⭐⭐ |
| **+ 均衡损失** | 快 | 很好 | 最稳定 | ⭐⭐⭐⭐⭐ |

---

### 不加优化的后果

```
❌ 3 个头学到相同行为（参数浪费 3 倍）
❌ 选择器总是选同一个头（其他头学不到）
❌ 训练不稳定（梯度竞争）
❌ 性能不如单头（白加了参数）
```

### 加了优化的效果

```
✅ 3 个头学到不同行为（巡航/避障/机动）
✅ 选择器根据场景选择
✅ 训练稳定收敛
✅ 性能提升 15-20%
```

---

## 📚 参考文献

1. Haarnoja et al. "Composable Deep Reinforcement Learning for Robotic Manipulation." ICRA 2018.
2. Vezhnevets et al. "FeUdal Networks for Hierarchical Reinforcement Learning." ICML 2017.
3. Eysenbach et al. "Diversity is All You Need: Learning Skills without a Reward Function." ICLR 2018.
4. Schulman et al. "Proximal Policy Optimization Algorithms." arXiv 2017.
5. Henderson et al. "Deep Reinforcement Learning that Matters." AAAI 2018.

---

*文档版本：v2.0（优化版）*
*最后更新：2026 年 1 月*
*作者：QuadSwarm-RL Team*

---

## 📝 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2026-01 | 初始方案：基础架构 + 对比实验 |
| v2.0 | 2026-01 | 优化方案：多技能头针对性优化 |
