# Adaptive Skill RL 使用指南

## 📋 文件结构

```
swarm_rl/
├── adaptive_skill/              # Adaptive Skill 模块
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── adaptive_skill_model.py    # 自定义模型（技能头 + 选择器）
│   └── losses/
│       ├── __init__.py
│       └── diversity_loss.py          # 多样性损失计算
│
├── train_adaptive_skill.py            # 训练脚本
└── runs/
    └── adaptive_skill/
        └── quads_use_adaptive_skill.py  # 运行配置
```

---

## 🚀 快速开始

### 方法 1: 使用训练脚本（推荐）

```bash
python -m swarm_rl.train_adaptive_skill \
    --algo=APPO \
    --env=quadrotor_multi \
    --quads_num_agents=1 \
    --quads_num_skills=3 \
    --diversity_loss_weight=0.5 \
    --train_for_env_steps=5000000 \
    --summary_dir=train_dir/adaptive_skill_001
```

### 方法 2: 使用 launcher.run（分布式训练）

```bash
python -m sample_factory.launcher.run \
    --run=swarm_rl.runs.adaptive_skill.quads_use_adaptive_skill \
    --max_parallel=4 \
    --experiments_per_gpu=1 \
    --num_gpus=1
```

### 方法 3: 使用 train.sh

```bash
bash train.sh
```

---

## ⚙️ 配置参数

### 技能配置

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--quads_num_skills` | 3 | 技能数量 |
| `--quads_use_adaptive_skill` | False | 启用 Adaptive Skill |
| `--quads_use_skill_bias` | True | 使用技能偏置 |

### 损失配置

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--diversity_loss_weight` | 0.5 | 多样性损失权重 |
| `--balance_loss_weight` | 0.1 | 均衡损失权重 |

### 选择器配置

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--gating_temperature` | 1.0 | 选择器温度系数 |

---

## 🎯 核心组件

### 1. 模型（`adaptive_skill_model.py`）

```python
class AdaptiveSkillPolicy(ActorCriticSharedWeights):
    """
    多技能策略网络
    
    - 共享编码器（复用 Sample Factory 基类）
    - 3 个技能头（每个输出完整动作）
    - 技能选择器（加权融合）
    """
```

### 2. 技能头（带偏置）

```python
class SkillHeadWithBias(nn.Module):
    """
    技能头（带偏置，引导分化）
    
    Skill 0: 推力偏置 [1, 0, 0, 0] → 巡航
    Skill 1: Yaw 偏置 [0, 0, 0, 1] → 避障
    Skill 2: Roll/Pitch偏置 [0, 1, 1, 0] → 机动
    """
```

### 3. 选择器（带温度）

```python
class GatingWithTemperature(nn.Module):
    """
    技能选择器（带温度系数）
    
    温度高 → 权重均匀 → 探索
    温度低 → 权重集中 → 利用
    """
```

---

## 📊 训练监控

### TensorBoard 指标

```bash
tensorboard --logdir=train_dir/adaptive_skill_001
```

### 关键指标

- `loss/diversity_loss`: 多样性损失（应该逐渐降低）
- `loss/total_loss`: 总损失
- `misc/skill_weights`: 技能权重分布

---

## ⚠️ 注意事项

### 1. 多样性损失

**问题**：Sample Factory 的 Learner 不支持直接扩展。

**解决**：多样性损失目前**只在模型中计算**，需要在训练后处理。

### 2. 技能分化

**建议**：
- 训练初期（0-1M 步）：多样性损失权重 0.5
- 训练中期（1M-3M 步）：多样性损失权重 0.3
- 训练后期（3M-5M 步）：多样性损失权重 0.1

### 3. 技能偏置

**作用**：加速技能分化，避免所有技能学到相同行为。

**建议**：始终开启（`--quads_use_skill_bias=True`）

---

## 🔧 故障排除

### 问题 1: 技能未分化

**症状**：3 个技能输出几乎相同的动作。

**解决**：
1. 增加多样性损失权重（`--diversity_loss_weight=1.0`）
2. 检查技能偏置是否启用（`--quads_use_skill_bias=True`）
3. 延长训练时间

### 问题 2: 选择器坍塌

**症状**：总是选择同一个技能（权重 [0.9, 0.05, 0.05]）。

**解决**：
1. 增加温度系数（`--gating_temperature=2.0`）
2. 添加均衡损失（`--balance_loss_weight=0.5`）

### 问题 3: 训练不稳定

**症状**：loss 震荡，性能不提升。

**解决**：
1. 降低学习率（`--learning_rate=1e-4`）
2. 增加梯度裁剪（`--max_grad_norm=0.5`）

---

## 📚 参考

- CBF 集成方式：`swarm_rl/models/quad_multi_model_rcbf.py`
- Sample Factory 文档：https://samplefactory.dev/
- 完整方案文档：`docs/Adaptive_Skill_RL_完整方案.md`
