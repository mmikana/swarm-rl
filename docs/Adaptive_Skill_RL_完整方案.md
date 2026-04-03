# Adaptive Skill RL 方案

> 多技能自适应强化学习用于四旋翼无人机敏捷飞行控制
>
> 版本：v3.0  
> 日期：2026 年 4 月

---

## 1. 研究问题

传统单一策略在无人机穿障任务中往往需要同时兼顾三类控制需求：

- 常规转向避障：主要依赖 `yaw` 调整航向，动作平滑，适合仍有转向空间的场景。
- 紧急机动避障：需要快速 `roll/pitch` 机动，适合近距离或高风险障碍。
- 推力稳定与能量管理：需要合理控制 `thrust`，维持速度、高度和恢复裕度。

因此，本方案的核心判断是：  
单一策略并不一定足以高效覆盖这些具有明显模式差异的控制需求，多技能策略更可能学出有用的分工。

---

## 2. 方法核心

### 2.1 总体结构

Adaptive Skill RL 采用共享表征加多技能头的结构：

`状态 -> 共享编码器 -> 3 个 skill heads + gating -> 融合动作`

其中：

- 共享编码器负责提取统一状态表征。
- 每个 skill head 输出完整动作，而不是只负责某一个维度。
- gating 网络根据当前状态给出各 skill 的权重。
- 最终动作由各 skill 的输出按权重融合得到。

这意味着本方案不是硬切换的层级控制器，而是一个带有结构先验的多模态策略。

### 2.2 技能设定

动作空间可理解为 `thrust / roll / pitch / yaw` 四个控制通道。  
本方案预期出现的三类技能分工为：

- Skill 1: `yaw` 主导的导航技能
- Skill 2: `roll/pitch` 主导的机动技能
- Skill 3: `thrust` 主导的稳定/能量技能

这里要强调两点：

- 这三类分工是研究假设和期望现象，不是训练时的硬标签。
- 每个 skill head 仍输出完整动作，只是预期在不同维度上形成不同的主导性。

### 2.3 Skill 融合与选择

本方案采用 soft gating，而不是一开始就做 hard routing。

原因是：

- 无人机控制本身是连续的，很多状态下确实需要多种控制模式共同作用。
- 完全硬切换虽然可解释，但容易导致训练不稳和早期死头。

但 soft gating 也有一个明确风险：  
如果始终过于平均，最终会退化成“多个 head 的平滑平均”，而不是形成有意义的模式选择。

因此，本方案采用 `soft-to-sharp` 的思路：

- 训练早期允许较软的混合，保证所有技能都能获得梯度。
- 训练中后期逐步变尖锐，让某一技能在关键状态下更明显地主导。
- 在低风险状态下允许平滑组合，在高风险状态下鼓励更明确的选择。

这里的 gating 只负责“如何选”，不负责“head 本身学什么”。

### 2.4 损失设计

主方案的训练目标保持简洁：

`L = L_PPO + lambda_div * L_diversity`

其中：

- `L_PPO` 是标准 PPO 损失。
- `L_diversity` 是 skill 之间的多样性正则。

本方案采用基于余弦相似度的弱约束形式：

`L_diversity = mean(relu(cos(a_i, a_j)))`

其含义是：

- 只惩罚 skill 之间过于相似的动作输出。
- 不预先规定哪个 skill 必须负责哪个动作维度。
- 只鼓励“不要学成一样”，而不强行指定“应该怎么不同”。

这也是为什么主方案先用 `cosine diversity`，而不是“主导维度约束”：

- `cosine diversity` 更适合验证 skill 是否能自动分化。
- 主导约束虽然更强、更可控，但会引入更重的人为先验。
- 如果自动分化不足，主导约束可以作为后续辅助消融，而不是主方案默认项。

### 2.5 正则项职责边界

本方案明确区分两类问题：

- `diversity_loss` 解决的是 `head collapse`
  不同 skill head 学成几乎相同的策略。

- gating 相关机制解决的是 `gate averaging / gate collapse`
  即选择器长期平均混合，或者过早塌成单头独占。

换句话说：

- `diversity_loss` 让“备选专家”彼此不同。
- gating 机制让“选择器”在合适的时候做出清晰选择。

### 2.6 当前主方案不使用 Skill-Specific Reward

本方案当前不把 `skill-specific reward` 作为主方法组成部分。

原因是：

- 主方案已经由多头结构、`diversity_loss` 和 `soft-to-sharp gating` 构成闭环。
- 额外的功能奖励会引入较强人为先验，削弱“自动分化”的研究结论。
- 设计不当的 skill reward 很容易演化成对动作幅度的奖励，带来明显的 reward hacking 风险。

因此，当前版本的核心方法只依赖：

- 多 skill heads
- diversity regularization
- soft-to-sharp gating

---

## 3. 实验设计

### 3.1 主研究问题

主实验围绕两个问题展开：

1. 仅靠多头结构，skill 是否会自发分化？
2. 加入 `diversity_loss` 后，分化质量和任务表现是否稳定提升？

### 3.2 主对比组

主实验只保留三组：

- `PPO baseline`
  单头策略，作为标准基线。

- `no_guidance`
  三头 adaptive-skill 结构，不加多样性引导。

- `weak_guidance`
  三头 adaptive-skill 结构，加 `diversity_loss`。

这三组已经足以回答主研究问题。  
更强的人为先验和工程技巧不纳入主对比轴。

### 3.3 控制变量

为保证对比干净，主实验中应固定：

- 相同的主干网络容量
- 相同的 PPO 超参数
- 相同的训练预算
- 相同的场景分布
- 相同的 gating 温度退火策略

主实验中真正变化的只有：

- 是否使用多头 adaptive skill
- 是否使用 `diversity_loss`

### 3.4 赛道与任务设置

建议采用障碍密度渐变的赛道，使不同技能的使用时机更容易被观察到：

- 开阔区域
- 稀疏障碍区域
- 密集障碍区域
- 再回到稀疏与开阔区域

这样能够同时覆盖：

- 常规导航
- 近障碍避障
- 恢复与稳定阶段

### 3.5 评估指标

任务性能指标：

- 完成率
- 平均完成时间
- 平均速度
- 碰撞率

分化质量指标：

- skill 间相关性
- 不同 skill 在各动作维度上的控制集中度
- 不同风险状态下的 skill 使用分布
- gating 熵随训练和场景变化的趋势

其中，最关键的不是“是否均匀使用所有 skill”，而是：

- 不同 skill 是否真的学出差异
- 不同场景下是否出现合理的技能切换

### 3.6 预期观察

预期结果如下：

- 单头 baseline 具备基本任务能力，但缺乏明显模式分工。
- `no_guidance` 可能出现一定程度的自发分化，但稳定性有限。
- `weak_guidance` 应该在分化质量和任务表现上都优于 `no_guidance`。

如果观察到以下现象，则说明方案方向成立：

- `yaw` 主导技能更多出现在可转向避障阶段
- `roll/pitch` 主导技能更多出现在高风险近障碍阶段
- `thrust` 主导技能更多出现在恢复、开阔或稳定段

---

## 4. 工程增强与扩展

以下内容不属于主方案结论依赖项，只作为后续工程增强或扩展消融方向。

### 4.1 Skill Bias

可以通过不同 head 的初始化偏置来打破对称性。  
这有助于加快早期分化，但属于人为增强，不纳入主研究轴。

### 4.2 防塌缩正则

如果训练中出现某个 head 长期几乎不被使用，可以引入很弱的防塌缩正则。  
它的作用只是防止“死头”，而不是追求所有 skill 全局均匀使用。

### 4.3 温度退火

温度退火被视为主实验中的共同训练设置，而不是可学习自由度。

当前方案选择：

- 固定的 `soft-to-sharp` 温度退火
- 不把温度作为可学习参数

后续实现时，可以复用现有训练框架中的退火机制按训练步数调度温度，但这属于实现细节，不影响方法本身的定义。

---

## 5. 论文层面的可表述结论

如果实验结果符合预期，本方案可以支持以下结论：

- 多技能策略结构有助于覆盖无人机敏捷飞行中的多模态控制需求。
- 即使不引入强先验，仅通过多头结构也可能出现一定程度的自发分化。
- 适度的 `diversity_loss` 能稳定提升 skill 分化质量与任务表现。
- `soft-to-sharp` gating 有助于在连续控制与明确选择之间取得平衡。

---

## 6. 成功标准

实验成功的最低标准可以定义为：

- 三组主实验都完成训练并可重复
- `weak_guidance` 相比 `no_guidance` 显示出更清晰的技能分化
- `weak_guidance` 相比 baseline 或 `no_guidance` 具有更好的任务指标
- skill 使用与场景风险之间存在可解释关联

---

## 7. 关键总结

- 这是一个以多头结构为先验的多模态策略方案，而不是显式层级规划器。
- 三个 skill 的 `yaw / roll-pitch / thrust` 分工是研究假设，不是训练硬标签。
- 主方案只依赖三件事：多头结构、`diversity_loss`、`soft-to-sharp gating`。
- 主实验只比较三组：单头 baseline、无引导三头、带多样性引导三头。
- 工程增强项可以做，但不应污染主研究结论。

---

## 参考文献

1. Haarnoja et al. "Composable Deep Reinforcement Learning for Robotic Manipulation." ICRA 2018.
2. Vezhnevets et al. "FeUdal Networks for Hierarchical Reinforcement Learning." ICML 2017.
3. Eysenbach et al. "Diversity is All You Need: Learning Skills without a Reward Function." ICLR 2018.
4. Schulman et al. "Proximal Policy Optimization Algorithms." arXiv 2017.
5. Henderson et al. "Deep Reinforcement Learning that Matters." AAAI 2018.

---

*文档版本：v3.0*
*最后更新：2026 年 4 月*
