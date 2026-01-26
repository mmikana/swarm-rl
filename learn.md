# 2026/1/6
## 奖励退火
- 基本概念：奖励退火是一种渐进式奖励调整策略，通过在训练过程中动态调整奖励系数，使智能体能够逐步适应更严格的约束条件。
- 在训练初期，agent行为随机、频繁碰撞使得很难学会基本飞行、容易陷入局部最优，因此通过动态调整奖励权重，使得agent在训练初期的碰撞惩罚较小，随着epi加大权重，使agent在早期注重学习基本飞行，在后期又学会避障。
- 数学表达式：collision_penalty(t) = min(final_penalty × (current_steps / anneal_steps),final_penalty)
- 曲线示例：
    奖励系数
     ↑
     |        /
     |       /
     |      /  ← 线性增长
     |     /
     |    /
     |   /
     |  /
     | /
     |/_________________________→ 训练步数
     0                    anneal_steps
- 优势：训练稳定、渐进式学习、避免局部最优

# 2026/1/7
## attention
- why attention？
    - long horizon：rnn处理长序列时，发生梯度消失和爆炸，难以捕捉远距离的信息关联；
    - bad parallel computation：rnn串行计算，训练效率低；
- what is attention mechanism？
    - 对于当前输入(Query)，计算Q与其他输入(K)的关联程度，使用关联程度对值向量(Value)加权求和，得到注意力输出。
    - Attention(Q,K,V)=softmax( Q·K^T / square(d_k) ) · V
- how to implement attention?
    - Input: x, d_m, d_k
    - QKV : Q = x · w_q     K = x · w_k     V = x · w_v     w ∈ R^( d_m × d_k)
    - corresponding matrix: Q·K^T
    - output: Attention(Q,K,V)=softmax( Q·K^T / square(d_k) ) · V
- multi-head attention
    - MultiHeadAtten(Q,K,V) = Concat(head_1,……，head_n) W^o
                            head_i = Attention(Q_i,K_i,V_i)
- detail
    - 缩放点积的核心作用是避免点积结果过大导致 softmax 函数梯度消失，从而保证注意力权重的合理分布;

# 2026/1/8
## para
- recurrence、adaptive_stddev、with_vtrace
- costume env
    - gym环境：定义观测和状态空间、reset、step、render、_update_state、_get_observation、_calculate_reward、_check_termination；
    - 注册环境：register_my_env()、make_my_env(env_name, cfg=None, **kwargs)
    - costume encoder

# 2026/1/14
## xavier_uniform
- xavier_uniform 是由 Glorot 和 Bengio 在 2010 年提出的权重初始化方法，核心目标是解决神经网络训练中的梯度消失 / 爆炸问题。给神经网络的权重 “设定合适的初始值范围”，让每一层的输入和输出的方差尽可能保持一致，就像给水管设计合适的管径，让水流（梯度）在各层间顺畅流动，既不会太细（梯度消失）也不会太粗（梯度爆炸）。
- xavier_uniform会将权重初始化为均匀分布[-limit,limit]，其中limit=square(6 / (f_an_in + f_an_out)),f_an_in为当前层的输入特征数，f_an_out为当前层输出特征数量。
- 适配激活函数：主要用于 sigmoid、tanh 这类对称且输出均值接近 0 的激活函数；
- 如果使用 ReLU、LeakyReLU 等 ReLU 类激活函数，更适合用 He 初始化（kaiming_uniform），因为 ReLU 会导致一半的神经元输出为 0，xavier_uniform 的方差适配性会变差。

## summary
- 这个项目只能用cpu训练了，cuda会发生内存泄漏的问题，暂时没法修复。我修改了baseline.py和quad_multi_mix_baseline.py，使用cpu进行训练，可以较好的完成任务。后继可以做一些对比实验，看看各个模块的作用，如attention、rnn、以及一些网络结构。