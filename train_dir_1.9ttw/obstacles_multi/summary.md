# single_pos(75M) train_dir/obstacles_multi/final_/00_final_see_0_q.n.age_1  quads_multi_obstacles.py
- demo现象：无人机可以较好到达目标位置，但在面对障碍物的时候，存在一定的急刹车，表面agent学会避障。
- 训练奖励：0-25M奖励逐步上升到达最大值8，25M-75M逐步下降到达3.5左右，这可能跟无人机的出生位置有关系，因为demo暂且可以完成任务。
- stats现象：0-25M和25M-75M出现了很多现象，master_process_memory_mb、memory_learner、memroy_policy_worker都发生了较大的变化。
- stats可能原因：0-25M作为训练初期，系统正在填充经验回放缓冲区，这需要额外的内存来存储状态、动作、奖励等数据；神经网络权重和其他参数的初始化可能会导致内存使用量的波动；多个工作进程（policy workers, learner等）的启动和初始化会增加内存使用；25M-75M 阶段是训练中期，在这个阶段，系统可能正在大量使用经验回放，需要同时维护当前经验和历史经验；定期保存模型检查点（checkpoints）可能导致临时内存使用增加；随着训练的进行，智能体的探索行为可能变得更加复杂，导致更密集的计算和内存使用。
std：在0-25Msteps下降较快，在25-75Msteps下降较慢。