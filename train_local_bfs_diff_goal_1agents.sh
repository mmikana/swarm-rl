#!/usr/bin/env bash
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.local_bfs_diff_goal_1agents --backend=processes --max_parallel=1 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1
