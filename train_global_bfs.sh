#!/usr/bin/env bash
python -m sample_factory.launcher.run --run=swarm_rl.runs.adaptive_skill.quads_guidance_global_bfs_gpu --backend=processes --max_parallel=1 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1
