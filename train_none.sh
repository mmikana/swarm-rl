#!/usr/bin/env bash
python -m sample_factory.launcher.run --run=swarm_rl.runs.adaptive_skill.quads_guidance_none --backend=processes --max_parallel=1 --pause_between=1 --experiments_per_gpu=0 --num_gpus=0
