"""
PPO baseline with global BFS guidance reward configured for GPU training.

Usage:
    python -m sample_factory.launcher.run --run=swarm_rl.runs.adaptive_skill.quads_guidance_global_bfs_gpu
"""

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8


_params = ParamGrid(
    [
        ("seed", [0]),
        ("quads_num_agents", [1]),
    ]
)


GUIDANCE_GLOBAL_BFS_GPU_CLI = (
    QUAD_BASELINE_CLI_8.replace("--device=cpu", "--device=gpu")
    .replace("--num_envs_per_worker=2", "--num_envs_per_worker=4")
    .replace("--batch_size=1024", "--batch_size=3072 --num_batches_per_epoch=2 --num_epochs=1")
) + (
    "--quads_mode=o_skill_hybrid --quads_room_dims 10 16 10 --quads_obst_spawn_area 10 16 "
    "--quads_guidance_type=global_bfs "
    "--quads_neighbor_visible_num=0 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention "
    "--with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones "
    "--wandb_group=adaptive_skill "
    "--quads_use_adaptive_skill=False "
    "--quads_use_diversity_loss=False "
    "--quads_action_type=omegathrust"
)


_experiment = Experiment(
    "new_scenario_guidance_global_bfs_gpu",
    GUIDANCE_GLOBAL_BFS_GPU_CLI,
    _params.generate_params(randomize=False),
)


RUN_DESCRIPTION = RunDescription("adaptive_skill_global_bfs_gpu", experiments=[_experiment])
