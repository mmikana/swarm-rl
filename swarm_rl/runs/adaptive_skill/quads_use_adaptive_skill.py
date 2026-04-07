"""
Adaptive Skill RL 运行配置

使用方式:
    python -m sample_factory.launcher.run --run=swarm_rl.runs.adaptive_skill.quads_use_adaptive_skill
"""

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8


# 参数网格
_params = ParamGrid(
    [
        ("seed", [0000]),
        ("quads_num_agents", [1]),
    ]
)

# 基础命令行（继承自 obstacle baseline）
# 当前配置用于单头 PPO baseline，并去掉 orient / spin 对机动的直接压制
ADAPTIVE_SKILL_CLI = QUAD_BASELINE_CLI_8 + (
    '--quads_mode=o_skill_hybrid --quads_room_dims 10 16 10 --quads_obst_spawn_area 8 16 '
    '--quads_neighbor_visible_num=0 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=adaptive_skill '
    '--quads_use_adaptive_skill=False '
    '--quads_use_diversity_loss=False '
    '--quads_action_type=omegathrust'
)


# 实验定义
_experiment = Experiment(
    "test_new_scenario",
    ADAPTIVE_SKILL_CLI,
    _params.generate_params(randomize=False),
)


RUN_DESCRIPTION = RunDescription("adaptive_skill", experiments=[_experiment])
