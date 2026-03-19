from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_num_agents", [1]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    '--quads_neighbor_visible_num=0 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=final '
    '--quads_cbf_k_omega=0 '
    '--quads_use_cbf=True --quads_cbf_alpha_1=0.1  --quads_cbf_alpha_2=0.1  --quads_cbf_k_omega =0  --quads_cbf_epsilon=0.1 '
    '--quads_cbf_R_obs=0.5'
)

_experiment = Experiment(
    "alpha_1=0.5 alpha_2=0.5 k_w=0",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_cbf", experiments=[_experiment])