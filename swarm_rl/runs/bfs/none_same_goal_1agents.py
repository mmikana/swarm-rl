"""
Single-agent PPO baseline with Euclidean reward only in same-goal environment.

Usage:
    python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.none_same_goal_1agents
"""

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

from swarm_rl.runs.bfs.common import build_bfs_cli


_params = ParamGrid(
    [
        ("seed", [0]),
        ("quads_num_agents", [1]),
    ]
)


CLI = build_bfs_cli(
    guidance_type="none",
    quads_mode="o_skill_hybrid_same_goal",
    neighbor_visible_num=0,
    gpu_profile="safe",
)


_experiment = Experiment(
    "none_same_goal_1agents",
    CLI,
    _params.generate_params(randomize=False),
)


RUN_DESCRIPTION = RunDescription("bfs", experiments=[_experiment])
