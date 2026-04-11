"""
Single-agent PPO with local BFS guidance reward in diff-goal environment.

Usage:
    python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.local_bfs_diff_goal_1agents
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
    guidance_type="local_bfs",
    quads_mode="o_skill_hybrid_diff_goal",
    neighbor_visible_num=0,
    gpu_profile="safe",
)


_experiment = Experiment(
    "local_bfs_diff_goal_1agents",
    CLI,
    _params.generate_params(randomize=False),
)


RUN_DESCRIPTION = RunDescription("bfs", experiments=[_experiment])
