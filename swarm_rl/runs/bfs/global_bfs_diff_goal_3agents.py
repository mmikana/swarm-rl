"""
Three-agent PPO with global BFS guidance reward in diff-goal environment.

Usage:
    python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.global_bfs_diff_goal_3agents
"""

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

from swarm_rl.runs.bfs.common import build_bfs_cli


_params = ParamGrid(
    [
        ("seed", [0]),
        ("quads_num_agents", [3]),
    ]
)


CLI = build_bfs_cli(
    guidance_type="global_bfs",
    quads_mode="o_skill_hybrid_diff_goal",
    neighbor_visible_num=2,
    gpu_profile="safe",
)


_experiment = Experiment(
    "global_bfs_diff_goal_3agents",
    CLI,
    _params.generate_params(randomize=False),
)


RUN_DESCRIPTION = RunDescription("bfs", experiments=[_experiment])
