# BFS Experiment Plan

## Goal

This document organizes the BFS-related experiment groups for the current paper direction.
The core claim is:

1. Pure distance-based RL reward is easily misled in concave or offset-gate environments.
2. Global BFS guidance provides topology-aware training signal and improves traversability.
3. The same idea can be extended from single-UAV navigation to multi-UAV navigation.

## Recommended Experiment Matrix

| ID | Name | Environment | Guidance | Agents | Neighbor Visible | Status | Main Use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | `none_same_goal_1agents` | `o_skill_hybrid_same_goal` | `none` | 1 | 0 | ready | single-agent baseline |
| E2 | `global_bfs_same_goal_1agents` | `o_skill_hybrid_same_goal` | `global_bfs` | 1 | 0 | ready | single-agent main result |
| E3 | `local_bfs_same_goal_1agents` | `o_skill_hybrid_same_goal` | `local_bfs` | 1 | 0 | ready with small cleanup suggested | weak-privilege comparison |
| E4 | `none_diff_goal_1agents` | `o_skill_hybrid_diff_goal` | `none` | 1 | 0 | ready | explicit diff-goal single-agent control |
| E5 | `global_bfs_diff_goal_1agents` | `o_skill_hybrid_diff_goal` | `global_bfs` | 1 | 0 | ready | explicit diff-goal single-agent control |
| E6 | `local_bfs_diff_goal_1agents` | `o_skill_hybrid_diff_goal` | `local_bfs` | 1 | 0 | ready | explicit diff-goal single-agent control |
| E7 | `none_diff_goal_3agents` | `o_skill_hybrid_diff_goal` | `none` | 3 | 2 | ready | multi-agent baseline |
| E8 | `global_bfs_diff_goal_3agents` | `o_skill_hybrid_diff_goal` | `global_bfs` | 3 | 2 | ready | multi-agent main result |
| E9 | `none_diff_goal_6agents` | `o_skill_hybrid_diff_goal` | `none` | 6 | 5 | ready | dense multi-agent baseline |
| E10 | `global_bfs_diff_goal_6agents` | `o_skill_hybrid_diff_goal` | `global_bfs` | 6 | 5 | ready | dense multi-agent main result |

## Minimum Set For Paper

If time is limited, run these first:

1. `none_same_goal_1agents`
2. `global_bfs_same_goal_1agents`
3. `none_diff_goal_6agents`
4. `global_bfs_diff_goal_6agents`
5. `local_bfs_same_goal_1agents`

This minimum set is enough to support:

1. single-agent effectiveness
2. multi-agent scalability
3. comparison between global BFS and local BFS

## Current Launcher Mapping

The following launcher files already exist in this folder:

| File | Current Meaning | Current Status |
| --- | --- | --- |
| `none_same_goal_1agents.py` | `none`, `1 agent`, `o_skill_hybrid_same_goal` | ready |
| `global_bfs_same_goal_1agents.py` | `global_bfs`, `1 agent`, `o_skill_hybrid_same_goal` | ready |
| `local_bfs_same_goal_1agents.py` | `local_bfs`, `1 agent`, `o_skill_hybrid_same_goal` | ready |
| `none_diff_goal_1agents.py` | `none`, `1 agent`, `o_skill_hybrid_diff_goal` | ready |
| `global_bfs_diff_goal_1agents.py` | `global_bfs`, `1 agent`, `o_skill_hybrid_diff_goal` | ready |
| `local_bfs_diff_goal_1agents.py` | `local_bfs`, `1 agent`, `o_skill_hybrid_diff_goal` | ready |
| `none_diff_goal_3agents.py` | `none`, `3 agents`, `o_skill_hybrid_diff_goal` | ready |
| `global_bfs_diff_goal_3agents.py` | `global_bfs`, `3 agents`, `o_skill_hybrid_diff_goal` | ready |
| `none_diff_goal_6agents.py` | `none`, `6 agents`, `o_skill_hybrid_diff_goal` | ready |
| `global_bfs_diff_goal_6agents.py` | `global_bfs`, `6 agents`, `o_skill_hybrid_diff_goal` | ready |

Current direct run commands:

```bash
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.none_same_goal_1agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.global_bfs_same_goal_1agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.local_bfs_same_goal_1agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.none_diff_goal_1agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.global_bfs_diff_goal_1agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.local_bfs_diff_goal_1agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.none_diff_goal_3agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.global_bfs_diff_goal_3agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.none_diff_goal_6agents
python -m sample_factory.launcher.run --run=swarm_rl.runs.bfs.global_bfs_diff_goal_6agents
```

## Naming Rule

The launcher and shell script names follow:

```text
{guidance_type}_{goal_mode}_{num_agents}agents
```

Examples:

```text
none_same_goal_1agents
global_bfs_same_goal_1agents
local_bfs_same_goal_1agents
none_diff_goal_1agents
global_bfs_diff_goal_1agents
local_bfs_diff_goal_1agents
none_diff_goal_3agents
global_bfs_diff_goal_3agents
none_diff_goal_6agents
global_bfs_diff_goal_6agents
```

The paired shell script names follow:

```text
train_{guidance_type}_{goal_mode}_{num_agents}agents.sh
```

Current shell scripts:

```text
train_none_same_goal_1agents.sh
train_global_bfs_same_goal_1agents.sh
train_local_bfs_same_goal_1agents.sh
train_none_diff_goal_1agents.sh
train_global_bfs_diff_goal_1agents.sh
train_local_bfs_diff_goal_1agents.sh
train_none_diff_goal_3agents.sh
train_global_bfs_diff_goal_3agents.sh
train_none_diff_goal_6agents.sh
train_global_bfs_diff_goal_6agents.sh
```

## Suggested Metrics

Report these metrics for every experiment:

1. success rate
2. collision rate
3. deadlock rate
4. average episode length
5. average trajectory length
6. normalized path cost

Normalized path cost is recommended as:

```text
actual trajectory length / BFS shortest feasible path length
```

For multi-agent experiments, also report:

1. per-agent success rate
2. all-agents success rate
3. inter-agent collision rate
4. throughput per episode

## Figures And Tables For Paper

Recommended paper artifacts:

1. Table 1: single-agent `none` vs `global_bfs` vs `local_bfs`
2. Table 2: multi-agent `none` vs `global_bfs` under `3 agents` and `6 agents`
3. Figure 1: BFS field visualization on the offset-gate map
4. Figure 2: representative trajectories for successful and failed runs
5. Figure 3: training curves of success, collision, and deadlock

## Recommended Execution Order

Run in this order:

1. `none_same_goal_1agents`
2. `global_bfs_same_goal_1agents`
3. `local_bfs_same_goal_1agents`
4. `none_diff_goal_6agents`
5. `global_bfs_diff_goal_6agents`
6. `none_diff_goal_3agents`
7. `global_bfs_diff_goal_3agents`
8. optional ablations after adding an actual no-randomization switch

This order prioritizes the experiments that are most likely to produce usable paper figures quickly.

## Notes

1. Single-agent main experiments should use `o_skill_hybrid_same_goal` for semantic clarity.
2. Multi-agent experiments should use `o_skill_hybrid_diff_goal` to avoid all agents being pulled toward the same target.
3. If local BFS remains unstable, move it to ablation or appendix instead of the main result section.
4. If seeds are limited, use at least `3` seeds for the main tables.
5. `global_bfs_6agents_no_rand` is not materialized as a launcher yet, because the environment does not currently expose a real switch that disables topology randomization.
6. The `*_diff_goal_1agents` launchers are added for naming clarity and explicit environment control. With `1` agent, `same_goal` and `diff_goal` are nearly equivalent in semantics because there is only one target.
