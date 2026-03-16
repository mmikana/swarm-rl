# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuadSwarm-RL is a reinforcement learning framework for training decentralized control policies for quadrotor swarms. It includes:
- A high-fidelity flight dynamics simulator (`gym_art/`) forked from gym_art and extended to support multi-agent swarms
- Integration with Sample Factory for distributed RL training
- Support for obstacle-dense environments with collision avoidance
- Multiple swarm coordination scenarios and architectures

**Current Research Focus**: The project is actively developing RCBF (Robust Control Barrier Function) integration for safe control, with detailed theoretical work documented in `docs/RCBF_理论推导.md`.

## Development Commands

### Installation

```bash
conda create -n swarm-rl python=3.11
conda activate swarm-rl
pip install -e .
```

### Training

**Using the default training script:**
```bash
bash train.sh
```

**Using runner scripts (recommended):**
```bash
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
```

**Using obstacle scenarios:**
```bash
python -m sample_factory.launcher.run --run=swarm_rl.runs.obstacles.quads_multi_obstacles --max_parallel=1 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1
```

Key training parameters:
- `--run`: Path to runner script (Python module path, e.g., `swarm_rl.runs.obstacles.quads_multi_obstacles`)
- `--max_parallel`: Number of parallel training runs
- `--experiments_per_gpu`: How many experiments to run per GPU
- `--num_gpus`: Total number of GPUs available

### Testing/Evaluation

```bash
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME --quads_view_mode topdown global chase
```

- `PATH_TO_TRAIN_DIR` and `EXPERIMENT_NAME` can be found in the `config.json` file of your trained model
- `--quads_view_mode`: Choose camera views from `[topdown, global, chase, side, corner0, corner1, corner2, corner3, topdownfollow]`

### Monitoring

```bash
cd train_dir_or_experiment_folder
tensorboard --logdir=./
```

**WandB Integration:**
```bash
wandb login  # First time setup
# Add --with_wandb=True --wandb_user=YOUR_USERNAME to training commands
```

### Unit Tests

```bash
./run_tests.sh
# Or directly:
python -m unittest
```

## Code Architecture

### Core Structure

```
gym_art/quadrotor_multi/     # Physics simulation and environment
├── quadrotor_multi.py       # Multi-agent environment wrapper
├── quadrotor_single.py      # Individual quadrotor simulator
├── quadrotor_dynamics.py    # Physics and dynamics model (Crazyflie)
├── quadrotor_control.py     # Control interfaces (RawControl, etc.)
├── scenarios/               # Task scenarios (goals, formations, etc.)
├── obstacles/               # Obstacle generation and SDF computation
├── collisions/              # Collision detection (drones, obstacles, walls)
└── aerodynamics/           # Downwash effects

swarm_rl/                    # RL training infrastructure
├── train.py                 # Main training entry point
├── enjoy.py                 # Evaluation/rendering script
├── models/                  # Neural network architectures
│   ├── quad_multi_model.py # Main encoder with attention/deepsets
│   └── attention_layer.py  # Multi-head attention for neighbors
├── env_wrappers/           # Environment utilities
│   ├── quad_utils.py       # Environment creation
│   ├── quadrotor_params.py # CLI argument definitions
│   └── reward_shaping.py   # Reward wrapper
└── runs/                   # Experiment configurations (runner scripts)
```

### Key Architectural Concepts

#### Multi-Agent Environment Design

The system uses a **decentralized architecture**:
- Each agent receives local observations (self state + neighbors + obstacles)
- A shared policy network outputs actions for all agents
- Agents coordinate implicitly through observation sharing, not communication

**Observation Structure** (defined in `quadrotor_params.py:20-22`):
- `quads_obs_repr`: Self observation space
  - `xyz_vxyz_R_omega`: Position, velocity, rotation matrix, angular velocity
  - `xyz_vxyz_R_omega_floor`: Above + floor distance
  - `xyz_vxyz_R_omega_wall`: Above + wall distances
- `quads_neighbor_obs_type`: Neighbor observations
  - `pos_vel`: Relative position and velocity of nearby agents
- `quads_obstacle_obs_type`: Obstacle observations
  - `octomap`: 3×3 SDF (Signed Distance Field) grid

#### Control Interface

**Action Space** (`quadrotor_control.py:30-66`):
- **Default**: `RawControl` with `raw_control_zero_middle=True`
- Actions are in range `[-1, 1]^4` (4 motor thrusts)
- Conversion to physical thrust: `thrust = 0.5 * (action + 1.0)` maps `[-1,1] → [0,1]`
- Physical parameters (Crazyflie): mass=0.028kg, max_thrust=0.130N per motor

#### Neighbor Encoder Architectures

The system supports multiple neighbor encoding schemes (`swarm_rl/models/quad_multi_model.py`):

1. **Attention** (`--quads_neighbor_encoder_type=attention`):
   - Multi-head attention mechanism from "Graph Neural Networks with Attention"
   - Learns importance weights for each neighbor
   - Outputs weighted sum of neighbor embeddings

2. **Mean Embedding** (`--quads_neighbor_encoder_type=mean_embed`):
   - DeepSets architecture (permutation invariant)
   - Embeds each neighbor independently, then averages

3. **MLP** (`--quads_neighbor_encoder_type=mlp`):
   - Simple concatenation + MLP
   - Requires fixed neighbor ordering

4. **No Encoder** (`--quads_neighbor_encoder_type=no_encoder`):
   - Blind agents (no neighbor information)

#### Scenarios and Tasks

Scenarios are defined in `gym_art/quadrotor_multi/scenarios/`:

**Obstacle-free scenarios:**
- `static_same_goal`: All agents go to same fixed goal
- `static_diff_goal`: Each agent has different fixed goal
- `dynamic_same_goal`, `dynamic_diff_goal`: Goals change over time
- `swap_goals`: Agents swap positions
- `swarm_vs_swarm`: Two teams with opposing objectives

**Obstacle scenarios** (`scenarios/obstacles/`):
- `o_static_same_goal`, `o_dynamic_same_goal`: Goal-reaching with obstacles
- `o_random`: Random goal generation in obstacle field
- `o_ep_rand_bezier`: Following Bezier curve trajectories

#### Obstacle Representation

**SDF (Signed Distance Field)** computation (`gym_art/quadrotor_multi/obstacles/utils.py`):
- 3×3 local grid centered on agent position
- Grid resolution: 0.1m (configurable)
- Values: distance to nearest obstacle surface (positive=safe, negative=collision)
- Gradient estimation from grid enables RCBF safety constraints

**Configuration** (`quadrotor_params.py:55-82`):
- `--quads_obst_density`: Obstacle density (0.0-1.0)
- `--quads_obst_size`: Obstacle diameter in meters
- `--quads_domain_random`: Enable domain randomization
- `--quads_obst_density_random`, `--quads_obst_size_random`: Randomize these parameters

### Sample Factory Integration

**Environment Registration** (`swarm_rl/train.py:16-19`):
```python
register_env("quadrotor_multi", make_quadrotor_env)
register_models()
```

**Configuration Flow**:
1. Runner scripts in `swarm_rl/runs/` define experiment parameters
2. `quadrotor_params.py` adds environment-specific arguments
3. `quad_utils.py:make_quadrotor_env_multi()` creates wrapped environment
4. `reward_shaping.py` applies reward transformations

**Key Training Parameters**:
- `--quads_num_agents`: Number of quadrotors (default: 8)
- `--quads_episode_duration`: Episode length in seconds (default: 15.0)
- `--quads_collision_reward`: Penalty for inter-agent collisions
- `--quads_obst_collision_reward`: Penalty for obstacle collisions
- `--anneal_collision_steps`: Gradually increase collision penalty over N steps

### RCBF Safety Layer (Experimental)

The project contains detailed RCBF theoretical work in `docs/RCBF_理论推导.md` for:
- Second-order barrier functions using SDF observations
- QP-based safety filters in thrust space
- Integration with RL policies for minimal intervention

**RCBF Parameters**:
- Collision radius: `quad_radius = 0.046m` (based on arm length)
- Obstacle radius: 0.15-0.5m depending on configuration
- CBF gains: `α₁, α₂ ∈ [1.0, 3.0]`

## Important Implementation Notes

### Coordinate Systems

- **Inertial frame**: `e₃ = [0, 0, 1]ᵀ` points upward (z-up convention)
- **Body frame**: z-axis points down (standard quadrotor convention)
- **Thrust direction**: `R @ e₃` gives thrust vector in inertial frame (upward when hovering)

### Observation and State Spaces

The codebase uses `gymnasium` (not the legacy `gym`) but maintains compatibility:
- Observation spaces defined in `QUADS_OBS_REPR` dictionary (`gym_art/quadrotor_multi/quad_utils.py`)
- Multi-agent observations are concatenated: `[self_obs, neighbor_obs₁, ..., neighbor_obsₙ, obstacle_obs]`
- Observation normalization happens in the policy encoder

### Collision Detection

Three collision types (`gym_art/quadrotor_multi/collisions/`):
1. **Agent-Agent**: Sphere-sphere collision with radius `collision_hitbox_radius * arm_length`
2. **Agent-Obstacle**: Uses precise SDF values from obstacle grid
3. **Agent-Boundary**: Room walls/ceiling/floor collisions

Collision penalties can be:
- Binary (`quadcol_bin`, `quadcol_bin_obst`)
- Smooth falloff (`collision_falloff_radius` > 0 enables distance-based penalty)

### Numba Acceleration

Set `--quads_use_numba=True` to enable JIT compilation for:
- Collision detection
- Downwash calculations
- Obstacle SDF computation

**Warning**: First run will be slow due to compilation. Disable for debugging or rendering.

### Domain Randomization

When enabled (`--quads_domain_random=True`), the system randomizes:
- Quadrotor dynamics (thrust noise, damping)
- Obstacle density and sizes (if `_random` flags enabled)
- Sensor noise levels

This improves sim-to-real transfer (see Sim2Real papers in README).

## Typical Workflow

### 1. Develop New Scenario

Create scenario class in `gym_art/quadrotor_multi/scenarios/` inheriting from `ScenarioBase`:
```python
from scenarios.base import ScenarioBase

class MyScenario(ScenarioBase):
    def reset(self, env):
        # Set initial positions, goals
        pass

    def update(self, env):
        # Update goals, check termination
        pass
```

Add to `scenarios/mix.py:create_scenario()` function.

### 2. Create Runner Script

Add experiment configuration in `swarm_rl/runs/`:
```python
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid([
    ('quads_num_agents', [4, 8, 16]),
    ('seed', [1111, 2222, 3333]),
])

venv = RunDescription('my_experiment', experiments=[Experiment('baseline', 'python -m swarm_rl.train', _params.generate_params(randomize=False))])
```

### 3. Train and Monitor

```bash
python -m sample_factory.launcher.run --run=swarm_rl.runs.my_module.venv --max_parallel=3 --num_gpus=1
tensorboard --logdir=train_dir/my_experiment/
```

### 4. Evaluate

```bash
python -m swarm_rl.enjoy --train_dir=train_dir/my_experiment --experiment=baseline_00 --quads_render=True --quads_view_mode topdown chase
```

## Related Subproject

**Mod-RL-RCBF** (`Mod-RL-RCBF/`): Standalone SAC-RCBF implementation for lower-dimensional systems (Unicycle, PVTOL, SimulatedCars). Uses differentiable CBF-QP layer with Gaussian Process disturbance learning. See its own `CLAUDE.md` for details.

## Troubleshooting

**Import errors**: Ensure you installed with `pip install -e .` from repository root.

**Rendering issues**: Set `--quads_use_numba=False` when using `--quads_render=True`.

**OOM errors**: Reduce `--num_workers`, `--batch_size`, or `--num_envs_per_worker`.

**Slow training**: Enable `--quads_use_numba=True` (but compile time on first run).

**Collision detection seems wrong**: Check `--quads_collision_hitbox_radius` (multiplier of arm length, default 2.0).
