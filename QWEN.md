# QuadSwarm-RL Project Context

## Project Overview

QuadSwarm-RL is a reinforcement learning codebase for training control policies for quadrotor swarms. It includes:
- A flight dynamics simulator forked from `gym_art` and extended to support swarms of quadrotor drones
- Scripts and wrappers to facilitate training of control policies with Sample Factory (https://github.com/alex-petrenko/sample-factory)

The project is designed for research on decentralized control of quadrotor swarms using end-to-end deep reinforcement learning, with capabilities for obstacle avoidance, navigation, and various swarm behaviors.

## Architecture

### Core Components
- `gym_art/`: Contains the quadrotor dynamics simulator (forked and extended from the original gym_art)
- `swarm_rl/`: Main RL training and evaluation code
  - `env_wrappers/`: Environment wrappers for compatibility and reward shaping
  - `models/`: Neural network architectures including attention mechanisms
  - `runs/`: Experiment configuration files for different scenarios
  - `sim2real/`: Sim-to-real transfer components
- `train.sh`: Main training script
- `run_tests.sh`: Unit test runner

### Key Technologies
- Python 3.11+
- PyTorch
- Sample Factory (reinforcement learning framework)
- Numba (for performance optimization)
- Various ML libraries (numpy, gym, transforms3d, etc.)

## Training and Running

### Installation
```bash
conda create -n swarm-rl python=3.11
conda activate swarm-rl
pip install -e .
```

### Training
The project uses Sample Factory for training. To run training:
```bash
bash train.sh
```

Or use the runner scripts in `swarm_rl/runs/`:
```bash
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
```

### Testing/Evaluation
To test a trained model:
```bash
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME --quads_view_mode CAMERA_VIEWS
```

### Unit Tests
Run unit tests with:
```bash
./run_tests.sh
```

## Key Features

### Environment Configuration
- Multiple swarm scenarios (static_same_goal, dynamic_diff_goal, swarm_vs_swarm, etc.)
- Configurable number of agents (default 8)
- Obstacle avoidance capabilities
- Various observation representations
- Neighbor encoding options (attention, mean_embed, mlp, no_encoder)

### Neural Network Models
- Multi-head attention mechanisms for swarm coordination
- Custom encoders for processing self, neighbor, and obstacle observations
- Support for different encoder types (attention, deepsets, MLP)

### Training Parameters
- APPO (Adaptive Proximal Policy Optimization) algorithm
- Configurable collision penalties with annealing
- Domain randomization capabilities
- Support for different reward shaping schemes

## Development Conventions

### Code Structure
- Training logic in `swarm_rl/train.py`
- Evaluation logic in `swarm_rl/enjoy.py`
- Environment creation in `swarm_rl/env_wrappers/quad_utils.py`
- Model definitions in `swarm_rl/models/quad_multi_model.py`
- Experiment configurations in `swarm_rl/runs/`

### Configuration
- Command-line arguments are managed through `swarm_rl/env_wrappers/quadrotor_params.py`
- Experiment parameters are defined in the `swarm_rl/runs/` directory
- Default values can be overridden in the configuration files

### Testing
- Unit tests are run with the standard Python unittest framework
- Tests are executed via the `run_tests.sh` script

## Key Papers

The project is associated with several research papers:
- QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control (ICRA Workshop 2023)
- Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors (IROS 2019)
- Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning (CoRL 2021)
- Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning (ICRA 2024)
- HyperPPO: A scalable method for finding small policies for robotic control (ICRA 2024)
- Tiny End-to-End Navigation for Quadrotor Swarms (ICRA 2025 Workshop)
- Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation (Under Review)

## Important Files and Directories

- `README.md`: Main project documentation with installation and usage instructions
- `setup.py`: Project dependencies and installation configuration
- `train.sh`: Default training script
- `run_tests.sh`: Test execution script
- `swarm_rl/runs/`: Experiment configuration files
- `swarm_rl/models/`: Neural network model implementations
- `swarm_rl/env_wrappers/`: Environment wrappers and utilities
- `gym_art/quadrotor_multi/`: Core quadrotor simulation environment

## Qwen Added Memories
- swarm-rl
- 20260107  qwen --resume f7be4949-c50d-42d9-8555-533608f88e8a
- 20260108 qwen --resume f7be4949-c50d-42d9-8555-533608f88e8a
- QuadSwarm-RL 项目研究：单无人机 RCBF（鲁棒控制屏障函数）安全控制。推力空间 RCBF-QP 公式已推导完成 - 直接在 4 维电机推力空间求解 QP，输出与 RL 策略对齐。核心创新：A_obs(x) = (2T_max/m)(p-p_obs)ᵀRe₃1ᵀ，完整约束矩阵公式已文档化于 docs/RCBF_理论推导.md
rcbf qwen --resume 25e1b023-cd07-4c14-a74f-ae1d84ba9116
controller_type qwen --resume 2a8caaf4-8f86-4685-b48c-29f9b59c2403