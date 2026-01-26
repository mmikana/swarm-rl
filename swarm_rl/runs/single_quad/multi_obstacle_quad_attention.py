from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid


_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

multi_obstacle_quad_attention_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --device=cpu --train_for_env_steps=30000000 --algo=APPO --use_rnn=false '
    '--num_workers=12 --num_envs_per_worker=2 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=True --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 --with_pbt=False '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=false --quads_num_agents=1 --quads_mode=static_same_goal --quads_episode_duration=15.0 '
    '--quads_neighbor_encoder_type=attention --quads_neighbor_hidden_size=256 --quads_neighbor_obs_type=pos_vel '
    '--quads_neighbor_visible_num=0 --replay_buffer_sample_prob=0.75 --anneal_collision_steps=30000000 '
    '--normalize_input=False --normalize_returns=False --reward_clip=10.0 --save_milestones_sec=3600 '
    '--quads_use_obstacles=False --quads_obst_spawn_area 8 8 --quads_obst_density=0.2 --quads_obst_size=0.6 '
    '--quads_obst_collision_reward=5.0 --quads_obstacle_obs_type=none --quads_use_downwash=True '
    '--with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_group=obstacle_with_replay --wandb_user=multi-drones'
)

_experiment = Experiment(
    'test',
    multi_obstacle_quad_attention_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('reward_test', experiments=[_experiment])