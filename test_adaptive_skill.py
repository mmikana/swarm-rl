"""
测试 AdaptiveSkillPolicy 在简单环境中是否能正常工作

使用 LunarLander-v2 快速验证：
- 网络结构是否正确
- 是否能正常训练收敛
- 技能是否能分化

运行：
python test_adaptive_skill.py
"""

import sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.algo.utils.context import global_model_factory
import gymnasium as gym


def make_pendulum(full_env_name, cfg, env_config, render_mode=None):
    """创建 Pendulum 环境（连续动作空间）"""
    import gymnasium as gym
    return gym.make("Pendulum-v1", render_mode=render_mode)


def make_adaptive_skill_actor_critic(cfg, obs_space, action_space):
    """创建 AdaptiveSkillPolicy"""
    from sample_factory.algo.utils.context import global_model_factory
    from swarm_rl.adaptive_skill.models.adaptive_skill_model import AdaptiveSkillPolicy

    model_factory = global_model_factory()
    return AdaptiveSkillPolicy(model_factory, obs_space, action_space, cfg)


def main():
    # 注册环境
    register_env("pendulum", make_pendulum)

    # 注册模型
    global_model_factory().make_actor_critic_func = make_adaptive_skill_actor_critic

    # 配置参数
    argv = [
        "--algo=APPO",
        "--env=pendulum",
        "--experiment=test_adaptive_skill_singlehead",
        "--train_dir=./test_train_dir",

        # 基础训练参数
        "--num_workers=12",
        "--num_envs_per_worker=2",
        "--batch_size=256",
        "--learning_rate=0.0001",
        "--train_for_env_steps=10000000000",
        "--device=cpu",  # 使用 CPU 避免 CUDA 问题

        # Adaptive Skill 参数
        "--quads_use_adaptive_skill=True",
        "--quads_num_skills=1",
        "--adaptive_stddev=False",
        "--initial_stddev=1.0",

        # 网络参数
        "--encoder_mlp_layers=128",
        "--encoder_mlp_layers=128",
        "--use_rnn=False",

        # 日志
        "--with_wandb=False",
    ]

    parser, partial_cfg = parse_sf_args(argv=argv)

    # 添加自定义参数
    parser.add_argument('--quads_use_adaptive_skill', type=bool, default=True)
    parser.add_argument('--quads_num_skills', type=int, default=3)

    cfg = parse_full_cfg(parser, argv)

    # 运行训练
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
