"""
评估并渲染 AdaptiveSkillPolicy

运行：
python eval_adaptive_skill.py
"""

import sys
import torch
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.enjoy import enjoy
import gymnasium as gym


def make_pendulum(full_env_name, cfg, env_config, render_mode=None):
    """创建 Pendulum 环境"""
    return gym.make("Pendulum-v1", render_mode=render_mode)


def make_adaptive_skill_actor_critic(cfg, obs_space, action_space):
    """创建 AdaptiveSkillPolicy"""
    from sample_factory.algo.utils.context import global_model_factory
    from swarm_rl.adaptive_skill.models.adaptive_skill_model import AdaptiveSkillPolicy

    model_factory = global_model_factory()
    return AdaptiveSkillPolicy(model_factory, obs_space, action_space, cfg)


def main():
    # 注册环境和模型
    register_env("pendulum", make_pendulum)
    global_model_factory().make_actor_critic_func = make_adaptive_skill_actor_critic

    # 配置参数
    argv = [
        "--algo=APPO",
        "--env=pendulum",
        "--experiment=test_adaptive_skill",
        "--train_dir=./test_train_dir",

        # 评估参数
        "--max_num_episodes=5",
        "--max_num_frames=1000",
        "--fps=30",

        # 模型参数（需要与训练时一致）
        "--quads_use_adaptive_skill=True",
        "--quads_num_skills=3",  # 改为 3 个技能
        "--adaptive_stddev=False",
        "--initial_stddev=1.0",
        "--encoder_mlp_layers=128",
        "--encoder_mlp_layers=128",
        "--use_rnn=False",
        "--device=cpu",
    ]

    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=True)
    parser.add_argument('--quads_use_adaptive_skill', type=bool, default=True)
    parser.add_argument('--quads_num_skills', type=int, default=1)
    cfg = parse_full_cfg(parser, argv)

    # 运行评估
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())

