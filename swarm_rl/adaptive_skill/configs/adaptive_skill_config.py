"""
Adaptive Skill RL 配置参数
"""


def add_adaptive_skill_args(parser):
    """
    添加 Adaptive Skill 参数到 argparse
    
    Args:
        parser: argparse.ArgumentParser
    """
    # 技能配置
    parser.add_argument(
        '--quads_num_skills',
        default=3,
        type=int,
        help='Number of skills to learn (default: 3)'
    )
    
    parser.add_argument(
        '--quads_use_adaptive_skill',
        default=True,
        type=bool,
        help='Whether to use adaptive skill policy'
    )
    
    parser.add_argument(
        '--quads_use_skill_bias',
        default=True,
        type=bool,
        help='Whether to use skill-specific bias'
    )
    
    # 损失配置
    parser.add_argument(
        '--diversity_loss_weight',
        default=0.5,
        type=float,
        help='Weight for diversity loss (default: 0.5)'
    )
    
    parser.add_argument(
        '--balance_loss_weight',
        default=0.1,
        type=float,
        help='Weight for skill balance loss (default: 0.1)'
    )
    
    # 选择器配置
    parser.add_argument(
        '--gating_temperature',
        default=1.0,
        type=float,
        help='Initial gating temperature (default: 1.0)'
    )
