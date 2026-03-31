"""
Adaptive Skill RL 配置参数

添加到 quadrotor_params.py 中
"""

from sample_factory.utils.utils import str2bool


def add_adaptive_skill_args(env, parser):
    """添加 Adaptive Skill RL 相关参数"""
    p = parser
    
    # ========== 技能架构配置 ==========
    p.add_argument('--quads_num_skills', default=4, type=int,
                   help='Number of skills to learn (default: 4)')
    
    p.add_argument('--quads_skill_hidden_size', default=256, type=int,
                   help='Hidden size for skill heads')
    
    # ========== 训练配置 ==========
    p.add_argument('--quads_use_diversity_loss', default=True, type=str2bool,
                   help='Whether to use diversity loss to encourage skill differentiation')
    
    p.add_argument('--quads_diversity_loss_weight', default=0.1, type=float,
                   help='Weight for diversity loss')
    
    # ========== 技能特定奖励权重 ==========
    # Skill 1: 高速巡航
    p.add_argument('--quads_skill1_speed_weight', default=1.0, type=float,
                   help='Speed reward weight for skill 1 (high-speed cruise)')
    
    # Skill 2: 紧急避障
    p.add_argument('--quads_skill2_safety_weight', default=1.0, type=float,
                   help='Safety reward weight for skill 2 (obstacle avoidance)')
    
    # Skill 3: 姿态恢复
    p.add_argument('--quads_skill3_stability_weight', default=1.0, type=float,
                   help='Stability reward weight for skill 3 (recovery)')
    
    # Skill 4: 精确穿越
    p.add_argument('--quads_skill4_precision_weight', default=1.0, type=float,
                   help='Precision reward weight for skill 4 (precision navigation)')
    
    # ========== 可视化配置 ==========
    p.add_argument('--quads_visualize_skills', default=False, type=str2bool,
                   help='Whether to visualize skill weights during evaluation')
