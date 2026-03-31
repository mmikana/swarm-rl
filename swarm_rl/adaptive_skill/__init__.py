"""
Adaptive Skill RL 模块

多技能自适应强化学习，通过共享编码器 + 多技能头 + 选择器实现
"""

from .models import register_adaptive_skill_model

__all__ = [
    'register_adaptive_skill_model',
]
