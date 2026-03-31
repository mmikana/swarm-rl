"""
Diversity Loss Hook for Adaptive Skill RL

这个模块提供多样性损失计算，通过 Sample Factory 的 batch 机制集成
"""

import torch
import torch.nn.functional as F


def compute_diversity_loss(skill_actions: torch.Tensor) -> torch.Tensor:
    """
    计算技能多样性损失
    
    通过余弦相似度鼓励技能差异化
    
    Args:
        skill_actions: [batch, num_skills, action_dim]
    
    Returns:
        diversity_loss: 标量
    """
    num_skills = skill_actions.shape[1]
    
    if num_skills < 2:
        return torch.tensor(0.0, device=skill_actions.device)
    
    # 计算技能对之间的余弦相似度
    similarities = []
    for i in range(num_skills):
        for j in range(i + 1, num_skills):
            sim = F.cosine_similarity(
                skill_actions[:, i],
                skill_actions[:, j],
                dim=-1
            )
            similarities.append(sim)
    
    # 平均相似度
    avg_sim = torch.mean(torch.stack(similarities)) if similarities else 0
    
    # 多样性损失 = -平均相似度
    diversity_loss = -avg_sim
    
    return diversity_loss
