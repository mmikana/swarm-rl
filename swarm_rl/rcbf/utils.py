"""
RCBF Utility Functions

Helper functions for state extraction, SDF gradient computation, etc.

These are pure functions (stateless) that provide common utilities
for the CBF-QP layer.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Union


def extract_state_from_obs(obs: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    从观测中提取 RCBF 需要的状态
    
    Args:
        obs: 环境观测，形状为 (batch_size, obs_dim) 或 (obs_dim,)
             obs 结构：[pos_rel(3), vel(3), rot(9), omega(3), sdf(9), ...]
    
    Returns:
        state: 状态字典
            - 'pos': 位置 (batch_size, 3) 或 (3,)
            - 'vel': 速度 (batch_size, 3) 或 (3,)
            - 'rot': 旋转矩阵 (batch_size, 3, 3) 或 (3, 3)
            - 'omega': 角速度 (batch_size, 3) 或 (3,)
    
    Examples:
        >>> obs = np.random.randn(4, 27)  # batch_size=4
        >>> state = extract_state_from_obs(obs)
        >>> state['pos'].shape
        (4, 3)
    """
    is_numpy = isinstance(obs, np.ndarray)
    is_single = (len(obs.shape) == 1)
    
    if is_single:
        if is_numpy:
            obs = obs[np.newaxis, :]
        else:
            obs = obs.unsqueeze(0)
    
    # 提取各个状态分量
    pos = obs[:, 0:3]
    vel = obs[:, 3:6]
    rot = obs[:, 6:15].reshape(-1, 3, 3)
    omega = obs[:, 15:18]
    
    state = {
        'pos': pos,
        'vel': vel,
        'rot': rot,
        'omega': omega,
    }
    
    # 如果是单样本，去掉 batch 维度
    if is_single:
        if is_numpy:
            state = {k: v[0] for k, v in state.items()}
        else:
            state = {k: v[0] for k, v in state.items()}
    
    return state


def extract_sdf_obs(obs: Union[np.ndarray, torch.Tensor], 
                    start_idx: int = 18, 
                    sdf_dim: int = 9) -> Union[np.ndarray, torch.Tensor]:
    """
    从观测中提取 SDF 观测
    
    Args:
        obs: 环境观测，形状为 (batch_size, obs_dim) 或 (obs_dim,)
        start_idx: SDF 观测起始索引，默认 18
        sdf_dim: SDF 观测维度，默认 9
    
    Returns:
        sdf_obs: SDF 网格值，形状为 (batch_size, sdf_dim) 或 (sdf_dim,)
    
    Examples:
        >>> obs = np.random.randn(4, 27)
        >>> sdf = extract_sdf_obs(obs)
        >>> sdf.shape
        (4, 9)
    """
    is_numpy = isinstance(obs, np.ndarray)
    is_single = (len(obs.shape) == 1)
    
    if is_single:
        if is_numpy:
            obs = obs[np.newaxis, :]
        else:
            obs = obs.unsqueeze(0)
    
    sdf_obs = obs[:, start_idx:start_idx + sdf_dim]
    
    if is_single:
        if is_numpy:
            sdf_obs = sdf_obs[0]
        else:
            sdf_obs = sdf_obs[0]
    
    return sdf_obs


def compute_sdf_gradient(sdf_obs: Union[np.ndarray, torch.Tensor], 
                         delta: float = 0.1) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    从 3×3 SDF 网格计算梯度（中心差分法）
    
    SDF 网格布局：
        [0] [1] [2]
        [3] [4] [5]  ← [4] 是中心点（当前点 SDF 值）
        [6] [7] [8]
    
    使用中心差分计算 x 和 y 方向的梯度：
        n_x = (sdf[5] - sdf[3]) / (2 * delta)
        n_y = (sdf[7] - sdf[1]) / (2 * delta)
    
    Args:
        sdf_obs: SDF 网格值，形状为 (batch_size, 9) 或 (9,)
        delta: SDF 网格分辨率（米），默认 0.1
    
    Returns:
        n: 梯度向量（法向量），形状为 (batch_size, 3) 或 (3,)
           z 分量始终为 0（2D 障碍物）
        h: 中心点 SDF 值，形状为 (batch_size,) 或 ()
    
    Examples:
        >>> sdf = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        >>> n, h = compute_sdf_gradient(sdf)
        >>> n.shape
        (3,)
        >>> h
        0.5
    """
    is_numpy = isinstance(sdf_obs, np.ndarray)
    is_single = (len(sdf_obs.shape) == 1)
    
    if is_single:
        if is_numpy:
            sdf_obs = sdf_obs[np.newaxis, :]
        else:
            sdf_obs = sdf_obs.unsqueeze(0)
    
    # 中心点 SDF 值
    h = sdf_obs[:, 4]
    
    # 中心差分计算梯度
    if is_numpy:
        n_x = (sdf_obs[:, 5] - sdf_obs[:, 3]) / (2 * delta)
        n_y = (sdf_obs[:, 7] - sdf_obs[:, 1]) / (2 * delta)
        n_z = np.zeros_like(h)
        n = np.stack([n_x, n_y, n_z], axis=1)
        
        # 归一化
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        # 只在 norm > 1e-6 时归一化，否则保持原方向（零向量）
        mask = (norms > 1e-6).flatten()  # 展平为 (batch_size,)
        n_normalized = np.zeros_like(n)
        if np.any(mask):
            n_normalized[mask] = n[mask] / norms[mask]
        n = n_normalized
    else:
        n_x = (sdf_obs[:, 5] - sdf_obs[:, 3]) / (2 * delta)
        n_y = (sdf_obs[:, 7] - sdf_obs[:, 1]) / (2 * delta)
        n_z = torch.zeros_like(h)
        n = torch.stack([n_x, n_y, n_z], dim=1)
        
        # 归一化
        norms = torch.norm(n, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-6)  # 避免除零
        n = n / norms
    
    if is_single:
        if is_numpy:
            n = n[0]
            h = h[0]
        else:
            n = n[0]
            h = h[0]
    
    return n, h
