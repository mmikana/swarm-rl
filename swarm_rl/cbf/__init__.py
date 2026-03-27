"""
CBF (Control Barrier Function) 模块

提供距离感知的姿态 CBF-QP 层，用于四旋翼无人机避障。
"""

from .cbf_layer import DistanceAwareCBFLayer

__all__ = ['DistanceAwareCBFLayer']
