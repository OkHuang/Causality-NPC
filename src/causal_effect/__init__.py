"""
因果效应估计模块

提供因果效应估计功能，基于因果发现的输出进行因果效应量化
"""

from .pipeline import run_causal_effect

__all__ = ['run_causal_effect']
