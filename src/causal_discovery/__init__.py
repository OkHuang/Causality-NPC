"""
因果发现模块

包含因果发现的完整流程：
- 数据加载和处理
- 特征编码
- 因果发现算法
- 约束管理
- 结果保存和报告
"""

from .config import NPCConfig
from .pipeline import run_causal_discovery

__all__ = ['NPCConfig', 'run_causal_discovery']
