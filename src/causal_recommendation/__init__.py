"""
因果推荐模块

基于因果发现的图和因果效应的ATE估计，为患者推荐药物
"""

from .pipeline import run_causal_recommendation
from .evaluation.evaluator import RecommendationEvaluator
from .evaluation.threshold_search import threshold_search

__all__ = [
    'run_causal_recommendation',
    'RecommendationEvaluator',
    'threshold_search'
]

__version__ = '2.0.0'
