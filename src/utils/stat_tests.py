"""
统计检验模块

功能：
- t检验
- 卡方检验
- Fisher精确检验
- Kolmogorov-Smirnov检验
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatisticalTests:
    """统计检验类"""

    @staticmethod
    def t_test(
        sample1: np.ndarray,
        sample2: np.ndarray,
        equal_var: bool = False,
    ) -> Dict[str, Any]:
        """
        独立样本t检验

        Parameters
        ----------
        sample1 : np.ndarray
            样本1
        sample2 : np.ndarray
            样本2
        equal_var : bool
            是否假设方差相等

        Returns
        -------
        dict
            检验结果
        """
        t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)

        # 计算均值差异
        mean_diff = sample1.mean() - sample2.mean()

        # 计算Cohen's d（效应量）
        n1, n2 = len(sample1), len(sample2)
        pooled_std = np.sqrt(((n1-1)*sample1.var() + (n2-1)*sample2.var()) / (n1+n2-2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        return {
            "test": "Independent t-test",
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def chi_square_test(
        contingency_table: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        卡方检验

        Parameters
        ----------
        contingency_table : pd.DataFrame
            列联表

        Returns
        -------
        dict
            检验结果
        """
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        return {
            "test": "Chi-square test",
            "chi_squared": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "expected_frequencies": expected,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def fisher_exact_test(
        contingency_table: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Fisher精确检验

        Parameters
        ----------
        contingency_table : pd.DataFrame
            2x2列联表

        Returns
        -------
        dict
            检验结果
        """
        odds_ratio, p_value = stats.fisher_exact(contingency_table.values)

        return {
            "test": "Fisher's exact test",
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def ks_test(
        sample1: np.ndarray,
        sample2: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov检验

        检验两个样本是否来自同一分布

        Parameters
        ----------
        sample1 : np.ndarray
            样本1
        sample2 : np.ndarray
            样本2

        Returns
        -------
        dict
            检验结果
        """
        ks_stat, p_value = stats.ks_2samp(sample1, sample2)

        return {
            "test": "Kolmogorov-Smirnov test",
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "same_distribution": p_value > 0.05,
        }

    @staticmethod
    def mann_whitney_u_test(
        sample1: np.ndarray,
        sample2: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Mann-Whitney U检验（非参数）

        Parameters
        ----------
        sample1 : np.ndarray
            样本1
        sample2 : np.ndarray
            样本2

        Returns
        -------
        dict
            检验结果
        """
        u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")

        return {
            "test": "Mann-Whitney U test",
            "u_statistic": u_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def anova(*samples: np.ndarray) -> Dict[str, Any]:
        """
        单因素方差分析（ANOVA）

        Parameters
        ----------
        *samples : np.ndarray
            多个样本

        Returns
        -------
        dict
            检验结果
        """
        f_stat, p_value = stats.f_oneway(*samples)

        # 计算效应量（eta squared）
        # TODO: 实现eta squared计算

        return {
            "test": "One-way ANOVA",
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def correlation_test(
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson",
    ) -> Dict[str, Any]:
        """
        相关性检验

        Parameters
        ----------
        x : np.ndarray
            变量1
        y : np.ndarray
            变量2
        method : str
            pearson, spearman, kendall

        Returns
        -------
        dict
            检验结果
        """
        if method == "pearson":
            corr, p_value = stats.pearsonr(x, y)
        elif method == "spearman":
            corr, p_value = stats.spearmanr(x, y)
        elif method == "kendall":
            corr, p_value = stats.kendalltau(x, y)
        else:
            raise ValueError(f"不支持的方法: {method}")

        return {
            "test": f"{method.capitalize()} correlation",
            "correlation": corr,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
