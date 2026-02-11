"""
评估指标模块

功能：
- 计算因果效应评估指标
- 模型性能指标
- 预测准确性指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "regression",
) -> Dict[str, float]:
    """
    计算评估指标

    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    task_type : str
        任务类型 (regression, classification, binary)

    Returns
    -------
    dict
        指标字典
    """
    metrics = {}

    if task_type == "regression":
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)

    elif task_type == "binary":
        y_pred_class = (y_pred > 0.5).astype(int)

        metrics["auc"] = roc_auc_score(y_true, y_pred)
        metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
        metrics["precision"] = precision_score(y_true, y_pred_class, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred_class, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred_class, zero_division=0)

    elif task_type == "classification":
        y_pred_class = np.argmax(y_pred, axis=1)

        metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
        metrics["macro_f1"] = f1_score(y_true, y_pred_class, average="macro", zero_division=0)
        metrics["weighted_f1"] = f1_score(y_true, y_pred_class, average="weighted", zero_division=0)

    else:
        raise ValueError(f"不支持的任务类型: {task_type}")

    return metrics


def calculate_ate_metrics(
    treatment_effect: float,
    se: float,
    p_value: float,
    ci_lower: float,
    ci_upper: float,
) -> Dict[str, Any]:
    """
    计算ATE相关指标

    Parameters
    ----------
    treatment_effect : float
        处理效应
    se : float
        标准误
    p_value : float
        p值
    ci_lower : float
        置信区间下限
    ci_upper : float
        置信区间上限

    Returns
    -------
    dict
        指标字典
    """
    metrics = {
        "ate": treatment_effect,
        "se": se,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "significant": p_value < 0.05,
        "effect_size_relative": abs(treatment_effect / se) if se > 0 else 0,  # 效应量（相对于标准误）
    }

    return metrics


def calculate_balance_metrics(
    treatment_group: pd.DataFrame,
    control_group: pd.DataFrame,
    variables: List[str],
) -> pd.DataFrame:
    """
    计算平衡性指标

    Parameters
    ----------
    treatment_group : pd.DataFrame
        处理组数据
    control_group : pd.DataFrame
        对照组数据
    variables : list
        需要检查的变量列表

    Returns
    -------
    pd.DataFrame
        平衡性指标表
    """
    results = []

    for var in variables:
        treat_mean = treatment_group[var].mean()
        ctrl_mean = control_group[var].mean()

        treat_std = treatment_group[var].std()
        ctrl_std = control_group[var].std()

        # 标准化差异
        pooled_std = np.sqrt((treat_std**2 + ctrl_std**2) / 2)
        std_diff = (treat_mean - ctrl_mean) / pooled_std if pooled_std > 0 else 0

        # 方差比
        variance_ratio = (treat_std**2) / (ctrl_std**2) if ctrl_std > 0 else np.nan

        results.append({
            "variable": var,
            "treatment_mean": treat_mean,
            "control_mean": ctrl_mean,
            "std_diff": std_diff,
            "variance_ratio": variance_ratio,
            "balanced": abs(std_diff) < 0.1,  # < 10%
        })

    return pd.DataFrame(results)
