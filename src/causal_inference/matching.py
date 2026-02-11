"""
患者匹配模块

功能：
- 基于倾向性评分进行患者匹配
- 支持多种匹配方法（最近邻、最优匹配、遗传匹配）
- 评估匹配质量
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


def match_patients(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    propensity_col: str = "propensity_score",
    method: str = "nearest_neighbor",
    ratio: int = 1,
    caliper: float = 0.2,
    replace: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    执行患者匹配

    Parameters
    ----------
    df : pd.DataFrame
        包含倾向性评分的数据
    treatment_col : str
        处理变量列名
    propensity_col : str
        倾向性评分列名
    method : str
        匹配方法
    ratio : int
        对照组:处理组比例
    caliper : float
        卡尺值（标准差倍数）
    replace : bool
        是否允许重复匹配

    Returns
    -------
    matched_df : pd.DataFrame
        匹配后的数据
    match_info : dict
        匹配信息
    """
    logger.info(f"开始患者匹配（方法: {method}）...")

    # 分离处理组和对照组
    treatment = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    n_treatment = len(treatment)
    n_control = len(control)

    logger.info(f"处理组: {n_treatment} 人, 对照组: {n_control} 人")

    # 执行匹配
    if method == "nearest_neighbor":
        matched_df, pairs = _nearest_neighbor_match(
            treatment, control,
            propensity_col, ratio, caliper, replace
        )
    elif method == "optimal":
        matched_df, pairs = _optimal_match(
            treatment, control,
            propensity_col, ratio
        )
    else:
        raise ValueError(f"不支持的匹配方法: {method}")

    # 统计信息
    match_info = {
        "method": method,
        "n_treatment_original": n_treatment,
        "n_control_original": n_control,
        "n_treatment_matched": len(matched_df[matched_df[treatment_col] == 1]),
        "n_control_matched": len(matched_df[matched_df[treatment_col] == 0]),
        "n_pairs": len(pairs),
        "caliper": caliper,
        "ratio": ratio,
    }

    logger.info(f"匹配完成: {match_info['n_pairs']} 对")

    return matched_df, match_info


def _nearest_neighbor_match(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    propensity_col: str,
    ratio: int,
    caliper: float,
    replace: bool,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    最近邻匹配
    """
    treatment_ps = treatment[propensity_col].values.reshape(-1, 1)
    control_ps = control[propensity_col].values.reshape(-1, 1)

    # 计算标准差（用于卡尺）
    ps_std = np.concatenate([treatment_ps.flatten(), control_ps.flatten()]).std()
    caliper_threshold = caliper * ps_std

    # 使用KD树查找最近邻
    nbrs = NearestNeighbors(n_neighbors=ratio, algorithm="kd_tree")
    nbrs.fit(control_ps)

    distances, indices = nbrs.kneighbors(treatment_ps)

    matched_control_indices = []
    matched_pairs = []
    used_control_indices = set()

    # 获取control的原始索引
    control_original_indices = control.index.tolist()

    for i, (treat_idx, neighbors, dists) in enumerate(zip(treatment.index, indices, distances)):
        # 找到符合条件的对照
        for j, (array_idx, dist) in enumerate(zip(neighbors, dists)):
            # 检查卡尺
            if dist > caliper_threshold:
                continue

            # 获取control的原始索引
            ctrl_idx = control_original_indices[array_idx]

            # 检查是否重复使用
            if not replace and ctrl_idx in used_control_indices:
                continue

            matched_control_indices.append(ctrl_idx)
            matched_pairs.append((treat_idx, ctrl_idx))
            used_control_indices.add(ctrl_idx)
            break

    # 构建匹配后的数据集
    matched_treatment = treatment.loc[[p[0] for p in matched_pairs]]
    matched_control = control.loc[[p[1] for p in matched_pairs]]

    # 添加配对ID
    matched_treatment["pair_id"] = range(len(matched_pairs))
    matched_control["pair_id"] = range(len(matched_pairs))

    matched_df = pd.concat([matched_treatment, matched_control], ignore_index=True)

    return matched_df, matched_pairs


def _optimal_match(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    propensity_col: str,
    ratio: int,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    最优匹配（全局优化）

    TODO: 实现最优匹配算法
    """
    raise NotImplementedError("最优匹配待实现")


def assess_match_quality(
    matched_df: pd.DataFrame,
    treatment_col: str,
    confounder_cols: List[str],
    propensity_col: str = "propensity_score",
) -> pd.DataFrame:
    """
    评估匹配质量

    计算匹配前后处理组和对照组的标准化差异

    Parameters
    ----------
    matched_df : pd.DataFrame
        匹配后的数据
    treatment_col : str
        处理变量列名
    confounder_cols : list
        混杂因素列名
    propensity_col : str
        倾向性评分列名

    Returns
    -------
    pd.DataFrame
        平衡性评估表
    """
    treatment = matched_df[matched_df[treatment_col] == 1]
    control = matched_df[matched_df[treatment_col] == 0]

    balance_metrics = []

    for col in confounder_cols + [propensity_col]:
        # 计算均值
        treat_mean = treatment[col].mean()
        ctrl_mean = control[col].mean()

        # 计算标准差
        treat_std = treatment[col].std()
        ctrl_std = control[col].std()

        # 标准化差异
        pooled_std = np.sqrt((treat_std**2 + ctrl_std**2) / 2)
        std_diff = (treat_mean - ctrl_mean) / pooled_std

        balance_metrics.append({
            "variable": col,
            "treatment_mean": treat_mean,
            "control_mean": ctrl_mean,
            "std_diff": std_diff,
            "balanced": abs(std_diff) < 0.1,  # < 10% 认为平衡
        })

    balance_df = pd.DataFrame(balance_metrics)

    logger.info(f"平衡性检查: {balance_df['balanced'].sum()}/{len(balance_df)} 个变量已平衡")

    return balance_df
