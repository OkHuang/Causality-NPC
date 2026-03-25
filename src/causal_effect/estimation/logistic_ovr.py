"""
One-vs-Rest 逻辑回归估计

支持二分类和多分类结果的因果效应估计
"""

import time
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
from typing import Dict, List, Tuple, Any, Optional


def _calculate_binary_logistic(
    Y_data: pd.Series,
    T_data: pd.Series,
    X_data: Optional[pd.DataFrame],
    treatment: str
) -> Tuple[float, float, float, Dict]:
    """
    使用逻辑回归计算 ATE（二分类结果）

    Parameters
    ----------
    Y_data : pd.Series
        结果变量（二分类）
    T_data : pd.Series
        处理变量
    X_data : pd.DataFrame or None
        混淆变量（协变量）
    treatment : str
        处理变量名称

    Returns
    -------
    Tuple[float, float, float, Dict]
        (ate, ci_lower, ci_upper, model_stats)
    """
    # 构建设计矩阵
    exog = pd.DataFrame({treatment: T_data})
    if X_data is not None and len(X_data.columns) > 0:
        exog = pd.concat([exog, X_data], axis=1)
    exog = sm.add_constant(exog)

    # 拟合逻辑回归（带降级策略）
    model = None
    for optimizer in ['newton', 'bfgs', 'lbfgs']:
        try:
            model = sm.Logit(Y_data, exog).fit(method=optimizer, maxiter=100, disp=0)
            if model.mle_retvals['converged']:
                break
        except Exception:
            continue

    if model is None or not model.mle_retvals['converged']:
        raise ValueError("逻辑回归未收敛")

    # 提取结果
    ate = model.params[treatment]
    conf_int = model.conf_int(alpha=0.05)
    ci_lower = conf_int.loc[treatment, 0]
    ci_upper = conf_int.loc[treatment, 1]

    # 统计指标
    model_stats = {
        'p_value': model.pvalues[treatment],
        'aic': model.aic,
        'bic': model.bic,
        'llf': model.llf,
        'prsquared': model.prsquared,
        'odds_ratio': np.exp(ate),
        'or_ci_lower': np.exp(ci_lower),
        'or_ci_upper': np.exp(ci_upper),
        'converged': True,
        'is_multiclass': False
    }

    return ate, ci_lower, ci_upper, model_stats


def _calculate_ovr_logistic(
    Y_data: pd.Series,
    T_data: pd.Series,
    X_data: Optional[pd.DataFrame],
    treatment: str
) -> Dict:
    """
    使用一对多逻辑回归计算 ATE（多分类结果）

    Parameters
    ----------
    Y_data : pd.Series
        结果变量（多分类）
    T_data : pd.Series
        处理变量
    X_data : pd.DataFrame or None
        混淆变量
    treatment : str
        处理变量名称

    Returns
    -------
    Dict
        包含加权平均ATE和各类别ATE的字典
    """
    unique_vals = sorted(Y_data.dropna().unique())
    category_results = []

    for target_class in unique_vals:
        # 将当前类别二值化
        Y_binary = (Y_data == target_class).astype(int)

        # 至少20个正样本
        if Y_binary.sum() < 20:
            continue

        try:
            ate, ci_lower, ci_upper, stats = _calculate_binary_logistic(
                Y_binary, T_data, X_data, treatment
            )
            category_results.append({
                'category': target_class,
                'ate': ate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'stats': stats,
                'n_samples': Y_binary.sum()
            })
        except Exception:
            continue

    if len(category_results) == 0:
        raise ValueError("所有类别的逻辑回归均未收敛或样本量不足")

    # 提取结果
    categories = [r['category'] for r in category_results]
    ate_list = [r['ate'] for r in category_results]
    model_stats_list = [r['stats'] for r in category_results]

    # 计算加权平均 ATE
    weights = np.array([r['n_samples'] for r in category_results])
    weights = weights / weights.sum()
    weighted_ate = np.average(ate_list, weights=weights)

    return {
        'ate': weighted_ate,
        'category_ates': ate_list,
        'ci_lower': None,
        'ci_upper': None,
        'model_stats': model_stats_list,
        'categories': categories,
        'is_multiclass': True,
        'category_results': category_results
    }


def estimate_logistic_ovr(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    source: str,
    target: str,
    bootstrap_iter: int = 50,
    confidence_level: float = 0.95,
    min_sample_size: int = 20,
    random_seed: int = 42
) -> Dict:
    """
    使用OVR逻辑回归估计单条边的因果效应

    Parameters
    ----------
    data : pd.DataFrame
        数据矩阵
    graph : nx.DiGraph
        因果图
    source : str
        处理变量
    target : str
        结果变量
    bootstrap_iter : int
        Bootstrap迭代次数
    confidence_level : float
        置信水平
    min_sample_size : int
        最小样本量
    random_seed : int
        随机种子

    Returns
    -------
    Dict
        估计结果，包含：
        - treatment: 处理变量名
        - outcome: 结果变量名
        - method: 'logistic_ovr'
        - ate: 平均处理效应
        - ci_lower: 置信区间下限
        - ci_upper: 置信区间上限
        - model_stats: 模型统计信息
        - timing: 耗时信息
    """
    # Step 1: 识别调整集
    identify_start = time.time()
    adjustment_set = list(graph.predecessors(source))
    valid_adj_set = [col for col in adjustment_set if col in data.columns and col != target]
    identify_time = time.time() - identify_start

    # Step 2: 准备数据
    cols_needed = [source, target] + valid_adj_set
    df_clean = data[cols_needed].dropna()

    if len(df_clean) < min_sample_size:
        raise ValueError(f"有效样本量不足 (N={len(df_clean)})")

    # Step 3: 估计效应
    estimate_start = time.time()

    # 核心计算函数
    def calculate_ate(Y, T, X):
        unique_vals = sorted(Y.dropna().unique())
        n_classes = len(unique_vals)

        # 二分类
        if n_classes == 2 and set(unique_vals).issubset({0, 1}):
            return _calculate_binary_logistic(Y, T, X, source)
        # 多分类或非标准二分类
        else:
            return _calculate_ovr_logistic(Y, T, X, source)

    ate = None
    ci_lower, ci_upper = None, None
    model_stats = None

    # Bootstrap
    if bootstrap_iter > 0:
        boot_ates = []
        rs = np.random.RandomState(random_seed)
        last_valid_result = None

        for i in range(bootstrap_iter):
            boot_df = df_clean.sample(frac=1.0, replace=True, random_state=rs.randint(0, 2**31-1))
            boot_Y = boot_df[target]
            boot_T = boot_df[source]
            boot_X = boot_df[valid_adj_set] if valid_adj_set else None

            boot_result = calculate_ate(boot_Y, boot_T, boot_X)

            if boot_result is not None:
                # 解析结果
                if isinstance(boot_result, dict) and boot_result.get('is_multiclass'):
                    boot_ate = boot_result['ate']
                elif isinstance(boot_result, tuple):
                    boot_ate = boot_result[0]
                else:
                    continue

                boot_ates.append(boot_ate)

                if i == bootstrap_iter - 1:
                    last_valid_result = boot_result

        if len(boot_ates) > 0:
            ate = np.mean(boot_ates)
            alpha = 1 - confidence_level
            ci_lower = np.percentile(boot_ates, alpha / 2 * 100)
            ci_upper = np.percentile(boot_ates, (1 - alpha / 2) * 100)

            if last_valid_result is not None:
                model_stats = last_valid_result
        else:
            raise ValueError("Bootstrap未能计算出有效ATE")
    else:
        # 单次估计
        Y = df_clean[target]
        T = df_clean[source]
        X = df_clean[valid_adj_set] if valid_adj_set else None

        result = calculate_ate(Y, T, X)

        if result is None:
            raise ValueError("未能计算出有效ATE")

        if isinstance(result, dict) and result.get('is_multiclass'):
            ate = result['ate']
            model_stats = result
        elif isinstance(result, tuple):
            ate, ci_lower, ci_upper, model_stats = result

    estimate_time = time.time() - estimate_start

    return {
        'treatment': source,
        'outcome': target,
        'method': 'logistic_ovr',
        'ate': ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'model_stats': model_stats,
        'timing': {
            'identify': identify_time,
            'estimate': estimate_time,
            'total': identify_time + estimate_time
        }
    }
