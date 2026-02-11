"""
平均处理效应（ATE）估计模块

功能：
- 估计平均处理效应（Average Treatment Effect）
- 支持多种估计方法（均值差、回归调整、IPW、双稳健）
- 提供置信区间和统计推断
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import resample
import logging

logger = logging.getLogger(__name__)


class ATEEstimator:
    """平均处理效应估计器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化估计器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}

    def estimate(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        method: str = "difference_in_means",
        confounder_cols: Optional[List[str]] = None,
        propensity_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        估计平均处理效应

        Parameters
        ----------
        df : pd.DataFrame
            数据（通常是匹配后的数据）
        treatment_col : str
            处理变量列名（0/1）
        outcome_col : str
            结果变量列名
        method : str
            估计方法
            - 'difference_in_means': 简单均值差
            - 'regression_adjustment': 回归调整
            - 'ipw': 逆概率加权（Inverse Probability Weighting）
            - 'doubly_robust': 双稳健估计
        confounder_cols : list, optional
            混杂因素列名
        propensity_col : str, optional
            倾向性评分列名（IPW和双稳健方法需要）

        Returns
        -------
        dict
            估计结果
        """
        logger.info(f"开始估计ATE（方法: {method}）...")

        # 根据方法选择估计器
        if method == "difference_in_means":
            results = self._difference_in_means(df, treatment_col, outcome_col)
        elif method == "regression_adjustment":
            if confounder_cols is None:
                raise ValueError("回归调整需要指定confounder_cols")
            results = self._regression_adjustment(df, treatment_col, outcome_col, confounder_cols)
        elif method == "ipw":
            if propensity_col is None:
                raise ValueError("IPW方法需要指定propensity_col")
            results = self._ipw(df, treatment_col, outcome_col, propensity_col)
        elif method == "doubly_robust":
            if confounder_cols is None or propensity_col is None:
                raise ValueError("双稳健方法需要指定confounder_cols和propensity_col")
            results = self._doubly_robust(df, treatment_col, outcome_col, confounder_cols, propensity_col)
        else:
            raise ValueError(f"不支持的估计方法: {method}")

        # 添加方法信息
        results["method"] = method

        logger.info(f"ATE估计完成: {results['ate']:.4f}")

        return results

    def _difference_in_means(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
    ) -> Dict[str, Any]:
        """
        简单均值差法

        ATE = E[Y|T=1] - E[Y|T=0]
        """
        treatment = df[df[treatment_col] == 1][outcome_col]
        control = df[df[treatment_col] == 0][outcome_col]

        ate = treatment.mean() - control.mean()

        # 计算标准误差
        n_treatment = len(treatment)
        n_control = len(control)
        var_treatment = treatment.var()
        var_control = control.var()

        se = np.sqrt(var_treatment / n_treatment + var_control / n_control)

        # 95% 置信区间
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # t统计量和p值
        t_stat = ate / se
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            "ate": ate,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_treatment": n_treatment,
            "n_control": n_control,
        }

    def _regression_adjustment(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: List[str],
    ) -> Dict[str, Any]:
        """
        回归调整法

        使用回归模型控制混杂因素
        """
        # 准备数据
        X = df[confounder_cols + [treatment_col]].copy()
        X = X.fillna(X.median())
        y = df[outcome_col].copy()

        # 拟合OLS模型
        model = LinearRegression()
        model.fit(X, y)

        # 处理变量的系数即为ATE
        ate = model.coef_[X.columns.get_loc(treatment_col)]

        # 使用Bootstrap计算标准误差
        n_bootstrap = self.config.get("n_bootstrap", 1000)
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            X_boot, y_boot = resample(X, y)
            model_boot = LinearRegression()
            model_boot.fit(X_boot, y_boot)
            boot_ate = model_boot.coef_[X.columns.get_loc(treatment_col)]
            bootstrap_ates.append(boot_ate)

        se = np.std(bootstrap_ates)
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)

        # p值
        t_stat = ate / se
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            "ate": ate,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_treatment": (df[treatment_col] == 1).sum(),
            "n_control": (df[treatment_col] == 0).sum(),
        }

    def _ipw(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        propensity_col: str,
    ) -> Dict[str, Any]:
        """
        逆概率加权法（Inverse Probability Weighting）

        使用倾向性评分的倒数作为权重
        """
        # 计算权重
        # 处理组: weight = 1 / PS
        # 对照组: weight = 1 / (1 - PS)
        weights = np.where(
            df[treatment_col] == 1,
            1 / (df[propensity_col] + 1e-10),
            1 / (1 - df[propensity_col] + 1e-10),
        )

        # 截断极端权重
        weight_percentiles = np.percentile(weights, [1, 99])
        weights = np.clip(weights, weight_percentiles[0], weight_percentiles[1])

        # 计算加权均值
        treatment_mask = df[treatment_col] == 1
        control_mask = df[treatment_col] == 0

        treatment_weighted_mean = np.average(
            df[treatment_mask][outcome_col],
            weights=weights[treatment_mask]
        )
        control_weighted_mean = np.average(
            df[control_mask][outcome_col],
            weights=weights[control_mask]
        )

        ate = treatment_weighted_mean - control_weighted_mean

        # 使用稳健标准误
        # 简化版本：使用加权的方差
        n_treatment = treatment_mask.sum()
        n_control = control_mask.sum()

        # 计算加权方差
        treatment_var = np.average(
            (df[treatment_mask][outcome_col] - treatment_weighted_mean) ** 2,
            weights=weights[treatment_mask]
        )
        control_var = np.average(
            (df[control_mask][outcome_col] - control_weighted_mean) ** 2,
            weights=weights[control_mask]
        )

        se = np.sqrt(treatment_var / n_treatment + control_var / n_control)

        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        t_stat = ate / se
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            "ate": ate,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_treatment": n_treatment,
            "n_control": n_control,
        }

    def _doubly_robust(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: List[str],
        propensity_col: str,
    ) -> Dict[str, Any]:
        """
        双稳健估计（Doubly Robust）

        结合回归调整和IPW的优势，只要其中一个模型正确，估计就是一致的
        """
        # 1. 拟合结果模型（Outcome Model）
        X_outcome = df[confounder_cols + [treatment_col]].copy()
        X_outcome = X_outcome.fillna(X_outcome.median())
        y = df[outcome_col].copy()

        outcome_model_treatment = LinearRegression()
        outcome_model_control = LinearRegression()

        # 分别拟合处理组和对照组
        treatment_mask = df[treatment_col] == 1
        control_mask = df[treatment_col] == 0

        outcome_model_treatment.fit(X_outcome[treatment_mask], y[treatment_mask])
        outcome_model_control.fit(X_outcome[control_mask], y[control_mask])

        # 预测潜在结果
        mu_1 = outcome_model_treatment.predict(X_outcome)
        mu_0 = outcome_model_control.predict(X_outcome)

        # 2. 计算IPW
        ps = df[propensity_col].values
        weights_treatment = 1 / (ps + 1e-10)
        weights_control = 1 / (1 - ps + 1e-10)

        # 截断极端权重
        weight_percentiles = np.percentile(weights_treatment, [1, 99])
        weights_treatment = np.clip(weights_treatment, weight_percentiles[0], weight_percentiles[1])
        weights_control = np.clip(weights_control, weight_percentiles[0], weight_percentiles[1])

        # 3. 双稳健估计
        # DR = mu_1 - mu_0 + (T * (Y - mu_1) / PS) - ((1-T) * (Y - mu_0) / (1-PS))
        T = df[treatment_col].values
        Y = y.values

        dr_estimates = (
            mu_1 - mu_0 +
            T * (Y - mu_1) * weights_treatment -
            (1 - T) * (Y - mu_0) * weights_control
        )

        ate = np.mean(dr_estimates)

        # 使用Bootstrap计算标准误差
        n_bootstrap = self.config.get("n_bootstrap", 500)
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            indices = resample(np.arange(len(df)))
            df_boot = df.iloc[indices].copy()

            # 重新拟合模型
            X_boot = df_boot[confounder_cols + [treatment_col]].copy()
            X_boot = X_boot.fillna(X_boot.median())
            y_boot = df_boot[outcome_col].copy()

            T_boot = df_boot[treatment_col].values
            Y_boot = y_boot.values
            ps_boot = df_boot[propensity_col].values

            treatment_mask_boot = T_boot == 1
            control_mask_boot = T_boot == 0

            # 简化：使用全局模型
            outcome_model = LinearRegression()
            outcome_model.fit(X_boot, Y_boot)

            # 预测
            X_boot_treat = X_boot.copy()
            X_boot_treat[treatment_col] = 1
            mu_1_boot = outcome_model.predict(X_boot_treat)

            X_boot_ctrl = X_boot.copy()
            X_boot_ctrl[treatment_col] = 0
            mu_0_boot = outcome_model.predict(X_boot_ctrl)

            # DR估计
            weights_t_boot = 1 / (ps_boot + 1e-10)
            weights_c_boot = 1 / (1 - ps_boot + 1e-10)

            dr_boot = (
                mu_1_boot - mu_0_boot +
                T_boot * (Y_boot - mu_1_boot) * weights_t_boot -
                (1 - T_boot) * (Y_boot - mu_0_boot) * weights_c_boot
            )

            bootstrap_ates.append(np.mean(dr_boot))

        se = np.std(bootstrap_ates)
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)

        t_stat = ate / se
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            "ate": ate,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_treatment": treatment_mask.sum(),
            "n_control": control_mask.sum(),
        }
