"""
平衡性验证模块

功能：
- 检查匹配后的平衡性
- 统计检验
- 可视化平衡性结果
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class BalanceValidator:
    """平衡性验证器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化验证器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.balance_results = {}

    def validate(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        confounder_cols: List[str],
        propensity_col: str = "propensity_score",
        tolerance: float = 0.1,
    ) -> pd.DataFrame:
        """
        执行平衡性验证

        Parameters
        ----------
        df : pd.DataFrame
            匹配后的数据
        treatment_col : str
            处理变量列名
        confounder_cols : list
            混杂因素列名
        propensity_col : str
            倾向性评分列名
        tolerance : float
            标准化差异容忍度

        Returns
        -------
        pd.DataFrame
            平衡性评估表
        """
        logger.info("开始平衡性验证...")

        treatment = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        balance_metrics = []

        for col in confounder_cols + [propensity_col]:
            # 连续变量
            if pd.api.types.is_numeric_dtype(df[col]):
                metric = self._continuous_balance(treatment, control, col)
            # 分类变量
            else:
                metric = self._categorical_balance(treatment, control, col)

            metric["balanced"] = abs(metric.get("std_diff", 1)) < tolerance
            balance_metrics.append(metric)

        balance_df = pd.DataFrame(balance_metrics)

        # 汇总
        n_balanced = balance_df["balanced"].sum()
        n_total = len(balance_df)

        logger.info(f"平衡性检查: {n_balanced}/{n_total} 个变量已平衡")

        self.balance_results = {
            "balance_df": balance_df,
            "n_balanced": n_balanced,
            "n_total": n_total,
        }

        return balance_df

    def _continuous_balance(
        self,
        treatment: pd.DataFrame,
        control: pd.DataFrame,
        col: str,
    ) -> Dict[str, Any]:
        """
        连续变量的平衡性指标
        """
        treat_vals = treatment[col].dropna()
        ctrl_vals = control[col].dropna()

        # 均值
        treat_mean = treat_vals.mean()
        ctrl_mean = ctrl_vals.mean()

        # 标准差
        treat_std = treat_vals.std()
        ctrl_std = ctrl_vals.std()

        # 标准化差异
        pooled_std = np.sqrt((treat_std**2 + ctrl_std**2) / 2)
        std_diff = (treat_mean - ctrl_mean) / pooled_std if pooled_std > 0 else 0

        # t检验
        t_stat, p_value = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False)

        return {
            "variable": col,
            "type": "continuous",
            "treatment_mean": treat_mean,
            "control_mean": ctrl_mean,
            "treatment_std": treat_std,
            "control_std": ctrl_std,
            "std_diff": std_diff,
            "t_statistic": t_stat,
            "p_value": p_value,
        }

    def _categorical_balance(
        self,
        treatment: pd.DataFrame,
        control: pd.DataFrame,
        col: str,
    ) -> Dict[str, Any]:
        """
        分类变量的平衡性指标
        """
        # 频数表
        treat_counts = treatment[col].value_counts(normalize=True)
        ctrl_counts = control[col].value_counts(normalize=True)

        # 所有类别
        all_categories = set(treat_counts.index) | set(ctrl_counts.index)

        # 计算差异
        max_diff = 0
        for cat in all_categories:
            treat_prop = treat_counts.get(cat, 0)
            ctrl_prop = ctrl_counts.get(cat, 0)
            diff = abs(treat_prop - ctrl_prop)
            max_diff = max(max_diff, diff)

        # 卡方检验
        contingency_table = pd.concat([
            treatment[col].value_counts(),
            control[col].value_counts()
        ], axis=1).fillna(0)

        try:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        except:
            chi2, p_value = np.nan, np.nan

        return {
            "variable": col,
            "type": "categorical",
            "max_diff": max_diff,
            "chi_squared": chi2,
            "p_value": p_value,
        }

    def plot_balance(
        self,
        balance_df: pd.DataFrame,
        save_path: str = None,
    ):
        """
        可视化平衡性结果

        Parameters
        ----------
        balance_df : pd.DataFrame
            平衡性评估表
        save_path : str, optional
            保存路径
        """
        # 筛选连续变量
        continuous = balance_df[balance_df["type"] == "continuous"].copy()

        if len(continuous) == 0:
            logger.warning("没有连续变量可绘制")
            return

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))

        # 变量名
        variables = continuous["variable"].values
        y_pos = np.arange(len(variables))

        # 标准化差异（绝对值）
        std_diffs = continuous["std_diff"].abs().values

        # 颜色：平衡为绿色，不平衡为红色
        colors = ["green" if balanced else "red" for balanced in continuous["balanced"]]

        # 柱状图
        ax.barh(y_pos, std_diffs, color=colors, alpha=0.7)

        # 阈值线
        ax.axvline(x=0.1, color="red", linestyle="--", linewidth=2, label="阈值 (0.1)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables)
        ax.set_xlabel("标准化差异（绝对值）")
        ax.set_title("匹配后平衡性检查")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"平衡性图已保存至: {save_path}")

        plt.show()

    def generate_report(self) -> str:
        """生成验证报告"""
        if not self.balance_results:
            return "尚未执行验证"

        balance_df = self.balance_results["balance_df"]

        report = "## 平衡性验证报告\n\n"
        report += f"### 总体情况\n\n"
        report += f"- 已平衡变量: {self.balance_results['n_balanced']}/{self.balance_results['n_total']}\n"
        report += f"- 平衡率: {self.balance_results['n_balanced']/self.balance_results['n_total']*100:.1f}%\n\n"

        report += "### 变量详情\n\n"
        report += "| 变量 | 类型 | 标准化差异 | 平衡 |\n"
        report += "|------|------|-----------|------|\n"

        for _, row in balance_df.iterrows():
            if row["type"] == "continuous":
                report += f"| {row['variable']} | 连续 | {row['std_diff']:.3f} | {'✓' if row['balanced'] else '✗'} |\n"
            else:
                report += f"| {row['variable']} | 分类 | {row.get('max_diff', 0):.3f} | {'✓' if row['balanced'] else '✗'} |\n"

        return report
