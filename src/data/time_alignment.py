"""
时间轴对齐模块

功能：
- 将每位患者的时间轴转换为相对时间
- 时间窗口归并
- 聚合同一窗口内的多条记录
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class TimeAligner:
    """时间轴对齐器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化时间对齐器

        Parameters
        ----------
        config : dict
            配置字典，包含：
            - time_window_days: 时间窗口大小（天）
            - aggregation_method: 聚合方法 (max, sum, union)
        """
        self.config = config or {}
        self.time_window_days = self.config.get("time_window_days", 30)
        self.aggregation_method = self.config.get("aggregation_method", "max")

    def align(self, df: pd.DataFrame, patient_col: str = "patient_id", time_col: str = "time") -> pd.DataFrame:
        """
        执行时间轴对齐

        Parameters
        ----------
        df : pd.DataFrame
            原始数据，必须包含 patient_id 和 time 列
        patient_col : str
            患者ID列名
        time_col : str
            时间列名

        Returns
        -------
        pd.DataFrame
            时间对齐后的数据，新增 time_step 列
        """
        logger.info("开始时间轴对齐...")

        if not all(col in df.columns for col in [patient_col, time_col]):
            raise ValueError(f"数据必须包含 {patient_col} 和 {time_col} 列")

        df_aligned = df.copy()

        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_aligned[time_col]):
            df_aligned[time_col] = pd.to_datetime(df_aligned[time_col])

        # 为每个患者计算相对时间
        df_aligned = self._compute_relative_time(df_aligned, patient_col, time_col)

        # 时间窗口归并
        df_aligned = self._bin_time_windows(df_aligned, patient_col)

        # 聚合同一窗口内的记录
        df_aligned = self._aggregate_within_window(df_aligned, patient_col)

        logger.info(f"时间对齐完成。最终数据: {df_aligned.shape}")

        return df_aligned

    def _compute_relative_time(
        self,
        df: pd.DataFrame,
        patient_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        """
        计算相对时间（以首次就诊为t0）
        """
        df = df.sort_values([patient_col, time_col])

        # 计算每个患者的首次就诊时间
        first_visit = df.groupby(patient_col)[time_col].transform("min")

        # 计算相对天数
        df["days_since_first"] = (df[time_col] - first_visit).dt.days

        logger.info("已计算相对时间轴")

        return df

    def _bin_time_windows(self, df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
        """
        将连续时间离散化为时间窗口
        """
        # 计算时间步长
        df["time_step"] = df["days_since_first"] // self.time_window_days

        logger.info(f"时间窗口大小: {self.time_window_days} 天")

        return df

    def _aggregate_within_window(
        self,
        df: pd.DataFrame,
        patient_col: str,
    ) -> pd.DataFrame:
        """
        聚合同一患者同一时间窗口内的多条记录

        策略：
        - 症状严重程度：取最大值（最严重的症状）
        - 药物：取并集（所有用过的药）
        - 证型：取并集
        - 数值变量：取均值
        """
        # 定义聚合规则
        agg_rules = self._get_aggregation_rules(df)

        # 执行聚合
        df_aggregated = df.groupby([patient_col, "time_step"], as_index=False).agg(agg_rules)

        # 计算每个窗口的实际时间（取该窗口内最后一次就诊的时间）
        time_in_window = df.groupby([patient_col, "time_step"])["time"].last().reset_index()
        df_aggregated = df_aggregated.merge(time_in_window, on=[patient_col, "time_step"], how="left")

        # 计算窗口内记录数（用于质量检查）
        record_counts = df.groupby([patient_col, "time_step"]).size().reset_index(name="records_in_window")
        df_aggregated = df_aggregated.merge(record_counts, on=[patient_col, "time_step"], how="left")

        # 重新计算 days_since_first（基于聚合后的时间）
        first_visit = df_aggregated.groupby(patient_col)["time"].transform("min")
        df_aggregated["days_since_first"] = (df_aggregated["time"] - first_visit).dt.days

        logger.info("已完成时间窗口内记录聚合")

        return df_aggregated

    def _get_aggregation_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        为每列定义聚合规则
        """
        agg_rules = {}

        for col in df.columns:
            if col in ["patient_id", "time_step", "days_since_first", "time"]:
                continue

            # 症状相关列（假设以 severity 或 score 结尾）
            if any(keyword in col.lower() for keyword in ["severity", "score", "严重"]):
                agg_rules[col] = "max"

            # 药物列（文本）
            elif any(keyword in col.lower() for keyword in ["medicine", "medicines", "药", "drug"]):
                agg_rules[col] = lambda x: self._union_lists(x)

            # 证型列
            elif any(keyword in col.lower() for keyword in ["diagnosis", "syndrome", "证"]):
                agg_rules[col] = lambda x: self._union_lists(x)

            # 其他数值列
            elif pd.api.types.is_numeric_dtype(df[col]):
                agg_rules[col] = "mean"

            # 其他文本列
            else:
                agg_rules[col] = "first"  # 保留第一条记录的值

        return agg_rules

    def _union_lists(self, series: pd.Series) -> str:
        """
        合并多个列表/字符串为一个并集

        Examples:
        - ["黄芪", "白术"] + ["黄芪", "丹参"] -> ["黄芪", "白术", "丹参"]
        """
        all_items = set()

        for val in series:
            if pd.isna(val):
                continue

            # 处理字符串格式的列表
            if isinstance(val, str):
                # 尝试解析列表格式
                if val.startswith("[") and val.endswith("]"):
                    try:
                        items = eval(val)
                        if isinstance(items, list):
                            all_items.update(items)
                            continue
                    except:
                        pass

                # 分隔符格式（逗号、顿号等）
                for sep in [",", "、", ";", " "]:
                    if sep in val:
                        all_items.update([item.strip() for item in val.split(sep) if item.strip()])
                        break
                else:
                    all_items.add(val.strip())

            elif isinstance(val, list):
                all_items.update(val)

        return str(sorted(list(all_items)))

    def create_lag_features(
        self,
        df: pd.DataFrame,
        patient_col: str = "patient_id",
        time_step_col: str = "time_step",
        lag: int = 1,
    ) -> pd.DataFrame:
        """
        创建滞后特征（用于时序因果发现）

        将 t 时刻的特征与 t+lag 时刻的结果配对

        Parameters
        ----------
        df : pd.DataFrame
            时间对齐后的数据
        patient_col : str
            患者ID列名
        time_step_col : str
            时间步列名
        lag : int
            滞后步长

        Returns
        -------
        pd.DataFrame
            包含滞后特征的数据，新增 _t 和 _tlag 后缀的列
        """
        df = df.sort_values([patient_col, time_step_col])

        # 计算下一个时间步
        df["next_time_step"] = df.groupby(patient_col)[time_step_col].shift(-lag)

        # 计算时间间隔
        df["time_gap_days"] = df.groupby(patient_col)["days_since_first"].shift(-lag) - df["days_since_first"]

        # 只保留有下一个时间步的记录
        df_with_lag = df.dropna(subset=["next_time_step"]).copy()

        # 过滤时间间隔过长的记录
        max_gap = self.config.get("max_time_gap_days", 180)
        df_with_lag = df_with_lag[df_with_lag["time_gap_days"] <= max_gap]

        logger.info(f"创建了 {len(df_with_lag)} 条时序对（lag={lag}）")

        return df_with_lag
