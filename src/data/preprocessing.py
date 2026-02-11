"""
数据预处理模块

功能：
- 数据清洗
- 缺失值处理
- 异常值检测
- 数据类型转换
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化预处理器

        Parameters
        ----------
        config : dict, optional
            配置字典，包含预处理参数
        """
        self.config = config or {}
        self.transformations = []  # 记录所有转换操作

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的数据清洗流程

        Parameters
        ----------
        df : pd.DataFrame
            原始数据

        Returns
        -------
        pd.DataFrame
            清洗后的数据
        """
        logger.info("开始数据清洗...")
        df_clean = df.copy()

        # 1. 去除重复行
        df_clean = self._remove_duplicates(df_clean)

        # 2. 处理缺失值
        # df_clean = self._handle_missing(df_clean)

        # 3. 处理异常值
        # df_clean = self._handle_outliers(df_clean)

        # 4. 数据类型转换
        df_clean = self._convert_dtypes(df_clean)

        logger.info(f"数据清洗完成。原始: {df.shape}, 清洗后: {df_clean.shape}")

        return df_clean

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """去除重复记录"""
        n_before = len(df)
        df_clean = df.drop_duplicates()
        n_removed = n_before - len(df_clean)

        if n_removed > 0:
            logger.info(f"去除了 {n_removed} 条重复记录")
            self.transformations.append(f"remove_duplicates: {n_removed} rows")

        return df_clean

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值

        策略：
        - 关键字段缺失比例 > 阈值：删除该行
        - 文本字段缺失：填充为"无"
        - 数值字段缺失：根据配置填充（均值/中位数/0）
        """
        threshold = self.config.get("missing_threshold", 0.5)

        # 检查每行的缺失比例
        missing_ratio = df.isnull().mean(axis=1)
        df_clean = df[missing_ratio <= threshold].copy()

        n_removed = len(df) - len(df_clean)
        if n_removed > 0:
            logger.info(f"因缺失值过多删除了 {n_removed} 行")
            self.transformations.append(f"high_missing: {n_removed} rows")

        # 填充剩余缺失值
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                # 文本字段
                df_clean[col].fillna("无", inplace=True)
            else:
                # 数值字段
                fill_strategy = self.config.get("numeric_fill", "median")
                if fill_strategy == "mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif fill_strategy == "median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna(0, inplace=True)

        return df_clean

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值

        使用IQR方法检测数值字段的异常值
        """
        if not self.config.get("outlier_detection", True):
            return df

        df_clean = df.copy()

        for col in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            n_outliers = outliers.sum()

            if n_outliers > 0:
                logger.warning(f"列 {col} 发现 {n_outliers} 个异常值")
                # 用边界值截断（winsorization）
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound

        return df_clean

    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据类型转换

        - 时间字段 -> datetime
        - 分类变量 -> category
        """
        df_clean = df.copy()

        # 转换时间字段
        time_cols = [col for col in df_clean.columns if "time" in col.lower() or "date" in col.lower()]
        for col in time_cols:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
                logger.info(f"转换 {col} 为 datetime 类型")
            except Exception as e:
                logger.warning(f"无法转换 {col} 为 datetime: {e}")

        # 转换低基数分类变量
        for col in df_clean.select_dtypes(include=["object"]).columns:
            unique_ratio = df_clean[col].nunique() / len(df_clean)
            if unique_ratio < 0.1:  # 唯一值占比小于10%
                df_clean[col] = df_clean[col].astype("category")

        return df_clean

    def filter_cohort(
        self,
        df: pd.DataFrame,
        min_follow_up: int = 30,
        min_visits: int = 2,
    ) -> pd.DataFrame:
        """
        筛选符合队列要求的患者

        Parameters
        ----------
        df : pd.DataFrame
            患者数据
        min_follow_up : int
            最小随访天数
        min_visits : int
            最小就诊次数

        Returns
        -------
        pd.DataFrame
            筛选后的数据
        """
        if "patient_id" not in df.columns:
            raise ValueError("数据必须包含 patient_id 列")

        # 计算每个患者的就诊次数和随访时长
        patient_stats = df.groupby("patient_id").agg(
            n_visits=("patient_id", "count"),
            follow_up_days=("time", lambda x: (x.max() - x.min()).days if pd.api.types.is_datetime64_any_dtype(x) else 0),
        )

        # 筛选
        valid_patients = patient_stats[
            (patient_stats["n_visits"] >= min_visits)
            & (patient_stats["follow_up_days"] >= min_follow_up)
        ].index

        df_filtered = df[df["patient_id"].isin(valid_patients)].copy()

        n_excluded = df["patient_id"].nunique() - df_filtered["patient_id"].nunique()
        logger.info(f"队列筛选: 排除了 {n_excluded} 名患者")
        logger.info(f"最终队列: {df_filtered['patient_id'].nunique()} 名患者, {len(df_filtered)} 条记录")

        return df_filtered

    def get_preprocessing_report(self) -> str:
        """获取预处理报告"""
        report = "## 数据预处理报告\n\n"
        report += "### 执行的操作\n\n"

        for i, transform in enumerate(self.transformations, 1):
            report += f"{i}. {transform}\n"

        return report
