"""
证型编码模块

功能：
- 将中医证型文本编码为数值特征
- 支持多热编码（multi-hot encoding）
- 处理证型的层次结构
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)


class SyndromeEncoder:
    """证型编码器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化证型编码器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.encoder = MultiLabelBinarizer()
        self.fitted = False

    def _parse_syndrome_string(self, syndrome_str: str) -> List[str]:
        """
        解析证型字符串为列表

        Examples:
        - "气虚血瘀证" -> ["气虚", "血瘀"]
        - "气虚证、血瘀证" -> ["气虚", "血瘀"]
        """
        if pd.isna(syndrome_str) or syndrome_str == "无" or syndrome_str == "":
            return []

        syndrome_str = syndrome_str.strip()

        # 去除"证"字
        syndrome_str = syndrome_str.replace("证", "")

        # 常见分隔符
        separators = ["、", "，", ",", ";", "兼", "伴", "+"]
        syndromes = [syndrome_str]

        for sep in separators:
            if sep in syndrome_str:
                syndromes = [s.strip() for s in syndrome_str.split(sep) if s.strip()]
                break

        # 进一步分解复合证型（如"气虚血瘀" -> ["气虚", "血瘀"]）
        expanded = []
        for syndrome in syndromes:
            # 尝试分解（这里需要证型词典支持）
            parts = self._split_complex_syndrome(syndrome)
            if parts:
                expanded.extend(parts)
            else:
                expanded.append(syndrome)

        return list(set(expanded))  # 去重

    def _split_complex_syndrome(self, syndrome: str) -> List[str]:
        """
        分解复合证型

        Examples:
        - "气虚血瘀" -> ["气虚", "血瘀"]
        - "阴虚火旺" -> ["阴虚", "火旺"]
        """
        # 常见证型要素
        elements = [
            "气虚", "血虚", "阴虚", "阳虚",
            "血瘀", "痰湿", "火热", "气滞",
            "湿阻", "毒热", "寒凝"
        ]

        found = []
        for element in elements:
            if element in syndrome:
                found.append(element)

        return found if len(found) > 1 else []

    def fit(self, syndromes: List[List[str]]):
        """
        拟合编码器

        Parameters
        ----------
        syndromes : list of list
            证型列表的列表
        """
        self.encoder.fit(syndromes)
        self.fitted = True

        logger.info(f"证型编码器已拟合，识别了 {len(self.encoder.classes_)} 个证型类别")

    def transform(self, syndromes: List[List[str]]) -> pd.DataFrame:
        """
        转换证型为编码

        Parameters
        ----------
        syndromes : list of list
            证型列表的列表

        Returns
        -------
        pd.DataFrame
            证型编码矩阵
        """
        if not self.fitted:
            raise ValueError("编码器未拟合，请先调用 fit()")

        encoded = self.encoder.transform(syndromes)
        df = pd.DataFrame(
            encoded,
            columns=[f"D_{cls}" for cls in self.encoder.classes_]
        )

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        syndrome_column: str = "chinese_diagnosis",
    ) -> pd.DataFrame:
        """
        拟合并转换

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        syndrome_column : str
            证型列名

        Returns
        -------
        pd.DataFrame
            包含证型编码的数据
        """
        logger.info("开始编码证型...")

        # 解析证型列表
        syndrome_lists = df[syndrome_column].apply(self._parse_syndrome_string).tolist()

        # 拟合并转换
        self.fit(syndrome_lists)
        encoded_df = self.transform(syndrome_lists)

        # 合并到原始数据
        result = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

        logger.info(f"编码了 {len(encoded_df.columns)} 个证型特征")

        return result

    def filter_rare_syndromes(
        self,
        df: pd.DataFrame,
        min_frequency: int = 5,
    ) -> List[str]:
        """
        筛选高频证型

        Parameters
        ----------
        df : pd.DataFrame
            编码后的数据
        min_frequency : int
            最小样本数

        Returns
        -------
        list
            高频证型列名
        """
        if not self.fitted:
            raise ValueError("编码器未拟合")

        # 统计每个证型的频率
        syndrome_cols = [f"D_{cls}" for cls in self.encoder.classes_]
        syndrome_counts = df[syndrome_cols].sum()

        # 筛选
        frequent_syndromes = syndrome_counts[syndrome_counts >= min_frequency].index.tolist()

        logger.info(f"保留了 {len(frequent_syndromes)}/{len(syndrome_cols)} 个高频证型")

        return frequent_syndromes

    def get_syndrome_statistics(self, df: pd.DataFrame, syndrome_column: str = "chinese_diagnosis") -> pd.DataFrame:
        """
        统计证型分布

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        syndrome_column : str
            证型列名

        Returns
        -------
        pd.DataFrame
            证型统计
        """
        syndrome_lists = df[syndrome_column].apply(self._parse_syndrome_string)

        # 展开所有证型
        all_syndromes = []
        for syndromes in syndrome_lists:
            all_syndromes.extend(syndromes)

        # 统计
        syndrome_counts = pd.Series(all_syndromes).value_counts().reset_index()
        syndrome_counts.columns = ["证型", "频次"]
        syndrome_counts["频率(%)"] = (syndrome_counts["频次"] / len(df) * 100).round(2)

        return syndrome_counts
