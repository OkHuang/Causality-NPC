"""
药物映射模块

功能：
- 将具体药物映射到功效类别
- 计算功效强度
- 生成药物特征向量
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any
import logging

logger = logging.getLogger(__name__)


class MedicineMapper:
    """药物映射器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化药物映射器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.herb_to_category = self._load_herb_mapping()
        self.dosage_standard = self._load_dosage_standard()

    def _load_herb_mapping(self) -> Dict[str, str]:
        """
        加载药物到功效类别的映射

        Returns:
            dict: {药物名: 功效类别}
        """
        # 这里应该从配置文件加载，这里提供示例
        return {
            # 补气药
            "黄芪": "补气药",
            "白术": "补气药",
            "党参": "补气药",
            "甘草": "补气药",

            # 养血药
            "当归": "养血药",
            "熟地黄": "养血药",
            "白芍": "养血药",
            "阿胶": "养血药",

            # 滋阴药
            "北沙参": "滋阴药",
            "麦冬": "滋阴药",
            "石斛": "滋阴药",
            "玉竹": "滋阴药",

            # 温阳药
            "附子": "温阳药",
            "肉桂": "温阳药",
            "淫羊藿": "温阳药",

            # 活血化瘀药
            "丹参": "活血化瘀药",
            "川芎": "活血化瘀药",
            "红花": "活血化瘀药",
            "桃仁": "活血化瘀药",

            # 清热解毒药
            "白花蛇舌草": "清热解毒药",
            "半枝莲": "清热解毒药",
            "野菊花": "清热解毒药",
            "金银花": "清热解毒药",

            # 化痰止咳药
            "半夏": "化痰止咳药",
            "陈皮": "化痰止咳药",
            "茯苓": "化痰止咳药",

            # 理气药
            "柴胡": "理气药",
            "香附": "理气药",
            "枳壳": "理气药",

            # 安神药
            "酸枣仁": "安神药",
            "远志": "安神药",
            "龙骨": "安神药",
        }

    def _load_dosage_standard(self) -> Dict[str, float]:
        """
        加载药物标准剂量

        Returns:
            dict: {药物名: 标准剂量(g)}
        """
        return {
            "黄芪": 30.0,
            "白术": 15.0,
            "丹参": 15.0,
            "当归": 10.0,
            # ... 更多药物
        }

    def map_medicines(self, medicine_list: List[str]) -> Dict[str, float]:
        """
        将药物列表映射到功效强度

        Parameters
        ----------
        medicine_list : list
            药物列表

        Returns
        -------
        dict
            {功效类别: 强度}
        """
        category_strength = {}

        for medicine in medicine_list:
            # 去除空格
            medicine = medicine.strip()

            # 查找映射
            category = self.herb_to_category.get(medicine)

            if category:
                # 累加强度（简单计数）
                category_strength[category] = category_strength.get(category, 0) + 1

        return category_strength

    def transform(
        self,
        df: pd.DataFrame,
        medicine_column: str = "chinese_medicines",
    ) -> pd.DataFrame:
        """
        将药物列转换为功效特征向量

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        medicine_column : str
            药物列名

        Returns
        -------
        pd.DataFrame
            包含功效强度列的数据
        """
        logger.info("开始映射药物...")

        # 解析药物列表
        medicine_vectors = df[medicine_column].apply(self._parse_medicine_string)

        # 映射到功效类别
        category_vectors = medicine_vectors.apply(self.map_medicines)

        # 展开为DataFrame
        category_df = pd.json_normalize(category_vectors)

        # 填充缺失值为0
        category_df = category_df.fillna(0)

        # 合并到原始数据
        result = pd.concat([df.reset_index(drop=True), category_df], axis=1)

        # 为功效列添加前缀
        category_cols = category_df.columns
        result = result.rename(columns={col: f"M_{col}" for col in category_cols})

        logger.info(f"映射为 {len(category_cols)} 个功效类别")

        return result

    def _parse_medicine_string(self, medicine_str: str) -> List[str]:
        """
        解析药物字符串为列表

        Examples:
        - "[黄芪, 白术, 丹参]" -> ["黄芪", "白术", "丹参"]
        - "黄芪、白术、丹参" -> ["黄芪", "白术", "丹参"]
        """
        if pd.isna(medicine_str) or medicine_str == "无":
            return []

        medicine_str = medicine_str.strip()

        # 尝试解析列表格式
        if medicine_str.startswith("[") and medicine_str.endswith("]"):
            try:
                medicines = eval(medicine_str)
                if isinstance(medicines, list):
                    return [m.strip() for m in medicines]
            except:
                pass

        # 尝试分隔符格式
        for sep in [",", "、", ";", "\n"]:
            if sep in medicine_str:
                return [m.strip() for m in medicine_str.split(sep) if m.strip()]

        # 单个药物
        return [medicine_str]

    def get_herb_statistics(self, df: pd.DataFrame, medicine_column: str = "chinese_medicines") -> pd.DataFrame:
        """
        统计药物使用频率

        Parameters
        ----------
        df : pd.DataFrame
            数据
        medicine_column : str
            药物列名

        Returns
        -------
        pd.DataFrame
            药物使用统计
        """
        all_medicines = []

        for medicine_list in df[medicine_column].apply(self._parse_medicine_string):
            all_medicines.extend(medicine_list)

        # 统计
        medicine_counts = pd.Series(all_medicines).value_counts().reset_index()
        medicine_counts.columns = ["药物", "频次"]
        medicine_counts["频率(%)"] = (medicine_counts["频次"] / len(df) * 100).round(2)

        return medicine_counts
