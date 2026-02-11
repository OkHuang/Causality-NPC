"""
症状提取模块

功能：
- 从主诉文本中提取症状实体
- 识别症状严重程度
- 量化症状为数值向量
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SymptomExtractor:
    """症状提取器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化症状提取器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.severity_keywords = self._load_severity_keywords()
        self.symptom_dictionary = self._load_symptom_dictionary()

    def _load_severity_keywords(self) -> Dict[str, int]:
        """
        加载严重程度关键词字典

        Returns:
            dict: {关键词: 严重程度等级}
        """
        return {
            # 0级：无/改善
            "无": 0,
            "改善": 0,
            "好转": 0,
            "消失": 0,
            "缓解": 0,
            "无XX": 0,  # 模板

            # 1级：轻度
            "偶有": 1,
            "轻度": 1,
            "稍": 1,
            "微": 1,

            # 2级：重度
            "明显": 2,
            "剧烈": 2,
            "严重": 2,
            "难以忍受": 2,
            "持续": 2,
            "加重": 2,
        }

    def _load_symptom_dictionary(self) -> Dict[str, List[str]]:
        """
        加载症状词典

        Returns:
            dict: {类别: [症状列表]}
        """
        # 这里应该从配置文件加载，这里提供示例
        return {
            "耳鼻症状": ["耳鸣", "听力下降", "鼻塞", "鼻衄", "回吸性涕血"],
            "全身症状": ["乏力", "消瘦", "发热", "盗汗"],
            "头颈部": ["头痛", "颈部肿块"],
            "消化道": ["恶心", "呕吐", "纳差", "口干"],
            "精神睡眠": ["失眠", "多梦", "焦虑"],
        }

    def extract(self, text: str, use_llm: bool = False) -> Dict[str, int]:
        """
        从文本中提取症状及其严重程度

        Parameters
        ----------
        text : str
            主诉文本
        use_llm : bool
            是否使用LLM提取

        Returns
        -------
        dict
            {症状名: 严重程度}
        """
        if pd.isna(text) or text == "无":
            return {}

        if use_llm:
            return self._extract_with_llm(text)
        else:
            return self._extract_with_rules(text)

    def _extract_with_rules(self, text: str) -> Dict[str, int]:
        """
        基于规则的症状提取
        """
        symptoms = {}

        # 遍历症状词典
        for category, symptom_list in self.symptom_dictionary.items():
            for symptom in symptom_list:
                if symptom in text:
                    # 判断严重程度
                    severity = self._detect_severity(text, symptom)
                    symptoms[symptom] = severity

        return symptoms

    def _detect_severity(self, text: str, symptom: str) -> int:
        """
        检测症状的严重程度

        在症状关键词前后查找程度修饰词
        """
        # 查找症状关键词在文本中的位置
        symptom_pos = text.find(symptom)
        if symptom_pos == -1:
            return 1  # 默认中度

        # 提取症状前后的上下文（前后各5个字符）
        context_start = max(0, symptom_pos - 5)
        context_end = min(len(text), symptom_pos + len(symptom) + 5)
        context = text[context_start:context_end]

        # 检查严重程度关键词
        for keyword, level in self.severity_keywords.items():
            if keyword in context:
                return level

        # 默认为1级
        return 1

    def _extract_with_llm(self, text: str) -> Dict[str, int]:
        """
        使用LLM提取症状

        TODO: 实现LLM调用逻辑
        """
        # 这里需要集成LLM API
        # 可以考虑使用 OpenAI API 或本地模型

        raise NotImplementedError("LLM提取功能待实现")

    def transform(
        self,
        df: pd.DataFrame,
        text_column: str = "chief_complaint",
    ) -> pd.DataFrame:
        """
        将主诉列转换为症状向量

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        text_column : str
            主诉列名

        Returns
        -------
        pd.DataFrame
            包含症状严重程度列的数据
        """
        logger.info("开始提取症状...")

        # 提取所有症状
        symptom_vectors = df[text_column].apply(self.extract)

        # 展开为DataFrame
        symptom_df = pd.json_normalize(symptom_vectors)

        # 填充缺失值为0（无该症状）
        symptom_df = symptom_df.fillna(0)

        # 合并到原始数据
        result = pd.concat([df.reset_index(drop=True), symptom_df], axis=1)

        # 为症状列添加前缀
        symptom_cols = symptom_df.columns
        result = result.rename(columns={col: f"S_{col}" for col in symptom_cols})

        logger.info(f"提取了 {len(symptom_cols)} 个症状特征")

        return result

    def get_symptom_list(self) -> List[str]:
        """
        获取所有症状列表
        """
        all_symptoms = []
        for symptom_list in self.symptom_dictionary.values():
            all_symptoms.extend(symptom_list)
        return all_symptoms
