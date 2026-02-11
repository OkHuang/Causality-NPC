"""
简易版数据加载模块

功能：
1. 读取CSV文件
2. 解析extracted_symptoms JSON字符串
3. 展开症状为多列
"""

import pandas as pd
import json
from typing import Dict, List
import numpy as np


class SimpleDataLoader:
    """简易版数据加载器"""

    def __init__(self, data_path: str):
        """
        初始化数据加载器

        Parameters
        ----------
        data_path : str
            CSV文件路径
        """
        self.data_path = data_path
        self.df_raw = None
        self.df_processed = None

    def load(self) -> pd.DataFrame:
        """
        加载CSV数据

        Returns
        -------
        pd.DataFrame
            原始数据
        """
        print(f"正在加载数据: {self.data_path}")
        self.df_raw = pd.read_csv(self.data_path)
        print(f"数据加载完成，共 {len(self.df_raw)} 行, {len(self.df_raw.columns)} 列")
        return self.df_raw

    def parse_symptoms(self, json_str: str) -> Dict[str, int]:
        """
        解析单个症状JSON字符串

        Parameters
        ----------
        json_str : str
            症状JSON字符串，格式: [{"name": "乏力", "severity": 1, "label": "轻度"}, ...]

        Returns
        -------
        Dict[str, int]
            症状名称到严重程度的映射，缺失则为0
        """
        if pd.isna(json_str) or json_str == "":
            return {}

        try:
            symptoms = json.loads(json_str)
            return {s['name']: s['severity'] for s in symptoms}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"解析症状JSON失败: {e}")
            return {}

    def expand_symptoms(self, df: pd.DataFrame, symptom_col: str = 'extracted_symptoms') -> pd.DataFrame:
        """
        将症状列展开为多列

        Parameters
        ----------
        df : pd.DataFrame
            包含extracted_symptoms列的数据框
        symptom_col : str
            症状列名，默认为'extracted_symptoms'

        Returns
        -------
        pd.DataFrame
            展开症状后的数据框
        """
        print(f"正在展开症状列...")

        # 解析症状
        symptom_dicts = df[symptom_col].apply(self.parse_symptoms)

        # 转换为DataFrame
        symptom_df = pd.DataFrame(symptom_dicts.tolist())

        # 填充缺失值为0（无症状）
        symptom_df = symptom_df.fillna(0)

        # 合并到原数据
        df_processed = pd.concat([df.reset_index(drop=True), symptom_df], axis=1)

        # 统计症状信息
        n_symptoms = len(symptom_df.columns)
        print(f"症状展开完成，共 {n_symptoms} 个症状")
        print(f"症状列表: {list(symptom_df.columns[:10])}..." if n_symptoms > 10 else f"症状列表: {list(symptom_df.columns)}")

        self.df_processed = df_processed
        return df_processed

    def load_and_process(self) -> pd.DataFrame:
        """
        加载并处理数据（一步完成）

        Returns
        -------
        pd.DataFrame
            处理后的数据框
        """
        # 加载原始数据
        df = self.load()

        # 展开症状
        df_processed = self.expand_symptoms(df)

        # 转换时间列为datetime
        if 'time' in df_processed.columns:
            df_processed['time'] = pd.to_datetime(df_processed['time'])
            print(f"时间列转换完成，范围: {df_processed['time'].min()} 到 {df_processed['time'].max()}")

        return df_processed

    def get_symptom_columns(self, df: pd.DataFrame) -> List[str]:
        """
        获取症状列名列表

        Parameters
        ----------
        df : pd.DataFrame
            数据框

        Returns
        -------
        List[str]
            症状列名列表
        """
        # 排除非症状列
        non_symptom_cols = [
            'checkup_id', 'patient_id', 'gender', 'age', 'time',
            'chief_complaint', 'chinese_diagnosis', 'western_diagnosis',
            'western_medicines', 'tongue_condition', 'pulse_condition',
            'physical_examination', 'acupuncture', 'test',
            'chinese_medicines', 'checkup_date', 'extracted_symptoms'
        ]

        symptom_cols = [col for col in df.columns if col not in non_symptom_cols]
        return symptom_cols


def test_loader():
    """测试数据加载器"""
    data_path = r"D:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv"

    loader = SimpleDataLoader(data_path)
    df = loader.load_and_process()

    print("\n=== 数据概览 ===")
    print(df[['patient_id', 'time', 'gender', 'age']].head(10))

    symptom_cols = loader.get_symptom_columns(df)
    print(f"\n=== 症状列 ===")
    print(f"共 {len(symptom_cols)} 个症状")
    print(symptom_cols[:20])

    return df


if __name__ == "__main__":
    test_loader()
