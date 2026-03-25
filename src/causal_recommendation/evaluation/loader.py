"""
评估数据加载模块

加载用于评估的真实数据（患者特征和真实用药）
"""

import pandas as pd
from pathlib import Path
from typing import List, Set, Tuple


class EvaluationDataLoader:
    """
    评估数据加载器

    加载患者特征数据和真实用药数据
    """

    def __init__(
        self,
        patient_data_path: str = "Data/processed/processed_data.csv",
        medicine_data_path: str = "Data/processed/processed_medicines.csv"
    ):
        """
        初始化加载器

        Parameters
        ----------
        patient_data_path : str
            患者特征数据CSV文件路径
        medicine_data_path : str
            真实用药数据CSV文件路径
        """
        self.patient_data_path = Path(patient_data_path)
        self.medicine_data_path = Path(medicine_data_path)

        # 加载数据
        self._load_data()

        # 过滤药物（延迟到设置图节点后）
        self.common_meds = None
        self.common_meds_original = None

    def _load_data(self):
        """加载患者数据和真实用药数据"""
        print(f"  加载患者数据: {self.patient_data_path}")
        self.df_patients = pd.read_csv(self.patient_data_path)
        print(f"    患者记录数: {len(self.df_patients)}")

        print(f"  加载用药数据: {self.medicine_data_path}")
        self.df_medicines = pd.read_csv(self.medicine_data_path)
        print(f"    用药记录数: {len(self.df_medicines)}")

        # 验证数据
        assert len(self.df_patients) == len(self.df_medicines), \
            f"患者数据({len(self.df_patients)}行)和用药数据({len(self.df_medicines)}行)行数不一致"

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取数据

        Returns
        -------
        df_patients : pd.DataFrame
            患者特征数据
        df_medicines : pd.DataFrame
            真实用药数据
        """
        return self.df_patients, self.df_medicines

    def filter_medicines(
        self,
        graph_meds: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        过滤药物，只保留图中存在的

        Parameters
        ----------
        graph_meds : List[str]
            图中的药物节点列表（格式：med_药品_t）

        Returns
        -------
        common_meds : List[str]
            共同药物（图中格式）
        common_meds_original : List[str]
            共同药物（原始列名）
        """
        # 从图中提取基础药品名
        graph_med_names = set()
        for med in graph_meds:
            if med.startswith('med_') and med.endswith('_t'):
                base_name = med[4:-2]  # 去掉 'med_' 和 '_t'
                graph_med_names.add(base_name)

        # 获取真实数据中的列名
        all_columns = set(self.df_medicines.columns)

        # 找出共同的药品
        common_med_names = sorted(graph_med_names & all_columns)

        # 转换为图中的格式
        common_meds = sorted([f"med_{name}_t" for name in common_med_names])

        # 保存到实例变量
        self.common_meds = common_meds
        self.common_meds_original = common_med_names

        print(f"  图中药物节点数: {len(graph_meds)}")
        print(f"  真实数据药物列数: {len(all_columns)}")
        print(f"  匹配的药物数: {len(common_med_names)}")

        if len(common_med_names) > 0:
            print(f"  匹配药物示例: {common_med_names[:min(10, len(common_med_names))]}")

        return common_meds, common_med_names

    def construct_patient_info(
        self,
        patient_row: pd.Series,
        all_nodes: Set[str]
    ) -> dict:
        """
        从患者数据行构建患者信息

        Parameters
        ----------
        patient_row : pd.Series
            患者数据行
        all_nodes : Set[str]
            图中所有节点集合

        Returns
        -------
        dict
            患者信息字典
        """
        patient_info = {}

        # 性别
        if 'gender' in patient_row.index:
            gender = patient_row['gender']
            if pd.notna(gender):
                patient_info['gender'] = '女' if str(gender) == '女' else '男'

        # 年龄
        if 'age' in patient_row.index and pd.notna(patient_row['age']):
            patient_info['age'] = int(patient_row['age'])

        # 遍历所有列，识别症状和诊断
        for col in patient_row.index:
            if col.startswith(('patient_id', 'checkup_id', 'time', 'gender', 'age')):
                continue

            value = patient_row[col]
            if pd.isna(value) or value == 0:
                continue

            # 症状
            if '证' not in col and not col.startswith('diagnosis_') and not col.startswith('med_'):
                node_name = f"{col}_t"
                if node_name in all_nodes:
                    patient_info[col] = float(value)

            # 诊断
            elif '证' in col:
                node_name = f"diagnosis_{col}_t"
                if node_name in all_nodes:
                    patient_info[col] = float(value)

        return patient_info
