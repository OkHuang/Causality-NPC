"""
简易版时序对构建模块

功能：
1. 按患者分组并按时间排序
2. 构建相邻时间点的配对 (t, t+1)
3. 生成用于因果发现的数据结构
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class SimplePairBuilder:
    """简易版时序对构建器"""

    def __init__(self, df: pd.DataFrame, patient_col: str = 'patient_id', time_col: str = 'time'):
        """
        初始化时序对构建器

        Parameters
        ----------
        df : pd.DataFrame
            包含患者ID和时间列的数据框
        patient_col : str
            患者ID列名
        time_col : str
            时间列名
        """
        self.df = df.copy()
        self.patient_col = patient_col
        self.time_col = time_col
        self.pairs_df = None
        self.pair_stats = {}

    def build_pairs(self) -> pd.DataFrame:
        """
        构建时序对

        Returns
        -------
        pd.DataFrame
            时序对数据框，每行代表一个 (t, t+1) 配对
        """
        print("\n=== 构建时序对 ===")

        # 确保时间列为datetime类型
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

        # 按患者分组并排序
        self.df = self.df.sort_values([self.patient_col, self.time_col])

        # 统计患者就诊次数
        visit_counts = self.df.groupby(self.patient_col).size()
        print(f"总患者数: {len(visit_counts)}")
        print(f"平均就诊次数: {visit_counts.mean():.2f}")
        print(f"最大就诊次数: {visit_counts.max()}")
        print(f"最小就诊次数: {visit_counts.min()}")

        # 只保留有至少2次就诊的患者
        valid_patients = visit_counts[visit_counts >= 2].index
        self.df = self.df[self.df[self.patient_col].isin(valid_patients)]
        print(f"有≥2次就诊的患者数: {len(valid_patients)}")

        # 构建时序对
        pairs = []
        for pid, group in self.df.groupby(self.patient_col):
            group = group.sort_values(self.time_col)
            for i in range(len(group) - 1):
                t = group.iloc[i]
                t1 = group.iloc[i + 1]

                # 计算时间间隔（天数）
                time_delta = (t1[self.time_col] - t[self.time_col]).days

                pair = {
                    'patient_id': pid,
                    'pair_id': f"{pid}_{i}",
                    'time_t': t[self.time_col],
                    'time_t1': t1[self.time_col],
                    'time_delta_days': time_delta,
                }

                # 添加t时刻的静态特征
                pair['gender'] = t['gender']
                pair['age_t'] = t['age']

                # 添加t时刻和t+1时刻的特征（使用列前缀区分）
                for col in self.df.columns:
                    if col not in [self.patient_col, self.time_col, 'gender', 'age']:
                        # t时刻
                        pair[f'{col}_t'] = t[col]
                        # t+1时刻
                        pair[f'{col}_t1'] = t1[col]

                pairs.append(pair)

        self.pairs_df = pd.DataFrame(pairs)

        # 统计信息
        self.pair_stats = {
            'n_pairs': len(self.pairs_df),
            'n_patients': self.pairs_df['patient_id'].nunique(),
            'mean_time_delta': self.pairs_df['time_delta_days'].mean(),
            'median_time_delta': self.pairs_df['time_delta_days'].median(),
        }

        print(f"\n时序对统计:")
        print(f"  总时序对数: {self.pair_stats['n_pairs']}")
        print(f"  涉及患者数: {self.pair_stats['n_patients']}")
        print(f"  平均时间间隔: {self.pair_stats['mean_time_delta']:.1f} 天")
        print(f"  中位时间间隔: {self.pair_stats['median_time_delta']:.1f} 天")

        return self.pairs_df

    def get_variable_groups(self) -> Dict[str, List[str]]:
        """
        获取变量分组

        Returns
        -------
        Dict[str, List[str]]
            变量分组字典
        """
        if self.pairs_df is None:
            raise ValueError("请先调用 build_pairs() 方法")

        # 静态特征
        static_vars = ['gender', 'age_t']

        # t时刻的特征（排除静态和时间列）
        t_vars = [col for col in self.pairs_df.columns if col.endswith('_t') and col not in static_vars]

        # t+1时刻的特征
        t1_vars = [col for col in self.pairs_df.columns if col.endswith('_t1')]

        # 识别不同类型的变量
        symptom_t_vars = [col for col in t_vars if not col.startswith('chinese_') and not col.startswith('western_') and not col.startswith('extracted_')]
        diagnosis_t_vars = [col for col in t_vars if 'diagnosis' in col.lower()]
        medicine_t_vars = [col for col in t_vars if 'medicines' in col.lower()]

        symptom_t1_vars = [col.replace('_t1', '_t') for col in t1_vars if not col.startswith('chinese_') and not col.startswith('western_') and not col.startswith('extracted_')]
        diagnosis_t1_vars = [col for col in t1_vars if 'diagnosis' in col.lower()]
        medicine_t1_vars = [col for col in t1_vars if 'medicines' in col.lower()]

        return {
            'static': static_vars,
            'symptoms_t': symptom_t_vars,
            'diagnosis_t': diagnosis_t_vars,
            'medicines_t': medicine_t_vars,
            'symptoms_t1': symptom_t1_vars,
            'diagnosis_t1': diagnosis_t1_vars,
        }

    def filter_by_time_delta(self, max_days: int = None) -> pd.DataFrame:
        """
        根据时间间隔筛选时序对

        Parameters
        ----------
        max_days : int, optional
            最大时间间隔（天数），None表示不筛选

        Returns
        -------
        pd.DataFrame
            筛选后的时序对数据框
        """
        if max_days is None:
            return self.pairs_df

        n_before = len(self.pairs_df)
        df_filtered = self.pairs_df[self.pairs_df['time_delta_days'] <= max_days].copy()
        n_after = len(df_filtered)

        print(f"\n时间间隔筛选 (≤{max_days}天):")
        print(f"  筛选前: {n_before} 对")
        print(f"  筛选后: {n_after} 对")
        print(f"  排除: {n_before - n_after} 对 ({(n_before - n_after) / n_before * 100:.1f}%)")

        return df_filtered

    def get_summary(self) -> Dict:
        """
        获取时序对的统计摘要

        Returns
        -------
        Dict
            统计信息字典
        """
        return self.pair_stats


def test_pair_builder():
    """测试时序对构建器"""
    from simple_loader import SimpleDataLoader

    # 加载数据
    data_path = r"D:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv"
    loader = SimpleDataLoader(data_path)
    df = loader.load_and_process()

    # 构建时序对
    builder = SimplePairBuilder(df)
    pairs_df = builder.build_pairs()

    print("\n=== 时序对示例 ===")
    print(pairs_df[['patient_id', 'time_t', 'time_t1', 'time_delta_days']].head(10))

    print("\n=== 变量分组 ===")
    groups = builder.get_variable_groups()
    for group_name, vars_list in groups.items():
        print(f"{group_name}: {len(vars_list)} 个变量")

    return pairs_df


if __name__ == "__main__":
    test_pair_builder()
