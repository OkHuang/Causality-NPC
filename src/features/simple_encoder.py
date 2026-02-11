"""
简易版特征编码模块

功能：
1. 对症状、诊断、药物进行编码
2. 基于频率筛选节点
3. 构建用于因果发现的数据矩阵
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import MultiLabelBinarizer


class SimpleFeatureEncoder:
    """简易版特征编码器"""

    def __init__(self,
                 symptom_threshold: float = 0.03,
                 medicine_threshold: float = 0.05,
                 diagnosis_threshold: float = 0.03,
                 age_bins: int = 5,
                 age_start: int = 20):
        """
        初始化特征编码器

        Parameters
        ----------
        symptom_threshold : float
            症状频率阈值（低于此值的症状将被删除）
        medicine_threshold : float
            药物频率阈值
        diagnosis_threshold : float
            诊断频率阈值
        age_bins : int
            年龄分箱数量
        age_start : int
            年龄分箱起始值
        """
        self.symptom_threshold = symptom_threshold
        self.medicine_threshold = medicine_threshold
        self.diagnosis_threshold = diagnosis_threshold
        self.age_bins = age_bins
        self.age_start = age_start

        self.selected_symptoms = []
        self.selected_medicines = []
        self.selected_diagnoses = []
        self.mlb_diagnosis = None
        self.mlb_medicines = None

    def encode_gender(self, df: pd.DataFrame, col: str = 'gender') -> pd.DataFrame:
        """
        编码性别列

        Parameters
        ----------
        df : pd.DataFrame
            数据框
        col : str
            性别列名

        Returns
        -------
        pd.DataFrame
            编码后的数据框（新增 gender_encoded 列）
        """
        df = df.copy()
        # 女=0, 男=1
        df['gender_encoded'] = (df[col] == '男').astype(int)
        return df

    def encode_age(self, df: pd.DataFrame, col: str = 'age_t') -> pd.DataFrame:
        """
        对年龄进行分箱编码

        Parameters
        ----------
        df : pd.DataFrame
            数据框
        col : str
            年龄列名

        Returns
        -------
        pd.DataFrame
            编码后的数据框（新增 age_binned 列）
        """
        df = df.copy()

        # 创建分箱
        bins = range(self.age_start, df[col].max() + 10, 5)
        labels = [f"[{bins[i]}-{bins[i+1]})" for i in range(len(bins) - 1)]

        df['age_binned'] = pd.cut(df[col], bins=bins, labels=labels, right=False)

        # 创建虚拟变量
        age_dummies = pd.get_dummies(df['age_binned'], prefix='age', drop_first=True)
        df = pd.concat([df, age_dummies], axis=1)

        print(f"年龄分箱: {len(labels)} 个区间")
        print(f"区间: {labels[:5]}..." if len(labels) > 5 else f"区间: {labels}")

        return df

    def encode_diagnosis(self, df: pd.DataFrame,
                        chinese_col: str = 'chinese_diagnosis_t',
                        western_col: str = 'western_diagnosis_t') -> pd.DataFrame:
        """
        对诊断进行多热编码

        Parameters
        ----------
        df : pd.DataFrame
            数据框
        chinese_col : str
            中医诊断列名
        western_col : str
            西医诊断列名

        Returns
        -------
        pd.DataFrame
            编码后的数据框
        """
        df = df.copy()

        # 检查列是否存在
        if chinese_col not in df.columns or western_col not in df.columns:
            print(f"警告: 诊断列 {chinese_col} 或 {western_col} 不存在，跳过诊断编码")
            return df

        # 提取时间后缀
        chinese_suffix = ''
        western_suffix = ''
        if chinese_col.endswith('_t') or chinese_col.endswith('_t1'):
            chinese_suffix = chinese_col[chinese_col.rfind('_'):]
        if western_col.endswith('_t') or western_col.endswith('_t1'):
            western_suffix = western_col[western_col.rfind('_'):]

        # 合并中医和西医诊断
        # 解析诊断字符串（空格分隔）
        df[chinese_col + '_list'] = df[chinese_col].fillna('').apply(
            lambda x: [d.strip() for d in str(x).split() if d.strip()]
        )
        df[western_col + '_list'] = df[western_col].fillna('').apply(
            lambda x: [d.strip() for d in str(x).split() if d.strip()]
        )

        # 使用MultiLabelBinarizer进行多热编码
        self.mlb_diagnosis = MultiLabelBinarizer()

        # 合并中西医诊断列表
        all_diagnoses = df[chinese_col + '_list'] + df[western_col + '_list']

        # 拟合并转换
        diagnosis_encoded = self.mlb_diagnosis.fit_transform(all_diagnoses)

        # 创建带时间后缀的列名（中医诊断用中文后缀，西医诊断用西文后缀）
        diagnosis_cols = []
        for d in self.mlb_diagnosis.classes_:
            # 这里简单处理：所有诊断都加上后缀
            diagnosis_cols.append(f'diagnosis_{d}{chinese_suffix}')

        # 添加到数据框
        df[diagnosis_cols] = diagnosis_encoded

        # 清理临时列
        df = df.drop(columns=[chinese_col + '_list', western_col + '_list'])

        print(f"诊断编码: {len(diagnosis_cols)} 个诊断")

        return df

    def encode_medicines(self, df: pd.DataFrame,
                        col: str = 'chinese_medicines_t') -> pd.DataFrame:
        """
        对药物进行多热编码

        Parameters
        ----------
        df : pd.DataFrame
            数据框
        col : str
            药物列名

        Returns
        -------
        pd.DataFrame
            编码后的数据框
        """
        df = df.copy()

        # 检查列是否存在
        if col not in df.columns:
            print(f"警告: 药物列 {col} 不存在，跳过药物编码")
            return df

        # 提取时间后缀（如果有）
        time_suffix = ''
        if col.endswith('_t') or col.endswith('_t1'):
            time_suffix = col[col.rfind('_'):]  # 获取最后一个 _ 之后的部分

        # 解析药物字符串（空格分隔）
        df[col + '_list'] = df[col].fillna('').apply(
            lambda x: [m.strip() for m in str(x).split() if m.strip()]
        )

        # 使用MultiLabelBinarizer
        self.mlb_medicines = MultiLabelBinarizer()
        medicines_encoded = self.mlb_medicines.fit_transform(df[col + '_list'])

        # 创建带时间后缀的列名
        medicine_cols = [f'med_{m}{time_suffix}' for m in self.mlb_medicines.classes_]

        # 添加到数据框
        df[medicine_cols] = medicines_encoded

        # 清理临时列
        df = df.drop(columns=[col + '_list'])

        print(f"药物编码: {len(medicine_cols)} 个药物")

        return df

    def filter_by_frequency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        根据频率筛选特征

        Parameters
        ----------
        df : pd.DataFrame
            编码后的数据框

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            筛选后的数据框和统计信息
        """
        print("\n=== 基于频率筛选节点 ===")

        stats = {}

        # 获取数值类型的列（症状、编码后的诊断、编码后的药物）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 筛选症状（排除已知的非症状列）
        exclude_cols = [
            'checkup_id', 'patient_id', 'pair_id',
            'gender_encoded', 'age_t',
            'time_delta_days'
        ]
        # 排除以age_开头的列
        symptom_cols = [col for col in numeric_cols if col not in exclude_cols
                       and not col.startswith('age_')
                       and not col.startswith('med_')
                       and not col.startswith('diagnosis_')
                       and not col.startswith('time_')]

        if symptom_cols:
            symptom_freq = (df[symptom_cols] > 0).mean()
            self.selected_symptoms = symptom_freq[symptom_freq >= self.symptom_threshold].index.tolist()
            stats['symptoms'] = {
                'n_original': len(symptom_cols),
                'n_selected': len(self.selected_symptoms),
                'removed': len(symptom_cols) - len(self.selected_symptoms)
            }
            print(f"症状: {stats['symptoms']['n_original']} -> {stats['symptoms']['n_selected']} "
                  f"(移除 {stats['symptoms']['removed']})")

        # 筛选药物
        medicine_cols = [col for col in numeric_cols if col.startswith('med_')]
        if medicine_cols:
            medicine_freq = (df[medicine_cols] > 0).mean()
            self.selected_medicines = medicine_freq[medicine_freq >= self.medicine_threshold].index.tolist()
            stats['medicines'] = {
                'n_original': len(medicine_cols),
                'n_selected': len(self.selected_medicines),
                'removed': len(medicine_cols) - len(self.selected_medicines)
            }
            print(f"药物: {stats['medicines']['n_original']} -> {stats['medicines']['n_selected']} "
                  f"(移除 {stats['medicines']['removed']})")

        # 筛选诊断
        diagnosis_cols = [col for col in numeric_cols if col.startswith('diagnosis_')]
        if diagnosis_cols:
            diagnosis_freq = (df[diagnosis_cols] > 0).mean()
            self.selected_diagnoses = diagnosis_freq[diagnosis_freq >= self.diagnosis_threshold].index.tolist()
            stats['diagnoses'] = {
                'n_original': len(diagnosis_cols),
                'n_selected': len(self.selected_diagnoses),
                'removed': len(diagnosis_cols) - len(self.selected_diagnoses)
            }
            print(f"诊断: {stats['diagnoses']['n_original']} -> {stats['diagnoses']['n_selected']} "
                  f"(移除 {stats['diagnoses']['removed']})")

        return df, stats

    def get_selected_columns(self, include_t1: bool = True) -> List[str]:
        """
        获取筛选后的列名列表（基于预筛选结果）

        Parameters
        ----------
        include_t1 : bool
            是否包含t+1时刻的变量

        Returns
        -------
        List[str]
            列名列表
        """
        selected_flat = ['gender_encoded']

        # 使用预筛选的结果
        if hasattr(self, 'prefiltered_symptoms_t'):
            # t时刻症状
            selected_flat.extend(self.prefiltered_symptoms_t)

            # t时刻药物和诊断
            selected_flat.extend([f'med_{m}_t' for m in self.prefiltered_medicines_t])
            selected_flat.extend([f'diagnosis_{d}_t' for d in self.prefiltered_diagnoses_t])

            if include_t1:
                # t1时刻症状
                selected_flat.extend(self.prefiltered_symptoms_t1)
                # t1时刻诊断
                selected_flat.extend([f'diagnosis_{d}_t1' for d in self.prefiltered_diagnoses_t1])
                # 注意：不包含t+1时刻的药物
        else:
            # 降级到旧的筛选逻辑
            selected_flat.extend([col for col in self.selected_symptoms if col.endswith('_t')])
            selected_flat.extend([col for col in self.selected_medicines if col.endswith('_t')])
            selected_flat.extend([col for col in self.selected_diagnoses if col.endswith('_t')])

            if include_t1:
                symptom_t1 = [col.replace('_t', '_t1') for col in self.selected_symptoms if col.endswith('_t')]
                selected_flat.extend(symptom_t1)
                selected_flat.extend([col for col in self.selected_diagnoses if col.endswith('_t1')])

        # 添加age相关的列
        age_cols = [col for col in getattr(self, 'selected_symptoms', []) if col.startswith('age_')]
        selected_flat.extend(age_cols)

        return selected_flat

    def prefilter_by_frequency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        在编码之前基于频率筛选原始文本数据

        Parameters
        ----------
        df : pd.DataFrame
            时序对数据框（未编码）

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            筛选后的数据框和统计信息
        """
        print("\n=== 编码前频率筛选（优化效率） ===")

        stats = {}

        # 定义非症状列（文本列和其他不相关列）
        non_symptom_patterns = [
            'chinese_', 'western_', 'extracted_', 'age_', 'gender_',
            'time_', 'checkup_id', 'patient_id', 'pair_id',
            'chief_complaint', 'tongue_condition', 'pulse_condition',
            'physical_examination', 'acupuncture', 'test'
        ]

        # 1. 筛选症状（t时刻）- 只选择数值类型的症状列
        print("\n--- 筛选症状 ---")
        symptom_cols_t = []
        for col in df.columns:
            if col.endswith('_t'):
                # 检查是否是非症状列
                is_non_symptom = any(pattern in col for pattern in non_symptom_patterns)
                if not is_non_symptom and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    symptom_cols_t.append(col)

        # 计算症状频率
        symptom_freq = {}
        for col in symptom_cols_t:
            freq = (df[col] > 0).mean()
            symptom_freq[col] = freq

        # 保留高频症状
        high_freq_symptoms = [col for col, freq in symptom_freq.items() if freq >= self.symptom_threshold]
        low_freq_symptoms = [col for col, freq in symptom_freq.items() if freq < self.symptom_threshold]

        stats['symptoms'] = {
            'n_original': len(symptom_cols_t),
            'n_selected': len(high_freq_symptoms),
            'removed': len(low_freq_symptoms),
            'removed_list': low_freq_symptoms
        }
        print(f"症状_t: {stats['symptoms']['n_original']} -> {stats['symptoms']['n_selected']} (移除 {stats['symptoms']['removed']})")

        # 对应的t1时刻症状也需要一起筛选
        symptom_cols_t1 = [col.replace('_t', '_t1') for col in high_freq_symptoms if col.replace('_t', '_t1') in df.columns]

        # 2. 筛选诊断（t时刻）
        print("\n--- 筛选诊断 ---")
        all_diagnoses_t = set()
        for d in df['chinese_diagnosis_t'].dropna():
            for diag in str(d).split():
                all_diagnoses_t.add(diag)
        for d in df['western_diagnosis_t'].dropna():
            for diag in str(d).split():
                all_diagnoses_t.add(diag)

        # 计算每个诊断的频率
        diagnosis_freq = {}
        for diag in all_diagnoses_t:
            count = 0
            for d in df['chinese_diagnosis_t'].dropna():
                if diag in str(d).split():
                    count += 1
            for d in df['western_diagnosis_t'].dropna():
                if diag in str(d).split():
                    count += 1
            diagnosis_freq[diag] = count / len(df)

        high_freq_diagnoses = [diag for diag, freq in diagnosis_freq.items() if freq >= self.diagnosis_threshold]
        low_freq_diagnoses = [diag for diag, freq in diagnosis_freq.items() if freq < self.diagnosis_threshold]

        stats['diagnoses'] = {
            'n_original': len(all_diagnoses_t),
            'n_selected': len(high_freq_diagnoses),
            'removed': len(low_freq_diagnoses),
            'selected_list': high_freq_diagnoses,
            'removed_list': low_freq_diagnoses
        }
        print(f"诊断_t: {stats['diagnoses']['n_original']} -> {stats['diagnoses']['n_selected']} (移除 {stats['diagnoses']['removed']})")

        # 统计t1时刻的诊断频率
        all_diagnoses_t1 = set()
        for d in df['chinese_diagnosis_t1'].dropna():
            for diag in str(d).split():
                all_diagnoses_t1.add(diag)
        for d in df['western_diagnosis_t1'].dropna():
            for diag in str(d).split():
                all_diagnoses_t1.add(diag)

        # 同样筛选t1时刻的诊断
        diagnosis_freq_t1 = {}
        for diag in all_diagnoses_t1:
            count = 0
            for d in df['chinese_diagnosis_t1'].dropna():
                if diag in str(d).split():
                    count += 1
            for d in df['western_diagnosis_t1'].dropna():
                if diag in str(d).split():
                    count += 1
            diagnosis_freq_t1[diag] = count / len(df)

        high_freq_diagnoses_t1 = [diag for diag, freq in diagnosis_freq_t1.items() if freq >= self.diagnosis_threshold]
        low_freq_diagnoses_t1 = [diag for diag, freq in diagnosis_freq_t1.items() if freq < self.diagnosis_threshold]

        stats['diagnoses_t1'] = {
            'n_original': len(all_diagnoses_t1),
            'n_selected': len(high_freq_diagnoses_t1),
            'removed': len(low_freq_diagnoses_t1),
            'removed_list': low_freq_diagnoses_t1
        }
        print(f"诊断_t1: {stats['diagnoses_t1']['n_original']} -> {stats['diagnoses_t1']['n_selected']} (移除 {stats['diagnoses_t1']['removed']})")

        # 3. 筛选药物（t时刻）
        print("\n--- 筛选药物 ---")
        all_medicines_t = set()
        for m in df['chinese_medicines_t'].dropna():
            for med in str(m).split():
                all_medicines_t.add(med)

        # 计算每个药物的频率
        medicine_freq = {}
        for med in all_medicines_t:
            count = sum([1 for m in df['chinese_medicines_t'].dropna() if med in str(m).split()])
            medicine_freq[med] = count / len(df)

        high_freq_medicines = [med for med, freq in medicine_freq.items() if freq >= self.medicine_threshold]
        low_freq_medicines = [med for med, freq in medicine_freq.items() if freq < self.medicine_threshold]

        stats['medicines'] = {
            'n_original': len(all_medicines_t),
            'n_selected': len(high_freq_medicines),
            'removed': len(low_freq_medicines),
            'selected_list': high_freq_medicines,
            'removed_list': low_freq_medicines
        }
        print(f"药物_t: {stats['medicines']['n_original']} -> {stats['medicines']['n_selected']} (移除 {stats['medicines']['removed']})")

        # 保存筛选结果供后续编码使用
        self.prefiltered_symptoms_t = high_freq_symptoms
        self.prefiltered_symptoms_t1 = symptom_cols_t1
        self.prefiltered_diagnoses_t = high_freq_diagnoses
        self.prefiltered_diagnoses_t1 = high_freq_diagnoses_t1
        self.prefiltered_medicines_t = high_freq_medicines

        return df, stats

    def encode_all(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        执行所有编码步骤

        Parameters
        ----------
        df : pd.DataFrame
            原始时序对数据框

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            编码后的数据框和统计信息
        """
        print("\n=== 特征编码 ===")

        # 0. 编码前预筛选（优化效率）
        df, prefilter_stats = self.prefilter_by_frequency(df)

        # 1. 编码性别
        df = self.encode_gender(df)

        # 2. 编码年龄
        df = self.encode_age(df)

        # 3. 编码诊断（t时刻）- 只编码筛选后的诊断
        df = self.encode_diagnosis_filtered(df, chinese_col='chinese_diagnosis_t', western_col='western_diagnosis_t',
                                            allowed_diagnoses=self.prefiltered_diagnoses_t, suffix='_t')

        # 4. 编码药物（t时刻）- 只编码筛选后的药物
        df = self.encode_medicines_filtered(df, col='chinese_medicines_t',
                                           allowed_medicines=self.prefiltered_medicines_t, suffix='_t')

        # 5. 编码诊断（t1时刻）- 只编码筛选后的诊断
        if 'chinese_diagnosis_t1' in df.columns and 'western_diagnosis_t1' in df.columns:
            df = self.encode_diagnosis_filtered(df, chinese_col='chinese_diagnosis_t1', western_col='western_diagnosis_t1',
                                                allowed_diagnoses=self.prefiltered_diagnoses_t1, suffix='_t1')

        # 6. 编码药物（t1时刻） - 已注释：不考虑t+1时刻的药物
        # if 'chinese_medicines_t1' in df.columns:
        #     df = self.encode_medicines_filtered(df, col='chinese_medicines_t1',
        #                                        allowed_medicines=self.prefiltered_medicines_t1, suffix='_t1')

        # 7. 筛选症状列（只保留预筛选后的症状）
        symptom_cols_to_keep = self.prefiltered_symptoms_t + self.prefiltered_symptoms_t1
        all_symptom_cols = [col for col in df.columns if col.endswith('_t') or col.endswith('_t1')]
        symptom_cols_to_drop = [col for col in all_symptom_cols if col not in symptom_cols_to_keep
                               and not col.startswith('diagnosis_') and not col.startswith('med_')
                               and not col.startswith('age_') and not col.startswith('time_')
                               and col not in ['gender_encoded', 'checkup_id', 'patient_id', 'pair_id']]

        if symptom_cols_to_drop:
            df = df.drop(columns=symptom_cols_to_drop)
            print(f"\n删除低频症状列: {len(symptom_cols_to_drop)} 列")

        # 8. 汇总统计信息
        stats = {
            'prefilter': prefilter_stats,
            'final': {
                'symptoms': prefilter_stats['symptoms'],
                'medicines': prefilter_stats['medicines'],
                'diagnoses': {
                    'n_original': prefilter_stats['diagnoses']['n_original'] + prefilter_stats['diagnoses_t1']['n_original'],
                    'n_selected': prefilter_stats['diagnoses']['n_selected'] + prefilter_stats['diagnoses_t1']['n_selected'],
                    'removed': prefilter_stats['diagnoses']['removed'] + prefilter_stats['diagnoses_t1']['removed']
                }
            }
        }

        print(f"\n编码完成，数据框形状: {df.shape}")

        return df, stats

    def encode_diagnosis_filtered(self, df: pd.DataFrame, chinese_col: str, western_col: str,
                                  allowed_diagnoses: list, suffix: str) -> pd.DataFrame:
        """
        对诊断进行多热编码（只编码允许的诊断）

        Parameters
        ----------
        df : pd.DataFrame
            数据框
        chinese_col : str
            中医诊断列名
        western_col : str
            西医诊断列名
        allowed_diagnoses : list
            允许编码的诊断列表
        suffix : str
            列名后缀

        Returns
        -------
        pd.DataFrame
            编码后的数据框
        """
        df = df.copy()

        # 检查列是否存在
        if chinese_col not in df.columns or western_col not in df.columns:
            print(f"警告: 诊断列 {chinese_col} 或 {western_col} 不存在，跳过诊断编码")
            return df

        # 为每个允许的诊断创建列
        for diag in allowed_diagnoses:
            col_name = f'diagnosis_{diag}{suffix}'
            df[col_name] = 0

            # 从中医诊断中标记
            for idx, val in df[chinese_col].items():
                if pd.notna(val) and diag in str(val).split():
                    df.at[idx, col_name] = 1

            # 从西医诊断中标记
            for idx, val in df[western_col].items():
                if pd.notna(val) and diag in str(val).split():
                    df.at[idx, col_name] = 1

        print(f"诊断编码({suffix}): {len(allowed_diagnoses)} 个诊断")

        return df

    def encode_medicines_filtered(self, df: pd.DataFrame, col: str,
                                  allowed_medicines: list, suffix: str) -> pd.DataFrame:
        """
        对药物进行多热编码（只编码允许的药物）

        Parameters
        ----------
        df : pd.DataFrame
            数据框
        col : str
            药物列名
        allowed_medicines : list
            允许编码的药物列表
        suffix : str
            列名后缀

        Returns
        -------
        pd.DataFrame
            编码后的数据框
        """
        df = df.copy()

        # 检查列是否存在
        if col not in df.columns:
            print(f"警告: 药物列 {col} 不存在，跳过药物编码")
            return df

        # 为每个允许的药物创建列
        for med in allowed_medicines:
            col_name = f'med_{med}{suffix}'
            df[col_name] = 0

            # 标记
            for idx, val in df[col].items():
                if pd.notna(val) and med in str(val).split():
                    df.at[idx, col_name] = 1

        print(f"药物编码({suffix}): {len(allowed_medicines)} 个药物")

        return df


def test_encoder():
    """测试特征编码器"""
    from simple_loader import SimpleDataLoader
    from simple_pair_builder import SimplePairBuilder

    # 加载数据并构建时序对
    data_path = r"D:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv"
    loader = SimpleDataLoader(data_path)
    df = loader.load_and_process()

    builder = SimplePairBuilder(df)
    pairs_df = builder.build_pairs()

    # 特征编码
    encoder = SimpleFeatureEncoder()
    encoded_df, stats = encoder.encode_all(pairs_df)

    print("\n=== 编码统计 ===")
    for var_type, stat in stats.items():
        print(f"{var_type}: {stat}")

    print("\n=== 选中的列 ===")
    selected_cols = encoder.get_selected_columns()
    print(f"共 {len(selected_cols)} 列")
    print(selected_cols[:20])

    return encoded_df


if __name__ == "__main__":
    test_encoder()
