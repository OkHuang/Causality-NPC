"""
推荐系统评估器

评估推荐系统的性能
"""

import networkx as nx
import pandas as pd
from typing import Dict, List
from pathlib import Path

from ..data.loader import RecommendationLoader
from ..data.patient_encoder import extract_mapping_rules
from ..recommendation.propagation import causal_propagation_recommend
from .loader import EvaluationDataLoader
from .metrics import calculate_metrics, aggregate_metrics


class RecommendationEvaluator:
    """
    推荐系统评估器

    计算推荐系统的分类评价指标
    """

    def __init__(
        self,
        config,
        patient_data_path: str = "Data/processed/processed_data.csv",
        medicine_data_path: str = "Data/processed/processed_medicines.csv"
    ):
        """
        初始化评估器

        Parameters
        ----------
        config : NPCConfig
            配置对象
        patient_data_path : str
            患者特征数据路径
        medicine_data_path : str
            真实用药数据路径
        """
        self.config = config
        self.threshold_positive = config.recommendation.threshold_positive
        self.threshold_negative = config.recommendation.threshold_negative

        # 加载图和ATE
        loader = RecommendationLoader(config)
        self.graph, self.ate_dict = loader.load_all()
        self.all_nodes = set(self.graph.nodes())

        # 提取映射规则
        self.mapping_rules = extract_mapping_rules(self.all_nodes)
        self.med_nodes = sorted([n for n in self.all_nodes if n.startswith('med_')])

        # 加载评估数据
        self.eval_loader = EvaluationDataLoader(patient_data_path, medicine_data_path)
        self.df_patients, self.df_medicines = self.eval_loader.get_data()

        # 过滤药物
        self.common_meds, self.common_meds_original = self.eval_loader.filter_medicines(
            self.med_nodes
        )

    def evaluate_patient(self, patient_idx: int) -> Dict:
        """
        评估单个患者

        Parameters
        ----------
        patient_idx : int
            患者数据索引

        Returns
        -------
        Dict
            评估结果
        """
        # 获取患者数据
        patient_row = self.df_patients.iloc[patient_idx]
        medicine_row = self.df_medicines.iloc[patient_idx]

        # 构建患者信息
        patient_info = self.eval_loader.construct_patient_info(
            patient_row, self.all_nodes
        )

        # 执行推荐
        result = causal_propagation_recommend(
            graph=self.graph,
            ate_dict=self.ate_dict,
            patient_info=patient_info,
            all_nodes=self.all_nodes,
            mapping_rules=self.mapping_rules,
            threshold_positive=self.threshold_positive,
            threshold_negative=self.threshold_negative
        )

        # 计算指标
        metrics = calculate_metrics(
            pred_scores=result['all_scores'],
            true_medicines=medicine_row,
            common_meds=self.common_meds,
            common_meds_original=self.common_meds_original,
            threshold_positive=self.threshold_positive
        )

        return {
            'patient_idx': patient_idx,
            'patient_info': patient_info,
            'metrics': metrics,
            'recommendation': result
        }

    def evaluate_batch(
        self,
        patient_indices: List[int] = None,
        max_patients = None
    ) -> Dict:
        """
        批量评估

        Parameters
        ----------
        patient_indices : List[int], optional
            患者索引列表
        max_patients : int, optional
            最大评估患者数

        Returns
        -------
        Dict
            批量评估结果
        """
        if patient_indices is None:
            patient_indices = range(len(self.df_patients))

        if max_patients is not None:
            patient_indices = list(patient_indices)[:max_patients]

        print(f"\n评估患者数: {len(patient_indices)}")

        results = []
        failed_count = 0

        for idx in patient_indices:
            try:
                result = self.evaluate_patient(idx)
                results.append(result)

                # 进度提示（每50个患者）
                if len(results) % 50 == 0:
                    print(f"  进度: {len(results)}/{len(patient_indices)}")

            except Exception as e:
                failed_count += 1
                print(f"  评估患者 {idx} 失败: {e}")
                continue

        print(f"  成功: {len(results)}, 失败: {failed_count}")

        # 汇总指标
        aggregated_metrics = aggregate_metrics(results)

        return {
            'threshold_positive': self.threshold_positive,
            'threshold_negative': self.threshold_negative,
            'num_patients': len(results),
            'num_failed': failed_count,
            'common_meds': self.common_meds,
            'patient_results': results,
            'aggregated_metrics': aggregated_metrics
        }

    def print_evaluation_report(self, eval_result: Dict):
        """
        打印评估报告

        Parameters
        ----------
        eval_result : Dict
            评估结果
        """
        print("\n" + "="*80)
        print("推荐系统评估报告")
        print("="*80)

        print(f"\n配置:")
        print(f"  推荐阈值: score >= {self.threshold_positive}")
        print(f"  警告阈值: score <= {self.threshold_negative}")
        print(f"  评估患者数: {eval_result['num_patients']}")
        if eval_result.get('num_failed', 0) > 0:
            print(f"  失败患者数: {eval_result['num_failed']}")
        print(f"  共同药物数: {len(eval_result['common_meds'])}")

        print(f"\n{'='*80}")
        print("微观平均 (Micro-Average)")
        print(f"{'='*80}")

        metrics = eval_result['aggregated_metrics']['micro']
        print(f"\n  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision):  {metrics['precision']:.4f}")
        print(f"  召回率 (Recall):     {metrics['recall']:.4f}")
        print(f"  F1分数 (F1-Score):   {metrics['f1']:.4f}")
        print(f"  MCC系数:            {metrics['mcc']:.4f}")

        print(f"\n  混淆矩阵:")
        print(f"    预测\\实际    用药(1)    不用药(0)")
        print(f"    推荐(1)      TP={metrics['tp']:>6}    FP={metrics['fp']:>6}")
        print(f"    不推荐(0)    FN={metrics['fn']:>6}    TN={metrics['tn']:>6}")

        print(f"\n{'='*80}")
        print("宏观平均 (Macro-Average)")
        print(f"{'='*80}")

        metrics = eval_result['aggregated_metrics']['macro']
        print(f"\n  准确率:  {metrics['accuracy']:.4f}")
        print(f"  精确率:  {metrics['precision']:.4f}")
        print(f"  召回率:  {metrics['recall']:.4f}")
        print(f"  F1分数:  {metrics['f1']:.4f}")
        print(f"  MCC系数:  {metrics['mcc']:.4f}")

        print("\n" + "="*80 + "\n")

    def save_results(self, eval_result: Dict, output_path: str = None):
        """
        保存评估结果到JSON文件

        Parameters
        ----------
        eval_result : Dict
            评估结果
        output_path : str, optional
            输出文件路径
        """
        import json

        if output_path is None:
            output_path = self.config.recommendation_output_dir / 'evaluation' / 'evaluation_results.json'

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化的格式
        eval_result_serializable = {
            'threshold_positive': eval_result['threshold_positive'],
            'threshold_negative': eval_result['threshold_negative'],
            'num_patients': eval_result['num_patients'],
            'num_failed': eval_result.get('num_failed', 0),
            'common_meds': eval_result['common_meds'],
            'aggregated_metrics': eval_result['aggregated_metrics']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_result_serializable, f, ensure_ascii=False, indent=2)

        print(f"评估结果已保存: {output_path}")

        return output_path
