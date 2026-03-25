"""
阈值搜索模块

搜索最优的推荐阈值组合
"""

from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path
from datetime import datetime


def threshold_search(
    evaluator,
    threshold_positive_range: List[float] = None,
    threshold_negative_range: List[float] = None,
    max_patients: int = 50,
    metric: str = "f1",
    verbose: bool = True
) -> Tuple[List[Dict], Dict, Tuple[float, float]]:
    """
    阈值搜索

    Parameters
    ----------
    evaluator : RecommendationEvaluator
        评估器实例
    threshold_positive_range : List[float], optional
        正向阈值搜索范围
    threshold_negative_range : List[float], optional
        负向阈值搜索范围
    max_patients : int
        每次评价的最大患者数
    metric : str
        优化目标（f1, accuracy, precision, recall, mcc）
    verbose : bool
        是否打印详细进度

    Returns
    -------
    results : List[Dict]
        所有阈值组合的结果
    best_result : Dict
        最优阈值组合的评估结果
    best_threshold : Tuple[float, float]
        最优阈值 (positive, negative)
    """
    if threshold_positive_range is None:
        threshold_positive_range = [0.01, 0.05, 0.1, 0.15]
    if threshold_negative_range is None:
        threshold_negative_range = [-0.15, -0.1, -0.05, -0.01]

    best_score = 0
    best_threshold = None
    best_result = None

    results = []

    if verbose:
        print("\n" + "="*80)
        print("阈值搜索 (Threshold Search)")
        print("="*80)
        print(f"\n搜索空间:")
        print(f"  正向阈值范围: {threshold_positive_range}")
        print(f"  负向阈值范围: {threshold_negative_range}")
        print(f"  总组合数: {len(threshold_positive_range) * len(threshold_negative_range)}")
        print(f"  每个组合评价患者数: {max_patients}")
        print(f"  优化目标: {metric}")

    for i, tp in enumerate(threshold_positive_range):
        for j, tn in enumerate(threshold_negative_range):
            # 更新阈值
            evaluator.threshold_positive = tp
            evaluator.threshold_negative = tn

            # 评价
            try:
                eval_result = evaluator.evaluate_batch(max_patients=max_patients)
                micro_metrics = eval_result['aggregated_metrics']['micro']

                result_entry = {
                    'threshold_positive': tp,
                    'threshold_negative': tn,
                    'accuracy': micro_metrics['accuracy'],
                    'precision': micro_metrics['precision'],
                    'recall': micro_metrics['recall'],
                    'f1': micro_metrics['f1'],
                    'mcc': micro_metrics['mcc']
                }
                results.append(result_entry)

                score = micro_metrics[metric]

                if verbose:
                    print(f"  [{i*len(threshold_negative_range) + j + 1:>2}/{len(threshold_positive_range) * len(threshold_negative_range)}] "
                          f"阈值: [{tn:>6}, {tp:>6}], "
                          f"F1: {micro_metrics['f1']:.4f}, "
                          f"Acc: {micro_metrics['accuracy']:.4f}, "
                          f"Prec: {micro_metrics['precision']:.4f}, "
                          f"Rec: {micro_metrics['recall']:.4f}, "
                          f"MCC: {micro_metrics['mcc']:.4f}")

                # 更新最优结果
                if score > best_score:
                    best_score = score
                    best_threshold = (tp, tn)
                    best_result = eval_result

            except Exception as e:
                print(f"  阈值 [{tn}, {tp}] 评价失败: {e}")
                continue

    if verbose:
        print(f"\n{'='*80}")
        print("最优阈值 (Best Threshold)")
        print(f"{'='*80}")
        print(f"  正向阈值: {best_threshold[0]}")
        print(f"  负向阈值: {best_threshold[1]}")
        print(f"  {metric.upper()}: {best_score:.4f}")
        print(f"\n  详细指标:")
        metrics = best_result['aggregated_metrics']['micro']
        print(f"    准确率: {metrics['accuracy']:.4f}")
        print(f"    精确率: {metrics['precision']:.4f}")
        print(f"    召回率: {metrics['recall']:.4f}")
        print(f"    F1分数: {metrics['f1']:.4f}")
        print(f"    MCC系数: {metrics['mcc']:.4f}")

    return results, best_result, best_threshold


def save_threshold_search_results(
    results: List[Dict],
    best_threshold: Tuple[float, float],
    output_path: str
):
    """
    保存阈值搜索结果

    Parameters
    ----------
    results : List[Dict]
        所有阈值组合的结果
    best_threshold : Tuple[float, float]
        最优阈值组合
    output_path : str
        输出文件路径
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 保存为CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 保存为JSON
    json_path = output_path.with_suffix('.json')
    search_result = {
        'timestamp': datetime.now().isoformat(),
        'best_threshold': {
            'positive': best_threshold[0],
            'negative': best_threshold[1]
        },
        'all_results': results
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(search_result, f, ensure_ascii=False, indent=2)

    print(f"\n阈值搜索结果已保存:")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")

    return csv_path, json_path
