"""
评估指标计算模块

计算推荐系统的分类评价指标
"""

from typing import Dict, List
import pandas as pd


def calculate_metrics(
    pred_scores: Dict[str, float],
    true_medicines: pd.Series,
    common_meds: List[str],
    common_meds_original: List[str],
    threshold_positive: float
) -> Dict:
    """
    计算单个患者的评估指标

    Parameters
    ----------
    pred_scores : Dict[str, float]
        预测分数 {med_药物_t: score}
    true_medicines : pd.Series
        真实用药数据行
    common_meds : List[str]
        图中的药品列表（格式：med_药品_t）
    common_meds_original : List[str]
        原始药品列名
    threshold_positive : float
        推荐阈值

    Returns
    -------
    Dict
        评价指标，包含：
        - tp, fp, tn, fn: 混淆矩阵元素
        - accuracy, precision, recall, f1, mcc: 评估指标
    """
    y_pred = []
    y_true = []

    for med_graph, med_orig in zip(common_meds, common_meds_original):
        # 预测：1=推荐，0=不推荐
        pred_label = 1 if pred_scores.get(med_graph, 0) >= threshold_positive else 0
        y_pred.append(pred_label)

        # 真实
        true_label = int(true_medicines.get(med_orig, 0))
        y_true.append(true_label)

    # 混淆矩阵
    tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)

    # 指标
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    # MCC
    mcc_numerator = tp * tn - fp * fn
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'num_recommended': sum(y_pred),
        'num_true_medications': sum(y_true)
    }


def aggregate_metrics(results: List[Dict]) -> Dict:
    """
    汇总多个患者的评估指标

    Parameters
    ----------
    results : List[Dict]
        单个患者评估结果列表

    Returns
    -------
    Dict
        汇总指标，包含：
        - micro: 微观平均
        - macro: 宏观平均
    """
    if not results:
        return {
            'micro': {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'tp', 'fp', 'tn', 'fn']},
            'macro': {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'mcc']}
        }

    # 累积混淆矩阵
    total_tp = sum(r['metrics']['tp'] for r in results)
    total_fp = sum(r['metrics']['fp'] for r in results)
    total_tn = sum(r['metrics']['tn'] for r in results)
    total_fn = sum(r['metrics']['fn'] for r in results)

    # 微观平均
    total = total_tp + total_tn + total_fp + total_fn
    micro_accuracy = (total_tp + total_tn) / total if total > 0 else 0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = 0

    mcc_numerator = total_tp * total_tn - total_fp * total_fn
    mcc_denominator = ((total_tp + total_fp) * (total_tp + total_fn) *
                       (total_tn + total_fp) * (total_tn + total_fn)) ** 0.5
    micro_mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

    # 宏观平均
    n = len(results)
    macro_accuracy = sum(r['metrics']['accuracy'] for r in results) / n
    macro_precision = sum(r['metrics']['precision'] for r in results) / n
    macro_recall = sum(r['metrics']['recall'] for r in results) / n
    macro_f1 = sum(r['metrics']['f1'] for r in results) / n
    macro_mcc = sum(r['metrics']['mcc'] for r in results) / n

    return {
        'micro': {
            'accuracy': micro_accuracy,
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1,
            'mcc': micro_mcc,
            'tp': total_tp, 'fp': total_fp, 'tn': total_tn, 'fn': total_fn
        },
        'macro': {
            'accuracy': macro_accuracy,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
            'mcc': macro_mcc
        }
    }
