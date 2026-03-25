"""
运行因果推荐评估

使用方法:
    # 单次评估
    python experiments/run_evaluation.py --mode evaluate --max-patients 100

    # 阈值搜索
    python experiments/run_evaluation.py --mode search --max-patients 50

    # 详细阈值搜索
    python experiments/run_evaluation.py --mode search --search-level detailed --max-patients 50
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.causal_discovery.config import NPCConfig
from src.causal_recommendation.evaluation.evaluator import RecommendationEvaluator
from src.causal_recommendation.evaluation.threshold_search import (
    threshold_search,
    save_threshold_search_results
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='因果推荐评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/base.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['evaluate', 'search'],
        default='evaluate',
        help='运行模式：evaluate=单次评估, search=阈值搜索'
    )

    parser.add_argument(
        '--patient-data-path',
        type=str,
        default='Data/processed/processed_data.csv',
        help='患者特征数据路径'
    )

    parser.add_argument(
        '--medicine-data-path',
        type=str,
        default='Data/processed/processed_medicines.csv',
        help='真实用药数据路径'
    )

    parser.add_argument(
        '--max-patients',
        type=int,
        default=50,
        help='每次评估的最大患者数'
    )

    parser.add_argument(
        '--search-level',
        type=str,
        choices=['basic', 'detailed'],
        default='basic',
        help='阈值搜索级别'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='f1',
        choices=['f1', 'accuracy', 'precision', 'recall', 'mcc'],
        help='优化目标指标'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*80)
    print("因果推荐评估实验")
    print("="*80 + "\n")

    # 加载配置
    print(f"加载配置: {args.config}")
    config = NPCConfig.from_yaml(args.config)

    # 初始化评估器
    print("初始化评估器...")
    evaluator = RecommendationEvaluator(
        config=config,
        patient_data_path=args.patient_data_path,
        medicine_data_path=args.medicine_data_path
    )
    print("评估器初始化成功\n")

    # 执行评估或阈值搜索
    if args.mode == 'evaluate':
        # 单次评估
        print("="*80)
        print("执行单次评估")
        print("="*80)
        print(f"  阈值: [{evaluator.threshold_negative}, {evaluator.threshold_positive}]")
        print(f"  患者数: {args.max_patients}")

        eval_result = evaluator.evaluate_batch(max_patients=args.max_patients)
        evaluator.print_evaluation_report(eval_result)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.recommendation_output_dir / 'evaluation' / f"evaluation_{timestamp}.json"
        evaluator.save_results(eval_result, str(output_path))

    elif args.mode == 'search':
        # 阈值搜索
        if args.search_level == 'basic':
            threshold_positive_range = [0.01, 0.05, 0.1, 0.15]
            threshold_negative_range = [-0.15, -0.1, -0.05, -0.01]
        else:
            threshold_positive_range = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
            threshold_negative_range = [-0.2, -0.15, -0.12, -0.1, -0.08, -0.05, -0.02, -0.01]

        results, best_result, best_threshold = threshold_search(
            evaluator=evaluator,
            threshold_positive_range=threshold_positive_range,
            threshold_negative_range=threshold_negative_range,
            max_patients=args.max_patients,
            metric=args.metric,
            verbose=True
        )

        # 保存搜索结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.recommendation_output_dir / 'evaluation' / f"threshold_search_{timestamp}"
        save_threshold_search_results(results, best_threshold, str(output_path))

        # 使用最优阈值进行完整评估
        print(f"\n{'='*80}")
        print("使用最优阈值进行完整评估")
        print(f"{'='*80}")

        evaluator.threshold_positive = best_threshold[0]
        evaluator.threshold_negative = best_threshold[1]

        final_eval_result = evaluator.evaluate_batch(max_patients=None)
        evaluator.print_evaluation_report(final_eval_result)

        # 保存最终评估结果
        output_path = config.recommendation_output_dir / 'evaluation' / f"evaluation_best_threshold_{timestamp}.json"
        evaluator.save_results(final_eval_result, str(output_path))

    print("\n" + "="*80)
    print("评估完成")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
