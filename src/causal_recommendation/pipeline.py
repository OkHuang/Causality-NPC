"""
因果推荐主流程

整合所有模块，执行完整的因果推荐流程
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_discovery.config import NPCConfig
from .data.loader import RecommendationLoader
from .data.patient_encoder import extract_mapping_rules
from .recommendation.propagation import causal_propagation_recommend
from .output.saver import RecommendationSaver
from .output.reporter import RecommendationReporter


def run_causal_recommendation(
    config: NPCConfig,
    patients: List[Dict]
) -> Dict:
    """
    运行因果推荐流程

    Parameters
    ----------
    config : NPCConfig
        配置对象
    patients : List[Dict]
        患者信息列表

    Returns
    -------
    Dict
        推荐结果字典
    """
    print(f"\n{'='*60}")
    print(f"因果推荐流程开始")
    print(f"实验: {config.experiment_name}")
    print(f"输出目录: {config.recommendation_output_dir}")
    print(f"{'='*60}\n")

    # 创建保存器
    saver = RecommendationSaver(config.recommendation_output_dir)

    # ========== 步骤1: 加载图和ATE ==========
    print("步骤1: 加载图和ATE估计")
    loader = RecommendationLoader(config)
    graph, ate_dict = loader.load_all()

    # ========== 步骤2: 提取映射规则 ==========
    print("\n步骤2: 提取映射规则")
    all_nodes = set(graph.nodes())
    mapping_rules = extract_mapping_rules(all_nodes)
    print(f"  映射规则数: {len(mapping_rules)}")

    # ========== 步骤3: 对每个患者执行推荐 ==========
    print(f"\n步骤3: 执行推荐")
    print(f"  患者数: {len(patients)}")

    results = []
    for i, patient_info in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 案例 {i}")

        # 执行推荐
        result = causal_propagation_recommend(
            graph=graph,
            ate_dict=ate_dict,
            patient_info=patient_info,
            all_nodes=all_nodes,
            mapping_rules=mapping_rules,
            threshold_positive=config.recommendation.threshold_positive,
            threshold_negative=config.recommendation.threshold_negative,
            top_k=config.recommendation.top_k,
            max_paths=config.recommendation.max_paths
        )

        # 添加患者信息到结果
        result['patient_info'] = patient_info

        results.append(result)

        # 打印简要结果
        recommended = result.get('recommended', {})
        not_recommended = result.get('not_recommended', {})
        neutral = result.get('neutral', [])

        print(f"  推荐: {len(recommended)} 味")
        if recommended:
            top_med = list(recommended.keys())[0]
            print(f"    Top-1: {top_med} ({recommended[top_med]:.4f})")

        print(f"  不推荐: {len(not_recommended)} 味")
        print(f"  中性: {len(neutral)} 味")

    # ========== 步骤4: 保存结果 ==========
    print(f"\n步骤4: 保存结果")
    saver.save_recommendations(results)
    saver.print_summary(results)

    # ========== 步骤5: 生成报告 ==========
    print(f"\n步骤5: 生成报告")
    reporter = RecommendationReporter()
    report = reporter.generate(results)
    report_path = saver.output_dir / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存: {report_path}")

    print(f"\n{'='*60}")
    print(f"因果推荐流程完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    return {
        'results': results,
        'report': report
    }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='因果推荐')
    parser.add_argument('--config', default='config/base.yaml', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置并运行
    config = NPCConfig.from_yaml(args.config)

    # 示例患者数据
    patients = [
        {
            'gender': '女',
            'age': 58,
            '乏力': 1.0,
            '畏寒': 1.0,
            '头晕': 1.0,
            '阴虚血瘀证': 1.0,
            '颃颡岩': 1.0,
        },
    ]

    run_causal_recommendation(config, patients)


if __name__ == "__main__":
    main()
