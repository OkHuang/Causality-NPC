"""
因果效应估计主流程

整合所有模块，执行完整的因果效应估计流程
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_discovery.config import NPCConfig
from .data.loader import DiscoveryLoader
from .data.edge_filter import filter_edges
from .estimation.logistic_ovr import estimate_logistic_ovr
from .output.saver import EffectSaver
from .output.reporter import EffectReporter


def run_causal_effect(config: NPCConfig) -> pd.DataFrame:
    """
    运行因果效应估计流程

    Parameters
    ----------
    config : NPCConfig
        配置对象

    Returns
    -------
    pd.DataFrame
        效应估计结果表
    """
    print(f"\n{'='*60}")
    print(f"因果效应估计流程开始")
    print(f"实验: {config.experiment_name}")
    print(f"输出目录: {config.effect_output_dir}")
    print(f"{'='*60}\n")

    # 创建保存器
    saver = EffectSaver(config.effect_output_dir)

    # ========== 步骤1: 加载因果发现输出 ==========
    print("步骤1: 加载因果发现输出")
    loader = DiscoveryLoader(config.discovery_output_dir)
    graph, edges, data = loader.load_all()

    # ========== 步骤2: 过滤边 ==========
    print("\n步骤2: 过滤边")
    filtered_edges = filter_edges(
        edges=edges,
        data=data,
        graph=graph,
        min_correlation=config.effect.min_correlation,
        min_sample_size=config.effect.min_sample_size
    )

    # ========== 步骤3: 批量估计效应 ==========
    print(f"\n步骤3: 批量估计效应")
    print(f"  待估计边数: {len(filtered_edges)}")

    results = []
    for i, edge_info in enumerate(filtered_edges):
        source = edge_info['source']
        target = edge_info['target']

        print(f"\n[{i+1}/{len(filtered_edges)}] {source} -> {target}")
        print(f"  相关系数: {edge_info['correlation']:.3f}")

        try:
            result = estimate_logistic_ovr(
                data=data,
                graph=graph,
                source=source,
                target=target,
                bootstrap_iter=config.effect.bootstrap.n_iterations if config.effect.bootstrap.enable else 0,
                confidence_level=config.effect.bootstrap.confidence_level,
                min_sample_size=config.effect.min_sample_size
            )

            result['correlation'] = edge_info['correlation']

            # 计算样本统计
            treatment_col = data[source]
            outcome_col = data[target]
            result['n_treated'] = int((treatment_col > 0).sum())
            result['n_outcome'] = int((outcome_col > 0).sum())
            result['total_samples'] = len(data)

            results.append(result)

            ate = result['ate']
            ci_lower = result.get('ci_lower')
            ci_upper = result.get('ci_upper')

            if ci_lower is not None:
                print(f"  ATE = {ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
            else:
                print(f"  ATE = {ate:.4f}")

        except Exception as e:
            print(f"  失败: {e}")
            results.append({
                'treatment': source,
                'outcome': target,
                'error': str(e),
                'method': 'logistic_ovr',
                'correlation': edge_info['correlation']
            })

    # ========== 步骤4: 保存结果 ==========
    print(f"\n步骤4: 保存结果")
    saver.save_models_pkl(results)
    csv_path = saver.save_results_csv(results)
    saver.print_summary(results)

    # ========== 步骤5: 生成报告 ==========
    print(f"\n步骤5: 生成报告")
    reporter = EffectReporter()

    # 读取CSV结果用于报告
    results_df = pd.read_csv(csv_path)

    # 收集统计信息
    stats = {
        'total_edges': len(filtered_edges),
        'successful': len([r for r in results if 'ate' in r]),
        'failed': len([r for r in results if 'error' in r]),
        'failed_list': [{'source': r['treatment'], 'target': r['outcome'], 'error': r['error']}
                        for r in results if 'error' in r]
    }

    report = reporter.generate(results_df, stats)
    report_path = saver.output_dir / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存: {report_path}")

    print(f"\n{'='*60}")
    print(f"因果效应估计流程完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    return results_df


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='因果效应估计')
    parser.add_argument('--config', default='config/base.yaml', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置并运行
    config = NPCConfig.from_yaml(args.config)
    run_causal_effect(config)


if __name__ == "__main__":
    main()
