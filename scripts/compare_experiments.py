"""
实验对比工具
比较不同参数配置下的结果
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.config_loader import load_config
import pandas as pd


def compare_experiments(exp_configs):
    """
    对比多个实验的结果

    Parameters
    ----------
    exp_configs : list of str
        实验配置文件路径列表
    """
    print("="*80)
    print("实验结果对比")
    print("="*80)

    results_summary = []

    for config_path in exp_configs:
        config = load_config(config_path)
        exp_name = config['experiment']['name']

        # 读取结果
        effects_path = Path(f"outputs/{exp_name}/causal_effects/estimated_effects_4_ovr.csv")

        if effects_path.exists():
            effects_df = pd.read_csv(effects_path)

            summary = {
                '实验名称': exp_name,
                '描述': config['experiment']['description'],
                '药物阈值': config['discovery']['medicine_threshold'],
                'Alpha': config['discovery']['alpha'],
                'Bootstrap次数': config['effect']['bootstrap']['n_iterations'],
                '边数量': len(effects_df),
                '平均ATE': effects_df['ate'].mean(),
                '显著边数量': len(effects_df[effects_df['p_value'] < 0.05]) if 'p_value' in effects_df.columns else 'N/A'
            }

            results_summary.append(summary)

            print(f"\n{exp_name}:")
            print(f"  描述: {config['experiment']['description']}")
            print(f"  边数量: {summary['边数量']}")
            print(f"  平均ATE: {summary['平均ATE']:.4f}")
            print(f"  显著边数量: {summary['显著边数量']}")
        else:
            print(f"\n{exp_name}: 结果文件不存在 ({effects_path})")

    # 生成对比表格
    if results_summary:
        comparison_df = pd.DataFrame(results_summary)
        print("\n" + "="*80)
        print("对比表格:")
        print("="*80)
        print(comparison_df.to_string(index=False))

        # 保存对比结果
        outputs_dir = Path('outputs')
        outputs_dir.mkdir(exist_ok=True)
        comparison_df.to_csv(outputs_dir / 'experiments_comparison.csv', index=False, encoding='utf-8-sig')
        print("\n对比结果已保存到: outputs/experiments_comparison.csv")
    else:
        print("\n未找到任何实验结果")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='对比多个实验的结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 对比所有药物频率阈值实验
  python scripts/compare_experiments.py config/experiments/exp_freq_*.yaml

  # 对比特定实验
  python scripts/compare_experiments.py config/experiments/exp_freq_0.10.yaml config/experiments/exp_freq_0.15.yaml

  # 对比所有实验
  python scripts/compare_experiments.py config/experiments/*.yaml
        """
    )

    parser.add_argument('configs', nargs='+',
                        help='实验配置文件路径（支持通配符）')

    args = parser.parse_args()

    try:
        compare_experiments(args.configs)
    except Exception as e:
        print(f"\n✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
