"""
鼻咽癌因果推断 - 全流程运行脚本
支持参数实验和结果自动组织
"""

import sys
import time
import argparse
from pathlib import Path

# 添加utils路径
sys.path.append(str(Path(__file__).parent))

from utils.config_loader import load_config, list_experiments
from utils.path_manager import PathManager


def run_all(config_path="config/base.yaml", skip_existing=False, verbose=True):
    """
    运行完整流程

    Parameters
    ----------
    config_path : str
        配置文件路径
    skip_existing : bool
        是否跳过已存在的输出（断点续传）
    verbose : bool
        是否打印详细信息
    """
    # 加载配置
    config = load_config(config_path)
    path_mgr = PathManager(config)

    exp_name = config['experiment']['name']
    exp_desc = config['experiment']['description']

    # 打印实验信息
    print("="*80)
    print("鼻咽癌因果推断 - 全流程运行")
    print("="*80)
    print(f"实验名称: {exp_name}")
    print(f"实验描述: {exp_desc}")
    print(f"配置文件: {config_path}")
    print(f"输出目录: {path_mgr.output_dir}")
    print("="*80)

    total_start = time.time()

    # 步骤1：因果发现
    print("\n[步骤 1/3] 因果发现")
    print("-"*80)

    if skip_existing and _check_discovery_outputs(path_mgr):
        print("✓ 输出文件已存在，跳过此步骤")
    else:
        start_time = time.time()
        from experiments.simple_causal_discovery import run_causal_discovery
        run_causal_discovery(config_path)
        elapsed = time.time() - start_time
        print(f"✓ 因果发现完成 (耗时: {elapsed:.2f}秒)")

    # 步骤2：因果效应估计
    print("\n[步骤 2/3] 因果效应估计")
    print("-"*80)

    if skip_existing and _check_effect_outputs(path_mgr):
        print("✓ 输出文件已存在，跳过此步骤")
    else:
        start_time = time.time()
        from experiments.simple_causal_effect_4_ovr import main as run_causal_effect
        run_causal_effect(config_path)
        elapsed = time.time() - start_time
        print(f"✓ 因果效应估计完成 (耗时: {elapsed:.2f}秒)")

    # 步骤3：因果推荐
    print("\n[步骤 3/3] 因果推荐")
    print("-"*80)

    if skip_existing and _check_recommendation_outputs(path_mgr):
        print("✓ 输出文件已存在，跳过此步骤")
    else:
        start_time = time.time()
        from experiments.simple_causal_recommendation import main as run_recommendation
        run_recommendation(config_path)
        elapsed = time.time() - start_time
        print(f"✓ 因果推荐完成 (耗时: {elapsed:.2f}秒)")

    total_elapsed = time.time() - total_start

    print("\n" + "="*80)
    print(f"全流程执行完成！ (总耗时: {total_elapsed:.2f}秒)")
    print(f"结果保存在: {path_mgr.output_dir}")
    print("="*80)


def _check_discovery_outputs(path_mgr):
    """检查因果发现步骤的输出是否存在"""
    required = [
        path_mgr.get_data_dir() / 'step4_data_matrix.csv',
        path_mgr.get_graphs_dir() / 'causal_dag.pkl',
        path_mgr.get_graphs_dir() / 'causal_edges.json'
    ]
    return all(p.exists() for p in required)


def _check_effect_outputs(path_mgr):
    """检查因果效应步骤的输出是否存在"""
    required = [
        path_mgr.get_effects_dir() / 'estimated_effects_4_ovr.csv',
        path_mgr.get_models_dir() / 'causal_estimates_4_ovr.pkl'
    ]
    return all(p.exists() for p in required)


def _check_recommendation_outputs(path_mgr):
    """检查推荐步骤的输出是否存在"""
    # 检查至少有一个推荐结果
    recommendations = list(path_mgr.get_effects_dir().glob('recommendation_case_*.json'))
    return len(recommendations) > 0


def list_all_experiments():
    """列出所有可用的实验配置"""
    print("\n可用的实验配置：")
    print("-"*80)

    experiments = list_experiments()

    if not experiments:
        print("未找到实验配置文件")
        return

    for name, info in experiments.items():
        print(f"\n{name}:")
        print(f"  描述: {info['description']}")
        print(f"  文件: {info['file']}")
        print(f"  标签: {', '.join(info['tags'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='运行完整因果推断流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认配置运行
  python scripts/run_all.py

  # 使用指定实验配置运行
  python scripts/run_all.py --config config/experiments/exp_freq_0.15.yaml

  # 断点续传（跳过已完成步骤）
  python scripts/run_all.py --config config/experiments/exp_freq_0.15.yaml --skip-existing

  # 列出所有可用实验
  python scripts/run_all.py --list

  # 对比多个实验
  python scripts/run_all.py --config config/experiments/exp_freq_0.10.yaml
  python scripts/run_all.py --config config/experiments/exp_freq_0.15.yaml
  python scripts/run_all.py --config config/experiments/exp_freq_0.20.yaml
        """
    )

    parser.add_argument('--config', default='config/base.yaml',
                        help='配置文件路径（默认: config/base.yaml）')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已存在的输出（断点续传）')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的实验配置')

    args = parser.parse_args()

    if args.list:
        list_all_experiments()
    else:
        try:
            run_all(args.config, args.skip_existing)
        except Exception as e:
            print(f"\n✗ 执行失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
