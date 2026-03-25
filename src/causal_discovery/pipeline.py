"""
因果发现主流程

整合所有模块，执行完整的因果发现流程
"""

import sys
from pathlib import Path
import pandas as pd
import networkx as nx
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .config import NPCConfig
from .data.loader import DataLoader
from .data.pair_builder import PairBuilder
from .data.cleaner import clean_data_matrix
from .features.encoder import FeatureEncoder
from .discovery.pc import run_pc_algorithm
from .discovery.constraints import ConstraintManager
from .discovery.cycle_handler import remove_cycles
from .visualization.plotter import CausalPlotter
from .output.saver import ResultSaver
from .output.reporter import DiscoveryReporter


def run_causal_discovery(config: NPCConfig) -> nx.DiGraph:
    """
    运行因果发现流程

    Parameters
    ----------
    config : NPCConfig
        配置对象

    Returns
    -------
    nx.DiGraph
        发现的因果图
    """
    print(f"\n{'='*60}")
    print(f"因果发现流程开始")
    print(f"实验: {config.experiment_name}")
    print(f"输出目录: {config.discovery_output_dir}")
    print(f"{'='*60}\n")

    # 创建结果保存器
    saver = ResultSaver(config.discovery_output_dir)

    # ========== 步骤1: 数据加载 ==========
    print("步骤1: 数据加载")
    loader = DataLoader(config.raw_data_path)
    df = loader.load_and_process()

    # ========== 步骤2: 构建时序对 ==========
    print("\n步骤2: 构建时序对")
    pair_builder = PairBuilder(df)
    pairs_df = pair_builder.build_pairs()
    saver.save_data(pairs_df, 'pairs.csv')

    # ========== 步骤3: 特征编码 ==========
    print("\n步骤3: 特征编码")
    encoder = FeatureEncoder(
        symptom_threshold=config.discovery.symptom_threshold,
        medicine_threshold=config.discovery.medicine_threshold,
        diagnosis_threshold=config.discovery.diagnosis_threshold,
    )
    encoded_df, encode_stats = encoder.encode_all(pairs_df)

    # ========== 步骤4: 数据清洗 ==========
    print("\n步骤4: 数据清洗")
    data_matrix, selected_cols = clean_data_matrix(encoded_df)

    # 只保留选中的列
    data_matrix = data_matrix[selected_cols].copy()
    saver.save_data(data_matrix, 'matrix.csv')

    # 获取变量分组
    variable_groups = {
        'static': ['gender_encoded'] + [col for col in selected_cols if col.startswith('age_')],
        'symptoms_t': [col for col in selected_cols if col.endswith('_t') and not col.startswith('age_')
                       and not col.startswith('med_') and not col.startswith('diagnosis_')],
        'diagnosis_t': [col for col in selected_cols if col.startswith('diagnosis_') and col.endswith('_t')],
        'medicines_t': [col for col in selected_cols if col.startswith('med_') and col.endswith('_t')],
        'symptoms_t1': [col for col in selected_cols if col.endswith('_t1') and not col.startswith('med_')
                        and not col.startswith('diagnosis_')],
        'diagnosis_t1': [col for col in selected_cols if col.startswith('diagnosis_') and col.endswith('_t1')],
    }

    # ========== 步骤5: 运行PC算法 ==========
    print("\n步骤5: 运行PC算法")
    G = run_pc_algorithm(
        data_matrix,
        node_names=selected_cols,
        alpha=config.discovery.alpha
    )

    # ========== 步骤6: 应用约束 ==========
    if config.discovery.apply_constraints:
        print("\n步骤6: 应用约束")
        constraint_manager = ConstraintManager()
        constraint_manager.set_variable_groups(variable_groups)
        G = constraint_manager.apply_constraints(G)

    # ========== 步骤7: 环处理 ==========
    if config.discovery.remove_cycles:
        print("\n步骤7: 环处理")
        G = remove_cycles(G)

    # ========== 步骤8: 保存图结构 ==========
    print("\n步骤8: 保存图结构")
    saver.save_graph(G)

    # ========== 步骤9: 可视化 ==========
    print("\n步骤9: 可视化")
    if G.number_of_edges() > 0:
        plotter = CausalPlotter()
        fig = plotter.plot_dag(
            G,
            variable_groups=variable_groups,
            save_path=str(saver.graph_dir / 'dag.png'),
            title="鼻咽癌中西医结合诊疗因果网络"
        )
        saver.save_visualization(fig)
    else:
        print("没有边可以绘制")

    # ========== 步骤10: 生成报告 ==========
    print("\n步骤10: 生成报告")
    reporter = DiscoveryReporter()

    # 收集统计信息
    stats = {
        **pair_builder.get_summary(),
        **encode_stats,
    }

    report = reporter.generate(G, stats, variable_groups)
    saver.save_report(report)

    print(f"\n{'='*60}")
    print(f"因果发现流程完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    return G


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='因果发现')
    parser.add_argument('--config', default='config/base.yaml', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置并运行
    config = NPCConfig.from_yaml(args.config)
    run_causal_discovery(config)


if __name__ == "__main__":
    main()
