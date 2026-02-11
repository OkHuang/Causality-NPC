"""
示例：因果发现完整流程（目标二）

探索性因果发现：构建症状-药物-证型的因果网络
"""

import sys
import os
# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 切换工作目录到项目根目录（这样所有相对路径都能正常工作）
os.chdir(project_root)

import pandas as pd
import numpy as np
from pathlib import Path

# 配置matplotlib中文显示
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入项目模块
from src.data import DataLoader, DataPreprocessor, TimeAligner
from src.features import SymptomExtractor, MedicineMapper, SyndromeEncoder
from src.causal_discovery import CausalDiscovery, StabilitySelector, ConstraintManager, GraphUtils
from src.visualization import DAGPlotter, ReportGenerator
from src.utils.config import load_config


def main():
    """主函数"""

    print("="*80)
    print("鼻咽癌因果推断 - 目标二：因果发现")
    print("="*80)

    # ============================================
    # 1. 加载配置
    # ============================================
    print("\n[1] 加载配置...")
    config = load_config(
        "config/causal_discovery.yaml",
        base_config_path="config/base.yaml"
    )

    # ============================================
    # 2. 加载数据
    # ============================================
    print("\n[2] 加载数据...")
    loader = DataLoader()
    # 使用实际存在的数据文件
    data_path = "Data/raw/npc_final.csv"
    if not os.path.exists(data_path):
        data_path = "data/raw/npc_data.csv"
    # 使用gbk编码读取CSV文件
    df = loader.load(data_path, encoding='gbk')

    print(f"原始数据: {df.shape}")
    print(f"患者数: {df['patient_id'].nunique()}")

    # ============================================
    # 3. 数据预处理
    # ============================================
    print("\n[3] 数据预处理...")

    # 3.1 基本清洗
    preprocessor = DataPreprocessor(config.get("preprocessing", {}))
    df_clean = preprocessor.clean(df)

    # 3.2 队列筛选
    df_cohort = preprocessor.filter_cohort(
        df_clean,
        min_follow_up=config.get("cohort", {}).get("min_follow_up", 30),
        min_visits=config.get("cohort", {}).get("min_visits", 2)
    )

    # ============================================
    # 4. 时间对齐
    # ============================================
    print("\n[4] 时间轴对齐...")
    time_aligner = TimeAligner(config.get("time_alignment", {}))
    df_aligned = time_aligner.align(df_cohort)

    # ============================================
    # 5. 特征提取
    # ============================================
    print("\n[5] 特征工程...")

    # 5.1 症状提取
    print("  - 提取症状...")
    symptom_extractor = SymptomExtractor(config.get("symptom_extraction", {}))
    df_features = symptom_extractor.transform(df_aligned, text_column="chief_complaint")

    # 5.2 药物映射
    print("  - 映射药物...")
    medicine_mapper = MedicineMapper(config.get("medicine_mapping", {}))
    df_features = medicine_mapper.transform(df_features, medicine_column="chinese_medicines")

    # 5.3 证型编码
    print("  - 编码证型...")
    syndrome_encoder = SyndromeEncoder(config.get("syndrome_encoding", {}))
    df_features = syndrome_encoder.fit_transform(df_features, syndrome_column="chinese_diagnosis")

    # 5.4 创建时序对
    print("  - 创建时序特征...")
    df_time = time_aligner.create_lag_features(df_features, lag=1)

    print(f"最终特征: {df_time.shape}")

    # ============================================
    # 6. 构建时序数据矩阵
    # ============================================
    print("\n[6] 构建时序数据矩阵...")

    # 为变量添加时间标签
    # t时刻：症状、证型、人口学
    # t+1时刻：症状

    # TODO: 根据实际数据构建时序变量
    # 示例：
    time_vars_t = [col for col in df_time.columns if col.startswith(("S_", "D_", "M_"))]
    time_vars_t1 = [col.replace("_t", "_t1") for col in time_vars_t if col.startswith("S_")]

    print(f"  t时刻变量: {len(time_vars_t)}")
    print(f"  t+1时刻变量: {len(time_vars_t1)}")

    # ============================================
    # 7. 变量筛选
    # ============================================
    print("\n[7] 变量筛选...")

    # 只保留高频变量（出现频率>10%）
    # 放宽阈值以保留更多变量（原10% -> 5%）
    min_prevalence = config.get("feature_reduction", {}).get("min_prevalence", 0.05)
    n_samples = len(df_time)

    selected_vars_t = []
    for var in time_vars_t:
        if (df_time[var] > 0).sum() / n_samples >= min_prevalence:
            selected_vars_t.append(var)

    print(f"  筛选后t时刻变量: {len(selected_vars_t)}")

    # 限制最大特征数（放宽限制以保留更多变量）
    max_features = config.get("feature_reduction", {}).get("max_features", 100)
    if len(selected_vars_t) > max_features:
        # 按频率排序，保留前N个
        var_freqs = [(var, (df_time[var] > 0).sum()) for var in selected_vars_t]
        var_freqs.sort(key=lambda x: x[1], reverse=True)
        selected_vars_t = [var for var, _ in var_freqs[:max_features]]

    print(f"  最终变量数: {len(selected_vars_t)}")

    # 准备因果发现数据
    discovery_data = df_time[selected_vars_t].copy()

    # ============================================
    # 8. 因果发现（单次运行）
    # ============================================
    print("\n[8] 运行因果发现算法...")

    cd = CausalDiscovery(config.get("algorithm", {}))

    try:
        graph = cd.discover(
            discovery_data,
            algorithm=config.get("algorithm", {}).get("name", "pc"),
            alpha=config.get("algorithm", {}).get("alpha", 0.05),
            independence_test=config.get("algorithm", {}).get("independence_test", "fisherz"),
        )

        print(f"  发现边数: {len(cd.get_edges())}")

        # 应用约束
        constraint_manager = ConstraintManager(config.get("constraints", {}))
        graph_constrained = constraint_manager.apply_constraints(graph, discovery_data.columns.tolist())

        print(f"  应用约束后边数: {graph_constrained.number_of_edges()}")

        # 绘制初始图
        cd.plot_graph(
            layout=config.get("visualization", {}).get("layout", "hierarchical"),
            save_path="outputs/graphs/causal_dag_initial.png"
        )

    except Exception as e:
        print(f"  错误: {e}")
        print("  请检查 causal-learn 是否已安装: pip install causal-learn")
        return

    # ============================================
    # 9. 稳定性选择
    # ============================================
    print("\n[9] Bootstrap稳定性选择...")
    print(f"  运行 {config.get('stability_selection', {}).get('n_bootstrap', 1000)} 次 Bootstrap...")

    stability_selector = StabilitySelector(config.get("stability_selection", {}))

    # 使用配置文件的参数
    algorithm_kwargs = {
        "alpha": config.get("algorithm", {}).get("alpha", 0.05),
        "independence_test": config.get("algorithm", {}).get("independence_test", "fisherz"),
    }

    stable_edges = stability_selector.select(
        discovery_data,
        cd.discover,
        algorithm_kwargs
    )

    print(f"  稳定边数: {len(stable_edges)}")

    # 获取边频率表
    edge_freq_df = stability_selector.get_frequency_table()
    edge_freq_df.to_csv("outputs/graphs/edge_frequencies.csv", index=False, encoding="utf-8-sig")

    # 绘制频率分布
    stability_selector.plot_frequencies(
        top_n=20,
        save_path="outputs/figures/edge_frequencies.png"
    )

    # ============================================
    # 10. 构建稳定因果图
    # ============================================
    print("\n[10] 构建稳定因果图...")

    # 重新运行因果发现以创建稳定图
    cd_stable = CausalDiscovery(config.get("algorithm", {}))
    stable_graph = cd_stable.discover(
        discovery_data,
        algorithm=config.get("algorithm", {}).get("name", "pc"),
        alpha=config.get("algorithm", {}).get("alpha", 0.05),
        independence_test=config.get("algorithm", {}).get("independence_test", "fisherz"),
    )

    # 只保留稳定边（边已保留方向，直接检查）
    for edge in list(stable_graph.edges()):
        if edge not in stable_edges:
            stable_graph.remove_edge(*edge)

    # 绘制稳定图
    plotter = DAGPlotter(config.get("visualization", {}))

    # 创建变量类型映射
    node_types = GraphUtils.create_variable_type_mapping(discovery_data.columns.tolist())

    # 边置信度映射
    edge_confidence = {edge: freq for edge, freq in stability_selector.get_edge_frequencies().items()}

    plotter.plot(
        stable_graph,
        layout="hierarchical",
        node_types=node_types,
        edge_confidence=edge_confidence,
        save_path="outputs/graphs/causal_dag_stable.png"
    )

    # ============================================
    # 11. 提取关键路径
    # ============================================
    print("\n[11] 提取关键因果路径...")

    graph_utils = GraphUtils()

    # 提取"证型->药物->症状"路径
    pattern_paths = graph_utils.extract_pattern_paths(
        stable_graph,
        pattern="syndrome -> medicine -> symptom",
        variable_types=node_types,
        max_length=3,
    )

    print(f"  找到 {len(pattern_paths)} 条'证型->药物->症状'路径")

    # 导出路径表
    if pattern_paths:
        paths_df = pd.DataFrame(pattern_paths)
        paths_df = graph_utils.export_path_table(
            stable_graph,
            [p["path"] for p in pattern_paths],
            save_path="outputs/graphs/key_paths.csv"
        )

    # ============================================
    # 12. 图统计分析
    # ============================================
    print("\n[12] 图统计分析...")

    graph_stats = graph_utils.analyze_graph_statistics(stable_graph)

    print(f"  节点数: {graph_stats['n_nodes']}")
    print(f"  边数: {graph_stats['n_edges']}")
    print(f"  图密度: {graph_stats['density']:.4f}")
    print(f"  最大入度: {graph_stats['max_in_degree']}")
    print(f"  最大出度: {graph_stats['max_out_degree']}")

    # ============================================
    # 13. 生成报告
    # ============================================
    print("\n[13] 生成报告...")

    report_gen = ReportGenerator()

    # 整合统计信息
    final_stats = {
        "algorithm": config.get("algorithm", {}).get("name", "pc"),
        "n_bootstrap": config.get("stability_selection", {}).get("n_bootstrap", 1000),
        "threshold": config.get("stability_selection", {}).get("selection_threshold", 0.85),
        **graph_stats,
    }

    # 如果有路径数据
    paths_df_final = paths_df if pattern_paths else pd.DataFrame()

    report_gen.generate_causal_discovery_report(
        final_stats,
        edge_freq_df,
        paths_df_final,
        "outputs/reports/causal_discovery_report.md"
    )

    print("\n" + "="*80)
    print("分析完成！")
    print(f"报告已保存至: outputs/reports/causal_discovery_report.md")
    print("="*80)


if __name__ == "__main__":
    main()
