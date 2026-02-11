"""
简易版因果发现主脚本

整合所有模块，执行完整的因果发现流程：
1. 数据加载与解析
2. 构建时序对
3. 特征编码与筛选
4. 运行因果发现算法
5. 应用约束
6. 可视化与路径提取
"""

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.simple_loader import SimpleDataLoader
from src.data.simple_pair_builder import SimplePairBuilder
from src.features.simple_encoder import SimpleFeatureEncoder
from src.causal_discovery.simple_constraints import SimpleConstraintManager
from src.visualization.simple_plotter import SimpleCausalPlotter


# ============== 配置参数 ==============
CONFIG = {
    # 数据路径
    'data_path': r"D:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv",

    # 特征筛选阈值 (针对中等数据集 200-300对)
    # 目标：约30-40个特征，为PC算法准备
    'symptom_threshold': 0.10,      # 症状频率阈值 (20%) - 至少53对
    'medicine_threshold': 0.10,     # 药物频率阈值 (20%) - 至少53对
    'diagnosis_threshold': 0.10,    # 诊断频率阈值 (20%) - 至少53对

    # 因果发现参数
    'alpha': 0.05,                  # 显著性水平（0.05 = 95%置信度）
    'independence_test': 'fisherz', # 独立性检验方法

    # 输出路径
    'output_dir': 'outputs',
    'figures_dir': 'outputs/figures',
    'graphs_dir': 'outputs/graphs',
    'reports_dir': 'outputs/reports',
    'data_dir': 'outputs/data',
}


def create_output_dirs():
    """创建输出目录"""
    for dir_path in [CONFIG['output_dir'], CONFIG['figures_dir'],
                     CONFIG['graphs_dir'], CONFIG['reports_dir'], CONFIG['data_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    print("输出目录已创建")


def run_causal_discovery():
    """运行因果发现流程"""

    print("=" * 60)
    print("简易版因果发现流程")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 创建输出目录
    create_output_dirs()

    # ========== 步骤1: 数据加载 ==========
    print("\n" + "=" * 60)
    print("步骤1: 数据加载与解析")
    print("=" * 60)

    loader = SimpleDataLoader(CONFIG['data_path'])
    df = loader.load_and_process()

    # 保存处理后的数据
    df.to_csv(os.path.join(CONFIG['data_dir'], 'step1_processed_data.csv'), index=False, encoding='utf-8-sig')
    print(f"数据已保存到: {os.path.join(CONFIG['data_dir'], 'step1_processed_data.csv')}")

    # ========== 步骤2: 构建时序对 ==========
    print("\n" + "=" * 60)
    print("步骤2: 构建时序对")
    print("=" * 60)

    builder = SimplePairBuilder(df)
    pairs_df = builder.build_pairs()

    # 保存时序对数据
    pairs_df.to_csv(os.path.join(CONFIG['data_dir'], 'step2_pairs_data.csv'), index=False, encoding='utf-8-sig')
    print(f"时序对数据已保存到: {os.path.join(CONFIG['data_dir'], 'step2_pairs_data.csv')}")

    # ========== 步骤3: 特征编码 ==========
    print("\n" + "=" * 60)
    print("步骤3: 特征编码与筛选")
    print("=" * 60)

    encoder = SimpleFeatureEncoder(
        symptom_threshold=CONFIG['symptom_threshold'],
        medicine_threshold=CONFIG['medicine_threshold'],
        diagnosis_threshold=CONFIG['diagnosis_threshold'],
    )

    encoded_df, encode_stats = encoder.encode_all(pairs_df)

    # 保存编码后的数据
    encoded_df.to_csv(os.path.join(CONFIG['data_dir'], 'step3_encoded_data.csv'), index=False, encoding='utf-8-sig')
    print(f"编码数据已保存到: {os.path.join(CONFIG['data_dir'], 'step3_encoded_data.csv')}")

    # ========== 步骤4: 准备因果发现数据 ==========
    print("\n" + "=" * 60)
    print("步骤4: 准备因果发现数据")
    print("=" * 60)

    # 获取选中的特征列
    selected_cols = encoder.get_selected_columns(include_t1=True)
    print(f"选中特征数: {len(selected_cols)}")

    # 过滤出实际存在的列
    existing_cols = [col for col in selected_cols if col in encoded_df.columns]
    print(f"实际存在的特征数: {len(existing_cols)}")

    # 构建数据矩阵
    data_matrix = encoded_df[existing_cols].copy()

    # 填充缺失值
    data_matrix = data_matrix.fillna(0)

    # 清理数据：移除常数特征和不需要的列
    print("\n--- 数据清洗 ---")

    # 0. 移除所有object类型的列（文本列）
    obj_cols = data_matrix.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print(f"移除 {len(obj_cols)} 个object类型列（文本列）:")
        for col in obj_cols:
            print(f"  - {col}")
        data_matrix = data_matrix.drop(columns=obj_cols)
        existing_cols = [col for col in existing_cols if col not in obj_cols]

    # 1. 移除checkup_id相关列
    checkup_cols = [col for col in data_matrix.columns if 'checkup_id' in col]
    if checkup_cols:
        print(f"移除 {len(checkup_cols)} 个checkup_id列")
        data_matrix = data_matrix.drop(columns=checkup_cols)
        existing_cols = [col for col in existing_cols if col not in checkup_cols]

    # 2. 移除常数特征（方差=0）
    variances = data_matrix.var()
    constant_features = variances[variances == 0].index.tolist()
    if constant_features:
        print(f"移除 {len(constant_features)} 个常数特征（方差=0）:")
        for feat in constant_features:
            print(f"  - {feat}")
        data_matrix = data_matrix.drop(columns=constant_features)
        existing_cols = [col for col in existing_cols if col not in constant_features]

    # 3. 检查并移除高度相关的特征（r > 0.99）
    corr_matrix = data_matrix.corr().abs()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.99:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    if high_corr_pairs:
        print(f"发现 {len(high_corr_pairs)} 对高度相关的特征 (r > 0.99)")
        # 移除相关性 > 0.99 的特征中的一个
        to_drop = set()
        for col1, col2 in high_corr_pairs:
            # 保留方差较大的特征
            if data_matrix[col1].var() >= data_matrix[col2].var():
                to_drop.add(col2)
            else:
                to_drop.add(col1)

        if to_drop:
            print(f"移除 {len(to_drop)} 个高度相关特征:")
            for feat in to_drop:
                print(f"  - {feat}")
            data_matrix = data_matrix.drop(columns=list(to_drop))
            existing_cols = [col for col in existing_cols if col not in to_drop]

    print(f"\n清洗后数据矩阵形状: {data_matrix.shape}")
    print(f"数据类型:\n{data_matrix.dtypes.value_counts()}")

    # 保存最终数据矩阵
    data_matrix.to_csv(os.path.join(CONFIG['data_dir'], 'step4_data_matrix.csv'), index=False, encoding='utf-8-sig')

    # ========== 步骤5: 运行因果发现 ==========
    print("\n" + "=" * 60)
    print("步骤5: 运行因果发现算法")
    print("=" * 60)

    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        # 准备numpy数组
        X = data_matrix.values.astype(float)

        print(f"运行PC算法 (alpha={CONFIG['alpha']})...")
        cg = pc(X, alpha=CONFIG['alpha'], indep_test=fisherz)

        # 创建NetworkX图
        G = nx.DiGraph()
        G.add_nodes_from(existing_cols)

        # 从因果图中提取边
        # cg.G.graph 是邻接矩阵，其中：
        # graph[i,j] = 1 且 graph[j,i] = -1: i -> j (有向边)
        # graph[i,j] = -1 且 graph[j,i] = 1: j -> i (有向边)
        # graph[i,j] = 0 且 graph[j,i] = 0: 无边
        graph = cg.G.graph

        # 添加有向边
        for i in range(len(existing_cols)):
            for j in range(len(existing_cols)):
                if i != j and graph[i, j] == 1 and graph[j, i] == -1:
                    G.add_edge(existing_cols[i], existing_cols[j])

        print(f"PC算法完成")
        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  边数: {G.number_of_edges()}")

    except ImportError:
        print("警告: causal-learn 未安装，跳过PC算法")
        print("请安装: pip install causal-learn")
        G = nx.DiGraph()
        G.add_nodes_from(existing_cols)

    except Exception as e:
        print(f"PC算法运行出错: {e}")
        print("创建空图")
        G = nx.DiGraph()
        G.add_nodes_from(existing_cols)

    # ========== 步骤6: 应用约束 ==========
    print("\n" + "=" * 60)
    print("步骤6: 应用约束")
    print("=" * 60)

    # 创建变量分组（不包含medicines_t1）
    variable_groups = {
        'static': ['gender_encoded'] + [col for col in existing_cols if col.startswith('age_')],
        'symptoms_t': [col for col in existing_cols if col.endswith('_t') and not col.startswith('age_')
                       and not col.startswith('med_') and not col.startswith('diagnosis_')
                       and not col.startswith('gender_') and not col.startswith('time_')
                       and col not in ['age_t', 'age_binned']],
        'diagnosis_t': [col for col in existing_cols if col.startswith('diagnosis_') and col.endswith('_t')],
        'medicines_t': [col for col in existing_cols if col.startswith('med_') and col.endswith('_t')],
        'symptoms_t1': [col for col in existing_cols if col.endswith('_t1') and not col.startswith('med_')
                        and not col.startswith('diagnosis_')],
        'diagnosis_t1': [col for col in existing_cols if col.startswith('diagnosis_') and col.endswith('_t1')],
    }

    # 应用约束
    constraint_manager = SimpleConstraintManager()
    constraint_manager.set_variable_groups(variable_groups)
    G_constrained = constraint_manager.apply_constraints(G, verbose=True)

    # ========== 步骤7: 可视化 ==========
    print("\n" + "=" * 60)
    print("步骤7: 可视化因果图")
    print("=" * 60)

    plotter = SimpleCausalPlotter()

    # 绘制完整DAG
    if G_constrained.number_of_edges() > 0:
        fig = plotter.plot_dag(
            G_constrained,
            variable_groups=variable_groups,
            save_path=os.path.join(CONFIG['graphs_dir'], 'causal_dag.png'),
            title="鼻咽癌中西医结合诊疗因果网络 (简易版)"
        )
        print("完整DAG已保存")

        # 提取并绘制关键路径
        print("\n--- 提取关键因果路径 ---")

        # 路径1: 诊断 -> 药物 -> 症状
        paths_med = plotter.extract_paths(G_constrained, 'diagnosis_t', 'symptoms_t1',
                                          max_length=2, variable_groups=variable_groups)
        print(f"\n路径 诊断->药物->症状 (共{len(paths_med)}条):")
        plotter.print_paths(paths_med, max_paths=10)

        # 路径2: 症状 -> 症状
        paths_sym = plotter.extract_paths(G_constrained, 'symptoms_t', 'symptoms_t1',
                                          max_length=1, variable_groups=variable_groups)
        print(f"\n路径 症状->症状 (共{len(paths_sym)}条):")
        plotter.print_paths(paths_sym, max_paths=10)

    else:
        print("没有边可以绘制")

    # ========== 步骤8: 生成报告 ==========
    print("\n" + "=" * 60)
    print("步骤8: 生成报告")
    print("=" * 60)

    report = generate_report(CONFIG, encode_stats, G_constrained, variable_groups, paths_med if 'paths_med' in locals() else [])
    report_path = os.path.join(CONFIG['reports_dir'], 'simple_causal_discovery_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存到: {report_path}")

    print("\n" + "=" * 60)
    print("因果发现流程完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return G_constrained, data_matrix


def generate_report(config, encode_stats, G, variable_groups, medication_paths):
    """生成分析报告"""

    report = f"""# 简易版因果发现报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 数据概览

### 1.1 时序对统计

- 总时序对数: {config.get('n_pairs', 'N/A')}
- 涉及患者数: {config.get('n_patients', 'N/A')}

### 1.2 特征编码统计（编码前筛选）

"""

    # 适配新的统计信息结构
    if 'prefilter' in encode_stats:
        prefilter = encode_stats['prefilter']
        if 'symptoms' in prefilter:
            s = prefilter['symptoms']
            report += f"- **症状**: {s['n_original']} → {s['n_selected']} (移除 {s['removed']}, {s['removed']/s['n_original']*100:.1f}%)\n"
        if 'medicines' in prefilter:
            m = prefilter['medicines']
            report += f"- **药物**: {m['n_original']} → {m['n_selected']} (移除 {m['removed']}, {m['removed']/m['n_original']*100:.1f}%)\n"
        if 'diagnoses' in prefilter:
            d_t = prefilter['diagnoses']
            d_t1 = prefilter.get('diagnoses_t1', {'n_original': 0, 'n_selected': 0, 'removed': 0})
            total_orig = d_t['n_original'] + d_t1['n_original']
            total_sel = d_t['n_selected'] + d_t1['n_selected']
            total_removed = d_t['removed'] + d_t1['removed']
            report += f"- **诊断**: {total_orig} → {total_sel} (移除 {total_removed}, {total_removed/total_orig*100:.1f}%)\n"
    else:
        # 降级到旧结构
        for var_type, stats in encode_stats.items():
            if isinstance(stats, dict) and 'n_original' in stats:
                report += f"- **{var_type}**: {stats['n_original']} → {stats['n_selected']} "
                report += f"(移除 {stats['removed']}, {stats['removed']/stats['n_original']*100:.1f}%)\n"

    report += f"""

## 2. 因果网络概览

### 2.1 网络统计

- 节点总数: {G.number_of_nodes()}
- 边总数: {G.number_of_edges()}

### 2.2 节点分布

"""

    for group_name, nodes in variable_groups.items():
        existing_nodes = [n for n in nodes if n in G.nodes()]
        report += f"- **{group_name}**: {len(existing_nodes)} 个节点\n"

    report += """

## 3. 关键发现

### 3.1 因果边列表

"""

    if G.number_of_edges() > 0:
        # 按节点类型分组显示边
        edges_by_type = {
            'static -> t': [],
            'symptoms_t -> diagnosis_t': [],
            'diagnosis_t -> medicines_t': [],
            'medicines_t -> symptoms_t1': [],
            'symptoms_t -> symptoms_t1': [],
            '其他': []
        }

        for u, v in G.edges():
            if u in ['gender_encoded'] or u.startswith('age_'):
                edges_by_type['static -> t'].append((u, v))
            elif u.startswith('med_') and v.endswith('_t1') and not v.startswith('med_') and not v.startswith('diagnosis_'):
                edges_by_type['medicines_t -> symptoms_t1'].append((u, v))
            elif v.startswith('diagnosis_') and v.endswith('_t') and not u.startswith('med_'):
                edges_by_type['symptoms_t -> diagnosis_t'].append((u, v))
            elif u.startswith('diagnosis_') and v.startswith('med_'):
                edges_by_type['diagnosis_t -> medicines_t'].append((u, v))
            elif u.endswith('_t') and v.endswith('_t1') and not u.startswith('med_') and not u.startswith('diagnosis_'):
                edges_by_type['symptoms_t -> symptoms_t1'].append((u, v))
            else:
                edges_by_type['其他'].append((u, v))

        for edge_type, edges in edges_by_type.items():
            if edges:
                report += f"\n#### {edge_type} ({len(edges)}条)\n\n"
                for u, v in edges[:20]:  # 最多显示20条
                    report += f"- {u} → {v}\n"
                if len(edges) > 20:
                    report += f"- ... 还有 {len(edges) - 20} 条\n"

    report += """

### 3.2 药物疗效路径

"""

    if medication_paths:
        report += f"发现 {len(medication_paths)} 条「诊断 → 药物 → 症状」路径:\n\n"
        for i, path in enumerate(medication_paths[:20]):
            report += f"{i+1}. {' → '.join(path)}\n"
        if len(medication_paths) > 20:
            report += f"\n... 还有 {len(medication_paths) - 20} 条路径\n"
    else:
        report += "未发现明显的药物疗效路径\n"

    report += """

## 4. 局限性

1. **未使用Bootstrap稳定性选择**: 结果可能存在假阳性，需要进一步验证
2. **样本量相对较小**: 963条原始数据，可能导致因果结构不稳定
3. **未考虑时间间隔**: 未对时间间隔进行加权或筛选
4. **特征编码简化**: 使用简单的频率筛选和二值化编码

## 5. 后续改进建议

- [ ] 添加Bootstrap稳定性选择 (1000次重采样)
- [ ] 考虑时间间隔的影响
- [ ] 引入更多专家知识约束
- [ ] 尝试其他因果发现算法 (FCI, GES等)
- [ ] 与中医专家验证结果

## 6. 输出文件

- 数据文件: `outputs/data/`
- 图表文件: `outputs/graphs/causal_dag.png`
- 报告文件: `outputs/reports/simple_causal_discovery_report.md`

---

**报告生成完毕**
"""

    return report


if __name__ == "__main__":
    # 运行因果发现流程
    G, data_matrix = run_causal_discovery()
