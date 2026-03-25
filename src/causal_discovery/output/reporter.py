"""
报告生成模块

生成因果发现报告
"""

import networkx as nx
from typing import Dict, List
from datetime import datetime


class DiscoveryReporter:
    """发现报告生成器"""

    def generate(self, G: nx.DiGraph, stats: Dict, variable_groups: Dict = None) -> str:
        """
        生成发现报告

        Parameters
        ----------
        G : nx.DiGraph
            因果图
        stats : Dict
            统计信息
        variable_groups : Dict, optional
            变量分组

        Returns
        -------
        str
            Markdown格式的报告
        """
        report = f"""# 因果发现报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 数据概览

### 1.1 时序对统计

"""

        # 添加统计信息
        if 'n_pairs' in stats:
            report += f"- 总时序对数: {stats['n_pairs']}\n"
            report += f"- 涉及患者数: {stats.get('n_patients', 'N/A')}\n"
            report += f"- 平均时间间隔: {stats.get('mean_time_delta', 'N/A'):.1f} 天\n"

        # 添加特征编码统计
        report += "\n### 1.2 特征编码统计\n\n"

        if 'prefilter' in stats:
            prefilter = stats['prefilter']

            if 'symptoms' in prefilter:
                s = prefilter['symptoms']
                report += f"- **症状**: {s['n_original']} -> {s['n_selected']} "
                report += f"(移除 {s['removed']}, {s['removed']/s['n_original']*100:.1f}%)\n"

            if 'medicines' in prefilter:
                m = prefilter['medicines']
                report += f"- **药物**: {m['n_original']} -> {m['n_selected']} "
                report += f"(移除 {m['removed']}, {m['removed']/m['n_original']*100:.1f}%)\n"

            if 'diagnoses' in prefilter:
                d_t = prefilter['diagnoses']
                d_t1 = prefilter.get('diagnoses_t1', {'n_original': 0, 'n_selected': 0, 'removed': 0})
                total_orig = d_t['n_original'] + d_t1['n_original']
                total_sel = d_t['n_selected'] + d_t1['n_selected']
                total_removed = d_t['removed'] + d_t1['removed']
                report += f"- **诊断**: {total_orig} -> {total_sel} "
                report += f"(移除 {total_removed}, {total_removed/total_orig*100:.1f}%)\n"

        # 添加网络统计
        report += "\n## 2. 因果网络概览\n\n"
        report += "### 2.1 网络统计\n\n"
        report += f"- 节点总数: {G.number_of_nodes()}\n"
        report += f"- 边总数: {G.number_of_edges()}\n"
        report += f"- 是否为DAG: {nx.is_directed_acyclic_graph(G)}\n"

        # 添加节点分布
        if variable_groups:
            report += "\n### 2.2 节点分布\n\n"
            for group_name, nodes in variable_groups.items():
                existing_nodes = [n for n in nodes if n in G.nodes()]
                report += f"- **{group_name}**: {len(existing_nodes)} 个节点\n"

        # 添加因果边列表
        report += "\n## 3. 因果边列表\n\n"

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
                    report += f"\n### {edge_type} ({len(edges)}条)\n\n"
                    for u, v in edges[:20]:  # 最多显示20条
                        report += f"- {u} -> {v}\n"
                    if len(edges) > 20:
                        report += f"- ... 还有 {len(edges) - 20} 条\n"

        report += "\n---\n\n**报告生成完毕**\n"

        return report
