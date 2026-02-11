"""
简易版因果图可视化模块

功能：
1. 绘制有向无环图（DAG）
2. 分层布局展示
3. 节点颜色编码
4. 提取和展示特定因果路径
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import numpy as np

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


class SimpleCausalPlotter:
    """简易版因果图绘制器"""

    # 节点颜色定义
    COLOR_STATIC = '#95a5a6'     # 灰色：静态特征
    COLOR_SYMPTOM = '#e74c3c'    # 红色：症状
    COLOR_DIAGNOSIS = '#3498db'  # 蓝色：诊断
    COLOR_MEDICINE = '#27ae60'   # 绿色：药物

    def __init__(self, figsize: Tuple[int, int] = (20, 12)):
        """
        初始化绘制器

        Parameters
        ----------
        figsize : Tuple[int, int]
            图形大小
        """
        self.figsize = figsize

    def get_node_color(self, node: str) -> str:
        """
        根据节点类型返回颜色

        Parameters
        ----------
        node : str
            节点名

        Returns
        -------
        str
            颜色代码
        """
        if node in ['gender_encoded'] or node.startswith('age_'):
            return self.COLOR_STATIC
        elif node.startswith('med_'):
            return self.COLOR_MEDICINE
        elif node.startswith('diagnosis_'):
            return self.COLOR_DIAGNOSIS
        else:
            return self.COLOR_SYMPTOM

    def get_time_layer(self, node: str) -> int:
        """
        获取节点的时间层

        Parameters
        ----------
        node : str
            节点名

        Returns
        -------
        int
            时间层 (0=static, 1=t, 2=t1)
        """
        if node in ['gender_encoded'] or node.startswith('age_'):
            return 0
        elif node.endswith('_t'):
            return 1
        elif node.endswith('_t1'):
            return 2
        else:
            return 1

    def create_hierarchical_layout(self, G: nx.DiGraph, variable_groups: Dict[str, List[str]] = None) -> Dict:
        """
        创建分层布局

        Parameters
        ----------
        G : nx.DiGraph
            因果图
        variable_groups : Dict[str, List[str]], optional
            变量分组字典

        Returns
        -------
        Dict
            节点位置字典
        """
        pos = {}

        if variable_groups:
            # 使用变量分组
            layer_height = 1000
            layer_width = 50

            # Layer 0: 静态特征
            for i, node in enumerate(variable_groups.get('static', [])):
                if node in G.nodes():
                    pos[node] = (i * layer_width, 3 * layer_height)

            # Layer 1: t时刻症状
            for i, node in enumerate(variable_groups.get('symptoms_t', [])):
                if node in G.nodes():
                    pos[node] = (i * layer_width, 2 * layer_height)

            # Layer 2: t时刻诊断
            for i, node in enumerate(variable_groups.get('diagnosis_t', [])):
                if node in G.nodes():
                    pos[node] = (i * layer_width, 1.5 * layer_height)

            # Layer 3: t时刻药物
            for i, node in enumerate(variable_groups.get('medicines_t', [])):
                if node in G.nodes():
                    pos[node] = (i * layer_width, 1 * layer_height)

            # Layer 4: t+1时刻症状
            for i, node in enumerate(variable_groups.get('symptoms_t1', [])):
                if node in G.nodes():
                    pos[node] = (i * layer_width, 0 * layer_height)

            # Layer 5: t+1时刻诊断
            for i, node in enumerate(variable_groups.get('diagnosis_t1', [])):
                if node in G.nodes():
                    pos[node] = (i * layer_width, -0.5 * layer_height)

        else:
            # 自动分层
            nodes_by_layer = {0: [], 1: [], 2: []}
            for node in G.nodes():
                layer = self.get_time_layer(node)
                nodes_by_layer[layer].append(node)

            layer_height = 1000
            for layer, nodes in nodes_by_layer.items():
                for i, node in enumerate(nodes):
                    pos[node] = (i * 50, (2 - layer) * layer_height)

        return pos

    def plot_dag(self, G: nx.DiGraph,
                 variable_groups: Dict[str, List[str]] = None,
                 save_path: str = None,
                 title: str = "鼻咽癌中西医结合诊疗因果网络",
                 show_labels: bool = True,
                 node_size: int = 500) -> plt.Figure:
        """
        绘制因果有向无环图

        Parameters
        ----------
        G : nx.DiGraph
            因果图
        variable_groups : Dict[str, List[str]], optional
            变量分组字典
        save_path : str, optional
            保存路径
        title : str
            图标题
        show_labels : bool
            是否显示节点标签
        node_size : int
            节点大小

        Returns
        -------
        plt.Figure
            matplotlib图形对象
        """
        plt.figure(figsize=self.figsize)

        # 创建布局
        pos = self.create_hierarchical_layout(G, variable_groups)

        # 获取节点颜色
        node_colors = [self.get_node_color(node) for node in G.nodes()]

        # 绘制图
        nx.draw(G, pos,
                with_labels=show_labels,
                node_color=node_colors,
                node_size=node_size,
                font_size=8,
                arrowsize=20,
                edge_color='gray',
                alpha=0.8,
                font_weight='bold')

        plt.title(title, fontsize=16, fontweight='bold')

        # 添加图例
        legend_patches = [
            mpatches.Patch(color=self.COLOR_STATIC, label='静态特征'),
            mpatches.Patch(color=self.COLOR_SYMPTOM, label='症状'),
            mpatches.Patch(color=self.COLOR_DIAGNOSIS, label='诊断'),
            mpatches.Patch(color=self.COLOR_MEDICINE, label='药物'),
        ]
        plt.legend(handles=legend_patches, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图已保存到: {save_path}")

        return plt.gcf()

    def plot_subgraph(self, G: nx.DiGraph,
                      nodes: List[str],
                      save_path: str = None,
                      title: str = "因果子图") -> plt.Figure:
        """
        绘制子图

        Parameters
        ----------
        G : nx.DiGraph
            原始因果图
        nodes : List[str]
            要显示的节点列表
        save_path : str, optional
            保存路径
        title : str
            图标题

        Returns
        -------
        plt.Figure
            matplotlib图形对象
        """
        # 提取子图
        subgraph = G.subgraph(nodes).copy()

        # 添加节点之间的边
        edges_to_add = []
        for u, v in G.edges():
            if u in nodes and v in nodes:
                edges_to_add.append((u, v))

        plt.figure(figsize=(12, 8))

        # 使用spring layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)

        # 获取节点颜色
        node_colors = [self.get_node_color(node) for node in subgraph.nodes()]

        # 绘制
        nx.draw(subgraph, pos,
                with_labels=True,
                node_color=node_colors,
                node_size=800,
                font_size=10,
                arrowsize=20,
                edge_color='gray',
                alpha=0.8,
                font_weight='bold')

        plt.title(title, fontsize=14, fontweight='bold')

        # 添加图例
        legend_patches = [
            mpatches.Patch(color=self.COLOR_STATIC, label='静态特征'),
            mpatches.Patch(color=self.COLOR_SYMPTOM, label='症状'),
            mpatches.Patch(color=self.COLOR_DIAGNOSIS, label='诊断'),
            mpatches.Patch(color=self.COLOR_MEDICINE, label='药物'),
        ]
        plt.legend(handles=legend_patches, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"子图已保存到: {save_path}")

        return plt.gcf()

    def extract_paths(self, G: nx.DiGraph,
                     source_pattern: str,
                     target_pattern: str,
                     max_length: int = 3,
                     variable_groups: Dict[str, List[str]] = None) -> List[List[str]]:
        """
        提取特定模式的因果路径

        Parameters
        ----------
        G : nx.DiGraph
            因果图
        source_pattern : str
            源节点类型 (static, symptom, diagnosis, medicine)
        target_pattern : str
            目标节点类型
        max_length : int
            最大路径长度
        variable_groups : Dict[str, List[str]], optional
            变量分组字典

        Returns
        -------
        List[List[str]]
            路径列表
        """
        if variable_groups is None:
            return []

        # 获取源节点和目标节点
        source_nodes = variable_groups.get(source_pattern, [])
        target_nodes = variable_groups.get(target_pattern.replace('_t', '_t').replace('_t1', '_t1'), [])

        paths = []

        for source in source_nodes:
            if source not in G.nodes():
                continue
            for target in target_nodes:
                if target not in G.nodes():
                    continue

                try:
                    # 查找最短路径
                    path = nx.shortest_path(G, source, target)
                    if len(path) <= max_length + 1:  # 路径长度 = 节点数 - 1
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue

        return paths

    def print_paths(self, paths: List[List[str]], max_paths: int = 10):
        """
        打印路径

        Parameters
        ----------
        paths : List[List[str]]
            路径列表
        max_paths : int
            最多显示的路径数
        """
        print(f"\n发现 {len(paths)} 条路径")
        for i, path in enumerate(paths[:max_paths]):
            print(f"  路径 {i+1}: {' → '.join(path)}")
        if len(paths) > max_paths:
            print(f"  ... 还有 {len(paths) - max_paths} 条路径")

    def plot_path(self, G: nx.DiGraph,
                  path: List[str],
                  save_path: str = None,
                  title: str = "因果路径") -> plt.Figure:
        """
        绘制单条路径

        Parameters
        ----------
        G : nx.DiGraph
            原始因果图
        path : List[str]
            路径节点列表
        save_path : str, optional
            保存路径
        title : str
            图标题

        Returns
        -------
        plt.Figure
            matplotlib图形对象
        """
        # 创建路径子图
        path_graph = nx.DiGraph()
        path_graph.add_nodes_from(path)
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                path_graph.add_edge(path[i], path[i + 1])

        plt.figure(figsize=(12, 4))

        # 线性布局
        pos = {}
        for i, node in enumerate(path):
            pos[node] = (i, 0)

        # 获取节点颜色
        node_colors = [self.get_node_color(node) for node in path_graph.nodes()]

        # 绘制
        nx.draw(path_graph, pos,
                with_labels=True,
                node_color=node_colors,
                node_size=1000,
                font_size=10,
                arrowsize=30,
                edge_color='black',
                alpha=0.8,
                font_weight='bold',
                width=2)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"路径图已保存到: {save_path}")

        return plt.gcf()


def test_plotter():
    """测试绘制器"""
    # 创建测试图
    G = nx.DiGraph()
    nodes = [
        'gender_encoded',
        'age_20-25',
        'fatigue_t',
        'headache_t',
        'diagnosis_qixu_t',
        'med_huangqi_t',
        'med_dangshen_t',
        'fatigue_t1',
        'headache_t1',
    ]
    G.add_nodes_from(nodes)

    edges = [
        ('gender_encoded', 'fatigue_t'),
        ('age_20-25', 'diagnosis_qixu_t'),
        ('fatigue_t', 'diagnosis_qixu_t'),
        ('diagnosis_qixu_t', 'med_huangqi_t'),
        ('med_huangqi_t', 'fatigue_t1'),
        ('fatigue_t', 'fatigue_t1'),
    ]
    G.add_edges_from(edges)

    # 绘制
    plotter = SimpleCausalPlotter()
    fig = plotter.plot_dag(G, save_path="outputs/graphs/test_dag.png")

    # 提取路径
    variable_groups = {
        'static': ['gender_encoded', 'age_20-25'],
        'symptoms_t': ['fatigue_t', 'headache_t'],
        'diagnosis_t': ['diagnosis_qixu_t'],
        'medicines_t': ['med_huangqi_t', 'med_dangshen_t'],
        'symptoms_t1': ['fatigue_t1', 'headache_t1'],
    }

    paths = plotter.extract_paths(G, 'diagnosis_t', 'symptoms_t1', max_length=2, variable_groups=variable_groups)
    plotter.print_paths(paths)

    return fig


if __name__ == "__main__":
    test_plotter()
    plt.show()
