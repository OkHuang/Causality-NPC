"""
PC算法模块

运行PC算法进行因果发现
"""

import pandas as pd
import networkx as nx
from typing import Tuple


def run_pc_algorithm(data_matrix: pd.DataFrame, node_names: list, alpha: float = 0.05) -> nx.DiGraph:
    """
    运行PC算法

    Parameters
    ----------
    data_matrix : pd.DataFrame
        数据矩阵
    node_names : list
        节点名称列表
    alpha : float
        显著性水平

    Returns
    -------
    nx.DiGraph
        发现的因果图
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        # 准备numpy数组
        X = data_matrix.values.astype(float)

        print(f"运行PC算法 (alpha={alpha})...")
        cg = pc(X, alpha=alpha, indep_test=fisherz)

        # 创建NetworkX图
        G = nx.DiGraph()
        G.add_nodes_from(node_names)

        # 从因果图中提取边
        # cg.G.graph 是邻接矩阵，其中：
        # graph[i,j] = 1 且 graph[j,i] = -1: i -> j (有向边)
        graph = cg.G.graph

        # 添加有向边
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if i != j and graph[i, j] == 1 and graph[j, i] == -1:
                    G.add_edge(node_names[i], node_names[j])

        print(f"PC算法完成")
        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  边数: {G.number_of_edges()}")

        return G

    except ImportError:
        print("警告: causal-learn 未安装，跳过PC算法")
        print("请安装: pip install causal-learn")
        return nx.DiGraph()
        G.add_nodes_from(node_names)

    except Exception as e:
        print(f"PC算法运行出错: {e}")
        print("创建空图")
        G = nx.DiGraph()
        G.add_nodes_from(node_names)
        return G
