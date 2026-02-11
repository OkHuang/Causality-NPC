"""
因果发现算法模块

功能：
- 实现PC算法
- 实现FCI算法（隐变量）
- 支持自定义约束
- 图结构输出
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import logging

logger = logging.getLogger(__name__)

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.GraphUtils import GraphUtils as CausalLearnGraphUtils
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    logger.warning("causal-learn未安装，PC/FCI算法将不可用")


class CausalDiscovery:
    """因果发现算法封装"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化因果发现器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.graph = None
        self.edges = []

    def discover(
        self,
        data: pd.DataFrame,
        algorithm: str = "pc",
        alpha: float = 0.05,
        independence_test: str = "fisherz",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> nx.DiGraph:
        """
        执行因果发现

        Parameters
        ----------
        data : pd.DataFrame
            数据矩阵（变量为列，样本为行）
        algorithm : str
            算法名称 (pc, fci)
        alpha : float
            显著性水平
        independence_test : str
            独立性检验方法
        constraints : dict, optional
            约束条件

        Returns
        -------
        nx.DiGraph
            因果图（NetworkX格式）
        """
        if not CAUSALLEARN_AVAILABLE:
            raise ImportError("请安装 causal-learn: pip install causal-learn")

        logger.info(f"开始因果发现（算法: {algorithm}）...")

        # 转换数据为numpy数组
        data_array = data.values
        variable_names = data.columns.tolist()

        # 执行算法
        if algorithm.lower() == "pc":
            graph = self._run_pc(data_array, alpha, independence_test)
        elif algorithm.lower() == "fci":
            graph = self._run_fci(data_array, alpha, independence_test)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        # 应用约束
        if constraints:
            graph = self._apply_constraints(graph, constraints, variable_names)

        # 转换为NetworkX格式
        self.graph = self._to_networkx(graph, variable_names)
        self.edges = list(self.graph.edges())

        logger.info(f"因果发现完成，发现 {len(self.edges)} 条边")

        return self.graph

    def _run_pc(
        self,
        data: np.ndarray,
        alpha: float,
        independence_test: str,
    ) -> Any:
        """
        运行PC算法
        """
        # PC算法
        graph = pc(
            data,
            alpha=alpha,
            indep_test=independence_test,
        )

        return graph

    def _run_fci(
        self,
        data: np.ndarray,
        alpha: float,
        independence_test: str,
    ) -> Any:
        """
        运行FCI算法
        """
        # FCI算法
        graph = fci(
            data,
            alpha=alpha,
            indep_test=independence_test,
        )

        return graph

    def _apply_constraints(
        self,
        graph: Any,
        constraints: Dict[str, Any],
        variable_names: List[str],
    ) -> Any:
        """
        应用领域约束

        Parameters
        ----------
        graph : causallearn graph
            原始图
        constraints : dict
            约束条件
        variable_names : list
            变量名列表

        Returns
        -------
        graph
            应用约束后的图
        """
        # TODO: 实现约束应用逻辑
        logger.info("应用领域约束...")

        return graph

    def _to_networkx(
        self,
        causallearn_graph: Any,
        variable_names: List[str],
    ) -> nx.DiGraph:
        """
        将causallearn图转换为NetworkX图
        """
        # 方法1：使用causallearn内置的转换方法
        try:
            # causallearn提供了to_nx_graph方法
            nx_graph = causallearn_graph.to_nx_graph()
            if nx_graph is not None and nx_graph.number_of_nodes() > 0:
                # 设置节点名称
                if len(variable_names) == nx_graph.number_of_nodes():
                    node_mapping = {i: name for i, name in enumerate(variable_names)}
                    nx_graph = nx.relabel_nodes(nx_graph, node_mapping)
                logger.info(f"成功转换图，节点数: {nx_graph.number_of_nodes()}, 边数: {nx_graph.number_of_edges()}")
                return nx_graph
        except Exception as e:
            logger.warning(f"使用内置方法转换图时出错: {e}，尝试手动解析...")

        # 方法2：手动解析（备选方案）
        G = nx.DiGraph()
        G.add_nodes_from(variable_names)

        try:
            # 通过G属性访问边
            edges = causallearn_graph.G.get_graph_edges()
            from causallearn.graph.Endpoint import Endpoint

            for edge in edges:
                node1 = edge.get_node1()
                node2 = edge.get_node2()

                # 获取节点索引
                idx1 = causallearn_graph.G.get_nodes().index(node1)
                idx2 = causallearn_graph.G.get_nodes().index(node2)

                # 获取节点名称
                name1 = variable_names[idx1] if idx1 < len(variable_names) else str(idx1)
                name2 = variable_names[idx2] if idx2 < len(variable_names) else str(idx2)

                # 获取端点类型
                endpoint1 = edge.get_endpoint1()
                endpoint2 = edge.get_endpoint2()

                # 添加有向边（TAIL -> ARROW）
                if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
                    G.add_edge(name1, name2)
                elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
                    G.add_edge(name2, name1)
                # 无向边（TAIL-TAIL）或不确定的边不添加，保持DAG结构

            logger.info(f"手动解析完成，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
        except Exception as e:
            logger.warning(f"手动解析图时出错: {e}")

        return G

    def get_edges(self) -> List[Tuple[str, str]]:
        """
        获取边列表
        """
        return self.edges

    def get_adjacency_matrix(self) -> pd.DataFrame:
        """
        获取邻接矩阵
        """
        if self.graph is None:
            raise ValueError("尚未运行因果发现")

        nodes = sorted(self.graph.nodes())
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=nodes)

        return pd.DataFrame(adj_matrix, index=nodes, columns=nodes)

    def plot_graph(
        self,
        layout: str = "spring",
        node_size: int = 1000,
        figsize: Tuple[int, int] = (12, 8),
        save_path: str = None,
    ):
        """
        绘制因果图

        Parameters
        ----------
        layout : str
            布局算法 (spring, hierarchical, circular)
        node_size : int
            节点大小
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        if self.graph is None:
            raise ValueError("尚未运行因果发现")

        import matplotlib.pyplot as plt

        # 选择布局
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "hierarchical":
            # 为分层布局添加subset属性（如果不存在）
            if not all("subset" in self.graph.nodes[node] for node in self.graph.nodes):
                # 简单地根据节点索引分配层级
                for i, node in enumerate(self.graph.nodes):
                    self.graph.nodes[node]["subset"] = i % 3  # 分为3层
            pos = nx.multipartite_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # 绘制
        fig, ax = plt.subplots(figsize=figsize)

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_size,
            node_color="lightblue",
            alpha=0.9
        )

        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color="gray",
            arrowsize=20,
            width=1.5,
            alpha=0.7
        )

        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=10,
            font_family="sans-serif"
        )

        ax.axis("off")
        ax.set_title("因果发现结果", fontsize=16)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"因果图已保存至: {save_path}")

        plt.show()

    def export_graph(self, filepath: str, format: str = "gml"):
        """
        导出图到文件

        Parameters
        ----------
        filepath : str
            文件路径
        format : str
            格式 (gml, graphml, json)
        """
        if self.graph is None:
            raise ValueError("尚未运行因果发现")

        if format == "gml":
            nx.write_gml(self.graph, filepath)
        elif format == "graphml":
            nx.write_graphml(self.graph, filepath)
        elif format == "json":
            nx.write_node_link_json(self.graph, filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")

        logger.info(f"图已导出至: {filepath}")
