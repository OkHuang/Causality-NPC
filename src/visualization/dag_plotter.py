"""
DAG绘制模块

功能：
- 绘制有向无环图
- 自定义节点和边样式
- 支持时序布局
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import logging

logger = logging.getLogger(__name__)


class DAGPlotter:
    """DAG绘制器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化绘制器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.node_styles = self.config.get("node_styles", {})
        self.edge_styles = self.config.get("edge_styles", {})

    def plot(
        self,
        graph: nx.DiGraph,
        layout: str = "hierarchical",
        node_types: Optional[Dict[str, str]] = None,
        edge_confidence: Optional[Dict[Tuple[str, str], float]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ):
        """
        绘制DAG

        Parameters
        ----------
        graph : nx.DiGraph
            因果图
        layout : str
            布局类型
        node_types : dict, optional
            节点类型（用于着色）
        edge_confidence : dict, optional
            边的置信度（用于边样式）
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 计算布局
        pos = self._get_layout(graph, layout)

        # 绘制边
        self._draw_edges(graph, pos, ax, edge_confidence)

        # 绘制节点
        self._draw_nodes(graph, pos, ax, node_types)

        # 绘制标签
        self._draw_labels(graph, pos, ax)

        # 图例
        if node_types:
            self._add_legend(node_types, ax)

        ax.axis("off")
        ax.set_title("因果DAG", fontsize=16, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"DAG已保存至: {save_path}")

        plt.show()

    def _get_layout(self, graph: nx.DiGraph, layout: str) -> Dict[str, Tuple[float, float]]:
        """
        计算节点布局
        """
        if layout == "spring":
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "hierarchical":
            # 分层布局（按时间步）
            pos = self._hierarchical_layout(graph)
        elif layout == "force_directed":
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)

        return pos

    def _hierarchical_layout(self, graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        分层布局（考虑时序）
        """
        # 根据节点名推断层级
        levels = {}
        for node in graph.nodes():
            if "_t" in node:
                level = 0
            elif "_t1" in node or "_t+1" in node:
                level = 1
            else:
                level = 0
            levels[node] = level

        # 使用networkx的分层布局
        try:
            pos = nx.multipartite_layout(graph, subset_key="level")
        except:
            # 备选：手动分层
            pos = {}
            level_nodes = {}
            for node, level in levels.items():
                if level not in level_nodes:
                    level_nodes[level] = []
                level_nodes[level].append(node)

            # 为每个层分配位置
            for level, nodes in level_nodes.items():
                n_nodes = len(nodes)
                for i, node in enumerate(nodes):
                    x = level * 2
                    y = (i - n_nodes / 2) * 1.5
                    pos[node] = (x, y)

        return pos

    def _draw_nodes(
        self,
        graph: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        ax: plt.Axes,
        node_types: Optional[Dict[str, str]],
    ):
        """
        绘制节点
        """
        if node_types is None:
            # 默认样式
            nx.draw_networkx_nodes(
                graph, pos,
                node_size=1500,
                node_color="lightblue",
                alpha=0.9,
                ax=ax
            )
        else:
            # 按类型绘制
            for node_type, nodes in self._group_nodes_by_type(graph, node_types).items():
                color = self.node_styles.get(node_type, {}).get("color", "lightblue")
                shape = self.node_styles.get(node_type, {}).get("shape", "o")

                nx.draw_networkx_nodes(
                    graph, pos,
                    nodelist=nodes,
                    node_size=1500,
                    node_color=color,
                    node_shape=shape,
                    alpha=0.9,
                    label=node_type,
                    ax=ax
                )

    def _draw_edges(
        self,
        graph: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        ax: plt.Axes,
        edge_confidence: Optional[Dict[Tuple[str, str], float]],
    ):
        """
        绘制边
        """
        if edge_confidence is None:
            # 默认样式
            nx.draw_networkx_edges(
                graph, pos,
                edge_color="gray",
                arrowsize=20,
                width=1.5,
                alpha=0.7,
                arrowstyle="->",
                ax=ax
            )
        else:
            # 按置信度绘制
            for edge in graph.edges():
                confidence = edge_confidence.get(edge, 0)

                if confidence >= 0.9:
                    style = self.edge_styles.get("confident", {})
                    color = style.get("color", "#2C3E50")
                    width = style.get("width", 2.0)
                elif confidence >= 0.7:
                    style = self.edge_styles.get("moderate", {})
                    color = style.get("color", "#7F8C8D")
                    width = style.get("width", 1.5)
                else:
                    style = self.edge_styles.get("weak", {})
                    color = style.get("color", "#BDC3C7")
                    width = style.get("width", 1.0)

                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=[edge],
                    edge_color=color,
                    arrowsize=20,
                    width=width,
                    alpha=0.7,
                    arrowstyle="->",
                    ax=ax
                )

    def _draw_labels(
        self,
        graph: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        ax: plt.Axes,
    ):
        """
        绘制节点标签
        """
        nx.draw_networkx_labels(
            graph, pos,
            font_size=9,
            font_family="sans-serif",
            font_weight="bold",
            ax=ax
        )

    def _group_nodes_by_type(
        self,
        graph: nx.DiGraph,
        node_types: Dict[str, str],
    ) -> Dict[str, List[str]]:
        """
        按类型分组节点
        """
        groups = {}
        for node in graph.nodes():
            node_type = node_types.get(node, "unknown")
            if node_type not in groups:
                groups[node_type] = []
            groups[node_type].append(node)
        return groups

    def _add_legend(self, node_types: Dict[str, str], ax: plt.Axes):
        """
        添加图例
        """
        unique_types = set(node_types.values())
        patches = []

        for node_type in unique_types:
            color = self.node_styles.get(node_type, {}).get("color", "gray")
            label = node_type
            patch = mpatches.Patch(color=color, label=label)
            patches.append(patch)

        ax.legend(handles=patches, loc="upper left")
