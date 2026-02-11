"""
图工具模块

功能：
- 路径提取
- 图可视化
- 图分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Set
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class GraphUtils:
    """图工具类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化图工具

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}

    @staticmethod
    def find_paths(
        graph: nx.DiGraph,
        source: str,
        target: str,
        max_length: int = 4,
        cutoff: int = None,
    ) -> List[List[str]]:
        """
        查找两个节点之间的所有路径

        Parameters
        ----------
        graph : nx.DiGraph
            因果图
        source : str
            源节点
        target : str
            目标节点
        max_length : int
            最大路径长度
        cutoff : int, optional
            路径截断长度

        Returns
        -------
        list of list
            路径列表
        """
        try:
            paths = nx.all_simple_paths(
                graph,
                source=source,
                target=target,
                cutoff=cutoff or max_length,
            )

            # 过滤长度
            paths = [p for p in paths if len(p) <= max_length + 1]

            return paths

        except nx.NetworkXNoPath:
            logger.warning(f"节点 {source} 和 {target} 之间没有路径")
            return []

    @staticmethod
    def extract_pattern_paths(
        graph: nx.DiGraph,
        pattern: str,
        variable_types: Dict[str, str],
        max_length: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        根据模式提取路径

        Parameters
        ----------
        graph : nx.DiGraph
            因果图
        pattern : str
            路径模式（如 "syndrome -> medicine -> symptom"）
        variable_types : dict
            变量类型映射（如 {"S_乏力": "symptom"}）
        max_length : int
            最大路径长度

        Returns
        -------
        list of dict
            匹配的路径信息
        """
        # 解析模式
        type_pattern = [t.strip() for t in pattern.split("->")]

        # 查找匹配的路径
        matched_paths = []

        for source in graph.nodes():
            for target in graph.nodes():
                # 获取路径
                paths = GraphUtils.find_paths(graph, source, target, max_length)

                for path in paths:
                    # 检查是否匹配模式
                    path_types = [variable_types.get(node, "unknown") for node in path]

                    if path_types == type_pattern:
                        matched_paths.append({
                            "path": path,
                            "types": path_types,
                            "length": len(path) - 1,
                        })

        logger.info(f"找到 {len(matched_paths)} 条匹配 '{pattern}' 的路径")

        return matched_paths

    @staticmethod
    def analyze_graph_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
        """
        分析图统计信息
        """
        stats = {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph),
            "n_components": nx.number_weakly_connected_components(graph),
        }

        # 节点度数
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())

        stats["max_in_degree"] = max(in_degrees.values()) if in_degrees else 0
        stats["max_out_degree"] = max(out_degrees.values()) if out_degrees else 0
        stats["avg_in_degree"] = np.mean(list(in_degrees.values())) if in_degrees else 0
        stats["avg_out_degree"] = np.mean(list(out_degrees.values())) if out_degrees else 0

        return stats

    @staticmethod
    def find_motifs(graph: nx.DiGraph, motif_size: int = 3) -> Dict[str, int]:
        """
        查找图motif（子图模式）

        Parameters
        ----------
        graph : nx.DiGraph
            因果图
        motif_size : int
            motif大小

        Returns
        -------
        dict
            motif计数
        """
        # TODO: 实现motif检测
        logger.warning("Motif检测功能待实现")
        return {}

    @staticmethod
    def export_path_table(
        graph: nx.DiGraph,
        paths: List[List[str]],
        save_path: str = None,
    ) -> pd.DataFrame:
        """
        导出路径表

        Parameters
        ----------
        graph : nx.DiGraph
            因果图
        paths : list of list
            路径列表
        save_path : str, optional
            保存路径

        Returns
        -------
        pd.DataFrame
            路径表
        """
        path_data = []

        for i, path in enumerate(paths):
            path_data.append({
                "path_id": i + 1,
                "path": " -> ".join(path),
                "length": len(path) - 1,
                "source": path[0],
                "target": path[-1],
                "intermediate": " -> ".join(path[1:-1]) if len(path) > 2 else "",
            })

        df = pd.DataFrame(path_data)

        if save_path:
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info(f"路径表已保存至: {save_path}")

        return df

    @staticmethod
    def create_variable_type_mapping(variable_names: List[str]) -> Dict[str, str]:
        """
        根据变量名前缀推断变量类型

        Parameters
        ----------
        variable_names : list
            变量名列表

        Returns
        -------
        dict
            {变量名: 类型}
        """
        mapping = {}

        for var in variable_names:
            if var.startswith("S_"):
                mapping[var] = "symptom"
            elif var.startswith("M_"):
                mapping[var] = "medicine"
            elif var.startswith("D_"):
                mapping[var] = "syndrome"
            elif var.startswith("P_"):
                mapping[var] = "demographic"
            else:
                mapping[var] = "unknown"

        return mapping
