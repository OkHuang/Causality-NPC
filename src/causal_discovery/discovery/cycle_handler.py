"""
环处理模块

检测并移除图中的环，确保结果为DAG
"""

import networkx as nx


def remove_cycles(G: nx.DiGraph, max_iterations: int = 100, verbose: bool = True) -> nx.DiGraph:
    """
    移除图中的环，确保结果是DAG

    Parameters
    ----------
    G : nx.DiGraph
        输入图
    max_iterations : int
        最大迭代次数
    verbose : bool
        是否打印详细信息

    Returns
    -------
    nx.DiGraph
        无环图（DAG）
    """
    G_copy = G.copy()
    iteration = 0

    if verbose:
        print("\n=== 环检测与处理 ===")

    if not nx.is_directed_acyclic_graph(G_copy):
        print("[WARNING] 图中存在环！")

        while not nx.is_directed_acyclic_graph(G_copy) and iteration < max_iterations:
            iteration += 1

            # 找到一个环
            try:
                cycle = nx.find_cycle(G_copy, orientation='ignore')
            except nx.NetworkXNoCycle:
                break

            # 获取环中的边
            cycle_edges = [(u, v) if G_copy.has_edge(u, v) else (v, u)
                          for u, v, _ in cycle]

            # 移除环中的第一条边
            if cycle_edges:
                edge_to_remove = cycle_edges[0]
                if G_copy.has_edge(edge_to_remove[0], edge_to_remove[1]):
                    G_copy.remove_edge(edge_to_remove[0], edge_to_remove[1])
                    if iteration <= 5:
                        print(f"  迭代 {iteration}: 移除边 {edge_to_remove[0]} -> {edge_to_remove[1]}")

        if verbose:
            print(f"环处理完成，迭代 {iteration} 次")
            print(f"  原始边数: {G.number_of_edges()}")
            print(f"  清理后边数: {G_copy.number_of_edges()}")
            print(f"  移除边数: {G.number_of_edges() - G_copy.number_of_edges()}")

        # 验证清理后的图
        if nx.is_directed_acyclic_graph(G_copy):
            print("[OK] 清理后的图是无环图（DAG）")
        else:
            print("[WARNING] 清理后仍有环存在")

    else:
        if verbose:
            print("[OK] 图中无环，是DAG")

    return G_copy
