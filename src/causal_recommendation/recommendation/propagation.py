"""
因果传播推荐算法

沿因果图线性传播激活值，计算药物推荐得分
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional


def _propagate_activation(
    graph: nx.DiGraph,
    ate_dict: Dict[Tuple[str, str], float],
    mapped_vars: Dict[str, float],
    all_nodes: Set[str]
) -> Dict[str, float]:
    """
    执行因果传播算法

    Parameters
    ----------
    graph : nx.DiGraph
        因果图
    ate_dict : Dict[Tuple[str, str], float]
        ATE字典
    mapped_vars : Dict[str, float]
        映射后的患者变量
    all_nodes : Set[str]
        所有节点集合

    Returns
    -------
    Dict[str, float]
        所有节点的激活值
    """
    # 初始化激活值
    activation = {node: 0.0 for node in all_nodes}

    # 设置初始激活值
    for var, value in mapped_vars.items():
        activation[var] = float(value)

    # 获取拓扑排序
    try:
        topo_order = list(nx.topological_sort(graph))
    except nx.NetworkXError:
        # 如果有环，使用简单的层级排序
        topo_order = sorted(all_nodes)

    # 按拓扑顺序传播
    for node in topo_order:
        # 跳过初始输入节点
        if node in mapped_vars:
            continue

        # 计算来自所有父节点的激活
        parents = list(graph.predecessors(node))
        if not parents:
            continue

        # 计算新的激活值
        new_value = 0.0
        for parent in parents:
            edge_key = (parent, node)
            if edge_key in ate_dict:
                ate = ate_dict[edge_key]
                new_value += activation[parent] * ate

        # 更新激活值
        activation[node] = new_value

    return activation


def generate_explanations(
    graph: nx.DiGraph,
    ate_dict: Dict[Tuple[str, str], float],
    patient_info: Dict,
    all_nodes: Set[str],
    mapping_rules: Dict,
    meds: List[str],
    max_paths: int = 5
) -> Dict[str, List[Dict]]:
    """
    生成推荐的因果路径解释

    Parameters
    ----------
    graph : nx.DiGraph
        因果图
    ate_dict : Dict[Tuple[str, str], float]
        ATE字典
    patient_info : Dict
        患者信息
    all_nodes : Set[str]
        所有节点集合
    mapping_rules : Dict
        映射规则
    meds : List[str]
        需要解释的药物列表
    max_paths : int
        每个药物的最大路径数

    Returns
    -------
    Dict[str, List[Dict]]
        每个药物的因果路径解释
    """
    from ..data.patient_encoder import map_patient_to_graph

    explanations = {}

    # 映射患者信息
    mapping_result = map_patient_to_graph(patient_info, all_nodes, mapping_rules)
    input_vars = mapping_result['mapped_vars']

    for med in meds:
        paths = []

        # 对每个输入变量，找到传播路径
        for input_var in input_vars.keys():
            try:
                # 找到所有简单路径
                all_paths = nx.all_simple_paths(
                    graph, input_var, med, cutoff=5
                )

                for path in all_paths:
                    if len(path) < 2:
                        continue

                    # 计算路径的贡献
                    contribution = 1.0
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        if edge in ate_dict:
                            contribution *= ate_dict[edge]

                    # 只记录贡献较大的路径
                    if abs(contribution) > 0.001:
                        path_info = {
                            'path': path,
                            'contribution': contribution,
                            'edges': [
                                {
                                    'source': path[i],
                                    'target': path[i+1],
                                    'ate': ate_dict.get((path[i], path[i+1]), 0)
                                }
                                for i in range(len(path) - 1)
                                if (path[i], path[i+1]) in ate_dict
                            ]
                        }
                        paths.append(path_info)

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        # 按贡献排序
        paths.sort(key=lambda x: abs(x['contribution']), reverse=True)

        if paths:
            explanations[med] = paths[:max_paths]

    return explanations


def causal_propagation_recommend(
    graph: nx.DiGraph,
    ate_dict: Dict[Tuple[str, str], float],
    patient_info: Dict,
    all_nodes: Set[str],
    mapping_rules: Dict,
    threshold_positive: float = 0.05,
    threshold_negative: float = -0.05,
    top_k: Optional[int] = None,
    max_paths: int = 5
) -> Dict:
    """
    使用因果传播算法进行药物推荐

    Parameters
    ----------
    graph : nx.DiGraph
        因果图
    ate_dict : Dict[Tuple[str, str], float]
        ATE字典 (source, target) -> ate
    patient_info : Dict
        患者信息
    all_nodes : Set[str]
        所有节点集合
    mapping_rules : Dict
        映射规则字典
    threshold_positive : float
        推荐阈值（正向）
    threshold_negative : float
        警告阈值（负向）
    top_k : int, optional
        返回前k个推荐药物
    max_paths : int
        最大解释路径数

    Returns
    -------
    Dict
        推荐结果，包含：
        - recommended: 推荐药物字典 {med: score}
        - not_recommended: 不推荐药物字典 {med: score}
        - neutral: 中性药物列表
        - explanations: 路径解释
        - all_scores: 所有药物得分
    """
    from ..data.patient_encoder import map_patient_to_graph

    # 1. 映射患者信息
    mapping_result = map_patient_to_graph(patient_info, all_nodes, mapping_rules)
    mapped_vars = mapping_result['mapped_vars']

    # 2. 执行传播
    activation = _propagate_activation(graph, ate_dict, mapped_vars, all_nodes)

    # 3. 提取药物节点
    med_nodes = sorted([n for n in all_nodes if n.startswith('med_')])
    med_scores = {med: activation[med] for med in med_nodes}

    # 4. 分类
    recommended = {
        med: score for med, score in med_scores.items()
        if score >= threshold_positive
    }

    not_recommended = {
        med: score for med, score in med_scores.items()
        if score <= threshold_negative
    }

    neutral = {
        med: score for med, score in med_scores.items()
        if threshold_negative < score < threshold_positive
    }

    # 5. 排序
    recommended = dict(sorted(recommended.items(), key=lambda x: -x[1]))
    not_recommended = dict(sorted(not_recommended.items(), key=lambda x: x[1]))

    # 6. 限制返回数量
    if top_k is not None:
        recommended = dict(list(recommended.items())[:top_k])

    # 7. 生成因果路径解释
    all_meds = list(recommended.keys()) + list(not_recommended.keys())
    explanations = generate_explanations(
        graph, ate_dict, patient_info, all_nodes,
        mapping_rules, all_meds, max_paths
    )

    return {
        'recommended': recommended,
        'not_recommended': not_recommended,
        'neutral': list(neutral.keys()),
        'explanations': explanations,
        'all_scores': med_scores,
        'mapping_result': mapping_result
    }
