"""
简易版因果约束管理模块

功能：
1. 定义因果发现的硬约束
2. 检查并移除违反约束的边
3. 应用领域知识约束
"""

import networkx as nx
from typing import List, Tuple, Dict


class SimpleConstraintManager:
    """简易版约束管理器"""

    def __init__(self, variable_groups: Dict[str, List[str]] = None):
        """
        初始化约束管理器

        Parameters
        ----------
        variable_groups : Dict[str, List[str]]
            变量分组字典
        """
        self.variable_groups = variable_groups or {}
        self.forbidden_patterns = []
        self._init_default_constraints()

    def _init_default_constraints(self):
        """初始化默认的硬约束"""
        # 禁止的边模式 (source_pattern, target_pattern)
        # 支持 * 通配符

        # 1. 未来不能影响过去
        self.forbidden_patterns.append(('*_t1', '*_t'))
        self.forbidden_patterns.append(('*_t1', 'age_*'))
        self.forbidden_patterns.append(('*_t1', 'gender_*'))

        # 2. 药物不能反向影响当前诊断
        self.forbidden_patterns.append(('med_*_t', 'diagnosis_*_t'))
        self.forbidden_patterns.append(('med_*_t', 'chinese_*_t'))
        self.forbidden_patterns.append(('med_*_t', 'western_*_t'))

        # 3. 诊断不能反向影响当前症状
        self.forbidden_patterns.append(('diagnosis_*_t', '*_t'))  # 诊断在t时刻已经是结果
        self.forbidden_patterns.append(('chinese_*_t', '*_t'))
        self.forbidden_patterns.append(('western_*_t', '*_t'))

        # 4. 人口学特征不能被其他变量影响
        self.forbidden_patterns.append(('*', 'age_*'))
        self.forbidden_patterns.append(('*', 'gender_*'))

        # 5. t+1时刻的变量不能指向t时刻的药物
        self.forbidden_patterns.append(('*_t1', 'med_*_t'))

        # 6. 时间相关变量之间不应该有边
        self.forbidden_patterns.append(('time_*', '*'))
        self.forbidden_patterns.append(('*', 'time_*'))

        print(f"已加载 {len(self.forbidden_patterns)} 个默认约束")

    def add_constraint(self, source_pattern: str, target_pattern: str):
        """
        添加自定义约束

        Parameters
        ----------
        source_pattern : str
            源节点模式，支持 * 通配符
        target_pattern : str
            目标节点模式，支持 * 通配符
        """
        self.forbidden_patterns.append((source_pattern, target_pattern))
        print(f"添加约束: {source_pattern} -> {target_pattern}")

    def matches_pattern(self, node: str, pattern: str) -> bool:
        """
        检查节点名是否匹配模式

        Parameters
        ----------
        node : str
            节点名
        pattern : str
            模式，支持 * 通配符

        Returns
        -------
        bool
            是否匹配
        """
        # 处理精确匹配
        if pattern == '*':
            return True

        # 处理后缀匹配
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return node.startswith(prefix)

        # 处理前缀匹配
        if pattern.startswith('*'):
            suffix = pattern[1:]
            return node.endswith(suffix)

        # 处理中间通配符
        if '*' in pattern:
            parts = pattern.split('*')
            if len(parts) == 2:
                return node.startswith(parts[0]) and node.endswith(parts[1])

        # 精确匹配
        return node == pattern

    def is_forbidden(self, source: str, target: str) -> bool:
        """
        检查边是否被禁止

        Parameters
        ----------
        source : str
            源节点
        target : str
            目标节点

        Returns
        -------
        bool
            是否被禁止
        """
        for src_pattern, tgt_pattern in self.forbidden_patterns:
            if self.matches_pattern(source, src_pattern) and self.matches_pattern(target, tgt_pattern):
                return True
        return False

    def apply_constraints(self, graph: nx.DiGraph, verbose: bool = True) -> nx.DiGraph:
        """
        应用约束，移除违反约束的边

        Parameters
        ----------
        graph : nx.DiGraph
            原始因果图
        verbose : bool
            是否打印详细信息

        Returns
        -------
        nx.DiGraph
            应用约束后的因果图
        """
        if verbose:
            print(f"\n=== 应用约束 ===")
            print(f"原始边数: {graph.number_of_edges()}")

        edges_to_remove = []

        # 检查每条边
        for u, v in graph.edges():
            if self.is_forbidden(u, v):
                edges_to_remove.append((u, v))
                if verbose:
                    print(f"  移除边: {u} -> {v}")

        # 移除边
        graph.remove_edges_from(edges_to_remove)

        if verbose:
            print(f"移除了 {len(edges_to_remove)} 条违反约束的边")
            print(f"剩余边数: {graph.number_of_edges()}")

        return graph

    def get_allowed_edges(self, nodes: List[str]) -> List[Tuple[str, str]]:
        """
        根据约束获取所有允许的边

        Parameters
        ----------
        nodes : List[str]
            节点列表

        Returns
        -------
        List[Tuple[str, str]]
            允许的边列表
        """
        allowed = []

        for source in nodes:
            for target in nodes:
                if source == target:
                    continue
                if not self.is_forbidden(source, target):
                    allowed.append((source, target))

        print(f"总可能的边: {len(nodes) * (len(nodes) - 1)}")
        print(f"允许的边: {len(allowed)}")
        print(f"禁止的边: {len(nodes) * (len(nodes) - 1) - len(allowed)}")

        return allowed

    def set_variable_groups(self, variable_groups: Dict[str, List[str]]):
        """
        设置变量分组

        Parameters
        ----------
        variable_groups : Dict[str, List[str]]
            变量分组字典
        """
        self.variable_groups = variable_groups
        print(f"设置变量分组: {list(variable_groups.keys())}")

    def get_node_type(self, node: str) -> str:
        """
        获取节点类型

        Parameters
        ----------
        node : str
            节点名

        Returns
        -------
        str
            节点类型 (static, symptom, diagnosis, medicine)
        """
        if node in ['gender_encoded'] or node.startswith('age_'):
            return 'static'
        elif node.startswith('med_'):
            return 'medicine'
        elif node.startswith('diagnosis_'):
            return 'diagnosis'
        else:
            return 'symptom'

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
            return 1  # 默认为t时刻


def test_constraints():
    """测试约束管理器"""
    # 创建测试图
    G = nx.DiGraph()
    nodes = [
        'gender_encoded',
        'age_20-25',
        'fatigue_t',
        'fatigue_t1',
        'med_huangqi_t',
        'diagnosis_qixu_t',
    ]
    G.add_nodes_from(nodes)

    # 添加一些边（包括违反约束的）
    edges = [
        ('gender_encoded', 'fatigue_t'),      # 允许
        ('fatigue_t', 'diagnosis_qixu_t'),    # 允许
        ('diagnosis_qixu_t', 'med_huangqi_t'), # 允许
        ('med_huangqi_t', 'fatigue_t1'),      # 允许
        ('fatigue_t1', 'fatigue_t'),          # 禁止：未来影响过去
        ('med_huangqi_t', 'diagnosis_qixu_t'), # 禁止：药物影响当前诊断
    ]
    G.add_edges_from(edges)

    # 应用约束
    manager = SimpleConstraintManager()
    G_constrained = manager.apply_constraints(G, verbose=True)

    print("\n=== 剩余的边 ===")
    for u, v in G_constrained.edges():
        print(f"  {u} -> {v}")

    return G_constrained


if __name__ == "__main__":
    test_constraints()
