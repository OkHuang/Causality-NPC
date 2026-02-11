"""
约束管理模块

功能：
- 定义时序约束
- 定义领域知识约束
- 应用约束到因果图
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Set, Tuple
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class ConstraintManager:
    """约束管理器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化约束管理器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.constraints = {
            "forbidden": set(),  # 禁止的边
            "required": set(),   # 必需的边
            "temporal": {},      # 时序约束
        }

        # 加载约束
        self._load_constraints()

    def _load_constraints(self):
        """
        从配置加载约束
        """
        # 加载时序约束
        temporal_rules = self.config.get("temporal", [])
        for rule in temporal_rules:
            self._add_temporal_constraint(rule)

        # 加载领域知识约束
        domain_rules = self.config.get("domain_knowledge", [])
        for rule in domain_rules:
            self._add_domain_constraint(rule)

        logger.info(f"已加载 {len(self.constraints['forbidden'])} 个禁止约束")
        logger.info(f"已加载 {len(self.constraints['required'])} 个必需约束")

    def _add_temporal_constraint(self, rule: Dict[str, Any]):
        """
        添加时序约束
        """
        rule_type = rule.get("type")
        rule_name = rule.get("rule")

        if rule_type == "forbid" and rule_name == "no_future_to_past":
            # 未来不能影响过去
            # 这在变量命名中体现：变量_t 不能指向变量_t-1
            self.constraints["temporal"]["no_future_to_past"] = True
            logger.info("添加约束: 禁止未来影响过去")

        elif rule_type == "forbid" and rule_name == "no_instantaneous_treatment":
            # 药物不能瞬时起效
            self.constraints["temporal"]["no_instantaneous_treatment"] = True
            logger.info("添加约束: 禁止药物瞬时起效")

    def _add_domain_constraint(self, rule: Dict[str, Any]):
        """
        添加领域知识约束
        """
        rule_type = rule.get("type")
        rule_name = rule.get("rule")

        if rule_type == "require" and rule_name == "symptoms_to_medicine_forbidden":
            # 症状不能指向药物（反向因果）
            self.constraints["temporal"]["symptoms_to_medicine"] = "forbid"
            logger.info("添加约束: 症状不能指向药物")

    def apply_constraints(
        self,
        graph: nx.DiGraph,
        variable_names: List[str],
    ) -> nx.DiGraph:
        """
        应用约束到图

        Parameters
        ----------
        graph : nx.DiGraph
            原始图
        variable_names : list
            变量名列表

        Returns
        -------
        nx.DiGraph
            应用约束后的图
        """
        logger.info("应用约束...")

        G = graph.copy()

        # 应用时序约束
        G = self._apply_temporal_constraints(G, variable_names)

        # 应用禁止边约束
        for edge in self.constraints["forbidden"]:
            if G.has_edge(*edge):
                G.remove_edge(*edge)
                logger.debug(f"移除禁止边: {edge}")

        # 确保必需边存在
        for edge in self.constraints["required"]:
            if not G.has_edge(*edge):
                logger.warning(f"缺少必需边: {edge}")

        logger.info(f"约束应用完成，剩余边数: {G.number_of_edges()}")

        return G

    def _apply_temporal_constraints(
        self,
        graph: nx.DiGraph,
        variable_names: List[str],
    ) -> nx.DiGraph:
        """
        应用时序约束
        """
        G = graph.copy()

        # 解析变量名的时间标签
        var_time_map = {}
        for var in variable_names:
            # 提取时间标签（如 "S_乏力_t" -> t, "S_乏力_t1" -> t1）
            if "_t" in var or "_t1" in var:
                parts = var.rsplit("_", 1)
                if len(parts) == 2:
                    name, time_tag = parts
                    var_time_map[var] = time_tag

        # 禁止未来指向过去
        if self.constraints["temporal"].get("no_future_to_past"):
            for edge in list(G.edges()):
                source, target = edge

                if source in var_time_map and target in var_time_map:
                    source_time = var_time_map[source]
                    target_time = var_time_map[target]

                    # 如果源时间 > 目标时间，移除该边
                    if source_time > target_time:
                        G.remove_edge(*edge)
                        logger.debug(f"移除边（违反时序）: {edge}")

        # 禁止药物瞬时起效
        if self.constraints["temporal"].get("no_instantaneous_treatment"):
            for edge in list(G.edges()):
                source, target = edge

                # 如果源是药物且时间是_t，目标是症状且时间是_t
                if ("M_" in source and "_t" in source) and ("S_" in target and "_t" in target):
                    G.remove_edge(*edge)
                    logger.debug(f"移除边（瞬时效应）: {edge}")

        return G

    def add_forbidden_edge(self, source: str, target: str):
        """
        手动添加禁止边
        """
        self.constraints["forbidden"].add((source, target))
        logger.info(f"添加禁止边: {source} -> {target}")

    def add_required_edge(self, source: str, target: str):
        """
        手动添加必需边
        """
        self.constraints["required"].add((source, target))
        logger.info(f"添加必需边: {source} -> {target}")

    def get_constraints_summary(self) -> Dict[str, Any]:
        """
        获取约束摘要
        """
        return {
            "n_forbidden": len(self.constraints["forbidden"]),
            "n_required": len(self.constraints["required"]),
            "temporal_rules": self.constraints["temporal"],
        }
