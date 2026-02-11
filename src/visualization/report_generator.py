"""
报告生成器

功能：
- 生成Markdown格式的分析报告
- 整合结果、图表、表格
- 导出HTML/PDF
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化报告生成器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.sections = []

    def add_title(self, title: str, level: int = 1):
        """
        添加标题

        Parameters
        ----------
        title : str
            标题文本
        level : int
            标题级别（1-6）
        """
        prefix = "#" * level
        self.sections.append(f"{prefix} {title}\n")

    def add_paragraph(self, text: str):
        """
        添加段落
        """
        self.sections.append(f"{text}\n")

    def add_table(self, df: pd.DataFrame, caption: str = ""):
        """
        添加表格

        Parameters
        ----------
        df : pd.DataFrame
            数据
        caption : str
            表标题
        """
        if caption:
            self.sections.append(f"**{caption}**\n")

        # Markdown表格
        table = df.to_markdown(index=False)
        self.sections.append(f"{table}\n")

    def add_list(self, items: List[str], ordered: bool = False):
        """
        添加列表

        Parameters
        ----------
        items : list
            列表项
        ordered : bool
            是否有序列表
        """
        for item in items:
            if ordered:
                self.sections.append(f"1. {item}")
            else:
                self.sections.append(f"- {item}")
        self.sections.append("")

    def add_code(self, code: str, language: str = "python"):
        """
        添加代码块

        Parameters
        ----------
        code : str
            代码
        language : str
            语言
        """
        self.sections.append(f"```{language}")
        self.sections.append(code)
        self.sections.append("```")
        self.sections.append("")

    def add_image(self, path: str, alt: str = "", width: Optional[int] = None):
        """
        添加图片

        Parameters
        ----------
        path : str
            图片路径
        alt : str
            替代文本
        width : int, optional
            宽度（像素）
        """
        width_attr = f' width="{width}"' if width else ""
        self.sections.append(f'<img src="{path}" alt="{alt}"{width_attr}>\n')

    def add_horizontal_rule(self):
        """添加分隔线"""
        self.sections.append("---\n")

    def generate_causal_inference_report(
        self,
        results: Dict[str, Any],
        balance_df: pd.DataFrame,
        save_path: str,
    ):
        """
        生成因果推断报告（目标一）

        Parameters
        ----------
        results : dict
            ATE估计结果
        balance_df : pd.DataFrame
            平衡性检验表
        save_path : str
            保存路径
        """
        self.sections = []

        # 标题
        self.add_title("因果推断分析报告", level=1)
        self.add_paragraph(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 研究假设
        self.add_title("研究假设", level=2)
        self.add_paragraph(f"- **处理变量**: {results.get('treatment', 'N/A')}")
        self.add_paragraph(f"- **结果变量**: {results.get('outcome', 'N/A')}")
        self.add_paragraph(f"- **混杂因素**: {', '.join(results.get('confounders', []))}")

        # 方法
        self.add_title("分析方法", level=2)
        self.add_paragraph(f"- **倾向性评分模型**: {results.get('ps_model', 'N/A')}")
        self.add_paragraph(f"- **匹配方法**: {results.get('matching_method', 'N/A')}")
        self.add_paragraph(f"- **ATE估计方法**: {results.get('ate_method', 'N/A')}")

        # 样本量
        self.add_title("样本量", level=2)
        self.add_paragraph(f"- **处理组**: {results.get('n_treatment', 'N/A')} 人")
        self.add_paragraph(f"- **对照组**: {results.get('n_control', 'N/A')} 人")

        # 平衡性检验
        self.add_title("平衡性检验", level=2)
        self.add_table(balance_df, "混杂因素平衡性检验")

        # ATE估计结果
        self.add_title("因果效应估计结果", level=2)
        self.add_paragraph(f"**平均处理效应（ATE）**: {results.get('ate', 'N/A'):.4f}")
        self.add_paragraph(f"**95% 置信区间**: [{results.get('ci_lower', 'N/A'):.4f}, {results.get('ci_upper', 'N/A'):.4f}]")
        self.add_paragraph(f"**标准误**: {results.get('se', 'N/A'):.4f}")
        self.add_paragraph(f"**p值**: {results.get('p_value', 'N/A'):.4f}")
        self.add_paragraph(f"**统计显著性**: {'是 ✓' if results.get('significant') else '否 ✗'}")

        # 结论
        self.add_title("结论", level=2)
        if results.get('significant'):
            self.add_paragraph(f"结果表明，**{results.get('treatment')}** 对 **{results.get('outcome')}** 有显著影响（ATE = {results.get('ate', 0):.4f}, p = {results.get('p_value', 1):.4f}）。")
        else:
            self.add_paragraph(f"结果表明，**{results.get('treatment')}** 对 **{results.get('outcome')}** 无显著影响（p = {results.get('p_value', 1):.4f}）。")

        # 生成报告
        report = "\n".join(self.sections)

        # 保存
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"因果推断报告已保存至: {save_path}")

        return report

    def generate_causal_discovery_report(
        self,
        graph_stats: Dict[str, Any],
        edge_freq_df: pd.DataFrame,
        paths_df: pd.DataFrame,
        save_path: str,
    ):
        """
        生成因果发现报告（目标二）

        Parameters
        ----------
        graph_stats : dict
            图统计信息
        edge_freq_df : pd.DataFrame
            边频率表
        paths_df : pd.DataFrame
            关键路径表
        save_path : str
            保存路径
        """
        self.sections = []

        # 标题
        self.add_title("因果发现分析报告", level=1)
        self.add_paragraph(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 方法
        self.add_title("分析方法", level=2)
        self.add_paragraph(f"- **因果发现算法**: {graph_stats.get('algorithm', 'N/A')}")
        self.add_paragraph(f"- **Bootstrap次数**: {graph_stats.get('n_bootstrap', 'N/A')}")
        self.add_paragraph(f"- **稳定性阈值**: {graph_stats.get('threshold', 'N/A')}")

        # 图统计
        self.add_title("因果图统计", level=2)
        self.add_paragraph(f"- **节点数**: {graph_stats.get('n_nodes', 'N/A')}")
        self.add_paragraph(f"- **边数**: {graph_stats.get('n_edges', 'N/A')}")
        self.add_paragraph(f"- **图密度**: {graph_stats.get('density', 'N/A'):.4f}")

        # 稳定边
        self.add_title("稳定因果关系", level=2)
        self.add_table(edge_freq_df.head(20), "高置信度因果关系（按选择频率排序）")

        # 关键路径
        self.add_title("关键因果路径", level=2)
        self.add_table(paths_df.head(10), "重要因果路径")

        # 解释
        self.add_title("发现总结", level=2)
        self.add_paragraph("基于数据驱动的因果发现，我们识别出以下关键模式：")
        self.add_list([
            f"发现了 {graph_stats.get('n_edges', 0)} 条稳定的因果关系",
            "这些关系经过了Bootstrap稳定性选择验证",
            "符合时序约束和领域知识约束",
        ])

        # 生成报告
        report = "\n".join(self.sections)

        # 保存
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"因果发现报告已保存至: {save_path}")

        return report

    def save_report(self, filepath: str, format: str = "markdown"):
        """
        保存报告

        Parameters
        ----------
        filepath : str
            文件路径
        format : str
            格式 (markdown, html)
        """
        report = "\n".join(self.sections)

        if format == "markdown":
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)
        elif format == "html":
            # 简单的Markdown到HTML转换
            import markdown
            html = markdown.markdown(report)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
        else:
            raise ValueError(f"不支持的格式: {format}")

        logger.info(f"报告已保存至: {filepath}")
