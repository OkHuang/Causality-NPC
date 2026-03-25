"""
报告生成模块

生成因果效应估计报告
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime


class EffectReporter:
    """因果效应估计报告生成器"""

    def generate(self, results: pd.DataFrame, stats: Dict) -> str:
        """
        生成Markdown报告

        Parameters
        ----------
        results : pd.DataFrame
            估计结果DataFrame
        stats : Dict
            统计信息

        Returns
        -------
        str
            Markdown格式的报告
        """
        lines = []

        # 标题
        lines.append("# 因果效应估计报告")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 1. 估计概览
        lines.append("## 1. 估计概览")
        lines.append(f"- 总估计边数: {stats.get('total_edges', 0)}")
        lines.append(f"- 成功估计: {stats.get('successful', 0)}")
        lines.append(f"- 失败估计: {stats.get('failed', 0)}")
        lines.append("")

        # 2. 效应分布
        if len(results) > 0:
            lines.append("## 2. 效应分布")
            lines.append(f"- 平均ATE: {results['ate'].mean():.4f}")
            lines.append(f"- 标准差: {results['ate'].std():.4f}")
            lines.append(f"- 最小值: {results['ate'].min():.4f}")
            lines.append(f"- 最大值: {results['ate'].max():.4f}")
            lines.append("")

            # 正效应Top10
            positive = results[results['ate'] > 0].nlargest(10, 'ate')
            if len(positive) > 0:
                lines.append("### 2.1 Top 正效应 (前10)")
                lines.append("")
                lines.append("| 源变量 | 目标变量 | ATE | 相关系数 |")
                lines.append("|--------|----------|-----|----------|")
                for _, row in positive.iterrows():
                    lines.append(f"| {row['source']} | {row['target']} | {row['ate']:.4f} | {row['correlation']:.4f} |")
                lines.append("")

            # 负效应Top10
            negative = results[results['ate'] < 0].nsmallest(10, 'ate')
            if len(negative) > 0:
                lines.append("### 2.2 Top 负效应 (前10)")
                lines.append("")
                lines.append("| 源变量 | 目标变量 | ATE | 相关系数 |")
                lines.append("|--------|----------|-----|----------|")
                for _, row in negative.iterrows():
                    lines.append(f"| {row['source']} | {row['target']} | {row['ate']:.4f} | {row['correlation']:.4f} |")
                lines.append("")

        # 3. 详细结果
        if len(results) > 0:
            lines.append("## 3. 详细结果")
            lines.append("")
            # 只显示关键列
            display_cols = ['source', 'target', 'ate', 'ci_lower', 'ci_upper']
            available_cols = [c for c in display_cols if c in results.columns]
            lines.append(results[available_cols].to_string(index=False))
            lines.append("")

        # 4. 失败列表
        if stats.get('failed', 0) > 0:
            lines.append("## 4. 失败的估计")
            lines.append("")
            for item in stats.get('failed_list', []):
                lines.append(f"- {item['source']} -> {item['target']}: {item['error']}")
            lines.append("")

        return "\n".join(lines)
