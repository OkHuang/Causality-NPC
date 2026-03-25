"""
报告生成模块

生成推荐结果的Markdown报告
"""

from typing import Dict, List


class RecommendationReporter:
    """
    推荐报告生成器
    """

    def generate(self, results: List[Dict]) -> str:
        """
        生成Markdown格式报告

        Parameters
        ----------
        results : List[Dict]
            推荐结果列表

        Returns
        -------
        str
            Markdown报告
        """
        lines = []

        # 标题
        lines.append("# 因果推荐报告\n")
        lines.append("---\n")

        # 概览
        lines.append("## 概览\n")
        lines.append(f"- **总案例数**: {len(results)}\n")

        # 统计信息
        total_recommended = sum(len(r.get('recommended', {})) for r in results)
        total_not_recommended = sum(len(r.get('not_recommended', {})) for r in results)
        total_neutral = sum(len(r.get('neutral', [])) for r in results)

        lines.append("### 推荐统计\n")
        lines.append(f"| 统计项 | 数量 |")
        lines.append(f"|--------|------|")
        lines.append(f"| 总推荐药物 | {total_recommended} |")
        lines.append(f"| 总不推荐药物 | {total_not_recommended} |")
        lines.append(f"| 总中性药物 | {total_neutral} |")
        lines.append(f"| 平均每案例推荐 | {total_recommended / len(results):.1f} 味 |\n")

        # 案例详情
        lines.append("## 案例详情\n")

        for i, result in enumerate(results, 1):
            lines.append(f"### 案例 {i}\n")

            # 患者信息
            patient_info = result.get('patient_info', {})
            if patient_info:
                lines.append("**患者信息**:\n")
                for key, value in patient_info.items():
                    lines.append(f"- {key}: {value}")
                lines.append("")

            # 映射信息
            mapping_result = result.get('mapping_result', {})
            mapped_vars = mapping_result.get('mapped_vars', {})
            invalid_vars = mapping_result.get('invalid_vars', [])
            unmapped_keys = mapping_result.get('unmapped_keys', [])

            if mapped_vars or invalid_vars or unmapped_keys:
                lines.append("**映射信息**:\n")
                if mapped_vars:
                    lines.append(f"- 成功映射: {len(mapped_vars)} 个变量")
                if invalid_vars:
                    lines.append(f"- 无效变量: {', '.join(invalid_vars)}")
                if unmapped_keys:
                    lines.append(f"- 未映射键: {', '.join(k for k, _ in unmapped_keys)}")
                lines.append("")

            # 推荐药物
            recommended = result.get('recommended', {})
            if recommended:
                lines.append(f"**推荐药物** ({len(recommended)}味):\n")
                for med, score in list(recommended.items())[:10]:
                    lines.append(f"- {med}: {score:.4f}")
                if len(recommended) > 10:
                    lines.append(f"- ... 还有 {len(recommended) - 10} 味")
                lines.append("")

            # 不推荐药物
            not_recommended = result.get('not_recommended', {})
            if not_recommended:
                lines.append(f"**不推荐药物** ({len(not_recommended)}味):\n")
                for med, score in list(not_recommended.items())[:5]:
                    lines.append(f"- {med}: {score:.4f}")
                if len(not_recommended) > 5:
                    lines.append(f"- ... 还有 {len(not_recommended) - 5} 味")
                lines.append("")

            # 中性药物
            neutral = result.get('neutral', [])
            if neutral:
                lines.append(f"**中性药物**: {len(neutral)}味\n")
                lines.append("")

            # 路径解释（仅显示推荐的药物）
            explanations = result.get('explanations', {})
            if explanations and recommended:
                lines.append("**推荐理由** (前3味药):\n")
                for med in list(recommended.keys())[:3]:
                    if med in explanations and explanations[med]:
                        lines.append(f"\n{med}:")
                        for j, path_info in enumerate(explanations[med][:2], 1):
                            path_str = ' -> '.join(path_info['path'])
                            contribution = path_info['contribution']
                            lines.append(f"  {j}. {path_str}")
                            lines.append(f"     贡献: {contribution:.4f}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)
