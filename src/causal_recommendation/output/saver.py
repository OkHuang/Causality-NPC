"""
结果保存模块

保存推荐结果到文件
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class RecommendationSaver:
    """
    推荐结果保存器
    """

    def __init__(self, output_dir: Path):
        """
        初始化保存器

        Parameters
        ----------
        output_dir : Path
            输出目录
        """
        self.output_dir = Path(output_dir)
        self.recommendations_dir = self.output_dir / 'recommendations'

        # 创建子目录
        self.recommendations_dir.mkdir(parents=True, exist_ok=True)

    def save_recommendations(self, results: List[Dict]) -> Path:
        """
        保存推荐结果为JSON

        Parameters
        ----------
        results : List[Dict]
            推荐结果列表

        Returns
        -------
        Path
            JSON文件路径
        """
        json_path = self.recommendations_dir / 'patient_recommendations.json'

        # 转换为可JSON序列化的格式
        serializable_results = []
        for i, result in enumerate(results, 1):
            serializable_result = {
                'case_id': i,
                'patient_info': result.get('patient_info', {}),
                'recommended': {
                    k: float(v) for k, v in result.get('recommended', {}).items()
                },
                'not_recommended': {
                    k: float(v) for k, v in result.get('not_recommended', {}).items()
                },
                'neutral': result.get('neutral', []),
                'explanations': result.get('explanations', {}),
                'mapping_info': {
                    'mapped_vars': result.get('mapping_result', {}).get('mapped_vars', {}),
                    'invalid_vars': result.get('mapping_result', {}).get('invalid_vars', []),
                    'unmapped_keys': [
                        {'key': k, 'value': v}
                        for k, v in result.get('mapping_result', {}).get('unmapped_keys', [])
                    ]
                }
            }
            serializable_results.append(serializable_result)

        # 保存到文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_cases': len(results),
                'cases': serializable_results
            }, f, ensure_ascii=False, indent=2)

        print(f"推荐结果已保存: {json_path}")
        print(f"  案例数: {len(results)}")

        return json_path

    def print_summary(self, results: List[Dict]):
        """
        打印结果摘要

        Parameters
        ----------
        results : List[Dict]
            推荐结果列表
        """
        print(f"\n=== 推荐结果摘要 ===")
        print(f"总案例数: {len(results)}")

        for i, result in enumerate(results, 1):
            recommended = result.get('recommended', {})
            not_recommended = result.get('not_recommended', {})
            neutral = result.get('neutral', [])

            print(f"\n案例 {i}:")
            print(f"  推荐: {len(recommended)} 味")
            print(f"  不推荐: {len(not_recommended)} 味")
            print(f"  中性: {len(neutral)} 味")

            if recommended:
                print(f"  推荐药物: {', '.join(list(recommended.keys())[:5])}")
