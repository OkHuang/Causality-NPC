"""
结果保存模块

保存因果效应估计结果
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


class EffectSaver:
    """因果效应结果保存器"""

    def __init__(self, output_dir: Path):
        """
        初始化保存器

        Parameters
        ----------
        output_dir : Path
            输出目录
        """
        self.output_dir = Path(output_dir)
        self.estimates_dir = self.output_dir / 'estimates'
        self.models_dir = self.output_dir / 'models'

        # 创建子目录
        self.estimates_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_results_csv(self, results: List[Dict]) -> Path:
        """
        保存结果为CSV

        Parameters
        ----------
        results : List[Dict]
            估计结果列表

        Returns
        -------
        Path
            CSV文件路径
        """
        successful = [r for r in results if 'ate' in r]
        csv_data = [self._build_result_row(r) for r in successful]

        df = pd.DataFrame(csv_data)
        csv_path = self.estimates_dir / 'estimated_effects.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"CSV结果已保存: {csv_path}")
        print(f"  成功: {len(successful)} 条")

        return csv_path

    def save_models_pkl(self, results: List[Dict]) -> Path:
        """
        保存完整结果为PKL

        Parameters
        ----------
        results : List[Dict]
            估计结果列表

        Returns
        -------
        Path
            PKL文件路径
        """
        pkl_path = self.models_dir / 'causal_estimates.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"完整结果已保存: {pkl_path}")

        return pkl_path

    def _build_result_row(self, r: Dict) -> Dict[str, Any]:
        """
        构建CSV结果行

        Parameters
        ----------
        r : Dict
            单条估计结果

        Returns
        -------
        Dict[str, Any]
            CSV行数据
        """
        row = {
            'source': r['treatment'],
            'target': r['outcome'],
            'method': r['method'],
            'ate': r['ate'],
            'correlation': r.get('correlation', 'N/A'),
            'n_treated': r.get('n_treated', 'N/A'),
            'n_outcome': r.get('n_outcome', 'N/A'),
            'total_samples': r.get('total_samples', 'N/A')
        }

        # 置信区间
        row['ci_lower'] = r.get('ci_lower', 'N/A') if r.get('ci_lower') is not None else 'N/A'
        row['ci_upper'] = r.get('ci_upper', 'N/A') if r.get('ci_upper') is not None else 'N/A'

        # model_stats
        model_stats = r.get('model_stats')
        if model_stats:
            if isinstance(model_stats, dict) and model_stats.get('is_multiclass'):
                # 多分类（返回的是完整的结果字典）
                row.update({
                    'categories': str(model_stats['categories']),
                    'category_ates': str([f'{ate:.4f}' for ate in model_stats['category_ates']]),
                    'n_classes': len(model_stats['categories']),
                    'p_value': 'N/A', 'aic': 'N/A', 'bic': 'N/A',
                    'odds_ratio': 'N/A', 'or_ci_lower': 'N/A', 'or_ci_upper': 'N/A'
                })
            elif isinstance(model_stats, dict) and not model_stats.get('is_multiclass'):
                # 二分类（返回的是stats字典）
                row.update({
                    'categories': 'N/A', 'category_ates': 'N/A', 'n_classes': 'N/A',
                    'odds_ratio': model_stats.get('odds_ratio', 'N/A'),
                    'or_ci_lower': model_stats.get('or_ci_lower', 'N/A'),
                    'or_ci_upper': model_stats.get('or_ci_upper', 'N/A'),
                    'p_value': model_stats.get('p_value', 'N/A'),
                    'aic': model_stats.get('aic', 'N/A'),
                    'bic': model_stats.get('bic', 'N/A')
                })
            else:
                # 兜底处理（其他情况）
                row.update({
                    'categories': 'N/A', 'category_ates': 'N/A', 'n_classes': 'N/A',
                    'odds_ratio': 'N/A', 'or_ci_lower': 'N/A', 'or_ci_upper': 'N/A',
                    'p_value': 'N/A', 'aic': 'N/A', 'bic': 'N/A'
                })
        else:
            row.update({
                'categories': 'N/A', 'category_ates': 'N/A', 'n_classes': 'N/A',
                'odds_ratio': 'N/A', 'or_ci_lower': 'N/A', 'or_ci_upper': 'N/A',
                'p_value': 'N/A', 'aic': 'N/A', 'bic': 'N/A'
            })

        return row

    def print_summary(self, results: List[Dict]):
        """
        打印结果摘要

        Parameters
        ----------
        results : List[Dict]
            估计结果列表
        """
        successful = [r for r in results if 'ate' in r]
        failed = [r for r in results if 'error' in r]

        print(f"\n=== 结果摘要 ===")
        print(f"成功: {len(successful)} 条")
        print(f"失败: {len(failed)} 条")

        if successful:
            df = pd.DataFrame([self._build_result_row(r) for r in successful])
            print(f"\n前10条结果:")
            print(df.head(10).to_string(index=False))

        if failed:
            print(f"\n失败的估计:")
            for r in failed[:5]:
                print(f"  - {r['treatment']} -> {r['outcome']}: {r['error']}")
