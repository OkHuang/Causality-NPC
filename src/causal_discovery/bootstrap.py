"""
稳定性选择模块

功能：
- Bootstrap重采样
- 边选择频率统计
- 稳定性筛选
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StabilitySelector:
    """稳定性选择器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化稳定性选择器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.n_bootstrap = self.config.get("n_bootstrap", 1000)
        self.sampling_ratio = self.config.get("sampling_ratio", 0.8)
        self.threshold = self.config.get("selection_threshold", 0.85)
        self.n_jobs = self.config.get("n_jobs", -1)

        self.edge_frequencies = {}
        self.stable_edges = []

    def select(
        self,
        data: pd.DataFrame,
        discovery_algorithm,
        algorithm_kwargs: Dict[str, Any] = None,
    ) -> List[Tuple[str, str]]:
        """
        执行稳定性选择

        Parameters
        ----------
        data : pd.DataFrame
            原始数据
        discovery_algorithm : callable
            因果发现算法函数
        algorithm_kwargs : dict, optional
            传递给算法的参数

        Returns
        -------
        list of tuple
            稳定边列表
        """
        logger.info(f"开始稳定性选择（Bootstrap {self.n_bootstrap} 次）...")

        algorithm_kwargs = algorithm_kwargs or {}

        # 并行运行Bootstrap
        edges_list = self._run_bootstrap(
            data, discovery_algorithm, algorithm_kwargs
        )

        # 统计边频率
        self._count_edge_frequencies(edges_list)

        # 筛选稳定边
        self.stable_edges = self._filter_stable_edges()

        logger.info(f"发现 {len(self.edge_frequencies)} 条不同的边")
        logger.info(f"其中 {len(self.stable_edges)} 条稳定边（频率 > {self.threshold}）")

        return self.stable_edges

    def _run_bootstrap(
        self,
        data: pd.DataFrame,
        discovery_algorithm,
        algorithm_kwargs: Dict[str, Any],
    ) -> List[List[Tuple[str, str]]]:
        """
        运行Bootstrap重采样
        """
        n_samples = int(len(data) * self.sampling_ratio)
        edges_list = []

        # 并行执行
        if self.n_jobs == -1:
            n_jobs = None  # 使用所有CPU
        else:
            n_jobs = self.n_jobs

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []

            for i in range(self.n_bootstrap):
                # 重采样
                sample = data.sample(n=n_samples, replace=True)

                # 提交任务
                future = executor.submit(
                    self._run_single_discovery,
                    sample, discovery_algorithm, algorithm_kwargs
                )
                futures.append(future)

            # 收集结果
            for future in tqdm(as_completed(futures), total=self.n_bootstrap):
                edges = future.result()
                edges_list.append(edges)

        return edges_list

    @staticmethod
    def _run_single_discovery(
        sample: pd.DataFrame,
        discovery_algorithm,
        algorithm_kwargs: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        """
        运行单次因果发现
        """
        try:
            graph = discovery_algorithm(sample, **algorithm_kwargs)

            # 提取边
            edges = list(graph.edges())

            return edges

        except Exception as e:
            logger.warning(f"单次发现失败: {e}")
            return []

    def _count_edge_frequencies(self, edges_list: List[List[Tuple[str, str]]]):
        """
        统计边出现频率（保留方向）
        """
        edge_counts = {}

        for edges in edges_list:
            for edge in edges:
                # 保留边的原始方向，不进行排序
                edge_key = edge  # 直接使用原始边 (source, target)

                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

        # 计算频率
        for edge, count in edge_counts.items():
            self.edge_frequencies[edge] = count / self.n_bootstrap

    def _filter_stable_edges(self) -> List[Tuple[str, str]]:
        """
        筛选高频边
        """
        stable_edges = [
            edge for edge, freq in self.edge_frequencies.items()
            if freq >= self.threshold
        ]

        # 按频率排序
        stable_edges.sort(
            key=lambda e: self.edge_frequencies[e],
            reverse=True
        )

        return stable_edges

    def get_edge_frequencies(self) -> Dict[Tuple[str, str], float]:
        """
        获取所有边的频率
        """
        return self.edge_frequencies

    def get_frequency_table(self) -> pd.DataFrame:
        """
        获取边频率表
        """
        df = pd.DataFrame([
            {
                "source": edge[0],
                "target": edge[1],
                "frequency": freq,
                "stable": freq >= self.threshold,
            }
            for edge, freq in self.edge_frequencies.items()
        ])

        df = df.sort_values("frequency", ascending=False)

        return df

    def plot_frequencies(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 6),
        save_path: str = None,
    ):
        """
        绘制边频率分布

        Parameters
        ----------
        top_n : int
            显示前N条边
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        import matplotlib.pyplot as plt

        # 获取频率表
        df = self.get_frequency_table().head(top_n)

        # 绘图
        fig, ax = plt.subplots(figsize=figsize)

        # 边标签
        edge_labels = [f"{row['source']} -> {row['target']}" for _, row in df.iterrows()]

        # 颜色：稳定为绿色，不稳定为红色
        colors = ['green' if stable else 'red' for stable in df['stable']]

        ax.barh(range(len(df)), df['frequency'], color=colors, alpha=0.7)

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(edge_labels)
        ax.set_xlabel('选择频率')
        ax.set_title(f'边选择频率分布（Top {top_n}）')
        ax.axvline(x=self.threshold, color='red', linestyle='--', label=f'阈值 ({self.threshold})')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"频率图已保存至: {save_path}")

        plt.show()
