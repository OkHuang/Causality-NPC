"""
结果保存模块

保存因果发现的结果文件
"""

import networkx as nx
import pickle
import json
from pathlib import Path
from typing import Dict, List


class ResultSaver:
    """结果保存器"""

    def __init__(self, output_dir: Path):
        """
        初始化结果保存器

        Parameters
        ----------
        output_dir : Path
            输出目录
        """
        self.output_dir = Path(output_dir)
        self.graph_dir = self.output_dir / 'graph'
        self.data_dir = self.output_dir / 'data'

        # 创建子目录
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_graph(self, G: nx.DiGraph) -> Dict[str, Path]:
        """
        保存图结构

        Parameters
        ----------
        G : nx.DiGraph
            因果图

        Returns
        -------
        Dict[str, Path]
            保存的文件路径
        """
        files = {}

        # 保存为pickle格式（NetworkX Graph对象）
        dag_path = self.graph_dir / 'dag.pkl'
        with open(dag_path, 'wb') as f:
            pickle.dump(G, f)
        files['dag'] = dag_path
        print(f"图结构已保存: {dag_path}")

        # 保存边列表为JSON格式（便于人工检查）
        edge_list = [(u, v) for u, v in G.edges()]
        edge_path = self.graph_dir / 'edges.json'
        with open(edge_path, 'w', encoding='utf-8') as f:
            json.dump(edge_list, f, ensure_ascii=False, indent=2)
        files['edges'] = edge_path
        print(f"边列表已保存: {edge_path}")
        print(f"边总数: {len(edge_list)}")

        return files

    def save_visualization(self, fig, filename: str = 'dag.png') -> Path:
        """
        保存可视化图像

        Parameters
        ----------
        fig : plt.Figure
            图形对象
        filename : str
            文件名

        Returns
        -------
        Path
            保存路径
        """
        save_path = self.graph_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化已保存: {save_path}")
        return save_path

    def save_data(self, df, filename: str) -> Path:
        """
        保存数据文件

        Parameters
        ----------
        df : pd.DataFrame
            数据框
        filename : str
            文件名

        Returns
        -------
        Path
            保存路径
        """
        save_path = self.data_dir / filename
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"数据已保存: {save_path}")
        return save_path

    def save_report(self, report: str, filename: str = 'report.md') -> Path:
        """
        保存报告

        Parameters
        ----------
        report : str
            报告内容（Markdown格式）
        filename : str
            文件名

        Returns
        -------
        Path
            保存路径
        """
        save_path = self.output_dir / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存: {save_path}")
        return save_path
