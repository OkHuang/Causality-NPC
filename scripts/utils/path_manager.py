"""
路径管理器
根据实验名称自动组织输出路径
"""

import os
from pathlib import Path
from typing import Dict


class PathManager:
    """路径管理器"""

    def __init__(self, config: Dict):
        """
        初始化路径管理器

        Parameters
        ----------
        config : Dict
            配置字典
        """
        self.config = config
        self.exp_name = config['experiment']['name']
        self.output_root = Path(config['paths']['outputs_root'])
        self.output_dir = self.output_root / self.exp_name

        # 创建输出目录结构
        self._create_directories()

    def _create_directories(self):
        """创建输出目录结构"""
        directories = [
            self.output_dir / 'data',
            self.output_dir / 'graphs',
            self.output_dir / 'causal_effects',
            self.output_dir / 'models',
            self.output_dir / 'reports',
            self.output_dir / 'logs'
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_data_dir(self) -> Path:
        """获取数据输出目录"""
        return self.output_dir / 'data'

    def get_graphs_dir(self) -> Path:
        """获取图输出目录"""
        return self.output_dir / 'graphs'

    def get_effects_dir(self) -> Path:
        """获取效应输出目录"""
        return self.output_dir / 'causal_effects'

    def get_models_dir(self) -> Path:
        """获取模型输出目录"""
        return self.output_dir / 'models'

    def get_reports_dir(self) -> Path:
        """获取报告输出目录"""
        return self.output_dir / 'reports'

    def get_logs_dir(self) -> Path:
        """获取日志输出目录"""
        return self.output_dir / 'logs'

    def get_raw_data_path(self) -> str:
        """获取原始数据路径"""
        return self.config['paths']['raw_data']
