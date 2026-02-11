"""
数据加载器

功能：
- 从文件加载原始数据（CSV, JSON, Excel等）
- 数据格式验证
- 基本统计信息生成
"""

import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据加载器

        Parameters
        ----------
        config : dict, optional
            配置字典
        """
        self.config = config or {}
        self.data = None
        self.metadata = {}

    def load(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        加载数据文件

        Parameters
        ----------
        filepath : str or Path
            数据文件路径
        **kwargs : dict
            传递给pandas读取函数的参数

        Returns
        -------
        pd.DataFrame
            加载的数据
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 根据文件扩展名选择读取方法
        suffix = filepath.suffix.lower()

        if suffix == ".csv":
            self.data = pd.read_csv(filepath, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            self.data = pd.read_excel(filepath, **kwargs)
        elif suffix == ".json":
            self.data = pd.read_json(filepath, **kwargs)
        elif suffix == ".parquet":
            self.data = pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

        logger.info(f"成功加载数据: {filepath}")
        logger.info(f"数据形状: {self.data.shape}")

        # 生成基本统计信息
        self._generate_metadata()

        return self.data

    def _generate_metadata(self):
        """生成数据元信息"""
        if self.data is None:
            return

        self.metadata = {
            "n_rows": len(self.data),
            "n_columns": len(self.data.columns),
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
        }

    def validate_schema(self, required_columns: list) -> bool:
        """
        验证数据是否包含必需的列

        Parameters
        ----------
        required_columns : list
            必需的列名列表

        Returns
        -------
        bool
            是否包含所有必需列
        """
        missing = set(required_columns) - set(self.data.columns)
        if missing:
            logger.error(f"缺少必需的列: {missing}")
            return False

        logger.info("数据验证通过")
        return True

    def get_summary(self) -> Dict[str, Any]:
        """
        获取数据摘要信息

        Returns
        -------
        dict
            数据摘要
        """
        if self.data is None:
            return {}

        summary = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
            "missing": self.data.isnull().sum().to_dict(),
            "n_unique": self.data.nunique().to_dict(),
        }

        return summary

    def save(self, filepath: Union[str, Path], format: str = "parquet"):
        """
        保存数据

        Parameters
        ----------
        filepath : str or Path
            保存路径
        format : str
            保存格式 (parquet, csv, json)
        """
        if self.data is None:
            raise ValueError("没有数据可保存")

        filepath = Path(filepath)

        if format == "parquet":
            self.data.to_parquet(filepath, index=False)
        elif format == "csv":
            self.data.to_csv(filepath, index=False)
        elif format == "json":
            self.data.to_json(filepath, orient="records", force_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")

        logger.info(f"数据已保存至: {filepath}")
