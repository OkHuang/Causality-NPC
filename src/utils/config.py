"""
配置管理模块

功能：
- 加载YAML配置文件
- 合并多个配置
- 配置验证
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(
    config_path: str,
    base_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    加载配置文件

    Parameters
    ----------
    config_path : str
        配置文件路径
    base_config_path : str, optional
        基础配置文件路径（会合并）

    Returns
    -------
    dict
        配置字典
    """
    config = {}

    # 先加载基础配置
    if base_config_path:
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
            config.update(base_config)
            logger.info(f"已加载基础配置: {base_config_path}")

    # 加载主配置（覆盖基础配置）
    with open(config_path, "r", encoding="utf-8") as f:
        main_config = yaml.safe_load(f)
        config.update(main_config)
        logger.info(f"已加载主配置: {config_path}")

    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典（后面的覆盖前面的）

    Parameters
    ----------
    *configs : dict
        多个配置字典

    Returns
    -------
    dict
        合并后的配置
    """
    merged = {}

    for config in configs:
        merged.update(config)

    return merged


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到文件

    Parameters
    ----------
    config : dict
        配置字典
    save_path : str
        保存路径
    """
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    logger.info(f"配置已保存至: {save_path}")


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    验证配置是否包含必需的键

    Parameters
    ----------
    config : dict
        配置字典
    required_keys : list
        必需的键列表（支持嵌套，如 "model.name"）

    Returns
    -------
    bool
        是否有效
    """
    for key in required_keys:
        keys = key.split(".")
        value = config

        try:
            for k in keys:
                value = value[k]
        except (KeyError, TypeError):
            logger.error(f"配置缺少必需的键: {key}")
            return False

    logger.info("配置验证通过")
    return True
