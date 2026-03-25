"""
实验流程公共函数
提供配置加载、实验信息打印等通用功能
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from utils.config_loader import load_config
from utils.path_manager import PathManager

# 统一边框宽度常量
DEFAULT_BORDER_WIDTH = 60


def get_default_config():
    """
    返回空配置字典

    当脚本不需要预定义配置时使用
    """
    return {}


def parse_config_arg(argv, default: str = "config/base.yaml") -> str:
    """
    解析命令行配置参数

    Parameters
    ----------
    argv : list
        sys.argv 参数列表
    default : str
        默认配置文件路径

    Returns
    -------
    str
        配置文件路径
    """
    return argv[1] if len(argv) > 1 else default


def load_experiment_config(config_path: str) -> Tuple[Dict, Optional[PathManager]]:
    """
    加载实验配置

    Parameters
    ----------
    config_path : str
        配置文件路径

    Returns
    -------
    config : Dict
        配置字典
    path_mgr : PathManager or None
        路径管理器，如果配置加载失败则为None
    """
    try:
        config = load_config(config_path)
        path_mgr = PathManager(config)
        return config, path_mgr
    except Exception as e:
        print(f"警告：无法加载配置文件 {config_path}: {e}")
        print("使用默认配置")
        # 返回空配置，让各个脚本自己决定默认值
        return {}, None


def print_experiment_header(config: Dict, path_mgr: PathManager, border_width: int = DEFAULT_BORDER_WIDTH):
    """
    打印实验标题信息

    Parameters
    ----------
    config : Dict
        配置字典
    path_mgr : PathManager
        路径管理器
    border_width : int
        边框宽度（默认60）
    """
    print("\n" + "=" * border_width)
    print(f"实验名称: {config['experiment']['name']}")
    print(f"实验描述: {config['experiment']['description']}")
    print(f"输出目录: {path_mgr.output_dir}")
    print("=" * border_width + "\n")


def print_phase_header(title: str, show_time: bool = True, border_width: int = DEFAULT_BORDER_WIDTH):
    """
    打印流程标题

    Parameters
    ----------
    title : str
        流程标题
    show_time : bool
        是否显示开始时间
    border_width : int
        边框宽度（默认60）
    """
    print("\n" + "=" * border_width)
    print(title)
    print("=" * border_width)

    if show_time:
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def print_phase_footer(title: str, duration: float, border_width: int = DEFAULT_BORDER_WIDTH):
    """
    打印流程结束信息

    Parameters
    ----------
    title : str
        流程标题
    duration : float
        运行时长（秒）
    border_width : int
        边框宽度（默认60）
    """
    print("\n" + "=" * border_width)
    print(f"[OK] {title}完成!")
    print("=" * border_width)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")


def initialize_experiment(
    config_path: str,
    print_header: bool = True,
    config_required: bool = False
) -> Tuple[Dict, Optional[PathManager]]:
    """
    完整的实验初始化流程

    Parameters
    ----------
    config_path : str
        配置文件路径
    print_header : bool
        是否打印实验标题
    config_required : bool
        如果为True且配置加载失败则报错；为False时返回空配置（默认False）

    Returns
    -------
    config : Dict
        配置字典
    path_mgr : PathManager or None
        路径管理器
    """
    config, path_mgr = load_experiment_config(config_path)

    if config_required and path_mgr is None:
        raise FileNotFoundError(f"无法加载配置文件: {config_path}")

    if path_mgr is not None and print_header:
        print_experiment_header(config, path_mgr)

    return config, path_mgr
