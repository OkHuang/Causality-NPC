"""
配置加载器
支持配置继承和覆盖
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件（支持继承）

    Parameters
    ----------
    config_path : str
        配置文件路径

    Returns
    -------
    Dict
        合并后的完整配置
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 处理继承
    if 'extends' in config:
        base_path = config_path.parent / config['extends']
        base_config = load_config(str(base_path))

        # 递归合并配置
        config = _deep_merge(base_config, config)

    # 添加实验信息（如果没有）
    if 'experiment' not in config:
        config['experiment'] = {
            'name': 'default',
            'description': '默认配置',
            'tags': ['default']
        }

    # 设置输出路径
    exp_name = config['experiment']['name']
    config['paths']['output_dir'] = f"{config['paths']['outputs_root']}/{exp_name}"

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典

    Parameters
    ----------
    base : Dict
        基础配置
    override : Dict
        覆盖配置

    Returns
    -------
    Dict
        合并后的配置
    """
    result = base.copy()

    for key, value in override.items():
        if key == 'extends':
            # 跳过继承标记
            continue
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并子字典
            result[key] = _deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = value

    return result


def list_experiments() -> Dict[str, Dict]:
    """
    列出所有可用的实验配置

    Returns
    -------
    Dict
        实验信息字典
    """
    experiments_dir = Path("config/experiments")

    if not experiments_dir.exists():
        return {}

    experiments = {}
    for exp_file in experiments_dir.glob("*.yaml"):
        try:
            config = load_config(str(exp_file))
            exp_name = config['experiment']['name']
            experiments[exp_name] = {
                'file': str(exp_file),
                'description': config['experiment'].get('description', ''),
                'tags': config['experiment'].get('tags', [])
            }
        except Exception as e:
            print(f"警告：无法加载实验配置 {exp_file}: {e}")

    return experiments


if __name__ == "__main__":
    # 测试：列出所有实验
    experiments = list_experiments()
    print("可用的实验配置：")
    for name, info in experiments.items():
        print(f"  - {name}: {info['description']}")
