"""
运行因果效应估计

使用方法:
    python experiments/run_effect.py --config config/base.yaml
    python experiments/run_effect.py --config config/experiments/exp_freq_0.15.yaml
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.causal_discovery.config import NPCConfig
from src.causal_effect.pipeline import run_causal_effect


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='运行因果效应估计')
    parser.add_argument('--config', default='config/base.yaml', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置并运行
    print(f"加载配置: {args.config}")
    config = NPCConfig.from_yaml(args.config)

    # 运行因果效应估计
    results = run_causal_effect(config)

    print(f"\n完成！估计了 {len(results)} 条边的因果效应")
