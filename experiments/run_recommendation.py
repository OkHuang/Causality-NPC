"""
运行因果推荐

使用方法:
    python experiments/run_recommendation.py --config config/base.yaml
    python experiments/run_recommendation.py --config config/experiments/exp_freq_0.15.yaml
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.causal_discovery.config import NPCConfig
from src.causal_recommendation.pipeline import run_causal_recommendation


def load_example_patients():
    """
    加载示例患者数据

    Returns
    -------
    list
        患者信息列表
    """
    patients = [
        # 案例1: 乏力畏寒的老年女性患者
        {
            'gender': '女',
            'age': 58,
            '乏力': 1.0,
            '畏寒': 1.0,
            '头痛': 0.0,
            '头晕': 1.0,
            '咽干': 0.0,
            '鼻塞': 0.0,
            '流涕': 0.0,
            '阴虚血瘀证': 1.0,
            '颃颡岩': 1.0,
        },

        # 案例2: 鼻塞流涕的男性患者
        {
            'gender': '男',
            'age': 45,
            '乏力': 0.0,
            '畏寒': 0.0,
            '头痛': 1.0,
            '头晕': 0.0,
            '咽干': 1.0,
            '鼻塞': 2.0,
            '流涕': 2.0,
            '耳鸣': 1.0,
            '痰热内扰证': 1.0,
        },

        # 案例3: 失眠易醒的中年女性患者
        {
            'gender': '女',
            'age': 52,
            '乏力': 1.0,
            '畏寒': 0.0,
            '头痛': 0.0,
            '头晕': 1.0,
            '咽干': 0.0,
            '失眠': 2.0,
            '易醒': 2.0,
            '胃胀': 1.0,
            '精神': 1.0,
            '气血亏虚证': 1.0,
        },

        # 案例4: 耳鸣听力下降的老年患者
        {
            'gender': '男',
            'age': 65,
            '耳鸣': 2.0,
            '听力下降': 2.0,
            '耳胀闷': 1.0,
            '颈痛': 1.0,
            '脾肾亏虚证': 1.0,
        },
    ]

    return patients


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='运行因果推荐')
    parser.add_argument('--config', default='config/base.yaml', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置并运行
    print(f"加载配置: {args.config}")
    config = NPCConfig.from_yaml(args.config)

    # 加载患者数据
    patients = load_example_patients()
    print(f"加载了 {len(patients)} 个示例患者\n")

    # 运行因果推荐
    result = run_causal_recommendation(config, patients)

    print(f"\n完成! 处理了 {len(result['results'])} 个患者的推荐")
