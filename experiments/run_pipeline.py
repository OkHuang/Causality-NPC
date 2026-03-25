"""
因果推断完整管道

一键运行从因果发现到评估的完整流程：
1. 因果发现 (Causal Discovery)
2. 因果效应估计 (Causal Effect Estimation)
3. 因果推荐 (Causal Recommendation)
4. 推荐评估 (Recommendation Evaluation)

使用方法:
    # 运行完整管道
    python experiments/run_pipeline.py --config config/experiments/exp_freq_0.20.yaml

    # 跳过评估步骤
    python experiments/run_pipeline.py --config config/experiments/exp_freq_0.20.yaml --skip-evaluation

    # 指定评估患者数
    python experiments/run_pipeline.py --config config/experiments/exp_freq_0.20.yaml --max-patients 100
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.causal_discovery.config import NPCConfig
from src.causal_discovery.pipeline import run_causal_discovery
from src.causal_effect.pipeline import run_causal_effect
from src.causal_recommendation.pipeline import run_causal_recommendation
from src.causal_recommendation.evaluation.evaluator import RecommendationEvaluator
from src.causal_recommendation.data.patient_encoder import extract_mapping_rules
from src.causal_recommendation.data.loader import RecommendationLoader


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


def load_real_patients(config: NPCConfig, max_patients = None):
    """
    从processed_data.csv加载真实患者数据

    Parameters
    ----------
    config : NPCConfig
        配置对象
    max_patients : int, optional
        最大加载患者数

    Returns
    -------
    list
        患者信息列表
    """
    import pandas as pd
    import networkx as nx

    # 加载患者数据
    patient_data_path = "Data/processed/processed_data.csv"
    print(f"加载真实患者数据: {patient_data_path}")
    df_patients = pd.read_csv(patient_data_path)
    print(f"  总患者记录数: {len(df_patients)}")

    # 加载图节点
    loader = RecommendationLoader(config)
    graph, _ = loader.load_all()
    all_nodes = set(graph.nodes())

    # 提取映射规则
    mapping_rules = extract_mapping_rules(all_nodes)

    # 构建患者信息列表
    patients = []
    limit = min(max_patients, len(df_patients)) if max_patients else len(df_patients)

    for idx in range(limit):
        patient_row = df_patients.iloc[idx]
        patient_info = {}

        # 性别
        if 'gender' in patient_row.index:
            gender = patient_row['gender']
            if pd.notna(gender):
                patient_info['gender'] = '女' if str(gender) == '女' else '男'

        # 年龄
        if 'age' in patient_row.index and pd.notna(patient_row['age']):
            patient_info['age'] = int(patient_row['age'])

        # 遍历所有列，识别症状和诊断
        for col in patient_row.index:
            if col.startswith(('patient_id', 'checkup_id', 'time', 'gender', 'age')):
                continue

            value = patient_row[col]
            if pd.isna(value) or value == 0:
                continue

            # 症状
            if '证' not in col and not col.startswith('diagnosis_') and not col.startswith('med_'):
                node_name = f"{col}_t"
                if node_name in all_nodes:
                    patient_info[col] = float(value)

            # 诊断
            elif '证' in col:
                node_name = f"diagnosis_{col}_t"
                if node_name in all_nodes:
                    patient_info[col] = float(value)

        patients.append(patient_info)

    print(f"  加载患者数: {len(patients)}")
    return patients


def run_pipeline(
    config: NPCConfig,
    patients: list = None,
    skip_evaluation: bool = False,
    max_eval_patients = None
):
    """
    运行完整的因果推断管道

    Parameters
    ----------
    config : NPCConfig
        配置对象
    patients : list, optional
        用于推荐的患者列表
    skip_evaluation : bool
        是否跳过评估步骤
    max_eval_patients : int, optional
        评估时使用的最大患者数（None表示全部患者）

    Returns
    -------
    dict
        管道运行结果
    """
    print("\n" + "="*80)
    print("因果推断完整管道")
    print("Causal Inference Pipeline")
    print("="*80)
    print(f"\n实验: {config.experiment_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    start_time = datetime.now()

    # ========== 步骤1: 因果发现 ==========
    print("\n" + "="*80)
    print("步骤1: 因果发现 (Causal Discovery)")
    print("="*80)

    try:
        discovery_df = run_causal_discovery(config)
        results['discovery'] = {
            'status': 'success',
            'num_edges': len(discovery_df)
        }
        print(f"[OK] 因果发现完成，发现 {len(discovery_df)} 条边")
    except Exception as e:
        results['discovery'] = {
            'status': 'failed',
            'error': str(e)
        }
        print(f"[ERROR] 因果发现失败: {e}")
        return results

    # ========== 步骤2: 因果效应估计 ==========
    print("\n" + "="*80)
    print("步骤2: 因果效应估计 (Causal Effect Estimation)")
    print("="*80)

    try:
        effect_df = run_causal_effect(config)
        results['effect'] = {
            'status': 'success',
            'num_edges': len(effect_df)
        }
        print(f"[OK] 因果效应估计完成，估计了 {len(effect_df)} 条边")
    except Exception as e:
        results['effect'] = {
            'status': 'failed',
            'error': str(e)
        }
        print(f"[ERROR] 因果效应估计失败: {e}")
        return results

    # ========== 步骤3: 因果推荐 ==========
    print("\n" + "="*80)
    print("步骤3: 因果推荐 (Causal Recommendation)")
    print("="*80)

    try:
        # 如果跳过评估，使用示例患者；否则使用真实患者数据
        if skip_evaluation:
            print("[INFO] 跳过评估，使用示例患者进行推荐验证")
            if patients is None:
                patients = load_example_patients()
        else:
            print("[INFO] 使用真实患者数据进行推荐")
            if patients is None:
                # 使用真实患者，但限制数量以控制输出
                patients = load_real_patients(config, max_patients=10)

        recommendation_result = run_causal_recommendation(config, patients)
        results['recommendation'] = {
            'status': 'success',
            'num_patients': len(recommendation_result['results'])
        }
        print(f"[OK] 因果推荐完成，处理了 {len(recommendation_result['results'])} 个患者")
    except Exception as e:
        results['recommendation'] = {
            'status': 'failed',
            'error': str(e)
        }
        print(f"[ERROR] 因果推荐失败: {e}")
        return results

    # ========== 步骤4: 推荐评估 ==========
    if not skip_evaluation:
        print("\n" + "="*80)
        print("步骤4: 推荐评估 (Recommendation Evaluation)")
        print("="*80)

        try:
            evaluator = RecommendationEvaluator(
                config=config,
                patient_data_path="Data/processed/processed_data.csv",
                medicine_data_path="Data/processed/processed_medicines.csv"
            )

            eval_result = evaluator.evaluate_batch(max_patients=max_eval_patients)
            evaluator.print_evaluation_report(eval_result)

            # 保存评估结果
            evaluator.save_results(eval_result)

            results['evaluation'] = {
                'status': 'success',
                'num_patients': eval_result['num_patients'],
                'metrics': eval_result['aggregated_metrics']
            }
            print(f"[OK] 推荐评估完成")
        except Exception as e:
            results['evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"[ERROR] 推荐评估失败: {e}")
    else:
        print("\n[INFO] 跳过评估步骤")
        results['evaluation'] = {'status': 'skipped'}

    # ========== 完成 ==========
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*80)
    print("管道运行完成")
    print("="*80)
    print(f"\n结果摘要:")
    print(f"  因果发现: {results['discovery']['status']}")
    if results['discovery']['status'] == 'success':
        print(f"    发现边数: {results['discovery']['num_edges']}")

    print(f"  因果效应: {results['effect']['status']}")
    if results['effect']['status'] == 'success':
        print(f"    估计边数: {results['effect']['num_edges']}")

    print(f"  因果推荐: {results['recommendation']['status']}")
    if results['recommendation']['status'] == 'success':
        print(f"    推荐患者数: {results['recommendation']['num_patients']}")

    print(f"  推荐评估: {results['evaluation']['status']}")
    if results['evaluation']['status'] == 'success':
        micro = results['evaluation']['metrics']['micro']
        print(f"    F1分数: {micro['f1']:.4f}")
        print(f"    准确率: {micro['accuracy']:.4f}")

    print(f"\n总耗时: {duration:.1f} 秒")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    results['duration'] = duration
    results['start_time'] = start_time.isoformat()
    results['end_time'] = end_time.isoformat()

    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='因果推断完整管道',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/base.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='跳过评估步骤'
    )

    parser.add_argument(
        '--max-patients',
        type=int,
        default=None,
        help='评估时使用的最大患者数（None表示全部患者）'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载配置
    print(f"加载配置: {args.config}")
    config = NPCConfig.from_yaml(args.config)

    # 加载示例患者
    patients = load_example_patients()

    # 运行管道
    results = run_pipeline(
        config=config,
        patients=patients,
        skip_evaluation=args.skip_evaluation,
        max_eval_patients=args.max_patients
    )

    # 检查是否有失败步骤
    failed_steps = [
        step for step, result in results.items()
        if isinstance(result, dict) and result.get('status') == 'failed'
    ]

    if failed_steps:
        print(f"\n[WARNING] 以下步骤失败: {', '.join(failed_steps)}")
        sys.exit(1)
    else:
        print(f"\n[OK] 所有步骤成功完成!")
        sys.exit(0)
