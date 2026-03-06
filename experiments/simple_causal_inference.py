"""
简易版因果效应估计主脚本

整合所有模块，执行完整的因果效应估计流程：
1. 加载因果发现结果（图结构和数据）
2. 提取药物→症状因果关系
3. 对每条关系估计因果效应
4. 根据数据质量自动选择方法
5. 生成效应汇总表和详细报告
"""

import os
import sys
import pandas as pd
import networkx as nx
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal_inference.effect_extractor import EffectExtractor


# ============== 配置参数 ==============
CONFIG = {
    # 数据路径
    'data_path': r"D:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv",
    'graph_path': 'outputs/graphs/causal_dag.pkl',
    'encoded_data_path': 'outputs/data/step3_encoded_data.csv',

    # 输出路径
    'output_dir': 'outputs',
    'data_dir': 'outputs/data',
    'figures_dir': 'outputs/figures',
    'graphs_dir': 'outputs/graphs',
    'reports_dir': 'outputs/reports',
}


def create_output_dirs():
    """创建输出目录"""
    for dir_path in [CONFIG['output_dir'], CONFIG['data_dir'],
                     CONFIG['figures_dir'], CONFIG['reports_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    print("输出目录已创建")


def check_prerequisites():
    """检查前置条件"""
    print("\n" + "=" * 60)
    print("检查前置条件")
    print("=" * 60)

    # 检查数据文件
    if not os.path.exists(CONFIG['encoded_data_path']):
        print(f"错误: 编码数据文件不存在: {CONFIG['encoded_data_path']}")
        print("请先运行因果发现流程生成此文件")
        return False

    # 检查图结构文件
    if not os.path.exists(CONFIG['graph_path']):
        print(f"错误: 图结构文件不存在: {CONFIG['graph_path']}")
        print("请先运行因果发现流程并保存图结构")
        print("\n提示：在 experiments/simple_causal_discovery.py 中，")
        print("确保步骤6.5（保存图结构）已被执行")
        return False

    print("✓ 编码数据文件存在")
    print("✓ 图结构文件存在")
    print("前置条件检查通过")
    return True


def run_causal_inference():
    """运行因果效应估计流程"""

    print("=" * 60)
    print("简易版因果效应估计流程")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 创建输出目录
    create_output_dirs()

    # 检查前置条件
    if not check_prerequisites():
        return

    # 创建效应提取器
    extractor = EffectExtractor(
        data_path=CONFIG['encoded_data_path'],
        graph_path=CONFIG['graph_path']
    )

    # 提取所有因果效应
    effects_summary = extractor.extract_all_effects()

    if not effects_summary:
        print("\n警告：未能提取任何因果效应")
        return

    # 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    summary_df = extractor.save_results(CONFIG['data_dir'])

    # 显示汇总表
    print("\n" + "=" * 60)
    print("因果效应汇总表")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    # 证据等级分布
    evidence_counts = summary_df['证据等级'].value_counts()
    print("\n证据等级分布:")
    for level, count in evidence_counts.items():
        print(f"  {level}级: {count}条")

    # 方向分布
    direction_counts = summary_df['解读'].str.contains('改善').sum()
    print(f"\n改善趋势: {direction_counts}条（占总数的{direction_counts/len(summary_df)*100:.1f}%）")

    # 显著性分布
    significant_count = (summary_df['p值'] < 0.05).sum()
    print(f"统计显著: {significant_count}条（占总数的{significant_count/len(summary_df)*100:.1f}%）")

    # 生成报告
    generate_report(extractor, summary_df, CONFIG['reports_dir'])

    print("\n" + "=" * 60)
    print("因果效应估计流程完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return extractor, summary_df


def generate_report(extractor: EffectExtractor, summary_df: pd.DataFrame,
                   reports_dir: str):
    """生成因果效应估计报告"""

    report_path = os.path.join(reports_dir, 'simple_causal_inference_report.md')

    # 获取当前时间
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 构建报告
    report = f"""# 鼻咽癌药物疗效因果效应估计报告

**生成时间**: {now}

---

## 1. 研究概述

### 1.1 目标

在因果发现的基础上，对发现的药物→症状因果关系进行**定量效应估计**。

### 1.2 方法

- **倾向性评分匹配 (PSM)**: 1:1 近邻匹配，卡尺=0.2×SD
- **ATE估计方法**: 双稳健估计 (Doubly Robust)
- **混杂因素识别**: 基于因果图的后门准则

### 1.3 数据概况

- **总时序对数**: {len(extractor.data)} 对
- **分析的关系数**: {len(extractor.effects)} 条药物→症状关系

---

## 2. 核心发现：药物效应汇总表

{summary_df.to_markdown(index=False)}

**图例说明**：
- ATE < 0: 症状改善（负向效应）
- ATE > 0: 症状恶化（正向效应）
- 证据等级: A（因果证据）、B（初步证据）、C（关联证据）

---

## 3. 证据等级分布

"""

    # 添加证据等级统计
    evidence_counts = summary_df['证据等级'].value_counts()
    for level in ['A', 'B', 'C']:
        count = evidence_counts.get(level, 0)
        report += f"- **{level}级**: {count}条\n"

    report += f"""
---

## 4. 方法局限性

### 4.1 样本量限制

部分药物使用频率较低，可能导致因果效应估计不稳定。

### 4.2 未观测混杂

可能存在未观测的混杂因素，影响因果效应的估计。

### 4.3 时间间隔异质

不同患者之间的就诊间隔差异较大，未对时间间隔进行加权或分层分析。

### 4.4 药物剂量未考虑

本分析仅考虑药物是否使用，未考虑药物剂量差异。

---

## 5. 下一步工作

- [ ] 对时间间隔进行分层分析
- [ ] 考虑药物剂量的影响
- [ ] 进行Bootstrap稳定性验证
- [ ] 与中医专家讨论结果的合理性

---

**报告生成完毕**
"""

    # 保存报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存到: {report_path}")


if __name__ == "__main__":
    # 运行因果效应估计流程
    run_causal_inference()
