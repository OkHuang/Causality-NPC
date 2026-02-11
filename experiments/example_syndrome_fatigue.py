"""
示例：证型-症状因果推断

验证性因果推断：评估"气虚证"对"乏力"的因果效应
验证中医理论：气虚证是否会导致乏力加重
"""

import sys
import os
# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 切换工作目录到项目根目录（这样所有相对路径都能正常工作）
os.chdir(project_root)

import pandas as pd
import numpy as np
from pathlib import Path

# 导入项目模块
from src.data import DataLoader, DataPreprocessor, TimeAligner
from src.features import SymptomExtractor, MedicineMapper, SyndromeEncoder
from src.causal_inference import PropensityScoreMatcher, ATEEstimator
from src.causal_inference.matching import match_patients, assess_match_quality
from src.causal_inference.validator import BalanceValidator
from src.visualization import ReportGenerator
from src.utils.config import load_config


def main():
    """主函数"""

    print("="*80)
    print("鼻咽癌因果推断 - 证型-症状关系分析")
    print("研究问题：气虚证是否导致乏力加重？")
    print("="*80)

    # ============================================
    # 1. 加载配置
    # ============================================
    print("\n[1] 加载配置...")
    config = load_config(
        "config/causal_inference.yaml",
        base_config_path="config/base.yaml"
    )

    # ============================================
    # 2. 加载数据
    # ============================================
    print("\n[2] 加载数据...")
    loader = DataLoader()
    # 使用实际存在的数据文件
    data_path = "Data/raw/npc_final.csv"
    if not os.path.exists(data_path):
        data_path = "data/raw/npc_data.csv"
    # 使用gbk编码读取CSV文件
    df = loader.load(data_path, encoding='gbk')

    print(f"原始数据: {df.shape}")
    print(f"患者数: {df['patient_id'].nunique()}")

    # ============================================
    # 3. 数据预处理
    # ============================================
    print("\n[3] 数据预处理...")

    # 3.1 基本清洗
    preprocessor = DataPreprocessor(config.get("preprocessing", {}))
    df_clean = preprocessor.clean(df)

    # 3.2 队列筛选
    df_cohort = preprocessor.filter_cohort(
        df_clean,
        min_follow_up=config.get("cohort", {}).get("min_follow_up", 30),
        min_visits=config.get("cohort", {}).get("min_visits", 2)
    )

    # ============================================
    # 4. 时间对齐
    # ============================================
    print("\n[4] 时间轴对齐...")
    time_aligner = TimeAligner(config.get("time_alignment", {}))
    df_aligned = time_aligner.align(df_cohort)

    # ============================================
    # 5. 特征提取
    # ============================================
    print("\n[5] 特征工程...")

    # 5.1 症状提取
    print("  - 提取症状...")
    symptom_extractor = SymptomExtractor(config.get("symptom_extraction", {}))
    df_features = symptom_extractor.transform(df_aligned, text_column="chief_complaint")

    # 5.2 药物映射
    print("  - 映射药物...")
    medicine_mapper = MedicineMapper(config.get("medicine_mapping", {}))
    df_features = medicine_mapper.transform(df_features, medicine_column="chinese_medicines")

    # 5.3 证型编码
    print("  - 编码证型...")
    syndrome_encoder = SyndromeEncoder(config.get("syndrome_encoding", {}))
    df_features = syndrome_encoder.fit_transform(df_features, syndrome_column="chinese_diagnosis")

    # 5.4 创建时序对
    print("  - 创建时序特征...")
    df_time = time_aligner.create_lag_features(df_features, lag=1)

    print(f"最终特征: {df_time.shape}")

    # ============================================
    # 6. 定义研究假设
    # ============================================
    print("\n[6] 定义研究假设...")

    # 处理变量：气虚证（组合所有气虚相关的证型）
    qi_deficiency_syndromes = [
        col for col in df_time.columns
        if "气虚" in col and col.startswith("D_")
    ]

    print(f"  检测到的气虚证型: {qi_deficiency_syndromes}")

    if qi_deficiency_syndromes:
        df_time["treatment"] = df_time[qi_deficiency_syndromes].max(axis=1)
    else:
        print("警告：未找到'气虚'证型，请检查证型提取")
        return

    # 结果变量：乏力变化（t+1时刻相对于t时刻的变化）
    if "S_乏力" in df_time.columns:
        df_time["outcome"] = df_time.groupby("patient_id")["S_乏力"].diff().shift(-1)
        # 注意：这里不取负号，正值表示乏力加重（符合直觉）
        print("  结果变量：乏力变化（正=加重，负=改善）")
    else:
        print("警告：未找到'乏力'症状，请检查症状提取")
        return

    # 混杂因素：年龄 + 其他证型（不包括气虚）
    confounders = ["age"]

    # 添加其他证型特征作为混杂因素
    other_syndrome_cols = [
        col for col in df_time.columns
        if col.startswith("D_") and col not in qi_deficiency_syndromes
    ]
    confounders.extend(other_syndrome_cols)

    # 添加基线症状
    if "S_乏力" in df_time.columns:
        confounders.append("S_乏力")

    # 确保混杂因素都是数值类型
    numeric_confounders = []
    for col in confounders:
        if col in df_time.columns and pd.api.types.is_numeric_dtype(df_time[col]):
            numeric_confounders.append(col)
    confounders = numeric_confounders

    # 移除缺失值
    df_analysis = df_time.dropna(subset=["treatment", "outcome"] + confounders)

    print(f"分析样本: {df_analysis.shape}")
    print(f"  气虚证组: {df_analysis['treatment'].sum()}")
    print(f"  非气虚组: {(df_analysis['treatment'] == 0).sum()}")

    # ============================================
    # 7. 倾向性评分匹配
    # ============================================
    print("\n[7] 倾向性评分匹配...")

    # 计算倾向性评分
    ps_matcher = PropensityScoreMatcher(config.get("propensity_score", {}))
    df_ps = ps_matcher.fit_transform(
        df_analysis,
        treatment_col="treatment",
        confounder_cols=confounders
    )

    # 检查共同支持
    common_support = ps_matcher.check_common_support(df_ps, "treatment", "propensity_score")

    # 匹配患者
    df_matched, match_info = match_patients(
        df_ps,
        treatment_col="treatment",
        propensity_col="propensity_score",
        method=config.get("matching", {}).get("method", "nearest_neighbor"),
        ratio=config.get("matching", {}).get("ratio", 1),
        caliper=config.get("matching", {}).get("caliper", 0.2),
    )

    print(f"匹配结果: {match_info['n_pairs']} 对")

    # ============================================
    # 8. 平衡性检验
    # ============================================
    print("\n[8] 平衡性检验...")

    balance_df = assess_match_quality(
        df_matched,
        treatment_col="treatment",
        confounder_cols=confounders,
    )

    print(balance_df)

    # 添加type列（用于绘图）
    balance_df["type"] = "continuous"

    # 可视化平衡性
    try:
        validator = BalanceValidator()
        validator.plot_balance(balance_df, save_path="outputs/figures/balance_check_syndrome.png")
    except Exception as e:
        print(f"绘图跳过（{e}）")

    # ============================================
    # 9. 估计因果效应
    # ============================================
    print("\n[9] 估计平均处理效应（ATE）...")

    ate_estimator = ATEEstimator(config.get("ate_estimation", {}))
    results = ate_estimator.estimate(
        df_matched,
        treatment_col="treatment",
        outcome_col="outcome",
        method=config.get("ate_estimation", {}).get("methods", ["difference_in_means"])[0],
        confounder_cols=confounders,
    )

    # ============================================
    # 10. 生成报告
    # ============================================
    print("\n[10] 生成报告...")

    report_gen = ReportGenerator()

    # 整合结果
    final_results = {
        "treatment": "气虚证",
        "outcome": "乏力变化",
        "confounders": confounders,
        "ps_model": config.get("propensity_score", {}).get("model"),
        "matching_method": config.get("matching", {}).get("method"),
        "ate_method": config.get("ate_estimation", {}).get("methods", [""])[0],
        **results,
    }

    report_gen.generate_causal_inference_report(
        final_results,
        balance_df,
        "outputs/reports/syndrome_fatigue_report.md"
    )

    print("\n" + "="*80)
    print("分析完成！")
    print(f"报告已保存至: outputs/reports/syndrome_fatigue_report.md")
    print("="*80)

    # 额外输出：解释结果
    print("\n【结果解读】")
    ate = results.get("ate", 0)
    p_value = results.get("p_value", 1)

    if ate > 0 and p_value < 0.05:
        print(f"✓ 气虚证显著导致乏力加重（ATE={ate:.4f}, p={p_value:.4f}）")
        print("  这符合中医理论：气虚则乏力")
    elif ate < 0 and p_value < 0.05:
        print(f"✗ 气虚证反而与乏力改善相关（ATE={ate:.4f}, p={p_value:.4f}）")
        print("  这与中医理论不符，需要进一步分析")
    else:
        print(f"○ 气虚证对乏力的影响无统计学意义（ATE={ate:.4f}, p={p_value:.4f}）")
        print("  可能原因：样本量不足、证型诊断不准确、或干预措施混淆")


if __name__ == "__main__":
    main()
