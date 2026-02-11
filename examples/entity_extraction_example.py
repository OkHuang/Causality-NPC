"""
实体提取使用示例

展示如何使用 entity_extraction 模块从临床主诉中提取结构化实体
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.entity_extraction import (
    EntityExtractor,
    PatientRecord,
    AzureConfig,
    flatten_patient_record,
    postprocess_results
)
import pandas as pd


def example_single_extraction():
    """示例 1: 单条主诉提取"""
    print("=" * 60)
    print("示例 1: 单条主诉提取")
    print("=" * 60)

    # 配置 Azure OpenAI (从环境变量加载)
    config = AzureConfig.from_env()
    extractor = EntityExtractor(config)

    # 示例主诉文本
    chief_complaint = (
        "鼻咽癌放化疗后11年余(2012年10月放疗结束,中肿)精神可,无恶风,怕冷,无头痛,头晕,咽干严重,无痰,无咳嗽,无鼻塞,喷嚏,流白清涕,无涕血,无耳鸣听力下降,无耳胀闷阻塞感,无耳流脓,无胃胀反酸,胃纳可,睡眠差,难入睡,易醒,多梦,夜尿1-2次,大便正常.舌硬活动不利.伸舌偏右,言语不利,吞咽欠顺畅."
    )

    # 执行提取
    result = extractor.extract_single(chief_complaint)

    if result:
        print("\n提取结果:")
        print(f"诊断: {result.medical_history.diagnosis}")
        print(f"治疗: {', '.join(result.medical_history.treatment)}")
        print(f"病程: {result.medical_history.duration}")
        print(f"\n症状数量: {len(result.symptoms)}")
        for symptom in result.symptoms[:5]:  # 只显示前5个
            print(f"  - {symptom.name}: {symptom.status} ({symptom.description})")

        # 展平为字典
        flat_dict = flatten_patient_record(result)
        print(f"\n展平后的字段数量: {len(flat_dict)}")
    else:
        print("提取失败")


def example_batch_extraction():
    """示例 2: 批量提取"""
    print("\n" + "=" * 60)
    print("示例 2: 批量提取")
    print("=" * 60)

    # 创建示例数据
    sample_data = [
        "鼻咽癌放化疗后11年余,精神可,无头痛,咽干严重,睡眠差,多梦.",
        "鼻咽癌化疗后6个月,胃纳差,反酸,失眠,夜尿3次.",
        "鼻咽癌放疗后2年,无特殊不适,胃纳可,睡眠好,大小便正常.",
        "鼻咽癌术后1年,时有咳嗽,痰少,鼻塞,流清涕.",
        "鼻咽癌综合治疗后5年,口干,耳鸣,听力下降,难入睡.",
    ]

    # 创建 DataFrame
    df = pd.DataFrame({
        "id": range(1, len(sample_data) + 1),
        "chief_complaint": sample_data
    })

    print(f"\n示例数据数量: {len(df)}")

    # 配置提取器
    config = AzureConfig.from_env()
    extractor = EntityExtractor(config)

    # 批量提取
    result_df = extractor.extract_from_dataframe(
        df,
        text_column="chief_complaint",
        batch_size=3  # 根据 Azure Rate Limit 调整
    )

    print(f"\n提取完成!")
    print(f"成功: {len(result_df[result_df['extracted_data'].notna()])} 条")

    # 后处理
    output_df = postprocess_results(result_df, "outputs/example_processed.xlsx")

    # 显示统计
    print("\n症状统计:")
    print(output_df[["diagnosis", "total_symptoms", "present_symptoms", "absent_symptoms"]])


def example_from_csv():
    """示例 3: 从 CSV 文件提取"""
    print("\n" + "=" * 60)
    print("示例 3: 从 CSV 文件提取")
    print("=" * 60)

    # 文件路径
    input_csv = "Data/raw/npc_final.csv"
    output_excel = "outputs/patients_extracted.xlsx"

    # 配置提取器
    config = AzureConfig.from_env()
    extractor = EntityExtractor(config)

    # 加载数据 (尝试不同编码)
    encodings = ["utf-8", "gbk", "gb2312", "gb18030"]
    df = None

    print("\n正在加载数据...")
    for encoding in encodings:
        try:
            df = pd.read_csv(input_csv, encoding=encoding)
            print(f"✓ 使用编码: {encoding}")
            break
        except Exception as e:
            continue

    if df is None:
        print("✗ 无法读取 CSV 文件")
        return

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 检查主诉列
    text_column = "chief_complaint"
    if text_column not in df.columns:
        print(f"✗ CSV 中没有列: {text_column}")
        return

    # 过滤空值
    valid_df = df[df[text_column].notna()].copy()
    print(f"\n有效记录数: {len(valid_df)}")

    # 限制处理数量（测试用）
    max_records = 50  # 设为 None 处理全部
    if max_records and len(valid_df) > max_records:
        print(f"测试模式: 只处理前 {max_records} 条记录")
        valid_df = valid_df.head(max_records)

    # 批量提取
    print(f"\n开始提取实体...")
    result_df = extractor.extract_from_dataframe(
        valid_df,
        text_column=text_column,
        batch_size=5  # 根据 Azure Rate Limit 调整
    )

    # 打印统计
    print(f"\n提取完成!")
    print(f"成功: {len(result_df[result_df['extracted_data'].notna()])} 条")
    print(f"失败: {len(extractor.errors)} 条")

    if extractor.errors:
        print("\n部分错误:")
        for i, error in enumerate(extractor.errors[:3]):
            print(f"  {i+1}. {error['error'][:100]}")

    # 后处理和保存
    print("\n正在后处理和保存...")
    output_df = postprocess_results(result_df, output_excel)

    # 显示统计摘要
    print("\n=== 提取统计 ===")
    print(f"主要诊断分布:")
    print(output_df["diagnosis"].value_counts().head())

    print(f"\n症状统计:")
    print(output_df[["total_symptoms", "present_symptoms", "absent_symptoms"]].describe())


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("实体提取使用示例")
    print("=" * 60)

    # 选择要运行的示例
    print("\n请选择要运行的示例:")
    print("1. 单条主诉提取")
    print("2. 批量提取 (5条示例数据)")
    print("3. 从 CSV 文件提取 (测试前50条)")
    print("4. 全部运行")

    choice = input("\n请输入选项 (1-4): ").strip()

    if choice == "1":
        example_single_extraction()
    elif choice == "2":
        example_batch_extraction()
    elif choice == "3":
        example_from_csv()
    elif choice == "4":
        example_single_extraction()
        example_batch_extraction()
        example_from_csv()
    else:
        print("无效选项，运行示例 1...")
        example_single_extraction()

    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
