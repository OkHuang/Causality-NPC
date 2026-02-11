"""
测试前20个样本的症状提取
"""

import sys
import os
from pathlib import Path
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.data.simplified_extraction import SimplifiedSymptomExtractor, AzureConfig, symptoms_to_dataframe

# 加载环境变量
load_dotenv()


def main():
    """测试前20个样本"""
    print("=" * 80)
    print("测试前20个样本的症状提取")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    input_path = "Data/raw/npc_final.csv"

    # 尝试不同编码读取
    encodings = ["utf-8", "gbk", "gb2312", "gb18030"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            print(f"成功使用编码: {encoding}")
            break
        except Exception as e:
            continue

    if df is None:
        print("无法读取 CSV 文件")
        return

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 2. 检查主诉列
    text_column = "chief_complaint"
    if text_column not in df.columns:
        print(f"CSV 中没有列: {text_column}")
        return

    # 3. 取前10条有效数据
    print(f"\n[2] 提取前10条有效数据...")
    valid_df = df[df[text_column].notna()].head(10)
    print(f"有效记录数: {len(valid_df)}")

    # 4. 创建提取器
    print(f"\n[3] 创建提取器...")
    config = AzureConfig.from_env()
    extractor = SimplifiedSymptomExtractor(config)

    # 5. 逐条提取并显示结果
    print(f"\n[4] 开始提取症状...")
    print("=" * 80)

    results = []
    for idx, row in valid_df.iterrows():
        chief_complaint = row[text_column]

        print(f"\n【样本 {idx + 1}】")
        print(f"主诉: {chief_complaint[:100]}...")  # 只显示前100字符

        try:
            result = extractor.extract_single(chief_complaint)

            if result and result.symptoms:
                # 分类统计
                absent = [s for s in result.symptoms if s.severity == 0]
                mild = [s for s in result.symptoms if s.severity == 1]
                severe = [s for s in result.symptoms if s.severity == 2]

                print(f"提取到 {len(result.symptoms)} 个症状:")
                print(f"  - 阴性(无): {len(absent)}个 - {', '.join([s.name for s in absent])}")
                print(f"  - 轻度: {len(mild)}个 - {', '.join([s.name for s in mild])}")
                print(f"  - 重度: {len(severe)}个 - {', '.join([s.name for s in severe])}")

                # 转换为向量
                symptom_df = symptoms_to_dataframe(result)
                results.append({
                    "index": idx,
                    "text": chief_complaint,
                    "symptoms": result.symptoms,
                    "vector": symptom_df
                })
            else:
                print("未提取到症状")

        except Exception as e:
            print(f"提取失败: {e}")

        print("-" * 80)

    # 6. 保存结果
    print(f"\n[5] 保存结果...")

    # 使用新的保存函数
    from src.data.simplified_extraction import save_with_json_column

    # 提取结果列表
    extracted_data = []
    for r in results:
        extracted_data.append(r["symptoms"] if r else None)

    # 保存为CSV（包含原始数据和提取的JSON）
    output_path = "Data/raw/npc_data_with_symptoms.csv"
    result_df = save_with_json_column(
        valid_df,
        extracted_data,
        output_path=output_path
    )

    # 7. 统计
    print(f"\n[6] 统计分析...")
    success_count = len([r for r in results if r])
    print(f"成功提取: {success_count}/{len(valid_df)}")
    if extractor.errors:
        print(f"失败: {len(extractor.errors)}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
