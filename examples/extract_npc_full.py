"""
批量抽取 npc_full.csv 中的症状

功能：
1. 使用 SimplifiedSymptomExtractor 抽取所有主诉中的症状
2. 每 10 条保存一次中间结果到 Data/temp/checkpoints（保留所有断点）
3. 最终输出 JSON 格式的症状列（UTF-8 编码）
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.data.simplified_extraction import (
    SimplifiedSymptomExtractor,
    AzureConfig,
    STANDARD_SYMPTOMS
)


def extract_npc_full_with_checkpoint(
    input_path: str = "Data/raw/npc_full.csv",
    output_path: str = "Data/raw/npc_full_with_symptoms.csv",
    checkpoint_dir: str = "Data/temp/checkpoints",
    checkpoint_interval: int = 10,
    max_records: Optional[int] = None
):
    """
    批量抽取症状，支持断点续传

    Parameters
    ----------
    input_path : str
        输入 CSV 文件路径
    output_path : str
        输出 CSV 文件路径
    checkpoint_dir : str
        断点保存目录（保留所有断点）
    checkpoint_interval : int
        保存中间结果的间隔（条数）
    max_records : int
        最大处理记录数（用于测试，None 表示处理全部）
    """

    print("=" * 80)
    print("批量抽取 npc_full.csv 中的症状")
    print("=" * 80)

    # 创建断点目录
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"\n[0] 断点保存目录: {checkpoint_path}")

    # 1. 加载数据
    print("\n[1] 加载数据...")
    print(f"  输入: {input_path}")

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            print(f"  [OK] 使用编码: {encoding}")
            break
        except Exception as e:
            continue

    if df is None:
        print("  [X] 无法读取 CSV 文件")
        return None

    print(f"  数据形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()}")

    # 2. 检查主诉列
    text_column = "chief_complaint"
    if text_column not in df.columns:
        print(f"  [X] CSV 中没有列: {text_column}")
        return None

    # 过滤有效数据
    valid_df = df[df[text_column].notna()].copy()

    # 限制处理数量（用于测试）
    if max_records and len(valid_df) > max_records:
        print(f"\n[2] 测试模式: 只处理前 {max_records} 条记录")
        valid_df = valid_df.head(max_records)
    else:
        print(f"\n[2] 有效主诉记录: {len(valid_df)} 条")

    # 3. 检查断点续传
    start_index = 0
    checkpoint_info_path = checkpoint_path / "checkpoint_info.json"

    if checkpoint_info_path.exists():
        try:
            with open(checkpoint_info_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                start_index = checkpoint.get('last_index', 0)
                print(f"\n[3] 检测到断点文件，从第 {start_index} 条继续...")
        except Exception as e:
            print(f"\n[3] 断点文件损坏，从头开始: {e}")
            start_index = 0
    else:
        print(f"\n[3] 未检测到断点文件，从头开始...")

    # 如果已保存部分结果，加载它
    if start_index > 0:
        # 找到最新的断点文件
        checkpoint_files = sorted(checkpoint_path.glob("checkpoint_*.csv"))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            try:
                result_df = pd.read_csv(latest_checkpoint, encoding='utf-8')
                print(f"  已加载断点: {latest_checkpoint.name}")
                print(f"  已处理: {len(result_df)} 条")
            except Exception as e:
                print(f"  [X] 无法加载断点，从头开始: {e}")
                result_df = valid_df.head(0).copy()
                result_df['extracted_symptoms'] = ''
                start_index = 0
        else:
            result_df = valid_df.head(0).copy()
            result_df['extracted_symptoms'] = ''
            start_index = 0
    else:
        result_df = valid_df.head(0).copy()
        result_df['extracted_symptoms'] = ''

    # 4. 创建提取器
    print(f"\n[4] 创建提取器...")
    config = AzureConfig.from_env()
    extractor = SimplifiedSymptomExtractor(config)
    print("  [OK] 提取器已创建")

    # 5. 批量抽取
    print(f"\n[5] 开始批量抽取...")
    print(f"  起始索引: {start_index}")
    print(f"  总记录数: {len(valid_df)}")
    print(f"  待处理: {len(valid_df) - start_index} 条")
    print(f"  保存间隔: 每 {checkpoint_interval} 条")

    start_time = time.time()

    # 逐条处理（支持断点续传）
    for idx in range(start_index, len(valid_df)):
        row = valid_df.iloc[idx]
        chief_complaint = row[text_column]

        # 抽取症状
        result = extractor.extract_single(chief_complaint)

        # 构建结果行
        new_row = row.copy()
        if result and result.symptoms:
            symptoms_data = []
            for s in result.symptoms:
                symptoms_data.append({
                    "name": s.name,
                    "severity": s.severity,
                    "label": {0: "无", 1: "轻度", 2: "重度"}[s.severity]
                })
            new_row['extracted_symptoms'] = json.dumps(symptoms_data, ensure_ascii=False)
        else:
            new_row['extracted_symptoms'] = ""

        # 添加到结果 DataFrame
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

        # 显示进度
        if (idx + 1) % 5 == 0 or idx == start_index:
            elapsed = time.time() - start_time
            success_count = len(result_df[result_df['extracted_symptoms'].astype(bool)])
            progress = (idx + 1) / len(valid_df) * 100
            print(f"  进度: {idx + 1}/{len(valid_df)} ({progress:.1f}%) | "
                  f"成功: {success_count} | "
                  f"失败: {len(extractor.errors)} | "
                  f"耗时: {elapsed:.1f}秒")

        # 定期保存中间结果（保留所有断点）
        if (idx + 1) % checkpoint_interval == 0 or (idx + 1) == len(valid_df):
            # 保存断点 CSV
            checkpoint_file = checkpoint_path / f"checkpoint_{idx + 1:04d}.csv"
            result_df.to_csv(checkpoint_file, index=False, encoding='utf-8')
            print(f"  [OK] 已保存断点: {checkpoint_file.name}")

            # 保存断点信息
            with open(checkpoint_info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'last_index': idx + 1,
                    'timestamp': datetime.now().isoformat(),
                    'total_records': len(valid_df),
                    'success_count': len(result_df[result_df['extracted_symptoms'].astype(bool)]),
                    'error_count': len(extractor.errors),
                    'latest_checkpoint': str(checkpoint_file)
                }, f, ensure_ascii=False, indent=2)

    # 6. 最终保存
    print(f"\n[6] 保存最终结果...")
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  [OK] 结果已保存至: {output_path}")
    print(f"  [OK] 编码: UTF-8")
    print(f"  [OK] 所有断点保存在: {checkpoint_dir}")

    # 7. 统计
    print(f"\n[7] 统计分析...")
    total_time = time.time() - start_time
    success_count = len(result_df[result_df['extracted_symptoms'].astype(bool)])
    error_count = len(extractor.errors)

    print(f"  总记录数: {len(valid_df)}")
    print(f"  成功提取: {success_count} ({success_count/len(valid_df)*100:.1f}%)")
    print(f"  提取失败: {error_count} ({error_count/len(valid_df)*100:.1f}%)")
    print(f"  总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  平均速度: {len(valid_df)/total_time:.2f} 条/秒")

    # 8. 症状统计
    if success_count > 0:
        print(f"\n[8] 症状分布统计...")

        # 统计每个症状的出现频率
        symptom_counts = {symptom: 0 for symptom in STANDARD_SYMPTOMS}

        for _, row in result_df.iterrows():
            if row['extracted_symptoms']:
                try:
                    symptoms = json.loads(row['extracted_symptoms'])
                    for s in symptoms:
                        if s['name'] in symptom_counts:
                            symptom_counts[s['name']] += 1
                except Exception:
                    pass

        # 排序并显示
        sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  最常见的 15 个症状:")
        for symptom, count in sorted_symptoms[:15]:
            if count > 0:
                print(f"    {symptom}: {count} 次")

    print("\n" + "=" * 80)
    print("抽取完成！")
    print("=" * 80)

    return result_df


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("NPC Full 数据集 - 批量症状抽取（测试模式：前20条）")
    print("=" * 80)

    # 运行抽取
    result_df = extract_npc_full_with_checkpoint(
        input_path="Data/raw/npc_full.csv",
        output_path="Data/raw/npc_full_with_symptoms.csv",
        checkpoint_dir="Data/temp/checkpoints",
        checkpoint_interval=10,  # 每 10 条保存一次
        max_records=None  # 处理全部数据
    )

    if result_df is not None:
        print("\n输出文件:")
        print("  - Data/raw/npc_full_with_symptoms.csv (最终结果)")
        print("  - Data/temp/checkpoints/ (所有断点)")
        print("\n下一步:")
        print("  1. 查看 npc_full_with_symptoms.csv 中的 extracted_symptoms 列")
        print("  2. 确认无误后，修改 max_records=None 处理全部数据")


if __name__ == "__main__":
    main()
