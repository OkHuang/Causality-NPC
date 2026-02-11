"""
合并 train.csv 和 test.csv 为 npc_full
"""

import pandas as pd
import os
from pathlib import Path

# 设置路径
data_dir = Path("Data/raw")

def main():
    print("=" * 80)
    print("合并 train.csv 和 test.csv")
    print("=" * 80)

    # 1. 读取数据
    print("\n[1] 读取数据...")

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    # 尝试不同编码读取
    encodings = ["utf-8", "gbk", "gb2312", "gb18030"]

    train_df = None
    test_df = None

    for encoding in encodings:
        try:
            if train_path.exists():
                train_df = pd.read_csv(train_path, encoding=encoding)
                print(f"[OK] 读取 train.csv (编码: {encoding})")
            break
        except Exception as e:
            continue

    for encoding in encodings:
        try:
            if test_path.exists():
                test_df = pd.read_csv(test_path, encoding=encoding)
                print(f"[OK] 读取 test.csv (编码: {encoding})")
            break
        except Exception as e:
            continue

    if train_df is None and test_df is None:
        print("[X] 无法读取文件")
        return

    # 2. 合并数据
    print(f"\n[2] 合并数据...")
    print(f"  train.csv: {train_df.shape if train_df is not None else '不存在'}")
    print(f"  test.csv: {test_df.shape if test_df is not None else '不存在'}")

    dfs = []
    if train_df is not None:
        dfs.append(train_df)
    if test_df is not None:
        dfs.append(test_df)

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"[OK] 合并后: {merged_df.shape}")

    # 3. 删除 id 列
    print(f"\n[3] 删除 id 列...")
    if 'id' in merged_df.columns:
        merged_df = merged_df.drop(columns=['id'])
        print(f"[OK] 已删除 id 列")
    else:
        print(f"  未找到 id 列")

    # 4. 保存
    print(f"\n[4] 保存文件...")
    output_path = data_dir / "npc_full.csv"

    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存至: {output_path}")
    print(f"  格式: UTF-8")
    print(f"  形状: {merged_df.shape}")

    # 同时保存为 Excel 格式
    output_xlsx_path = data_dir / "npc_full.xlsx"
    merged_df.to_excel(output_xlsx_path, index=False, engine="openpyxl")
    print(f"[OK] 已保存至: {output_xlsx_path}")
    print(f"  格式: Excel (UTF-8)")

    print("\n" + "=" * 80)
    print("合并完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
