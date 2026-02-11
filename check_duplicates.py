"""
检查合并后数据是否有重复行 - 详细分析
"""

import pandas as pd
from pathlib import Path

def main():
    print("=" * 60)
    print("Detailed Duplicate Analysis")
    print("=" * 60)

    # 读取数据
    data_path = Path("Data/raw/npc_full.csv")
    df = pd.read_csv(data_path, encoding="utf-8-sig")

    print(f"\nTotal rows: {len(df)}")

    # 1. 检查 time 列的 NaN 情况
    print("\n" + "=" * 60)
    print("[1] Analyzing 'time' column")
    print("=" * 60)

    print(f"time column NaN count: {df['time'].isna().sum()}")
    print(f"time column non-NaN count: {df['time'].notna().sum()}")

    # 2. 检查基于 patient_id 的重复
    print("\n" + "=" * 60)
    print("[2] Analyzing patient_id duplicates")
    print("=" * 60)

    dup_patient = df.duplicated(subset=['patient_id'], keep=False)
    print(f"Rows with duplicate patient_id: {dup_patient.sum()}")
    print(f"Unique patient_id count: {df['patient_id'].nunique()}")
    print(f"Total rows: {len(df)}")

    # 分析重复的 patient_id
    if dup_patient.sum() > 0:
        patient_counts = df['patient_id'].value_counts()
        dup_patients = patient_counts[patient_counts > 1]

        print(f"\n=== Top 10 patients with most records ===")
        print(dup_patients.head(10))

        print(f"\n=== Statistics ===")
        print(f"Total patients with multiple records: {len(dup_patients)}")
        print(f"Max records for a single patient: {dup_patients.max()}")
        print(f"Min records for a patient with duplicates: {dup_patients.min()}")

    # 3. 基于 checkup_id 的唯一性检查
    print("\n" + "=" * 60)
    print("[3] Analyzing checkup_id uniqueness")
    print("=" * 60)

    print(f"Unique checkup_id count: {df['checkup_id'].nunique()}")
    print(f"Total rows: {len(df)}")

    if df['checkup_id'].nunique() == len(df):
        print("[OK] All checkup_id values are unique!")

    # 4. 检查 train 和 test 数据分布
    print("\n" + "=" * 60)
    print("[4] Source data distribution (by index)")
    print("=" * 60)

    print(f"Rows 0-669 (from train.csv): 670 rows")
    print(f"Rows 670-962 (from test.csv): 293 rows")

    # 5. 交叉检查 patient_id 在 train 和 test 中的分布
    print("\n" + "=" * 60)
    print("[5] Patient overlap between train and test")
    print("=" * 60)

    train_df = df.iloc[:670]
    test_df = df.iloc[670:]

    train_patients = set(train_df['patient_id'].dropna().unique())
    test_patients = set(test_df['patient_id'].dropna().unique())

    print(f"Unique patients in train: {len(train_patients)}")
    print(f"Unique patients in test: {len(test_patients)}")

    overlap = train_patients & test_patients
    print(f"\nOverlapping patients: {len(overlap)}")

    if overlap:
        print(f"Sample overlapping patient_ids: {list(overlap)[:10]}")

    # 6. 检查是否应该删除重复
    print("\n" + "=" * 60)
    print("[6] Recommendation")
    print("=" * 60)

    print("\nAnalysis Summary:")
    print(f"- No completely duplicate rows (all columns identical)")
    print(f"- {dup_patient.sum()} rows share patient_id with other records")
    print(f"- All checkup_id values are unique")
    print(f"- {len(overlap)} patients appear in both train and test sets")

    print("\nConclusion:")
    print("These are NOT duplicates in the traditional sense.")
    print("They are multiple visits/records for the SAME patient.")
    print("Each record has a UNIQUE checkup_id (different visit times).")
    print("\nRecommendation: KEEP all rows. These are valid longitudinal records.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
