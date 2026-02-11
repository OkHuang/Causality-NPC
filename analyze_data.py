import csv
import json
import re
import sys
from collections import Counter

# 设置输出编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 存储所有数据
chinese_medicines_set = set()
chinese_diagnoses_set = set()
western_diagnoses_set = set()
chinese_medicine_counter = Counter()

row_count = 0

with open(r'd:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_count += 1

        # 分析中药
        chinese_medicines = row.get('chinese_medicines', '')
        if chinese_medicines and chinese_medicines.strip():
            # 按空格分割
            medicines = [m.strip() for m in chinese_medicines.split() if m.strip()]
            for med in medicines:
                chinese_medicines_set.add(med)
                chinese_medicine_counter[med] += 1

        # 分析中医证型
        chinese_diagnosis = row.get('chinese_diagnosis', '')
        if chinese_diagnosis and chinese_diagnosis.strip():
            # 中医证型可能包含多个，用空格分割
            diagnoses = [d.strip() for d in chinese_diagnosis.split() if d.strip()]
            for diag in diagnoses:
                chinese_diagnoses_set.add(diag)

        # 分析西医诊断
        western_diagnosis = row.get('western_diagnosis', '')
        if western_diagnosis and western_diagnosis.strip():
            # 西医诊断可能包含多个，用空格分割
            diagnoses = [d.strip() for d in western_diagnosis.split() if d.strip()]
            for diag in diagnoses:
                western_diagnoses_set.add(diag)

print('=' * 60)
print(f'数据集统计 (总行数: {row_count})')
print('=' * 60)

print(f'\n【中药统计】')
print(f'中药种类数: {len(chinese_medicines_set)}')
print(f'\n最常用的20味中药:')
for i, (med, count) in enumerate(chinese_medicine_counter.most_common(20), 1):
    print(f'{i}. {med}: {count}次')

print(f'\n【中医证型统计】')
print(f'中医证型种类数: {len(chinese_diagnoses_set)}')
print(f'\n所有中医证型列表 (按字母顺序排序):')
for i, diag in enumerate(sorted(chinese_diagnoses_set), 1):
    print(f'{i}. {diag}')

print(f'\n【西医诊断统计】')
print(f'西医诊断种类数: {len(western_diagnoses_set)}')
print(f'\n所有西医诊断列表 (按字母顺序排序):')
for i, diag in enumerate(sorted(western_diagnoses_set), 1):
    print(f'{i}. {diag}')
