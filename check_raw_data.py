# -*- coding: utf-8 -*-
"""
检查原始数据中的"牛蒡子"和"淋"相关内容
"""
import pandas as pd
import json

# 读取原始CSV
df = pd.read_csv(r'D:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv')

print('=' * 60)
print('检查原始数据')
print('=' * 60)

# 1. 检查chinese_medicines列
print('\n【1. 检查chinese_medicines列】')
med_col = df['chinese_medicines'].dropna()

# 查找包含"牛蒡"的记录
niu_bang = med_col[med_col.str.contains('牛蒡', na=False)]
print(f'\n包含"牛蒡"的记录数: {len(niu_bang)}')
if len(niu_bang) > 0:
    print('示例:')
    for i, med in enumerate(niu_bang.head(5)):
        print(f'  {i+1}. {med}')

# 查找包含"Z"的药物记录
has_Z = med_col[med_col.str.contains('Z', na=False)]
print(f'\n包含"Z"的记录数: {len(has_Z)}')
if len(has_Z) > 0:
    print('示例:')
    for i, med in enumerate(has_Z.head(10)):
        print(f'  {i+1}. {med}')

# 2. 检查诊断列
print('\n【2. 检查诊断列】')
diag_col = df['chinese_diagnosis'].dropna()

# 查找包含"淋"的记录
lin_records = diag_col[diag_col.str.contains('淋', na=False)]
print(f'\n包含"淋"的诊断记录数: {len(lin_records)}')
if len(lin_records) > 0:
    print('示例:')
    for i, diag in enumerate(lin_records.head(10)):
        print(f'  {i+1}. {diag}')

# 查找包含"子淋"的记录
zi_lin_records = diag_col[diag_col.str.contains('子淋', na=False)]
print(f'\n包含"子淋"的诊断记录数: {len(zi_lin_records)}')
if len(zi_lin_records) > 0:
    print('示例:')
    for i, diag in enumerate(zi_lin_records.head(10)):
        print(f'  {i+1}. {diag}')

# 3. 检查是否有分隔符问题
print('\n【3. 检查分隔符】')
# 查看包含多种空格/分隔符的记录
import re
complex_meds = med_col[med_col.str.contains(r'[,\t、]|  ', na=False)]
print(f'\n包含多种分隔符的药物记录数: {len(complex_meds)}')
if len(complex_meds) > 0:
    print('示例:')
    for i, med in enumerate(complex_meds.head(10)):
        print(f'  {i+1}. [{repr(med)}]')

print('\n' + '=' * 60)
