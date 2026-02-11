"""
验证抽取结果
"""
import pandas as pd
import json

df = pd.read_csv(r'D:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv', encoding='utf-8')

print('Shape:', df.shape)
print('Columns:', df.columns.tolist())

print('\n=== First 3 rows ===')
for i, row in df.head(3).iterrows():
    print(f'\nRow {i}:')
    print(f'  Chief Complaint: {row["chief_complaint"][:100]}...')
    print(f'  Extracted Symptoms: {row["extracted_symptoms"][:150]}...')

print('\n=== Statistics ===')
print(f'Total rows: {len(df)}')
print(f'Non-empty symptoms: {df["extracted_symptoms"].astype(bool).sum()}')
