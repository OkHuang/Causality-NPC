import csv
import json
import re
import sys

# 设置输出编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')

symptoms_set = set()
row_count = 0
error_count = 0

with open(r'd:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_count += 1
        extracted_symptoms = row.get('extracted_symptoms', '')
        if extracted_symptoms and extracted_symptoms.strip():
            try:
                # 尝试解析JSON
                symptoms = json.loads(extracted_symptoms)
                for symptom in symptoms:
                    name = symptom.get('name', '')
                    if name:
                        symptoms_set.add(name)
            except Exception as e:
                # 如果JSON解析失败，使用正则表达式提取
                try:
                    pattern = r'"name":\s*"([^"]+)"'
                    matches = re.findall(pattern, extracted_symptoms)
                    for match in matches:
                        if match:
                            symptoms_set.add(match)
                except:
                    error_count += 1
                    pass

print(f'总行数: {row_count}')
print(f'解析错误行数: {error_count}')
print(f'\n数据集中提取出的症状总数: {len(symptoms_set)}')
print(f'\n所有症状列表 (按字母顺序排序):')
for i, symptom in enumerate(sorted(symptoms_set), 1):
    print(f'{i}. {symptom}')
