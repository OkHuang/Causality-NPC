"""
检查步骤3的编码数据
"""
import pandas as pd
import numpy as np

# 读取编码数据
df = pd.read_csv(r'D:\WorkProject\Causality-NPC\outputs\data\step3_encoded_data.csv')

print('=' * 60)
print('步骤3编码数据检查报告')
print('=' * 60)

# 1. 基本信息
print(f'\n【1. 基本信息】')
print(f'数据形状: {df.shape}')
print(f'样本数（配对数）: {df.shape[0]}')
print(f'总特征数: {df.shape[1]}')

# 2. 检查各类型特征
print(f'\n【2. 特征类型分布】')
symptom_t = [c for c in df.columns if c.endswith('_t') and not c.startswith(('med_', 'diagnosis_', 'age_', 'gender_', 'time_'))]
med_t = [c for c in df.columns if c.startswith('med_') and c.endswith('_t')]
diagnosis_t = [c for c in df.columns if c.startswith('diagnosis_') and c.endswith('_t')]
symptom_t1 = [c for c in df.columns if c.endswith('_t1') and not c.startswith(('med_', 'diagnosis_', 'age_', 'gender_', 'time_'))]
med_t1 = [c for c in df.columns if c.startswith('med_') and c.endswith('_t1')]
diagnosis_t1 = [c for c in df.columns if c.startswith('diagnosis_') and c.endswith('_t1')]
age_cols = [c for c in df.columns if c.startswith('age_')]
static_cols = ['gender_encoded'] if 'gender_encoded' in df.columns else []

print(f'  症状_t: {len(symptom_t)} 个')
print(f'  药物_t: {len(med_t)} 个')
print(f'  诊断_t: {len(diagnosis_t)} 个')
print(f'  症状_t1: {len(symptom_t1)} 个')
print(f'  药物_t1: {len(med_t1)} 个')
print(f'  诊断_t1: {len(diagnosis_t1)} 个')
print(f'  年龄分箱: {len(age_cols)} 个')
print(f'  静态特征: {len(static_cols)} 个')
print(f'  总计: {len(symptom_t)+len(med_t)+len(diagnosis_t)+len(symptom_t1)+len(med_t1)+len(diagnosis_t1)+len(age_cols)+len(static_cols)} 个')

# 3. 检查t和t1对称性
print(f'\n【3. t和t1对称性检查】')
sym_match = "OK" if len(symptom_t) == len(symptom_t1) else "X NOT MATCH"
diag_match = "OK" if len(diagnosis_t) == len(diagnosis_t1) else "X NOT MATCH"
med_match = "OK" if len(med_t) == len(med_t1) else "X NOT MATCH"
print(f'  症状: {len(symptom_t)} (t) vs {len(symptom_t1)} (t1) - {sym_match}')
print(f'  诊断: {len(diagnosis_t)} (t) vs {len(diagnosis_t1)} (t1) - {diag_match}')
print(f'  药物: {len(med_t)} (t) vs {len(med_t1)} (t1) - {med_match}')

# 4. 检查频率（阈值40%）
print(f'\n【4. 特征频率检查（阈值40%）】')
n_pairs = len(df)
threshold_count = int(n_pairs * 0.40)
print(f'  配对数: {n_pairs}')
print(f'  40%阈值: 至少 {threshold_count} 对 ({threshold_count/n_pairs*100:.1f}%)')

def check_freq(cols, name):
    if not cols:
        print(f'  {name}: 无特征')
        return []

    freqs = []
    for col in cols:
        count = (df[col] > 0).sum()
        freq = count / n_pairs
        freqs.append((col, count, freq))

    # 找出低于阈值的
    low_freq = [(col, count, f'{freq*100:.1f}%') for col, count, freq in freqs if freq < 0.40]

    # 找出高于阈值的
    high_freq = [(col, count, f'{freq*100:.1f}%') for col, count, freq in freqs if freq >= 0.40]

    if low_freq:
        print(f'  {name} - 低于40%: {len(low_freq)}个, 高于40%: {len(high_freq)}个')
        print(f'    保留的特征(>=40%):')
        for col, count, pct in high_freq[:10]:
            print(f'      - {col}: {count} ({pct})')
        if len(high_freq) > 10:
            print(f'      ... 还有 {len(high_freq)-10} 个')
    else:
        print(f'  {name}: OK 所有特征都>=40%')

    return high_freq

high_symptoms = check_freq(symptom_t, '症状_t')
high_meds = check_freq(med_t, '药物_t')
high_diagnoses = check_freq(diagnosis_t, '诊断_t')

# 5. 检查数据类型
print(f'\n【5. 数据类型检查】')
print(df.dtypes.value_counts())

# 6. 检查缺失值
print(f'\n【6. 缺失值检查】')
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f'  发现 {len(missing_cols)} 列有缺失值:')
    print(missing_cols.head(10))
else:
    print(f'  OK 无缺失值')

# 7. 检查方差（寻找常数特征）
print(f'\n【7. 方差检查】')
variances = df.var()
constant_features = variances[variances == 0].index.tolist()
if constant_features:
    print(f'  WARNING 发现 {len(constant_features)} 个常数特征(方差=0):')
    for feat in constant_features[:10]:
        print(f'    - {feat}')
else:
    print(f'  OK 无常数特征')

# 8. 检查症状值分布
print(f'\n【8. 症状值分布检查（前5个症状）】')
for col in symptom_t[:5]:
    print(f'  {col}:')
    print(f'    值分布: {df[col].value_counts().sort_index().to_dict()}')

print('\n' + '=' * 60)
print('检查完成')
print('=' * 60)
