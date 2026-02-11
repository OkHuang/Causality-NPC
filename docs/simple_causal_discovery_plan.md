# 简易版因果发现计划

**版本**: v1.0-simple
**创建日期**: 2026-02-10
**目标**: 构建鼻咽癌患者诊疗数据的简易因果网络，不使用Bootstrap稳定性选择

---

## 1. 项目概述

本计划旨在从963条鼻咽癌患者的中西医结合诊疗数据中，构建一个**时序因果网络**，发现"症状→诊断→药物→后续症状"的因果链条。

### 与完整版的区别

| 特性 | 完整版 | 简易版 |
|------|--------|--------|
| Bootstrap稳定性选择 | ✓ (1000次) | ✗ |
| 时间对齐 | ✓ (相对时间轴) | ✗ (直接使用原始时间) |
| 时间窗口归并 | ✓ | ✗ |
| 节点降维 | ✓ (药物归类、症状聚类) | 简化版 (基于频率删减) |
| 代码复杂度 | 高 | 低 |

---

## 2. 数据结构分析

### 原始数据字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `patient_id` | int | 患者唯一标识 |
| `gender` | str | 性别 (女/男) |
| `age` | int | 年龄 |
| `time` | date | 就诊时间 |
| `extracted_symptoms` | JSON字符串 | 症状列表，格式: `[{"name": "乏力", "severity": 1, "label": "轻度"}]` |
| `chinese_diagnosis` | str | 中医诊断证型 |
| `western_diagnosis` | str | 西医诊断 |
| `chinese_medicines` | str | 中药处方，空格分隔 |

### extracted_symptoms 示例

```json
[
  {"name": "乏力", "severity": 1, "label": "轻度"},
  {"name": "畏寒", "severity": 1, "label": "轻度"},
  {"name": "头痛", "severity": 0, "label": "无"},
  ...
]
```

---

## 3. 因果网络结构设计

### 3.1 整体架构

```
时刻 t                                    时刻 t+1
┌─────────────────────┐                  ┌─────────────────────┐
│                     │                  │                     │
│  静态特征           │                  │  状态特征           │
│  - gender           │                  │  - symptoms_t+1     │
│  - age              │      ────────>   │  - chinese_dx_t+1   │
│                     │                  │  - western_dx_t+1   │
│  动态特征           │                  │                     │
│  - symptoms_t       │                  │                     │
│  - chinese_dx_t     │                  │                     │
│  - western_dx_t     │                  │                     │
│                     │                  │                     │
└─────────────────────┘                  └─────────────────────┘
            │
            │ 决定
            ↓
┌─────────────────────┐
│                     │
│  处理变量 (t)       │
│  - chinese_medicines│
│                     │
└─────────────────────┘
```

### 3.2 因果路径假设

根据中医诊疗逻辑和时序关系，我们假设以下因果路径：

**路径1: 症状→诊断→药物→后续症状**
```
symptoms_t → chinese_diagnosis_t → chinese_medicines_t → symptoms_t+1
symptoms_t → western_diagnosis_t → chinese_medicines_t → symptoms_t+1
```

**路径2: 人口学特征影响一切**
```
gender, age → [所有t时刻变量]
```

**路径3: 诊断内部的自回归**
```
chinese_diagnosis_t → chinese_diagnosis_t+1
western_diagnosis_t → western_diagnosis_t+1
```

**路径4: 症状内部的自回归**
```
symptoms_t → symptoms_t+1
```

### 3.3 禁止的边（硬约束）

根据时序因果原则，以下边**不允许**出现：

1. **未来影响过去**: t+1时刻的任何变量不能指向t时刻的变量
2. **药物不直接影响当前诊断**: `chinese_medicines_t → chinese_diagnosis_t` (医生是根据诊断开药的)
3. **人口学特征时不变**: `gender_t → gender_t+1` 等没有意义

---

## 4. 实现步骤

### 步骤1: 数据加载与解析

**输入**: `D:\WorkProject\Causality-NPC\Data\raw\npc_full_with_symptoms.csv`

**任务**:
1. 读取CSV文件
2. 解析 `extracted_symptoms` JSON字符串
3. 将症状展开为多列 (每个症状一列，值为severity)

**输出示例**:

| patient_id | time | gender | age | 乏力 | 畏寒 | 头痛 | chinese_diagnosis | ... | chinese_medicines |
|------------|------|--------|-----|------|------|------|-------------------|-----|-------------------|
| 29121      | 2023-10-02 | 女 | 58 | 1 | 1 | 1 | 痰瘀互结证 | ... | 郁金 肉桂 瓜蒌皮 |
| 29121      | 2024-03-21 | 女 | 58 | 1 | 1 | 1 | 痰瘀互结证 | ... | 熟地黄 生山萸肉 |

**代码伪代码**:
```python
import pandas as pd
import json

# 读取数据
df = pd.read_csv("Data/raw/npc_full_with_symptoms.csv")

# 解析extracted_symptoms
def parse_symptoms(json_str):
    symptoms = json.loads(json_str)
    return {s['name']: s['severity'] for s in symptoms}

# 展开症状列
symptom_df = df['extracted_symptoms'].apply(parse_symptoms).apply(pd.Series)
df = pd.concat([df, symptom_df], axis=1)
```

---

### 步骤2: 构建时序对 (t → t+1)

**任务**:
1. 按 `patient_id` 分组
2. 对每个患者，按 `time` 排序
3. 构建相邻时间点的配对：(visit_i, visit_{i+1})

**输出结构**:

| patient_id | pair_id | gender | age_t | 乏力_t | 畏寒_t | chinese_dx_t | chinese_meds_t | 乏力_t1 | 畏寒_t1 | chinese_dx_t1 |
|------------|---------|--------|-------|--------|--------|--------------|----------------|---------|---------|---------------|
| 29121      | 0       | 女     | 58    | 1      | 1      | 痰瘀互结证   | 郁金 肉桂      | 1       | 1       | 痰瘀互结证    |
| 29121      | 1       | 女     | 58    | 1      | 1      | 痰瘀互结证   | 熟地黄 生山萸肉| 1       | 1       | 痰瘀互结证    |

**注意**:
- gender 和 age 使用 t 时刻的值
- 如果患者只有一次就诊，则无法构建时序对，需排除

**代码伪代码**:
```python
# 按患者分组并排序
df_sorted = df.sort_values(['patient_id', 'time'])

# 构建时序对
pairs = []
for pid, group in df_sorted.groupby('patient_id'):
    if len(group) < 2:
        continue  # 跳过只有一次就诊的患者
    for i in range(len(group) - 1):
        t = group.iloc[i]
        t1 = group.iloc[i + 1]
        pairs.append({
            'patient_id': pid,
            'gender': t['gender'],
            'age_t': t['age'],
            # ... t时刻特征
            'chinese_meds_t': t['chinese_medicines'],
            # ... t+1时刻特征
        })

df_pairs = pd.DataFrame(pairs)
```

---

### 步骤3: 特征编码

#### 3.1 二值化/数值化

| 变量类型 | 编码方式 | 示例 |
|----------|----------|------|
| 症状 (severity) | 保持原值 (0/1/2) | 乏力_t: 0/1/2 |
| 性别 | 二值化 | gender: 0=女, 1=男 |
| 年龄 | 从20开始分箱，bin=5 | "[50-55)" |
| 中医诊断 | 多热编码 | 痰瘀互结证: 1, 气虚证: 1 |
| 西医诊断 | 多热编码 | 高血压: 1, 糖尿病: 0 |
| 中药药物 | 多热编码 | 黄芪: 1, 党参: 1 |

#### 3.2 处理缺失值

- 对于缺失的症状，默认 `severity = 0` (无症状)
- 对于缺失的诊断，认为该诊断不存在

**代码伪代码**:
```python
# 诊断多热编码
from sklearn.preprocessing import MultiLabelBinarizer

# 解析诊断 (空格分隔)
df_pairs['chinese_dx_list'] = df_pairs['chinese_dx_t'].str.split(' ')
mlb = MultiLabelBinarizer()
dx_encoded = mlb.fit_transform(df_pairs['chinese_dx_list'])
dx_cols = [f'chinese_dx_{c}' for c in mlb.classes_]
df_pairs[dx_cols] = dx_encoded

# 药物多热编码
df_pairs['meds_list'] = df_pairs['chinese_meds_t'].str.split(' ')
mlb_meds = MultiLabelBinarizer()
meds_encoded = mlb_meds.fit_transform(df_pairs['meds_list'])
meds_cols = [f'med_{c}' for c in mlb_meds.classes_]
df_pairs[meds_cols] = meds_encoded
```

---

### 步骤4: 节点筛选 (基于频率)

**目的**: 如果节点太多（症状/药物/诊断种类过多），根据出现频率删减低频节点。

**筛选策略**:

1. **统计频率**: 计算每个节点（变量）在数据中的非零比例
2. **设定阈值**: 保留出现频率 > threshold 的节点
   - 建议阈值: 5% (至少在5%的时序对中出现)
3. **分层筛选**:
   - 症状节点: 保留频率 > 3% 的症状
   - 药物节点: 保留频率 > 5% 的药物
   - 诊断节点: 保留频率 > 3% 的诊断

**代码伪代码**:
```python
# 计算频率
symptom_freq = (symptom_cols_df > 0).mean()
med_freq = (med_cols_df > 0).mean()
dx_freq = (dx_cols_df > 0).mean()

# 筛选
selected_symptoms = symptom_freq[symptom_freq > 0.03].index.tolist()
selected_meds = med_freq[med_freq > 0.05].index.tolist()
selected_dx = dx_freq[dx_freq > 0.03].index.tolist()

print(f"原始症状数: {len(symptom_freq)}, 筛选后: {len(selected_symptoms)}")
print(f"原始药物数: {len(med_freq)}, 筛选后: {len(selected_meds)}")
print(f"原始诊断数: {len(dx_freq)}, 筛选后: {len(selected_dx)}")
```

**预期结果**:
- 症状: 从60+ → 20-30个
- 药物: 从100+ → 20-40个
- 诊断: 保持原样（通常诊断种类不会太多）

---

### 步骤5: 构建因果发现数据集

**输入**: 筛选后的特征表

**输出**: 用于因果发现算法的数据矩阵

**列组织**:
```
[静态特征_t] + [症状_t] + [诊断_t] + [药物_t] + [症状_t1] + [诊断_t1]
```

**示例**:
```
gender, age_t, 乏力_t, 畏寒_t, ..., chinese_dx_痰瘀_t, med_黄芪_t, ..., 乏力_t1, 畏寒_t1, ..., chinese_dx_痰瘀_t1
```

**变量分组** (用于后续约束):
```python
variable_groups = {
    'static': ['gender', 'age_t'],
    'symptoms_t': [f'{s}_t' for s in selected_symptoms],
    'diagnosis_t': [f'{d}_t' for d in selected_dx],
    'medicines_t': [f'{m}_t' for m in selected_meds],
    'symptoms_t1': [f'{s}_t1' for s in selected_symptoms],
    'diagnosis_t1': [f'{d}_t1' for d in selected_dx],
}
```

---

### 步骤6: 定义因果约束

**硬约束** (禁止的边):

```python
forbidden_edges = [
    # 未来不能影响过去
    ('*_t1', '*_t'),

    # 药物不能反向影响当前诊断
    ('medicines_t', 'diagnosis_t'),

    # 诊断不能反向影响当前症状
    ('diagnosis_t', 'symptoms_t'),

    # 人口学特征不能被影响
    ('*', 'static'),

    # t+1时刻的变量不能指向t时刻的药物
    ('*_t1', 'medicines_t'),
]
```

**允许的边** (基于时序和领域知识):

```python
allowed_edges = [
    # 静态特征 → t时刻所有变量
    ('static', '*_t'),

    # t时刻症状 → t时刻诊断 (医生根据症状诊断)
    ('symptoms_t', 'diagnosis_t'),

    # t时刻诊断 → t时刻药物 (根据诊断开药)
    ('diagnosis_t', 'medicines_t'),

    # t时刻药物 → t+1时刻症状 (药物治疗效果)
    ('medicines_t', 'symptoms_t1'),

    # t时刻症状 → t+1时刻症状 (症状自然演变)
    ('symptoms_t', 'symptoms_t1'),

    # t时刻诊断 → t+1时刻诊断 (证型演变)
    ('diagnosis_t', 'diagnosis_t1'),

    # t时刻药物 → t+1时刻诊断 (药物影响证型)
    ('medicines_t', 'diagnosis_t1'),
]
```

---

### 步骤7: 运行因果发现算法

**算法选择**: PC算法 (基于约束的因果发现)

**库**: `causal-learn` (Python)

**参数设置**:
```python
from causallearn.search.ConstraintBased.PC import pc

alpha = 0.05  # 显著性水平
indep_test = 'fisherz'  # 独立性检验方法 (适用于连续变量)
# 或
indep_test = 'chisq'  # 适用于离散变量
```

**代码框架**:
```python
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

# 准备数据矩阵
X = df_pairs[all_columns].values  # numpy array

# 运行PC算法
cg = pc(X, alpha=0.05, indep_test=fisherz)

# 获取图的边
edges = cg.get_graph_edges()

# 转换为NetworkX图
import networkx as nx
G = nx.DiGraph()
G.add_nodes_from(all_columns)
for edge in edges:
    G.add_edge(edge[0], edge[1])
```

---

### 步骤8: 应用约束并剪枝

**任务**: 移除违反硬约束的边

**代码框架**:
```python
def apply_constraints(G, forbidden_edges):
    """
    移除违反约束的边
    """
    edges_to_remove = []

    for u, v in G.edges():
        # 检查是否违反任何禁止规则
        for pattern in forbidden_edges:
            if matches_pattern(u, v, pattern):
                edges_to_remove.append((u, v))
                break

    G.remove_edges_from(edges_to_remove)
    print(f"移除了 {len(edges_to_remove)} 条违反约束的边")
    return G

def matches_pattern(u, v, pattern):
    """
    检查边 (u, v) 是否匹配禁止模式
    支持通配符 *
    """
    src_pattern, dst_pattern = pattern
    return (src_pattern == '*' or u.endswith(src_pattern.rstrip('*'))) and \
           (dst_pattern == '*' or v.endswith(dst_pattern.rstrip('*')))
```

---

### 步骤9: 可视化因果图

**工具**: `networkx` + `matplotlib` 或 `graphviz`

**可视化策略**:

1. **分层布局**:
   - Layer 1: 静态特征 (gender, age)
   - Layer 2: t时刻症状
   - Layer 3: t时刻诊断
   - Layer 4: t时刻药物
   - Layer 5: t+1时刻症状
   - Layer 6: t+1时刻诊断

2. **节点颜色编码**:
   - 静态特征: 灰色
   - 症状: 红色系
   - 诊断: 蓝色系
   - 药物: 绿色系

3. **边的粗细**: 可以用边的权重表示置信度 (暂时没有Bootstrap，所以统一粗细)

**代码框架**:
```python
import matplotlib.pyplot as plt
import networkx as nx

# 设置节点位置 (分层布局)
pos = {}
layer_height = 1000

# Layer 1: 静态特征
for i, node in enumerate(variable_groups['static']):
    pos[node] = (i * 50, 5 * layer_height)

# Layer 2: 症状_t
for i, node in enumerate(variable_groups['symptoms_t']):
    pos[node] = (i * 30, 4 * layer_height)

# ... 以此类推

# 绘制
plt.figure(figsize=(20, 12))
node_colors = [get_node_color(node) for node in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        node_size=500, font_size=8, arrowsize=20)
plt.title("鼻咽癌中西医结合诊疗因果网络")
plt.savefig("outputs/graphs/simple_causal_dag.png", dpi=300, bbox_inches='tight')
```

---

### 步骤10: 路径提取与分析

**任务**: 提取特定的因果路径，生成可解释的报告

**重点路径**:

1. **药物疗效路径**: `诊断_t → 药物_t → 症状_t1`
2. **症状演变路径**: `症状_t → 症状_t1`
3. **证型演变路径**: `诊断_t → 诊断_t1`

**代码框架**:
```python
def extract_paths(G, pattern, max_length=3):
    """
    提取符合特定模式的路径
    pattern: 例如 'diagnosis_t -> medicines_t -> symptoms_t1'
    """
    node_types = get_node_types(G)  # 获取每个节点的类型
    paths = []

    for source in G.nodes():
        if node_types[source] != pattern[0]:
            continue
        # 深度优先搜索
        for target in G.nodes():
            if node_types[target] != pattern[-1]:
                continue
            try:
                path = nx.shortest_path(G, source, target)
                if len(path) <= max_length:
                    if matches_path_pattern(path, pattern, node_types):
                        paths.append(path)
            except nx.NetworkXNoPath:
                continue

    return paths

# 示例：提取"诊断→药物→症状改善"路径
medication_paths = extract_paths(G, ['diagnosis_t', 'medicines_t', 'symptoms_t1'])

print(f"发现 {len(medication_paths)} 条药物治疗路径")
for path in medication_paths[:10]:
    print(" → ".join(path))
```

---

## 5. 预期输出

### 5.1 数据文件

| 文件 | 说明 |
|------|------|
| `outputs/data/pairs_data.csv` | 时序对数据 |
| `outputs/data/selected_features.csv` | 筛选后的特征表 |
| `outputs/data/edge_list.csv` | 因果边列表 |

### 5.2 图表文件

| 文件 | 说明 |
|------|------|
| `outputs/graphs/simple_causal_dag.png` | 因果有向无环图 |
| `outputs/graphs/medication_paths.png` | 药物疗效路径子图 |
| `outputs/graphs/symptom_network.png` | 症状网络子图 |

### 5.3 报告文件

| 文件 | 说明 |
|------|------|
| `outputs/reports/simple_causal_discovery_report.md` | 分析报告 |

---

## 6. 报告模板

```markdown
# 简易版因果发现报告

## 1. 数据概览

- 总患者数: XXX
- 总时序对数: XXX
- 平均每人就诊次数: X.XX

## 2. 特征统计

- 选中症状数: XX (原始XX个)
- 选中药物数: XX (原始XX个)
- 选中诊断数: XX (原始XX个)

## 3. 因果网络概览

- 节点总数: XXX
- 边总数: XXX

## 4. 关键发现

### 4.1 药物疗效路径

发现以下"诊断→药物→症状改善"路径:

1. 气虚证 → (黄芪, 党参) → 乏力改善
2. 瘀血证 → (丹参, 川芎) → 疼痛减轻
...

### 4.2 症状演变路径

发现以下症状之间的因果关系:

- 乏力 → 失眠 (OR: X.XX)
...

### 4.3 证型演变路径

- 气虚证 → 阴虚证 (频率: XX%)

## 5. 局限性

1. 未使用Bootstrap稳定性选择，结果可能存在假阳性
2. 样本量相对较小
3. 未考虑时间间隔的影响

## 6. 后续改进

- [ ] 添加Bootstrap稳定性选择
- [ ] 考虑时间间隔权重
- [ ] 引入专家知识约束
- [ ] 与专家验证结果
```

---

## 7. 依赖库

```txt
# 核心依赖
pandas>=2.0.0
numpy>=1.24.0
networkx>=3.0
causal-learn>=0.1.3
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
graphviz>=0.20.0  # 可选，用于更好的可视化

# 可选依赖
pygraphviz>=1.11  # 更好的图布局
```

---

## 8. 实现时间估计

| 步骤 | 预计时间 |
|------|----------|
| 步骤1-2: 数据加载与时序对构建 | 2-3小时 |
| 步骤3-4: 特征编码与筛选 | 2-3小时 |
| 步骤5-6: 数据集构建与约束定义 | 1-2小时 |
| 步骤7-8: 因果发现与约束应用 | 2-3小时 |
| 步骤9-10: 可视化与路径提取 | 3-4小时 |
| **总计** | **10-15小时** |

---

## 9. 下一步行动

1. 审阅本计划，确认是否符合需求
2. 如有修改意见，更新计划
3. 开始编码实现
4. 先在小样本上测试流程
5. 应用到全量数据
6. 生成报告并与专家验证

---

**计划文档版本**: v1.0
**最后更新**: 2026-02-10
**状态**: 待审核
