# Causality-NPC 因果推荐模块

鼻咽癌中西医结合诊疗因果推荐模块，基于因果发现的图结构和因果效应的ATE估计，为患者推荐中药方剂。

## 功能概述

本模块实现了完整的因果推荐流程：

1. **加载上游输出** - 读取因果发现的图和因果效应的ATE估计
2. **患者编码** - 将患者信息映射到图变量
3. **因果传播** - 沿因果图线性传播激活值
4. **推荐生成** - 根据药物节点激活值进行推荐
5. **路径解释** - 生成因果路径解释
6. **结果保存** - 保存JSON结果和Markdown报告

## 目录结构

```
src/causal_recommendation/
├── __init__.py
├── pipeline.py               # 主流程入口
├── data/                     # 数据处理子模块
│   ├── loader.py            # 加载图和ATE
│   └── patient_encoder.py   # 患者信息编码
├── recommendation/           # 推荐算法子模块
│   └── propagation.py       # 因果传播算法
└── output/                   # 输出管理子模块
    ├── saver.py             # 结果保存
    └── reporter.py          # 报告生成
```

## 快速开始

### 运行因果推荐

```bash
# 使用默认配置（需先运行因果发现和因果效应）
python experiments/run_recommendation.py

# 使用指定配置文件
python experiments/run_recommendation.py --config config/base.yaml

# 使用实验配置
python experiments/run_recommendation.py --config config/experiments/exp_freq_0.15.yaml
```

### Python代码调用

```python
from src.causal_discovery.config import NPCConfig
from src.causal_recommendation.pipeline import run_causal_recommendation

# 加载配置
config = NPCConfig.from_yaml("config/base.yaml")

# 准备患者数据
patients = [
    {
        'gender': '女',
        'age': 58,
        '乏力': 1.0,
        '畏寒': 1.0,
        '阴虚血瘀证': 1.0,
    }
]

# 运行因果推荐
result = run_causal_recommendation(config, patients)

# 查看结果
print(f"处理了 {len(result['results'])} 个患者")
```

## 配置说明

### 配置文件结构

配置文件使用YAML格式，位于 `config/` 目录：

```yaml
# config/base.yaml
project:
  name: "Causality-NPC"
  version: "2.0"

paths:
  raw_data: "Data/raw/npc_full_with_symptoms.csv"
  outputs_root: "outputs"

recommendation:
  threshold_positive: 0.05    # 推荐阈值（正向）
  threshold_negative: -0.05   # 警告阈值（负向）
  top_k: 5                    # 返回Top-K推荐
  max_paths: 5                # 最大解释路径数
```

### 实验配置

在 `config/experiments/` 目录创建实验配置：

```yaml
# config/experiments/my_exp.yaml
experiment:
  name: "my_exp"

recommendation:
  threshold_positive: 0.10    # 覆盖：提高推荐阈值
  top_k: 10                   # 覆盖：返回更多推荐
```

### 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| threshold_positive | 推荐阈值（正向），得分>=此值推荐 | 0.05 |
| threshold_negative | 警告阈值（负向），得分<=此值不推荐 | -0.05 |
| top_k | 返回前k个推荐药物 | 5 |
| max_paths | 每个药物的最大解释路径数 | 5 |

## 输出说明

### 输出目录结构

```
outputs/{实验名称}/causal_recommendation/
├── recommendations/
│   └── patient_recommendations.json  # 推荐结果
└── report.md                          # 推荐报告
```

### 输出文件说明

| 文件 | 说明 |
|------|------|
| patient_recommendations.json | 完整的推荐结果，包含所有患者的推荐详情 |
| report.md | 包含统计信息、推荐结果的Markdown报告 |

### 推荐结果格式

```json
{
  "generated_at": "2026-03-25 16:40:33",
  "total_cases": 4,
  "cases": [
    {
      "case_id": 1,
      "patient_info": {
        "gender": "女",
        "age": 58,
        "乏力": 1.0
      },
      "recommended": {
        "med_黄芪_t": 213.77,
        "med_党参_t": 1.23
      },
      "not_recommended": {},
      "neutral": ["med_人参片_t", ...],
      "explanations": {
        "med_黄芪_t": [
          {
            "path": ["gender_encoded", "med_黄芪_t"],
            "contribution": 1.05,
            "edges": [
              {"source": "gender_encoded", "target": "med_黄芪_t", "ate": 1.05}
            ]
          }
        ]
      },
      "mapping_info": {
        "mapped_vars": {"gender_encoded": 1, "乏力_t": 1.0},
        "invalid_vars": [],
        "unmapped_keys": []
      }
    }
  ]
}
```

## 推荐算法

### 因果传播算法

1. **输入编码**：将患者信息映射到图变量
2. **激活传播**：沿因果图按拓扑顺序传播激活值
3. **药物得分**：收集药物节点的激活值
4. **结果分类**：根据阈值分类为推荐/不推荐/中性

### 患者信息映射

| 输入类型 | 映射规则 | 示例 |
|----------|----------|------|
| 症状 | `{症状名}_t` | `'乏力'` → `'乏力_t'` |
| 诊断 | `diagnosis_{诊断名}_t` | `'贫血'` → `'diagnosis_贫血_t'` |
| 性别 | `gender_encoded` | `'女'` → `1`, `'男'` → `0` |

### 路径解释生成

- 使用NetworkX查找所有简单路径（最大长度5）
- 计算路径贡献：路径上所有边的ATE乘积
- 返回贡献最大的前N条路径

## 算法切换

如需使用其他推荐算法（如协同过滤），只需修改两个位置：

### 1. 创建新的推荐函数

```python
# src/causal_recommendation/recommendation/collaborative_filtering.py
import networkx as nx
from typing import Dict

def collaborative_filtering_recommend(
    graph: nx.DiGraph,
    ate_dict: Dict,
    patient_info: Dict,
    **kwargs
) -> Dict:
    """使用协同过滤进行药物推荐"""
    # CF实现
    # ...

    return {
        'recommended': {...},
        'not_recommended': {...},
        'neutral': [...],
        'explanations': {...}
    }
```

### 2. 修改主流程

```python
# src/causal_recommendation/pipeline.py

# 改导入
# from .recommendation.propagation import causal_propagation_recommend
from .recommendation.collaborative_filtering import collaborative_filtering_recommend

# 改调用
# result = causal_propagation_recommend(...)
result = collaborative_filtering_recommend(...)
```

## 模块说明

### 数据处理模块 (data/)

| 模块/类 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| RecommendationLoader | 加载图和ATE | 配置对象 | 图、ATE字典 |
| extract_mapping_rules | 提取映射规则 | 节点集合 | 映射规则字典 |
| map_patient_to_graph | 映射患者信息 | 患者信息、节点、规则 | 映射结果 |

### 推荐模块 (recommendation/)

| 函数 | 功能 | 输入 | 输出 |
|------|------|------|------|
| causal_propagation_recommend | 因果传播推荐 | 图、ATE、患者信息 | 推荐结果 |
| generate_explanations | 生成路径解释 | 图、ATE、患者、药物 | 解释字典 |

### 输出管理模块 (output/)

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| RecommendationSaver | 保存结果文件 | 结果列表 | JSON文件 |
| RecommendationReporter | 生成Markdown报告 | 结果列表 | 报告字符串 |

## 示例

### 示例1：单个患者推荐

```python
from src.causal_discovery.config import NPCConfig
from src.causal_recommendation.pipeline import run_causal_recommendation

config = NPCConfig.from_yaml("config/base.yaml")
patient = {
    'gender': '女',
    'age': 58,
    '乏力': 1.0,
    '畏寒': 1.0,
    '阴虚血瘀证': 1.0,
}

result = run_causal_recommendation(config, [patient])
recommendation = result['results'][0]

print("推荐药物:", list(recommendation['recommended'].keys()))
```

### 示例2：查看路径解释

```python
recommendation = result['results'][0]
explanations = recommendation['explanations']

for med, paths in explanations.items():
    print(f"\n{med}:")
    for i, path_info in enumerate(paths, 1):
        print(f"  路径{i}: {' -> '.join(path_info['path'])}")
        print(f"  贡献: {path_info['contribution']:.4f}")
```

### 示例3：修改推荐阈值

```python
from src.causal_discovery.config import NPCConfig

config = NPCConfig.from_yaml("config/base.yaml")

# 修改阈值
config.recommendation.threshold_positive = 0.10  # 提高推荐阈值
config.recommendation.top_k = 10  # 返回更多推荐

result = run_causal_recommendation(config, patients)
```

## 注意事项

1. **依赖顺序**：必须先运行因果发现和因果效应，再运行因果推荐
2. **输入路径**：确保因果发现和因果效应的输出文件存在
3. **患者信息**：症状值为0表示无此症状，会被跳过
4. **未映射键**：某些患者信息可能无法映射到图节点，会记录在unmapped_keys中

## 常见问题

### Q: 如何修改推荐阈值？

A: 修改配置文件中的 `recommendation.threshold_positive` 值，或在代码中修改 `config.recommendation.threshold_positive`。

### Q: 为什么某些药物没有解释？

A: 可能是从患者输入到该药物没有因果路径，或路径贡献太小（<0.001）。

### Q: 如何添加新的症状或诊断？

A: 症状和诊断是从因果图中自动提取的，确保新的节点命名遵循约定（症状以`_t`结尾，诊断以`diagnosis_`开头）。

### Q: 输出目录在哪里？

A: 默认在 `outputs/default/causal_recommendation/`，实验配置在 `outputs/{实验名}/causal_recommendation/`。

### Q: 如何切换到其他推荐算法？

A: 参考本文档的"算法切换"章节，创建新函数并修改导入。
