# Causality-NPC 因果效应估计模块

鼻咽癌中西医结合诊疗因果效应估计模块，基于因果发现的图结构量化因果关系强度。

## 功能概述

本模块实现了完整的因果效应估计流程：

1. **加载发现输出** - 读取因果发现的图、边列表、数据矩阵
2. **边过滤** - 根据相关性和样本量过滤边
3. **调整集识别** - 使用后门准则识别混淆变量
4. **效应估计** - 使用OVR逻辑回归+Bootstrap估计因果效应
5. **结果保存** - 保存CSV摘要和PKL完整结果
6. **报告生成** - 生成Markdown格式分析报告

## 目录结构

```
src/causal_effect/
├── __init__.py
├── pipeline.py               # 主流程入口
├── data/                     # 数据处理子模块
│   ├── loader.py            # 加载发现输出
│   └── edge_filter.py       # 边过滤
├── estimation/               # 估计方法子模块
│   └── logistic_ovr.py      # OVR逻辑回归
└── output/                   # 输出管理子模块
    ├── saver.py             # 结果保存
    └── reporter.py          # 报告生成
```

## 快速开始

### 运行因果效应估计

```bash
# 使用默认配置（需先运行因果发现）
python experiments/run_effect.py

# 使用指定配置文件
python experiments/run_effect.py --config config/base.yaml

# 使用实验配置
python experiments/run_effect.py --config config/experiments/exp_freq_0.15.yaml
```

### Python代码调用

```python
from src.causal_discovery.config import NPCConfig
from src.causal_effect.pipeline import run_causal_effect

# 加载配置
config = NPCConfig.from_yaml("config/base.yaml")

# 运行因果效应估计
results = run_causal_effect(config)

# 查看结果
print(f"估计了 {len(results)} 条边的因果效应")
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

effect:
  # 质量过滤
  min_correlation: 0.0        # 最小相关系数
  min_sample_size: 20         # 最小样本量

  # 估计方法
  method: "logistic_ovr"      # 固定为OVR逻辑回归

  # Bootstrap参数
  bootstrap:
    enable: true              # 启用Bootstrap
    n_iterations: 50          # 迭代次数
    confidence_level: 0.95    # 置信水平
```

### 实验配置

在 `config/experiments/` 目录创建实验配置：

```yaml
# config/experiments/my_exp.yaml
experiment:
  name: "my_exp"

effect:
  min_correlation: 0.1        # 覆盖：最小相关系数
  bootstrap:
    n_iterations: 100         # 覆盖：更多Bootstrap迭代
```

### 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| min_correlation | 最小相关系数阈值 | 0.0 |
| min_sample_size | 最小样本量 | 20 |
| method | 估计方法（当前仅支持logistic_ovr） | logistic_ovr |
| bootstrap.enable | 是否启用Bootstrap | true |
| bootstrap.n_iterations | Bootstrap迭代次数 | 50 |
| bootstrap.confidence_level | 置信水平 | 0.95 |

## 输出说明

### 输出目录结构

```
outputs/{实验名称}/causal_effects/
├── estimates/
│   └── estimated_effects.csv     # 结果摘要
├── models/
│   └── causal_estimates.pkl      # 完整结果对象
└── report.md                      # 分析报告
```

### 输出文件说明

| 文件 | 说明 |
|------|------|
| estimated_effects.csv | 结果摘要，每行一条边的估计结果 |
| causal_estimates.pkl | 完整的Python对象，包含所有估计详情 |
| report.md | 包含统计信息、Top效应、失败列表的分析报告 |

## 估计方法

### OVR逻辑回归

- **二分类结果**：标准逻辑回归，返回log-odds尺度的ATE
- **多分类结果**：One-vs-Rest策略，返回加权平均ATE
- **置信区间**：Bootstrap百分位法

### 调整集识别

使用直接后门准则：在DAG中，treatment的父节点即为最小调整集。

## 算法切换

如需使用其他估计方法（如倾向得分匹配），只需修改两个位置：

### 1. 创建新的估计函数

```python
# src/causal_effect/estimation/psm.py
import pandas as pd
import networkx as nx
from typing import Dict

def estimate_psm(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    source: str,
    target: str,
    **kwargs
) -> Dict:
    """使用倾向得分匹配估计因果效应"""
    # PSM实现
    # ...

    return {
        'treatment': source,
        'outcome': target,
        'method': 'psm',
        'ate': ate_value,
        'ci_lower': ci_lower_value,
        'ci_upper': ci_upper_value,
        # ...
    }
```

### 2. 修改主流程

```python
# src/causal_effect/pipeline.py

# 改导入
# from .estimation.logistic_ovr import estimate_logistic_ovr
from .estimation.psm import estimate_psm

# 改调用
# result = estimate_logistic_ovr(...)
result = estimate_psm(...)
```

旧算法保留在代码库中，未被导入即不生效。

## 模块说明

### 数据处理模块 (data/)

| 模块/函数 | 功能 | 输入 | 输出 |
|-----------|------|------|------|
| DiscoveryLoader | 加载因果发现输出 | 发现输出目录 | 图、边、数据 |
| filter_edges | 过滤边 | 边列表、数据、图 | 过滤后的边信息 |

### 估计模块 (estimation/)

| 函数 | 功能 | 输入 | 输出 |
|------|------|------|------|
| estimate_logistic_ovr | OVR逻辑回归估计 | 数据、图、源、目标 | 估计结果字典 |

### 输出管理模块 (output/)

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| EffectSaver | 保存结果文件 | 结果列表 | CSV、PKL文件 |
| EffectReporter | 生成Markdown报告 | 结果DataFrame | 报告字符串 |

## 示例

### 示例1：查看估计结果

```python
import pandas as pd
import pickle

# 读取CSV摘要
df = pd.read_csv('outputs/default/causal_effects/estimates/estimated_effects.csv')
print(df.head())

# 读取完整结果
with open('outputs/default/causal_effects/models/causal_estimates.pkl', 'rb') as f:
    results = pickle.load(f)

# 查看第一条结果的详细信息
print(results[0])
```

### 示例2：筛选正效应

```python
# 筛选正效应（ate > 0）
positive_effects = df[df['ate'] > 0].sort_values('ate', ascending=False)
print(positive_effects.head(10))

# 筛选负效应（ate < 0）
negative_effects = df[df['ate'] < 0].sort_values('ate')
print(negative_effects.head(10))
```

### 示例3：Python脚本运行

```python
from src.causal_discovery.config import NPCConfig
from src.causal_effect.pipeline import run_causal_effect

# 加载配置
config = NPCConfig.from_yaml("config/base.yaml")

# 修改参数
config.effect.min_correlation = 0.1  # 提高相关性阈值
config.effect.bootstrap.n_iterations = 100  # 更多Bootstrap迭代

# 运行
results = run_causal_effect(config)
```

## 注意事项

1. **依赖顺序**：必须先运行因果发现，再运行因果效应估计
2. **运行时间**：Bootstrap会多次拟合模型，大图可能需要较长时间
3. **收敛警告**：某些边可能因数据稀疏导致收敛失败，这是正常现象
4. **输出目录**：不同实验的结果会保存在不同目录，互不干扰

## 常见问题

### Q: 如何修改相关性阈值？

A: 修改配置文件中的 `effect.min_correlation` 值。

### Q: 如何调整Bootstrap迭代次数？

A: 修改 `effect.bootstrap.n_iterations` 参数，越多越稳定但越慢。

### Q: 输出目录在哪里？

A: 默认在 `outputs/default/causal_effects/`，实验配置在 `outputs/{实验名}/causal_effects/`。

### Q: 如何切换到其他估计方法？

A: 参考本文档的"算法切换"章节，创建新函数并修改导入。

### Q: ATE是什么尺度？

A: 对于二分类结果，ATE是log-odds尺度；对于多分类结果，是加权平均的log-odds。
