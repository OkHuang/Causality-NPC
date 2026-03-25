# Causality-NPC 因果发现模块

鼻咽癌中西医结合诊疗因果发现模块，基于PC算法从时序患者数据中发现因果关系。

## 功能概述

本模块实现了完整的因果发现流程：

1. **数据加载** - 读取原始数据并解析症状JSON
2. **时序对构建** - 构建相邻时间点的配对 (t, t+1)
3. **特征编码** - 对症状、诊断、药物进行编码和频率筛选
4. **数据清洗** - 移除常数特征和高度相关特征
5. **PC算法** - 使用Fisher-Z独立性检验发现因果边
6. **约束应用** - 应用领域知识约束（如未来不能影响过去）
7. **环处理** - 检测并移除图中的环，确保生成DAG
8. **结果保存** - 保存图结构、可视化和报告

## 目录结构

```
src/causal_discovery/
├── __init__.py
├── config.py                 # 统一配置类
├── pipeline.py               # 主流程入口
├── data/                     # 数据处理子模块
│   ├── loader.py            # 数据加载器
│   ├── pair_builder.py      # 时序对构建器
│   └── cleaner.py           # 数据清洗
├── features/                 # 特征工程子模块
│   └── encoder.py           # 特征编码器
├── discovery/                # 因果发现子模块
│   ├── pc.py                # PC算法
│   ├── constraints.py       # 约束管理
│   └── cycle_handler.py     # 环处理
├── output/                   # 输出管理子模块
│   ├── saver.py             # 结果保存
│   └── reporter.py          # 报告生成
└── visualization/            # 可视化子模块
    └── plotter.py           # 图可视化
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行因果发现

```bash
# 使用默认配置
python experiments/run_discovery.py

# 使用指定配置文件
python experiments/run_discovery.py --config config/base.yaml

# 使用实验配置
python experiments/run_discovery.py --config config/experiments/exp_freq_0.15.yaml
```

### Python代码调用

```python
from src.causal_discovery.config import NPCConfig
from src.causal_discovery.pipeline import run_causal_discovery

# 加载配置
config = NPCConfig.from_yaml("config/base.yaml")

# 运行因果发现
G = run_causal_discovery(config)

# 查看结果
print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")
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

discovery:
  # 特征筛选阈值
  symptom_threshold: 0.10      # 症状频率阈值
  medicine_threshold: 0.10     # 药物频率阈值
  diagnosis_threshold: 0.10    # 诊断频率阈值

  # PC算法参数
  alpha: 0.05                  # 显著性水平
  independence_test: "fisherz"  # 独立性检验方法
  depth: -1                    # 搜索深度

  # 处理选项
  apply_constraints: true      # 是否应用约束
  remove_cycles: true          # 是否移除环
```

### 实验配置

在 `config/experiments/` 目录创建实验配置，继承并覆盖基础配置：

```yaml
# config/experiments/exp_freq_0.15.yaml
experiment:
  name: "exp_freq_0.15"

discovery:
  medicine_threshold: 0.15     # 覆盖：药物频率阈值
  alpha: 0.03                   # 覆盖：更严格的显著性水平
```

### 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| symptom_threshold | 症状频率阈值，低于此值的症状将被移除 | 0.10 |
| medicine_threshold | 药物频率阈值 | 0.10 |
| diagnosis_threshold | 诊断频率阈值 | 0.10 |
| alpha | PC算法显著性水平，越小越保守 | 0.05 |
| independence_test | 独立性检验方法 | fisherz |
| apply_constraints | 是否应用领域知识约束 | true |
| remove_cycles | 是否移除图中的环 | true |

## 输出说明

### 输出目录结构

```
outputs/{实验名称}/causal_discovery/
├── data/                     # 处理后的数据
│   ├── pairs.csv            # 时序对数据
│   └── matrix.csv           # 用于PC算法的数据矩阵
├── graph/                    # 图文件
│   ├── dag.pkl              # NetworkX图对象
│   ├── edges.json           # 边列表（JSON格式）
│   └── dag.png              # 可视化图像
└── report.md                # 发现报告
```

### 输出文件说明

| 文件 | 说明 |
|------|------|
| pairs.csv | 构建的时序对数据，包含t时刻和t+1时刻的所有特征 |
| matrix.csv | 用于PC算法的数据矩阵，已清洗和筛选 |
| dag.pkl | NetworkX有向图对象，可用Python加载 |
| edges.json | 因果边列表，JSON格式，便于人工检查 |
| dag.png | 因果网络可视化图像 |
| report.md | 包含统计信息、因果边列表的分析报告 |

## 算法切换

如需使用其他因果发现算法（如FCI），只需修改两个位置：

### 1. 创建新的算法文件

```python
# src/causal_discovery/discovery/fci.py
import pandas as pd
import networkx as nx

def run_fci_algorithm(data_matrix: pd.DataFrame, config, alpha: float = 0.05):
    """运行FCI算法"""
    from causallearn.search.ConstraintBased.FCI import fci

    X = data_matrix.values.astype(float)
    cg = fci(X, alpha=alpha)

    # 转换为NetworkX图
    G = nx.DiGraph()
    G.add_nodes_from(data_matrix.columns)
    # ... 添加边的逻辑 ...

    return G
```

### 2. 修改主流程

```python
# src/causal_discovery/pipeline.py

# 改导入
# from .discovery.pc import run_pc_algorithm
from .discovery.fci import run_fci_algorithm

# 改调用
# G = run_pc_algorithm(data_matrix, node_names, config.discovery.alpha)
G = run_fci_algorithm(data_matrix, config, config.discovery.alpha)
```

## 模块说明

### 数据处理模块 (data/)

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| DataLoader | 加载CSV并解析症状JSON | 原始数据路径 | 处理后的DataFrame |
| PairBuilder | 构建时序对(t, t+1) | 处理后的DataFrame | 时序对DataFrame |
| Cleaner | 清洗数据矩阵 | 编码后的DataFrame | 清洗后的数据矩阵 |

### 特征工程模块 (features/)

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| FeatureEncoder | 编码特征并筛选 | 时序对DataFrame | 编码后的DataFrame、统计信息 |

### 因果发现模块 (discovery/)

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| run_pc_algorithm | 运行PC算法 | 数据矩阵、节点名、alpha | 因果图 |
| ConstraintManager | 应用领域约束 | 因果图 | 约束后的图 |
| remove_cycles | 移除环 | 有环图 | DAG |

### 输出管理模块 (output/)

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| ResultSaver | 保存结果文件 | 图、数据、报告 | 文件保存路径 |
| DiscoveryReporter | 生成Markdown报告 | 图、统计信息 | 报告字符串 |

## 示例

### 示例1：使用不同阈值

```bash
# 创建实验配置 config/experiments/my_exp.yaml
cat > config/experiments/my_exp.yaml << EOF
experiment:
  name: "my_exp"

discovery:
  symptom_threshold: 0.15
  medicine_threshold: 0.20
  diagnosis_threshold: 0.15
  alpha: 0.03
EOF

# 运行实验
python experiments/run_discovery.py --config config/experiments/my_exp.yaml
```

### 示例2：Python脚本运行

```python
from src.causal_discovery.config import NPCConfig
from src.causal_discovery.pipeline import run_causal_discovery
import networkx as nx

# 加载配置
config = NPCConfig.from_yaml("config/base.yaml")

# 修改参数
config.discovery.alpha = 0.01  # 更严格的显著性水平

# 运行
G = run_causal_discovery(config)

# 分析结果
print(f"发现的因果边: {G.number_of_edges()}")

# 保存为其他格式
nx.write_gexf(G, "my_graph.gexf")
```

## 注意事项

1. **数据要求**：原始数据必须包含 `extracted_symptoms` 列（JSON格式的症状数据）
2. **内存使用**：大规模数据集可能需要较多内存
3. **运行时间**：PC算法的时间复杂度较高，大图可能需要较长时间
4. **输出目录**：不同实验的结果会保存在不同目录，互不干扰

## 常见问题

### Q: 如何修改显著性水平？

A: 修改配置文件中的 `discovery.alpha` 值，越小越保守。

### Q: 如何调整特征筛选阈值？

A: 修改 `discovery.symptom_threshold`、`discovery.medicine_threshold` 等参数。

### Q: 输出目录在哪里？

A: 默认在 `outputs/default/causal_discovery/`，实验配置在 `outputs/{实验名}/causal_discovery/`。

### Q: 如何禁用某些约束？

A: 设置 `discovery.apply_constraints: false`，或在 `src/causal_discovery/discovery/constraints.py` 中修改约束定义。
