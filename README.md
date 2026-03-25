# Causality-NPC

鼻咽癌中西医结合诊疗的因果推断系统。

## 项目概述

Causality-NPC 是一个完整的因果推断工作流系统，专为鼻咽癌（NPC）诊疗设计。系统从原始医疗数据出发，通过因果发现算法构建因果图，量化因果效应，并基于因果传播为患者提供个性化中药推荐。

### 核心功能

- **因果发现**: 使用 PC 算法从时序患者数据中发现因果关系，构建有向无环图（DAG）
- **因果效应估计**: 基于 OVR 逻辑回归量化每条因果边的平均处理效应（ATE）
- **因果推荐**: 通过因果传播算法为患者推荐中药方剂，并提供可解释的因果路径
- **推荐评估**: 评估推荐系统的准确性，支持阈值优化

### 技术栈

- **因果推断**: causal-learn, dowhy, CausalML
- **数据分析**: pandas, numpy, scipy, scikit-learn
- **网络分析**: networkx, graphviz
- **可视化**: matplotlib, seaborn, plotly

---

## 项目架构

### 系统架构

```
                        +-------------------+
                        |   实验入口层       |
                        |  run_pipeline.py  |
                        +-------------------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
+-------------------+  +-------------------+  +-------------------+
|  因果发现模块      |  |  因果效应模块      |  |  因果推荐模块      |
| causal_discovery  |->| causal_effect     |->|causal_recommend  |
+-------------------+  +-------------------+  +-------------------+
         |                       |                       |
         v                       v                       v
    因果图 (DAG)            ATE 估计表              推荐结果
```

### 数据流程

```
原始患者数据 (CSV)
      |
      v
[因果发现] -> 构建时序对 -> 特征编码 -> PC算法 -> 因果图 (DAG)
      |
      v
[因果效应估计] -> 加载因果图 -> 边过滤 -> OVR回归 -> ATE估计
      |
      v
[因果推荐] -> 患者信息映射 -> 因果传播 -> 推荐药物列表
      |
      v
[推荐评估] -> 对比真实数据 -> 计算指标 -> 评估报告
```

---

## 安装说明

### 环境要求

- Python >= 3.8
- conda 或 venv

### 安装步骤

1. 克隆项目

```bash
git clone <repository-url>
cd Causality-NPC
```

2. 创建虚拟环境

```bash
# 使用 conda
conda create -n causalnex_env python=3.9
conda activate causalnex_env

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置环境变量（可选）

如需使用 LLM 进行文本提取，复制 `.env.example` 为 `.env` 并填写 API 密钥：

```bash
cp .env.example .env
```

---

## 使用方法

### 运行完整管道

一键运行从因果发现到推荐的完整流程：

```bash
python experiments/run_pipeline.py --config config/base.yaml
```

跳过评估步骤：

```bash
python experiments/run_pipeline.py --config config/base.yaml --skip-evaluation
```

限制评估患者数量：

```bash
python experiments/run_pipeline.py --config config/base.yaml --max-patients 100
```

### 单步运行

单独运行因果发现：

```bash
python experiments/run_discovery.py --config config/base.yaml
```

单独运行因果效应估计：

```bash
python experiments/run_effect.py --config config/base.yaml
```

单独运行因果推荐：

```bash
python experiments/run_recommendation.py --config config/base.yaml
```

单独运行推荐评估：

```bash
python experiments/run_evaluation.py --mode evaluate
```

运行阈值搜索优化：

```bash
python experiments/run_evaluation.py --mode search --search-level medium
```

### 批量运行实验

使用 scripts 工具批量运行多个实验：

```bash
python scripts/run_all.py --config config/base.yaml
```

比较不同实验的结果：

```bash
python scripts/compare_experiments.py config/experiments/exp_freq_0.10.yaml config/experiments/exp_freq_0.15.yaml
```

---

## 配置说明

### 基础配置文件

配置文件位于 `config/base.yaml`，包含以下主要部分：

#### 因果发现配置

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|----------|
| `symptom_threshold` | 症状最小频率 | 0.10 | 越高保留症状越少 |
| `medicine_threshold` | 药物最小频率 | 0.10 | 常调参数，影响推荐范围 |
| `diagnosis_threshold` | 诊断最小频率 | 0.10 | 越高保留诊断越少 |
| `alpha` | 显著性水平 | 0.05 | 越小边越少，越大边越多 |
| `apply_constraints` | 是否应用约束 | true | 建议开启 |
| `remove_cycles` | 是否移除环 | true | 建议开启 |

#### 因果效应配置

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|----------|
| `min_correlation` | 最小相关系数 | 0.0 | 提高可减少噪声边 |
| `min_sample_size` | 最小样本量 | 20 | 根据数据量调整 |
| `bootstrap.n_iterations` | Bootstrap 次数 | 50 | 越多越稳定，但越慢 |

#### 因果推荐配置

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|----------|
| `threshold_positive` | 推荐阈值 | 0.05 | 越高推荐越严格 |
| `threshold_negative` | 警告阈值 | -0.05 | 越低警告越多 |
| `top_k` | 返回推荐数量 | 5 | 根据需求调整 |

### 实验配置

项目提供多个预定义实验配置，位于 `config/experiments/`：

- `exp_freq_0.05.yaml` ~ `exp_freq_0.25.yaml`: 不同药物频率阈值
- `exp_alpha_0.01.yaml`: 严格显著性水平
- `exp_bootstrap_100.yaml`: 更多 Bootstrap 迭代

使用实验配置：

```bash
python experiments/run_pipeline.py --config config/experiments/exp_freq_0.15.yaml
```

---

## 项目结构

```
Causality-NPC/
├── config/                      # 配置文件
│   ├── base.yaml                # 基础配置
│   └── experiments/             # 实验配置
│
├── Data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   └── processed/               # 处理后数据
│
├── docs/                        # 文档
│   ├── causal_discovery_README.md
│   ├── causal_effect_README.md
│   └── causal_recommendation_README.md
│
├── examples/                    # 示例代码
│   └── extract_npc_full.py
│
├── experiments/                 # 入口脚本
│   ├── run_pipeline.py          # 主入口
│   ├── run_discovery.py
│   ├── run_effect.py
│   ├── run_recommendation.py
│   └── run_evaluation.py
│
├── outputs/                     # 输出结果
│   └── {experiment_name}/       # 按实验名称隔离
│
├── scripts/                     # 工具脚本
│   ├── run_all.py
│   ├── compare_experiments.py
│   └── utils/
│
└── src/                         # 核心模块
    ├── causal_discovery/        # 因果发现模块
    │   ├── pipeline.py
    │   ├── config.py
    │   ├── data/
    │   ├── features/
    │   ├── discovery/
    │   ├── visualization/
    │   └── output/
    │
    ├── causal_effect/           # 因果效应模块
    │   ├── pipeline.py
    │   ├── data/
    │   ├── estimation/
    │   └── output/
    │
    └── causal_recommendation/   # 因果推荐模块
        ├── pipeline.py
        ├── data/
        ├── recommendation/
        ├── evaluation/
        └── output/
```

---

## 输出说明

### 输出目录结构

每个实验的结果独立保存在 `outputs/{experiment_name}/` 目录下：

```
outputs/{experiment_name}/
├── causal_discovery/            # 因果发现输出
│   ├── data/
│   │   ├── pairs.csv            # 时序对数据
│   │   └── matrix.csv           # 数据矩阵
│   ├── graph/
│   │   ├── dag.pkl              # 因果图对象
│   │   ├── edges.json           # 边列表
│   │   └── dag.png              # 可视化图
│   └── report.md                # 发现报告
│
├── causal_effects/              # 因果效应输出
│   ├── estimates/
│   │   └── estimated_effects_4_ovr.csv  # ATE 估计表
│   ├── models/                  # 模型对象
│   └── report.md                # 估计报告
│
├── causal_recommendation/       # 因果推荐输出
│   ├── recommendations/
│   │   └── patient_recommendations.json  # 推荐结果
│   └── report.md                # 推荐报告
│
└── evaluation/                  # 评估输出（如启用）
    ├── evaluation_*.json        # 评估指标
    └── threshold_search_*/      # 阈值搜索结果
```

### 主要输出文件

#### estimated_effects_4_ovr.csv

因果效应估计结果表，主要列：

- `treatment`: 处理变量（原因）
- `outcome`: 结果变量（结果）
- `ate`: 平均处理效应
- `ci_lower`, `ci_upper`: 置信区间
- `p_value`: 统计显著性
- `correlation`: 相关系数
- `n_treated`, `n_control`: 样本量

#### patient_recommendations.json

推荐结果 JSON，每个患者包含：

- `patient_info`: 患者基本信息
- `recommended`: 推荐药物及得分
- `not_recommended`: 不推荐药物
- `neutral`: 中性药物
- `explanations`: 因果路径解释

---

## 许可证

本项目仅供学术研究使用。
