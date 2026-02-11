# 鼻咽癌因果推断项目 (Causality-NPC)

**中西医结合诊疗的因果推断与动态网络发现**

版本: 0.1.0

---

## 项目简介

本项目旨在利用因果推断方法，从963条鼻咽癌患者的中西医结合诊疗数据中，**量化中药疗效**并**发现因果机制**。

### 研究目标

#### 目标一：验证性因果推断
- 模拟随机对照试验（RCT）
- 评估特定药物/疗法对症状的因果效应
- 方法：倾向性评分匹配（PSM）+ ATE估计

#### 目标二：因果发现
- 构建症状-药物-证型的因果网络
- 挖掘潜在的因果路径
- 方法：PC/FCI算法 + Bootstrap稳定性选择

---

## 项目结构

```
Causality-NPC/
├── config/                    # 配置文件
│   ├── base.yaml              # 基础配置
│   ├── preprocessing.yaml     # 预处理配置
│   ├── causal_inference.yaml  # 因果推断配置
│   └── causal_discovery.yaml  # 因果发现配置
│
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   │   ├── loader.py          # 数据加载
│   │   ├── preprocessing.py   # 数据清洗
│   │   └── time_alignment.py  # 时间对齐
│   │
│   ├── features/              # 特征工程模块
│   │   ├── symptom_extractor.py   # 症状提取
│   │   ├── medicine_mapper.py     # 药物映射
│   │   └── syndrome_encoder.py    # 证型编码
│   │
│   ├── causal_inference/      # 目标一：因果推断
│   │   ├── propensity_score.py    # 倾向性评分
│   │   ├── matching.py            # 患者匹配
│   │   ├── ate_estimator.py       # ATE估计
│   │   └── validator.py           # 平衡性验证
│   │
│   ├── causal_discovery/      # 目标二：因果发现
│   │   ├── algorithms.py          # PC/FCI算法
│   │   ├── bootstrap.py           # 稳定性选择
│   │   ├── constraints.py         # 约束管理
│   │   └── graph_utils.py         # 图工具
│   │
│   ├── visualization/         # 可视化模块
│   │   ├── dag_plotter.py         # DAG绘制
│   │   └── report_generator.py    # 报告生成
│   │
│   └── utils/                 # 工具函数
│       ├── metrics.py             # 评估指标
│       ├── stat_tests.py          # 统计检验
│       └── config.py              # 配置管理
│
├── experiments/               # 实验脚本
│   ├── 01_data_exploration.ipynb   # 数据探索
│   ├── 02_preprocessing_pipeline.py # 预处理流程
│   ├── 03_propensity_score.ipynb   # 倾向性评分实验
│   ├── 04_causal_inference.py      # 因果推断实验
│   ├── 05_causal_discovery.py      # 因果发现实验
│   └── 06_stability_analysis.py    # 稳定性分析
│
├── outputs/                   # 输出结果
│   ├── figures/               # 图表
│   ├── reports/               # 分析报告
│   └── graphs/                # 因果图
│
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 预处理后数据
│   ├── features/              # 特征文件
│   └── models/                # 训练好的模型
│
├── docs/                      # 文档
│   ├── plan.md                # 项目计划
│   └── knowledge_base/        # 领域知识文档
│       ├── tcm_symptoms.md    # 中医症状词典
│       ├── herbs_mapping.md   # 中药功效映射
│       └── expert_rules.md    # 专家规则
│
├── tests/                     # 单元测试
│
└── requirements.txt           # 依赖包
```

---

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-repo/Causality-NPC.git
cd Causality-NPC

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

将原始数据放入 `data/raw/` 目录，确保数据包含以下字段：
- `patient_id`: 患者ID
- `time`: 就诊时间
- `chief_complaint`: 主诉文本
- `chinese_medicines`: 中药处方
- `chinese_diagnosis`: 中医证型
- `western_medicines`: 西医治疗（可选）

### 3. 运行实验

#### 目标一：因果推断（验证性）

```python
from src import DataLoader, SymptomExtractor, MedicineMapper
from src.causal_inference import PropensityScoreMatcher, ATEEstimator

# 加载数据
loader = DataLoader()
df = loader.load("data/raw/npc_data.csv")

# 特征提取
symptom_extractor = SymptomExtractor()
medicine_mapper = MedicineMapper()
df = symptom_extractor.transform(df, text_column="chief_complaint")
df = medicine_mapper.transform(df, medicine_column="chinese_medicines")

# 定义研究假设
treatment = "M_补气药"  # 处理变量
outcome = "S_乏力_t1"   # 结果变量（t+1时刻）
confounders = ["age", "gender", "D_气虚"]  # 混杂因素

# 倾向性评分匹配
ps_matcher = PropensityScoreMatcher()
df_matched = ps_matcher.fit_transform(df, treatment, confounders)

# 匹配患者
from src.causal_inference import match_patients, assess_match_quality
df_matched, match_info = match_patients(df_matched, treatment)
balance_df = assess_match_quality(df_matched, treatment, confounders)

# 估计ATE
ate_estimator = ATEEstimator()
results = ate_estimator.estimate(df_matched, treatment, outcome)

# 生成报告
from src.visualization import ReportGenerator
report_gen = ReportGenerator()
report_gen.generate_causal_inference_report(
    results, balance_df, "outputs/reports/causal_inference_report.md"
)
```

#### 目标二：因果发现（探索性）

```python
from src import CausalDiscovery, StabilitySelector
from src.causal_discovery import ConstraintManager

# 准备时序数据
# 确保变量名包含时间标签（如 "S_乏力_t", "M_补气药_t1"）

# 运行因果发现
cd = CausalDiscovery()
graph = cd.discover(
    data=df_time,
    algorithm="pc",
    alpha=0.05,
    independence_test="fisherz"
)

# 稳定性选择
stability_selector = StabilitySelector(n_bootstrap=1000)
stable_edges = stability_selector.select(
    df_time, cd.discover, {"alpha": 0.05}
)

# 应用约束
constraint_manager = ConstraintManager()
graph_constrained = constraint_manager.apply_constraints(graph, df_time.columns)

# 可视化
cd.plot_graph(layout="hierarchical", save_path="outputs/graphs/causal_dag.png")
```

---

## 核心功能

### 数据处理模块
- ✅ 数据加载与验证
- ✅ 时间轴对齐
- ✅ 症状文本提取（支持规则+LLM）
- ✅ 药物功效映射
- ✅ 证型编码

### 因果推断模块（目标一）
- ✅ 倾向性评分估计（逻辑回归、随机森林、梯度提升）
- ✅ 患者匹配（最近邻、最优匹配）
- ✅ 平衡性检验（标准化差异、t检验）
- ✅ ATE估计（均值差、回归调整、双鲁棒）
- ✅ 敏感性分析（Rosenbaum界、E-value）

### 因果发现模块（目标二）
- ✅ PC算法
- ✅ FCI算法（隐变量）
- ✅ Bootstrap稳定性选择
- ✅ 时序约束
- ✅ 领域知识约束
- ✅ 路径提取与可视化

### 可视化模块
- ✅ DAG绘制（支持分层布局）
- ✅ 平衡性图
- ✅ 边频率分布图
- ✅ Markdown报告生成

---

## 配置说明

所有配置文件位于 `config/` 目录：

- **base.yaml**: 项目全局配置（路径、随机种子、日志等）
- **preprocessing.yaml**: 数据预处理配置（时间窗口、症状分级等）
- **causal_inference.yaml**: 因果推断配置（匹配方法、ATE估计方法等）
- **causal_discovery.yaml**: 因果发现配置（算法选择、约束定义等）

可通过修改YAML文件调整参数，无需修改代码。

---

## 依赖说明

### 核心依赖
- **numpy, pandas**: 数据处理
- **scikit-learn**: 机器学习
- **causal-learn**: PC/FCI算法实现

### 可选依赖
- **dowhy**: 因果推断验证
- **openai/anthropic**: LLM文本提取

---

## 常见问题

### Q1: 如何添加新的症状词典？

编辑 `docs/knowledge_base/tcm_symptoms.md`，代码会自动加载。

### Q2: Bootstrap太慢怎么办？

在 `config/base.yaml` 中设置 `n_bootstrap: 100`（快速测试）或使用并行计算。

### Q3: 如何解读因果图？

- 实线箭头：有向因果关系
- 箭头方向：时间流向（过去 → 未来）
- 边的粗细：置信度（通过Bootstrap频率计算）

---

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至：your.email@example.com

---

**最后更新**: 2026-01-28
