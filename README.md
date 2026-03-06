# 鼻咽癌因果推断项目 (Causality-NPC)

**中西医结合诊疗的因果推断研究**

版本: 1.0.0 | 最后更新: 2026-03-06

---

## 📋 项目简介

本项目旨在利用**因果推断方法**，从鼻咽癌患者的中西医结合诊疗数据中：

1. **提取结构化症状**：从自由文本主诉中提取标准化症状
2. **发现因果关系**：构建症状-药物-证型的因果网络
3. **量化因果效应**：计算药物对症状的真实疗效（**开发中**）

---

## 🎯 已实现功能

### 功能1：中医症状实体抽取 ✅

**目标**：将患者自由文本主诉转换为结构化症状数据

#### 技术方案
- **LLM引擎**：Azure OpenAI GPT-4
- **框架**：LangChain + Pydantic
- **症状词典**：41种标准化症状
- **严重程度分级**：0=无，1=轻度，2=重度

#### 标准症状词典
```python
{
    # 头颈部
    "精神", "头痛", "头晕", "畏寒",

    # 耳鼻喉
    "耳鸣", "听力下降", "耳胀闷", "鼻塞", "流涕", "咽干", "咽痛",

    # 呼吸道
    "咳嗽", "咳痰", "气促",

    # 消化系统
    "恶心", "呕吐", "胃胀", "反酸", "口干", "口苦",

    # 精神睡眠
    "失眠", "多梦", "易醒",

    # 排泄
    "便秘", "腹泻", "夜尿",

    # 疼痛相关
    "颈痛", "肩痛", "骨痛",

    # 全身症状
    "发热", "消瘦", "心悸"
}
```

#### 使用方法
```bash
# 运行症状抽取
python examples/extract_npc_full.py
```

#### 输入输出
**输入**：`Data/raw/npc_full.csv`
```csv
patient_id, chief_complaint, ...
001, "鼻咽癌放化疗后11年余，精神疲倦乏力，咽干严重，偶有咳嗽"
```

**输出**：`Data/raw/npc_full_with_symptoms.csv`
```csv
patient_id, chief_complaint, extracted_symptoms
001, "...", "[{"name": "乏力", "severity": 1, "label": "轻度"},
             {"name": "咽干", "severity": 2, "label": "重度"}]"
```

#### 数据规模
- **原始记录**：963条就诊记录
- **患者总数**：239人
- **提取症状**：41种标准症状

---

### 功能2：简单因果发现 ✅

**目标**：从时序数据中自动发现变量间的因果关系

#### 技术方案
- **算法**：PC算法（Peter-Clark算法）
- **独立性检验**：Fisher-Z检验
- **约束管理**：时序约束 + 领域知识约束
- **可视化**：NetworkX + Matplotlib

#### 核心流程
```mermaid
graph LR
    A[原始数据] --> B[时序对构建]
    B --> C[特征编码]
    C --> D[PC算法]
    D --> E[约束应用]
    E --> F[因果图可视化]
```

#### 使用方法
```bash
# 运行因果发现
python experiments/simple_causal_discovery.py
```

#### 数据处理流程
1. **时序对构建**：724对相邻就诊记录（139名患者）
2. **特征筛选**：
   - 症状：41 → 20（频率≥10%）
   - 药物：272 → 40（频率≥10%）
   - 诊断：324 → 28（频率≥10%）
3. **因果发现**：107个节点，180条因果边

#### 发现的因果关系

##### 药物→症状（11条）
| 药物 | 目标症状 | 说明 |
|------|----------|------|
| 威灵仙 | 颈痛、腹泻 | 通络止痛 |
| 人参片 | 头晕 | 大补元气 |
| 石菖蒲 | 精神 | 开窍醒神 |
| 辛夷 | 流涕 | 通鼻窍 |
| 陈皮 | 鼻塞 | 理气健脾 |

##### 症状持续性（11条）
- 畏寒、头痛、头晕、咽干、咳痰、咳嗽、鼻塞、耳鸣、听力下降、耳胀闷、反酸、精神

#### 输出文件
```
outputs/
├── data/
│   ├── step1_processed_data.csv      # 处理后数据
│   ├── step2_pairs_data.csv          # 时序对数据
│   ├── step3_encoded_data.csv        # 编码数据
│   └── step4_data_matrix.csv         # 最终数据矩阵
├── graphs/
│   ├── causal_dag.pkl                # NetworkX图对象
│   ├── causal_edges.json             # 边列表
│   └── causal_dag.png                # 可视化图
└── reports/
    └── simple_causal_discovery_report.md
```

---

## 🚀 下一步开发计划

### 阶段3：因果推断（开发中）⚙️

**目标**：基于因果图，量化药物对症状的因果效应

#### 技术方案
- **方法**：因果效应估计（Causal Effect Estimation）
- **核心技术**：
  - 倾向性评分匹配（PSM）
  - 平均处理效应（ATE）估计
  - 双稳健估计（Doubly Robust Estimation）

#### 实现计划
```python
# 伪代码示例
for treatment_outcome in causal_edges:
    if treatment_outcome.type == "药物→症状":
        # 1. 识别混杂因素
        confounders = identify_confounders(
            treatment=treatment_outcome.source,
            outcome=treatment_outcome.target,
            causal_graph=causal_dag
        )

        # 2. 倾向性评分匹配
        matched_data = propensity_score_matching(
            data=data_matrix,
            treatment=treatment_outcome.source,
            confounders=confounders
        )

        # 3. 估计ATE
        ate_result = estimate_ate(
            data=matched_data,
            treatment=treatment_outcome.source,
            outcome=treatment_outcome.target
        )

        # 4. 生成报告
        print(f"{treatment_outcome.source} → {treatment_outcome.target}")
        print(f"  ATE: {ate_result.ate} (95% CI: {ate_result.ci})")
        print(f"  p值: {ate_result.p_value}")
```

#### 预期输出
- 每条药物→症状因果边的效应大小
- 置信区间和显著性检验
- 带权重的因果图（边上标注ATE值）

---

## 📁 项目结构

```
Causality-NPC/
├── Data/
│   └── raw/
│       ├── npc_full.csv                  # 原始数据
│       └── npc_full_with_symptoms.csv    # 提取症状后
│
├── src/
│   ├── data/
│   │   ├── simple_loader.py              # 数据加载
│   │   ├── simple_pair_builder.py        # 时序对构建
│   │   └── simplified_extraction.py      # 症状抽取
│   ├── features/
│   │   └── simple_encoder.py             # 特征编码
│   ├── causal_discovery/
│   │   └── simple_constraints.py         # 因果约束
│   └── visualization/
│       └── simple_plotter.py             # 可视化
│
├── experiments/
│   └── simple_causal_discovery.py       # 因果发现主脚本
│
├── examples/
│   └── extract_npc_full.py              # 症状抽取脚本
│
└── outputs/
    ├── data/                             # 处理后数据
    ├── graphs/                           # 因果图
    └── reports/                          # 分析报告
```

---

## 🔧 环境配置

### Python环境
```bash
# 创建环境
conda create -n causalnex_env python=3.8
conda activate causalnex_env

# 安装依赖
pip install -r requirements.txt
```

### 必需依赖
```bash
# 核心依赖
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
networkx>=2.6.0
causal-learn>=0.1.3

# 可视化
matplotlib>=3.5.0

# LLM（症状抽取）
langchain-openai>=0.1.0
langchain-core>=0.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### Azure OpenAI配置
创建 `.env` 文件：
```bash
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

---

## 📊 数据说明

### 原始数据字段
| 字段 | 说明 | 示例 |
|------|------|------|
| patient_id | 患者ID | 001 |
| time | 就诊时间 | 2023-10-02 |
| gender | 性别 | 女 |
| age | 年龄 | 58 |
| chief_complaint | 主诉文本 | "精神疲倦乏力，咽干严重" |
| chinese_diagnosis | 中医诊断 | "痰瘀互结证" |
| western_diagnosis | 西医诊断 | "高血压" |
| chinese_medicines | 中药处方 | "黄芪 党参 甘草" |

### 时序对数据结构
```
患者A: 第1次就诊(t) → 第2次就诊(t+1)
      │                      │
   症状_t                症状_t1
   诊断_t                诊断_t1
   药物_t                (不使用)
```

---

## 📈 核心发现

### 因果网络统计
- **节点数**：107个
- **边数**：180条
- **网络密度**：1.6%

### 关键药物疗效证据
1. **威灵仙 → 颈痛**：符合中医药理论（通络止痛）
2. **人参片 → 头晕**：符合"气虚则头晕"理论
3. **石菖蒲 → 精神**：符合开窍醒神功效

### 症状持续性模式
12种症状具有强持续性，构成鼻咽癌患者的"核心症状群"

---

## 🎓 技术亮点

1. **LLM驱动的症状提取**
   - 使用GPT-4从自由文本中提取结构化症状
   - 症状名称标准化（41种标准症状）
   - 严重程度自动分级（0/1/2）

2. **时序因果发现**
   - 基于PC算法的自动因果发现
   - 时序约束（未来不能影响过去）
   - 领域知识约束（医学合理性）

3. **端到端流程**
   - 从原始文本到因果图的完整流程
   - 支持断点续传（症状抽取）
   - 自动化报告生成