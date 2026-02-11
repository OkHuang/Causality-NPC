# 中医临床主诉实体提取 - 使用指南

## 概述

本项目使用 LangChain + Azure OpenAI 从非结构化的中医临床患者主诉中提取结构化实体。

### 主要功能

- 从临床主诉文本中提取结构化医学实体
- 识别症状（包括阳性/阴性状态）
- 提取病史信息（诊断、治疗、病程）
- 识别舌脉象和体征
- 批量处理支持（带进度条）
- 结果导出为 Excel 格式

### 技术栈

- **语言**: Python 3.8+
- **LLM 后端**: Azure OpenAI
- **框架**: LangChain (LCEL 语法)
- **Schema**: Pydantic v2
- **输出**: Excel (.xlsx)

---

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括:
- `langchain` - LangChain 核心库
- `langchain-openai` - Azure OpenAI 集成
- `langchain-core` - LangChain 核心接口
- `pydantic` - 数据验证
- `pandas` - 数据处理
- `openpyxl` - Excel 导出
- `tqdm` - 进度条
- `python-dotenv` - 环境变量管理

### 2. 配置 Azure OpenAI

复制 `.env.example` 为 `.env` 并填入你的 Azure OpenAI 凭证:

```bash
cp .env.example .env
```

编辑 `.env` 文件:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

---

## 数据 Schema

### PatientRecord (主模型)

| 字段 | 类型 | 描述 |
|------|------|------|
| `medical_history` | `MedicalHistory` | 基本病史信息 |
| `symptoms` | `List[Symptom]` | 症状清单 |
| `physical_signs` | `List[PhysicalSign]` | 舌脉象/体征 |
| `general_condition` | `str` | 一般状态 |
| `digestion_status` | `str` | 消化系统状态 |
| `sleep_status` | `str` | 睡眠状态 |
| `elimination_status` | `str` | 排泄状态 |

### MedicalHistory (病史)

| 字段 | 类型 | 描述 |
|------|------|------|
| `diagnosis` | `str` | 主要诊断 |
| `treatment` | `List[str]` | 治疗手段列表 |
| `duration` | `str` | 病程时间 |
| `treatment_details` | `str` | 治疗细节 |

### Symptom (症状)

| 字段 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 症状名称 |
| `status` | `str` | present/absent/unknown |
| `description` | `str` | 具体描述 |
| `body_part` | `str` | 涉及部位 |

### PhysicalSign (体征)

| 字段 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 体征名称 |
| `description` | `str` | 具体描述 |

---

## 使用方法

### 方法 1: 使用示例脚本

运行交互式示例:

```bash
python examples/entity_extraction_example.py
```

### 方法 2: 在代码中使用

#### 单条提取

```python
from src.data.entity_extraction import EntityExtractor, AzureConfig

# 配置
config = AzureConfig.from_env()
extractor = EntityExtractor(config)

# 单条提取
chief_complaint = "鼻咽癌放化疗后11年余,精神可,无头痛,咽干严重,睡眠差..."
result = extractor.extract_single(chief_complaint)

if result:
    print(f"诊断: {result.medical_history.diagnosis}")
    print(f"症状数: {len(result.symptoms)}")
```

#### 批量提取

```python
import pandas as pd
from src.data.entity_extraction import EntityExtractor, postprocess_results

# 加载数据
df = pd.read_csv("Data/raw/npc_final.csv", encoding="gbk")

# 配置提取器
config = AzureConfig.from_env()
extractor = EntityExtractor(config)

# 批量提取
result_df = extractor.extract_from_dataframe(
    df,
    text_column="chief_complaint",
    batch_size=5  # 根据 Azure Rate Limit 调整
)

# 后处理和保存
output_df = postprocess_results(result_df, "outputs/processed_patients.xlsx")
```

---

## 输出格式

### Excel 输出列

| 列名 | 描述 |
|------|------|
| `diagnosis` | 诊断 |
| `treatments` | 治疗手段 (逗号分隔) |
| `duration` | 病程 |
| `treatment_details` | 治疗细节 |
| `general_condition` | 一般状态 |
| `digestion_status` | 消化状态 |
| `sleep_status` | 睡眠状态 |
| `elimination_status` | 排泄状态 |
| `total_symptoms` | 总症状数 |
| `present_symptoms` | 阳性症状数 |
| `absent_symptoms` | 阴性症状数 |
| `symptoms_json` | 症状列表 (JSON) |
| `signs_json` | 体征列表 (JSON) |
| `symptom_头痛` | 头痛状态 (present/absent/unknown) |
| `symptom_头晕` | 头晕状态 |
| `symptom_咽干` | 咽干状态 |
| ... | 其他常见症状 |

---

## 性能优化

### Rate Limit 处理

Azure OpenAI 有 TPM (Token Per Minute) 和 RPM (Request Per Minute) 限制。

调整 `batch_size` 参数来控制并发:

```python
# 保守设置 (适合低限额)
extractor.extract_from_dataframe(df, batch_size=3)

# 中等设置
extractor.extract_from_dataframe(df, batch_size=5)

# 激进设置 (适合高限额)
extractor.extract_from_dataframe(df, batch_size=10)
```

### 处理大数据集

对于 900+ 条数据，建议分批处理:

```python
# 分批处理，每批 100 条
batch_size = 100
total = len(df)

for i in range(0, total, batch_size):
    batch_df = df.iloc[i:i+batch_size]
    result = extractor.extract_from_dataframe(batch_df, batch_size=5)
    # 保存中间结果
    postprocess_results(result, f"outputs/batch_{i}.xlsx")
```

---

## 错误处理

提取失败时，`extractor.errors` 会记录错误信息:

```python
extractor = EntityExtractor(config)
result_df = extractor.extract_from_dataframe(df)

if extractor.errors:
    print(f"失败数量: {len(extractor.errors)}")
    for error in extractor.errors[:10]:
        print(f"错误: {error['error']}")
```

---

## 示例输入输出

### 输入

```
鼻咽癌放化疗后11年余(2012年10月放疗结束,中肿)精神可,无恶风,怕冷,无头痛,头晕,咽干严重,无痰,无咳嗽,无鼻塞,喷嚏,流白清涕,无涕血,无耳鸣听力下降,无耳胀闷阻塞感,无耳流脓,无胃胀反酸,胃纳可,睡眠差,难入睡,易醒,多梦,夜尿1-2次,大便正常.舌硬活动不利.伸舌偏右,言语不利,吞咽欠顺畅.
```

### 输出 (JSON)

```json
{
  "medical_history": {
    "diagnosis": "鼻咽癌",
    "treatment": ["放疗", "化疗"],
    "duration": "11年余",
    "treatment_details": "2012年10月放疗结束,中肿"
  },
  "symptoms": [
    {"name": "头痛", "status": "absent", "description": "", "body_part": "头"},
    {"name": "咽干", "status": "present", "description": "严重", "body_part": "咽"},
    {"name": "咳嗽", "status": "absent", "description": "", "body_part": ""},
    {"name": "睡眠", "status": "present", "description": "差,难入睡,易醒,多梦", "body_part": ""}
  ],
  "physical_signs": [
    {"name": "舌硬", "description": "活动不利"},
    {"name": "伸舌偏右", "description": ""}
  ],
  "general_condition": "精神可",
  "digestion_status": "无胃胀反酸,胃纳可",
  "sleep_status": "睡眠差,难入睡,易醒,多梦",
  "elimination_status": "夜尿1-2次,大便正常"
}
```

---

## 常见问题

### Q: 提取速度慢怎么办?

A: 减小 `batch_size` 或升级 Azure 定额。

### Q: 某些记录提取失败?

A: 检查 `extractor.errors`，可能是文本格式问题或 API 限流。

### Q: 如何调整提取精度?

A: 修改 `entity_extraction.py` 中的 `system_message` 提示词。

### Q: 支持哪些模型?

A: 支持 Azure OpenAI 上的 GPT-3.5/GPT-4 系列模型。修改 `AZURE_OPENAI_DEPLOYMENT_NAME` 环境变量。

---

## 文件结构

```
Causality-NPC/
├── src/data/
│   └── entity_extraction.py      # 主模块
├── examples/
│   └── entity_extraction_example.py  # 使用示例
├── Data/raw/
│   └── npc_final.csv             # 输入数据
├── outputs/
│   └── processed_patients.xlsx   # 输出结果
├── .env                          # 环境变量配置
├── .env.example                  # 配置示例
└── requirements.txt              # 依赖列表
```

---

## 下一步

1. 运行示例脚本验证配置
2. 根据实际数据调整提示词
3. 优化批处理参数
4. 集成到数据处理流水线
