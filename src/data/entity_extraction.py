"""
中医临床主诉实体提取模块

使用 LangChain + Azure OpenAI 从非结构化临床主诉中提取结构化实体
"""

import os
import json
import asyncio
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# LangChain 相关导入
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field


# ==================== Step 1: Pydantic Schema 定义 ====================

class Symptom(BaseModel):
    """症状实体"""
    name: str = Field(description="症状名称，如: 头痛、睡眠、咽干、咳嗽等")
    status: Literal["present", "absent", "unknown"] = Field(
        description="症状状态: present(阳性/存在), absent(阴性/无), unknown(未提及或不明确)"
    )
    description: Optional[str] = Field(
        default="",
        description="症状的具体描述，如: 严重、难入睡、易醒、白清涕等"
    )
    body_part: Optional[str] = Field(
        default="",
        description="症状涉及的部位，如: 舌、耳、胃、咽等"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "头痛",
                "status": "absent",
                "description": "",
                "body_part": "头"
            }
        }


class MedicalHistory(BaseModel):
    """基本病史信息"""
    diagnosis: str = Field(description="主要诊断，如: 鼻咽癌")
    treatment: List[str] = Field(
        default_factory=list,
        description="治疗手段，如: 放疗、化疗、手术等"
    )
    duration: str = Field(
        default="",
        description="病程时间描述，如: 11年余、5个月、1年等"
    )
    treatment_details: Optional[str] = Field(
        default="",
        description="治疗的具体细节，如: 2012年10月放疗结束、中肿等"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "diagnosis": "鼻咽癌",
                "treatment": ["放疗", "化疗"],
                "duration": "11年余",
                "treatment_details": "2012年10月放疗结束,中肿"
            }
        }


class PhysicalSign(BaseModel):
    """舌脉象/体征"""
    name: str = Field(description="体征名称，如: 舌硬、伸舌偏右、舌苔黄等")
    description: Optional[str] = Field(default="", description="体征的具体描述")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "舌硬",
                "description": "活动不利"
            }
        }


class PatientRecord(BaseModel):
    """患者记录主模型"""
    medical_history: MedicalHistory = Field(description="患者的基本病史信息")
    symptoms: List[Symptom] = Field(
        default_factory=list,
        description="患者的所有症状清单"
    )
    physical_signs: List[PhysicalSign] = Field(
        default_factory=list,
        description="舌脉象和其他体征"
    )
    general_condition: Optional[str] = Field(
        default="",
        description="一般状态描述，如: 精神可、胃纳可、睡眠差等"
    )
    digestion_status: Optional[str] = Field(
        default="",
        description="消化系统相关描述，如: 胃纳可、无胃胀反酸等"
    )
    sleep_status: Optional[str] = Field(
        default="",
        description="睡眠相关描述，如: 睡眠差、难入睡、易醒、多梦等"
    )
    elimination_status: Optional[str] = Field(
        default="",
        description="排泄相关描述，如: 夜尿1-2次、大便正常等"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "medical_history": {
                    "diagnosis": "鼻咽癌",
                    "treatment": ["放疗", "化疗"],
                    "duration": "11年余",
                    "treatment_details": "2012年10月放疗结束,中肿"
                },
                "symptoms": [
                    {
                        "name": "头痛",
                        "status": "absent",
                        "description": "",
                        "body_part": "头"
                    },
                    {
                        "name": "咽干",
                        "status": "present",
                        "description": "严重",
                        "body_part": "咽"
                    }
                ],
                "physical_signs": [
                    {
                        "name": "舌硬",
                        "description": "活动不利"
                    },
                    {
                        "name": "伸舌偏右",
                        "description": ""
                    }
                ],
                "general_condition": "精神可",
                "digestion_status": "胃纳可,无胃胀反酸",
                "sleep_status": "睡眠差,难入睡,易醒,多梦",
                "elimination_status": "夜尿1-2次,大便正常"
            }
        }


# ==================== Step 2: Azure OpenAI 配置 ====================

@dataclass
class AzureConfig:
    """Azure OpenAI 配置"""
    api_key: str
    endpoint: str
    api_version: str
    deployment_name: str

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """从环境变量加载配置"""
        load_dotenv()

        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

        if not api_key or not endpoint:
            raise ValueError(
                "请设置环境变量 AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT"
            )

        return cls(
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            deployment_name=deployment_name
        )


def create_azure_llm(config: AzureConfig, temperature: float = 0.0):
    """
    创建 Azure ChatOpenAI 实例

    Parameters
    ----------
    config : AzureConfig
        Azure 配置
    temperature : float
        温度参数，默认 0.0 保证稳定性

    Returns
    -------
    AzureChatOpenAI
        Azure OpenAI 模型实例
    """
    return AzureChatOpenAI(
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
        api_version=config.api_version,
        deployment_name=config.deployment_name,
        temperature=temperature,
        max_tokens=2000,
    )


# ==================== Step 3: Extraction Chain (LCEL) ====================

def create_extraction_prompt() -> ChatPromptTemplate:
    """
    创建实体提取的提示词模板

    Returns
    -------
    ChatPromptTemplate
        LangChain 提示词模板
    """
    system_message = """你是一位专业的中医临床信息提取专家。你的任务是从患者的临床主诉文本中准确地提取结构化的医学实体。

**提取规则:**
1. **症状状态判断:**
   - "无XXX"、"未XXX"、"无"开头的症状 → status="absent"
   - 直接描述症状（如"咽干严重"、"难入睡"）→ status="present"
   - 不确定或未明确提及 → status="unknown"

2. **基本病史提取:**
   - 提取主要诊断（如: 鼻咽癌）
   - 提取治疗方式（如: 放疗、化疗、手术）
   - 提取病程时间（如: 11年余、5个月）
   - 提取治疗细节（如: 具体日期、医院名称）

3. **体征/舌脉象:**
   - 舌象相关: 舌硬、伸舌偏右、舌苔等
   - 脉象相关: 如有提及
   - 其他体征

4. **分类提取:**
   - general_condition: 精神状态、整体感觉
   - digestion_status: 胃纳、反酸、胃胀等
   - sleep_status: 睡眠质量、入睡难易、是否多梦
   - elimination_status: 小便（夜尿次数）、大便情况

5. **提取要求:**
   - 尽可能完整地提取所有症状
   - 保持描述的原意，不要添加额外信息
   - 对于"无症状"的情况，也要记录（status=absent）
   - body_part 尽可能明确

**输出格式:**
严格按照 JSON Schema 格式输出，确保所有字段都正确填写。"""

    human_message = """请从以下患者主诉中提取结构化的医学实体:

患者主诉:
{chief_complaint}

请返回符合 JSON Schema 的结构化数据。"""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def create_extraction_chain(llm: AzureChatOpenAI):
    """
    创建实体提取链（使用 LCEL 语法）

    Parameters
    ----------
    llm : AzureChatOpenAI
        Azure OpenAI 模型实例

    Returns
    -------
    Runnable
        LangChain 提取链
    """
    prompt = create_extraction_prompt()

    # 使用 with_structured_output 方法强制输出 Pydantic 模型格式
    extraction_chain = prompt | llm.with_structured_output(schema=PatientRecord)

    return extraction_chain


# ==================== Step 4: 批处理逻辑 ====================

class EntityExtractor:
    """实体提取器主类"""

    def __init__(self, config: Optional[AzureConfig] = None):
        """
        初始化实体提取器

        Parameters
        ----------
        config : AzureConfig, optional
            Azure 配置，默认从环境变量加载
        """
        self.config = config or AzureConfig.from_env()
        self.llm = create_azure_llm(self.config)
        self.chain = create_extraction_chain(self.llm)
        self.results = []
        self.errors = []

    def extract_single(self, chief_complaint: str) -> Optional[PatientRecord]:
        """
        从单条主诉中提取实体

        Parameters
        ----------
        chief_complaint : str
            患者主诉文本

        Returns
        -------
        PatientRecord or None
            提取的患者记录，失败返回 None
        """
        try:
            result = self.chain.invoke({"chief_complaint": chief_complaint})
            return result
        except Exception as e:
            self.errors.append({
                "chief_complaint": chief_complaint[:100],  # 只保存前100字符
                "error": str(e)
            })
            return None

    def extract_batch(
        self,
        chief_complaints: List[str],
        batch_size: int = 10,
        show_progress: bool = True
    ) -> List[Optional[PatientRecord]]:
        """
        批量提取实体（带并发和进度条）

        Parameters
        ----------
        chief_complaints : List[str]
            患者主诉文本列表
        batch_size : int
            批处理大小，用于控制并发
        show_progress : bool
            是否显示进度条

        Returns
        -------
        List[PatientRecord or None]
            提取结果列表
        """
        results = []
        total = len(chief_complaints)

        # 使用 LangChain 的 batch 方法（自动处理并发）
        # 注意: 需要根据 Azure 的 Rate Limit 调整 batch_size
        inputs = [{"chief_complaint": text} for text in chief_complaints]

        if show_progress:
            # 分批处理，每批显示进度
            with tqdm(total=total, desc="提取实体", unit="条") as pbar:
                for i in range(0, total, batch_size):
                    batch_inputs = inputs[i:i + batch_size]
                    batch_results = self._process_batch_with_retry(batch_inputs)
                    results.extend(batch_results)
                    pbar.update(len(batch_results))
        else:
            for i in range(0, total, batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_results = self._process_batch_with_retry(batch_inputs)
                results.extend(batch_results)

        self.results = results
        return results

    def _process_batch_with_retry(
        self,
        batch_inputs: List[Dict[str, str]],
        max_retries: int = 2
    ) -> List[Optional[PatientRecord]]:
        """
        处理单个批次（带重试机制）

        Parameters
        ----------
        batch_inputs : List[Dict]
            批次输入
        max_retries : int
            最大重试次数

        Returns
        -------
        List[PatientRecord or None]
            处理结果
        """
        results = []

        for item in batch_inputs:
            success = False
            retry_count = 0
            result = None

            while not success and retry_count <= max_retries:
                try:
                    result = self.chain.invoke(item)
                    success = True
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.errors.append({
                            "chief_complaint": item["chief_complaint"][:100],
                            "error": str(e)
                        })
                        result = None

            results.append(result)

        return results

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "chief_complaint",
        batch_size: int = 10
    ) -> pd.DataFrame:
        """
        从 DataFrame 中提取实体

        Parameters
        ----------
        df : pd.DataFrame
            输入数据
        text_column : str
            包含主诉文本的列名
        batch_size : int
            批处理大小

        Returns
        -------
        pd.DataFrame
            包含提取结果的 DataFrame
        """
        # 确保列存在
        if text_column not in df.columns:
            raise ValueError(f"DataFrame 中没有列: {text_column}")

        # 过滤掉空值
        valid_df = df[df[text_column].notna()].copy()
        chief_complaints = valid_df[text_column].tolist()

        # 批量提取
        results = self.extract_batch(chief_complaints, batch_size=batch_size)

        # 将结果添加到 DataFrame
        valid_df = valid_df.reset_index(drop=True)
        valid_df["extracted_data"] = results

        return valid_df


# ==================== Step 5: 后处理和保存 ====================

def flatten_patient_record(record: PatientRecord) -> Dict[str, Any]:
    """
    将嵌套的 PatientRecord 展平为字典

    Parameters
    ----------
    record : PatientRecord
        患者记录

    Returns
    -------
    Dict
        展平后的字典
    """
    flat_data = {
        # 基本病史
        "diagnosis": record.medical_history.diagnosis,
        "treatments": ",".join(record.medical_history.treatment),
        "duration": record.medical_history.duration,
        "treatment_details": record.medical_history.treatment_details,

        # 分类状态
        "general_condition": record.general_condition or "",
        "digestion_status": record.digestion_status or "",
        "sleep_status": record.sleep_status or "",
        "elimination_status": record.elimination_status or "",

        # 症状统计
        "total_symptoms": len(record.symptoms),
        "present_symptoms": len([s for s in record.symptoms if s.status == "present"]),
        "absent_symptoms": len([s for s in record.symptoms if s.status == "absent"]),

        # 症状列表 (JSON 字符串)
        "symptoms_json": json.dumps(
            [s.model_dump() for s in record.symptoms],
            ensure_ascii=False
        ),

        # 体征列表 (JSON 字符串)
        "signs_json": json.dumps(
            [s.model_dump() for s in record.physical_signs],
            ensure_ascii=False
        ),
    }

    # 提取常见症状作为独立列
    common_symptoms = {
        "头痛": None, "头晕": None, "咽干": None, "咳嗽": None,
        "鼻塞": None, "耳鸣": None, "听力": None, "睡眠": None,
        "胃纳": None, "反酸": None, "胃胀": None
    }

    for symptom in record.symptoms:
        if symptom.name in common_symptoms:
            common_symptoms[symptom.name] = symptom.status

    for name, status in common_symptoms.items():
        flat_data[f"symptom_{name}"] = status or "unknown"

    return flat_data


def postprocess_results(
    df: pd.DataFrame,
    output_path: str = "processed_patients.xlsx"
) -> pd.DataFrame:
    """
    后处理提取结果并保存为 Excel

    Parameters
    ----------
    df : pd.DataFrame
        包含 extracted_data 列的 DataFrame
    output_path : str
        输出 Excel 文件路径

    Returns
    -------
    pd.DataFrame
        处理后的展平 DataFrame
    """
    # 展平所有记录
    flattened_records = []

    for _, row in df.iterrows():
        if row["extracted_data"] is not None:
            flat_record = flatten_patient_record(row["extracted_data"])

            # 保留原始数据的某些列
            for col in ["id", "patient_id", "age", "gender", "time"]:
                if col in row:
                    flat_record[col] = row[col]

            flattened_records.append(flat_record)

    # 创建新的 DataFrame
    result_df = pd.DataFrame(flattened_records)

    # 保存为 Excel
    result_df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"结果已保存至: {output_path}")
    print(f"总共处理 {len(result_df)} 条记录")

    return result_df


# ==================== 主函数 ====================

def main():
    """主函数示例"""
    # 1. 加载环境变量和配置
    config = AzureConfig.from_env()

    # 2. 创建提取器
    extractor = EntityExtractor(config)

    # 3. 加载数据
    input_path = "Data/raw/npc_final.csv"
    print(f"正在加载数据: {input_path}")

    # 尝试不同的编码
    encodings = ["utf-8", "gbk", "gb2312", "gb18030"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            print(f"成功使用编码: {encoding}")
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None:
        raise ValueError("无法读取 CSV 文件，请检查文件编码")

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 4. 提取实体
    result_df = extractor.extract_from_dataframe(
        df,
        text_column="chief_complaint",
        batch_size=5  # 根据 Azure Rate Limit 调整
    )

    # 5. 打印错误统计
    print(f"\n处理完成!")
    print(f"成功: {len(result_df)} 条")
    print(f"失败: {len(extractor.errors)} 条")

    if extractor.errors:
        print("\n错误示例:")
        for error in extractor.errors[:3]:
            print(f"  - {error}")

    # 6. 后处理和保存
    output_df = postprocess_results(result_df, "outputs/processed_patients.xlsx")

    # 打印统计信息
    print("\n=== 提取统计 ===")
    print(output_df.describe())

    return output_df


if __name__ == "__main__":
    main()
