"""
简化版症状提取模块

功能：
1. 只提取症状（不提取病史、体征等）
2. 症状分级：0=无，1=轻度，2=重度
3. 症状名称标准化（使用标准词典）
"""

import os
import json
from typing import List, Optional, Dict, Literal
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# LangChain 相关导入
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ==================== Step 1: 标准症状词典 ====================

STANDARD_SYMPTOMS = {
    # 头颈部
    "精神", "头痛", "头晕", "畏寒"

    # 耳鼻喉
    "耳鸣", "听力下降", "耳胀闷", "耳流脓", "鼻塞", "涕血", "流涕", "咽干", "咽痛", "声音嘶哑",

    # 呼吸道
    "咳嗽", "咳痰", "气促", "呼吸困难",

    # 消化系统
    "恶心", "呕吐", "胃纳", "胃胀", "反酸", "口干", "口苦", "吞咽", "失语",

    # 精神睡眠
    "失眠", "多梦", "易醒", "嗜睡", "盗汗",

    # 排泄
    "便秘", "腹泻", "大便粘腻", "夜尿",

    # 疼痛相关
    "颈痛", "肩痛", "骨痛",

    # 全身症状
    "发热", "消瘦", "心悸"
}


def find_closest_symptom(symptom: str, threshold: float = 0.6) -> Optional[str]:
    """
    使用模糊匹配找到最接近的标准症状名

    Parameters
    ----------
    symptom : str
        原始症状名
    threshold : float
        相似度阈值（0-1）

    Returns
    -------
    str or None
        标准症状名，如果不匹配返回None
    """
    if symptom in STANDARD_SYMPTOMS:
        return symptom

    # 找到最匹配的标准症状
    best_match = None
    best_ratio = 0

    for standard in STANDARD_SYMPTOMS:
        ratio = SequenceMatcher(None, symptom, standard).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = standard

    if best_ratio >= threshold:
        return best_match

    return None


# ==================== Step 2: 简化的 Pydantic Schema ====================

class SymptomItem(BaseModel):
    """单个症状项"""
    name: str = Field(description="症状名称（请尽量使用标准症状名称）")
    severity: Literal[0, 1, 2] = Field(
        description="症状严重程度: 0=无/消失, 1=轻度/偶有, 2=重度/持续/严重"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "咽干",
                "severity": 2
            }
        }


class SimplifiedSymptomRecord(BaseModel):
    """简化的症状记录（只包含症状）"""
    symptoms: List[SymptomItem] = Field(
        default_factory=list,
        description="患者的所有症状列表"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": [
                    {"name": "头痛", "severity": 0},
                    {"name": "咽干", "severity": 2},
                    {"name": "失眠", "severity": 1},
                ]
            }
        }


# ==================== Step 3: Azure OpenAI 配置 ====================

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
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-chat")

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


def create_azure_llm(config: AzureConfig):
    """创建 Azure ChatOpenAI 实例"""
    return AzureChatOpenAI(
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
        api_version=config.api_version,
        deployment_name=config.deployment_name,
        temperature=1.0,
    )


# ==================== Step 4: 优化的提示词 ====================

def create_extraction_prompt() -> ChatPromptTemplate:
    """
    创建简化的症状提取提示词模板（使用思维链结构）
    """
    # 格式化标准症状列表
    symptoms_list = "\n".join([f"  - {s}" for s in sorted(STANDARD_SYMPTOMS)])

    system_message = f"""你是一位专业的中医临床信息提取专家。你的任务是从患者的临床主诉文本中准确地提取症状及其严重程度。

**标准症状词表（请优先使用这些名称）：**
{symptoms_list}

**严重程度分级标准：**
- 0级（无）："无"、"消失"、"缓解"、"改善"、"未XXX"、"可"
- 1级（轻度）："偶有"、"轻度"、"稍"、"微"
- 2级（重度）："严重"、"剧烈"、"明显"、"持续"、"难以忍受"、"加重"

**提取流程（请按此步骤思考）：**

第一步：扫描文本，识别所有症状关键词
- 逐词扫描，找出所有症状相关的词汇
- 注意：不要提取诊断、治疗、病史等非症状信息

第二步：判断每个症状的严重程度
- 检查症状前后的程度修饰词
- 根据分级标准确定严重程度（0/1/2）
- 特别注意："无XX"或"X可"应标记为0级

第三步：标准化症状名称
- 优先使用标准症状词表中的名称
- 如果症状不在词表中，映射到最接近的标准名称
- 不要创建词表之外的新症状名

第四步：输出结构化结果
- 按照JSON Schema格式输出
- 确保所有症状都在标准词表中

**示例分析：**

输入："鼻咽癌放化疗后11年余，精神可，无头痛，咽干严重，偶有咳嗽，睡眠差，难入睡"

分析过程：
- 第一步识别症状：头痛(阴性)、咽干(阳性)、咳嗽(阳性)
- 第二步判断程度，尽量保持严谨：
  * 无头痛 → 0级
  * 咽干严重 → 2级
  * 偶有咳嗽 → 1级
  * 睡眠差 → 1级
  * 难入睡 → 1级
- 第三步标准化：睡眠→失眠，入睡难->失眠
- 第四步输出：JSON格式的症状列表

现在，请按照上述流程分析患者主诉。"""

    human_message = """患者主诉:
{chief_complaint}

请按照思维链步骤分析，并返回符合 JSON Schema 的结构化数据（只包含症状列表）。"""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def create_extraction_chain(llm: AzureChatOpenAI):
    """创建简化的症状提取链"""
    prompt = create_extraction_prompt()
    extraction_chain = prompt | llm.with_structured_output(schema=SimplifiedSymptomRecord)
    return extraction_chain


# ==================== Step 5: 简化的提取器 ====================

class SimplifiedSymptomExtractor:
    """简化的症状提取器"""

    def __init__(self, config: Optional[AzureConfig] = None):
        """
        初始化提取器

        Parameters
        ----------
        config : AzureConfig, optional
            Azure 配置，默认从环境变量加载
        """
        self.config = config or AzureConfig.from_env()
        self.llm = create_azure_llm(self.config)
        self.chain = create_extraction_chain(self.llm)
        self.errors = []

    def extract_single(self, chief_complaint: str, max_retries: int = 1) -> Optional[SimplifiedSymptomRecord]:
        """
        从单条主诉中提取症状（支持重试）

        Parameters
        ----------
        chief_complaint : str
            患者主诉文本
        max_retries : int
            最大重试次数（默认1次，总共最多2次机会）

        Returns
        -------
        SimplifiedSymptomRecord or None
            提取的症状记录，失败返回 None
        """
        for attempt in range(max_retries + 1):
            try:
                result = self.chain.invoke({"chief_complaint": chief_complaint})

                # 后处理：标准化症状名称
                if result and result.symptoms:
                    for symptom in result.symptoms:
                        standard_name = find_closest_symptom(symptom.name)
                        if standard_name:
                            symptom.name = standard_name

                return result  # 成功，直接返回

            except Exception as e:
                # 最后一次尝试失败，记录错误
                if attempt == max_retries:
                    self.errors.append({
                        "chief_complaint": chief_complaint[:100],
                        "error": str(e),
                        "attempts": attempt + 1
                    })
                    return None
                # 还有重试机会，继续循环

        return None

    def extract_batch(
        self,
        chief_complaints: List[str],
        batch_size: int = 10,
        show_progress: bool = True
    ) -> List[Optional[SimplifiedSymptomRecord]]:
        """
        批量提取症状

        Parameters
        ----------
        chief_complaints : List[str]
            患者主诉文本列表
        batch_size : int
            批处理大小
        show_progress : bool
            是否显示进度条

        Returns
        -------
        List[SimplifiedSymptomRecord or None]
            提取结果列表
        """
        results = []
        total = len(chief_complaints)

        if show_progress:
            with tqdm(total=total, desc="提取症状", unit="条") as pbar:
                for i in range(0, total, batch_size):
                    batch_texts = chief_complaints[i:i + batch_size]
                    batch_results = [self.extract_single(text) for text in batch_texts]
                    results.extend(batch_results)
                    pbar.update(len(batch_results))
        else:
            for i in range(0, total, batch_size):
                batch_texts = chief_complaints[i:i + batch_size]
                batch_results = [self.extract_single(text) for text in batch_texts]
                results.extend(batch_results)

        return results

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "chief_complaint",
        batch_size: int = 10
    ) -> pd.DataFrame:
        """
        从 DataFrame 中提取症状

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
        if text_column not in df.columns:
            raise ValueError(f"DataFrame 中没有列: {text_column}")

        # 过滤掉空值
        valid_df = df[df[text_column].notna()].copy()
        chief_complaints = valid_df[text_column].tolist()

        # 批量提取
        results = self.extract_batch(chief_complaints, batch_size=batch_size)

        # 将结果添加到 DataFrame
        valid_df = valid_df.reset_index(drop=True)
        valid_df["extracted_symptoms"] = results

        return valid_df


# ==================== Step 6: 结果后处理 ====================

def symptoms_to_dataframe(record: SimplifiedSymptomRecord) -> pd.DataFrame:
    """
    将症状记录转换为宽格式 DataFrame

    Parameters
    ----------
    record : SimplifiedSymptomRecord
        症状记录

    Returns
    -------
    pd.DataFrame
        每个标准症状一列，值为严重程度（0/1/2）
    """
    # 创建所有标准症状的列（初始值为0）
    data = {symptom: 0 for symptom in STANDARD_SYMPTOMS}

    # 填充提取的症状
    if record and record.symptoms:
        for symptom in record.symptoms:
            if symptom.name in STANDARD_SYMPTOMS:
                data[symptom.name] = symptom.severity

    return pd.DataFrame([data])


def save_with_json_column(
    original_df: pd.DataFrame,
    extracted_data: List,
    output_path: str = "Data/raw/npc_data_with_symptoms.csv"
) -> pd.DataFrame:
    """
    将提取的JSON结果作为新列，与原始数据一起保存

    Parameters
    ----------
    original_df : pd.DataFrame
        原始数据
    extracted_data : List
        提取的症状结果列表（可以是 SimplifiedSymptomRecord 或 .symptoms 列表）
    output_path : str
        输出文件路径（默认：Data/raw/npc_data_with_symptoms.csv）

    Returns
    -------
    pd.DataFrame
        包含提取结果的数据
    """
    import json

    # 复制原始数据
    result_df = original_df.copy()

    # 添加提取结果列
    extracted_json_list = []

    for item in extracted_data:
        # 处理不同类型的输入
        symptoms_list = None
        if item is None:
            symptoms_list = None
        elif hasattr(item, 'symptoms'):
            # SimplifiedSymptomRecord 对象
            symptoms_list = item.symptoms
        elif isinstance(item, list):
            # 已经是列表
            symptoms_list = item
        else:
            symptoms_list = None

        if symptoms_list:
            # 转换为JSON格式
            symptoms_data = []
            for s in symptoms_list:
                symptoms_data.append({
                    "name": s.name,
                    "severity": s.severity,
                    "label": {0: "无", 1: "轻度", 2: "重度"}[s.severity]
                })
            extracted_json_list.append(json.dumps(symptoms_data, ensure_ascii=False))
        else:
            # 提取失败，记录为空
            extracted_json_list.append("")

    # 添加新列
    result_df["extracted_symptoms"] = extracted_json_list

    # 保存为CSV
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存至: {output_path}")
    print(f"总共处理 {len(result_df)} 条记录")

    return result_df


def postprocess_results(
    df: pd.DataFrame,
    output_path: str = "outputs/symptoms_extracted.xlsx"
) -> pd.DataFrame:
    """
    后处理提取结果并保存

    Parameters
    ----------
    df : pd.DataFrame
        包含 extracted_symptoms 列的 DataFrame
    output_path : str
        输出 Excel 文件路径

    Returns
    -------
    pd.DataFrame
        处理后的 DataFrame（宽格式）
    """
    # 为每条记录转换为宽格式
    symptom_dfs = []

    for _, row in df.iterrows():
        symptom_df = symptoms_to_dataframe(row["extracted_symptoms"])

        # 添加原始数据的某些列
        for col in ["id", "patient_id", "age", "gender", "time"]:
            if col in row:
                symptom_df[col] = row[col]

        symptom_dfs.append(symptom_df)

    # 合并所有记录
    if symptom_dfs:
        result_df = pd.concat(symptom_dfs, ignore_index=True)

        # 保存为 Excel
        result_df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"结果已保存至: {output_path}")
        print(f"总共处理 {len(result_df)} 条记录")

        return result_df
    else:
        return pd.DataFrame()


# ==================== 主函数 ====================

def main():
    """主函数示例"""
    # 1. 加载配置
    config = AzureConfig.from_env()

    # 2. 创建提取器
    extractor = SimplifiedSymptomExtractor(config)

    # 3. 测试单条提取
    test_text = (
        "鼻咽癌放化疗后11年余，精神可，无头痛，咽干严重，无咳嗽，"
        "偶有耳鸣，睡眠差，难入睡，夜尿1-2次。"
    )

    print("=" * 60)
    print("测试单条提取")
    print("=" * 60)
    print(f"输入: {test_text}\n")

    result = extractor.extract_single(test_text)

    if result:
        print("提取结果:")
        for symptom in result.symptoms:
            print(f"  - {symptom.name}: 严重程度 {symptom.severity}")

        # 转换为 DataFrame
        symptom_df = symptoms_to_dataframe(result)
        print("\n宽格式:")
        print(symptom_df.T[symptom_df.T[0] > 0])

    # 4. 从CSV批量提取
    print("\n" + "=" * 60)
    print("批量提取")
    print("=" * 60)

    input_path = "Data/raw/npc_final.csv"
    output_path = "outputs/symptoms_extracted.xlsx"

    # 尝试不同编码读取
    encodings = ["utf-8", "gbk", "gb2312", "gb18030"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            print(f"✓ 使用编码: {encoding}")
            break
        except Exception:
            continue

    if df is None:
        print("✗ 无法读取 CSV 文件")
        return

    print(f"数据形状: {df.shape}")

    # 批量提取（限制前50条测试）
    test_df = df.head(50)
    result_df = extractor.extract_from_dataframe(
        test_df,
        text_column="chief_complaint",
        batch_size=5
    )

    print(f"\n提取完成!")
    print(f"成功: {len(result_df[result_df['extracted_symptoms'].notna()])} 条")
    print(f"失败: {len(extractor.errors)} 条")

    # 后处理
    output_df = postprocess_results(result_df, output_path)

    # 统计
    print("\n=== 症状统计 ===")
    symptom_cols = list(STANDARD_SYMPTOMS)
    symptom_counts = (output_df[symptom_cols] > 0).sum().sort_values(ascending=False)
    print("最常见的10个症状:")
    print(symptom_counts.head(10))


if __name__ == "__main__":
    main()
