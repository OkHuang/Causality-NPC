"""
测试 Azure OpenAI 配置是否正确
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.data.simplified_extraction import AzureConfig, SimplifiedSymptomExtractor

def test_env_loading():
    """测试环境变量加载"""
    print("=" * 60)
    print("【测试1】检查环境变量")
    print("=" * 60)

    load_dotenv()

    vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]

    all_ok = True
    for var in vars:
        value = os.getenv(var)
        if value:
            # 隐藏 API Key 的中间部分
            if "KEY" in var:
                masked = value[:8] + "..." + value[-8:]
                print(f"[OK] {var}: {masked}")
            else:
                print(f"[OK] {var}: {value}")
        else:
            print(f"[X] {var}: 未设置")
            all_ok = False

    return all_ok


def test_config_creation():
    """测试配置对象创建"""
    print("\n" + "=" * 60)
    print("【测试2】创建配置对象")
    print("=" * 60)

    try:
        config = AzureConfig.from_env()
        print("[OK] 配置对象创建成功")
        print(f"  Endpoint: {config.endpoint}")
        print(f"  Deployment: {config.deployment_name}")
        return True, config
    except ValueError as e:
        print(f"[X] 配置创建失败: {e}")
        return False, None


def test_llm_connection(config):
    """测试 LLM 连接"""
    print("\n" + "=" * 60)
    print("【测试3】测试 LLM 连接")
    print("=" * 60)

    try:
        extractor = SimplifiedSymptomExtractor(config)
        print("[OK] 提取器创建成功")

        # 测试简单提取
        test_text = "无头痛"
        print(f"\n测试文本: {test_text}")

        result = extractor.extract_single(test_text)

        if result:
            print("[OK] LLM 调用成功")
            print(f"  提取到 {len(result.symptoms)} 个症状")
            for s in result.symptoms:
                severity_label = {0: "无", 1: "轻度", 2: "重度"}[s.severity]
                print(f"  - {s.name}: {severity_label}")
            return True
        else:
            print("[X] 提取失败（返回None）")
            if extractor.errors:
                print(f"  错误: {extractor.errors[-1]['error']}")
            return False

    except Exception as e:
        print(f"[X] LLM 连接失败")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")

        # 常见错误提示
        error_str = str(e).lower()
        if "unauthorized" in error_str or "401" in error_str:
            print("\n  [TIP] API Key 可能错误")
        elif "not found" in error_str or "404" in error_str:
            print("\n  [TIP] Endpoint 或 Deployment Name 可能错误")
        elif "timeout" in error_str or "connection" in error_str:
            print("\n  [TIP] 网络连接问题，请检查 Endpoint")

        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Azure OpenAI 配置测试")
    print("=" * 60)

    # 测试1: 环境变量
    env_ok = test_env_loading()
    if not env_ok:
        print("\n[X] 环境变量未配置，请检查 .env 文件")
        return

    # 测试2: 配置对象
    config_ok, config = test_config_creation()
    if not config_ok:
        return

    # 测试3: LLM连接
    llm_ok = test_llm_connection(config)

    # 总结
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    if llm_ok:
        print("[OK] 所有测试通过！配置正确，可以开始使用。")
        print("\n下一步：运行示例")
        print("  python examples/simplified_extraction_example.py")
    else:
        print("[X] 部分测试失败，请根据上述提示修正配置。")

    print("=" * 60)


if __name__ == "__main__":
    main()
