"""
JSON处理工具函数

统一处理LLM响应中的JSON解析，去除重复代码。
"""

import json
from typing import Dict, Any


def extract_json_from_llm_response(text: str) -> Dict[str, Any]:
    """
    从LLM响应中提取JSON，统一处理Markdown包装

    Args:
        text: LLM返回的原始文本，可能包含Markdown代码块包装

    Returns:
        解析后的JSON字典

    Raises:
        json.JSONDecodeError: JSON解析失败时抛出
        ValueError: 文本格式异常时抛出
    """
    if not text or not text.strip():
        raise ValueError("输入文本为空")

    cleaned_text = text.strip()

    # 处理```json包装
    if "```json" in cleaned_text:
        parts = cleaned_text.split("```json", 1)
        if len(parts) > 1:
            cleaned_text = parts[1].split("```", 1)[0]
        else:
            raise ValueError("JSON代码块格式不完整")

    # 处理通用```包装
    elif "```" in cleaned_text:
        parts = cleaned_text.split("```", 1)
        if len(parts) > 1:
            # 取第一个代码块的内容
            cleaned_text = parts[1].split("```", 1)[0]
        else:
            raise ValueError("代码块格式不完整")

    # 清理空白字符
    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        raise ValueError("提取的JSON内容为空")

    return json.loads(cleaned_text)


def safe_json_loads(text: str, default=None) -> Any:
    """
    安全的JSON解析，失败时返回默认值

    Args:
        text: 要解析的JSON字符串
        default: 解析失败时的默认返回值

    Returns:
        解析结果或默认值
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def format_json_for_llm(data: Any, indent: int = 2) -> str:
    """
    格式化数据为LLM友好的JSON字符串

    Args:
        data: 要格式化的数据
        indent: 缩进空格数

    Returns:
        格式化的JSON字符串
    """
    return json.dumps(data, ensure_ascii=False, indent=indent)
