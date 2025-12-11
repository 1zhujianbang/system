"""
LLM工具函数测试

测试utils/llm_utils.py中的LLM相关功能
"""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_utils import (
    call_llm_with_retry, create_extraction_prompt,
    create_deduplication_prompt, create_event_deduplication_prompt
)


class TestLLMCalls:
    """LLM调用功能测试"""

    def test_create_extraction_prompt(self):
        """测试实体提取提示创建"""
        title = "新闻标题"
        content = "新闻内容"
        entity_definitions = "实体定义"

        prompt = create_extraction_prompt(title, content, entity_definitions)

        assert title in prompt
        assert content in prompt
        assert entity_definitions in prompt
        assert "提取" in prompt or "extract" in prompt.lower()

    def test_create_deduplication_prompt(self):
        """测试实体去重提示创建"""
        entities_batch = {
            "entity1": {"sources": ["src1"], "original_forms": ["Form1"]},
            "entity2": {"sources": ["src2"], "original_forms": ["Form2"]}
        }
        evidence_map = {
            "entity1": ["evidence1", "evidence2"],
            "entity2": ["evidence3"]
        }

        prompt = create_deduplication_prompt(entities_batch, evidence_map)

        assert "entity1" in prompt
        assert "entity2" in prompt
        assert "evidence1" in prompt
        assert "重复" in prompt or "duplicate" in prompt.lower()

    def test_create_event_deduplication_prompt(self):
        """测试事件去重提示创建"""
        events_batch = {
            "event1": {
                "entities": ["实体A"],
                "event_summary": "事件摘要1",
                "sources": ["src1"]
            },
            "event2": {
                "entities": ["实体B"],
                "event_summary": "事件摘要2",
                "sources": ["src2"]
            }
        }

        prompt = create_event_deduplication_prompt(events_batch)

        assert "事件摘要1" in prompt
        assert "事件摘要2" in prompt
        assert "实体A" in prompt
        assert "实体B" in prompt
        assert "重复" in prompt or "duplicate" in prompt.lower()

    def test_create_extraction_prompt(self):
        """测试实体提取提示创建"""
        title = "新闻标题"
        content = "新闻内容"
        entity_definitions = "实体定义"

        prompt = create_extraction_prompt(title, content, entity_definitions)

        assert title in prompt
        assert content in prompt
        assert entity_definitions in prompt
        assert "提取" in prompt or "extract" in prompt.lower()

    def test_create_deduplication_prompt(self):
        """测试实体去重提示创建"""
        entities_batch = {
            "entity1": {"sources": ["src1"], "original_forms": ["Form1"]},
            "entity2": {"sources": ["src2"], "original_forms": ["Form2"]}
        }
        evidence_map = {
            "entity1": ["evidence1", "evidence2"],
            "entity2": ["evidence3"]
        }

        prompt = create_deduplication_prompt(entities_batch, evidence_map)

        assert "entity1" in prompt
        assert "entity2" in prompt
        assert "evidence1" in prompt
        assert "重复" in prompt or "duplicate" in prompt.lower()

    def test_create_event_deduplication_prompt(self):
        """测试事件去重提示创建"""
        events_batch = {
            "event1": {
                "entities": ["实体A"],
                "event_summary": "事件摘要1",
                "sources": ["src1"]
            },
            "event2": {
                "entities": ["实体B"],
                "event_summary": "事件摘要2",
                "sources": ["src2"]
            }
        }

        prompt = create_event_deduplication_prompt(events_batch)

        assert "事件摘要1" in prompt
        assert "事件摘要2" in prompt
        assert "实体A" in prompt
        assert "实体B" in prompt
        assert "重复" in prompt or "duplicate" in prompt.lower()


class TestPromptContent:
    """提示内容测试"""

    def test_extraction_prompt_structure(self):
        """测试提取提示的结构"""
        prompt = create_extraction_prompt("Title", "Content", "Definitions")

        # 检查是否包含必要的部分
        assert "Title" in prompt
        assert "Content" in prompt
        assert "Definitions" in prompt

        # 检查是否包含输出格式要求
        assert "JSON" in prompt or "json" in prompt.lower()

    def test_deduplication_prompt_format(self):
        """测试去重提示的格式"""
        entities_batch = {"entity1": {"sources": ["src1"]}}
        evidence_map = {"entity1": ["evidence1"]}

        prompt = create_deduplication_prompt(entities_batch, evidence_map)

        # 检查是否包含批次信息
        assert "entity1" in prompt
        assert "evidence1" in prompt

        # 检查是否包含格式要求
        assert "格式" in prompt or "format" in prompt.lower()

    def test_event_deduplication_prompt_format(self):
        """测试事件去重提示的格式"""
        events_batch = {
            "event1": {
                "entities": ["实体A"],
                "event_summary": "摘要1",
                "sources": ["src1"]
            }
        }

        prompt = create_event_deduplication_prompt(events_batch)

        # 检查是否包含事件信息
        assert "摘要1" in prompt
        assert "实体A" in prompt

        # 检查是否包含格式要求
        assert "格式" in prompt or "format" in prompt.lower()
        assert "摘要" in prompt or "summary" in prompt.lower()
        assert "参与实体" in prompt or "entities" in prompt.lower()
