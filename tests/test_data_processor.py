import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_utils import DataNormalizer


class TestDataNormalizer:
    """DataNormalizer 单元测试"""

    @pytest.fixture
    def data_normalizer(self):
        """创建数据标准化器实例"""
        return DataNormalizer()

    def test_normalize_dict_list(self, data_normalizer):
        """测试字典列表标准化"""
        input_data = [
            {"title": "News 1", "content": "Content 1"},
            {"title": "News 2", "content": "Content 2"}
        ]

        result = data_normalizer.normalize_event_input(input_data)
        assert len(result) == 2
        assert result[0]["title"] == "News 1"
        assert result[1]["title"] == "News 2"

    def test_normalize_json_string(self, data_normalizer):
        """测试JSON字符串标准化"""
        json_str = '[{"title": "News 1"}, {"title": "News 2"}]'
        result = data_normalizer.normalize_event_input(json_str)
        assert len(result) == 2
        assert result[0]["title"] == "News 1"

    def test_normalize_single_dict(self, data_normalizer):
        """测试单个字典标准化"""
        input_data = {"title": "Single News", "content": "Content"}
        result = data_normalizer.normalize_event_input(input_data)
        assert len(result) == 1
        assert result[0]["title"] == "Single News"

    def test_normalize_file_path(self, data_normalizer):
        """测试文件路径标准化"""
        mock_data = [
            {"title": "File News 1"},
            {"title": "File News 2"}
        ]

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_data))):

            result = data_normalizer.normalize_event_input("test.json")
            assert len(result) == 2
            assert result[0]["title"] == "File News 1"

    def test_normalize_invalid_json_string(self, data_normalizer):
        """测试无效JSON字符串"""
        invalid_json = '{"title": "News", invalid}'
        result = data_normalizer.normalize_event_input(invalid_json)
        assert result == []  # 应该返回空列表

    def test_normalize_file_not_found(self, data_normalizer):
        """测试文件不存在"""
        with patch('pathlib.Path.exists', return_value=False):
            result = data_normalizer.normalize_event_input("nonexistent.json")
            assert result == []

    def test_normalize_empty_input(self, data_normalizer):
        """测试空输入"""
        result = data_normalizer.normalize_event_input([])
        assert result == []

        result = data_normalizer.normalize_event_input("")
        assert result == []

        result = data_normalizer.normalize_event_input(None)
        assert result == []

    def test_normalize_nested_structures(self, data_normalizer):
        """测试嵌套结构标准化"""
        nested_data = {
            "articles": [
                {"title": "News 1", "meta": {"author": "Author 1"}},
                {"title": "News 2", "meta": {"author": "Author 2"}}
            ]
        }

        result = data_normalizer.normalize_event_input(nested_data)
        # 应该只处理顶层字典，返回包含nested_data的列表
        assert len(result) == 1
        assert "articles" in result[0]

    def test_normalize_with_event_keys(self, data_normalizer):
        """测试包含事件键的标准化"""
        events_dict = {
            "event1": {"title": "Event 1", "entities": ["A", "B"]},
            "event2": {"title": "Event 2", "entities": ["C", "D"]}
        }

        result = data_normalizer.normalize_event_input(events_dict)
        assert len(result) == 2
        # 结果应该包含abstract字段
        assert "abstract" in result[0] or "title" in result[0]

    def test_data_cleaning(self, data_normalizer):
        """测试数据清理功能"""
        dirty_data = [
            {"title": "  News with spaces  ", "content": None},
            {"title": "", "content": "Valid content"},
            {"title": "Valid title", "content": "Valid content"}
        ]

        result = data_normalizer.normalize_event_input(dirty_data)
        assert len(result) == 3

        # 检查数据清理
        assert result[0]["title"] == "News with spaces"  # 去除了前后空格

    def test_large_dataset_handling(self, data_normalizer):
        """测试大数据集处理"""
        # 创建1000个新闻项目
        large_dataset = [
            {"title": f"News {i}", "content": f"Content {i}"}
            for i in range(1000)
        ]

        result = data_normalizer.normalize_event_input(large_dataset)
        assert len(result) == 1000

        # 验证所有项目都被正确处理
        for i, item in enumerate(result):
            assert item["title"] == f"News {i}"
            assert item["content"] == f"Content {i}"

    def test_mixed_data_types(self, data_normalizer):
        """测试混合数据类型"""
        mixed_data = [
            {"title": "String title", "count": 42},
            {"title": "Another title", "count": "25"},  # 字符串数字
            {"title": "Third title", "active": True}
        ]

        result = data_normalizer.normalize_event_input(mixed_data)
        assert len(result) == 3

        # 验证数据类型保持
        assert isinstance(result[0]["count"], int)
        assert isinstance(result[1]["count"], str)
        assert isinstance(result[2]["active"], bool)

    def test_unicode_handling(self, data_normalizer):
        """测试Unicode字符处理"""
        unicode_data = [
            {"title": "新闻标题", "content": "新闻内容"},
            {"title": "News Title", "content": "News Content"}
        ]

        result = data_normalizer.normalize_event_input(unicode_data)
        assert len(result) == 2
        assert result[0]["title"] == "新闻标题"
        assert result[1]["title"] == "News Title"

    def test_error_resilience(self, data_normalizer):
        """测试错误恢复能力"""
        problematic_data = [
            {"title": "Valid news"},
            "invalid_string_entry",
            {"title": None},
            {"title": "Another valid", "content": "Content"}
        ]

        result = data_normalizer.normalize_event_input(problematic_data)

        # 应该只返回有效的字典条目
        valid_entries = [item for item in result if isinstance(item, dict) and item.get("title")]
        assert len(valid_entries) >= 2  # 至少有两个有效条目
