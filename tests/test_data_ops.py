"""
数据操作工具测试

测试utils/data_ops.py中的数据操作功能
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_ops import (
    write_jsonl_file, read_jsonl_file, write_json_file, read_json_file,
    sanitize_datetime_fields, create_temp_file_path, update_entities,
    update_abstract_map, merge_entity_data, merge_event_data,
    validate_entity_data, validate_event_data, cleanup_duplicate_entities,
    cleanup_duplicate_events, backup_data_file, restore_from_backup
)


class TestDataOperations:
    """数据操作基础功能测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_write_read_jsonl_file(self, temp_dir):
        """测试JSONL文件读写"""
        test_data = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200}
        ]
        file_path = temp_dir / "test.jsonl"

        # 写入
        write_jsonl_file(file_path, test_data)

        # 读取
        result = read_jsonl_file(file_path)

        assert len(result) == 2
        assert result[0]["name"] == "test1"
        assert result[1]["value"] == 200

    def test_write_read_json_file(self, temp_dir):
        """测试JSON文件读写"""
        test_data = {"config": {"debug": True, "port": 8080}}
        file_path = temp_dir / "test.json"

        # 写入
        write_json_file(file_path, test_data)

        # 读取
        result = read_json_file(file_path)

        assert result["config"]["debug"] is True
        assert result["config"]["port"] == 8080

    def test_sanitize_datetime_fields(self):
        """测试日期时间字段清理"""
        from datetime import datetime

        test_data = [
            {"id": 1, "timestamp": datetime(2023, 1, 1, 12, 0, 0)},
            {"id": 2, "created_at": datetime(2023, 1, 2, 13, 30, 0)}
        ]

        result = sanitize_datetime_fields(test_data)

        assert isinstance(result[0]["timestamp"], str)
        assert isinstance(result[1]["created_at"], str)
        assert "2023-01-01" in result[0]["timestamp"]
        assert "2023-01-02" in result[1]["created_at"]

    def test_create_temp_file_path(self, temp_dir):
        """测试临时文件路径创建"""
        base_dir = temp_dir / "temp"
        temp_path = create_temp_file_path(base_dir, prefix="test", suffix="data", extension="jsonl")

        assert temp_path.parent == base_dir
        assert temp_path.name.startswith("test")
        assert temp_path.name.endswith("data.jsonl")
        assert len(temp_path.name) > 20  # 包含时间戳

    @patch('src.utils.data_utils.safe_save_data')
    @patch('src.utils.data_utils.load_json_data')
    def test_update_entities(self, mock_load, mock_safe_save, temp_dir):
        """测试实体更新"""
        mock_load.return_value = {}  # 空字典表示没有现有数据
        mock_safe_save.return_value = True

        entities = ["实体A", "实体B"]
        entities_original = ["Entity A", "Entity B"]

        result = update_entities(entities, entities_original, "test_source", "2023-01-01")

        assert result is True
        mock_safe_save.assert_called_once()

    @patch('src.utils.data_utils.safe_save_data')
    @patch('src.utils.data_utils.load_json_data')
    def test_update_abstract_map(self, mock_load, mock_safe_save, temp_dir):
        """测试抽象映射更新"""
        mock_load.return_value = {}  # 空字典表示没有现有数据
        mock_safe_save.return_value = True

        extracted_list = [
            {"abstract": "事件1", "entities": ["实体A"], "event_summary": "摘要1"}
        ]

        result = update_abstract_map(extracted_list, "test_source", "2023-01-01")

        assert result is True
        mock_safe_save.assert_called_once()


class TestDataMerging:
    """数据合并功能测试"""

    def test_merge_entity_data(self):
        """测试实体数据合并"""
        target = {
            "first_seen": "2023-01-01",
            "sources": ["source1"],
            "original_forms": ["Entity A"]
        }

        source = {
            "first_seen": "2023-01-02",
            "sources": ["source2"],
            "original_forms": ["Entity B"]
        }

        merge_entity_data(target, source)

        assert "source2" in target["sources"]
        assert "Entity B" in target["original_forms"]
        assert target["first_seen"] == "2023-01-02"  # 取较新的时间

    def test_merge_event_data(self):
        """测试事件数据合并"""
        target = {
            "entities": ["实体A"],
            "event_summary": "Short summary",
            "sources": ["source1"],
            "first_seen": "2023-01-01"
        }

        source = {
            "entities": ["实体B"],
            "event_summary": "This is a much longer and more detailed summary of the event",
            "sources": ["source2"],
            "first_seen": "2023-01-02"
        }

        merge_event_data(target, source)

        assert "实体B" in target["entities"]
        assert "source2" in target["sources"]
        assert len(target["event_summary"]) > len("Short summary")  # 保留更长的摘要
        assert target["first_seen"] == "2023-01-02"  # 取较新的时间


class TestDataValidation:
    """数据验证功能测试"""

    def test_validate_entity_data_valid(self):
        """测试有效实体数据验证"""
        entity = {
            "first_seen": "2023-01-01",
            "sources": ["source1", "source2"],
            "original_forms": ["Entity A", "Entity B"]
        }

        errors = validate_entity_data(entity)
        assert len(errors) == 0

    def test_validate_entity_data_missing_fields(self):
        """测试缺少字段的实体数据验证"""
        entity = {
            "sources": ["source1"],
            "original_forms": ["Entity A"]
        }

        errors = validate_entity_data(entity)
        assert len(errors) > 0
        assert any("first_seen" in error for error in errors)

    def test_validate_entity_data_invalid_types(self):
        """测试类型无效的实体数据验证"""
        entity = {
            "first_seen": "2023-01-01",
            "sources": "not_a_list",  # 应该是列表
            "original_forms": ["Entity A"]
        }

        errors = validate_entity_data(entity)
        assert len(errors) > 0
        assert any("sources" in error for error in errors)

    def test_validate_event_data_valid(self):
        """测试有效事件数据验证"""
        event = {
            "entities": ["实体A", "实体B"],
            "event_summary": "事件摘要",
            "sources": ["source1"],
            "first_seen": "2023-01-01"
        }

        errors = validate_event_data(event)
        assert len(errors) == 0

    def test_validate_event_data_missing_fields(self):
        """测试缺少字段的事件数据验证"""
        event = {
            "entities": ["实体A"],
            "sources": ["source1"]
        }

        errors = validate_event_data(event)
        assert len(errors) > 0
        assert any("event_summary" in error for error in errors)


class TestDataCleanup:
    """数据清理功能测试"""

    def test_cleanup_duplicate_entities(self):
        """测试实体去重清理"""
        entities_dict = {
            "entity1": {"sources": ["src1"], "original_forms": ["Form1"]},
            "entity2": {"sources": ["src2"], "original_forms": ["Form2"]}
        }

        stats = cleanup_duplicate_entities(entities_dict)

        assert isinstance(stats, dict)
        assert "removed" in stats
        assert "merged" in stats

    def test_cleanup_duplicate_events(self):
        """测试事件去重清理"""
        events_dict = {
            "event1": {"entities": ["A"], "event_summary": "Summary1"},
            "event2": {"entities": ["B"], "event_summary": "Summary2"}
        }

        stats = cleanup_duplicate_events(events_dict)

        assert isinstance(stats, dict)
        assert "removed" in stats
        assert "merged" in stats


class TestDataBackup:
    """数据备份恢复功能测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_backup_data_file(self, temp_dir):
        """测试数据文件备份"""
        source_file = temp_dir / "source.json"
        source_file.write_text('{"test": "data"}')

        backup_path = backup_data_file(source_file)

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.name.endswith(".bak")
        assert backup_path.read_text() == '{"test": "data"}'

    def test_backup_nonexistent_file(self):
        """测试备份不存在的文件"""
        nonexistent = Path("nonexistent.json")
        backup_path = backup_data_file(nonexistent)

        assert backup_path is None

    def test_restore_from_backup(self, temp_dir):
        """测试从备份恢复"""
        backup_file = temp_dir / "backup.json.bak"
        backup_file.write_text('{"restored": "data"}')

        target_file = temp_dir / "target.json"

        result = restore_from_backup(backup_file, target_file)

        assert result is True
        assert target_file.exists()
        assert target_file.read_text() == '{"restored": "data"}'

    def test_restore_from_nonexistent_backup(self, temp_dir):
        """测试从不存在的备份恢复"""
        nonexistent_backup = Path("nonexistent.bak")
        target_file = temp_dir / "target.json"

        result = restore_from_backup(nonexistent_backup, target_file)

        assert result is False
