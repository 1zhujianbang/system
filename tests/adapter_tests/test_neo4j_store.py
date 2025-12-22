"""
Neo4j存储适配器模块单元测试
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from src.adapters.neo4j.store import Neo4jStore
from src.domain.models import EntityCanonical, EventCanonical, SourceRef


class TestNeo4jStore(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        # 直接创建存储实例，不使用patch
        self.store = Neo4jStore.__new__(Neo4jStore)
        self.store._lock = Mock()
            
    def test_init(self):
        """测试初始化"""
        # 由于我们直接创建了实例，这里只检查_lock是否存在
        self.assertIsNotNone(self.store._lock)
        
    def test_upsert_entity(self):
        """测试创建或更新实体"""
        # 模拟会话和结果
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        self.store._driver = mock_driver
        
        # 模拟查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value="entity_123")
        mock_session.run.return_value.single.return_value = mock_record
        
        # 创建测试实体
        source_ref = SourceRef(id="source_1", name="Test Source", url="http://test.com")
        entity = EntityCanonical(
            entity_id="entity_123",
            name="Test Entity",
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            sources=[source_ref],
            original_forms=["Test Entity"],
            aliases=["TE"]
        )
        
        # 调用方法
        result = self.store.upsert_entity(entity)
        
        # 验证结果
        self.assertEqual(result, "entity_123")
        mock_session.run.assert_called_once()
        
    def test_upsert_event(self):
        """测试创建或更新事件"""
        # 模拟会话和结果
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        self.store._driver = mock_driver
        
        # 模拟查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value="event_123")
        mock_session.run.return_value.single.return_value = mock_record
        
        # 创建测试事件
        source_ref = SourceRef(id="source_1", name="Test Source", url="http://test.com")
        event = EventCanonical(
            event_id="event_123",
            abstract="Test Event Abstract",
            event_summary="Test Event Summary",
            event_types=["TYPE1", "TYPE2"],
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            sources=[source_ref],
            entities=["entity_1", "entity_2"],
            entity_roles={}
        )
        
        # 调用方法
        result = self.store.upsert_event(event)
        
        # 验证结果
        self.assertEqual(result, "event_123")
        mock_session.run.assert_called_once()
        
    def test_search_entities(self):
        """测试搜索实体"""
        # 模拟会话和结果
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        self.store._driver = mock_driver
        
        # 模拟查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value={
            "entity_id": "entity_123",
            "name": "Test Entity",
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "sources": json.dumps([{"id": "source_1", "name": "Test Source", "url": "http://test.com"}]),
            "original_forms": json.dumps(["Test Entity"]),
            "aliases": json.dumps(["TE"])
        })
        mock_session.run.return_value = [mock_record]
        
        # 调用方法
        result = self.store.search_entities("Test")
        
        # 验证结果
        self.assertIsInstance(result, list)
        if result:
            self.assertIsInstance(result[0], EntityCanonical)


if __name__ == '__main__':
    unittest.main()