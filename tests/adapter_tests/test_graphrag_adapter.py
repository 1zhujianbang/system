"""
GraphRAG适配器模块单元测试
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.adapters.llm.graphrag_adapter import GraphRAGAdapter
from src.ports.llm_client import LLMResponse, LLMCallConfig, LLMProviderType


class TestGraphRAGAdapter(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        # 模拟LLM客户端
        self.mock_llm_client = Mock()
        
        # 直接创建适配器，不使用patch
        self.adapter = GraphRAGAdapter.__new__(GraphRAGAdapter)
        self.adapter._llm_client = self.mock_llm_client
        self.adapter._graph_database_uri = "bolt://localhost:7687"
        self.adapter._graph_database_auth = ("neo4j", "password")
        self.adapter._logger = Mock()
            
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.adapter._llm_client, self.mock_llm_client)
        self.assertEqual(self.adapter._graph_database_uri, "bolt://localhost:7687")
        
    def test_query_graph_success(self):
        """测试成功查询图谱"""
        # 模拟会话和结果
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        self.adapter._driver = mock_driver
        
        # 模拟图数据库查询结果
        # 第一次调用返回实体
        mock_entity_record = Mock()
        mock_entity_record.__getitem__ = Mock(side_effect=lambda key: {
            "name": "Test Entity",
            "description": "Test Description",
            "score": 0.9
        }[key])
        
        # 第二次调用返回事件
        mock_event_record = Mock()
        mock_event_record.__getitem__ = Mock(side_effect=lambda key: {
            "abstract": "Test Event",
            "summary": "Test Summary",
            "score": 0.8
        }[key])
        
        # 第三次调用返回关系
        mock_relation_record = Mock()
        mock_relation_record.__getitem__ = Mock(side_effect=lambda key: {
            "source": "Entity1",
            "relation": "related_to",
            "target": "Entity2",
            "description": "Test Relation"
        }[key])
        
        # 设置session.run的返回值序列
        mock_session.run.side_effect = [
            [mock_entity_record],  # 实体查询结果
            [mock_event_record],   # 事件查询结果
            [mock_relation_record] # 关系查询结果
        ]
        
        # 模拟LLM响应
        mock_llm_response = LLMResponse(
            content="This is a test response",
            model="test-model",
            provider=LLMProviderType.OPENAI
        )
        self.mock_llm_client.call.return_value = mock_llm_response
        
        # 调用方法
        result = self.adapter.query_graph("Test query")
        
        # 验证结果
        self.assertTrue(result.success)
        self.assertEqual(result.content, "This is a test response")
        self.mock_llm_client.call.assert_called_once()
        
    def test_query_graph_failure(self):
        """测试查询图谱失败"""
        # 模拟会话异常
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        self.adapter._driver = mock_driver
        
        # 模拟图数据库查询异常
        mock_session.run.side_effect = Exception("Database error")
        
        # 调用方法
        result = self.adapter.query_graph("Test query")
        
        # 验证结果
        self.assertFalse(result.success)
        self.assertIn("Database error", result.error)
        
    def test_construct_enhanced_prompt(self):
        """测试构造增强提示词"""
        # 准备测试上下文
        context = {
            "entities": [
                {"name": "Entity1", "description": "Description1", "score": 0.9}
            ],
            "events": [
                {"abstract": "Event1", "summary": "Summary1", "score": 0.8}
            ],
            "relations": [
                {"source": "Entity1", "relation": "related_to", "target": "Entity2", "description": "Relation description"}
            ]
        }
        
        # 调用方法
        result = self.adapter._construct_enhanced_prompt("Test query", context)
        
        # 验证结果
        self.assertIsInstance(result, str)
        self.assertIn("Test query", result)
        self.assertIn("Entity1", result)
        self.assertIn("Event1", result)
        self.assertIn("related_to", result)


if __name__ == '__main__':
    unittest.main()