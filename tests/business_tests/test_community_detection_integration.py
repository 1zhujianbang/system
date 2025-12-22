"""
社区检测集成测试
"""
import unittest
from unittest.mock import Mock, patch
import networkx as nx

from src.app.business.community_detection import CommunityDetector
from src.adapters.neo4j.store import Neo4jStore


class TestCommunityDetectionIntegration(unittest.TestCase):
    
    @patch('src.adapters.neo4j.store.GraphDatabase')
    def setUp(self, mock_graph_db):
        """测试初始化"""
        self.store = Neo4jStore("bolt://localhost:7687", ("neo4j", "password"))
        self.detector = CommunityDetector()
        
    @patch('src.app.business.community_detection.get_neo4j_store')
    def test_community_detection_flow(self, mock_get_store):
        """测试社区检测完整流程"""
        # 模拟存储
        mock_get_store.return_value = self.store
        
        # 模拟存储会话和结果
        mock_session = Mock()
        mock_driver = Mock()
        mock_driver.session.return_value = mock_session
        self.store._driver = mock_driver
        
        # 模拟数据库查询结果
        mock_session.run.return_value = [
            Mock(__getitem__=Mock(side_effect=lambda key: {
                "entity1": "EntityA",
                "entity2": "EntityB",
                "avg_strength": 0.8
            }[key]))
        ]
        
        # 调用方法
        result = self.detector.detect_communities_louvain(("2023-01-01", "2023-01-31"))
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("communities", result)
        self.assertIn("statistics", result)
        
    @patch('src.app.business.community_detection.get_neo4j_store')
    def test_viewpoint_camps_flow(self, mock_get_store):
        """测试观点阵营检测完整流程"""
        # 模拟存储
        mock_get_store.return_value = self.store
        
        # 模拟存储会话和结果
        mock_session = Mock()
        mock_driver = Mock()
        mock_driver.session.return_value = mock_session
        self.store._driver = mock_driver
        
        # 模拟数据库查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            "source": "EntityA",
            "target": "EntityB",
            "avg_tone": 0.5,
            "relation_count": 3
        }[key])
        mock_session.run.return_value.single.return_value = mock_record
        mock_session.run.return_value = [mock_record]
        
        # 调用方法
        result = self.detector.detect_viewpoint_camps(("2023-01-01", "2023-01-31"))
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("positive_camp", result)
        self.assertIn("negative_camp", result)


if __name__ == '__main__':
    unittest.main()