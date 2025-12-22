"""
社区检测模块单元测试
"""
import unittest
from unittest.mock import Mock, patch
import networkx as nx
from datetime import datetime

from src.app.business.community_detection import CommunityDetector


class TestCommunityDetector(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        # 直接模拟detector的store属性
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        mock_store = Mock()
        mock_store._driver = mock_driver
        
        # 创建detector并直接设置store属性
        self.detector = CommunityDetector.__new__(CommunityDetector)
        self.detector.store = mock_store
        # 手动设置logger
        with patch('src.app.business.community_detection.get_logger') as mock_get_logger:
            mock_get_logger.return_value = Mock()
            self.detector.logger = mock_get_logger.return_value
        
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector.logger)
        
    @patch('src.app.business.community_detection.get_neo4j_store')
    def test_find_central_entities(self, mock_get_store):
        """测试查找中心实体"""
        # 创建测试图
        graph = nx.Graph()
        graph.add_edge('A', 'B', weight=1.0)
        graph.add_edge('A', 'C', weight=0.5)
        graph.add_edge('B', 'C', weight=0.8)
        
        # 调用方法
        result = self.detector._find_central_entities(graph)
        
        # 验证结果
        self.assertIsInstance(result, list)
        if result:  # 如果有结果
            self.assertIsInstance(result[0], dict)
            self.assertIn('name', result[0])
            self.assertIn('centrality_score', result[0])
            
    def test_get_community_by_entity(self):
        """测试根据实体获取社区信息"""
        # 直接使用setUp中设置的mocked store
            
        # 获取mock session
        mock_session = self.detector.store._driver.session()
            
        # 模拟查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            'community_id': 'community_1', 
            'size': 5,
            'density': 0.6,
            'central_entities': [{'name': 'A', 'centrality_score': 0.8}]
        }[key])
        mock_session.run.return_value.single.return_value = mock_record
            
        # 调用方法
        result = self.detector.get_community_by_entity('A')
            
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result['community_id'], 'community_1')
        self.assertEqual(result['size'], 5)
        
    def test_detect_viewpoint_camps(self):
        """测试检测观点阵营"""
        # 直接使用setUp中设置的mocked store
        
        # 获取mock session
        mock_session = self.detector.store._driver.session()
        
        # 模拟查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            'source': 'A',
            'target': 'B',
            'avg_tone': 0.5,
            'relation_count': 3
        }[key])
        mock_session.run.return_value.single.return_value = mock_record
        mock_session.run.return_value = [mock_record]  # 返回可迭代对象
        
        # 调用方法
        result = self.detector.detect_viewpoint_camps(('2023-01-01', '2023-01-31'))
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn('positive_camp', result)
        self.assertIn('negative_camp', result)


if __name__ == '__main__':
    unittest.main()