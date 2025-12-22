"""
因果网络模块单元测试
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import pandas as pd
from datetime import datetime

from src.app.business.causal_network import CausalNetworkBuilder


class TestCausalNetworkBuilder(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        # 直接模拟builder的store属性
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        mock_store = Mock()
        mock_store._driver = mock_driver
        
        # 创建builder并直接设置store属性
        self.builder = CausalNetworkBuilder.__new__(CausalNetworkBuilder)
        self.builder.store = mock_store
        self.builder.graph = nx.DiGraph()
        # 手动设置logger
        with patch('src.app.business.causal_network.get_logger') as mock_get_logger:
            mock_get_logger.return_value = Mock()
            self.builder.logger = mock_get_logger.return_value
        
    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.builder.graph, nx.DiGraph)
        self.assertIsNotNone(self.builder.logger)
        
    @patch('src.app.business.causal_network.get_neo4j_store')
    def test_build_event_sequences(self, mock_get_store):
        """测试构建事件序列"""
        # 模拟存储返回
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        mock_driver.session.return_value = mock_session
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_store = Mock()
        mock_store._driver = mock_driver
        mock_get_store.return_value = mock_store
        
        # 准备测试数据
        events = [
            {"event_code": "010", "date": "2023-01-01", "goldstein_scale": 1.0, "num_mentions": 5},
            {"event_code": "010", "date": "2023-01-02", "goldstein_scale": 1.2, "num_mentions": 3},
            {"event_code": "020", "date": "2023-01-01", "goldstein_scale": -1.0, "num_mentions": 2}
        ]
        
        # 调用方法
        result = self.builder._build_event_sequences(events)
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 2)  # 两种事件类型
        self.assertEqual(len(result), 2)  # 两天的数据
        
    @patch('src.app.business.causal_network.get_neo4j_store')
    def test_calculate_edge_weight(self, mock_get_store):
        """测试计算边权重"""
        # 模拟存储返回
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        mock_driver.session.return_value = mock_session
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_store = Mock()
        mock_store._driver = mock_driver
        mock_get_store.return_value = mock_store
        
        # 准备测试数据
        data = pd.DataFrame({
            '010': [1, 2, 3],
            '020': [2, 1, 2]
        })
        
        # 调用方法
        weight = self.builder._calculate_edge_weight(data, '010', '020')
        
        # 验证结果
        self.assertIsInstance(weight, float)
        self.assertGreaterEqual(weight, 0)
        
    def test_query_causal_probability(self):
        """测试查询因果概率"""
        # 直接使用setUp中设置的mocked store
        
        # 获取mock session
        mock_session = self.builder.store._driver.session()
        
        # 模拟查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            'probability': 0.75,
            'created_at': '2023-01-01T00:00:00'
        }[key])
        mock_session.run.return_value.single.return_value = mock_record
        
        # 调用方法
        result = self.builder.query_causal_probability('010', '020')
        
        # 验证结果
        self.assertEqual(result['source'], '010')
        self.assertEqual(result['target'], '020')
        self.assertEqual(result['probability'], 0.75)
        
    def test_trace_causal_chain(self):
        """测试追踪因果链"""
        # 直接使用setUp中设置的mocked store
        
        # 获取mock session
        mock_session = self.builder.store._driver.session()
        
        # 模拟查询结果 - 需要使结果可迭代
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value=['010', '020', '030'])
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        mock_session.run.return_value = mock_result
        
        # 调用方法
        result = self.builder.trace_causal_chain('010')
        
        # 验证结果
        self.assertIsInstance(result, list)
        if result:  # 如果有结果
            self.assertIsInstance(result[0], list)


if __name__ == '__main__':
    unittest.main()