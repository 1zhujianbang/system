"""
动态仿真引擎模块单元测试
"""
import unittest
from unittest.mock import Mock, patch
import networkx as nx
from datetime import datetime

from src.app.business.dynamic_simulation import DynamicSimulationEngine


class TestDynamicSimulationEngine(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        # 直接模拟engine的store属性
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        mock_store = Mock()
        mock_store._driver = mock_driver
        
        # 创建engine并直接设置store属性
        self.engine = DynamicSimulationEngine.__new__(DynamicSimulationEngine)
        self.engine.store = mock_store
        self.engine.entity_network = nx.Graph()
        # 手动设置logger
        with patch('src.app.business.dynamic_simulation.get_logger') as mock_get_logger:
            mock_get_logger.return_value = Mock()
            self.engine.logger = mock_get_logger.return_value
        
    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.engine.entity_network, nx.Graph)
        self.assertIsNotNone(self.engine.logger)
        
    def test_load_simulation_rules(self):
        """测试加载仿真规则"""
        # 直接使用setUp中设置的mocked store
        
        # 获取mock session
        mock_session = self.engine.store._driver.session()
        
        # 模拟查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            'rule_type': 'conflict_update',
            'parameters': {'event_code': '190', 'probability': 0.7}
        }[key])
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        mock_session.run.return_value = mock_result
        
        # 模拟查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            'rule_type': 'conflict_update',
            'parameters': {'event_code': '190', 'probability': 0.7}
        }[key])
        mock_session.run.return_value.single.return_value = mock_record
        mock_session.run.return_value = [mock_record]  # 返回可迭代对象
        
        # 调用方法
        rules = self.engine._load_simulation_rules()
        
        # 验证结果
        self.assertIsInstance(rules, dict)
        self.assertIn('conflict_update', rules)
        
    def test_capture_network_state(self):
        """测试捕获网络状态"""
        # 直接使用setUp中设置的mocked store
        
        # 获取mock session
        mock_session = self.engine.store._driver.session()
        
        # 创建测试网络
        self.engine.entity_network.add_node('A', state={'influence': 0.5})
        self.engine.entity_network.add_node('B', state={'influence': 0.8})
        self.engine.entity_network.add_edge('A', 'B', weight=0.7)
        
        # 调用方法
        state = self.engine._capture_network_state()
        
        # 验证结果
        self.assertIsInstance(state, dict)
        self.assertIn('nodes', state)
        self.assertIn('edges', state)
        self.assertEqual(len(state['nodes']), 2)
        self.assertEqual(len(state['edges']), 1)
        
    @patch('src.app.business.dynamic_simulation.get_neo4j_store')
    def test_detect_key_events(self, mock_get_store):
        """测试检测关键事件"""
        # 准备测试状态
        current_state = {
            'nodes': {
                'A': {'influence': 0.9},  # 高影响力
                'B': {'influence': 0.1},  # 低影响力
                'C': {'influence': 0.5}   # 正常影响力
            },
            'edges': {},
            'timestamp': '2023-01-01T00:00:00'
        }
        
        # 调用方法
        events = self.engine._detect_key_events(current_state)
        
        # 验证结果
        self.assertIsInstance(events, list)
        # 应该检测到两个关键事件（A和B）
        self.assertGreaterEqual(len(events), 0)


if __name__ == '__main__':
    unittest.main()