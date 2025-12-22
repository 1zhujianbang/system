"""
动态仿真引擎集成测试
"""
import unittest
from unittest.mock import Mock, patch
import networkx as nx

from src.app.business.dynamic_simulation import DynamicSimulationEngine
from src.adapters.neo4j.store import Neo4jStore


class TestDynamicSimulationIntegration(unittest.TestCase):
    
    @patch('src.adapters.neo4j.store.GraphDatabase')
    def setUp(self, mock_graph_db):
        """测试初始化"""
        self.store = Neo4jStore("bolt://localhost:7687", ("neo4j", "password"))
        self.engine = DynamicSimulationEngine()
        
    @patch('src.app.business.dynamic_simulation.get_neo4j_store')
    def test_simulation_initialization_flow(self, mock_get_store):
        """测试仿真初始化完整流程"""
        # 模拟存储
        mock_get_store.return_value = self.store
        
        # 模拟存储会话和结果
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        self.store._driver = mock_driver
        
        # 模拟数据库查询结果
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            "entity1": "EntityA",
            "entity2": "EntityB",
            "avg_strength": 0.8,
            "relation_count": 2
        }[key])
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        mock_session.run.return_value = mock_result
        
        # 调用方法
        self.engine.initialize_simulation(("2023-01-01", "2023-01-31"))
        
        # 验证结果
        self.assertGreater(len(self.engine.entity_network.nodes()), 0)
        
    @patch('src.app.business.dynamic_simulation.get_neo4j_store')
    def test_simulation_run_flow(self, mock_get_store):
        """测试仿真运行完整流程"""
        # 模拟存储
        mock_get_store.return_value = self.store
        
        # 模拟存储会话和结果
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver = Mock()
        mock_driver.session = Mock(return_value=mock_session)
        self.store._driver = mock_driver
        
        # 初始化仿真网络
        self.engine.entity_network.add_node('A', state={'influence': 0.5})
        self.engine.entity_network.add_node('B', state={'influence': 0.8})
        self.engine.entity_network.add_edge('A', 'B', weight=0.7)
        
        # 模拟数据库查询结果
        mock_session.run.return_value = [
            Mock(__getitem__=Mock(side_effect=lambda key: {
                "rule_type": "conflict_update",
                "parameters": {"event_code": "190", "probability": 0.7}
            }[key]))
        ]
        
        # 调用方法
        result = self.engine.run_simulation(iterations=2)
        
        # 验证结果
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()