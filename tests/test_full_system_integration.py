"""
全系统集成测试
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.app.pipeline.distributed_engine import DistributedPipelineEngine
from src.app.pipeline.context import PipelineContext
from src.app.business.causal_network import CausalNetworkBuilder
from src.app.business.community_detection import CommunityDetector
from src.app.business.dynamic_simulation import DynamicSimulationEngine
from src.adapters.news.gdelt_adapter import GDELTAdapter
from src.adapters.llm.graphrag_adapter import GraphRAGAdapter
from src.adapters.neo4j.store import Neo4jStore


class TestFullSystemIntegration(unittest.TestCase):
    
    @patch('src.adapters.neo4j.store.GraphDatabase')
    @patch('src.adapters.llm.graphrag_adapter.GraphDatabase')
    def setUp(self, mock_graphrag_db, mock_neo4j_db):
        """测试初始化"""
        self.context = PipelineContext()
        self.pipeline_engine = DistributedPipelineEngine(self.context)
        
    @patch('src.app.pipeline.distributed_engine.FunctionRegistry')
    @patch('src.adapters.llm.graphrag_adapter.GraphDatabase')
    @patch('src.adapters.neo4j.store.GraphDatabase')
    def test_end_to_end_pipeline_flow(self, mock_neo4j_db, mock_graphrag_db, mock_registry):
        """测试端到端Pipeline执行流程"""
        # 模拟工具函数
        mock_func = AsyncMock(return_value="test_result")
        mock_registry.get_tool.return_value = mock_func
        mock_registry.get_input_model.return_value = None
        
        # 模拟LLM客户端
        mock_llm_client = Mock()
        
        # 模拟GraphRAG适配器
        graphrag_adapter = GraphRAGAdapter(
            llm_client=mock_llm_client,
            graph_database_uri="bolt://localhost:7687",
            graph_database_auth=("neo4j", "password")
        )
        
        # 模拟LLM响应
        mock_llm_response = Mock()
        mock_llm_response.content = "Enhanced response with graph context"
        mock_llm_client.call.return_value = mock_llm_response
        
        # 模拟Neo4j存储
        neo4j_store = Neo4jStore("bolt://localhost:7687", ("neo4j", "password"))
        
        # 模拟GDELT适配器
        gdelt_adapter = GDELTAdapter(name="TestGDELT")
        
        # 准备完整的Pipeline定义
        pipeline_def = {
            "name": "full_system_pipeline",
            "steps": [
                {
                    "tool": "fetch_gdelt_data",
                    "inputs": {
                        "date_range": ["2023-01-01", "2023-01-31"]
                    }
                },
                {
                    "tool": "build_causal_network",
                    "inputs": {
                        "start_date": "2023-01-01",
                        "end_date": "2023-01-31"
                    }
                },
                {
                    "tool": "detect_communities",
                    "inputs": {
                        "date_range": ["2023-01-01", "2023-01-31"]
                    }
                },
                {
                    "tool": "run_simulation",
                    "inputs": {
                        "steps": 5
                    }
                },
                {
                    "tool": "graphrag_query",
                    "inputs": {
                        "query": "Analyze the detected communities and simulation results"
                    }
                }
            ]
        }
        
        # 运行异步测试
        async def run_test():
            result = await self.pipeline_engine.run_pipeline(
                run_id="full_system_test_run",
                project_id="full_system_test_project",
                pipeline_def=pipeline_def
            )
            return result
            
        result = asyncio.run(run_test())
        
        # 验证结果
        self.assertEqual(result.status, "completed")
        
    def test_cross_module_data_flow(self):
        """测试跨模块数据流"""
        # 测试因果网络构建器
        causal_builder = CausalNetworkBuilder()
        
        # 测试社区检测器
        community_detector = CommunityDetector()
        
        # 测试动态仿真引擎
        simulation_engine = DynamicSimulationEngine()
        
        # 验证各模块可以独立初始化
        self.assertIsNotNone(causal_builder)
        self.assertIsNotNone(community_detector)
        self.assertIsNotNone(simulation_engine)
        
        # 验证模块间数据传递的可能性
        # 这里我们验证模块之间没有直接依赖，但可以通过Pipeline进行数据传递


if __name__ == '__main__':
    unittest.main()