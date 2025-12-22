"""
分布式Pipeline集成测试
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.app.pipeline.distributed_engine import DistributedPipelineEngine
from src.app.pipeline.context import PipelineContext
from src.adapters.llm.graphrag_adapter import GraphRAGAdapter
from src.adapters.neo4j.store import Neo4jStore


class TestDistributedPipelineIntegration(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        self.context = PipelineContext()
        self.engine = DistributedPipelineEngine(self.context)
        
    @patch('src.app.pipeline.distributed_engine.FunctionRegistry')
    @patch('src.adapters.llm.graphrag_adapter.GraphDatabase')
    def test_pipeline_with_graphrag_flow(self, mock_graph_db, mock_registry):
        """测试包含GraphRAG的Pipeline执行流程"""
        # 模拟工具函数
        mock_func = AsyncMock(return_value="test_result")
        mock_registry.get_tool.return_value = mock_func
        mock_registry.get_input_model.return_value = None
        
        # 模拟GraphRAG适配器
        mock_llm_client = Mock()
        with patch('src.adapters.neo4j.store.GraphDatabase'):
            graphrag_adapter = GraphRAGAdapter(
                llm_client=mock_llm_client,
                graph_database_uri="bolt://localhost:7687",
                graph_database_auth=("neo4j", "password")
            )
            
        # 模拟LLM响应
        mock_llm_response = Mock()
        mock_llm_response.content = "Enhanced response with graph context"
        mock_llm_client.call.return_value = mock_llm_response
        
        # 准备测试Pipeline定义
        pipeline_def = {
            "name": "test_graphrag_pipeline",
            "steps": [
                {
                    "tool": "graphrag_query",
                    "inputs": {
                        "query": "Test query"
                    }
                }
            ]
        }
        
        # 运行异步测试
        async def run_test():
            result = await self.engine.run_pipeline(
                run_id="test_run",
                project_id="test_project",
                pipeline_def=pipeline_def
            )
            return result
            
        result = asyncio.run(run_test())
        
        # 验证结果
        self.assertEqual(result.status, "completed")
        
    def test_pipeline_error_handling_flow(self):
        """测试Pipeline错误处理流程"""
        # 准备包含错误的Pipeline定义
        pipeline_def = {
            "name": "test_error_pipeline",
            "steps": [
                "invalid_step"  # 无效步骤格式
            ]
        }
        
        # 运行异步测试
        async def run_test():
            result = await self.engine.run_pipeline(
                run_id="test_run",
                project_id="test_project",
                pipeline_def=pipeline_def
            )
            return result
            
        result = asyncio.run(run_test())
        
        # 验证结果
        self.assertEqual(result.status, "failed")
        self.assertIn("Invalid step definition", result.error)


if __name__ == '__main__':
    unittest.main()