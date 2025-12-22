"""
分布式Pipeline引擎模块单元测试
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime

from src.app.pipeline.distributed_engine import DistributedPipelineEngine
from src.app.pipeline.context import PipelineContext
from src.app.pipeline.models import PipelineRunState


class TestDistributedPipelineEngine(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        self.context = PipelineContext()
        self.engine = DistributedPipelineEngine(self.context)
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.engine.ctx, self.context)
        self.assertFalse(self.engine._tools_loaded)
        
    def test_resolve_inputs(self):
        """测试解析输入参数"""
        # 准备测试数据
        step = {
            "inputs": {
                "param1": "value1",
                "param2": "$var1"  # 从context获取
            }
        }
        
        # 设置context值
        self.context.set("var1", "context_value")
        
        # 调用方法
        result = self.engine._resolve_inputs(step)
        
        # 验证结果
        self.assertEqual(result["param1"], "value1")
        self.assertEqual(result["param2"], "context_value")
        
    @patch('src.app.pipeline.distributed_engine.FunctionRegistry')
    def test_run_step_local(self, mock_registry):
        """测试本地运行步骤"""
        # 模拟工具函数
        mock_func = AsyncMock(return_value="test_result")
        mock_registry.get_tool.return_value = mock_func
        mock_registry.get_input_model.return_value = None
        
        # 准备测试数据
        step = {
            "tool": "test_tool",
            "inputs": {}
        }
        
        # 创建异步事件循环并运行
        async def run_test():
            result = await self.engine.run_step(None, 0, step)
            return result
            
        result = asyncio.run(run_test())
        
        # 验证结果
        self.assertEqual(result, "test_result")
        mock_func.assert_called_once()
        
    def test_run_step_invalid_step(self):
        """测试运行无效步骤"""
        # 准备测试数据
        steps = [
            "invalid_step"  # 不是dict类型
        ]
        
        # 创建异步事件循环并运行
        async def run_test():
            with self.assertRaises(ValueError):
                await self.engine.run_pipeline(
                    run_id="test_run",
                    project_id="test_project",
                    pipeline_def={"steps": steps}
                )
                
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()