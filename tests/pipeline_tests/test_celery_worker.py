"""
Celery工作器模块单元测试
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime

# 注意：由于Celery工作器是顶层模块，我们主要测试其任务函数
from src.app.pipeline.celery_worker import run_pipeline_task, run_step_task, health_check_task


class TestCeleryWorker(unittest.TestCase):
    
    @patch('src.app.pipeline.celery_worker.PipelineEngine')
    @patch('src.app.pipeline.celery_worker.PipelineContext')
    def test_run_pipeline_task_success(self, mock_context_class, mock_engine_class):
        """测试成功运行Pipeline任务"""
        # 模拟Pipeline引擎和上下文
        mock_context = Mock()
        mock_context.to_dict.return_value = {"test_key": "test_value"}
        mock_context_class.return_value = mock_context
        
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # 模拟异步运行结果
        async def mock_run_pipeline(*args, **kwargs):
            return mock_context
        mock_engine.run_pipeline = mock_run_pipeline
        
        # 准备测试数据
        run_id = "test_run_123"
        project_id = "test_project_456"
        pipeline_def = {"name": "test_pipeline", "steps": []}
        context_data = {"input_key": "input_value"}
        
        # 运行任务
        result = run_pipeline_task(run_id, project_id, pipeline_def, context_data)
        
        # 验证结果
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(result["project_id"], project_id)
        
    @patch('src.app.pipeline.celery_worker.PipelineEngine')
    @patch('src.app.pipeline.celery_worker.PipelineContext')
    def test_run_pipeline_task_failure(self, mock_context_class, mock_engine_class):
        """测试运行Pipeline任务失败"""
        # 模拟异常
        mock_context_class.side_effect = Exception("Test error")
        
        # 准备测试数据
        run_id = "test_run_123"
        project_id = "test_project_456"
        pipeline_def = {"name": "test_pipeline", "steps": []}
        context_data = {"input_key": "input_value"}
        
        # 运行任务
        result = run_pipeline_task(run_id, project_id, pipeline_def, context_data)
        
        # 验证结果
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(result["project_id"], project_id)
        self.assertIn("Test error", result["error"])
        
    @patch('src.app.pipeline.celery_worker.PipelineEngine')
    @patch('src.app.pipeline.celery_worker.PipelineContext')
    def test_run_step_task_success(self, mock_context_class, mock_engine_class):
        """测试成功运行步骤任务"""
        # 模拟Pipeline引擎和上下文
        mock_context = Mock()
        mock_context.to_dict.return_value = {"test_key": "test_value"}
        mock_context_class.return_value = mock_context
        
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # 模拟异步运行结果
        async def mock_run_step(*args, **kwargs):
            return "step_result"
        mock_engine.run_step = mock_run_step
        
        # 准备测试数据
        run_id = "test_run_123"
        step_idx = 0
        step = {"tool": "test_tool"}
        context_data = {"input_key": "input_value"}
        
        # 运行任务
        result = run_step_task(run_id, step_idx, step, context_data)
        
        # 验证结果
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(result["step_idx"], step_idx)
        self.assertEqual(result["result"], "step_result")
        
    def test_health_check_task(self):
        """测试健康检查任务"""
        # 运行任务
        result = health_check_task()
        
        # 验证结果
        self.assertEqual(result["status"], "healthy")
        self.assertIn("timestamp", result)
        self.assertIn("worker", result)


if __name__ == '__main__':
    unittest.main()