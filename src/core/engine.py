import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union
from .registry import FunctionRegistry
from .context import PipelineContext
from .task_executor import TaskExecutor
from .logging import LoggerManager

class PipelineEngine:
    """
    流程执行引擎
    负责解析配置，调度原子函数，并管理上下文。
    """
    def __init__(self, context: Optional[PipelineContext] = None) -> None:
        self.context: PipelineContext = context or PipelineContext()
        self.task_executor: TaskExecutor = TaskExecutor(self.context)
        self.logger: logging.Logger = LoggerManager.get_logger(__name__)

    async def run_task(self, task_config: Dict[str, Any]) -> Any:
        """
        运行单个任务（重构版）

        Task Config 结构示例:
        {
            "id": "fetch_news_step",
            "tool": "fetch_news_stream",
            "inputs": { ... },
            "output": "raw_news_data",
            "retry": 3,              # 可选：重试次数
            "continue_on_error": false # 可选：出错是否继续
        }
        """
        task_id = task_config.get("id", task_config.get("tool"))
        tool_name = task_config.get("tool")
        continue_on_error = task_config.get("continue_on_error", False)

        self.logger.info(f"开始执行任务: {task_id} (工具: {tool_name})")

        try:
            # 将任务配置传递给TaskExecutor处理
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.task_executor.execute_task, task_config
            )

            self.logger.info(f"任务 {task_id} 完成")
            return result

        except Exception as e:
            self.logger.error(f"任务 {task_id} 执行失败: {e}")

            if continue_on_error:
                self.logger.warning(f"根据配置continue_on_error=True，继续执行")
                return None
            else:
                raise


    async def run_pipeline(self, pipeline_config: List[Dict[str, Any]]) -> PipelineContext:
        """
        执行完整流程

        Args:
            pipeline_config: 任务列表

        Returns:
            执行完成的上下文
        """
        self.logger.info("启动流程执行...")

        for i, task_config in enumerate(pipeline_config):
            try:
                self.logger.debug(f"执行流程步骤 {i+1}/{len(pipeline_config)}")
                await self.run_task(task_config)
            except Exception as e:
                # 如果 run_task 抛出异常，流程中止
                self.logger.error(f"流程执行被中断于步骤{i+1}: {e}")
                raise e

        self.logger.info("流程执行结束")
        return self.context
