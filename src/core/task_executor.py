"""
TaskExecutor - 任务执行器
从PipelineEngine拆分出来的任务执行逻辑，提高代码可维护性
"""

import asyncio
import time
from typing import Dict, Any, Optional
from .logging import LoggerManager
from .registry import FunctionRegistry


class TaskExecutor:
    """任务执行器 - 处理单个任务的执行逻辑"""

    def __init__(self, context, logger=None):
        """
        初始化任务执行器

        Args:
            context: PipelineContext实例
            logger: 日志记录器
        """
        self.context = context
        self.logger = logger or LoggerManager.get_logger(__name__)

    def execute_task(self, task_config: Dict[str, Any]) -> Any:
        """
        执行单个任务的主流程

        Args:
            task_config: 任务配置

        Returns:
            任务执行结果
        """
        # 1. 解析任务配置
        tool_name = self._extract_tool_info(task_config)

        # 2. 解析工具函数
        func = self._resolve_tool_function(tool_name)

        # 3. 准备输入参数
        inputs = self._prepare_inputs(task_config)

        # 4. 执行任务（带重试）
        result = self._execute_with_retry(func, inputs, task_config)

        # 5. 处理输出
        self._handle_output(result, task_config)

        return result

    def _extract_tool_info(self, config: Dict[str, Any]) -> str:
        """职责1: 解析配置，提取工具信息"""
        return config.get("tool")

    def _resolve_tool_function(self, name: str):
        """职责2: 解析工具函数"""
        func = FunctionRegistry.get_tool(name)
        if not func:
            error_msg = f"找不到工具: {name}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return func

    def _prepare_inputs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """职责3: 准备输入参数"""
        input_mapping = config.get("inputs", {})
        inputs = {}

        for arg_name, value_mapping in input_mapping.items():
            if isinstance(value_mapping, str) and value_mapping.startswith("$"):
                # 从上下文获取变量
                key = value_mapping[1:]
                val = self.context.get(key)
                inputs[arg_name] = val
            else:
                # 直接字面量
                inputs[arg_name] = value_mapping

        # Pydantic校验
        tool_name = config.get("tool")
        input_model = FunctionRegistry.get_input_model(tool_name)
        if input_model:
            try:
                validated_data = input_model(**inputs)
                inputs = validated_data.model_dump()
            except Exception as ve:
                error_msg = f"参数校验失败: {ve}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        return inputs

    def _execute_with_retry(self, func, inputs: Dict[str, Any], config: Dict[str, Any]) -> Any:
        """职责4: 执行任务（带重试逻辑）"""
        retry_count = config.get("retry", 0)
        continue_on_error = config.get("continue_on_error", False)

        attempt = 0
        last_error = None

        while attempt <= retry_count:
            try:
                # 根据函数类型选择执行方式
                if asyncio.iscoroutinefunction(func):
                    # 异步函数 - 需要在事件循环中运行
                    if asyncio.get_event_loop().is_running():
                        # 如果已经在事件循环中，使用asyncio.create_task
                        import nest_asyncio
                        nest_asyncio.apply()
                        result = asyncio.get_event_loop().run_until_complete(func(**inputs))
                    else:
                        # 正常异步执行
                        result = asyncio.run(func(**inputs))
                else:
                    # 同步函数
                    result = func(**inputs)

                return result

            except Exception as e:
                last_error = e
                attempt += 1

                if attempt <= retry_count:
                    self.logger.warning(f"任务重试 (第{attempt}/{retry_count}次): {e}")
                    time.sleep(1 * attempt)  # 简单的指数退避

        # 重试耗尽
        error_msg = f"任务执行失败，已重试{retry_count}次: {last_error}"
        self.logger.error(error_msg)

        if continue_on_error:
            self.logger.warning("根据配置continue_on_error=True，继续执行")
            return None
        else:
            raise last_error

    def _handle_output(self, result: Any, config: Dict[str, Any]):
        """职责5: 处理输出结果"""
        output_key = config.get("output")
        if output_key:
            self.context.set(output_key, result)
            self.logger.debug(f"任务结果已存入上下文: {output_key}")
        else:
            self.logger.debug("任务完成，无输出存储")
