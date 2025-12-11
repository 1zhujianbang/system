from .registry import FunctionRegistry, register_tool
from .context import PipelineContext
from .engine import PipelineEngine
from .task_executor import TaskExecutor

# 新增的共享基础设施组件
from .config import ConfigManager
from .key_manager import KeyManager, get_key_manager, store_api_key, get_api_key
from .di_container import DependencyContainer, GlobalContainer, get_container, get_service, register_service, register_service_factory, ServiceLifetime
from ..utils.llm_utils import AsyncExecutor, RateLimiter
from .data_utils import DataNormalizer, DataPipeline, StandardEventPipeline, BatchDataProcessor
from .agent_base import BaseAgent, PipelineAgent, TaskAgent, AgentRegistry, register_agent, AgentStatus
from .agent_manager import AgentManager, AgentWorkflow, get_agent_manager
from .logging import LoggerManager
from .serialization import Serializer
from .imports import ImportManager
from .news_processing import process_news_batch_async, build_published_at, load_processed_ids, save_processed_id
from .agent_config import AgentConfigLoader, get_agent_config
from ..agents.api_client import LLMAPIPool
from ..utils.tool_function import tools
# 工具模块
from ..utils import json_utils, llm_utils, file_utils, data_utils, data_ops
from . import async_file_utils
from .exceptions import (
    NewsAgentException, ConfigError, ValidationError, NetworkError,
    APIError, ProcessingError, FileOperationError, ConcurrencyError,
    handle_errors, handle_async_errors, ErrorHandler
)
from .singleton import SingletonBase, SingletonMeta, ThreadLocalSingleton, singleton, get_singleton_instance

# 全局实例工厂函数 (避免循环导入)
def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    return ConfigManager()

def get_logger_manager() -> LoggerManager:
    """获取日志管理器实例"""
    return LoggerManager()

__all__ = [
    # 原有组件
    'FunctionRegistry', 'register_tool', 'PipelineContext', 'PipelineEngine', 'TaskExecutor',
    # 新增组件
    'ConfigManager', 'KeyManager', 'get_key_manager', 'store_api_key', 'get_api_key',
    'DependencyContainer', 'GlobalContainer', 'get_container', 'get_service', 'register_service', 'register_service_factory', 'ServiceLifetime',
    'AsyncExecutor', 'RateLimiter', 'DataNormalizer',
    'DataPipeline', 'StandardEventPipeline', 'BatchDataProcessor',
    'BaseAgent', 'PipelineAgent', 'TaskAgent', 'AgentRegistry', 'register_agent', 'AgentStatus',
    'AgentManager', 'AgentWorkflow', 'get_agent_manager',
    'LoggerManager', 'Serializer', 'ImportManager',
    # 异常处理
    'NewsAgentException', 'ConfigError', 'ValidationError', 'NetworkError',
    'APIError', 'ProcessingError', 'FileOperationError', 'ConcurrencyError',
    'handle_errors', 'handle_async_errors', 'ErrorHandler',
    # 单例模式
    'SingletonBase', 'SingletonMeta', 'ThreadLocalSingleton', 'singleton', 'get_singleton_instance',
    # 全局实例工厂
    'get_config_manager', 'get_logger_manager',
    # 新闻处理工具
    'process_news_batch_async', 'build_published_at', 'load_processed_ids', 'save_processed_id',
    # Agent配置
    'AgentConfigLoader', 'get_agent_config',
    # API客户端
    'LLMAPIPool',
    # 工具函数
    'tools',
    # 工具模块
    'json_utils', 'llm_utils', 'file_utils', 'data_utils', 'data_ops',
    # 异步文件工具
    'async_file_utils'
]

