"""
标准Agent基类 (BE-01)
提供统一的Agent开发框架和生命周期管理
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from .config import ConfigManager
from .logging import LoggerManager
from ..utils.llm_utils import AsyncExecutor
from .serialization import Serializer
from .context import PipelineContext
from .engine import PipelineEngine


class AgentStatus:
    """Agent状态枚举"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class BaseAgent(ABC):
    """
    标准Agent基类
    提供统一的Agent生命周期管理和基础设施集成
    """

    def __init__(self, agent_name: str, config_section: str = None):
        """
        初始化Agent

        Args:
            agent_name: Agent名称，用于日志和配置
            config_section: 配置section名称，默认使用agent_name+"_config"
        """
        self.agent_name = agent_name
        self.config_section = config_section or f"{agent_name}_config"
        self.status = AgentStatus.INITIALIZING

        # 初始化基础设施组件
        self.config = ConfigManager()
        self.logger = LoggerManager.get_logger(f"Agent.{agent_name}")
        self.executor = AsyncExecutor()
        self.serializer = Serializer()

        # Agent状态管理
        self.context = PipelineContext()
        self.pipeline_engine = PipelineEngine(self.context)
        self.start_time = None
        self.last_execution_time = None
        self.execution_count = 0

        # 加载Agent配置
        self._load_agent_config()

        self.logger.info(f"Agent {agent_name} 初始化完成")

    def _load_agent_config(self):
        """加载Agent配置"""
        try:
            # 基础配置
            self.max_workers = self.config.get_concurrency_limit(self.config_section)
            self.rate_limit = self.config.get_rate_limit(self.config_section)

            # Agent特定配置（子类可覆盖）
            self._load_custom_config()

            self.logger.debug(f"Agent配置加载完成: max_workers={self.max_workers}, rate_limit={self.rate_limit}")
        except Exception as e:
            self.logger.error(f"Agent配置加载失败: {e}")
            # 使用默认配置
            self.max_workers = 3
            self.rate_limit = 10.0

    def _load_custom_config(self):
        """加载Agent特定配置（子类可覆盖）"""
        pass

    async def initialize(self) -> bool:
        """
        初始化Agent（异步）
        子类可覆盖此方法进行特定初始化逻辑

        Returns:
            初始化是否成功
        """
        try:
            self.status = AgentStatus.READY
            self.start_time = datetime.now()
            self.logger.info(f"Agent {self.agent_name} 初始化成功")

            # 调用子类初始化
            await self._initialize_custom()
            return True

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Agent {self.agent_name} 初始化失败: {e}")
            return False

    async def _initialize_custom(self):
        """子类特定初始化逻辑（可选覆盖）"""
        pass

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行Agent任务

        Args:
            **kwargs: 执行参数

        Returns:
            执行结果
        """
        if self.status != AgentStatus.READY:
            raise RuntimeError(f"Agent {self.agent_name} 状态为 {self.status}，无法执行")

        self.status = AgentStatus.RUNNING
        self.execution_count += 1
        self.last_execution_time = datetime.now()

        try:
            self.logger.info(f"开始执行Agent {self.agent_name} (第{self.execution_count}次)")

            # 执行前准备
            await self._pre_execute()

            # 执行主逻辑
            result = await self._execute_main(**kwargs)

            # 执行后处理
            await self._post_execute(result)

            self.status = AgentStatus.READY
            self.logger.info(f"Agent {self.agent_name} 执行完成")

            return {
                "status": "success",
                "result": result,
                "execution_time": datetime.now(),
                "execution_count": self.execution_count
            }

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Agent {self.agent_name} 执行失败: {e}")

            return {
                "status": "error",
                "error": str(e),
                "execution_time": datetime.now(),
                "execution_count": self.execution_count
            }

    @abstractmethod
    async def _execute_main(self, **kwargs) -> Any:
        """
        主执行逻辑（子类必须实现）

        Args:
            **kwargs: 执行参数

        Returns:
            执行结果
        """
        pass

    async def _pre_execute(self):
        """执行前准备（子类可覆盖）"""
        pass

    async def _post_execute(self, result: Any):
        """执行后处理（子类可覆盖）"""
        pass

    async def pause(self):
        """暂停Agent"""
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.PAUSED
            self.logger.info(f"Agent {self.agent_name} 已暂停")

    async def resume(self):
        """恢复Agent"""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.READY
            self.logger.info(f"Agent {self.agent_name} 已恢复")

    async def stop(self):
        """停止Agent"""
        self.status = AgentStatus.STOPPED
        self.logger.info(f"Agent {self.agent_name} 已停止")

    def save_state(self) -> str:
        """
        保存Agent状态

        Returns:
            序列化的状态字符串
        """
        state = {
            "agent_name": self.agent_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "execution_count": self.execution_count,
            "context": self.context.to_json()
        }

        return self.serializer.safe_json_dumps(state)

    def load_state(self, state_json: str):
        """
        加载Agent状态

        Args:
            state_json: 序列化的状态字符串
        """
        try:
            state = self.serializer.safe_json_loads(state_json)

            self.status = state.get("status", AgentStatus.INITIALIZING)
            self.start_time = datetime.fromisoformat(state["start_time"]).replace(tzinfo=None) if state.get("start_time") else None
            self.last_execution_time = datetime.fromisoformat(state["last_execution_time"]).replace(tzinfo=None) if state.get("last_execution_time") else None
            self.execution_count = state.get("execution_count", 0)

            # 恢复上下文
            if state.get("context"):
                # 这里需要实现上下文的反序列化
                pass

            self.logger.info(f"Agent {self.agent_name} 状态加载完成")
        except Exception as e:
            self.logger.error(f"Agent {self.agent_name} 状态加载失败: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态信息"""
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "start_time": self.start_time,
            "last_execution_time": self.last_execution_time,
            "execution_count": self.execution_count,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }


class PipelineAgent(BaseAgent):
    """
    基于Pipeline的Agent
    使用PipelineEngine编排任务流程
    """

    def __init__(self, agent_name: str, pipeline_config: List[Dict[str, Any]]):
        """
        初始化Pipeline Agent

        Args:
            agent_name: Agent名称
            pipeline_config: Pipeline配置
        """
        super().__init__(agent_name)
        self.pipeline_config = pipeline_config

    async def _execute_main(self, **kwargs) -> Any:
        """使用PipelineEngine执行任务"""
        # 合并执行参数到pipeline配置
        execution_config = self.pipeline_config.copy()

        # 可以在这里根据kwargs修改pipeline配置

        # 执行pipeline
        result = await self.pipeline_engine.run_pipeline(execution_config)
        return result


class TaskAgent(BaseAgent):
    """
    基于Task的Agent
    直接使用TaskExecutor执行单个任务
    """

    def __init__(self, agent_name: str, task_config: Dict[str, Any]):
        """
        初始化Task Agent

        Args:
            agent_name: Agent名称
            task_config: 任务配置
        """
        super().__init__(agent_name)
        self.task_config = task_config

    async def _execute_main(self, **kwargs) -> Any:
        """使用TaskExecutor执行任务"""
        # 合并执行参数到任务配置
        execution_config = self.task_config.copy()
        execution_config.update(kwargs)

        # 执行任务
        from .task_executor import TaskExecutor
        task_executor = TaskExecutor(self.context)

        # 这里需要异步执行任务，可能需要调整TaskExecutor
        # 暂时使用同步方式
        result = task_executor.execute_task(execution_config)
        return result


class AgentRegistry:
    """
    Agent注册中心
    管理所有已注册的Agent
    """

    _agents: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_agent(cls, agent_class: type, name: str, description: str = "", **metadata):
        """
        注册Agent类

        Args:
            agent_class: Agent类
            name: Agent名称
            description: Agent描述
            **metadata: 其他元数据
        """
        cls._agents[name] = {
            "class": agent_class,
            "name": name,
            "description": description,
            "metadata": metadata
        }

    @classmethod
    def get_agent_class(cls, name: str) -> Optional[type]:
        """获取Agent类"""
        return cls._agents.get(name, {}).get("class")

    @classmethod
    def list_agents(cls) -> Dict[str, Dict[str, Any]]:
        """列出所有注册的Agent"""
        return cls._agents.copy()

    @classmethod
    def create_agent(cls, name: str, **kwargs) -> Optional[BaseAgent]:
        """创建Agent实例"""
        agent_class = cls.get_agent_class(name)
        if agent_class:
            return agent_class(**kwargs)
        return None


# 便捷函数
def register_agent(name: str, description: str = "", **metadata):
    """Agent注册装饰器"""
    def decorator(agent_class: type):
        AgentRegistry.register_agent(agent_class, name, description, **metadata)
        return agent_class
    return decorator
