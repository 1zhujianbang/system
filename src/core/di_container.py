"""
Dependency Injection Container
提供依赖注入服务，支持组件生命周期管理和依赖解析
"""

import inspect
import threading
from typing import Any, Dict, Type, TypeVar, Optional, Callable, Union, List
from enum import Enum
from .logging import LoggerManager
from .singleton import singleton


T = TypeVar('T')


class ServiceLifetime(Enum):
    """服务生命周期枚举"""
    SINGLETON = "singleton"      # 单例模式，整个应用生命周期内只有一个实例
    SCOPED = "scoped"           # 作用域模式，每个作用域内只有一个实例
    TRANSIENT = "transient"     # 瞬时模式，每次请求都创建新实例


class ServiceDescriptor:
    """服务描述符"""

    def __init__(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                 factory: Optional[Callable] = None, lifetime: ServiceLifetime = ServiceLifetime.SINGLETON):
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.factory = factory
        self.lifetime = lifetime
        self.instance: Optional[T] = None
        self._lock = threading.Lock()

    def get_instance(self, container: 'DependencyContainer') -> T:
        """获取服务实例"""
        if self.lifetime == ServiceLifetime.SINGLETON:
            if self.instance is None:
                with self._lock:
                    if self.instance is None:
                        self.instance = self._create_instance(container)
            return self.instance
        elif self.lifetime == ServiceLifetime.TRANSIENT:
            return self._create_instance(container)
        else:
            # SCOPED 模式暂时按单例处理（可以后续扩展）
            if self.instance is None:
                with self._lock:
                    if self.instance is None:
                        self.instance = self._create_instance(container)
            return self.instance

    def _create_instance(self, container: 'DependencyContainer') -> T:
        """创建服务实例"""
        if self.factory:
            # 使用工厂方法创建
            return self.factory(container)
        else:
            # 使用构造函数注入
            return container._create_instance_with_injection(self.implementation_type)


class DependencyContainer:
    """依赖注入容器"""

    def __init__(self):
        self._services: Dict[Type[T], ServiceDescriptor] = {}
        self._logger = LoggerManager.get_logger(__name__)

    def register(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                lifetime: ServiceLifetime = ServiceLifetime.SINGLETON) -> 'DependencyContainer':
        """
        注册服务

        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型，如果为None则使用service_type
            lifetime: 服务生命周期

        Returns:
            容器实例，支持链式调用
        """
        descriptor = ServiceDescriptor(service_type, implementation_type, lifetime=lifetime)
        self._services[service_type] = descriptor
        self._logger.debug(f"Registered service: {service_type.__name__} -> {implementation_type.__name__ if implementation_type else service_type.__name__}")
        return self

    def register_factory(self, service_type: Type[T], factory: Callable[['DependencyContainer'], T],
                        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON) -> 'DependencyContainer':
        """
        注册工厂方法

        Args:
            service_type: 服务类型
            factory: 工厂方法，接收容器作为参数
            lifetime: 服务生命周期

        Returns:
            容器实例，支持链式调用
        """
        descriptor = ServiceDescriptor(service_type, factory=factory, lifetime=lifetime)
        self._services[service_type] = descriptor
        self._logger.debug(f"Registered factory for service: {service_type.__name__}")
        return self

    def register_instance(self, service_type: Type[T], instance: T) -> 'DependencyContainer':
        """
        注册已存在的实例

        Args:
            service_type: 服务类型
            instance: 服务实例

        Returns:
            容器实例，支持链式调用
        """
        descriptor = ServiceDescriptor(service_type, lifetime=ServiceLifetime.SINGLETON)
        descriptor.instance = instance
        self._services[service_type] = descriptor
        self._logger.debug(f"Registered instance for service: {service_type.__name__}")
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """
        解析服务

        Args:
            service_type: 服务类型

        Returns:
            服务实例

        Raises:
            ValueError: 如果服务未注册
        """
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} is not registered")

        descriptor = self._services[service_type]
        return descriptor.get_instance(self)

    def is_registered(self, service_type: Type[T]) -> bool:
        """检查服务是否已注册"""
        return service_type in self._services

    def _create_instance_with_injection(self, implementation_type: Type[T]) -> T:
        """
        通过构造函数注入创建实例

        Args:
            implementation_type: 实现类型

        Returns:
            创建的实例
        """
        # 获取构造函数参数
        init_signature = inspect.signature(implementation_type.__init__)
        init_params = init_signature.parameters

        # 准备构造函数参数
        kwargs = {}

        for param_name, param in init_params.items():
            if param_name == 'self':
                continue

            # 检查是否有类型注解
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation

                # 处理特殊类型
                if param_type == DependencyContainer:
                    kwargs[param_name] = self
                elif hasattr(param_type, '__origin__') and param_type.__origin__ == Union:
                    # 处理 Optional[T] 等联合类型
                    for arg in param_type.__args__:
                        if arg != type(None) and self.is_registered(arg):
                            kwargs[param_name] = self.resolve(arg)
                            break
                elif self.is_registered(param_type):
                    # 如果参数类型已注册，则注入
                    kwargs[param_name] = self.resolve(param_type)
                elif param.default != inspect.Parameter.empty:
                    # 使用默认值
                    continue
                else:
                    # 无法解析的参数，记录警告
                    self._logger.warning(f"Cannot resolve parameter '{param_name}' of type {param_type} for {implementation_type.__name__}")
            elif param.default != inspect.Parameter.empty:
                # 使用默认值
                continue
            else:
                self._logger.warning(f"Parameter '{param_name}' has no type annotation and no default value")

        try:
            instance = implementation_type(**kwargs)
            self._logger.debug(f"Created instance of {implementation_type.__name__}")
            return instance
        except Exception as e:
            self._logger.error(f"Failed to create instance of {implementation_type.__name__}: {e}")
            raise

    def get_registered_services(self) -> List[Type[T]]:
        """获取所有已注册的服务类型"""
        return list(self._services.keys())


# 全局容器实例
@singleton
class GlobalContainer:
    """全局依赖注入容器"""

    def __init__(self):
        self._container = DependencyContainer()
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self):
        """初始化全局容器"""
        with self._lock:
            if self._initialized:
                return

            self._setup_default_services()
            self._initialized = True

    def _setup_default_services(self):
        """设置默认服务"""
        from .config import ConfigManager
        from .key_manager import KeyManager
        from ..utils.llm_utils import AsyncExecutor, RateLimiter
        from .logging import LoggerManager
        from ..agents.api_client import LLMAPIPool

        # 核心服务
        self._container.register_instance(DependencyContainer, self._container)
        self._container.register(ConfigManager, lifetime=ServiceLifetime.SINGLETON)
        self._container.register(KeyManager, lifetime=ServiceLifetime.SINGLETON)
        self._container.register(LoggerManager, lifetime=ServiceLifetime.SINGLETON)

        # 异步和并发服务
        self._container.register(AsyncExecutor, lifetime=ServiceLifetime.SINGLETON)

        # API客户端 - 使用工厂方法以支持延迟初始化
        def create_llm_api_pool(container):
            return LLMAPIPool()

        self._container.register_factory(LLMAPIPool, create_llm_api_pool, ServiceLifetime.SINGLETON)

        # 速率限制器工厂
        def create_rate_limiter(container):
            config_manager = container.resolve(ConfigManager)
            rate_limit = config_manager.get_config_value("rate_limit_per_sec", 1.0)
            return RateLimiter(rate_limit)

        self._container.register_factory(RateLimiter, create_rate_limiter, ServiceLifetime.SINGLETON)

    @property
    def container(self) -> DependencyContainer:
        """获取容器实例"""
        if not self._initialized:
            self.initialize()
        return self._container


# 便捷函数
def get_container() -> DependencyContainer:
    """获取全局容器实例"""
    global_container = GlobalContainer()
    return global_container.container


def get_service(service_type: Type[T]) -> T:
    """获取服务实例的便捷函数"""
    return get_container().resolve(service_type)


def register_service(service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON):
    """注册服务的便捷函数"""
    get_container().register(service_type, implementation_type, lifetime)


def register_service_factory(service_type: Type[T], factory: Callable[[DependencyContainer], T],
                           lifetime: ServiceLifetime = ServiceLifetime.SINGLETON):
    """注册服务工厂的便捷函数"""
    get_container().register_factory(service_type, factory, lifetime)
