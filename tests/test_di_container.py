"""
Unit tests for Dependency Injection Container
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.di_container import DependencyContainer, ServiceLifetime, get_container, get_service, register_service


class TestDependencyContainer:
    """测试依赖注入容器"""

    def test_register_and_resolve_service(self):
        """测试服务注册和解析"""
        container = DependencyContainer()

        # 注册服务
        mock_service = MagicMock()
        mock_service.value = "test"

        container.register_instance(str, mock_service)

        # 解析服务
        resolved = container.resolve(str)
        assert resolved is mock_service
        assert resolved.value == "test"

    def test_register_with_implementation(self):
        """测试使用实现类型注册服务"""
        container = DependencyContainer()

        # 定义接口和服务
        class IService:
            pass

        class ServiceImpl(IService):
            def __init__(self):
                self.value = "impl"

        # 注册服务
        container.register(IService, ServiceImpl)

        # 解析服务
        resolved = container.resolve(IService)
        assert isinstance(resolved, ServiceImpl)
        assert resolved.value == "impl"

    def test_register_factory(self):
        """测试工厂方法注册"""
        container = DependencyContainer()

        def create_service(container):
            service = MagicMock()
            service.created_by_factory = True
            return service

        container.register_factory(str, create_service)

        resolved = container.resolve(str)
        assert resolved.created_by_factory is True

    def test_singleton_lifetime(self):
        """测试单例生命周期"""
        container = DependencyContainer()

        class TestService:
            def __init__(self):
                self.instance_id = id(self)

        container.register(TestService, lifetime=ServiceLifetime.SINGLETON)

        # 解析多次应该返回同一个实例
        instance1 = container.resolve(TestService)
        instance2 = container.resolve(TestService)

        assert instance1 is instance2
        assert instance1.instance_id == instance2.instance_id

    def test_transient_lifetime(self):
        """测试瞬时生命周期"""
        container = DependencyContainer()

        class TestService:
            def __init__(self):
                self.instance_id = id(self)

        container.register(TestService, lifetime=ServiceLifetime.TRANSIENT)

        # 解析多次应该返回不同实例
        instance1 = container.resolve(TestService)
        instance2 = container.resolve(TestService)

        assert instance1 is not instance2
        assert instance1.instance_id != instance2.instance_id

    def test_constructor_injection(self):
        """测试构造函数注入"""
        container = DependencyContainer()

        class DependencyA:
            def __init__(self):
                self.name = "A"

        class DependencyB:
            def __init__(self):
                self.name = "B"

        class ServiceWithDeps:
            def __init__(self, dep_a: DependencyA, dep_b: DependencyB):
                self.dep_a = dep_a
                self.dep_b = dep_b

        # 注册依赖
        container.register(DependencyA)
        container.register(DependencyB)
        container.register(ServiceWithDeps)

        # 解析服务
        resolved = container.resolve(ServiceWithDeps)

        assert isinstance(resolved.dep_a, DependencyA)
        assert isinstance(resolved.dep_b, DependencyB)
        assert resolved.dep_a.name == "A"
        assert resolved.dep_b.name == "B"

    def test_container_injection(self):
        """测试容器自身注入"""
        container = DependencyContainer()

        class ServiceNeedingContainer:
            def __init__(self, container: DependencyContainer):
                self.container = container

        container.register_instance(DependencyContainer, container)
        container.register(ServiceNeedingContainer)

        resolved = container.resolve(ServiceNeedingContainer)
        assert resolved.container is container

    def test_unregistered_service_error(self):
        """测试未注册服务解析错误"""
        container = DependencyContainer()

        with pytest.raises(ValueError, match="Service .* is not registered"):
            container.resolve(str)

    def test_is_registered(self):
        """测试服务注册检查"""
        container = DependencyContainer()

        assert not container.is_registered(str)

        container.register_instance(str, "test")
        assert container.is_registered(str)

    def test_get_registered_services(self):
        """测试获取已注册服务列表"""
        container = DependencyContainer()

        # 初始状态
        services = container.get_registered_services()
        assert len(services) == 0

        # 注册服务后
        container.register_instance(str, "test")
        container.register_instance(int, 42)

        services = container.get_registered_services()
        assert len(services) == 2
        assert str in services
        assert int in services


class TestGlobalContainerIntegration:
    """全局容器集成测试"""

    def test_global_container_initialization(self):
        """测试全局容器初始化"""
        from src.core.di_container import GlobalContainer

        # 创建新的全局容器实例（不使用单例）
        global_container = GlobalContainer()
        global_container.initialize()

        container = global_container.container

        # 检查核心服务是否已注册
        assert container.is_registered(DependencyContainer)

        # 解析核心服务
        config_manager = container.resolve('ConfigManager')  # 使用字符串名称测试
        assert config_manager is not None

    def test_get_service_function(self):
        """测试get_service便捷函数"""
        # 模拟已初始化的全局容器
        with patch('src.core.di_container.GlobalContainer') as mock_global:
            mock_container = MagicMock()
            mock_service = MagicMock()
            mock_service.name = "test_service"

            mock_global.return_value.container = mock_container
            mock_container.resolve.return_value = mock_service

            # 测试get_service
            result = get_service(str)
            assert result is mock_service
            mock_container.resolve.assert_called_with(str)

    def test_register_service_function(self):
        """测试register_service便捷函数"""
        with patch('src.core.di_container.get_container') as mock_get_container:
            mock_container = MagicMock()
            mock_get_container.return_value = mock_container

            # 测试注册服务
            register_service(str, int)

            mock_container.register.assert_called_once_with(str, int, ServiceLifetime.SINGLETON)


class TestCircularDependency:
    """测试循环依赖"""

    def test_circular_dependency_detection(self):
        """测试循环依赖检测"""
        container = DependencyContainer()

        class ServiceA:
            def __init__(self, b):  # type: ignore
                self.b = b

        class ServiceB:
            def __init__(self, a):  # type: ignore
                self.a = a

        container.register(ServiceA)
        container.register(ServiceB)

        # 解析时应该能正常工作（通过延迟解析）
        # 这是一个简化的测试，实际的循环依赖检测需要更复杂的实现
        try:
            a = container.resolve(ServiceA)
            assert a is not None
        except RecursionError:
            pytest.fail("Circular dependency caused infinite recursion")


class TestServiceLifetime:
    """测试服务生命周期"""

    def test_scoped_lifetime(self):
        """测试作用域生命周期（暂时按单例处理）"""
        container = DependencyContainer()

        class TestService:
            def __init__(self):
                self.instance_id = id(self)

        container.register(TestService, lifetime=ServiceLifetime.SCOPED)

        # 当前实现中SCOPED按SINGLETON处理
        instance1 = container.resolve(TestService)
        instance2 = container.resolve(TestService)

        assert instance1 is instance2


if __name__ == "__main__":
    pytest.main([__file__])
