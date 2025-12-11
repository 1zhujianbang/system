"""
优雅的单例模式实现
支持线程安全的单例创建和实例管理
"""

import threading
from typing import Type, TypeVar, Dict, Any, Optional
from abc import ABC

T = TypeVar('T')


class SingletonMeta(type):
    """
    单例元类 - 线程安全的单例实现

    使用双重检查锁定模式确保线程安全，同时避免不必要的锁竞争。
    """
    _instances: Dict[Type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # 双重检查锁定
        if cls not in cls._instances:
            with cls._lock:
                # 再次检查（防止在等待锁时其他线程创建了实例）
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class SingletonBase(metaclass=SingletonMeta):
    """
    单例基类

    所有需要单例模式的类都可以继承此类来获得线程安全的单例行为。
    """

    def __init__(self) -> None:
        # 防止重复初始化
        if hasattr(self, '_singleton_initialized'):
            return

        self._singleton_initialized = True
        self._init_singleton()

    def _init_singleton(self) -> None:
        """
        子类可以重写此方法来进行单例初始化逻辑

        注意：此方法只会在实例第一次创建时调用一次
        """
        pass


class ThreadLocalSingleton:
    """
    线程本地单例

    为每个线程提供独立的单例实例。
    适用于需要线程隔离的场景。
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def get_instance(self, cls: Type[T], *args, **kwargs) -> T:
        """
        获取线程本地的实例

        Args:
            cls: 要创建实例的类
            *args, **kwargs: 传递给构造函数的参数

        Returns:
            线程本地的实例
        """
        instance_key = f"_instance_{cls.__name__}"

        if not hasattr(self._local, instance_key):
            instance = cls(*args, **kwargs)
            setattr(self._local, instance_key, instance)

        return getattr(self._local, instance_key)


def singleton(cls: Type[T]) -> Type[T]:
    """
    单例装饰器

    用法：
    @singleton
    class MyClass:
        pass
    """
    class SingletonWrapper(cls):
        _instance: Optional[T] = None
        _lock = threading.Lock()

        def __new__(cls_inner, *args, **kwargs):
            if cls_inner._instance is None:
                with cls_inner._lock:
                    if cls_inner._instance is None:
                        cls_inner._instance = super().__new__(cls_inner)
            return cls_inner._instance

        def __init__(self, *args, **kwargs):
            # 只在第一次创建时初始化
            if not hasattr(self, '_initialized'):
                super().__init__(*args, **kwargs)
                self._initialized = True

    # 保持原始类的名称和文档
    SingletonWrapper.__name__ = cls.__name__
    SingletonWrapper.__doc__ = cls.__doc__
    SingletonWrapper.__module__ = cls.__module__

    return SingletonWrapper


# 便捷函数
def get_singleton_instance(cls: Type[T], *args, **kwargs) -> T:
    """
    获取类的单例实例

    Args:
        cls: 要获取单例实例的类
        *args, **kwargs: 传递给构造函数的参数

    Returns:
        单例实例
    """
    return cls(*args, **kwargs)  # 如果类使用了单例元类，这里会返回单例实例
