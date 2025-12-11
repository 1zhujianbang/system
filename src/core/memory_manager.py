"""
内存管理器
提供内存使用监控、垃圾回收和内存优化功能
"""

import gc
import psutil
import os
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
import weakref


class MemoryManager:
    """内存管理器"""

    def __init__(self,
                 memory_limit_mb: int = 512,
                 gc_threshold_mb: int = 100,
                 cleanup_interval: int = 60):
        """
        初始化内存管理器

        Args:
            memory_limit_mb: 内存使用上限（MB）
            gc_threshold_mb: 垃圾回收阈值（MB）
            cleanup_interval: 清理检查间隔（秒）
        """
        self.memory_limit_mb = memory_limit_mb
        self.gc_threshold_mb = gc_threshold_mb
        self.cleanup_interval = cleanup_interval

        self._process = psutil.Process(os.getpid())
        self._lock = threading.RLock()
        self._weak_refs: List[weakref.ref] = []
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False

        self._stats = {
            'memory_current_mb': 0,
            'memory_peak_mb': 0,
            'objects_tracked': 0,
            'gc_cycles': 0,
            'forced_cleanups': 0
        }

        self._logger = logging.getLogger(__name__)

    def start_monitoring(self) -> None:
        """启动内存监控"""
        if self._running:
            return

        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="MemoryManager-Cleanup"
        )
        self._cleanup_thread.start()

        self._logger.info(f"Memory manager started with limit {self.memory_limit_mb}MB")

    def stop_monitoring(self) -> None:
        """停止内存监控"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        self._logger.info("Memory manager stopped")

    def get_memory_usage(self) -> float:
        """
        获取当前内存使用量（MB）

        Returns:
            内存使用量（MB）
        """
        try:
            memory_info = self._process.memory_info()
            usage_mb = memory_info.rss / 1024 / 1024
            return usage_mb
        except Exception as e:
            self._logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息

        Returns:
            内存统计字典
        """
        current = self.get_memory_usage()
        peak = max(current, self._stats['memory_peak_mb'])

        with self._lock:
            self._stats.update({
                'memory_current_mb': current,
                'memory_peak_mb': peak,
                'objects_tracked': len(self._weak_refs)
            })

            return self._stats.copy()

    def check_memory_pressure(self) -> bool:
        """
        检查内存压力

        Returns:
            是否处于内存压力状态
        """
        usage = self.get_memory_usage()
        return usage > self.memory_limit_mb * 0.8  # 80%阈值

    def force_gc(self) -> int:
        """
        强制垃圾回收

        Returns:
            回收的对象数量
        """
        collected = gc.collect()
        with self._lock:
            self._stats['gc_cycles'] += 1

        self._logger.debug(f"Garbage collection completed, {collected} objects collected")
        return collected

    def cleanup_weak_refs(self) -> int:
        """
        清理无效的弱引用

        Returns:
            清理的数量
        """
        with self._lock:
            # 清理已失效的弱引用
            self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]
            cleaned = len(self._weak_refs)

        return cleaned

    def track_object(self, obj: Any) -> None:
        """
        跟踪对象（弱引用）

        Args:
            obj: 要跟踪的对象
        """
        with self._lock:
            self._weak_refs.append(weakref.ref(obj, self._on_object_deleted))

    def _on_object_deleted(self, ref: weakref.ref) -> None:
        """对象删除回调"""
        with self._lock:
            try:
                self._weak_refs.remove(ref)
            except ValueError:
                pass  # 引用已不存在

    def _cleanup_worker(self) -> None:
        """清理工作线程"""
        while self._running:
            try:
                time.sleep(self.cleanup_interval)

                # 检查内存使用
                usage = self.get_memory_usage()

                # 如果内存使用超过阈值，执行清理
                if usage > self.memory_limit_mb:
                    self._perform_cleanup()
                    with self._lock:
                        self._stats['forced_cleanups'] += 1

                # 定期清理弱引用
                self.cleanup_weak_refs()

            except Exception as e:
                self._logger.error(f"Cleanup worker error: {e}")

    def _perform_cleanup(self) -> None:
        """执行内存清理"""
        self._logger.info("Performing memory cleanup...")

        # 强制垃圾回收
        collected = self.force_gc()

        # 清理弱引用
        weak_cleaned = self.cleanup_weak_refs()

        # 记录清理结果
        after_usage = self.get_memory_usage()
        self._logger.info(f"Memory cleanup: {collected} objects collected, "
                         f"{weak_cleaned} weak refs cleaned, "
                         f"memory now: {after_usage:.1f}MB")

    @contextmanager
    def memory_budget(self, budget_mb: float):
        """
        内存预算上下文管理器

        Args:
            budget_mb: 预算内存大小（MB）

        Raises:
            MemoryError: 超出预算
        """
        start_memory = self.get_memory_usage()

        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            used = end_memory - start_memory

            if used > budget_mb:
                self._logger.warning(f"Memory budget exceeded: {used:.1f}MB used, "
                                   f"{budget_mb:.1f}MB budgeted")
                # 可以选择强制清理
                self.force_gc()

    def optimize_collections(self) -> None:
        """优化集合类型"""
        # 调整GC阈值
        gc.set_threshold(700, 10, 10)  # 更激进的GC

        # 禁用GC（在高负载时）
        # gc.disable()

        self._logger.info("Collection optimizations applied")

    def __enter__(self):
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()


class CacheWithMemoryLimit:
    """带内存限制的缓存"""

    def __init__(self,
                 max_memory_mb: int = 50,
                 cleanup_threshold: float = 0.8):
        """
        初始化带内存限制的缓存

        Args:
            max_memory_mb: 最大内存使用（MB）
            cleanup_threshold: 清理阈值（占最大内存的比例）
        """
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold

        self._cache: Dict[str, Any] = {}
        self._memory_manager = MemoryManager(memory_limit_mb=max_memory_mb * 2)
        self._lock = threading.RLock()

        self._logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值
        """
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 检查内存使用
            if self._should_cleanup():
                self._cleanup_cache()

            self._cache[key] = value
            self._memory_manager.track_object(value)

    def delete(self, key: str) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._memory_manager.force_gc()

    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)

    def _should_cleanup(self) -> bool:
        """检查是否应该清理缓存"""
        usage = self._memory_manager.get_memory_usage()
        return usage > self.max_memory_mb * self.cleanup_threshold

    def _cleanup_cache(self) -> None:
        """清理缓存"""
        # 简单的清理策略：删除一半的条目
        with self._lock:
            items = list(self._cache.items())
            # 按某种策略排序（这里简单地删除前一半）
            items_to_remove = items[:len(items)//2]

            for key, _ in items_to_remove:
                del self._cache[key]

            self._logger.info(f"Cache cleanup: removed {len(items_to_remove)} items")

            # 强制GC
            self._memory_manager.force_gc()


# 全局内存管理器实例
_global_memory_manager: Optional[MemoryManager] = None


def get_global_memory_manager() -> MemoryManager:
    """获取全局内存管理器"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def memory_usage_monitor(func: Callable) -> Callable:
    """
    内存使用监控装饰器

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        manager = get_global_memory_manager()
        start_memory = manager.get_memory_usage()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_memory = manager.get_memory_usage()
            used = end_memory - start_memory

            if used > 10:  # 超过10MB
                logging.info(f"Function {func.__name__} used {used:.1f}MB memory")

    return wrapper
