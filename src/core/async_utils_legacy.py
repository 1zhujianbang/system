"""
AsyncExecutor and RateLimiter - 异步执行和限速控制工具
"""

import asyncio
import threading
import time
import logging
from typing import List, Any, Optional, Callable, Awaitable, TypeVar
from .logging import LoggerManager

T = TypeVar('T')

class AsyncExecutor:
    """统一异步执行器"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger: logging.Logger = logger or LoggerManager.get_logger(__name__)

    async def run_concurrent_tasks(
        self,
        tasks: List[Callable[[], Awaitable[Any]]],
        concurrency: int = 6,
        semaphore: Optional[asyncio.Semaphore] = None
    ) -> List[Any]:
        """
        并发执行异步任务

        Args:
            tasks: 异步任务函数列表，每个函数应是 () -> Awaitable[Any]
            concurrency: 最大并发数
            semaphore: 可选的信号量，如果不提供则自动创建

        Returns:
            任务结果列表
        """
        if not semaphore:
            semaphore = asyncio.Semaphore(concurrency)

        async def run_task(task_func):
            async with semaphore:
                try:
                    return await task_func()
                except Exception as e:
                    self.logger.error(f"Task execution failed: {e}")
                    raise

        try:
            results = await asyncio.gather(*[run_task(task) for task in tasks])
            self.logger.debug(f"Successfully executed {len(tasks)} concurrent tasks")
            return results
        except Exception as e:
            self.logger.error(f"Concurrent task execution failed: {e}")
            raise

    async def run_with_timeout(
        self,
        coro: Awaitable[Any],
        timeout: float = 30.0
    ) -> Any:
        """
        带超时的协程执行

        Args:
            coro: 要执行的协程
            timeout: 超时时间（秒）

        Returns:
            协程结果

        Raises:
            asyncio.TimeoutError: 超时异常
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Coroutine execution timed out after {timeout}s")
            raise

    def run_threaded_tasks(
        self,
        tasks: List[Any],
        func: Callable,
        max_workers: int = 7
    ) -> List[Any]:
        """
        使用线程池执行任务

        Args:
            tasks: 任务参数列表
            func: 处理函数
            max_workers: 最大线程数

        Returns:
            任务结果列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        if max_workers <= 1:
            # 顺序执行
            for task in tasks:
                try:
                    result = func(task)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Task execution failed: {e}")
                    results.append(None)
        else:
            # 并发执行
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(func, task): task for task in tasks}
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Threaded task execution failed: {e}")
                        results.append(None)

        self.logger.debug(f"Completed {len(tasks)} threaded tasks")
        return results


class RateLimiter:
    """线程安全的令牌桶限速器"""

    def __init__(self, rate_per_sec: float, logger: Optional[logging.Logger] = None) -> None:
        """
        初始化限速器

        Args:
            rate_per_sec: 每秒允许的请求数
            logger: 日志记录器
        """
        self.interval: float = 1.0 / rate_per_sec if rate_per_sec > 0 else 0
        self._lock: threading.Lock = threading.Lock()
        self._next: float = 0.0
        self.logger: logging.Logger = logger or LoggerManager.get_logger(__name__)

    def acquire(self) -> None:
        """
        获取许可，阻塞直到可以执行
        """
        if self.interval <= 0:
            return

        with self._lock:
            now = time.time()
            if now < self._next:
                sleep_time = self._next - now
                self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
                time.sleep(sleep_time)
            self._next = max(self._next, now) + self.interval

    async def acquire_async(self) -> None:
        """
        异步获取许可
        """
        if self.interval <= 0:
            return

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

        def _acquire_blocking() -> None:
            with self._lock:
                now = time.time()
                if now < self._next:
                    sleep_time = self._next - now
                    time.sleep(sleep_time)
                self._next = max(self._next, now) + self.interval

        await loop.run_in_executor(None, _acquire_blocking)

    def try_acquire(self) -> bool:
        """
        尝试获取许可，不阻塞

        Returns:
            True如果获取成功，False如果需要等待
        """
        if self.interval <= 0:
            return True

        with self._lock:
            now = time.time()
            if now >= self._next:
                self._next = max(self._next, now) + self.interval
                return True
            return False

    def get_rate_per_sec(self) -> float:
        """获取当前的速率限制"""
        return 1.0 / self.interval if self.interval > 0 else float('inf')

    def set_rate_per_sec(self, rate_per_sec: float):
        """动态调整速率限制"""
        self.interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0
        self.logger.info(f"Rate limit updated to {rate_per_sec} requests per second")
