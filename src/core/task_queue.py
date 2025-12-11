"""
异步任务队列
提供高效的任务调度和执行，支持优先级队列和资源限制
"""

import asyncio
import time
import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import logging


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class Task:
    """任务数据类"""

    def __init__(self, priority: TaskPriority, created_at: float, task_id: str,
                 func: Callable, args: Tuple = None, kwargs: Dict = None,
                 retry_count: int = 0, max_retries: int = 3,
                 callback: Optional[Callable] = None):
        self.priority = priority
        self.created_at = created_at
        self.task_id = task_id
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.callback = callback

        # 用于优先队列的排序键
        self._sort_key = (-self.priority.value, self.created_at, self.task_id)

    def __lt__(self, other):
        """比较运算符，用于优先队列"""
        return self._sort_key < other._sort_key

    def __le__(self, other):
        return self._sort_key <= other._sort_key

    def __gt__(self, other):
        return self._sort_key > other._sort_key

    def __ge__(self, other):
        return self._sort_key >= other._sort_key


class AsyncTaskQueue:
    """异步任务队列"""

    def __init__(self,
                 max_workers: int = 4,
                 max_queue_size: int = 1000,
                 task_timeout: float = 30.0):
        """
        初始化异步任务队列

        Args:
            max_workers: 最大并发工作线程数
            max_queue_size: 队列最大大小
            task_timeout: 任务执行超时时间（秒）
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_timeout = task_timeout

        self._queue: List[Task] = []
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'queue_size': 0,
            'active_workers': 0
        }

        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """启动任务队列"""
        if self._running:
            return

        self._running = True
        self._workers = []

        # 启动工作线程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)

        self._logger.info(f"Task queue started with {self.max_workers} workers")

    async def stop(self, timeout: float = 5.0) -> None:
        """停止任务队列"""
        if not self._running:
            return

        self._running = False

        # 等待工作线程完成
        if self._workers:
            await asyncio.wait(self._workers, timeout=timeout)

            # 取消未完成的任务
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()

        self._logger.info("Task queue stopped")

    async def submit_task(self,
                         func: Callable,
                         *args,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         task_id: Optional[str] = None,
                         max_retries: int = 3,
                         callback: Optional[Callable] = None,
                         **kwargs) -> str:
        """
        提交任务到队列

        Args:
            func: 要执行的函数
            *args: 函数位置参数
            priority: 任务优先级
            task_id: 任务ID（自动生成如果未提供）
            max_retries: 最大重试次数
            callback: 任务完成回调函数
            **kwargs: 函数关键字参数

        Returns:
            任务ID

        Raises:
            QueueFullError: 队列已满
        """
        if not self._running:
            await self.start()

        async with self._lock:
            if len(self._queue) >= self.max_queue_size:
                raise QueueFullError("Task queue is full")

            if task_id is None:
                task_id = f"task_{int(time.time() * 1000000)}_{len(self._queue)}"

            task = Task(
                priority=priority,
                created_at=time.time(),
                task_id=task_id,
                func=func,
                args=args,
                kwargs=kwargs,
                max_retries=max_retries,
                callback=callback
            )

            heapq.heappush(self._queue, task)
            self._stats['queue_size'] = len(self._queue)

            self._logger.debug(f"Task {task_id} submitted to queue")

            return task_id

    async def get_queue_size(self) -> int:
        """获取队列大小"""
        async with self._lock:
            return len(self._queue)

    async def get_stats(self) -> Dict[str, int]:
        """获取队列统计信息"""
        async with self._lock:
            stats = self._stats.copy()
            stats['queue_size'] = len(self._queue)
            return stats

    async def _worker_loop(self, worker_id: int) -> None:
        """工作线程主循环"""
        self._logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # 获取任务
                task = await self._get_next_task()
                if task is None:
                    await asyncio.sleep(0.1)  # 短暂等待
                    continue

                async with self._lock:
                    self._stats['active_workers'] += 1

                # 执行任务
                await self._execute_task(task)

            except Exception as e:
                self._logger.error(f"Worker {worker_id} error: {e}")
            finally:
                async with self._lock:
                    self._stats['active_workers'] = max(0, self._stats['active_workers'] - 1)

        self._logger.debug(f"Worker {worker_id} stopped")

    async def _get_next_task(self) -> Optional[Task]:
        """获取下一个要执行的任务"""
        async with self._lock:
            if not self._queue:
                return None

            task = heapq.heappop(self._queue)
            self._stats['queue_size'] = len(self._queue)
            return task

    async def _execute_task(self, task: Task) -> None:
        """执行单个任务"""
        try:
            # 设置超时
            result = await asyncio.wait_for(
                self._run_task_func(task),
                timeout=self.task_timeout
            )

            # 调用回调
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(result, task.task_id)
                    else:
                        task.callback(result, task.task_id)
                except Exception as e:
                    self._logger.error(f"Callback error for task {task.task_id}: {e}")

            async with self._lock:
                self._stats['tasks_processed'] += 1

            self._logger.debug(f"Task {task.task_id} completed successfully")

        except asyncio.TimeoutError:
            await self._handle_task_failure(task, "Task timeout")
        except Exception as e:
            await self._handle_task_failure(task, str(e))

    async def _run_task_func(self, task: Task) -> Any:
        """运行任务函数"""
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(*task.args, **task.kwargs)
        else:
            # 在线程池中运行同步函数
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, task.func, *task.args, **task.kwargs
            )

    async def _handle_task_failure(self, task: Task, error_msg: str) -> None:
        """处理任务失败"""
        async with self._lock:
            self._stats['tasks_failed'] += 1

        self._logger.warning(f"Task {task.task_id} failed: {error_msg}")

        # 重试逻辑
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            async with self._lock:
                self._stats['tasks_retried'] += 1

            self._logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")

            # 重新提交任务（降低优先级）
            await self.submit_task(
                task.func,
                *task.args,
                priority=max(TaskPriority.LOW, TaskPriority(task.priority.value - 1)),
                task_id=f"{task.task_id}_retry_{task.retry_count}",
                max_retries=task.max_retries,
                callback=task.callback,
                **task.kwargs
            )
        else:
            self._logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} retries")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class QueueFullError(Exception):
    """队列已满异常"""
    pass


# 全局任务队列实例
_global_queue: Optional[AsyncTaskQueue] = None
_queue_lock = threading.Lock()


def get_global_task_queue() -> AsyncTaskQueue:
    """获取全局任务队列实例"""
    global _global_queue
    with _queue_lock:
        if _global_queue is None:
            _global_queue = AsyncTaskQueue()
    return _global_queue


async def submit_global_task(func: Callable,
                           *args,
                           priority: TaskPriority = TaskPriority.NORMAL,
                           **kwargs) -> str:
    """
    提交任务到全局队列

    Args:
        func: 要执行的函数
        *args: 函数参数
        priority: 任务优先级
        **kwargs: 函数关键字参数

    Returns:
        任务ID
    """
    queue = get_global_task_queue()
    return await queue.submit_task(func, *args, priority=priority, **kwargs)