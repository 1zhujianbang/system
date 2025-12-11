import pytest
import asyncio
import time
import tempfile
import aiofiles
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_utils import AsyncExecutor, RateLimiter
from src.utils.file_utils import AsyncFileOperations, AsyncFileLock


class TestAsyncExecutor:
    """AsyncExecutor 单元测试"""

    @pytest.fixture
    def async_executor(self):
        """创建异步执行器实例"""
        return AsyncExecutor()

    def test_async_executor_creation(self, async_executor):
        """测试异步执行器创建"""
        assert async_executor is not None
        assert hasattr(async_executor, 'run_concurrent_tasks')
        assert hasattr(async_executor, 'run_threaded_tasks')

    @pytest.mark.asyncio
    async def test_run_threaded_tasks(self, async_executor):
        """测试线程任务执行"""
        def cpu_intensive_task(x):
            time.sleep(0.1)  # 模拟CPU密集型操作
            return x ** 2

        tasks = [i for i in range(3)]
        results = await async_executor.run_threaded_tasks(
            tasks=tasks,
            func=cpu_intensive_task,
            max_workers=2
        )

        assert len(results) == 3
        assert set(results) == {0, 1, 4}  # 0^2, 1^2, 2^2

    def test_concurrency_limits(self, async_executor):
        """测试并发限制"""
        assert async_executor.max_concurrent == float('inf')  # 默认无限制

        # 创建有限并发执行器
        limited_executor = AsyncExecutor(max_concurrent=2)
        assert limited_executor.max_concurrent == 2

    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_tasks(self, async_executor):
        """测试并发任务中的错误处理"""
        async def failing_task():
            raise ValueError("Test error")

        async def success_task():
            return "success"

        tasks = [
            lambda: failing_task(),
            lambda: success_task()
        ]

        # 应该抛出异常
        with pytest.raises(ValueError, match="Test error"):
            await async_executor.run_concurrent_tasks(tasks)

    @pytest.mark.asyncio
    async def test_empty_task_list(self, async_executor):
        """测试空任务列表"""
        results = await async_executor.run_concurrent_tasks([])
        assert results == []

    @pytest.mark.asyncio
    async def test_single_task_execution(self, async_executor):
        """测试单个任务执行"""
        async def simple_task():
            return 42

        results = await async_executor.run_concurrent_tasks([lambda: simple_task()])
        assert results == [42]


class TestRateLimiter:
    """RateLimiter 单元测试"""

    @pytest.fixture
    def rate_limiter(self):
        """创建速率限制器实例"""
        return RateLimiter(rate_per_second=2.0)

    def test_initialization(self, rate_limiter):
        """测试初始化"""
        assert rate_limiter.rate_per_second == 2.0
        assert hasattr(rate_limiter, '_tokens')
        assert hasattr(rate_limiter, '_last_update')
        assert hasattr(rate_limiter, '_lock')

    @pytest.mark.asyncio
    async def test_rate_limiting(self, rate_limiter):
        """测试速率限制"""
        start_time = time.time()

        # 执行多个请求
        for i in range(3):
            await rate_limiter.acquire_async()

        end_time = time.time()

        # 3个请求，速率限制为2每秒，应该至少需要1秒
        duration = end_time - start_time
        assert duration >= 1.0

    @pytest.mark.asyncio
    async def test_zero_rate(self):
        """测试零速率限制"""
        limiter = RateLimiter(rate_per_second=0)

        # 应该立即返回（无限制）
        start_time = time.time()
        await limiter.acquire_async()
        await limiter.acquire_async()
        end_time = time.time()

        duration = end_time - start_time
        assert duration < 0.1  # 应该非常快

    @pytest.mark.asyncio
    async def test_high_rate(self):
        """测试高速率限制"""
        limiter = RateLimiter(rate_per_second=100.0)

        start_time = time.time()
        # 执行多个请求
        for i in range(10):
            await limiter.acquire_async()
        end_time = time.time()

        duration = end_time - start_time
        assert duration < 0.2  # 高速率应该很快完成

    def test_thread_safety(self, rate_limiter):
        """测试线程安全性"""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            async def run():
                try:
                    await rate_limiter.acquire_async()
                    results.append(worker_id)
                except Exception as e:
                    errors.append(e)

            # 在新的事件循环中运行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run())
            loop.close()

        # 创建多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) == 5
        assert len(errors) == 0
        assert set(results) == set(range(5))

    @pytest.mark.asyncio
    async def test_burst_requests(self, rate_limiter):
        """测试突发请求"""
        # 一次性获取多个令牌
        await rate_limiter.acquire_async()
        await rate_limiter.acquire_async()

        # 接下来应该有延迟
        start_time = time.time()
        await rate_limiter.acquire_async()  # 应该等待补充令牌
        end_time = time.time()

        duration = end_time - start_time
        assert duration >= 0.4  # 至少需要0.5秒补充令牌，但有误差


class TestAsyncFileOperations:
    """异步文件操作测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_read_write_json_async(self, temp_dir):
        """测试异步JSON文件读写"""
        test_data = {"test": "data", "number": 42, "list": [1, 2, 3]}
        file_path = temp_dir / "test.json"

        # 写入
        await AsyncFileOperations.write_json_async(file_path, test_data)

        # 读取
        result = await AsyncFileOperations.read_json_async(file_path)

        assert result["test"] == "data"
        assert result["number"] == 42
        assert result["list"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_read_write_text_async(self, temp_dir):
        """测试异步文本文件读写"""
        test_content = "Hello, World!\nThis is a test file.\n"
        file_path = temp_dir / "test.txt"

        # 写入
        await AsyncFileOperations.write_text_async(file_path, test_content)

        # 读取
        result = await AsyncFileOperations.read_text_async(file_path)

        assert result == test_content

    @pytest.mark.asyncio
    async def test_append_text_async(self, temp_dir):
        """测试异步文本追加"""
        file_path = temp_dir / "append_test.txt"

        # 初始写入
        await AsyncFileOperations.write_text_async(file_path, "Line 1\n")

        # 追加内容
        await AsyncFileOperations.append_text_async(file_path, "Line 2\n")
        await AsyncFileOperations.append_text_async(file_path, "Line 3\n")

        # 读取验证
        result = await AsyncFileOperations.read_text_async(file_path)
        expected = "Line 1\nLine 2\nLine 3\n"

        assert result == expected

    @pytest.mark.asyncio
    async def test_atomic_write_json(self, temp_dir):
        """测试原子JSON写入（使用临时文件）"""
        test_data = {"atomic": "write", "test": True}
        file_path = temp_dir / "atomic_test.json"

        # 检查临时文件是否被清理
        temp_file = file_path.with_suffix('.tmp')

        await AsyncFileOperations.write_json_async(file_path, test_data)

        # 临时文件应该被清理
        assert not temp_file.exists()
        # 主文件应该存在并包含正确数据
        assert file_path.exists()
        result = await AsyncFileOperations.read_json_async(file_path)
        assert result == test_data


class TestAsyncFileLock:
    """异步文件锁测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_file_lock_basic(self, temp_dir):
        """测试基本文件锁功能"""
        lock_file = temp_dir / "test.lock"
        lock = AsyncFileLock(lock_file)

        # 第一次获取锁应该成功
        acquired = await lock.acquire()
        assert acquired is True

        # 第二次获取锁应该失败（因为锁已被持有）
        acquired2 = await lock.acquire()
        assert acquired2 is True  # 我们的实现允许重入

        # 释放锁
        await lock.release()

        # 再次获取锁应该成功
        acquired3 = await lock.acquire()
        assert acquired3 is True

    @pytest.mark.asyncio
    async def test_file_lock_context_manager(self, temp_dir):
        """测试文件锁上下文管理器"""
        lock_file = temp_dir / "context.lock"
        test_file = temp_dir / "test_data.txt"
        test_file.write_text("initial")

        lock = AsyncFileLock(lock_file)

        async with lock:
            # 在锁内修改文件
            current_content = test_file.read_text()
            test_file.write_text(current_content + " modified")

        # 锁外验证修改
        final_content = test_file.read_text()
        assert final_content == "initial modified"

    @pytest.mark.asyncio
    async def test_file_lock_stale_cleanup(self, temp_dir):
        """测试过期锁文件清理"""
        lock_file = temp_dir / "stale.lock"

        # 创建一个假的过期锁文件（5分钟前）
        import time
        old_timestamp = time.time() - 400  # 400秒前（超过5分钟的300秒）
        async with aiofiles.open(lock_file, 'w') as f:
            await f.write(f"99999\n{old_timestamp}")

        lock = AsyncFileLock(lock_file)

        # 应该能够获取锁（清理了过期锁）
        acquired = await lock.acquire()
        assert acquired is True
