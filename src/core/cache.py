"""
智能缓存系统
提供多级缓存支持，包括内存缓存、文件缓存和分布式缓存
"""

import time
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union, List
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading


class MemoryCache:
    """内存缓存"""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        初始化内存缓存

        Args:
            max_size: 最大缓存条目数
            ttl: 默认TTL（秒）
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在或过期返回None
        """
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                return None

            return entry['value']

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: TTL（秒），None使用默认值
        """
        with self._lock:
            # 检查缓存大小限制
            if len(self._cache) >= self._max_size:
                self._evict_oldest()

            self._cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl or self._default_ttl
            }

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
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """检查条目是否过期"""
        return time.time() - entry['timestamp'] > entry['ttl']

    def _evict_oldest(self) -> None:
        """清除最旧的条目（简单LRU）"""
        if not self._cache:
            return

        # 找到最旧的条目
        oldest_key = min(self._cache.keys(),
                        key=lambda k: self._cache[k]['timestamp'])

        del self._cache[oldest_key]


class FileCache:
    """文件缓存"""

    def __init__(self, cache_dir: Union[str, Path], max_size_mb: int = 100):
        """
        初始化文件缓存

        Args:
            cache_dir: 缓存目录
            max_size_mb: 最大缓存大小（MB）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def get(self, key: str) -> Optional[Any]:
        """
        异步获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值
        """
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return None

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(self._executor, self._load_cache_file, cache_file)
            return data
        except Exception:
            return None

    async def set(self, key: str, value: Any) -> None:
        """
        异步设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
        """
        cache_file = self._get_cache_file(key)

        # 检查缓存大小
        await self._ensure_cache_size()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._save_cache_file, cache_file, value)
        except Exception:
            pass  # 忽略缓存写入失败

    async def delete(self, key: str) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        cache_file = self._get_cache_file(key)

        if cache_file.exists():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, cache_file.unlink)
            return True
        return False

    async def clear(self) -> None:
        """清空所有缓存"""
        import shutil
        if self.cache_dir.exists():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, shutil.rmtree, self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用MD5哈希作为文件名
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"

    def _load_cache_file(self, cache_file: Path) -> Any:
        """加载缓存文件"""
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def _save_cache_file(self, cache_file: Path, value: Any) -> None:
        """保存缓存文件"""
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)

    async def _ensure_cache_size(self) -> None:
        """确保缓存大小不超过限制"""
        loop = asyncio.get_event_loop()
        total_size = await loop.run_in_executor(self._executor, self._calculate_cache_size)

        if total_size > self.max_size_bytes:
            await loop.run_in_executor(self._executor, self._cleanup_old_files)

    def _calculate_cache_size(self) -> int:
        """计算缓存总大小"""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                total_size += cache_file.stat().st_size
            except Exception:
                pass
        return total_size

    def _cleanup_old_files(self) -> None:
        """清理最旧的文件"""
        cache_files = []
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                mtime = cache_file.stat().st_mtime
                cache_files.append((cache_file, mtime))
            except Exception:
                pass

        # 按修改时间排序（最旧的在前）
        cache_files.sort(key=lambda x: x[1])

        # 删除最旧的文件直到大小合适
        target_size = self.max_size_bytes * 0.8  # 目标大小为80%
        current_size = self._calculate_cache_size()

        for cache_file, _ in cache_files:
            if current_size <= target_size:
                break
            try:
                size = cache_file.stat().st_size
                cache_file.unlink()
                current_size -= size
            except Exception:
                pass


class SmartCache:
    """智能多级缓存"""

    def __init__(self,
                 memory_ttl: int = 300,
                 file_cache_dir: Optional[Union[str, Path]] = None,
                 file_cache_size_mb: int = 100):
        """
        初始化智能缓存

        Args:
            memory_ttl: 内存缓存TTL（秒）
            file_cache_dir: 文件缓存目录
            file_cache_size_mb: 文件缓存大小限制（MB）
        """
        self.memory_cache = MemoryCache(ttl=memory_ttl)

        if file_cache_dir:
            self.file_cache = FileCache(file_cache_dir, file_cache_size_mb)
        else:
            self.file_cache = None

    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值（先内存，后文件）

        Args:
            key: 缓存键

        Returns:
            缓存值
        """
        # 先检查内存缓存
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # 再检查文件缓存
        if self.file_cache:
            value = await self.file_cache.get(key)
            if value is not None:
                # 写入内存缓存
                self.memory_cache.set(key, value)
                return value

        return None

    async def set(self, key: str, value: Any, memory_only: bool = False) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            memory_only: 是否只缓存到内存
        """
        # 设置内存缓存
        self.memory_cache.set(key, value)

        # 设置文件缓存（如果启用且不是内存专用）
        if self.file_cache and not memory_only:
            await self.file_cache.set(key, value)

    async def delete(self, key: str) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        memory_deleted = self.memory_cache.delete(key)
        file_deleted = False

        if self.file_cache:
            file_deleted = await self.file_cache.delete(key)

        return memory_deleted or file_deleted

    async def clear(self) -> None:
        """清空所有缓存"""
        self.memory_cache.clear()

        if self.file_cache:
            await self.file_cache.clear()

    def get_memory_stats(self) -> Dict[str, int]:
        """获取内存缓存统计"""
        return {
            'size': self.memory_cache.size(),
            'max_size': self.memory_cache._max_size
        }


# 全局缓存实例
_default_cache: Optional[SmartCache] = None


def get_global_cache() -> SmartCache:
    """获取全局缓存实例"""
    global _default_cache
    if _default_cache is None:
        _default_cache = SmartCache()
    return _default_cache


@asynccontextmanager
async def cached_operation(cache_key: str, operation: Callable, *args, **kwargs):
    """
    带缓存的操作上下文管理器

    Args:
        cache_key: 缓存键
        operation: 要执行的操作
        *args: 操作参数
        **kwargs: 操作关键字参数

    Yields:
        操作结果
    """
    cache = get_global_cache()

    # 尝试从缓存获取
    result = await cache.get(cache_key)
    if result is not None:
        yield result
        return

    # 执行操作
    result = operation(*args, **kwargs)
    if asyncio.iscoroutine(result):
        result = await result

    # 缓存结果
    await cache.set(cache_key, result)

    yield result
