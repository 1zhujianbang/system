"""
文件管理工具函数

统一处理文件和目录操作，减少重复代码。
"""

import time
import aiofiles
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
import tempfile
import shutil
from ..utils.tool_function import tools


def ensure_dir(dir_path: Path) -> None:
    """
    确保目录存在，如果不存在则创建

    Args:
        dir_path: 目录路径
    """
    dir_path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(*dir_paths: Path) -> None:
    """
    确保多个目录存在

    Args:
        *dir_paths: 目录路径列表
    """
    for dir_path in dir_paths:
        ensure_dir(dir_path)


def safe_unlink(file_path: Path, log_prefix: str = "", missing_ok: bool = True) -> bool:
    """
    安全删除文件，带日志记录

    Args:
        file_path: 文件路径
        log_prefix: 日志前缀
        missing_ok: 如果文件不存在是否忽略错误

    Returns:
        删除是否成功
    """
    try:
        if file_path.exists():
            file_path.unlink(missing_ok=missing_ok)
            if log_prefix:
                tools.log(f"🗑️ 删除{log_prefix}文件: {file_path}")
            return True
        return False
    except Exception as e:
        tools.log(f"⚠️ 删除{log_prefix}文件失败 {file_path}: {e}")
        return False


def safe_unlink_multiple(file_paths: List[Path], log_prefix: str = "") -> int:
    """
    安全删除多个文件

    Args:
        file_paths: 文件路径列表
        log_prefix: 日志前缀

    Returns:
        成功删除的文件数量
    """
    deleted_count = 0
    for file_path in file_paths:
        if safe_unlink(file_path, log_prefix):
            deleted_count += 1
    return deleted_count


def generate_timestamp(format_str: str = "%Y%m%d%H%M%S") -> str:
    """
    生成统一格式的时间戳

    Args:
        format_str: 时间格式字符串

    Returns:
        格式化的时间戳字符串
    """
    return time.strftime(format_str)


def get_file_size_mb(file_path: Path) -> float:
    """
    获取文件大小（MB）

    Args:
        file_path: 文件路径

    Returns:
        文件大小（MB）
    """
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)


def cleanup_temp_files(temp_dir: Path, pattern: str = "*", max_age_hours: int = 24) -> int:
    """
    清理临时目录中的过期文件

    Args:
        temp_dir: 临时目录路径
        pattern: 文件匹配模式
        max_age_hours: 最大文件年龄（小时）

    Returns:
        删除的文件数量
    """
    if not temp_dir.exists():
        return 0

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0

    for file_path in temp_dir.glob(pattern):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                if safe_unlink(file_path, "过期临时"):
                    deleted_count += 1

    return deleted_count


class AsyncFileOperations:
    """异步文件操作工具类"""

    @staticmethod
    async def read_json_async(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        异步读取JSON文件

        Args:
            file_path: 文件路径

        Returns:
            解析后的JSON数据

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON解析错误
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)

    @staticmethod
    async def write_json_async(file_path: Union[str, Path], data: Any, indent: int = 2, ensure_ascii: bool = False) -> None:
        """
        异步写入JSON文件（原子操作）

        Args:
            file_path: 文件路径
            data: 要写入的数据
            indent: JSON缩进
            ensure_ascii: 是否确保ASCII编码
        """
        file_path = Path(file_path)
        ensure_dir(file_path.parent)

        # 使用临时文件实现原子写入
        temp_file = file_path.with_suffix('.tmp')

        try:
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                content = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
                await f.write(content)

            # 原子重命名
            if temp_file.exists():
                temp_file.replace(file_path)

        except Exception as e:
            # 清理临时文件
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise e

    @staticmethod
    async def read_text_async(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        异步读取文本文件

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            文件内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
            return await f.read()

    @staticmethod
    async def write_text_async(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
        """
        异步写入文本文件

        Args:
            file_path: 文件路径
            content: 要写入的内容
            encoding: 文件编码
        """
        file_path = Path(file_path)
        ensure_dir(file_path.parent)

        async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
            await f.write(content)

    @staticmethod
    async def append_text_async(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
        """
        异步追加文本文件

        Args:
            file_path: 文件路径
            content: 要追加的内容
            encoding: 文件编码
        """
        file_path = Path(file_path)
        ensure_dir(file_path.parent)

        async with aiofiles.open(file_path, 'a', encoding=encoding) as f:
            await f.write(content)


class AsyncFileLock:
    """异步文件锁"""

    def __init__(self, lock_file: Union[str, Path]):
        self.lock_file = Path(lock_file)
        self._locked = False

    async def acquire(self) -> bool:
        """
        获取文件锁

        Returns:
            是否成功获取锁
        """
        if self._locked:
            return True

        try:
            # 尝试创建锁文件
            async with aiofiles.open(self.lock_file, 'x', encoding='utf-8') as f:
                await f.write(f"{os.getpid()}\n{time.time()}")
            self._locked = True
            return True
        except FileExistsError:
            # 锁文件已存在，检查是否是死锁
            if await self._is_stale_lock():
                # 移除死锁文件
                try:
                    self.lock_file.unlink()
                    # 重新尝试获取锁
                    async with aiofiles.open(self.lock_file, 'x', encoding='utf-8') as f:
                        await f.write(f"{os.getpid()}\n{time.time()}")
                    self._locked = True
                    return True
                except FileExistsError:
                    pass
            return False
        except Exception:
            return False

    async def release(self) -> None:
        """释放文件锁"""
        if self._locked and self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except Exception:
                pass
            self._locked = False

    async def _is_stale_lock(self) -> bool:
        """检查是否是过期的锁文件"""
        try:
            async with aiofiles.open(self.lock_file, 'r', encoding='utf-8') as f:
                lines = (await f.read()).strip().split('\n')
                if len(lines) >= 2:
                    lock_time = float(lines[1])
                    # 如果锁文件超过5分钟，认为已过期
                    return time.time() - lock_time > 300
        except Exception:
            pass
        return False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
