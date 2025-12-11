"""
异步文件操作工具
提供高效的异步文件读写功能，支持大文件处理和原子写入
"""

import aiofiles
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import asynccontextmanager
import tempfile
import shutil


class AsyncFileHandler:
    """异步文件处理器"""

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
    async def write_json_async(file_path: Union[str, Path], data: Any, indent: int = 2) -> None:
        """
        异步写入JSON文件（原子操作）

        Args:
            file_path: 文件路径
            data: 要写入的数据
            indent: JSON缩进
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 原子写入：先写临时文件，再重命名
        temp_file = file_path.with_suffix('.tmp')

        try:
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                json_str = json.dumps(data, ensure_ascii=False, indent=indent)
                await f.write(json_str)

            # 原子重命名
            if file_path.exists():
                # Windows上需要特殊处理
                temp_file.replace(file_path)
            else:
                temp_file.rename(file_path)

        except Exception:
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
            raise

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
            content: 文件内容
            encoding: 文件编码
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
            await f.write(content)

    @staticmethod
    async def file_exists_async(file_path: Union[str, Path]) -> bool:
        """
        异步检查文件是否存在

        Args:
            file_path: 文件路径

        Returns:
            文件是否存在
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: Path(file_path).exists())

    @staticmethod
    async def get_file_size_async(file_path: Union[str, Path]) -> int:
        """
        异步获取文件大小

        Args:
            file_path: 文件路径

        Returns:
            文件大小（字节）
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: Path(file_path).stat().st_size)

    @staticmethod
    async def read_lines_async(file_path: Union[str, Path], encoding: str = 'utf-8') -> list[str]:
        """
        异步按行读取文件

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            文件行列表
        """
        content = await AsyncFileHandler.read_text_async(file_path, encoding)
        return content.splitlines()

    @staticmethod
    @asynccontextmanager
    async def open_async(file_path: Union[str, Path], mode: str = 'r', encoding: str = 'utf-8'):
        """
        异步文件上下文管理器

        Args:
            file_path: 文件路径
            mode: 打开模式
            encoding: 文件编码

        Yields:
            文件对象
        """
        file_path = Path(file_path)
        async with aiofiles.open(file_path, mode, encoding=encoding) as f:
            yield f


class FileLock:
    """文件锁（简单实现）"""

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
            async with AsyncFileHandler.open_async(self.lock_file, 'w') as f:
                await f.write(str(os.getpid()))
            self._locked = True
            return True
        except Exception:
            return False

    async def release(self) -> None:
        """释放文件锁"""
        if self._locked and self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except Exception:
                pass  # 忽略删除失败
            finally:
                self._locked = False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
