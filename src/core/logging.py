"""
LoggerManager - 统一日志管理器
提供统一的日志配置和获取接口
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class LoggerManager:
    """统一日志管理器"""

    _loggers: dict = {}
    _configured: bool = False

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取或创建配置好的logger

        Args:
            name: logger名称，通常使用 __name__

        Returns:
            配置好的logger实例
        """
        if not cls._configured:
            cls._configure_logging()

        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)

        return cls._loggers[name]

    @classmethod
    def _configure_logging(cls):
        """统一日志配置"""
        if cls._configured:
            return

        # 根日志器配置
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # 避免重复添加处理器
        if root_logger.handlers:
            cls._configured = True
            return

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # 文件处理器 (可选)
        try:
            from ..utils.tool_function import tools
            log_file = tools.LOG_FILE
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # 文件日志失败不影响程序运行，只在控制台输出警告
            console_handler.emit(
                logging.LogRecord(
                    'LoggerManager', logging.WARNING, __file__, 0,
                    f"文件日志配置失败: {e}", (), None
                )
            )

        cls._configured = True

    @classmethod
    def set_level(cls, level: str):
        """设置全局日志级别"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if level.upper() in level_map:
            logging.getLogger().setLevel(level_map[level.upper()])

    @classmethod
    def add_file_handler(cls, file_path: Path, level: str = 'INFO'):
        """添加额外的文件处理器"""
        try:
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)

            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }
            file_handler.setLevel(level_map.get(level.upper(), logging.INFO))

            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            logging.getLogger('LoggerManager').warning(f"添加文件处理器失败: {e}")

    @classmethod
    def get_all_loggers(cls) -> dict:
        """获取所有已创建的logger（调试用）"""
        return cls._loggers.copy()
