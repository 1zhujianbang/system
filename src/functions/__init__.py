# 安全导入所有子模块以确保工具被注册
from ..core.imports import ImportManager
from ..core.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

__all__ = []

# 明确指定要导入的模块列表，避免意外导入
_MODULES_TO_IMPORT = [
    'data_fetch',
    'extraction',
    'graph_ops',
    'reporting'
]

# 使用安全导入
try:
    ImportManager.safe_import_modules(_MODULES_TO_IMPORT, __package__, logger)
    logger.info("Functions package initialized successfully")
except ImportError as e:
    logger.error(f"Failed to initialize functions package: {e}")
    raise

