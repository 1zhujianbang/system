"""
ImportManager - 安全模块导入管理器
提供安全的动态模块导入功能
"""

import importlib
import pkgutil
from typing import List, Optional
from .logging import LoggerManager


class ImportManager:
    """安全模块导入管理器"""

    @staticmethod
    def safe_import_modules(module_names: List[str], package: str, logger=None) -> List:
        """
        安全批量导入模块，失败时记录但不中断

        Args:
            module_names: 要导入的模块名称列表
            package: 包路径
            logger: 日志记录器，如果为None则自动创建

        Returns:
            成功导入的模块列表

        Raises:
            ImportError: 如果有模块导入失败
        """
        if logger is None:
            logger = LoggerManager.get_logger(__name__)

        imported_modules = []
        failed_modules = []

        for module_name in module_names:
            try:
                module = importlib.import_module(f".{module_name}", package)
                imported_modules.append(module)
                logger.debug(f"Successfully imported {module_name}")
            except ImportError as e:
                error_msg = f"Failed to import {module_name}: {e}"
                logger.error(error_msg)
                failed_modules.append((module_name, str(e)))
            except Exception as e:
                error_msg = f"Unexpected error importing {module_name}: {e}"
                logger.warning(error_msg)
                failed_modules.append((module_name, str(e)))

        if failed_modules:
            error_msg = f"Failed to import {len(failed_modules)} modules: {failed_modules}"
            logger.error(error_msg)
            raise ImportError(error_msg)

        logger.info(f"Successfully imported {len(imported_modules)} modules")
        return imported_modules

    @staticmethod
    def get_available_modules(package_path: str) -> List[str]:
        """
        安全获取可用模块列表

        Args:
            package_path: 包路径

        Returns:
            可用模块名称列表
        """
        try:
            modules = []
            for importer, name, is_pkg in pkgutil.iter_modules([package_path]):
                if not name.startswith('_'):  # 排除私有模块
                    modules.append(name)
            return modules
        except Exception as e:
            logger = LoggerManager.get_logger(__name__)
            logger.warning(f"Failed to get available modules from {package_path}: {e}")
            return []

    @staticmethod
    def safe_import_optional(module_name: str, package: str = None, logger=None) -> Optional:
        """
        可选导入，如果失败返回None

        Args:
            module_name: 模块名称
            package: 包路径
            logger: 日志记录器

        Returns:
            导入的模块或None
        """
        if logger is None:
            logger = LoggerManager.get_logger(__name__)

        try:
            if package:
                module = importlib.import_module(f".{module_name}", package)
            else:
                module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported optional module {module_name}")
            return module
        except ImportError:
            logger.debug(f"Optional module {module_name} not available")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error importing optional module {module_name}: {e}")
            return None

    @staticmethod
    def validate_module_interface(module, required_attributes: List[str], logger=None) -> bool:
        """
        验证模块接口是否完整

        Args:
            module: 要验证的模块
            required_attributes: 必需的属性列表
            logger: 日志记录器

        Returns:
            接口是否完整
        """
        if logger is None:
            logger = LoggerManager.get_logger(__name__)

        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(module, attr):
                missing_attributes.append(attr)

        if missing_attributes:
            logger.error(f"Module {module.__name__} missing required attributes: {missing_attributes}")
            return False

        logger.debug(f"Module {module.__name__} interface validation passed")
        return True
