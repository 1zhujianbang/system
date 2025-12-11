"""
Serializer - 安全序列化工具
提供安全的JSON序列化功能，避免序列化失败导致程序崩溃
"""

import json
from datetime import datetime, date
from typing import Any, Dict, List
from .logging import LoggerManager


class Serializer:
    """安全序列化工具"""

    @staticmethod
    def safe_json_dumps(obj: Any, **kwargs) -> str:
        """
        安全的JSON序列化，自动处理常见不可序列化对象

        Args:
            obj: 要序列化的对象
            **kwargs: 传递给json.dumps的其他参数

        Returns:
            JSON字符串
        """
        def safe_serialize(o):
            """自定义序列化函数"""
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            elif hasattr(o, '__dict__'):
                # 对于自定义对象，返回类型信息
                return f"<Object: {type(o).__name__}>"
            elif isinstance(o, (set, frozenset)):
                # 集合转换为列表
                return list(o)
            elif isinstance(o, complex):
                # 复数转换为字符串
                return str(o)
            elif callable(o):
                # 函数对象
                return f"<Callable: {o.__name__ if hasattr(o, '__name__') else type(o).__name__}>"
            else:
                raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

        # 默认参数
        json_kwargs = {
            'ensure_ascii': False,
            'indent': 2,
            'default': safe_serialize
        }
        json_kwargs.update(kwargs)

        try:
            return json.dumps(obj, **json_kwargs)
        except Exception as e:
            # 最终降级方案
            logger = LoggerManager.get_logger(__name__)
            logger.warning(f"序列化失败，使用降级方案: {e}")

            return json.dumps({
                "error": f"Serialization failed: {e}",
                "object_type": type(obj).__name__,
                "fallback": True,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False)

    @staticmethod
    def serialize_context(context) -> str:
        """
        专门用于PipelineContext的序列化

        Args:
            context: PipelineContext实例

        Returns:
            序列化后的JSON字符串
        """
        try:
            # 安全地序列化存储的数据
            serializable_store = {}
            for k, v in context._store.items():
                try:
                    # 尝试序列化
                    json.dumps(v, default=str)
                    serializable_store[k] = v
                except (TypeError, ValueError):
                    # 序列化失败时，使用安全表示
                    serializable_store[k] = f"<Non-serializable: {type(v).__name__}>"

            # 构建结果对象
            result = {
                "store": serializable_store,
                "logs": context._logs if hasattr(context, '_logs') else [],
                "execution_history": context._execution_history if hasattr(context, '_execution_history') else []
            }

            return Serializer.safe_json_dumps(result)

        except Exception as e:
            logger = LoggerManager.get_logger(__name__)
            logger.error(f"Context序列化完全失败: {e}")

            # 最终安全保障
            return json.dumps({
                "error": f"Context serialization failed: {e}",
                "store_keys": list(context._store.keys()) if hasattr(context, '_store') else [],
                "logs_count": len(context._logs) if hasattr(context, '_logs') else 0,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False)

    @staticmethod
    def serialize_for_logging(obj: Any) -> str:
        """
        为日志记录优化的序列化，简化输出

        Args:
            obj: 要序列化的对象

        Returns:
            简化的字符串表示
        """
        try:
            if isinstance(obj, dict):
                # 对于字典，只显示键和类型信息
                return f"Dict with keys: {list(obj.keys())}"
            elif isinstance(obj, (list, tuple)):
                return f"{type(obj).__name__} with {len(obj)} items"
            elif isinstance(obj, str) and len(obj) > 100:
                return f"String({len(obj)} chars): {obj[:50]}..."
            else:
                return Serializer.safe_json_dumps(obj, indent=None)
        except Exception:
            return f"<{type(obj).__name__}>"
