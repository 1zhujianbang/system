"""
Streamlit Key 管理器
提供全局唯一的key生成和管理机制，防止重复key错误
"""

from typing import Set, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StreamlitKeyManager:
    """Streamlit Key 全局管理器"""

    def __init__(self):
        self.used_keys: Set[str] = set()
        self.context_stack: list = []
        self.key_patterns: Dict[str, str] = {}

        # 预定义的key模式
        self._init_key_patterns()

    def _init_key_patterns(self):
        """初始化key命名模式"""
        self.key_patterns = {
            'gnews': 'gnews_{element}_{context}',
            'env': 'env_{element}_{context}',
            'pipeline': 'pipeline_{element}_{context}',
            'data': 'data_{element}_{context}',
            'config': 'config_{element}_{context}',
            'tool': 'tool_{element}_{context}',
            'system': 'system_{element}_{context}'
        }

    def push_context(self, context: str):
        """推入上下文"""
        self.context_stack.append(context)

    def pop_context(self):
        """弹出上下文"""
        if self.context_stack:
            return self.context_stack.pop()
        return None

    def get_current_context(self) -> str:
        """获取当前上下文"""
        return '_'.join(self.context_stack) if self.context_stack else 'global'

    def generate_key(self, element_type: str, element_name: str = '',
                    index: Optional[int] = None, custom_context: str = None) -> str:
        """
        生成唯一的key

        Args:
            element_type: 元素类型 (gnews, env, pipeline, etc.)
            element_name: 元素名称 (category, query, table, etc.)
            index: 可选的索引号
            custom_context: 自定义上下文

        Returns:
            唯一的key字符串
        """
        context = custom_context or self.get_current_context()

        # 获取基础模式
        base_pattern = self.key_patterns.get(element_type, '{element_type}_{element}_{context}')

        # 构建key
        key = base_pattern.format(
            element_type=element_type,
            element=element_name,
            context=context
        )

        # 添加索引
        if index is not None:
            key += f'_{index}'

        # 确保唯一性
        original_key = key
        counter = 1
        while key in self.used_keys:
            key = f"{original_key}_{counter}"
            counter += 1

        # 注册key
        self.used_keys.add(key)
        logger.debug(f"Generated unique key: {key}")

        return key

    def register_key(self, key: str) -> bool:
        """
        手动注册key

        Args:
            key: 要注册的key

        Returns:
            是否注册成功
        """
        if key in self.used_keys:
            logger.warning(f"Key already registered: {key}")
            return False

        self.used_keys.add(key)
        return True

    def is_key_used(self, key: str) -> bool:
        """检查key是否已被使用"""
        return key in self.used_keys

    def get_used_keys_count(self) -> int:
        """获取已使用的key数量"""
        return len(self.used_keys)

    def clear_context(self):
        """清空上下文栈"""
        self.context_stack.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'used_keys_count': len(self.used_keys),
            'current_context': self.get_current_context(),
            'context_stack_depth': len(self.context_stack)
        }


# 全局实例
key_manager = StreamlitKeyManager()


def get_unique_key(element_type: str, element_name: str = '',
                  index: Optional[int] = None, context: str = None) -> str:
    """
    便捷函数：生成唯一的Streamlit key

    使用示例:
        key = get_unique_key('gnews', 'category', context='config_tab')
        key = get_unique_key('env', 'table', index=0)
    """
    return key_manager.generate_key(element_type, element_name, index, context)


def push_context(context: str):
    """推入上下文"""
    key_manager.push_context(context)


def pop_context() -> Optional[str]:
    """弹出上下文"""
    return key_manager.pop_context()


class KeyContext:
    """上下文管理器"""

    def __init__(self, context: str):
        self.context = context

    def __enter__(self):
        push_context(self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pop_context()


# 使用示例:
"""
# 基本用法
key = get_unique_key('gnews', 'category', context='config_tab')

# 上下文管理
with KeyContext('config_tab'):
    category_key = get_unique_key('gnews', 'category')  # 自动包含context
    query_key = get_unique_key('gnews', 'query')

# 手动注册
key_manager.register_key('custom_key_123')
"""
