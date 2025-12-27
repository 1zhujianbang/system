"""
端口层 - 图存储接口

定义图数据库的抽象操作接口。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class GraphStore(ABC):
    """图存储抽象接口"""

    @abstractmethod
    def is_available(self) -> bool:
        """检查图数据库是否可用"""
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭连接"""
        ...

    @abstractmethod
    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行 Cypher 查询
        
        Args:
            cypher: Cypher 查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        ...
        
    @abstractmethod
    def execute_batch(self, operations: List[Dict[str, Any]]) -> None:
        """
        批量执行操作（用于迁移或大量写入）
        
        Args:
            operations: 操作列表，每个操作包含 'cypher' 和 'params'
        """
        ...
