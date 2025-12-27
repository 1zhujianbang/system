"""
适配器层 - Neo4j 图存储实现

实现 GraphStore 端口，提供 Neo4j 数据库访问。
"""
import logging
import os
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver

from ...ports.graph_store import GraphStore
from ...infra.config import get_config_manager

logger = logging.getLogger(__name__)


class Neo4jAdapter(GraphStore):
    """Neo4j 图存储适配器"""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        初始化 Neo4j 适配器
        
        如果参数未提供，尝试从环境变量或配置管理器加载。
        """
        config = get_config_manager()
        
        self._uri = uri or os.getenv("NEO4J_URI") or config.get_config_value("neo4j.uri", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or config.get_config_value("neo4j.user", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD") or config.get_config_value("neo4j.password", "password")
        self._database = (
            (database or os.getenv("NEO4J_DATABASE") or config.get_config_value("neo4j.database", "")) or None
        )
        
        self._driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        """建立连接"""
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            # 验证连接
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self._uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {self._uri}: {e}")
            self._driver = None

    def is_available(self) -> bool:
        """检查图数据库是否可用"""
        if not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """关闭连接"""
        if self._driver:
            self._driver.close()
            logger.info("Closed Neo4j connection")

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行 Cypher 查询"""
        if not self._driver:
            self._connect()
            if not self._driver:
                raise ConnectionError("Neo4j driver is not available")

        params = params or {}
        try:
            session_kwargs = {"database": self._database} if self._database else {}
            with self._driver.session(**session_kwargs) as session:
                result = session.run(cypher, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}\nQuery: {cypher}\nParams: {params}")
            raise

    def execute_batch(self, operations: List[Dict[str, Any]]) -> None:
        """
        批量执行操作
        
        Args:
            operations: List of dicts with 'cypher' and 'params' keys
        """
        if not self._driver:
            self._connect()
            if not self._driver:
                raise ConnectionError("Neo4j driver is not available")
        
        if not operations:
            return

        try:
            session_kwargs = {"database": self._database} if self._database else {}
            with self._driver.session(**session_kwargs) as session:
                with session.begin_transaction() as tx:
                    for op in operations:
                        tx.run(op['cypher'], op.get('params', {}))
                    tx.commit()
            logger.info(f"Executed batch of {len(operations)} operations")
        except Exception as e:
            logger.error(f"Neo4j batch execution failed: {e}")
            raise
