"""
连接池管理器
提供HTTP连接池、数据库连接池等资源复用管理
"""

import asyncio
import aiohttp
from typing import Optional, Dict, Any, Union, List
import time
import logging
from contextlib import asynccontextmanager
import ssl


class HTTPConnectionPool:
    """HTTP连接池管理器"""

    def __init__(self,
                 max_connections: int = 20,
                 max_connections_per_host: int = 5,
                 timeout: aiohttp.ClientTimeout = None,
                 headers: Dict[str, str] = None):
        """
        初始化HTTP连接池

        Args:
            max_connections: 最大总连接数
            max_connections_per_host: 每主机最大连接数
            timeout: 请求超时设置
            headers: 默认请求头
        """
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.timeout = timeout or aiohttp.ClientTimeout(total=30)
        self.default_headers = headers or {}

        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()

        self._stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'connections_active': 0,
            'connections_created': 0
        }

        self._logger = logging.getLogger(__name__)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self) -> None:
        """初始化连接池"""
        if self._session is not None:
            return

        async with self._lock:
            if self._session is not None:
                return

            # 创建连接器
            self._connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                ttl_dns_cache=300,  # DNS缓存5分钟
                use_dns_cache=True,
                keepalive_timeout=60,  # 保持连接60秒
                enable_cleanup_closed=True,
                # SSL配置
                ssl=ssl.create_default_context(),
                ssl_check_hostname=True,
                verify_ssl=True
            )

            # 创建会话
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=self.timeout,
                headers=self.default_headers
            )

            self._logger.info(f"HTTP connection pool initialized with {self.max_connections} max connections")

    async def close(self) -> None:
        """关闭连接池"""
        if self._session is None:
            return

        async with self._lock:
            if self._session is None:
                return

            await self._session.close()
            self._session = None
            self._connector = None

            self._logger.info("HTTP connection pool closed")

    async def get(self,
                  url: str,
                  headers: Dict[str, str] = None,
                  params: Dict[str, Any] = None,
                  **kwargs) -> aiohttp.ClientResponse:
        """
        执行GET请求

        Args:
            url: 请求URL
            headers: 请求头
            params: 查询参数
            **kwargs: 其他参数

        Returns:
            响应对象
        """
        return await self._request('GET', url, headers=headers, params=params, **kwargs)

    async def post(self,
                   url: str,
                   data: Any = None,
                   json: Any = None,
                   headers: Dict[str, str] = None,
                   **kwargs) -> aiohttp.ClientResponse:
        """
        执行POST请求

        Args:
            url: 请求URL
            data: 请求数据
            json: JSON数据
            headers: 请求头
            **kwargs: 其他参数

        Returns:
            响应对象
        """
        return await self._request('POST', url, data=data, json=json, headers=headers, **kwargs)

    async def put(self,
                  url: str,
                  data: Any = None,
                  json: Any = None,
                  headers: Dict[str, str] = None,
                  **kwargs) -> aiohttp.ClientResponse:
        """
        执行PUT请求

        Args:
            url: 请求URL
            data: 请求数据
            json: JSON数据
            headers: 请求头
            **kwargs: 其他参数

        Returns:
            响应对象
        """
        return await self._request('PUT', url, data=data, json=json, headers=headers, **kwargs)

    async def delete(self,
                     url: str,
                     headers: Dict[str, str] = None,
                     **kwargs) -> aiohttp.ClientResponse:
        """
        执行DELETE请求

        Args:
            url: 请求URL
            headers: 请求头
            **kwargs: 其他参数

        Returns:
            响应对象
        """
        return await self._request('DELETE', url, headers=headers, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """执行HTTP请求"""
        if self._session is None:
            raise RuntimeError("Connection pool not initialized. Call initialize() first.")

        self._stats['requests_total'] += 1

        try:
            async with self._session.request(method, url, **kwargs) as response:
                if response.status < 400:
                    self._stats['requests_success'] += 1
                else:
                    self._stats['requests_failed'] += 1

                return response

        except Exception as e:
            self._stats['requests_failed'] += 1
            self._logger.error(f"HTTP request failed: {e}")
            raise

    def get_stats(self) -> Dict[str, int]:
        """获取连接池统计信息"""
        return self._stats.copy()

    async def health_check(self, test_url: str = "https://httpbin.org/get") -> bool:
        """
        健康检查

        Args:
            test_url: 测试URL

        Returns:
            是否健康
        """
        try:
            async with self.get(test_url) as response:
                return response.status == 200
        except Exception:
            return False


class DatabaseConnectionPool:
    """数据库连接池管理器（通用接口）"""

    def __init__(self,
                 pool_type: str = "sqlite",
                 max_connections: int = 10,
                 **pool_kwargs):
        """
        初始化数据库连接池

        Args:
            pool_type: 数据库类型 ('sqlite', 'postgres', 'mysql')
            max_connections: 最大连接数
            **pool_kwargs: 连接池参数
        """
        self.pool_type = pool_type
        self.max_connections = max_connections
        self.pool_kwargs = pool_kwargs

        self._pool = None
        self._lock = asyncio.Lock()

        self._logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """初始化连接池"""
        if self._pool is not None:
            return

        async with self._lock:
            if self._pool is not None:
                return

            if self.pool_type == "sqlite":
                # SQLite连接池
                try:
                    import aiosqlite
                    self._pool = await aiosqlite.create_pool(
                        **self.pool_kwargs,
                        max_size=self.max_connections
                    )
                except ImportError:
                    self._logger.warning("aiosqlite not available, using synchronous SQLite")
                    self._pool = None

            elif self.pool_type == "postgres":
                try:
                    import asyncpg
                    self._pool = await asyncpg.create_pool(
                        **self.pool_kwargs,
                        max_size=self.max_connections
                    )
                except ImportError:
                    raise ImportError("asyncpg required for PostgreSQL support")

            elif self.pool_type == "mysql":
                try:
                    import aiomysql
                    self._pool = await aiomysql.create_pool(
                        **self.pool_kwargs,
                        maxsize=self.max_connections
                    )
                except ImportError:
                    raise ImportError("aiomysql required for MySQL support")

            else:
                raise ValueError(f"Unsupported pool type: {self.pool_type}")

    async def close(self) -> None:
        """关闭连接池"""
        if self._pool is None:
            return

        async with self._lock:
            if self._pool is None:
                return

            if hasattr(self._pool, 'close'):
                await self._pool.close()

            self._pool = None

    @asynccontextmanager
    async def connection(self):
        """获取数据库连接"""
        if self._pool is None:
            await self.initialize()

        if self.pool_type == "sqlite":
            # SQLite特殊处理
            conn = await self._pool.acquire()
            try:
                yield conn
            finally:
                await self._pool.release(conn)
        else:
            async with self._pool.acquire() as conn:
                yield conn

    async def execute(self, query: str, *args) -> Any:
        """
        执行数据库查询

        Args:
            query: SQL查询
            *args: 查询参数

        Returns:
            查询结果
        """
        async with self.connection() as conn:
            if self.pool_type == "sqlite":
                cursor = await conn.execute(query, args)
                await conn.commit()
                return await cursor.fetchall()
            else:
                return await conn.execute(query, *args)


# 全局连接池实例
_http_pool: Optional[HTTPConnectionPool] = None
_db_pool: Optional[DatabaseConnectionPool] = None


def get_http_pool() -> HTTPConnectionPool:
    """获取全局HTTP连接池"""
    global _http_pool
    if _http_pool is None:
        _http_pool = HTTPConnectionPool()
    return _http_pool


def get_db_pool(pool_type: str = "sqlite", **kwargs) -> DatabaseConnectionPool:
    """获取全局数据库连接池"""
    global _db_pool
    if _db_pool is None or _db_pool.pool_type != pool_type:
        _db_pool = DatabaseConnectionPool(pool_type, **kwargs)
    return _db_pool


@asynccontextmanager
async def http_session(headers: Dict[str, str] = None):
    """
    HTTP会话上下文管理器

    Args:
        headers: 额外的请求头

    Yields:
        HTTPConnectionPool实例
    """
    pool = get_http_pool()
    if headers:
        # 创建带有额外头的临时会话
        temp_pool = HTTPConnectionPool(headers={**pool.default_headers, **headers})
        async with temp_pool:
            yield temp_pool
    else:
        async with pool:
            yield pool
