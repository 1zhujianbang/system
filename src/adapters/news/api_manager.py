"""
适配器层 - GNews API 管理器

实现 NewsSource 端口，提供 GNews 新闻抓取功能。
"""
from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, List, Dict, AsyncIterator

from ...ports.extraction import (
    NewsSource, NewsSourceType, NewsSourcePool,
    NewsItem, FetchConfig, FetchResult
)
from ...infra import get_logger

# 导入GDELT适配器
from .gdelt_adapter import GDELTAdapter


class GNewsAdapter(NewsSource):
    """GNews 新闻源适配器"""

    BASE_URL = "https://gnews.io/api/v4/"

    def __init__(
        self,
        api_key: str,
        language: str = "zh",
        country: Optional[str] = None,
        name: str = "GNews",
        timeout: int = 30
    ):
        self._api_key = api_key
        self._language = language
        self._country = country
        self._name = name
        self._timeout = timeout
        self._session = None
        self._logger = get_logger(__name__)

    @property
    def source_type(self) -> NewsSourceType:
        return NewsSourceType.NEWSAPI

    @property
    def source_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def fetch(self, config: Optional[FetchConfig] = None) -> FetchResult:
        """抓取新闻"""
        config = config or FetchConfig()
        items = []

        try:
            import aiohttp

            params = {
                "lang": config.language or self._language,
                "max": min(config.max_items, 100),
                "apikey": self._api_key
            }

            if self._country:
                params["country"] = self._country

            if config.keywords:
                params["q"] = " ".join(config.keywords)

            if config.category:
                params["category"] = config.category

            # 处理日期范围参数
            if config.from_date:
                # 转换为 GNews API 所需的 ISO 8601 格式
                params["from"] = config.from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            if config.to_date:
                # 转换为 GNews API 所需的 ISO 8601 格式
                params["to"] = config.to_date.strftime("%Y-%m-%dT%H:%M:%SZ")

            # 使用 ThreadedResolver 避免 Windows 上 aiodns 的兼容性问题
            # 显式禁用 aiodns，使用默认的 ThreadedResolver
            try:
                from aiohttp.resolver import ThreadedResolver
                resolver = ThreadedResolver()
            except ImportError:
                resolver = None
            connector = aiohttp.TCPConnector(resolver=resolver)
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                url = f"{self.BASE_URL}top-headlines"
                    
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        return FetchResult(
                            items=[],
                            total_fetched=0,
                            success=False,
                            error=f"API error: {response.status} - {text}",
                            fetch_time=datetime.now(timezone.utc)
                        )

                    data = await response.json()
                    articles = data.get("articles", [])

                    for article in articles[:config.max_items]:
                        items.append(self._parse_article(article))

            return FetchResult(
                items=items,
                total_fetched=len(items),
                success=True,
                fetch_time=datetime.now(timezone.utc)
            )

        except Exception as e:
            self._logger.error(f"GNews fetch error: {e}")
            return FetchResult(
                items=[],
                total_fetched=0,
                success=False,
                error=str(e),
                fetch_time=datetime.now(timezone.utc)
            )
    async def fetch_stream(self, config: Optional[FetchConfig] = None) -> AsyncIterator[NewsItem]:
        """流式抓取新闻"""
        result = await self.fetch(config)
        for item in result.items:
            yield item

    def _parse_article(self, article: Dict) -> NewsItem:
        """解析文章为 NewsItem"""
        published_at = None
        if article.get("publishedAt"):
            try:
                published_at = datetime.fromisoformat(
                    article["publishedAt"].replace("Z", "+00:00")
                )
            except Exception:
                pass

        source = article.get("source", {})

        return NewsItem(
            id=article.get("url", ""),
            title=article.get("title", ""),
            content=article.get("content") or article.get("description", ""),
            source_name=source.get("name", self._name),
            source_url=article.get("url"),
            published_at=published_at,
            author=None,
            category=None,
            language=self._language,
            raw_data=article
        )


class NewsAPIManager(NewsSourcePool):
    """新闻 API 管理器"""

    _instance: Optional["NewsAPIManager"] = None

    def __init__(self, auto_load: bool = True):
        self._sources: Dict[str, NewsSource] = {}
        self._logger = get_logger(__name__)
        self._api_key_pool: List[str] = []
        
        # 自动从环境变量加载配置
        if auto_load:
            self.load_from_env()

    @classmethod
    def get_instance(cls) -> "NewsAPIManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = NewsAPIManager()
        return cls._instance

    def register(self, source: NewsSource) -> None:
        """注册新闻源"""
        self._sources[source.source_name] = source
        self._logger.info(f"Registered news source: {source.source_name}")

    def get_source(self, name: str) -> Optional[NewsSource]:
        """获取指定新闻源"""
        return self._sources.get(name)

    def list_sources(self) -> List[str]:
        """列出所有新闻源"""
        return list(self._sources.keys())

    async def fetch_all(self, config: Optional[FetchConfig] = None) -> Dict[str, FetchResult]:
        """从所有新闻源抓取"""
        results = {}
        tasks = []

        for name, source in self._sources.items():
            if source.is_available():
                tasks.append((name, source.fetch(config)))

        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                results[name] = FetchResult(
                    items=[],
                    total_fetched=0,
                    success=False,
                    error=str(e)
                )

        return results

    def load_from_env(self) -> None:
        """从环境变量或 KeyManager 加载配置"""
        # 尝试1: 从环境变量加载 API 密钥池
        gnews_pool = os.getenv("GNEWS_APIS_POOL", "")
        if gnews_pool:
            try:
                self._api_key_pool = json.loads(gnews_pool.strip("'"))
            except Exception as e:
                self._logger.warning(f"Failed to parse GNEWS_APIS_POOL: {e}")
        
        # 尝试2: 从 KeyManager 加载 gnews 密钥
        if not self._api_key_pool:
            try:
                from ...core import get_key_manager
                key_manager = get_key_manager()
                services = key_manager.list_services()
                gnews_keys = [s for s in services if s.startswith('gnews')]
                for service_key in gnews_keys:
                    api_key = key_manager.get_api_key(service_key)
                    if api_key:
                        self._api_key_pool.append(api_key)
                if self._api_key_pool:
                    self._logger.info(f"Loaded {len(self._api_key_pool)} GNews API keys from KeyManager")
            except Exception as e:
                self._logger.debug(f"Could not load from KeyManager: {e}")
        
        # 尝试3: 从 .env.local 文件加载
        if not self._api_key_pool:
            try:
                from pathlib import Path
                from dotenv import load_dotenv
                project_root = Path(__file__).parent.parent.parent.parent.resolve()
                dotenv_path = project_root / "config" / ".env.local"
                if dotenv_path.exists():
                    load_dotenv(dotenv_path)
                    gnews_pool = os.getenv("GNEWS_APIS_POOL", "")
                    if gnews_pool:
                        self._api_key_pool = json.loads(gnews_pool.strip("'"))
            except Exception as e:
                self._logger.debug(f"Could not load from .env.local: {e}")

        # 创建更多默认 GNews 源（扩展到20个以上）
        default_configs = [
            {"name": "GNews-cn", "language": "zh", "country": "cn"},
            {"name": "GNews-us", "language": "en", "country": "us"},
            {"name": "GNews-hk", "language": "zh", "country": "hk"},
            {"name": "GNews-tw", "language": "zh", "country": "tw"},
            {"name": "GNews-sg", "language": "en", "country": "sg"},
            {"name": "GNews-gb", "language": "en", "country": "gb"},
            {"name": "GNews-au", "language": "en", "country": "au"},
            {"name": "GNews-ca", "language": "en", "country": "ca"},
            {"name": "GNews-fr", "language": "fr", "country": "fr"},
            {"name": "GNews-de", "language": "de", "country": "de"},
            {"name": "GNews-jp", "language": "ja", "country": "jp"},
            {"name": "GNews-kr", "language": "ko", "country": "kr"},
            {"name": "GNews-ru", "language": "ru", "country": "ru"},
            {"name": "GNews-ua", "language": "uk", "country": "ua"},
            {"name": "GNews-br", "language": "pt", "country": "br"},
            {"name": "GNews-ar", "language": "es", "country": "ar"},
            {"name": "GNews-mx", "language": "es", "country": "mx"},
            {"name": "GNews-es", "language": "es", "country": "es"},
            {"name": "GNews-it", "language": "it", "country": "it"},
            {"name": "GNews-in", "language": "en", "country": "in"},
        ]

        for idx, cfg in enumerate(default_configs):
            if self._api_key_pool:
                api_key = self._api_key_pool[idx % len(self._api_key_pool)]
                source = GNewsAdapter(
                    api_key=api_key,
                    language=cfg["language"],
                    country=cfg["country"],
                    name=cfg["name"]
                )
                self.register(source)
        
        # 注册GDELT数据源
        gdelt_source = GDELTAdapter(name="GDELT")
        self.register(gdelt_source)

    def get_collector(self, name: str) -> Optional[Any]:
        """兼容旧 API：获取收集器（支持动态创建）"""
        source = self.get_source(name)
        if source:
            return source
        
        # 尝试动态创建 GNews 源
        if name.startswith("GNews-") and self._api_key_pool:
            parts = name.split("-")
            if len(parts) >= 2:
                country = parts[1].lower()
                lang_map = {
                    "cn": "zh", "hk": "zh", "tw": "zh", "sg": "en",
                    "us": "en", "gb": "en", "au": "en", "ca": "en",
                    "fr": "fr", "de": "de", "jp": "ja", "kr": "ko",
                    "ru": "ru", "ua": "uk", "br": "pt", "ar": "es",
                    "mx": "es", "es": "es", "it": "it", "in": "en",
                }
                language = lang_map.get(country, "en")
                api_key = self._api_key_pool[len(self._sources) % len(self._api_key_pool)]
                new_source = GNewsAdapter(
                    api_key=api_key,
                    language=language,
                    country=country,
                    name=name
                )
                self.register(new_source)
                return new_source
        
        raise ValueError(f"News source '{name}' not found and cannot be created")

    def list_available_sources(self) -> List[str]:
        """兼容旧 API：列出可用源"""
        return [name for name, source in self._sources.items() if source.is_available()]


def get_news_manager() -> NewsAPIManager:
    """获取新闻管理器实例"""
    return NewsAPIManager.get_instance()
