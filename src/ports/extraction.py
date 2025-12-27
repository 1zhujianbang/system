"""
端口层 - 新闻源与抽取接口

定义新闻数据采集和实体/事件抽取的抽象接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncIterator


# =============================================================================
# NewsSource（新闻源端口）
# =============================================================================


class NewsSourceType(str, Enum):
    """新闻源类型"""
    NEWSAPI = "newsapi"
    RSS = "rss"
    SCRAPER = "scraper"
    LOCAL = "local"
    CUSTOM = "custom"
    GDELT = "gdelt"


@dataclass
class NewsItem:
    """单条新闻数据"""
    id: str
    title: str
    content: str
    source_name: str
    source_url: Optional[str] = None
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    category: Optional[str] = None
    language: str = "zh"
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "author": self.author,
            "category": self.category,
            "language": self.language,
            "metadata": self.metadata,
        }


@dataclass
class FetchConfig:
    """抓取配置"""
    max_items: int = 100
    language: str = "zh"
    category: Optional[str] = None
    keywords: Optional[List[str]] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    timeout_seconds: float = 30.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FetchResult:
    """抓取结果"""
    items: List[NewsItem]
    total_fetched: int
    success: bool
    error: Optional[str] = None
    fetch_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NewsSource(ABC):
    """新闻源抽象接口"""

    @property
    @abstractmethod
    def source_type(self) -> NewsSourceType:
        """获取新闻源类型"""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """获取新闻源名称"""
        ...

    @abstractmethod
    async def fetch(self, config: Optional[FetchConfig] = None) -> FetchResult:
        """
        抓取新闻

        Args:
            config: 抓取配置

        Returns:
            抓取结果
        """
        ...

    @abstractmethod
    async def fetch_stream(self, config: Optional[FetchConfig] = None) -> AsyncIterator[NewsItem]:
        """
        流式抓取新闻

        Args:
            config: 抓取配置

        Yields:
            新闻条目
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查新闻源是否可用"""
        ...


class NewsSourcePool(ABC):
    """新闻源池抽象接口"""

    @abstractmethod
    def register(self, source: NewsSource) -> None:
        """注册新闻源"""
        ...

    @abstractmethod
    def get_source(self, name: str) -> Optional[NewsSource]:
        """获取指定新闻源"""
        ...

    @abstractmethod
    def list_sources(self) -> List[str]:
        """列出所有新闻源"""
        ...

    @abstractmethod
    async def fetch_all(self, config: Optional[FetchConfig] = None) -> Dict[str, FetchResult]:
        """从所有新闻源抓取"""
        ...


# =============================================================================
# EntityExtractor（实体抽取端口）
# =============================================================================


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    name: str
    entity_type: str  # PERSON, ORG, LOCATION, etc.
    confidence: float = 1.0
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityExtractionResult:
    """实体抽取结果"""
    entities: List[ExtractedEntity]
    source_text: str
    success: bool
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None


class EntityExtractor(ABC):
    """实体抽取抽象接口"""

    @abstractmethod
    async def extract(self, text: str, context: Optional[Dict[str, Any]] = None) -> EntityExtractionResult:
        """
        从文本中抽取实体

        Args:
            text: 输入文本
            context: 额外上下文信息

        Returns:
            抽取结果
        """
        ...

    @abstractmethod
    async def extract_batch(
        self,
        texts: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[EntityExtractionResult]:
        """
        批量抽取实体

        Args:
            texts: 输入文本列表
            context: 额外上下文信息

        Returns:
            抽取结果列表
        """
        ...


# =============================================================================
# EventExtractor（事件抽取端口）
# =============================================================================


@dataclass
class ExtractedEvent:
    """抽取的事件"""
    abstract: str
    event_type: Optional[str] = None
    time: Optional[datetime] = None
    location: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventExtractionResult:
    """事件抽取结果"""
    events: List[ExtractedEvent]
    source_text: str
    success: bool
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None


class EventExtractor(ABC):
    """事件抽取抽象接口"""

    @abstractmethod
    async def extract(self, text: str, context: Optional[Dict[str, Any]] = None) -> EventExtractionResult:
        """
        从文本中抽取事件

        Args:
            text: 输入文本
            context: 额外上下文信息

        Returns:
            抽取结果
        """
        ...

    @abstractmethod
    async def extract_batch(
        self,
        texts: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[EventExtractionResult]:
        """
        批量抽取事件

        Args:
            texts: 输入文本列表
            context: 额外上下文信息

        Returns:
            抽取结果列表
        """
        ...


# =============================================================================
# Deduplicator（去重端口）
# =============================================================================


@dataclass
class DeduplicationResult:
    """去重结果"""
    unique_items: List[Any]
    duplicate_groups: List[List[Any]]
    total_input: int
    total_unique: int
    total_duplicates: int


class Deduplicator(ABC):
    """去重抽象接口"""

    @abstractmethod
    def deduplicate(self, items: List[Any], key_func=None) -> DeduplicationResult:
        """
        对条目进行去重

        Args:
            items: 输入条目列表
            key_func: 提取去重键的函数

        Returns:
            去重结果
        """
        ...

    @abstractmethod
    def is_duplicate(self, item: Any, existing_items: List[Any]) -> bool:
        """
        检查单个条目是否重复

        Args:
            item: 待检查条目
            existing_items: 现有条目列表

        Returns:
            是否重复
        """
        ...

    @abstractmethod
    def compute_similarity(self, item1: Any, item2: Any) -> float:
        """
        计算两个条目的相似度

        Args:
            item1: 条目1
            item2: 条目2

        Returns:
            相似度分数 (0.0-1.0)
        """
        ...


class TextDeduplicator(Deduplicator):
    """文本去重抽象接口（支持SimHash等算法）"""

    @abstractmethod
    def compute_hash(self, text: str) -> str:
        """
        计算文本指纹哈希

        Args:
            text: 输入文本

        Returns:
            指纹哈希
        """
        ...

    @abstractmethod
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        计算两个哈希的汉明距离

        Args:
            hash1: 哈希1
            hash2: 哈希2

        Returns:
            汉明距离
        """
        ...
