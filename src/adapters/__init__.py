"""
适配器层（Adapters Layer）。

提供对外部依赖的具体实现：
- sqlite: SQLite 存储适配器
- llm: LLM 提供商适配器
- export: JSON 导出/快照适配器
- news: 新闻源适配器
- extraction: 实体/事件抽取适配器
"""

from .sqlite.kg_read_store import SQLiteKGReadStore
from .llm import (
    OpenAIAdapter,
    KimiAdapter,
    AliyunAdapter,
    DefaultLLMClientPool,
    create_llm_client,
)
from .llm.pool import DefaultLLMPool, get_llm_pool, set_llm_pool
from .export import (
    JsonSnapshotWriter,
    JsonSnapshotReader,
    CompatJsonExporter,
)
from .news.api_manager import GNewsAdapter, NewsAPIManager, get_news_manager
from .news.gdelt_adapter import GDELTAdapter
from .extraction import LLMEntityExtractor, LLMEventExtractor

__all__ = [
    # SQLite
    "SQLiteKGReadStore",
    # LLM
    "OpenAIAdapter",
    "KimiAdapter",
    "AliyunAdapter",
    "DefaultLLMClientPool",
    "create_llm_client",
    "DefaultLLMPool",
    "get_llm_pool",
    "set_llm_pool",
    # Export
    "JsonSnapshotWriter",
    "JsonSnapshotReader",
    "CompatJsonExporter",
    # News
    "GNewsAdapter",
    "GDELTAdapter",
    "NewsAPIManager",
    "get_news_manager",
    # Extraction
    "LLMEntityExtractor",
    "LLMEventExtractor",
]



