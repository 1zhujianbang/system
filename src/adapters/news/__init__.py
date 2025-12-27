"""
适配器层 - 新闻 API 适配器

实现新闻源端口，支持 GNews 等新闻 API。
"""

from .fetch_utils import (
    normalize_news_items,
    fetch_from_collector,
    fetch_from_multiple_sources,
)
from .gdelt_adapter import GDELTAdapter

__all__ = [
    "normalize_news_items",
    "fetch_from_collector",
    "fetch_from_multiple_sources",
    "GDELTAdapter",
]
