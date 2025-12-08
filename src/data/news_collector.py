# src/data/news_collector.py
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
from typing import Dict, List, Optional, Any
import json
from enum import Enum
import os
import hashlib

os.environ["AIODNS_NO_winloop"] = "1"

import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


from ..utils.tool_function import tools

tools = tools()

API_POOL = None

def init_api_pool():
    """惰性初始化 DataAPIPool，避免与 api_client 形成循环导入。"""
    from .api_client import DataAPIPool  # 本地导入，延后到运行时

    global API_POOL
    if API_POOL is None:
        API_POOL = DataAPIPool()

def _json_serializer(obj):
    """支持 datetime 的 JSON 序列化辅助函数"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

class NewsType(Enum):
    """新闻类型枚举"""
    FLASH = "flash"  # 快讯
    ARTICLE = "article"  # 文章
    IMPORTANT = "push"  # 重要新闻

class NewsCollector:
    def __init__(self):
        pass

    async def data_extract(
        self,
        limit: int = 10,
        category: Optional[str] = None,
        query: Optional[str] = None,
        from_: Optional[str] = None,
        to: Optional[str] = None,
        nullable: Optional[str] = None,
        truncate: Optional[str] = None,
        sortby: Optional[str] = None,
        in_fields: Optional[str] = None,
        page: Optional[int] = None,
    ):
        """
        抓取配置中的所有新闻源（如 GNews），统一写入 raw_news 目录。
        额外支持 GNews 可选参数（category/query/from/to/nullable/truncate/sortby/in/page）。
        
        Args:
            limit: 每个数据源抓取的最大条数
            category: GNews 分类
            query: 关键词（如提供则优先使用 search 端点）
            from_: ISO8601 起始时间
            to: ISO8601 结束时间
            nullable: 允许为 null 的字段，如 "description,content"
            truncate: 截断字段设置，如 "content"
            sortby: 排序方式（publishedAt|relevance）
            in_fields: 搜索字段列表，如 "title,description"
            page: 页码
        """
        tools.log(f"[数据获取] 🚀 开始执行 NewsCollector.data_extract (limit={limit})")
        init_api_pool()  # 初始化 DataAPIPool
        if API_POOL is None:
            tools.log("[数据获取] ❌ API 池未初始化")
            return []

        try:
            sources = API_POOL.list_available_sources()
            tools.log(f"[数据获取] ℹ️ 可用数据源: {sources}")
            if not sources:
                tools.log("[数据获取] ⚠️ 未在环境变量 DATA_APIS 中配置任何新闻数据源")
                return []

            all_dfs: List[pd.DataFrame] = []

            for source_name in sources:
                try:
                    tools.log(f"[数据获取] 🔍 准备获取来源: {source_name}")
                    collector = API_POOL.get_collector(source_name)

                    async def fetch_one(col):
                        async with col:
                            # 如果提供 query，优先使用 search 端点；否则使用 top-headlines
                            if query:
                                news_list = await col.search(
                                    query=query,
                                    from_=from_,
                                    to=to,
                                    limit=limit,
                                    in_fields=in_fields,
                                    nullable=nullable,
                                    sortby=sortby,
                                    page=page,
                                    truncate=truncate,
                                )
                            else:
                                news_list = await col.get_top_headlines(
                                    category=category,
                                    limit=limit,
                                    nullable=nullable,
                                    from_=from_,
                                    to=to,
                                    query=query,
                                    page=page,
                                    truncate=truncate,
                                )
                            df = col.news_to_dataframe(news_list)
                            return df

                    tools.log(f"[数据获取] ⏱ 异步抓取 {source_name} 新闻中...")
                    df = await fetch_one(collector)
                    if not df.empty:
                        all_dfs.append(df)
                        tools.log(f"[数据获取] ✅ {source_name} 获取到 {len(df)} 条新闻")
                    else:
                        tools.log(f"[数据获取] ⚠️ {source_name} 未获取到任何新闻")
                except Exception as e:
                    tools.log(f"[数据获取] ❌ 来源 {source_name} 抓取失败: {e}")

            if not all_dfs:
                tools.log("[数据获取] ⚠️ 所有来源均未获取到新闻")
                return []

            merged_df = pd.concat(all_dfs, ignore_index=True)

            timestamp = int(time.time())
            # 写入 tmp/raw_news，便于 Agent1 读取处理
            output_file = tools.RAW_NEWS_TMP_DIR / f"raw_{timestamp}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for _, row in merged_df.iterrows():
                    f.write(
                        json.dumps(
                            row.to_dict(),
                            default=_json_serializer,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            tools.log(
                f"[数据获取] ✅ 共保存 {len(merged_df)} 条新闻到 {output_file.name}"
            )

        except Exception as e:
            tools.log(f"[数据获取] ❌ 抓取失败: {e}")
        
        



class GNewsCollector:
    """
    GNews 新闻数据收集器

    文档: https://gnews.io/api/v4/{endpoint}?{parameters}&apikey=YOUR_API_KEY
    """

    BASE_URL = "https://gnews.io/api/v4/"

    def __init__(
        self,
        api_key: str,
        language: str = "zh",
        country: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        初始化 GNews 收集器

        Args:
            api_key: GNews API Key
            language: 语言代码, 如 'zh', 'en'
            country: 国家代码, 如 'cn', 'us'；可选
            timeout: 超时时间（秒）
        """
        self.api_key = api_key
        self.language = language
        self.country = country
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def __aenter__(self):
        if not self.session:
            self._connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
        if self._connector:
            await self._connector.close()
            self._connector = None

    async def _ensure_session(self):
        if not self.session:
            self._connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )

    async def _make_request(self, endpoint: str, params: Dict) -> Dict[str, Any]:
        await self._ensure_session()

        url = f"{self.BASE_URL}{endpoint}"
        # 创建参数副本，避免修改原始参数
        request_params = dict(params or {})
        
        # 使用当前收集器的API key
        request_params["apikey"] = self.api_key

        try:
            # 调试：打印本次请求的关键信息（不打印完整 key）
            safe_params = {k: (v if k != "apikey" else "***") for k, v in request_params.items()}
            print(f"[数据获取][GNews] 请求 {url} 参数: {safe_params}")
            async with self.session.get(url, params=request_params) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"GNews API 请求失败: {response.status} - {text}")
                    
                data = await response.json()
                print(f"[数据获取][GNews] 响应状态: {response.status}, 文章数: {len(data.get('articles', []) if isinstance(data, dict) else [])}")
                return data
        except aiohttp.ClientError as e:
            raise Exception(f"GNews 网络请求错误: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"GNews JSON 解析错误: {e}")

    async def get_top_headlines(
        self,
        category: Optional[str] = None,
        limit: int = 10,
        nullable: Optional[str] = None,
        from_: Optional[str] = None,
        to: Optional[str] = None,
        query: Optional[str] = None,
        page: Optional[int] = None,
        truncate: Optional[str] = None,
    ) -> List[Dict]:
        """
        获取头条新闻（Top Headlines Endpoint）

        对应 GNews 参数:
        - category: 分类，如 general, world, business, technology 等
        - lang:     语言（已由实例属性 language 决定）
        - country:  国家（已由实例属性 country 决定，可选）
        - max:      返回条数（limit）
        - nullable: 允许为 null 的字段，如 "description,content"
        - from/to:  ISO8601 时间范围
        - q:        关键字（可选）
        - page:     页码
        - truncate: 内容截断设置，如 "content"
        """
        params: Dict[str, Any] = {
            "lang": self.language,
            "max": min(limit, 100),
        }
        if self.country:
            params["country"] = self.country
        if category:
            params["category"] = category
        if nullable:
            params["nullable"] = nullable
        if from_:
            params["from"] = from_
        if to:
            params["to"] = to
        if query:
            params["q"] = query
        if page is not None:
            params["page"] = page
        if truncate:
            params["truncate"] = truncate

        data = await self._make_request("top-headlines", params)
        articles = data.get("articles", []) or []

        for art in articles:
            self._process_timestamp(art)

        return articles[:limit]

    async def search(
        self,
        query: str,
        from_: Optional[str] = None,
        to: Optional[str] = None,
        limit: int = 10,
        in_fields: Optional[str] = None,
        nullable: Optional[str] = None,
        sortby: Optional[str] = None,
        page: Optional[int] = None,
        truncate: Optional[str] = None,
    ) -> List[Dict]:
        """
        使用 Search Endpoint 按关键字搜索新闻

        对应 GNews 参数:
        - q:       关键字（必填）
        - lang:    语言（已由实例属性 language 决定）
        - country: 国家（已由实例属性 country 决定，可选）
        - max:     返回条数（limit）
        - in:      搜索字段，如 "title,description"
        - nullable: 允许为 null 的字段，如 "description,content"
        - from / to: ISO8601 时间范围
        - sortby:  "publishedAt" | "relevance"
        - page:    页码
        - truncate: 内容截断设置，如 "content"
        """
        
        params: Dict[str, Any] = {
            "q": query,
            "lang": self.language,
            "max": min(limit, 100),
        }

        if self.country:
            params["country"] = self.country
        if from_:
            params["from"] = from_
        if to:
            params["to"] = to
        if in_fields:
            params["in"] = in_fields
        if nullable:
            params["nullable"] = nullable
        if sortby:
            params["sortby"] = sortby
        if page is not None:
            params["page"] = page
        if truncate:
            params["truncate"] = truncate

        data = await self._make_request("search", params)
        articles = data.get("articles", []) or []

        for art in articles:
            self._process_timestamp(art)

        return articles[:limit]

    def _process_timestamp(self, article: Dict) -> None:
        """
        处理 GNews 的 publishedAt 字段，转换为 datetime 和本地格式化时间
        """
        ts = article.get("publishedAt")
        if not ts:
            article["datetime"] = None
            article["formatted_time"] = "未知时间"
            return
        try:
            # 例如: 2025-12-04T09:30:00Z
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            article["datetime"] = dt
            article["formatted_time"] = dt.astimezone(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception:
            article["datetime"] = None
            article["formatted_time"] = "未知时间"

    async def get_latest_important_news(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的重要新闻
        """
        return await self.get_top_headlines(limit=limit)

    def news_to_dataframe(self, news_list: List[Dict]) -> pd.DataFrame:
        """
        将 GNews 文章列表转换为与 Agent1 兼容的 DataFrame 结构
        """
        if not news_list:
            return pd.DataFrame()

        processed: List[Dict[str, Any]] = []
        source_name = "gnews"

        for article in news_list:
            url = article.get("url", "")
            title = article.get("title", "") or ""
            content = article.get("content") or article.get("description", "") or ""
            img = article.get("image", "")
            src = article.get("source", {}) or {}
            src_name = src.get("name") or source_name

            processed.append(
                {
                    # 使用 URL 作为全局唯一 ID，后续 Agent1 会组合为 "gnews:<url>"
                    "id": url or hashlib.md5(title.encode("utf-8")).hexdigest(),
                    "source": src_name,
                    "title": title,
                    "content": content,
                    "type": "article",
                    "link": url,
                    "image_url": img,
                    "create_time": article.get("formatted_time", ""),
                    "timestamp": article.get("datetime"),
                    "is_original": False,
                    "column": src_name,
                    "entities": [],
                    "event_type": None,
                    "raw_json": json.dumps(
                        article, default=_json_serializer, ensure_ascii=False
                    ),
                }
            )

        df = pd.DataFrame(processed)
        if not df.empty and "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        return df
    
    async def get_news_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取指定时间范围内的新闻摘要
        
        Args:
            hours: 时间范围（小时）
            
        Returns:
            新闻摘要统计
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        # 获取最近的重要新闻
        all_news = await self.get_latest_important_news(limit=100)
        
        # 过滤时间范围内的新闻
        recent_news = []
        for news in all_news:
            news_time = news.get("datetime")
            if news_time and start_time <= news_time <= end_time:
                recent_news.append(news)
        
        # 统计信息
        flash_count = sum(1 for news in recent_news if "content" in news)
        article_count = len(recent_news) - flash_count
        
        # 提取热门关键词（简单实现）
        all_titles = " ".join([news.get("title", "") for news in recent_news])
        words = all_titles.split()
        from collections import Counter
        word_freq = Counter(words)
        top_keywords = [word for word, count in word_freq.most_common(10) if len(word) > 1]
        
        return {
            "total_news": len(recent_news),
            "flash_count": flash_count,
            "article_count": article_count,
            "time_range": f"最近{hours}小时",
            "top_keywords": top_keywords[:5],
            "latest_news": recent_news[:10]  # 最新10条新闻
        }
    
    def clear_cache(self):
        """清空缓存"""
        pass 

# 使用示例和测试

if __name__ == "__main__":
    pass
    
