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

import aiodns

loop = asyncio.get_event_loop()
resolver = aiodns.DNSResolver(loop=loop)

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

class Language(Enum):
    """语言类型枚举"""
    CN = "cn"  # 中文
    EN = "en"  # 英文
    CHT = "cht"  # 繁体中文

class NewsCollector:
    def __init__(self):
        pass

    async def data_extract(self):
        """
        抓取配置中的所有新闻源（如 Blockbeats、GNews），统一写入 raw_news 目录。
        """
        tools.log("[数据获取] 🚀 开始执行 NewsCollector.data_extract")
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
                            # 约定：所有 collector 都实现 get_latest_important_news + news_to_dataframe
                            news_list = await col.get_latest_important_news(limit=50)
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
            output_file = tools.RAW_NEWS_DIR / f"raw_{timestamp}.jsonl"
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
        
        


class BlockbeatsNewsCollector:
    """Blockbeats新闻数据收集器"""
    
    BASE_URL = "https://api.theblockbeats.news/v1/"
    
    def __init__(self, language: Language = Language.CN, timeout: int = 30):
        """
        初始化新闻收集器
        
            language: 语言类型
            timeout: 超时时间（秒）
        """
        self.language = language
        self.timeout = timeout
        self.session = None
        self.cache = {}  # 简单的内存缓存
        self.cache_ttl = 300  # 缓存有效期5分钟
        self._connector = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if not self.session:
            self._connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.session.close()
    
    async def close(self):
        """显式关闭连接"""
        if self.session:
            await self.session.close()
            self.session = None
        if self._connector:
            await self._connector.close()
            self._connector = None

    async def ensure_session(self):
        """确保会话存在"""
        if not self.session:
            self._connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """生成缓存键"""
        return f"{endpoint}:{json.dumps(params, sort_keys=True)}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key in self.cache:
            cached_time, _ = self.cache[cache_key]
            return (time.time() - cached_time) < self.cache_ttl
        return False
    
    async def _make_request(self, endpoint: str, params: Dict) -> Dict[str, Any]:
        """发送API请求"""
        await self.ensure_session()  # 确保会话存在

        cache_key = self._get_cache_key(endpoint, params)
        
        # 检查缓存
        if self._is_cache_valid(cache_key):
            _, cached_data = self.cache[cache_key]
            print(f"📰 使用缓存数据: {endpoint}")
            return cached_data
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API请求失败: {response.status} - {error_text}")
                
                result = await response.json()
                
                # 缓存结果
                self.cache[cache_key] = (time.time(), result)
                
                return result
                
        except aiohttp.ClientError as e:
            raise Exception(f"网络请求错误: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"JSON解析错误: {str(e)}")
    
    async def get_flash_news(self, 
                           page: int = 1, 
                           size: int = 10,
                           news_type: NewsType = NewsType.IMPORTANT) -> List[Dict]:
        """
        获取快讯新闻
        
        Args:
            page: 页码
            size: 每页数量
            news_type: 新闻类型
            
        Returns:
            快讯新闻列表
        """
        endpoint = "open-api/open-flash"
        params = {
            "page": page,
            "size": size,
            "type": news_type.value,
            "lang": self.language.value
        }
        
        try:
            result = await self._make_request(endpoint, params)
            
            if result.get("status") == 0:
                data = result.get("data", {})
                news_list = data.get("data", [])
                
                # 处理时间戳
                for news in news_list:
                    news = self._process_news_timestamp(news)
                
                print(f"✅ 获取到 {len(news_list)} 条快讯新闻")
                return news_list
            else:
                error_msg = result.get("message", "未知错误")
                raise Exception(f"API返回错误: {error_msg}")
                
        except Exception as e:
            print(f"❌ 获取快讯新闻失败: {str(e)}")
            return []
    
    async def get_articles(self, 
                         page: int = 1, 
                         size: int = 10,
                         news_type: NewsType = NewsType.IMPORTANT) -> List[Dict]:
        """
        获取文章
        
        Args:
            page: 页码
            size: 每页数量
            news_type: 新闻类型
            
        Returns:
            文章列表
        """
        endpoint = "open-api/open-information"
        params = {
            "page": page,
            "size": size,
            "type": news_type.value,
            "lang": self.language.value
        }
        
        try:
            result = await self._make_request(endpoint, params)
            
            if result.get("status") == 0:
                data = result.get("data", {})
                articles = data.get("data", [])
                
                # 处理时间戳
                for article in articles:
                    article = self._process_news_timestamp(article)
                
                print(f"✅ 获取到 {len(articles)} 篇文章")
                return articles
            else:
                error_msg = result.get("message", "未知错误")
                raise Exception(f"API返回错误: {error_msg}")
                
        except Exception as e:
            print(f"❌ 获取文章失败: {str(e)}")
            return []
    
    def _process_news_timestamp(self, news_item: Dict) -> Dict:
        """处理新闻时间戳"""
        create_time = news_item.get("create_time")
        if create_time:
            try:
                # 将时间戳转换为datetime对象
                timestamp = int(create_time)
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                news_item["datetime"] = dt
                news_item["formatted_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                news_item["datetime"] = None
                news_item["formatted_time"] = "未知时间"
        
        return news_item
    
    async def get_latest_important_news(self, limit: int = 20) -> List[Dict]:
        """
        获取最新的重要新闻（快讯+文章）
        
        Args:
            limit: 总数量限制
            
        Returns:
            合并的重要新闻列表
        """
        # 获取快讯和文章
        flash_news = await self.get_flash_news(page=1, size=limit//2, news_type=NewsType.IMPORTANT)
        articles = await self.get_articles(page=1, size=limit//2, news_type=NewsType.IMPORTANT)
        
        # 合并并排序
        all_news = flash_news + articles
        all_news.sort(key=lambda x: x.get("datetime") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        
        # 限制数量
        return all_news[:limit]
    
    async def search_news_by_keyword(self, 
                                   keyword: str, 
                                   news_type: NewsType = None,
                                   limit: int = 50) -> List[Dict]:
        """
        根据关键词搜索新闻（通过获取多页数据实现简单搜索）
        
        Args:
            keyword: 搜索关键词
            news_type: 新闻类型，None表示搜索所有类型
            limit: 最大结果数量
            
        Returns:
            包含关键词的新闻列表
        """
        all_results = []
        page = 1
        page_size = 20
        
        while len(all_results) < limit:
            try:
                # 获取快讯
                if news_type is None or news_type == NewsType.FLASH:
                    flash_news = await self.get_flash_news(page=page, size=page_size)
                    all_results.extend(flash_news)
                
                # 获取文章
                if news_type is None or news_type == NewsType.ARTICLE:
                    articles = await self.get_articles(page=page, size=page_size)
                    all_results.extend(articles)
                
                # 如果没有更多数据，停止搜索
                if not flash_news and not articles:
                    break
                
                page += 1
                # 避免请求过快
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ 搜索新闻时出错: {str(e)}")
                break
        
        # 过滤包含关键词的新闻
        filtered_results = []
        for news in all_results:
            title = news.get("title", "").lower()
            content = news.get("content", "").lower()
            description = news.get("description", "").lower()
            
            if (keyword.lower() in title or 
                keyword.lower() in content or 
                keyword.lower() in description):
                filtered_results.append(news)
        
        # 按时间排序
        filtered_results.sort(key=lambda x: x.get("datetime") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        
        return filtered_results[:limit]
    
    def news_to_dataframe(self, news_list: List[Dict]) -> pd.DataFrame:
        """
        将新闻列表转换为DataFrame
        
        Args:
            news_list: 新闻列表
            
        Returns:
            DataFrame格式的新闻数据
        """
        if not news_list:
            return pd.DataFrame()
        
        # 提取关键字段
        processed_news = []
        source_name = "blockbeats"
        for news in news_list:
            processed_news.append({
                "id": news.get("id"),
                "source": source_name,
                "title": news.get("title", ""),
                "content": news.get("content", news.get("description", "")),
                "type": "flash" if "content" in news else "article",
                "link": news.get("link", ""),
                "image_url": news.get("pic", ""),
                "create_time": news.get("formatted_time", ""),
                "timestamp": news.get("datetime"),
                "is_original": news.get("is_original", False),
                "column": news.get("column", ""),
                # === 新增字段：用于知识图谱构建 ===
                "entities": [],          # 预留：由智能体1填充实体列表，如 ["BTC", "以太坊"]
                "event_type": None,      # 预留：事件类型，如 "regulation", "hack"
                "raw_json": json.dumps(news, default=_json_serializer, ensure_ascii=False)  # 预留：原始数据回溯
            })
        
        df = pd.DataFrame(processed_news)
        if not df.empty and "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=False)
            df = df.reset_index(drop=True)
        
        return df


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
        limit: int = 50,
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
        limit: int = 50,
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

    async def get_latest_important_news(self, limit: int = 50) -> List[Dict]:
        """
        为了与 BlockbeatsNewsCollector 接口统一，这里使用 Top Headlines 作为“重要新闻”
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
        self.cache.clear()
        print("✅ 新闻缓存已清空")


# 使用示例和测试
async def news_collector_demo():
    """新闻收集器使用示例"""
    
    async with BlockbeatsNewsCollector(language=Language.CN) as collector:
        print("🚀 Blockbeats新闻收集器演示")
        print("=" * 50)
        
        # 1. 获取快讯新闻
        print("\n1. 获取重要快讯:")
        flash_news = await collector.get_flash_news(page=1, size=5, news_type=NewsType.IMPORTANT)
        for i, news in enumerate(flash_news, 1):
            print(f"   {i}. {news.get('title')} [{news.get('formatted_time')}]")
        
        # 2. 获取文章
        print("\n2. 获取重要文章:")
        articles = await collector.get_articles(page=1, size=3, news_type=NewsType.IMPORTANT)
        for i, article in enumerate(articles, 1):
            print(f"   {i}. {article.get('title')} [{article.get('formatted_time')}]")
        
        # 3. 获取最新重要新闻
        print("\n3. 最新重要新闻:")
        important_news = await collector.get_latest_important_news(limit=5)
        for i, news in enumerate(important_news, 1):
            news_type = "快讯" if "content" in news else "文章"
            print(f"   {i}. [{news_type}] {news.get('title')}")
        
        # 4. 搜索新闻
        print("\n4. 搜索'BTC'相关新闻:")
        btc_news = await collector.search_news_by_keyword("BTC", limit=3)
        for i, news in enumerate(btc_news, 1):
            print(f"   {i}. {news.get('title')}")
        
        # 5. 获取新闻摘要
        print("\n5. 24小时新闻摘要:")
        summary = await collector.get_news_summary(hours=24)
        print(f"   总新闻数: {summary['total_news']}")
        print(f"   快讯数: {summary['flash_count']}")
        print(f"   文章数: {summary['article_count']}")
        print(f"   热门关键词: {', '.join(summary['top_keywords'])}")
        
        # 6. 转换为DataFrame
        print("\n6. 转换为DataFrame:")
        df = collector.news_to_dataframe(important_news)
        if not df.empty:
            print(f"   DataFrame形状: {df.shape}")
            print(f"   列名: {list(df.columns)}")
            print(f"   前3条新闻标题:")
            for title in df['title'].head(3):
                print(f"     - {title}")

if __name__ == "__main__":
    # 运行演示
    asyncio.run(news_collector_demo())