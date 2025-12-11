from typing import List, Dict, Any, Optional, Set
import pandas as pd
from ..core.registry import register_tool
from ..data import news_collector
from ..utils.tool_function import tools as Tools
from ..core import ConfigManager, AsyncExecutor, RateLimiter
from ..utils.data_utils import update_entities, update_abstract_map
from ..functions.extraction import llm_extract_events, NewsDeduplicator, persist_expanded_news_to_tmp
from ..utils.file_utils import safe_unlink_multiple, safe_unlink
from ..utils.data_utils import sanitize_datetime_fields, write_jsonl_file
from pathlib import Path
import json
import asyncio
import time
from datetime import datetime, timedelta, timezone

@register_tool(
    name="fetch_news_stream",
    description="从所有配置的数据源（当前仅 GNews）获取最新新闻",
    category="Data Fetch"
)
async def fetch_news_stream(
    limit: int = 50,
    sources: Optional[List[str]] = None,
    # GNews 可选参数
    category: Optional[str] = None,
    query: Optional[str] = None,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    nullable: Optional[str] = None,
    truncate: Optional[str] = None,
    sortby: Optional[str] = None,
    in_fields: Optional[str] = None,
    page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    获取全渠道新闻数据。
    
    Args:
        limit: 每个源获取的最大条数
        sources: 指定源列表 (如 ["GNews-cn"]), 默认为所有可用源
        
    Returns:
        新闻列表 (List[Dict])
    """
    tools = Tools()
    # 配置驱动的并发上限（使用统一配置管理器）
    config_manager = ConfigManager()
    concurrency = config_manager.get_concurrency_limit("agent1_config")
    
    # 初始化 API Pool
    news_collector.init_api_pool()
    if news_collector.API_POOL is None:
        raise RuntimeError("API Pool failed to initialize")

    available_sources = news_collector.API_POOL.list_available_sources()
    if sources:
        target_sources = [s for s in sources if s in available_sources]
    else:
        target_sources = available_sources
    
    if not target_sources:
        tools.log("Warning: No valid sources to fetch from.")
        return []

    # 使用统一异步执行器
    async_executor = AsyncExecutor()

    async def fetch_one(source_name: str) -> List[Dict[str, Any]]:
        try:
            collector = news_collector.API_POOL.get_collector(source_name)
            async with collector:
                if query:
                    news = await collector.search(
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
                    news = await collector.get_top_headlines(
                        category=category,
                        limit=limit,
                        nullable=nullable,
                        from_=from_,
                        to=to,
                        query=query,
                        page=page,
                        truncate=truncate,
                    )
                for item in news:
                    if "source" not in item:
                        item["source"] = source_name
                    if "datetime" in item and hasattr(item["datetime"], "isoformat"):
                        item["datetime"] = item["datetime"].isoformat()
                tools.log(f"Fetched {len(news)} items from {source_name}")
                return news
        except Exception as e:
            tools.log(f"Error fetching from {source_name}: {e}")
            return []

    # 使用AsyncExecutor进行并发执行
    results = await async_executor.run_concurrent_tasks(
        tasks=[lambda src=src: fetch_one(src) for src in target_sources],
        concurrency=concurrency
    )

    all_news = []
    for news in results:
        all_news.extend(news)

    # 按时间倒序排序
    all_news.sort(key=lambda x: x.get("datetime") or "", reverse=True)
    return all_news


def _load_entity_equivs() -> Dict[str, Set[str]]:
    """
    构建实体及同义词索引：实体库 original_forms + 合并规则的别名。
    返回 dict: 词 -> 同义集合（包含自身）。
    """
    tools = Tools()
    idx: Dict[str, Set[str]] = {}

    def add_forms(key: str, forms: List[str]):
        if not key:
            return
        bucket = idx.setdefault(key, set())
        for f in forms:
            if f:
                bucket.add(f)
        bucket.add(key)

    # 实体库 original_forms
    try:
        if tools.ENTITIES_FILE.exists():
            with open(tools.ENTITIES_FILE, "r", encoding="utf-8") as f:
                ents = json.load(f)
            for name, data in ents.items():
                forms = data.get("original_forms", []) if isinstance(data, dict) else []
                if isinstance(forms, list):
                    flat = []
                    for x in forms:
                        if isinstance(x, str):
                            flat.append(x)
                        elif isinstance(x, list):
                            flat.extend([str(i) for i in x])
                    add_forms(name, flat)
                else:
                    add_forms(name, [])
    except Exception:
        pass

    # 合并规则（alias -> primary），双向加入
    try:
        merge_rules_file = tools.CONFIG_DIR / "entity_merge_rules.json"
        if merge_rules_file.exists():
            with open(merge_rules_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("merge_rules", {}) if isinstance(data, dict) else {}
            inv: Dict[str, List[str]] = {}
            for alias, primary in rules.items():
                add_forms(alias, [primary])
                inv.setdefault(primary, []).append(alias)
            for primary, aliases in inv.items():
                add_forms(primary, aliases)
    except Exception:
        pass

    # 展开：确保互相包含
    for k, forms in list(idx.items()):
        for f in list(forms):
            if f in idx:
                forms.update(idx[f])
    return idx


def _expand_keywords(keywords: List[str]) -> List[List[str]]:
    """
    将每个关键词扩展为同义集合列表（按输入顺序保留组）。
    """
    idx = _load_entity_equivs()
    groups: List[List[str]] = []
    for kw in keywords:
        kw_norm = (kw or "").strip()
        if not kw_norm:
            continue
        forms = set([kw_norm])
        if kw_norm in idx:
            forms.update(idx[kw_norm])
        groups.append(list(forms))
    return groups


def _build_boolean_query(groups: List[List[str]]) -> str:
    """
    根据分组构造 (A1 OR A2) AND (B1 OR B2) 形式的查询。
    """
    clauses = []
    for g in groups:
        ors = []
        for term in g:
            t = str(term).strip()
            if not t:
                continue
            # 去除内部引号，外层加引号保证短语/特殊字符安全
            t = t.replace('"', "")
            ors.append(f'"{t}"')
        if ors:
            clauses.append("(" + " OR ".join(ors) + ")")
    return " AND ".join(clauses)


@register_tool(
    name="search_news_by_keywords",
    description="按关键词搜索新闻（当前仅 GNews），支持可选时间范围与排序",
    category="Data Fetch"
)
async def search_news_by_keywords(
    keywords: List[str],
    apis: Optional[List[str]] = None,
    limit: int = 50,
    category: Optional[str] = None,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    nullable: Optional[str] = None,
    truncate: Optional[str] = None,
    sortby: Optional[str] = None,
    in_fields: Optional[str] = None,
    page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    根据关键词列表搜索新闻（GNews），返回合并后的结果。
    """
    tools = Tools()
    if not keywords:
        return []

    # 构造 (A OR A2) AND (B OR B2) 查询串
    groups = _expand_keywords(keywords)
    query_str = _build_boolean_query(groups)
    if not query_str:
        return []

    news_collector.init_api_pool()
    if news_collector.API_POOL is None:
        raise RuntimeError("API Pool failed to initialize")

    # 并发上限（使用统一配置管理器）
    config_manager = ConfigManager()
    concurrency = config_manager.get_concurrency_limit("agent2_config")

    # 使用统一异步执行器
    async_executor = AsyncExecutor()

    available_sources = news_collector.API_POOL.list_available_sources()
    if apis:
        target_sources = [s for s in apis if s in available_sources]
    else:
        target_sources = available_sources

    if not target_sources:
        tools.log("Warning: No valid sources to search from.")
        return []

    async def search_one(source_name: str) -> List[Dict[str, Any]]:
        try:
            collector = news_collector.API_POOL.get_collector(source_name)
            async with collector:
                news = await collector.search(
                    query=query_str,
                    from_=from_,
                    to=to,
                    limit=limit,
                    in_fields=in_fields,
                    nullable=nullable,
                    sortby=sortby,
                    page=page,
                    truncate=truncate,
                )
                if not news and not query_str:
                    news = await collector.get_top_headlines(
                        category=category,
                        limit=limit,
                        nullable=nullable,
                        from_=from_,
                        to=to,
                        query=None,
                        page=page,
                        truncate=truncate,
                    )
                for item in news:
                    if "source" not in item:
                        item["source"] = source_name
                    if "datetime" in item and hasattr(item["datetime"], "isoformat"):
                        item["datetime"] = item["datetime"].isoformat()
                tools.log(f"Fetched {len(news)} items from {source_name} with query: {query_str}")
                return news
        except Exception as e:
            tools.log(f"Error searching from {source_name}: {e}")
            return []

    # 使用AsyncExecutor进行并发执行
    results = await async_executor.run_concurrent_tasks(
        tasks=[lambda src=src: search_one(src) for src in target_sources],
        concurrency=concurrency
    )
    all_news: List[Dict[str, Any]] = []
    for news in results:
        all_news.extend(news)
    all_news.sort(key=lambda x: x.get("datetime") or "", reverse=True)
    return all_news


async def expand_news_by_entities(entities: List[Dict], limit_per_entity: int = 10, time_window_days: int = 30, full_search: bool = False) -> List[Dict]:
    """
    根据实体列表搜索相关新闻，支持使用原始词进行检索

    Args:
        entities: 实体列表，每个实体包含name和original_forms字段
        limit_per_entity: 每个实体搜索的新闻数量限制
        time_window_days: 默认检索时间窗口（天），默认为30天
        full_search: 是否进行全面检索，如果为True则从当前时间向前检索多个30天或更小的天数直到2020年

    Returns:
        搜索到的相关新闻列表
    """
    expanded_news = []
    news_id_set = set()  # 用于去重

    # 获取所有可用的新闻收集器
    news_collectors = []
    available_sources = news_collector.API_POOL.list_available_sources()

    # 与更新后的API池兼容，移除可能的备用逻辑
    for source_name in available_sources:
        try:
            collector = news_collector.API_POOL.get_collector(source_name)
            news_collectors.append(collector)
        except Exception as e:
            Tools().log(f"⚠️ 无法创建新闻收集器 {source_name}: {e}")

    if not news_collectors:
        Tools().log("❌ 未找到可用的新闻收集器")
        return expanded_news

    # 为每个实体搜索相关新闻
    for entity in entities:
        entity_name = entity['name']
        original_forms = entity.get('original_forms', [])

        # 构建使用OR操作符连接的搜索查询：实体名称 + 所有原始词
        all_terms = [entity_name] + original_forms

        # 生成查询批次以避免超过200字符限制
        query_batches = []
        current_batch = []
        current_length = 0

        for term in all_terms:
            quoted_term = f'"{term}"'
            term_length = len(quoted_term)

            # 如果是第一个词，直接添加；否则需要考虑OR操作符的长度
            if current_batch:
                required_length = current_length + 4 + term_length  # 4是" OR "的长度
                if required_length > 200:
                    # 如果添加当前词会超过限制，保存当前批次并开始新批次
                    query_batches.append(" OR ".join(current_batch))
                    current_batch = [quoted_term]
                    current_length = term_length
                else:
                    current_batch.append(quoted_term)
                    current_length = required_length
            else:
                current_batch.append(quoted_term)
                current_length = term_length

        # 添加最后一个批次
        if current_batch:
            query_batches.append(" OR ".join(current_batch))

        Tools().log(f"🔍 为实体 '{entity_name}' 搜索相关新闻，使用OR查询连接 {len(original_forms)} 个原始词...")
        Tools().log(f"   📝 生成了 {len(query_batches)} 个查询批次以避免超过200字符限制")

        # 获取时间范围
        time_ranges = get_time_ranges(time_window_days, full_search)

        for collector in news_collectors:
            try:
                # 对每个查询批次和时间范围进行搜索
                for time_range in time_ranges:
                    start_date = time_range['start']
                    end_date = time_range['end']

                    Tools().log(f"   📅 搜索时间范围: {start_date} 至 {end_date}")

                    for batch_index, batch_query in enumerate(query_batches):
                        Tools().log(f"   📝 执行查询批次 {batch_index + 1}/{len(query_batches)}: '{batch_query}'")

                        try:
                            # 使用搜索功能获取相关新闻，传入时间范围参数
                            search_params = {
                                'keyword' if hasattr(collector, 'search_news_by_keyword') else 'query': batch_query,
                                'limit': limit_per_entity // (len(query_batches) * len(time_ranges)) + 1  # 平均分配限制
                            }

                            # 如果收集器支持时间范围参数，则添加
                            if hasattr(collector, 'search_news_by_keyword'):
                                if 'start_date' in collector.search_news_by_keyword.__code__.co_varnames:
                                    search_params['start_date'] = start_date
                                    search_params['end_date'] = end_date
                                elif 'from_date' in collector.search_news_by_keyword.__code__.co_varnames:
                                    search_params['from_date'] = start_date
                                    search_params['to_date'] = end_date
                            elif hasattr(collector, 'search'):
                                if 'start_date' in collector.search.__code__.co_varnames:
                                    search_params['start_date'] = start_date
                                    search_params['end_date'] = end_date
                                elif 'from_date' in collector.search.__code__.co_varnames:
                                    search_params['from_date'] = start_date
                                    search_params['to_date'] = end_date

                            # 使用更新后的API调用方法
                            # 优先使用search_news_by_keyword方法，确保与更新后的API池兼容
                            if hasattr(collector, 'search_news_by_keyword'):
                                news_list = await collector.search_news_by_keyword(**search_params)
                            elif hasattr(collector, 'search'):
                                news_list = await collector.search(**search_params)
                            else:
                                Tools().log(f"⚠️ 收集器 {collector.__class__.__name__} 没有支持的搜索方法")
                                continue

                            # 为每条新闻添加实体标签并去重
                            for news in news_list:
                                # 生成唯一标识符用于去重
                                news_id = f"{news.get('url', '')}_{news.get('publishedAt', '')}"
                                if news_id not in news_id_set:
                                    news_id_set.add(news_id)
                                    news['expanded_from_entity'] = entity_name
                                    news['search_term'] = batch_query  # 记录使用的搜索词
                                    news['source'] = collector.__class__.__name__.replace('Collector', '').lower()
                                    news['query_batch'] = batch_index + 1  # 记录查询批次
                                    news['search_time_range'] = f"{start_date} to {end_date}"  # 记录搜索时间范围
                                    expanded_news.append(news)
                        except Exception as batch_error:
                            Tools().log(f"⚠️ 查询批次 {batch_index + 1} 执行失败: {batch_error}")
            except Exception as e:
                Tools().log(f"⚠️ 从 {collector.__class__.__name__} 搜索失败: {e}")

    return expanded_news


def get_time_ranges(default_days: int = 30, full_search: bool = False) -> List[Dict]:
    """
    获取搜索的时间范围列表，满足以下要求：
    1. 时间范围只能从2020年至今
    2. 默认检索前30天内的新闻
    3. 全面检索时从当前时间向前检索多个30天或更小的天数直到2020年

    Args:
        default_days: 默认的时间窗口天数，默认为30天
        full_search: 是否进行全面检索

    Returns:
        时间范围列表，每个元素包含start和end日期字符串
    """
    time_ranges = []
    now = datetime.now(timezone.utc)

    # 定义2020年1月1日作为起始时间
    start_date_2020 = datetime(2020, 1, 1, tzinfo=timezone.utc)

    if not full_search:
        # 非全面检索，只返回默认时间窗口
        end_date = now
        start_date = max(start_date_2020, now - timedelta(days=default_days))
        time_ranges.append({
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        })
    else:
        # 全面检索，从当前时间向前检索多个30天或更小的天数直到2020年
        Tools().log("🔄 执行全面检索，从当前时间向前检索多个30天批次直到2020年...")

        end_date = now
        batch_count = 0

        while end_date > start_date_2020:
            start_date = max(start_date_2020, end_date - timedelta(days=default_days))

            # 确保不重复添加相同的时间范围
            if not time_ranges or time_ranges[-1]['start'] != start_date.strftime('%Y-%m-%d'):
                time_ranges.append({
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d')
                })

            batch_count += 1
            end_date = start_date - timedelta(days=1)  # 避免日期重叠

        Tools().log(f"✅ 生成了 {batch_count} 个时间范围批次")

    return time_ranges


def get_recent_entities(time_window_days: int = 30, limit: int = 50) -> List[Dict]:
    """
    获取最近时间窗口内的实体列表，包含原始词信息

    Args:
        time_window_days: 时间窗口（天）
        limit: 返回的实体数量限制

    Returns:
        最近的实体列表，每个实体包含名称和原始词信息
    """
    entities = []
    tools = Tools()

    if not tools.ENTITIES_FILE.exists():
        tools.log("⚠️ 实体库文件不存在")
        return entities

    # 读取实体库
    with open(tools.ENTITIES_FILE, "r", encoding="utf-8") as f:
        entity_data = json.load(f)

    # 根据 first_seen 排序，获取最近的实体
    sorted_entities = sorted(
        entity_data.items(),
        key=lambda x: x[1].get('first_seen', ''),
        reverse=True
    )

    # 过滤时间窗口内的实体
    now = datetime.now(timezone.utc)
    time_window = timedelta(days=time_window_days)

    for entity_name, entity_info in sorted_entities:
        first_seen = entity_info.get('first_seen')
        if first_seen:
            try:
                # 解析时间字符串
                if 'T' in first_seen:
                    # ISO格式时间
                    seen_time = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                else:
                    # 普通格式时间
                    seen_time = datetime.strptime(first_seen, '%Y-%m-%d %H:%M:%S')
                    seen_time = seen_time.replace(tzinfo=timezone.utc)

                # 检查是否在时间窗口内
                if now - seen_time <= time_window:
                    entity_info = {
                        'name': entity_name,
                        'original_forms': entity_data[entity_name].get('original_forms', [])
                    }
                    entities.append(entity_info)
                    if len(entities) >= limit:
                        break
            except Exception as e:
                tools.log(f"⚠️ 解析实体 '{entity_name}' 的时间戳失败: {e}")

    tools.log(f"✅ 获取了 {len(entities)} 个最近实体")
    return entities


async def process_expanded_news(expanded_news: List[Dict], rate_limit: float = 1.0, max_workers: int = 3) -> int:
    """
    处理拓展的新闻，提取实体和事件

    Args:
        expanded_news: 拓展的新闻列表
        rate_limit: 每秒速率限制
        max_workers: 最大并发数

    Returns:
        处理的新闻数量
    """
    processed_count = 0
    tools = Tools()

    # 初始化新闻去重器
    deduplicator = NewsDeduplicator(threshold=tools.get_dedupe_threshold())

    # 创建去重集合（ID去重）
    seen_news = set()

    # 使用统一异步执行器和限速器
    async_executor = AsyncExecutor()
    limiter = RateLimiter(rate_limit) if rate_limit > 0 else None

    async def handle_one(news: Dict) -> int:
        nonlocal processed_count
        try:
            news_id = news.get('id')
            source = news.get('source', 'unknown')
            if news_id:
                news_key = f"{source}:{news_id}"
                if news_key in seen_news:
                    return 0
                seen_news.add(news_key)

            title = news.get('title', '')
            content = news.get('content', '')
            if not title:
                return 0

            news_text = f"{title} {content}".strip()
            if deduplicator.is_duplicate(news_text):
                return 0

            # 应用限速
            if limiter:
                await limiter.acquire_async()

            loop = asyncio.get_running_loop()
            api_pool = news_collector.API_POOL.get_llm_api_pool() if hasattr(news_collector.API_POOL, 'get_llm_api_pool') else None
            if api_pool is None:
                from ..core import LLMAPIPool
                api_pool = LLMAPIPool()
            extracted = await loop.run_in_executor(None, llm_extract_events, title, content, api_pool)

            if extracted:
                all_entities = []
                for ev in extracted:
                    all_entities.extend(ev['entities'])

                if all_entities:
                    published_at = news.get('datetime')
                    if published_at and isinstance(published_at, datetime):
                        published_at = published_at.isoformat()

                    all_entities_original = all_entities
                    update_entities(all_entities, all_entities_original, source, published_at)
                    update_abstract_map(extracted, source, published_at)
                    return 1
        except Exception as e:
            tools.log(f"⚠️ 处理拓展新闻失败: {e}")
        return 0

    tasks = [handle_one(news) for news in expanded_news]
    # 使用AsyncExecutor进行并发执行
    results = await async_executor.run_concurrent_tasks(
        tasks=tasks,
        concurrency=max_workers
    )
    processed_count = sum(results)

    return processed_count


@register_tool(
    name="expand_news_by_recent_entities",
    description="[工作流] 根据最近实体搜索相关新闻并提取事件",
    category="Workflow"
)
async def expand_news_by_recent_entities(
    entity_limit: int = 1,
    time_window_days: int = 30,
    limit_per_entity: int = 120,
    full_search: bool = False,
    rate_limit: float = 1.0,
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    根据最近实体搜索相关新闻的工作流

    Args:
        entity_limit: 实体数量限制
        time_window_days: 时间窗口（天）
        limit_per_entity: 每个实体搜索新闻数量
        full_search: 是否全面检索
        rate_limit: 速率限制
        max_workers: 最大并发数

    Returns:
        处理结果统计
    """
    tools = Tools()
    processed_ids = set()
    if tools.PROCESSED_IDS_FILE.exists():
        with open(tools.PROCESSED_IDS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            processed_ids = set(line.strip() for line in f if line.strip())

    # 获取最近实体
    recent_entities = get_recent_entities(time_window_days=time_window_days, limit=entity_limit)
    if not recent_entities:
        tools.log("📭 没有可用的实体进行新闻拓展")
        return {"processed_count": 0, "expanded_news_count": 0}

    # 搜索相关新闻
    tools.log(f"🔍 开始搜索 {len(recent_entities)} 个实体的相关新闻...")
    expanded_news = await expand_news_by_entities(
        recent_entities,
        limit_per_entity=limit_per_entity,
        time_window_days=time_window_days,
        full_search=full_search
    )
    tools.log(f"✅ 共搜索到 {len(expanded_news)} 条相关新闻")

    # 处理搜索到的新闻
    if expanded_news:
        deduped_path = persist_expanded_news_to_tmp(expanded_news, processed_ids)
        processed_count = 0
        if deduped_path and deduped_path.exists():
            tools.log(f"📄 开始处理拓展的新闻 (deduped: {deduped_path.name}) ...")
            news_list = []
            with open(deduped_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        news_list.append(json.loads(line))
                    except Exception as e:
                        tools.log(f"⚠️ 跳过无效行: {e}")
            processed_count = await process_expanded_news(news_list, rate_limit=rate_limit, max_workers=max_workers)
            # 清理 tmp 文件
            try:
                raw_file = tools.RAW_NEWS_TMP_DIR / deduped_path.name.replace("_deduped", "")
                file_paths = [raw_file, deduped_path]
                safe_unlink_multiple(file_paths, "临时")
            except Exception as e:
                tools.log(f"⚠️ 删除临时文件失败: {e}")
        tools.log(f"✅ 成功处理 {processed_count} 条拓展新闻")
        return {"processed_count": processed_count, "expanded_news_count": len(expanded_news)}

    return {"processed_count": 0, "expanded_news_count": 0}

