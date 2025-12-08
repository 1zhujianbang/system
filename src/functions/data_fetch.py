from typing import List, Dict, Any, Optional, Set
import pandas as pd
from ..core.registry import register_tool
from ..data import news_collector
from ..utils.tool_function import tools as Tools
import json

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

    all_news = []
    
    for source_name in target_sources:
        try:
            collector = news_collector.API_POOL.get_collector(source_name)
            
            # 使用 async with 确保连接正确管理
            async with collector:
                # 若提供 query 则使用 search，否则使用 top_headlines
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

                # 确保 source 字段存在
                for item in news:
                    if "source" not in item:
                        item["source"] = source_name
                    
                    # 转换 datetime 为 ISO 字符串以便序列化
                    if "datetime" in item and hasattr(item["datetime"], "isoformat"):
                        item["datetime"] = item["datetime"].isoformat()
                        
                all_news.extend(news)
                tools.log(f"Fetched {len(news)} items from {source_name}")
                 
        except Exception as e:
            tools.log(f"Error fetching from {source_name}: {e}")

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

    available_sources = news_collector.API_POOL.list_available_sources()
    if apis:
        target_sources = [s for s in apis if s in available_sources]
    else:
        target_sources = available_sources

    if not target_sources:
        tools.log("Warning: No valid sources to search from.")
        return []

    all_news: List[Dict[str, Any]] = []

    for source_name in target_sources:
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
                # fallback: 若未提供 query（极端情况），尝试 top-headlines
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
                all_news.extend(news)
                tools.log(f"Fetched {len(news)} items from {source_name} with query: {query_str}")
        except Exception as e:
            tools.log(f"Error searching from {source_name}: {e}")

    all_news.sort(key=lambda x: x.get("datetime") or "", reverse=True)
    return all_news

