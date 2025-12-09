# src/agents/agent2.py
"""
智能体2：实体拓展新闻

核心功能：
1. 从实体库中获取已提取的实体
2. 使用这些实体作为关键词搜索相关新闻
3. 对搜索到的新闻进行处理，提取更多相关实体和事件
4. 更新实体库和事件映射
"""

import os
import sys
import json
import asyncio
import time
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from ..utils.tool_function import tools
from ..data.api_client import DataAPIPool
from ..data.news_collector import NewsType
from .agent1 import llm_extract_events, NewsDeduplicator
from ..utils.entity_updater import update_entities, update_abstract_map
from .agent3 import refresh_graph  # 导入知识图谱刷新功能

# 初始化工具
tools = tools()

# 环境变量/配置加载
load_dotenv(dotenv_path=tools.CONFIG_DIR / ".env.local")

def _load_agent2_settings():
    defaults = {
        "max_workers": 3,
        "rate_limit_per_sec": 1.0,
    }
    cfg_path = tools.CONFIG_DIR / "config.yaml"
    try:
        import yaml
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            cfg = data.get("agent2_config", {})
            if isinstance(cfg, dict):
                for k, v in cfg.items():
                    if k in defaults:
                        defaults[k] = v
    except Exception:
        pass
    return defaults

_agent2_settings = _load_agent2_settings()
AGENT2_MAX_WORKERS = _agent2_settings["max_workers"]
AGENT2_RATE_LIMIT = _agent2_settings["rate_limit_per_sec"]

# 初始化数据API池 - 使用更新后的API池实现
data_api_pool = DataAPIPool()

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
    available_sources = data_api_pool.list_available_sources()
    
    # 与更新后的API池兼容，移除可能的备用逻辑
    for source_name in available_sources:
        try:
            collector = data_api_pool.get_collector(source_name)
            news_collectors.append(collector)
        except Exception as e:
            tools.log(f"⚠️ 无法创建新闻收集器 {source_name}: {e}")
    
    if not news_collectors:
        tools.log("❌ 未找到可用的新闻收集器")
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
        
        tools.log(f"🔍 为实体 '{entity_name}' 搜索相关新闻，使用OR查询连接 {len(original_forms)} 个原始词...")
        tools.log(f"   📝 生成了 {len(query_batches)} 个查询批次以避免超过200字符限制")
        
        # 获取时间范围
        time_ranges = get_time_ranges(time_window_days, full_search)
        
        for collector in news_collectors:
            try:
                # 对每个查询批次和时间范围进行搜索
                for time_range in time_ranges:
                    start_date = time_range['start']
                    end_date = time_range['end']
                    
                    tools.log(f"   📅 搜索时间范围: {start_date} 至 {end_date}")
                    
                    for batch_index, batch_query in enumerate(query_batches):
                        tools.log(f"   📝 执行查询批次 {batch_index + 1}/{len(query_batches)}: '{batch_query}'")
                        
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
                                tools.log(f"⚠️ 收集器 {collector.__class__.__name__} 没有支持的搜索方法")
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
                            tools.log(f"⚠️ 查询批次 {batch_index + 1} 执行失败: {batch_error}")
            except Exception as e:
                tools.log(f"⚠️ 从 {collector.__class__.__name__} 搜索失败: {e}")
    
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
        tools.log("🔄 执行全面检索，从当前时间向前检索多个30天批次直到2020年...")
        
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
            
        tools.log(f"✅ 生成了 {batch_count} 个时间范围批次")
    
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

async def process_expanded_news(expanded_news: List[Dict]) -> int:
    """
    处理拓展的新闻，提取实体和事件
    
    Args:
        expanded_news: 拓展的新闻列表
        
    Returns:
        处理的新闻数量
    """
    processed_count = 0
    
    # 初始化新闻去重器
    deduplicator = NewsDeduplicator(threshold=tools.DEDUPE_THRESHOLD if hasattr(tools, 'DEDUPE_THRESHOLD') else 3)
    
    # 创建去重集合（ID去重）
    seen_news = set()
    
    # 并发控制
    sem = asyncio.Semaphore(AGENT2_MAX_WORKERS if AGENT2_MAX_WORKERS > 0 else 1)
    limiter_interval = 1.0 / AGENT2_RATE_LIMIT if AGENT2_RATE_LIMIT > 0 else 0
    limiter_lock = asyncio.Lock()

    async def rate_limit():
        if limiter_interval <= 0:
            return
        async with limiter_lock:
            # 简单串行限速
            await asyncio.sleep(limiter_interval)

    async def handle_one(news: Dict) -> int:
        nonlocal processed_count
        try:
            async with sem:
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

                await rate_limit()
                loop = asyncio.get_running_loop()
                extracted = await loop.run_in_executor(None, llm_extract_events, title, content)

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
    results = await asyncio.gather(*tasks)
    processed_count = sum(results)

    return processed_count


def persist_expanded_news_to_tmp(expanded_news: List[Dict], processed_ids: Set[str]) -> Optional[Path]:
    """
    将拓展新闻写入 tmp 原始文件并做去重，返回去重后的文件路径。
    """
    if not expanded_news:
        return None
    tools.RAW_NEWS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tools.DEDUPED_NEWS_TMP_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d%H%M%S")
    raw_path = tools.RAW_NEWS_TMP_DIR / f"expanded_{ts}.jsonl"
    deduped_path = tools.DEDUPED_NEWS_TMP_DIR / f"expanded_{ts}_deduped.jsonl"

    def _sanitize(item: Dict) -> Dict:
        clean = {}
        for k, v in item.items():
            if isinstance(v, datetime):
                clean[k] = v.isoformat()
            else:
                clean[k] = v
        return clean

    with open(raw_path, "w", encoding="utf-8") as f:
        for news in expanded_news:
            safe_news = _sanitize(news)
            f.write(json.dumps(safe_news, ensure_ascii=False) + "\n")

    deduper = NewsDeduplicator(threshold=tools.DEDUPE_THRESHOLD if hasattr(tools, 'DEDUPE_THRESHOLD') else 3)
    deduper.dedupe_file(raw_path, deduped_path, processed_ids)
    return deduped_path


def load_merge_rules(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("merge_rules", {})
    except Exception:
        return {}


def build_equiv_index(entities_file: Path, merge_rules_file: Path) -> Dict[str, Set[str]]:
    """
    构建实体等价词索引：实体名/原始词/合并规则别名互相指向。
    """
    idx: Dict[str, Set[str]] = defaultdict(set)
    entities = {}
    if entities_file.exists():
        try:
            entities = json.loads(entities_file.read_text(encoding="utf-8"))
        except Exception as e:
            tools.log(f"⚠️ 加载实体库失败: {e}")

    merge_rules = load_merge_rules(merge_rules_file)
    # 建反向索引 primary -> duplicates
    rev_rules: Dict[str, Set[str]] = defaultdict(set)
    for dup, primary in merge_rules.items():
        rev_rules[primary].add(dup)

    for name, data in entities.items():
        forms = set()
        if name:
            forms.add(name)
        for f in data.get("original_forms", []):
            if isinstance(f, str) and f.strip():
                forms.add(f.strip())
        if name in merge_rules:  # name 是别名
            forms.add(merge_rules[name])
        if name in rev_rules:    # 有别名指向 name
            forms.update(rev_rules[name])
        for f in forms:
            idx[f].update(forms)

    # 规则里出现但未在实体库的别名/主名
    for dup, primary in merge_rules.items():
        idx[dup].add(primary)
        idx[dup].add(dup)
        idx[primary].add(primary)
        idx[primary].add(dup)
    return idx


def expand_keywords_with_equivs(keywords: List[str], idx: Dict[str, Set[str]]) -> List[Dict[str, List[str]]]:
    """
    将输入关键词扩展为实体及其原始形态列表，供 OR 合并使用。
    """
    expanded = []
    for kw in keywords:
        kw_norm = kw.strip()
        if not kw_norm:
            continue
        forms = set([kw_norm])
        if kw_norm in idx:
            forms.update(idx[kw_norm])
        expanded.append({
            "name": kw_norm,
            "original_forms": [f for f in forms if f != kw_norm]
        })
    return expanded

async def main(args: Optional[argparse.Namespace] = None):
    """
    主函数，可通过命令行参数指定关键词、时间窗口、数量等。
    """
    tools.log("🚀 启动 Agent2：实体拓展新闻...")
    processed_ids = set()
    if tools.PROCESSED_IDS_FILE.exists():
        with open(tools.PROCESSED_IDS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            processed_ids = set(line.strip() for line in f if line.strip())
    
    # 1. 获取实体来源：命令行关键词或最近实体
    if args and args.keywords:
        merge_rules_file = tools.CONFIG_DIR / "entity_merge_rules.json"
        idx = build_equiv_index(tools.ENTITIES_FILE, merge_rules_file)
        recent_entities = expand_keywords_with_equivs(args.keywords, idx)
        tools.log(f"🔖 使用命令行关键词 {len(recent_entities)} 个作为实体（含等价词扩展）")
    else:
        entity_limit = args.entity_limit if args else 1
        window_days = args.time_window_days if args else 30
        recent_entities = get_recent_entities(time_window_days=window_days, limit=entity_limit)
        if not recent_entities:
            tools.log("📭 没有可用的实体进行新闻拓展")
            return
    
    # 2. 使用实体搜索相关新闻
    limit_per_entity = args.limit_per_entity if args else 120
    window_days = args.time_window_days if args else 30
    full_search = args.full_search if args else False
    tools.log(f"🔍 开始搜索 {len(recent_entities)} 个实体的相关新闻...")
    expanded_news = await expand_news_by_entities(
        recent_entities,
        limit_per_entity=limit_per_entity,
        time_window_days=window_days,
        full_search=full_search
    )
    tools.log(f"✅ 共搜索到 {len(expanded_news)} 条相关新闻")
    
    # 3. 处理搜索到的新闻
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
            processed_count = await process_expanded_news(news_list)
            # 清理 tmp 文件
            try:
                raw_file = tools.RAW_NEWS_TMP_DIR / deduped_path.name.replace("_deduped", "")
                if raw_file.exists():
                    raw_file.unlink()
                    tools.log(f"🗑️ 删除临时原始文件: {raw_file}")
                deduped_path.unlink()
                tools.log(f"🗑️ 删除临时去重文件: {deduped_path}")
            except Exception as e:
                tools.log(f"⚠️ 删除临时文件失败: {e}")
        tools.log(f"✅ 成功处理 {processed_count} 条拓展新闻")
    
    tools.log("🎉 实体拓展新闻任务完成！")


def parse_args():
    parser = argparse.ArgumentParser(description="Agent2 实体拓展新闻")
    parser.add_argument("--keywords", "-k", nargs="+", help="指定实体关键词列表，替代最近实体")
    parser.add_argument("--entity-limit", type=int, default=1, help="从最近实体库选择的数量（未指定关键词时生效）")
    parser.add_argument("--time-window-days", type=int, default=30, help="最近实体时间窗口 / 搜索时间窗口（天）")
    parser.add_argument("--limit-per-entity", type=int, default=120, help="每个实体搜索新闻数量上限")
    parser.add_argument("--full-search", action="store_true", help="是否全面检索至2020年")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
