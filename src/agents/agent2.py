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
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from ..utils.tool_function import tools
from ..data.api_client import DataAPIPool
from ..data.news_collector import NewsType
from .agent1 import llm_extract_events, update_entities, update_abstract_map, NewsDeduplicator

# 初始化工具
tools = tools()

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
    
    for news in expanded_news:
        try:
            # 1. 检查新闻是否已处理（ID去重）
            news_id = news.get('id')
            source = news.get('source', 'unknown')
            if news_id:
                news_key = f"{source}:{news_id}"
                if news_key in seen_news:
                    continue
                seen_news.add(news_key)
            
            title = news.get('title', '')
            content = news.get('content', '')
            
            if not title:
                continue
                
            # 2. 使用内容相似度去重（基于simhash）
            news_text = f"{title} {content}".strip()
            if deduplicator.is_duplicate(news_text):
                continue
            
            # 提取实体和事件
            extracted = llm_extract_events(title, content)
            
            if extracted:
                all_entities = []
                for ev in extracted:
                    all_entities.extend(ev['entities'])
                
                if all_entities:
                    # 优先使用新闻自身的时间戳
                    published_at = news.get('datetime')
                    if published_at and isinstance(published_at, datetime):
                        published_at = published_at.isoformat()
                    
                    # 更新实体库和事件映射（在agent2中，我们没有提取原始词，所以使用实体名称作为原始词）
                    all_entities_original = all_entities  # 使用实体名称作为原始词
                    update_entities(all_entities, all_entities_original, source, published_at)
                    update_abstract_map(extracted, source, published_at)
                    processed_count += 1
                    
        except Exception as e:
            tools.log(f"⚠️ 处理拓展新闻失败: {e}")
    
    return processed_count

async def main():
    """
    主函数
    """
    tools.log("🚀 启动 Agent2：实体拓展新闻...")
    
    # 1. 获取最近的实体
    recent_entities = get_recent_entities(time_window_days=30, limit=1)
    
    if not recent_entities:
        tools.log("📭 没有可用的实体进行新闻拓展")
        return
    
    # 2. 使用实体搜索相关新闻
    # 默认只搜索最近30天的新闻，设置full_search=True可进行全面检索
    tools.log(f"🔍 开始搜索 {len(recent_entities)} 个实体的相关新闻...")
    expanded_news = await expand_news_by_entities(recent_entities, limit_per_entity=120, time_window_days=30, full_search=False)
    tools.log(f"✅ 共搜索到 {len(expanded_news)} 条相关新闻")
    
    # 3. 处理搜索到的新闻
    if expanded_news:
        tools.log("📄 开始处理拓展的新闻...")
        processed_count = await process_expanded_news(expanded_news)
        tools.log(f"✅ 成功处理 {processed_count} 条拓展新闻")
    
    tools.log("🎉 实体拓展新闻任务完成！")

if __name__ == "__main__":
    asyncio.run(main())
