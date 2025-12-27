"""
新闻数据获取公共工具函数

统一处理新闻数据获取、转换和处理的重复逻辑。
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
logger = logging.getLogger(__name__)


def normalize_news_items(news_list: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    """
    标准化新闻数据项

    Args:
        news_list: 原始新闻列表
        source_name: 数据源名称

    Returns:
        标准化后的新闻列表
    """
    for item in news_list:
        # 设置数据源
        if "source" not in item:
            item["source"] = source_name

        # 转换datetime对象为ISO格式字符串
        if "datetime" in item and hasattr(item["datetime"], "isoformat"):
            item["datetime"] = item["datetime"].isoformat()

    return news_list


async def fetch_from_collector(
    collector: Any,
    source_name: str,
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    nullable: Optional[str] = None,
    truncate: Optional[str] = None,
    sortby: Optional[str] = None,
    in_fields: Optional[str] = None,
    page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    从新闻收集器获取数据的统一接口

    Args:
        collector: 新闻收集器实例（GNewsAdapter 或其他 NewsSource）
        source_name: 数据源名称
        query: 搜索关键词
        category: 分类
        limit: 限制条数
        from_: 开始时间
        to: 结束时间
        nullable: 可空字段
        truncate: 截断字段
        sortby: 排序方式
        in_fields: 搜索字段
        page: 页码

    Returns:
        新闻数据列表
    """
    from ...ports.extraction import FetchConfig
    
    logger.info(f"开始从数据源 {source_name} 获取数据")
    
    try:
        # 构建 FetchConfig
        keywords = query.split() if query else None
        # 将字符串格式的日期转换为 datetime 对象
        from_date = None
        to_date = None
        if from_:
            try:
                from_date = datetime.fromisoformat(from_.replace("Z", "+00:00"))
            except Exception:
                pass
        if to:
            try:
                to_date = datetime.fromisoformat(to.replace("Z", "+00:00"))
            except Exception:
                pass
                
        config = FetchConfig(
            max_items=limit,
            keywords=keywords,
            category=category,
            language=None,  # 使用 collector 默认语言
            from_date=from_date,
            to_date=to_date,
            extra=extra or {},
        )
        logger.info(f"构建 FetchConfig 完成: {config}")
        
        # 调用 collector.fetch() 方法
        logger.info(f"调用 {source_name}.fetch() 方法")
        result = await collector.fetch(config)
        logger.info(f"{source_name}.fetch() 返回结果: success={result.success}, items={len(result.items)}, error={result.error}")
        
        if not result.success:
            logger.error(f"从 {source_name} 获取数据失败: {result.error}")
            return []
        
        # 将 NewsItem 转换为字典格式
        news = []
        for item in result.items:
            news.append({
                "id": item.id,
                "title": item.title,
                "content": item.content,
                "source": item.source_name,
                "url": item.source_url,
                "datetime": item.published_at.isoformat() if item.published_at else None,
                "author": item.author,
                "category": item.category,
                "language": item.language,
            })
        
        # 标准化数据
        normalized_news = normalize_news_items(news, source_name)
        logger.debug(f"成功从 {source_name} 获取 {len(normalized_news)} 条新闻")
        return normalized_news

    except Exception as e:
        logger.error(f"从 {source_name} 获取数据失败: {e}")
        return []


async def fetch_from_multiple_sources(
    api_pool: Any,
    source_names: List[str],
    concurrency_limit: int,
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    nullable: Optional[str] = None,
    truncate: Optional[str] = None,
    sortby: Optional[str] = None,
    in_fields: Optional[str] = None,
    page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    从多个数据源并发获取新闻数据

    Args:
        api_pool: API池实例
        source_names: 数据源名称列表
        concurrency_limit: 并发限制
        其他参数同 fetch_from_collector

    Returns:
        合并后的新闻数据列表
    """
    from ...infra.async_utils import AsyncExecutor

    logger.info(f"开始从多个数据源获取数据: {source_names}, 并发限制: {concurrency_limit}")
    
    async_executor = AsyncExecutor()

    async def fetch_one(source_name: str) -> List[Dict[str, Any]]:
        logger.info(f"开始处理数据源: {source_name}")
        try:
            collector = api_pool.get_collector(source_name)
            logger.info(f"获取到 {source_name} 的 collector")
            result = await fetch_from_collector(
                collector=collector,
                source_name=source_name,
                query=query,
                category=category,
                limit=limit,
                from_=from_,
                to=to,
                extra=extra,
                nullable=nullable,
                truncate=truncate,
                sortby=sortby,
                in_fields=in_fields,
                page=page,
            )
            logger.info(f"数据源 {source_name} 处理完成，获取到 {len(result)} 条数据")
            return result
        except Exception as e:
            logger.error(f"获取数据源 {source_name} 失败: {e}", exc_info=True)
            return []

    # 并发执行
    logger.info(f"准备并发执行 {len(source_names)} 个任务")
    tasks = [lambda src=src: fetch_one(src) for src in source_names]
    logger.info(f"任务列表创建完成，共 {len(tasks)} 个任务")
    results = await async_executor.run_concurrent_tasks(
        tasks=tasks,
        concurrency=concurrency_limit
    )
    logger.info(f"并发执行完成，共获取到 {len(results)} 个结果")

    # 合并结果
    all_news = []
    for i, news in enumerate(results):
        logger.info(f"处理第 {i+1} 个结果，包含 {len(news)} 条数据")
        all_news.extend(news)

    logger.info(f"合并完成，总共获取到 {len(all_news)} 条新闻")
    
    # 按时间排序
    all_news.sort(key=lambda x: x.get("datetime") or "", reverse=True)

    logger.info(f"排序完成，返回 {len(all_news)} 条新闻")
    return all_news
