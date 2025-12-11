"""
共享的新闻处理工具函数
提取agent1和agent2中的重复逻辑
"""

import asyncio
from typing import List, Dict, Set, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime, timezone

from ..utils.llm_utils import AsyncExecutor, RateLimiter
from .logging import LoggerManager


async def process_news_batch_async(
    news_list: List[Dict],
    extract_func: Callable,
    api_pool,
    processed_ids: Set[str],
    limiter: Optional[RateLimiter] = None,
    max_workers: int = 6,
    update_entities_func: Optional[Callable] = None,
    update_abstract_func: Optional[Callable] = None
) -> Tuple[int, Set[str]]:
    """
    异步批量处理新闻
    
    Args:
        news_list: 新闻列表
        extract_func: 提取函数（如llm_extract_events）
        api_pool: LLM API池
        processed_ids: 已处理的ID集合
        limiter: 速率限制器
        max_workers: 最大并发数
        update_entities_func: 实体更新函数
        update_abstract_func: 事件映射更新函数
        
    Returns:
        (处理成功的数量, 更新后的已处理ID集合)
    """
    logger = LoggerManager.get_logger(__name__)
    async_executor = AsyncExecutor()
    total_processed = 0
    new_processed_ids = processed_ids.copy()

    async def extract_task_async(
        global_id: str, 
        title: str, 
        content: str, 
        source: str, 
        published_at: Optional[str]
    ) -> Tuple[str, str, Optional[str], List[Dict]]:
        """单个新闻提取任务"""
        try:
            if limiter:
                await limiter.acquire_async()
            
            loop = asyncio.get_running_loop()
            extracted = await loop.run_in_executor(
                None, 
                extract_func, 
                title, 
                content, 
                api_pool
            )
            return global_id, source, published_at, extracted
        except Exception as e:
            logger.error(f"任务 {global_id} 提取失败: {e}")
            return global_id, source, published_at, []

    # 构建任务列表
    tasks = []
    for news in news_list:
        raw_id = str(news.get("id", "")).strip()
        source = news.get("source", "unknown").strip().lower()

        if not raw_id or not source:
            logger.warning("⚠️ 跳过无 ID 或无 source 的新闻")
            continue

        global_id = f"{source}:{raw_id}"
        if global_id in new_processed_ids:
            continue

        title = news.get("title", "")
        content = news.get("content", "")
        MAX_CONTENT_CHARS = 2000
        if isinstance(content, str) and len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS] + "……【后文已截断】"

        published_at = build_published_at(news.get("timestamp") or news.get("datetime"))

        # 创建异步任务
        tasks.append(
            lambda gid=global_id, t=title, c=content, s=source, p=published_at: 
            extract_task_async(gid, t, c, s, p)
        )

    if not tasks:
        return 0, new_processed_ids

    logger.info(f"🔄 开始并发处理 {len(tasks)} 个新闻提取任务")
    
    # 使用AsyncExecutor统一管理并发执行
    results = await async_executor.run_concurrent_tasks(
        tasks=tasks,
        concurrency=max_workers
    )

    for result in results:
        try:
            global_id, source, published_at, extracted = result
            if not extracted:
                logger.debug(f"⏳ 新闻 {global_id}：LLM 未返回有效事件")
                continue

            all_entities = []
            all_entities_original = []
            for ev in extracted:
                all_entities.extend(ev["entities"])
                all_entities_original.extend(ev.get("entities_original", ev["entities"]))

            if all_entities and len(all_entities) == len(all_entities_original):
                if update_entities_func:
                    update_entities_func(all_entities, all_entities_original, source, published_at)
                if update_abstract_func:
                    update_abstract_func(extracted, source, published_at)
                total_processed += 1
                new_processed_ids.add(global_id)
            else:
                logger.debug(f"🔍 新闻 {global_id}：LLM 返回事件但无有效实体")
        except Exception as e:
            logger.error(f"⚠️ 处理提取结果失败: {e}")

    logger.info(f"✅ 完成！共处理 {total_processed} 条含有效实体的新闻")
    return total_processed, new_processed_ids


def build_published_at(ts: Optional[str]) -> Optional[str]:
    """构建标准化的发布时间字符串"""
    if not ts:
        return None
    try:
        if isinstance(ts, datetime):
            return ts.isoformat()
        return ts if isinstance(ts, str) else str(ts)
    except Exception:
        return None


def load_processed_ids(file_path: Path) -> Set[str]:
    """加载已处理的ID集合"""
    processed_ids = set()
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            processed_ids = set(line.strip() for line in f if line.strip())
    return processed_ids


def save_processed_id(file_path: Path, global_id: str):
    """保存已处理的ID"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(global_id + "\n")

