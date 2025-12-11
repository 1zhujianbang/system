from typing import List, Dict, Any, Optional, Set, Tuple
from ..core.registry import register_tool
from ..core import ConfigManager, RateLimiter, LLMAPIPool, AsyncExecutor, tools
from ..utils.data_utils import update_entities, update_abstract_map
from ..utils.json_utils import extract_json_from_llm_response
from ..utils.llm_utils import call_llm_with_retry, create_extraction_prompt
from ..utils.file_utils import ensure_dirs, safe_unlink, generate_timestamp
from ..utils.data_utils import write_jsonl_file, sanitize_datetime_fields, create_temp_file_path
import json
from pathlib import Path
import time
import threading
import asyncio
import datetime
from datetime import datetime as dt, timezone, timedelta
from collections import defaultdict

class NewsDeduplicator:
    """新闻去重器，支持依赖注入"""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.seen_hashes: Set[int] = set()

    @staticmethod
    def _news_key(news: Dict) -> str:
        """构造用于去重的唯一键，包含 source 前缀，兼容多数据源。"""
        return f"{news.get('source', 'unknown')}:{news.get('id')}"

    def is_duplicate(self, text: str) -> bool:
        h = tools.simhash(text)
        for seen_h in self.seen_hashes:
            if tools.hamming_distance(h, seen_h) <= self.threshold:
                return True
        self.seen_hashes.add(h)
        return False

    def dedupe_file(self, input_path: Path, output_path: Path, processed_ids: Optional[Set[str]] = None):
        """
        对单个原始文件做去重：
        - 先用 processed_ids（全局已处理 ID，如 blockbeats:323066）过滤历史已处理新闻
        - 再结合已有去重文件 & simhash 去掉本批内/跨批的重复内容
        """
        tools.log(f"🔍 去重中: {input_path.name}")

        # 先加载"全局已处理 ID"，避免老新闻再次进入去重结果
        seen_ids: Set[str] = set(processed_ids or set())
        if processed_ids:
            tools.log(f"🔍 已有历史 processed_ids 数量: {len(processed_ids)}")

        # 再加载已有去重结果文件中的 ID，实现跨批次的本地去重
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        seen_ids.add(self._news_key(item))
                    except Exception as e:
                        tools.log(f"⚠️ 读取历史去重文件时跳过无效行: {e}")

        kept, skipped_id, skipped_sim = 0, 0, 0
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "a", encoding="utf-8") as fout:
            for line in fin:
                try:
                    news = json.loads(line)
                    key = self._news_key(news)

                    # 1) 按全局 ID 去重（包括 processed_ids 和已有去重文件中的 ID）
                    if key in seen_ids:
                        skipped_id += 1
                        continue

                    # 2) 构造文本，按内容相似度去重
                    raw_text = (news.get("title", "") + " " + news.get("content", "")).strip()
                    if not raw_text:
                        continue
                    if self.is_duplicate(raw_text):
                        skipped_sim += 1
                        continue
                    fout.write(line)
                    seen_ids.add(key)
                    kept += 1
                except Exception as e:
                    tools.log(f"⚠️ 跳过无效行: {e}")
        tools.log(f"✅ 去重完成: 保留 {kept} 条, 按 ID 跳过 {skipped_id} 条, 按相似度跳过 {skipped_sim} 条")


def llm_extract_events(title: str, content: str, api_pool: LLMAPIPool, max_retries=2) -> List[Dict]:
    """使用LLM提取事件，支持依赖注入"""
    if api_pool is None:
        tools.log("[LLM请求] ❌ API 池未初始化")
        return []

    # 使用工具函数创建提示
    entity_definitions = """✅ 必须满足以下任一条件：
- 自然人（如 Elon Musk、Cathie Wood、Warren Buffett）
- 注册公司（如 Apple Inc.、Goldman Sachs、中国工商银行、Volkswagen AG）
- 政府机构或部门（如 美国证券交易委员会、中国人民银行、欧盟委员会、日本金融厅）
- 主权国家或明确行政区（如 美国、新加坡、加利福尼亚州、香港特别行政区、德意志联邦共和国）
- 国际组织（如 国际货币基金组织、世界银行、联合国、金融稳定理事会）
- 重要产品/品牌/型号（由明确主体生产/提供的具体产品或品牌，如 iPhone 15 Pro、Tesla Model 3、ChatGPT、Windows 11、Redmi 12C）

❌ 以下内容**不得**视为实体：
- 只作为新闻通讯源的实体不应该作为实体
- 抽象概念（如 "市场波动"、"系统性风险"、"资本流动"）
- 技术或金融术语（如 "期权定价"、"资产负债表"、"量化宽松"）
- 金融工具或资产名称（如 "标普500指数"、"10年期美债"、"黄金期货"、"BTC"）——除非指代其发行方、管理方或关联法人（如 "标普道琼斯指数公司"）
- 泛称（如 "投资者"、"监管机构"、"某银行"、"大型科技公司"、"智能手机"）
- 情绪/行情描述（如 "暴涨"、"抛售潮"、"经济衰退担忧"）"""

    prompt = create_extraction_prompt(title, content, entity_definitions)

    # 使用统一的LLM调用函数
    raw_content = call_llm_with_retry(
        llm_pool=api_pool,
        prompt=prompt,
        max_tokens=1500,
        timeout=55,
        retries=max_retries
    )

    if not raw_content:
        return []

    # 使用统一的JSON解析函数
    try:
        data = extract_json_from_llm_response(raw_content)
        events = data.get("events", [])
        result = []
        for item in events:
            abstract = item.get("abstract", "").strip()
            # 确保entities和entities_original一一对应，且都有效
            entities_raw = item.get("entities", [])
            entities_original_raw = item.get("entities_original", [])
            entities = []
            entities_original = []

            # 遍历并过滤，确保索引对应
            for ent, ent_original in zip(entities_raw, entities_original_raw):
                if tools.is_valid_entity(ent) and tools.is_valid_entity(ent_original):
                    entities.append(ent)
                    entities_original.append(ent_original)
            summary = item.get("event_summary", "").strip()
            if abstract and entities and summary:
                result.append({
                    "abstract": abstract,
                    "entities": entities,
                    "entities_original": entities_original,
                    "event_summary": summary
                })
        return result
    except Exception as e:
        tools.log(f"[LLM获取] ❌ LLM 返回内容解析失败: {e}")
        return []


@register_tool(
    name="extract_entities_events",
    description="使用 LLM 从新闻标题和内容中提取实体和事件",
    category="Information Extraction"
)
def extract_entities_events(title: str, content: str) -> List[Dict[str, Any]]:
    """
    从新闻中提取实体和事件

    Args:
        title: 新闻标题
        content: 新闻内容

    Returns:
        事件列表，每项包含 entities, event_summary 等
    """
    api_pool = LLMAPIPool()
    return llm_extract_events(title, content, api_pool)

@register_tool(
    name="deduplicate_news_batch",
    description="对新闻列表进行批量去重 (基于 SimHash)",
    category="Data Processing"
)
def deduplicate_news_batch(news_list: List[Dict[str, Any]], threshold: int = 3) -> List[Dict[str, Any]]:
    """
    批量去重
    
    Args:
        news_list: 新闻字典列表
        threshold: SimHash 汉明距离阈值
        
    Returns:
        去重后的新闻列表
    """
    if not news_list:
        return []
        
    deduper = NewsDeduplicator(threshold=threshold)
    unique_news = []
    
    for news in news_list:
        # 构造指纹文本
        text = (news.get("title", "") + " " + news.get("content", "")).strip()
        if not text: 
            continue
            
        # 检查重复
        if not deduper.is_duplicate(text):
            unique_news.append(news)
            
    return unique_news

def get_unprocessed_news_files() -> List[Path]:
    """
    仅使用 tmp 目录的去重与处理。
    tmp 用于新抓取的待处理数据，处理完成后会删除对应 raw/deduped。
    """
    processed_ids = set()
    if tools.PROCESSED_IDS_FILE.exists():
        with open(tools.PROCESSED_IDS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            processed_ids = set(line.strip() for line in f if line.strip())

    unprocessed: List[Path] = []
    raw_dir = tools.RAW_NEWS_TMP_DIR
    dedup_dir = tools.DEDUPED_NEWS_TMP_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in sorted(raw_dir.glob("*.jsonl")):
        deduped_file = dedup_dir / f"{raw_file.stem}_deduped.jsonl"
        if not deduped_file.exists():
            deduper = NewsDeduplicator(threshold=tools.get_dedupe_threshold())
            deduper.dedupe_file(raw_file, deduped_file, processed_ids)
        unprocessed.append(deduped_file)
    return unprocessed


@register_tool(
    name="process_news_pipeline",
    description="[工作流] 处理新闻管道：从tmp文件读取、提取实体事件、更新图谱",
    category="Workflow"
)
async def process_news_pipeline(max_workers: int = 3, rate_limit_per_sec: float = 1.0) -> Dict[str, Any]:
    """
    主处理流程：并发实体提取

    Args:
        max_workers: 最大并发数
        rate_limit_per_sec: 每秒速率限制

    Returns:
        处理统计信息
    """
    tools.log(f"🚀 启动新闻处理管道 | workers={max_workers}, rate={rate_limit_per_sec}/s")
    files = get_unprocessed_news_files()
    if not files:
        tools.log("📭 无可处理新闻文件")
        return {"processed_count": 0, "files_processed": 0}

    processed_ids = set()
    if tools.PROCESSED_IDS_FILE.exists():
        with open(tools.PROCESSED_IDS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            processed_ids = set(line.strip() for line in f if line.strip())

    limiter = RateLimiter(rate_limit_per_sec)
    async_executor = AsyncExecutor()
    logger = tools.get_logger(__name__)
    api_pool = LLMAPIPool()
    total_processed = 0

    def build_published_at(ts: Optional[str]) -> Optional[str]:
        if not ts:
            return None
        try:
            return ts if isinstance(ts, str) else str(ts)
        except Exception:
            return None

    async def extract_task_async(global_id: str, title: str, content: str, source: str, published_at: Optional[str]) -> Tuple[str, str, Optional[str], List[Dict]]:
        try:
            await limiter.acquire_async()
            loop = asyncio.get_running_loop()
            extracted = await loop.run_in_executor(None, llm_extract_events, title, content, api_pool)
            return global_id, source, published_at, extracted
        except Exception as e:
            logger.error(f"任务 {global_id} 提取失败: {e}")
            return global_id, source, published_at, []

    with open(tools.PROCESSED_IDS_FILE, "a") as id_log:
        for file_path in files:
            logger.info(f"📄 处理文件: {file_path.name}")

            # 收集需要处理的新闻任务
            news_tasks = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        news = json.loads(line)
                        raw_id = str(news.get("id", "")).strip()
                        source = news.get("source", "unknown").strip().lower()

                        if not raw_id or not source:
                            logger.warning("⚠️ 跳过无 ID 或无 source 的新闻")
                            continue

                        global_id = f"{source}:{raw_id}"
                        if global_id in processed_ids:
                            continue

                        title = news.get("title", "")
                        content = news.get("content", "")
                        MAX_CONTENT_CHARS = 2000
                        if isinstance(content, str) and len(content) > MAX_CONTENT_CHARS:
                            content = content[:MAX_CONTENT_CHARS] + "……【后文已截断】"

                        published_at = build_published_at(news.get("timestamp"))

                        # 创建异步任务
                        news_tasks.append(
                            lambda gid=global_id, t=title, c=content, s=source, p=published_at: extract_task_async(gid, t, c, s, p)
                        )
                    except Exception as e:
                        logger.error(f"⚠️ 解析新闻行失败: {e}")

            if news_tasks:
                logger.info(f"🔄 开始并发处理 {len(news_tasks)} 个新闻提取任务")
                # 使用AsyncExecutor统一管理并发执行
                results = await async_executor.run_concurrent_tasks(
                    tasks=news_tasks,
                    concurrency=max_workers
                )

                for result in results:
                    try:
                        global_id, source, published_at, extracted = result
                        if not extracted:
                            logger.debug(f"⏳ 新闻 {global_id}：LLM 未返回有效事件，保留重试机会")
                            continue

                        all_entities = []
                        all_entities_original = []
                        for ev in extracted:
                            all_entities.extend(ev["entities"])
                            all_entities_original.extend(ev["entities_original"])

                        if all_entities and len(all_entities) == len(all_entities_original):
                            update_entities(all_entities, all_entities_original, source, published_at)
                            update_abstract_map(extracted, source, published_at)
                            total_processed += 1
                            id_log.write(global_id + "\n")
                            processed_ids.add(global_id)
                        else:
                            logger.debug(f"🔍 新闻 {global_id}：LLM 返回事件但无有效实体，暂不标记")
                    except Exception as e:
                        logger.error(f"⚠️ 处理提取结果失败: {e}")

            try:
                # 处理 tmp 目录下的 raw/deduped 文件
                raw_dir = tools.RAW_NEWS_TMP_DIR
                raw_file_name = file_path.stem.replace("_deduped", "") + ".jsonl"
                raw_file_path = raw_dir / raw_file_name

                # 使用统一的删除函数
                safe_unlink(raw_file_path, "原始新闻")
                safe_unlink(file_path, "去重新闻")
            except Exception as e:
                tools.log(f"⚠️ 删除文件失败: {e}")

    tools.log(f"✅ 完成！共处理 {total_processed} 条含有效实体的新闻")
    return {"processed_count": total_processed, "files_processed": len(files)}


@register_tool(
    name="batch_process_news",
    description="[工作流] 批量处理新闻：去重并提取事件",
    category="Workflow"
)
async def batch_process_news(news_list: List[Dict[str, Any]], limit: int = -1) -> List[Dict[str, Any]]:
    """
    批量处理新闻：
    1. 去重
    2. 提取实体和事件
    3. 附加元数据 (source, published_at)

    Args:
        news_list: 新闻列表
        limit: 限制处理的新闻数量，-1 表示不限制。用于测试/节省 Token。

    Returns:
        扁平化的事件列表
    """
    # 1. 去重
    unique_news = deduplicate_news_batch(news_list)
    if limit > 0:
        unique_news = unique_news[:limit]

    # 读取并发/限速配置（使用统一配置管理器）
    config_manager = ConfigManager()
    max_workers = config_manager.get_concurrency_limit("agent1_config")
    rate_limit = config_manager.get_rate_limit("agent1_config")

    # 使用统一限速器
    limiter = RateLimiter(rate_limit)

    def process_one(news: Dict[str, Any]) -> (List[Dict[str, Any]], Optional[str]):
        events_out = []
        processed_id = None
        try:
            title = news.get("title", "")
            content = news.get("content", "")
            source = news.get("source", "unknown")
            timestamp = news.get("datetime") or news.get("formatted_time")
            news_id = str(news.get("id", "")).strip()
            if news_id and source:
                processed_id = f"{source}:{news_id}"
            limiter.acquire()
            api_pool = LLMAPIPool()
            extracted = llm_extract_events(title, content, api_pool)
            for ev in extracted:
                ev["source"] = source
                ev["published_at"] = timestamp
                ev["news_id"] = news.get("id")
                events_out.append(ev)
        except Exception as e:
            print(f"Extraction failed for news {news.get('id', '')}: {e}")
        return events_out, processed_id

    all_events: List[Dict[str, Any]] = []
    processed_ids: List[str] = []
    if not unique_news:
        return all_events

    if max_workers <= 1:
        for n in unique_news:
            evs, pid = process_one(n)
            all_events.extend(evs)
            if pid:
                processed_ids.append(pid)
    else:
        # 使用AsyncExecutor统一管理线程并发
        async_executor = AsyncExecutor()
        task_results = async_executor.run_threaded_tasks(
            tasks=unique_news,
            func=process_one,
            max_workers=max_workers
        )

        for evs, pid in task_results:
            all_events.extend(evs or [])
            if pid:
                processed_ids.append(pid)

    # 记录 processed_ids，避免重复处理
    if processed_ids:
        try:
            tools.PROCESSED_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(tools.PROCESSED_IDS_FILE, "a", encoding="utf-8") as f:
                for pid in processed_ids:
                    f.write(pid + "\n")
        except Exception:
            pass

    return all_events


@register_tool(
    name="persist_expanded_news_tmp",
    description="将拓展新闻写入 tmp/raw_news & tmp/deduped_news，并返回路径",
    category="Workflow"
)
def persist_expanded_news_tmp(expanded_news: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    为前端调用的包装：落地拓展新闻到 tmp，并返回文件路径。
    """
    processed_ids = set()
    if Path(tools.PROCESSED_IDS_FILE).exists():
        try:
            processed_ids = set(
                line.strip()
                for line in Path(tools.PROCESSED_IDS_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except Exception:
            processed_ids = set()

    deduped_path = persist_expanded_news_to_tmp(expanded_news, processed_ids)
    return {
        "deduped_path": str(deduped_path) if deduped_path else "",
        "raw_path": str(tools.RAW_NEWS_TMP_DIR) if deduped_path else "",
    }


@register_tool(
    name="save_extracted_events_tmp",
    description="将提取的事件列表写入 data/tmp/extracted_events_*.jsonl，并返回路径",
    category="Data Processing"
)
def save_extracted_events_tmp(events: List[Dict[str, Any]]) -> Dict[str, str]:
    if not events:
        return {"path": ""}

    out_path = create_temp_file_path(tools.DATA_TMP_DIR, "extracted_events")
    write_jsonl_file(out_path, events, ensure_ascii=False)
    return {"path": str(out_path)}


def persist_expanded_news_to_tmp(expanded_news: List[Dict], processed_ids: Set[str]) -> Optional[Path]:
    """
    将拓展新闻写入 tmp 原始文件并做去重，返回去重后的文件路径。
    """
    if not expanded_news:
        return None

    # 确保目录存在
    ensure_dirs(tools.RAW_NEWS_TMP_DIR, tools.DEDUPED_NEWS_TMP_DIR)

    # 创建临时文件路径
    ts = generate_timestamp()
    raw_path = tools.RAW_NEWS_TMP_DIR / f"expanded_{ts}.jsonl"
    deduped_path = tools.DEDUPED_NEWS_TMP_DIR / f"expanded_{ts}_deduped.jsonl"

    # 写入原始数据（处理datetime字段）
    sanitized_news = sanitize_datetime_fields(expanded_news)
    write_jsonl_file(raw_path, sanitized_news, ensure_ascii=False)

    # 去重处理
    deduper = NewsDeduplicator(threshold=tools.get_dedupe_threshold())
    deduper.dedupe_file(raw_path, deduped_path, processed_ids)
    return deduped_path
