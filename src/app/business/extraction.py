from typing import List, Dict, Any, Optional, Set, Tuple
from ...infra.registry import register_tool
from ...core import ConfigManager, RateLimiter, LLMAPIPool, AsyncExecutor, tools, get_config_manager, get_llm_pool
from ...domain.data_operations import update_entities, update_abstract_map
from ...infra.serialization import extract_json_from_llm_response
from ...infra.async_utils import call_llm_with_retry, create_extraction_prompt
from ...infra.file_utils import ensure_dirs, safe_unlink, generate_timestamp
from ...domain.data_operations import write_jsonl_file, sanitize_datetime_fields, create_temp_file_path
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


def llm_extract_events(
    title: str,
    content: str,
    api_pool: LLMAPIPool,
    max_retries: int = 2,
    reported_at: Optional[str] = None
) -> List[Dict]:
    """使用LLM提取事件，支持依赖注入"""
    tools.log(f"[LLM请求] 开始处理新闻: {title[:100]}...")
    if api_pool is None:
        tools.log("[LLM请求] ❌ API 池未初始化")
        return []

    # 使用工具函数创建提示
    tools.log("[LLM请求] 构建提示词")
    entity_definitions = """
必要条件：
- 是构成该事件必要的实体，若该实体在事件中缺失则可能导致事件不完备的实体

✅ 必须满足以下任一条件：
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
- 情绪/行情描述（如 "暴涨"、"抛售潮"、"经济衰退担忧"）

额外约束：
- 实体必须“原子化”：不要把多个实体粘连成一个实体名称（例如“美国总统特朗普”必须拆分为“美国总统”和“特朗普”，不要输出粘连形式）
- 同一主体多种表述时：entities 用最规范且不歧义的主名称，entities_original 保留对应原文表述并逐一对齐"""

    prompt = create_extraction_prompt(title, content, entity_definitions, reported_at=reported_at)
    tools.log(f"[LLM请求] 提示词长度: {len(prompt)} 字符")

    # 使用统一的LLM调用函数
    tools.log("[LLM请求] 调用LLM API")
    raw_content = call_llm_with_retry(
        llm_pool=api_pool,
        prompt=prompt,
        max_tokens=1500,
        timeout=55,
        retries=max_retries
    )
    tools.log(f"[LLM请求] LLM返回内容长度: {len(raw_content) if raw_content else 0} 字符")

    if not raw_content:
        tools.log("[LLM请求] LLM返回空内容")
        return []

    # 使用统一的JSON解析函数
    try:
        tools.log("[LLM请求] 解析LLM响应")
        data = extract_json_from_llm_response(raw_content)
        events = data.get("events", [])
        tools.log(f"[LLM请求] 解析到 {len(events)} 个事件")
        result = []

        for item in events:
            abstract = item.get("abstract", "").strip()
            # ----------------------------
            # 1) entities / entities_original：对齐 & 容错
            # ----------------------------
            entities_raw = item.get("entities", []) or []
            entities_original_raw = item.get("entities_original", []) or []
            if isinstance(entities_raw, str):
                entities_raw = [entities_raw]
            if isinstance(entities_original_raw, str):
                entities_original_raw = [entities_original_raw]

            entities: List[str] = []
            entities_original: List[str] = []
            for i, ent in enumerate(entities_raw if isinstance(entities_raw, list) else []):
                if not isinstance(ent, str):
                    continue
                ent = ent.strip()
                if not ent or not tools.is_valid_entity(ent):
                    continue

                ent_original = ""
                if isinstance(entities_original_raw, list) and i < len(entities_original_raw):
                    ent_original = entities_original_raw[i]
                if not isinstance(ent_original, str):
                    ent_original = ""
                ent_original = ent_original.strip()

                # 原始表述缺失时回退到实体名（避免因 zip 截断/缺失导致实体整体被丢弃）
                if not ent_original or not tools.is_valid_entity(ent_original):
                    ent_original = ent

                entities.append(ent)
                entities_original.append(ent_original)

            # ----------------------------
            # 2) entity_roles：实体语义角色（key 必须来自 entities）
            # ----------------------------
            roles_raw = item.get("entity_roles", {}) or {}
            entity_roles: Dict[str, List[str]] = {}
            if isinstance(roles_raw, dict):
                allowed = set(entities)
                for k, v in roles_raw.items():
                    if not isinstance(k, str):
                        continue
                    ek = k.strip()
                    if ek not in allowed:
                        continue
                    roles_list: List[str] = []
                    if isinstance(v, str):
                        roles_list = [v]
                    elif isinstance(v, list):
                        roles_list = [r for r in v if isinstance(r, str)]
                    cleaned_roles = []
                    seen = set()
                    for r in roles_list:
                        rr = r.strip()
                        if not rr:
                            continue
                        if rr not in seen:
                            seen.add(rr)
                            cleaned_roles.append(rr)
                    if cleaned_roles:
                        entity_roles[ek] = cleaned_roles

            # ----------------------------
            # 3) event_types：事件类型标签
            # ----------------------------
            types_raw = item.get("event_types", []) or []
            event_types: List[str] = []
            if isinstance(types_raw, str):
                types_raw = [types_raw]
            if isinstance(types_raw, list):
                seen_t = set()
                for t in types_raw:
                    if not isinstance(t, str):
                        continue
                    tt = t.strip()
                    if not tt:
                        continue
                    if tt not in seen_t:
                        seen_t.add(tt)
                        event_types.append(tt)

            # ----------------------------
            # 4) event_start_time：事件起始时间（与 reported_at 区分）
            # ----------------------------
            event_start_time = item.get("event_start_time", "")
            event_start_time_text = item.get("event_start_time_text", "")
            event_start_time_precision = item.get("event_start_time_precision", "unknown")
            if not isinstance(event_start_time, str):
                event_start_time = ""
            if not isinstance(event_start_time_text, str):
                event_start_time_text = ""
            if not isinstance(event_start_time_precision, str):
                event_start_time_precision = "unknown"
            event_start_time = event_start_time.strip()
            event_start_time_text = event_start_time_text.strip()
            event_start_time_precision = event_start_time_precision.strip() or "unknown"

            # ----------------------------
            # 5) relations：(实体, 关系, 实体) 三元组
            # ----------------------------
            relations_raw = item.get("relations", []) or []
            relations: List[Dict[str, str]] = []
            allowed_entities = set(entities)
            seen_rel = set()

            def _add_relation(s: str, p: str, o: str, ev: str = "", relation_kind: Any = ""):
                ss = s.strip() if isinstance(s, str) else ""
                pp = p.strip() if isinstance(p, str) else ""
                oo = o.strip() if isinstance(o, str) else ""
                ee = ev.strip() if isinstance(ev, str) else ""
                if not ss or not pp or not oo:
                    return
                if ss not in allowed_entities or oo not in allowed_entities:
                    return
                if ss == oo:
                    return
                rk_raw = relation_kind if isinstance(relation_kind, str) else str(relation_kind or "")
                rk = rk_raw.strip().lower()
                if rk not in {"state", "event"}:
                    rk = ""
                key = (ss, pp, oo, rk)
                if key in seen_rel:
                    return
                seen_rel.add(key)
                relations.append({
                    "subject": ss,
                    "predicate": pp,
                    "object": oo,
                    "relation_kind": rk,
                    "evidence": ee
                })

            if isinstance(relations_raw, dict):
                relations_raw = [relations_raw]
            if isinstance(relations_raw, list):
                for rel in relations_raw:
                    # 支持 dict 形式：{"subject","predicate","object","evidence"}
                    if isinstance(rel, dict):
                        _add_relation(
                            rel.get("subject", ""),
                            rel.get("predicate", ""),
                            rel.get("object", ""),
                            rel.get("evidence", "") or rel.get("text", ""),
                            rel.get("relation_kind", "") or rel.get("kind", "") or rel.get("type", ""),
                        )
                        continue
                    # 兼容 tuple/list 形式：[s,p,o] 或 [s,p,o,evidence]
                    if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                        s, p, o = rel[0], rel[1], rel[2]
                        ev = rel[3] if len(rel) >= 4 else ""
                        _add_relation(s, p, o, ev, "")

            summary = item.get("event_summary", "").strip()
            if abstract and entities and summary:
                result.append({
                    "abstract": abstract,
                    "entities": entities,
                    "entities_original": entities_original,
                    "entity_roles": entity_roles,
                    "event_types": event_types,
                    "event_start_time": event_start_time,
                    "event_start_time_text": event_start_time_text,
                    "event_start_time_precision": event_start_time_precision,
                    "relations": relations,
                    "event_summary": summary
                })
        tools.log(f"[LLM请求] 提取完成，共 {len(result)} 个有效事件")
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
    api_pool = get_llm_pool()
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
    # 从数据库获取已处理的ID
    from src.adapters.sqlite.store import get_store
    processed_ids = get_store().get_processed_ids()

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

    # 从数据库获取已处理的ID
    from src.adapters.sqlite.store import get_store
    processed_ids = get_store().get_processed_ids()

    limiter = RateLimiter(rate_limit_per_sec)
    async_executor = AsyncExecutor()
    logger = tools.get_logger(__name__)
    api_pool = get_llm_pool()
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
            extracted = await loop.run_in_executor(
                None,
                lambda: llm_extract_events(title, content, api_pool, reported_at=published_at)
            )
            return global_id, source, published_at, extracted
        except Exception as e:
            logger.error(f"任务 {global_id} 提取失败: {e}")
            return global_id, source, published_at, []

    # 收集所有需要记录的已处理ID，稍后批量插入数据库
    processed_ids_to_add = []
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
                            # 收集需要记录的ID
                            processed_ids_to_add.append((global_id, source, raw_id))
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

    # 批量插入已处理的ID到数据库
    if processed_ids_to_add:
        try:
            from src.adapters.sqlite.store import get_store
            count = get_store().add_processed_ids(processed_ids_to_add)
            tools.log(f"✅ 批量记录 {count} 个已处理ID到数据库")
        except Exception as e:
            tools.log(f"⚠️ 记录已处理ID到数据库失败: {e}")
    
    tools.log(f"✅ 完成！共处理 {total_processed} 条含有效实体的新闻")
    # SQLite 为主存储：批量处理结束后统一导出兼容 JSON（避免每条新闻都写一次大文件）
    if total_processed > 0:
        try:
            from src.adapters.sqlite.store import get_store
            get_store().export_compat_json_files()
        except Exception as e:
            tools.log(f"⚠️ 导出兼容JSON失败（不影响主存储SQLite）: {e}")
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
    from ...infra.paths import tools as Tools
    tools = Tools()
    tools.log(f"[batch_process_news] 开始处理，新闻数量: {len(news_list)}, limit: {limit}")
    # 1. 去重
    unique_news = deduplicate_news_batch(news_list)
    tools.log(f"[batch_process_news] 去重后新闻数量: {len(unique_news)}")
    if limit > 0:
        unique_news = unique_news[:limit]
        tools.log(f"[batch_process_news] 应用limit后新闻数量: {len(unique_news)}")

    # 读取并发/限速配置（使用统一配置管理器）
    config_manager = get_config_manager()
    max_workers = config_manager.get_concurrency_limit("agent1_config")
    rate_limit = config_manager.get_rate_limit("agent1_config")

    # 使用统一限速器
    limiter = RateLimiter(rate_limit)

    # 复用一个 API pool，避免每条新闻都初始化 LLMAPIPool 触发“迁移/加载服务”刷屏
    api_pool = get_llm_pool()

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
            extracted = llm_extract_events(title, content, api_pool, reported_at=timestamp)
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
        tools.log("[batch_process_news] 没有新闻需要处理")
        return all_events
    else:
        tools.log(f"[batch_process_news] 准备处理 {len(unique_news)} 条唯一新闻")

    if max_workers <= 1:
        tools.log(f"[batch_process_news] 开始串行处理 {len(unique_news)} 条新闻")
        for i, n in enumerate(unique_news):
            tools.log(f"[batch_process_news] 处理第 {i+1} 条新闻: {n.get('title', '')[:50]}...")
            evs, pid = process_one(n)
            tools.log(f"[batch_process_news] 第 {i+1} 条新闻提取到 {len(evs)} 个事件")
            all_events.extend(evs)
            if pid:
                processed_ids.append(pid)
    else:
        # 使用AsyncExecutor统一管理线程并发
        tools.log(f"[batch_process_news] 开始并发处理 {len(unique_news)} 条新闻，最大并发数: {max_workers}")
        async_executor = AsyncExecutor()
        task_results = async_executor.run_threaded_tasks(
            tasks=unique_news,
            func=process_one,
            max_workers=max_workers
        )
        tools.log(f"[batch_process_news] 并发处理完成，获取到 {len(task_results)} 个结果")

        for evs, pid in task_results:
            all_events.extend(evs or [])
            if pid:
                processed_ids.append(pid)

    # 写入SQLite数据库（实体和事件）
    if all_events:
        tools.log(f"[batch_process_news] 开始写入 {len(all_events)} 个事件到SQLite")
        try:
            # 按来源分组事件，批量写入
            events_by_source = {}
            for ev in all_events:
                source = ev.get("source", "unknown")
                if source not in events_by_source:
                    events_by_source[source] = []
                events_by_source[source].append(ev)
            
            # 逐个来源写入
            for source, events in events_by_source.items():
                # 收集所有实体
                all_entities = []
                all_entities_original = []
                for ev in events:
                    all_entities.extend(ev.get("entities", []))
                    all_entities_original.extend(ev.get("entities_original", []))
                
                # 写入实体
                if all_entities and len(all_entities) == len(all_entities_original):
                    published_at = events[0].get("published_at") if events else None
                    update_entities(all_entities, all_entities_original, source, published_at)
                    tools.log(f"[batch_process_news] 已写入 {len(all_entities)} 个实体 (来源: {source})")
                
                # 写入事件
                update_abstract_map(events, source, events[0].get("published_at") if events else None)
                tools.log(f"[batch_process_news] 已写入 {len(events)} 个事件 (来源: {source})")
        except Exception as e:
            tools.log(f"[batch_process_news] 写入SQLite失败: {e}")

    # 记录 processed_ids 到数据库，避免重复处理
    if processed_ids:
        tools.log(f"[batch_process_news] 记录 {len(processed_ids)} 个已处理的ID")
        try:
            # 从 processed_ids 中提取 source 和 news_id
            ids_to_add = []
            for pid in processed_ids:
                parts = pid.split(':', 1)
                if len(parts) == 2:
                    source, news_id = parts
                    ids_to_add.append((pid, source, news_id))
            
            from src.adapters.sqlite.store import get_store
            count = get_store().add_processed_ids(ids_to_add)
            tools.log(f"[batch_process_news] 成功记录 {count} 个已处理ID到数据库")
        except Exception as e:
            tools.log(f"[batch_process_news] 记录已处理ID到数据库时出错: {e}")
    else:
        tools.log("[batch_process_news] 没有需要记录的已处理ID")

    # 导出SQLite数据到JSON文件（兼容旧逻辑）
    if all_events:
        try:
            from src.adapters.sqlite.store import get_store
            get_store().export_compat_json_files()
            tools.log(f"[batch_process_news] ✅ 已导出SQLite数据到JSON文件")
        except Exception as e:
            tools.log(f"[batch_process_news] ⚠️ 导出JSON文件失败（不影响SQLite主存储）: {e}")

    tools.log(f"[batch_process_news] 处理完成，总共提取到 {len(all_events)} 个事件")
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
    # 从数据库获取已处理的ID
    from src.adapters.sqlite.store import get_store
    processed_ids = get_store().get_processed_ids()

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
