"""
Application Services 实现。

提供 IngestionService、ReviewService、KnowledgeGraphService 的具体实现。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .services import (
    IngestionService, IngestionConfig, IngestionResult,
    ReviewService, ReviewConfig, ReviewResult,
    KnowledgeGraphService, SnapshotConfig, KGRefreshResult, SnapshotResult,
)
from ..domain import (
    EntityCanonical, EntityMention, EventCanonical, EventMention, EventEdge
)
from ..ports.extraction import EntityExtractor, EventExtractor
from ..ports.store import UnifiedStore
from ..infra import get_logger, IdFactory, utc_now
from ..infra.paths import tools as Tools
from ..domain.models import SourceRef, EventEdgeType


class IngestionServiceImpl(IngestionService):
    """入库服务实现"""

    def __init__(
        self,
        store: UnifiedStore,
        entity_extractor: Optional[EntityExtractor] = None,
        event_extractor: Optional[EventExtractor] = None,
    ):
        self._store = store
        self._entity_extractor = entity_extractor
        self._event_extractor = event_extractor
        self._logger = get_logger(__name__)

    def ingest_news(
        self,
        news_items: List[Dict[str, Any]],
        config: Optional[IngestionConfig] = None,
    ) -> IngestionResult:
        """入库新闻"""
        config = config or IngestionConfig()
        result = IngestionResult(
            success=True,
            run_id=IdFactory.run_id(),
            started_at=utc_now()
        )

        for item in news_items:
            try:
                # 1. 处理新闻项
                source_id = item.get("id", "")
                text = item.get("content", "") or item.get("title", "")

                if not text:
                    continue

                # 2. 抽取实体
                entities = self.extract_entities(text, source_id)
                for entity in entities:
                    # TODO: 实际写入存储
                    result.mentions_created += 1

                # 3. 抽取事件
                events = self.extract_events(text, source_id)
                for event in events:
                    result.mentions_created += 1

                result.sources_processed += 1

            except Exception as e:
                self._logger.error(f"Failed to ingest news item: {e}")

        result.finished_at = utc_now()
        return result

    async def ingest_news_async(
        self,
        news_items: List[Dict[str, Any]],
        config: Optional[IngestionConfig] = None,
    ) -> IngestionResult:
        """异步入库新闻"""
        # 简单实现：调用同步方法
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.ingest_news, news_items, config
        )

    def extract_entities(
        self,
        text: str,
        source_id: str = "",
    ) -> List[EntityMention]:
        """从文本中抽取实体"""
        if not self._entity_extractor:
            return []

        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self._entity_extractor.extract(text)
        )

        if not result.success:
            return []

        mentions = []
        for entity in result.entities:
            mention = EntityMention(
                mention_id=IdFactory.mention_id(entity.name, source_id, utc_now().isoformat()),
                name_text=entity.name,
                reported_at=utc_now(),
                source=None,  # TODO: 创建 SourceRef
                confidence=entity.confidence,
                created_at=utc_now()
            )
            mentions.append(mention)

        return mentions

    def extract_events(
        self,
        text: str,
        source_id: str = "",
    ) -> List[EventMention]:
        """从文本中抽取事件"""
        if not self._event_extractor:
            return []

        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self._event_extractor.extract(text)
        )

        if not result.success:
            return []

        mentions = []
        for event in result.events:
            mention = EventMention(
                mention_id=IdFactory.mention_id(event.abstract, source_id, utc_now().isoformat()),
                abstract_text=event.abstract,
                reported_at=utc_now(),
                source=None,
                event_type=event.event_type,
                confidence=event.confidence,
                created_at=utc_now()
            )
            mentions.append(mention)

        return mentions


class ReviewServiceImpl(ReviewService):
    """审查服务实现"""

    def __init__(self, store: UnifiedStore):
        self._store = store
        self._logger = get_logger(__name__)

    def generate_entity_merge_candidates(
        self,
        min_similarity: float = 0.92,
        max_pairs: int = 200,
    ) -> int:
        """生成实体合并候选"""
        # TODO: 实现候选生成逻辑
        self._logger.info(f"Generating entity merge candidates (similarity >= {min_similarity})")
        return 0

    def generate_event_merge_candidates(
        self,
        shared_entity_min: int = 2,
        days_window: int = 14,
        max_pairs: int = 200,
    ) -> int:
        """生成事件合并候选"""
        self._logger.info(f"Generating event merge candidates (shared >= {shared_entity_min})")
        return 0

    def run_review_worker(
        self,
        task_type: str,
        max_tasks: int = 20,
        rate_limit_per_sec: float = 0.5,
    ) -> Dict[str, int]:
        """运行审查 worker"""
        self._logger.info(f"Running review worker for {task_type}")
        return {"done": 0, "failed": 0}

    def apply_entity_merges(
        self,
        max_actions: int = 50,
    ) -> Dict[str, int]:
        """应用实体合并"""
        return {"applied": 0, "skipped": 0}

    def apply_event_decisions(
        self,
        max_actions: int = 50,
    ) -> Dict[str, int]:
        """应用事件决策"""
        return {"merged": 0, "edges_added": 0, "skipped": 0}

    def run_end_to_end(
        self,
        config: Optional[ReviewConfig] = None,
    ) -> ReviewResult:
        """端到端审查流程"""
        config = config or ReviewConfig()
        result = ReviewResult(success=True, started_at=utc_now())

        # 1. 生成候选
        result.candidates_generated = self.generate_entity_merge_candidates(
            min_similarity=config.min_similarity,
            max_pairs=config.max_pairs
        )

        # 2. 运行审查
        worker_result = self.run_review_worker("entity_merge", config.max_review_tasks)
        result.tasks_reviewed = worker_result.get("done", 0)
        result.tasks_failed = worker_result.get("failed", 0)

        # 3. 应用合并
        apply_result = self.apply_entity_merges(config.max_apply)
        result.merges_applied = apply_result.get("applied", 0)
        result.skipped = apply_result.get("skipped", 0)

        result.finished_at = utc_now()
        return result

    def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计"""
        return {
            "pending_entity_merges": 0,
            "pending_event_decisions": 0,
            "completed_today": 0
        }


class KnowledgeGraphServiceImpl(KnowledgeGraphService):
    """知识图谱服务实现"""

    def __init__(self, store: UnifiedStore):
        self._store = store
        self._logger = get_logger(__name__)
        self._tools = Tools()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._tools.SQLITE_DB_FILE))
        conn.row_factory = sqlite3.Row
        return conn

    def _parse_dt(self, ts: str) -> Optional[datetime]:
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _parse_sources(self, sources_json: str) -> List[SourceRef]:
        try:
            raw = json.loads(sources_json or "[]")
        except Exception:
            raw = []
        out: List[SourceRef] = []
        if isinstance(raw, list):
            for x in raw:
                if isinstance(x, dict):
                    out.append(SourceRef.from_dict(x))
                elif isinstance(x, str) and x.strip():
                    out.append(SourceRef(id="", name=x.strip(), url=""))
        return out

    def refresh(self) -> KGRefreshResult:
        """刷新知识图谱"""
        result = KGRefreshResult(success=True, started_at=utc_now())

        stats = self.get_stats()
        result.entities_count = stats.get("entities", 0)
        result.events_count = stats.get("events", 0)
        result.relations_count = stats.get("relations", 0)
        result.edges_count = stats.get("edges", 0)

        result.finished_at = utc_now()
        return result

    def export_compat_json(self) -> Dict[str, str]:
        """导出兼容 JSON"""
        try:
            if hasattr(self._store, "export_compat_json_files"):
                self._store.export_compat_json_files()
        except Exception as e:
            self._logger.error(f"export_compat_json failed: {e}")
        return {
            "entities": str(self._tools.ENTITIES_FILE),
            "abstract_map": str(self._tools.ABSTRACT_MAP_FILE),
        }

    def generate_snapshots(
        self,
        config: Optional[SnapshotConfig] = None,
    ) -> SnapshotResult:
        """生成快照"""
        config = config or SnapshotConfig()
        result = SnapshotResult(
            success=True,
            started_at=utc_now(),
            graph_types=[]
        )
        try:
            from .snapshot_service import SnapshotService

            svc = SnapshotService(db_path=self._tools.SQLITE_DB_FILE, out_dir=self._tools.SNAPSHOTS_DIR)
            out = svc.generate(
                top_entities=int(config.top_entities),
                top_events=int(config.top_events),
                max_edges=int(config.max_edges),
                days_window=int(config.days_window),
                gap_days=int(config.gap_days),
            )
            if out.get("status") == "ok":
                result.paths = out.get("paths", {}) or {}
                result.graph_types = sorted(list(result.paths.keys()))
            else:
                result.success = False
                result.error = str(out.get("message") or "generate_snapshots failed")
        except Exception as e:
            result.success = False
            result.error = str(e)
        result.finished_at = utc_now()
        return result

    def get_stats(self) -> Dict[str, int]:
        """获取统计"""
        conn = self._connect_db()
        try:
            entities = int(conn.execute("SELECT COUNT(1) FROM entities").fetchone()[0])
            events = int(conn.execute("SELECT COUNT(1) FROM events").fetchone()[0])
            relations = int(conn.execute("SELECT COUNT(1) FROM relations").fetchone()[0])
            edges = int(conn.execute("SELECT COUNT(1) FROM event_edges").fetchone()[0])
            return {"entities": entities, "events": events, "relations": relations, "edges": edges}
        finally:
            conn.close()

    def query_entity(
        self,
        entity_name: str,
    ) -> Optional[EntityCanonical]:
        """查询实体"""
        name = (entity_name or "").strip()
        if not name:
            return None
        conn = self._connect_db()
        try:
            row = conn.execute(
                "SELECT entity_id, name, first_seen, last_seen, sources_json, original_forms_json FROM entities WHERE name=?",
                (name,),
            ).fetchone()
            if row is None:
                return None
            aliases_rows = conn.execute(
                "SELECT alias FROM entity_aliases WHERE entity_id=? ORDER BY created_at ASC",
                (str(row["entity_id"]),),
            ).fetchall()
            aliases = [str(r["alias"]) for r in aliases_rows if str(r["alias"] or "").strip()]
            try:
                forms = json.loads(row["original_forms_json"] or "[]")
                if not isinstance(forms, list):
                    forms = []
            except Exception:
                forms = []
            return EntityCanonical(
                entity_id=str(row["entity_id"]),
                name=str(row["name"] or name),
                first_seen=self._parse_dt(str(row["first_seen"] or "")) or utc_now(),
                last_seen=self._parse_dt(str(row["last_seen"] or "")) or utc_now(),
                sources=self._parse_sources(str(row["sources_json"] or "[]")),
                original_forms=[x for x in forms if isinstance(x, str) and x.strip()],
                aliases=aliases,
            )
        finally:
            conn.close()

    def query_event(
        self,
        abstract: str,
    ) -> Optional[EventCanonical]:
        """查询事件"""
        abs_key = (abstract or "").strip()
        if not abs_key:
            return None
        conn = self._connect_db()
        try:
            row = conn.execute(
                "SELECT * FROM events WHERE abstract=?",
                (abs_key,),
            ).fetchone()
            if row is None:
                alias_row = conn.execute(
                    "SELECT event_id FROM event_aliases WHERE abstract=?",
                    (abs_key,),
                ).fetchone()
                if alias_row is None:
                    return None
                event_id = str(alias_row["event_id"] or "")
                for _ in range(20):
                    red = conn.execute(
                        "SELECT to_event_id FROM event_redirects WHERE from_event_id=?",
                        (event_id,),
                    ).fetchone()
                    if red is None:
                        break
                    nxt = str(red["to_event_id"] or "")
                    if not nxt or nxt == event_id:
                        break
                    event_id = nxt
                row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
                if row is None:
                    return None

            event_id = str(row["event_id"] or "")
            abs_canonical = str(row["abstract"] or abs_key)
            types: List[str] = []
            try:
                raw_types = json.loads(row["event_types_json"] or "[]")
                if isinstance(raw_types, list):
                    types = [x.strip() for x in raw_types if isinstance(x, str) and x.strip()]
            except Exception:
                types = []

            parts = conn.execute(
                """
                SELECT ent.name AS entity_name, p.roles_json AS roles_json
                FROM participants p
                JOIN entities ent ON ent.entity_id = p.entity_id
                WHERE p.event_id=?
                """,
                (event_id,),
            ).fetchall()
            entities: List[str] = []
            entity_roles: Dict[str, List[str]] = {}
            for r in parts:
                en = str(r["entity_name"] or "").strip()
                if not en:
                    continue
                if en not in entities:
                    entities.append(en)
                roles: List[str] = []
                try:
                    rr = json.loads(r["roles_json"] or "[]")
                    if isinstance(rr, list):
                        roles = [x.strip() for x in rr if isinstance(x, str) and x.strip()]
                except Exception:
                    roles = []
                if roles:
                    entity_roles[en] = roles

            return EventCanonical(
                event_id=event_id,
                abstract=abs_canonical,
                event_summary=str(row["event_summary"] or ""),
                event_types=types,
                event_start_time=self._parse_dt(str(row["event_start_time"] or "")),
                event_start_time_text=str(row["event_start_time_text"] or ""),
                event_start_time_precision=str(row["event_start_time_precision"] or "unknown"),
                reported_at=self._parse_dt(str(row["reported_at"] or "")),
                first_seen=self._parse_dt(str(row["first_seen"] or "")),
                last_seen=self._parse_dt(str(row["last_seen"] or "")),
                sources=self._parse_sources(str(row["sources_json"] or "[]")),
                entities=entities,
                entity_roles=entity_roles,
            )
        finally:
            conn.close()

    def query_entity_timeline(
        self,
        entity_name: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """查询实体时间线"""
        from ..adapters.sqlite.kg_read_store import SQLiteKGReadStore

        store = SQLiteKGReadStore(self._tools.SQLITE_DB_FILE)
        rows = store.fetch_entity_timeline(entity_name)
        return (rows or [])[: int(limit) if int(limit) > 0 else 50]

    def query_event_edges(
        self,
        event_id: str,
    ) -> List[EventEdge]:
        """查询事件演化边"""
        eid = (event_id or "").strip()
        if not eid:
            return []
        conn = self._connect_db()
        try:
            rows = conn.execute(
                """
                SELECT id, from_event_id, to_event_id, edge_type, time, reported_at, confidence, evidence_json, decision_input_hash
                FROM event_edges
                WHERE from_event_id=? OR to_event_id=?
                ORDER BY time ASC
                """,
                (eid, eid),
            ).fetchall()
            out: List[EventEdge] = []
            for r in rows:
                et = str(r["edge_type"] or "related").strip().lower() or "related"
                try:
                    edge_type = EventEdgeType(et)
                except Exception:
                    edge_type = EventEdgeType.RELATED
                try:
                    ev = json.loads(r["evidence_json"] or "[]")
                    if not isinstance(ev, list):
                        ev = []
                except Exception:
                    ev = []
                out.append(
                    EventEdge(
                        id=int(r["id"]) if r["id"] is not None else None,
                        from_event_id=str(r["from_event_id"] or ""),
                        to_event_id=str(r["to_event_id"] or ""),
                        edge_type=edge_type,
                        time=self._parse_dt(str(r["time"] or "")) or utc_now(),
                        reported_at=self._parse_dt(str(r["reported_at"] or "")),
                        confidence=float(r["confidence"] or 0.0),
                        evidence=[x for x in ev if isinstance(x, str) and x.strip()],
                        decision_input_hash=str(r["decision_input_hash"] or ""),
                    )
                )
            return out
        finally:
            conn.close()


# =============================================================================
# 服务工厂
# =============================================================================

_services_registry: Dict[str, Any] = {}


def get_ingestion_service() -> IngestionService:
    """获取入库服务"""
    if "ingestion" not in _services_registry:
        from src.adapters.sqlite.store import get_store
        store = get_store()
        _services_registry["ingestion"] = IngestionServiceImpl(store)
    return _services_registry["ingestion"]


def get_review_service() -> ReviewService:
    """获取审查服务"""
    if "review" not in _services_registry:
        from src.adapters.sqlite.store import get_store
        store = get_store()
        _services_registry["review"] = ReviewServiceImpl(store)
    return _services_registry["review"]


def get_kg_service() -> KnowledgeGraphService:
    """获取图谱服务"""
    if "kg" not in _services_registry:
        from src.adapters.sqlite.store import get_store
        store = get_store()
        _services_registry["kg"] = KnowledgeGraphServiceImpl(store)
    return _services_registry["kg"]
