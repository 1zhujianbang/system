"""
Migration tools: 数据迁移与回填
"""
from __future__ import annotations

import json
import hashlib
import sqlite3
from typing import Any, Dict, List
from datetime import datetime, timezone

from ...infra.registry import register_tool
from ...utils.tool_function import tools as Tools
from ...adapters.sqlite.store import get_store


_tools = Tools()


@register_tool(
    name="migrate_json_to_sqlite",
    description="一次性迁移：从 data/entities.json + data/abstract_to_event_map.json 导入 SQLite 主存储，并重新导出兼容 JSON",
    category="Storage",
)
def migrate_json_to_sqlite() -> Dict[str, Any]:
    """
    迁移原则：
    - SQLite 为主存储
    - 每条 participant/relation 强制带 time（缺失则从 event_start_time/reported_at/first_seen 推导）
    - 导入完成后，使用 SQLite 重新导出 entities.json / abstract_to_event_map.json（统一 schema）
    """
    store = get_store()

    # 读取现有 JSON（可能是旧 schema；store 内会尽量归一化）
    entities_path = _tools.ENTITIES_FILE
    abstract_path = _tools.ABSTRACT_MAP_FILE

    entities = {}
    if entities_path.exists():
        try:
            entities = json.loads(entities_path.read_text(encoding="utf-8"))
        except Exception:
            entities = {}

    abstract_map = {}
    if abstract_path.exists():
        try:
            abstract_map = json.loads(abstract_path.read_text(encoding="utf-8"))
        except Exception:
            abstract_map = {}

    # 迁移 entities：把 name/original_forms/sources/first_seen 尽量保留
    ent_count = 0
    for name, data in (entities or {}).items():
        if not isinstance(name, str) or not isinstance(data, dict):
            continue
        first_seen = str(data.get("first_seen") or "").strip() or None
        sources = data.get("sources", [])
        original_forms = data.get("original_forms", [])
        if not isinstance(original_forms, list) or not original_forms:
            original_forms = [name]
        # upsert_entities 需要同长度
        store.upsert_entities([name], [str(original_forms[0] or name)], source=(sources[0] if isinstance(sources, list) and sources else "migrated"), reported_at=first_seen)
        ent_count += 1

    # 迁移 events：逐条喂给 upsert_events（其内部会写 participants/relations，并补 time）
    evt_count = 0
    for abstract, data in (abstract_map or {}).items():
        if not isinstance(abstract, str) or not isinstance(data, dict):
            continue
        # 尽量从旧结构推断 reported_at（若没有就用 first_seen）
        reported_at = str(data.get("reported_at") or data.get("published_at") or data.get("first_seen") or "").strip() or None
        # source 尽量保留（可能是 list[str] 或 list[dict]）
        src = None
        sources = data.get("sources", [])
        if isinstance(sources, list) and sources:
            src = sources[0]
        if not src:
            src = "migrated"

        event_obj = {
            "abstract": abstract,
            "event_summary": data.get("event_summary", ""),
            "event_types": data.get("event_types", []),
            "entities": data.get("entities", []),
            "entity_roles": data.get("entity_roles", {}),
            "relations": data.get("relations", []),
            "event_start_time": data.get("event_start_time", ""),
            "event_start_time_text": data.get("event_start_time_text", ""),
            "event_start_time_precision": data.get("event_start_time_precision", "unknown"),
        }
        store.upsert_events([event_obj], source=src, reported_at=reported_at)
        evt_count += 1

    # 最终统一导出兼容 JSON
    store.export_compat_json_files()

    return {
        "status": "success",
        "entities_imported": ent_count,
        "events_imported": evt_count,
        "sqlite_db": str(_tools.SQLITE_DB_FILE),
        "exported": [str(_tools.ENTITIES_FILE), str(_tools.ABSTRACT_MAP_FILE)],
    }


@register_tool(
    name="backfill_mentions",
    description="一次性回填 mentions：基于现有 SQLite entities/events 生成 entity_mentions/event_mentions（用于 mention-first 审计层）",
    category="Storage",
)
def backfill_mentions(limit_entities: int = 0, limit_events: int = 0) -> Dict[str, Any]:
    """
    回填策略（MVP）：
    - 每个 entity 生成 1 条 mention（reported_at=first_seen，source_json=entities.sources_json）
    - 每个 event 生成 1 条 mention（reported_at=first_seen/report_at，source_json=events.sources_json）
    说明：这是"最小可用"的历史补齐；后续可按 participants/relations 颗粒度回填更多 mention。
    """
    store = get_store()
    now = datetime.now(timezone.utc).isoformat()

    def sha1_text(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

    with store._lock:
        conn = store._connect()
        try:
            ent_rows = conn.execute(
                "SELECT entity_id, name, first_seen, sources_json FROM entities ORDER BY first_seen ASC"
                + (f" LIMIT {int(limit_entities)}" if int(limit_entities) > 0 else "")
            ).fetchall()
            evt_rows = conn.execute(
                "SELECT event_id, abstract, first_seen, reported_at, sources_json FROM events ORDER BY first_seen ASC"
                + (f" LIMIT {int(limit_events)}" if int(limit_events) > 0 else "")
            ).fetchall()

            ent_added = 0
            for r in ent_rows:
                entity_id = str(r["entity_id"])
                name = str(r["name"] or "")
                ts = str(r["first_seen"] or "") or now
                src_json = str(r["sources_json"] or "[]")
                mid = sha1_text(f"ent_mention_backfill:{entity_id}|{ts}|{src_json}")
                cur = conn.execute(
                    """
                    INSERT INTO entity_mentions(mention_id, name_text, reported_at, source_json, resolved_entity_id, confidence, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(mention_id) DO NOTHING
                    """,
                    (mid, name, ts, src_json, entity_id, 1.0, ts),
                )
                if cur.rowcount and cur.rowcount > 0:
                    ent_added += 1

            evt_added = 0
            for r in evt_rows:
                event_id = str(r["event_id"])
                abstract = str(r["abstract"] or "")
                ts = str(r["first_seen"] or "") or str(r["reported_at"] or "") or now
                src_json = str(r["sources_json"] or "[]")
                mid = sha1_text(f"evt_mention_backfill:{event_id}|{ts}|{src_json}")
                cur = conn.execute(
                    """
                    INSERT INTO event_mentions(mention_id, abstract_text, reported_at, source_json, resolved_event_id, confidence, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(mention_id) DO NOTHING
                    """,
                    (mid, abstract, ts, src_json, event_id, 1.0, ts),
                )
                if cur.rowcount and cur.rowcount > 0:
                    evt_added += 1

            conn.commit()

            return {
                "status": "ok",
                "entities_considered": len(ent_rows),
                "events_considered": len(evt_rows),
                "entity_mentions_added": ent_added,
                "event_mentions_added": evt_added,
            }
        finally:
            conn.close()


@register_tool(
    name="migrate_sqlite_to_neo4j",
    description="一次性迁移：从 SQLite 主存储导入 Neo4j（entities/events/participants/relations）",
    category="Storage",
)
def migrate_sqlite_to_neo4j(batch_size: int = 200) -> Dict[str, Any]:
    sqlite_store = get_store()
    try:
        from ...adapters.graph_store.neo4j_adapter import get_neo4j_store
    except Exception as e:
        return {"status": "error", "message": f"Neo4j adapter unavailable: {e}"}

    neo = get_neo4j_store()

    entities = sqlite_store.export_entities_json() or {}
    abstract_map = sqlite_store.export_abstract_map_json() or {}

    ent_items = [(k, v) for k, v in (entities or {}).items() if isinstance(k, str) and isinstance(v, dict)]
    evt_items = [(k, v) for k, v in (abstract_map or {}).items() if isinstance(k, str) and isinstance(v, dict)]

    ent_done = 0
    for i in range(0, len(ent_items), int(batch_size)):
        chunk = ent_items[i : i + int(batch_size)]
        names: List[str] = []
        originals: List[str] = []
        src = "migrated"
        ts = None
        for name, data in chunk:
            names.append(name)
            of = data.get("original_forms", [])
            if isinstance(of, list) and of:
                originals.append(str(of[0] or name))
            else:
                originals.append(name)
            if src == "migrated":
                sources = data.get("sources", [])
                if isinstance(sources, list) and sources:
                    src = sources[0]
            if ts is None:
                ts = str(data.get("first_seen") or "").strip() or None
        neo.upsert_entities(names, originals, source=src, reported_at=ts)
        ent_done += len(chunk)

    evt_done = 0
    for i in range(0, len(evt_items), int(batch_size)):
        chunk = evt_items[i : i + int(batch_size)]
        events_list: List[Dict[str, Any]] = []
        src = "migrated"
        ts = None
        for abstract, data in chunk:
            events_list.append(
                {
                    "abstract": abstract,
                    "event_summary": data.get("event_summary", ""),
                    "event_types": data.get("event_types", []),
                    "entities": data.get("entities", []),
                    "entity_roles": data.get("entity_roles", {}),
                    "relations": data.get("relations", []),
                    "event_start_time": data.get("event_start_time", ""),
                    "event_start_time_text": data.get("event_start_time_text", ""),
                    "event_start_time_precision": data.get("event_start_time_precision", "unknown"),
                }
            )
            if src == "migrated":
                sources = data.get("sources", [])
                if isinstance(sources, list) and sources:
                    src = sources[0]
            if ts is None:
                ts = str(data.get("reported_at") or data.get("first_seen") or "").strip() or None

        neo.upsert_events(events_list, source=src, reported_at=ts)
        evt_done += len(chunk)

    return {
        "status": "success",
        "entities_migrated": ent_done,
        "events_migrated": evt_done,
        "neo4j_uri": getattr(neo, "_uri", ""),
        "neo4j_database": getattr(neo, "_database", None),
    }




