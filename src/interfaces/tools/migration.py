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
from ...infra.paths import tools as Tools
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
    if not getattr(neo, "is_available", lambda: False)():
        return {
            "status": "error",
            "message": "Neo4j is not available. Please check NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD and service status.",
            "neo4j_uri": getattr(neo, "_uri", ""),
            "neo4j_database": getattr(neo, "_database", None),
        }

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


@register_tool(
    name="backfill_relation_kind",
    description="离线回填 relations.relation_kind（规则推断，无需 LLM）",
    category="Storage",
)
def backfill_relation_kind(limit: int = 0, rebuild_relation_states: int = 1) -> Dict[str, Any]:
    from ...adapters.sqlite.store import _infer_relation_kind, _norm_relation_kind

    store = get_store()
    limit_n = int(limit) if int(limit) > 0 else 0
    do_rebuild = int(rebuild_relation_states or 0) != 0

    updated = 0
    triples: Dict[str, int] = {}
    with store._lock:
        conn = store._connect()
        try:
            try:
                rows = conn.execute(
                    """
                    SELECT id, subject_entity_id, predicate, object_entity_id, relation_kind
                    FROM relations
                    WHERE relation_kind='' OR relation_kind IS NULL
                    ORDER BY id ASC
                    """
                ).fetchall()
            except sqlite3.OperationalError as e:
                return {"status": "error", "message": f"relations 表不可用或缺少字段: {e}"}

            if limit_n and len(rows) > limit_n:
                rows = rows[:limit_n]

            for r in rows or []:
                rid = int(r["id"])
                predicate = str(r["predicate"] or "")
                current = str(r["relation_kind"] or "")
                rk = _norm_relation_kind(current) or _infer_relation_kind(predicate)
                if not rk or rk == current:
                    continue
                conn.execute("UPDATE relations SET relation_kind=? WHERE id=?", (rk, rid))
                updated += 1
                s_id = str(r["subject_entity_id"] or "").strip()
                o_id = str(r["object_entity_id"] or "").strip()
                if s_id and predicate and o_id:
                    triples[f"{s_id}::{predicate}::{o_id}"] = 1

            conn.commit()
        finally:
            conn.close()

    rebuilt = 0
    if do_rebuild and triples:
        for key in triples.keys():
            parts = key.split("::", 2)
            if len(parts) != 3:
                continue
            s_id, predicate, o_id = parts
            rebuilt += int(store.rebuild_relation_states_for_triple(s_id, predicate, o_id) or 0)

    return {
        "status": "ok",
        "relations_updated": updated,
        "triples_touched": len(triples),
        "relation_states_written": rebuilt,
    }


@register_tool(
    name="build_event_observations",
    description="构建内部事件观测层：为事件生成可回放的 event_observations",
    category="Storage",
)
def build_event_observations(event_id: str = "", limit: int = 200) -> Dict[str, Any]:
    store = get_store()
    eid = str(event_id or "").strip()
    if eid:
        wrote = store.seed_event_observations(eid)
        return {"status": "ok", "mode": "single", "event_id": eid, "observations_written": wrote}

    limit_n = int(limit) if int(limit) > 0 else 200
    with store._lock:
        conn = store._connect()
        try:
            rows = conn.execute(
                """
                SELECT e.event_id
                FROM events e
                LEFT JOIN (
                    SELECT DISTINCT event_id AS event_id
                    FROM event_observations
                ) o ON o.event_id = e.event_id
                WHERE o.event_id IS NULL
                ORDER BY e.first_seen ASC
                LIMIT ?
                """,
                (limit_n,),
            ).fetchall()
            ids = [str(r["event_id"]) for r in rows or []]
        finally:
            conn.close()

    total = 0
    done = 0
    for x in ids:
        total += store.seed_event_observations(x)
        done += 1
    return {"status": "ok", "mode": "batch_missing_only", "events_considered": done, "observations_written": total}


@register_tool(
    name="validate_events_against_signals",
    description="将内部观测投影与外部 event_signals 做一致性校验，输出报告",
    category="Storage",
)
def validate_events_against_signals(event_id: str = "", limit: int = 50) -> Dict[str, Any]:
    store = get_store()
    eid = str(event_id or "").strip()
    if eid:
        return store.validate_event_against_signals(eid)

    limit_n = int(limit) if int(limit) > 0 else 50
    with store._lock:
        conn = store._connect()
        try:
            rows = conn.execute(
                """
                SELECT s.event_id
                FROM event_signals s
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit_n,),
            ).fetchall()
            ids = [str(r["event_id"]) for r in rows or []]
        finally:
            conn.close()

    reports: List[Dict[str, Any]] = []
    summary = {"match": 0, "mismatch": 0, "unknown": 0}
    for x in ids:
        r = store.validate_event_against_signals(x)
        reports.append(r)
        for c in r.get("checks") or []:
            st = str(c.get("status") or "unknown")
            if st in summary:
                summary[st] += 1
            else:
                summary["unknown"] += 1

    return {"status": "ok", "events": len(ids), "check_summary": summary, "reports": reports}


def _trim_text(s: Any, limit: int) -> str:
    t = str(s or "")
    if len(t) <= limit:
        return t
    return t[:limit]


def _build_event_observations_prompt(payload: Dict[str, Any]) -> str:
    base = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""你是“事件观测构建专家”。你的输出将直接写入数据库 event_observations，用于默认字段投影与可回放审计。你必须严格遵守以下规则：

1) 只使用输入 JSON 里的信息，不得引入任何外部知识或推测。
2) 事件为过程性表达：时间字段以“开始时间 start_time”为核心，尽可能精确；无法精确时必须标注精度。
3) 主观字段不得量化：
   - color（事件色彩程度）只能输出无立场、过程性的客观描述文本，禁止数值/打分/百分比。
   - credibility（可信度）只能输出“基于证据的文字陈述”，包含不确定性来源，禁止数值/打分/概率。
4) 证据必须可回放：每个观测必须给出 evidence 数组，元素至少包含 mention_id 与 quote（引用原文片段，短且直接）。
5) 输出必须是严格 JSON 对象，且只输出 JSON，不要任何解释文字、不要 markdown 代码块。

你将收到一个输入对象 input。请返回如下结构（字段名必须完全一致）：
{{
  "observations": [
    {{
      "field": "start_time|description|color|credibility",
      "value_text": "用于展示的主文本（字符串）",
      "value_json": {{}},
      "evidence": [{{"mention_id": "…", "quote": "…"}}]
    }}
  ]
}}

其中 start_time 的 value_json 必须包含：
- time: ISO8601 时间字符串（无法确定则置空字符串）
- time_text: 原文时间片段或规范化展示文本
- precision: 取值限定为 year|month|day|hour|minute|second|unknown
- candidates: 候选数组，每个候选包含 time/time_text/precision/evidence/reason
- selection_reason: 为什么选择该候选作为默认（短句，基于证据，不要泛泛而谈）

description 的 value_text 必须是“无情绪、无立场”的一句话事件描述，尽量包含谁对谁做了什么。

输入 input 如下：
{base}
"""


def _fetch_event_context(store, event_id: str, max_mentions: int, max_mention_chars: int) -> Dict[str, Any]:
    eid = str(event_id or "").strip()
    if not eid:
        return {}

    with store._lock:
        conn = store._connect()
        try:
            evt = conn.execute(
                """
                SELECT event_id, abstract, event_summary, event_types_json,
                       event_start_time, event_start_time_text, event_start_time_precision,
                       reported_at, first_seen, last_seen
                FROM events
                WHERE event_id=?
                """,
                (eid,),
            ).fetchone()
            if evt is None:
                return {}

            rows = conn.execute(
                """
                SELECT m.mention_id, m.abstract_text, m.reported_at, m.source_json, m.confidence
                FROM event_mentions m
                WHERE m.resolved_event_id=?
                ORDER BY m.reported_at ASC
                """,
                (eid,),
            ).fetchall()

            participants = conn.execute(
                """
                SELECT e.name AS name, p.roles_json AS roles_json
                FROM participants p
                JOIN entities e ON e.entity_id = p.entity_id
                WHERE p.event_id=?
                ORDER BY e.name ASC
                """,
                (eid,),
            ).fetchall()

            rels = conn.execute(
                """
                SELECT es.name AS subject, r.predicate AS predicate, eo.name AS object, r.evidence_json AS evidence_json
                FROM relations r
                JOIN entities es ON es.entity_id = r.subject_entity_id
                JOIN entities eo ON eo.entity_id = r.object_entity_id
                WHERE r.event_id=?
                ORDER BY r.id ASC
                """,
                (eid,),
            ).fetchall()
        finally:
            conn.close()

    evt_d = dict(evt)
    try:
        event_types = json.loads(str(evt_d.get("event_types_json") or "[]"))
        if not isinstance(event_types, list):
            event_types = []
    except Exception:
        event_types = []

    mentions_out: List[Dict[str, Any]] = []
    for r in (rows or [])[: max(0, int(max_mentions))]:
        d = dict(r)
        mentions_out.append(
            {
                "mention_id": str(d.get("mention_id") or ""),
                "reported_at": str(d.get("reported_at") or ""),
                "text": _trim_text(d.get("abstract_text") or "", int(max_mention_chars)),
                "confidence": d.get("confidence"),
                "source_json": str(d.get("source_json") or ""),
            }
        )

    parts_out: List[Dict[str, Any]] = []
    for p in participants or []:
        d = dict(p)
        roles = []
        try:
            roles = json.loads(str(d.get("roles_json") or "[]"))
            if not isinstance(roles, list):
                roles = []
        except Exception:
            roles = []
        parts_out.append({"name": str(d.get("name") or ""), "roles": roles})

    rels_out: List[Dict[str, Any]] = []
    for r in rels or []:
        d = dict(r)
        ev = []
        try:
            ev = json.loads(str(d.get("evidence_json") or "[]"))
            if not isinstance(ev, list):
                ev = []
        except Exception:
            ev = []
        rels_out.append(
            {
                "subject": str(d.get("subject") or ""),
                "predicate": str(d.get("predicate") or ""),
                "object": str(d.get("object") or ""),
                "evidence": ev,
            }
        )

    return {
        "event": {
            "event_id": str(evt_d.get("event_id") or ""),
            "abstract": str(evt_d.get("abstract") or ""),
            "event_summary": str(evt_d.get("event_summary") or ""),
            "event_types": event_types,
            "event_start_time": str(evt_d.get("event_start_time") or ""),
            "event_start_time_text": str(evt_d.get("event_start_time_text") or ""),
            "event_start_time_precision": str(evt_d.get("event_start_time_precision") or ""),
            "reported_at": str(evt_d.get("reported_at") or ""),
            "first_seen": str(evt_d.get("first_seen") or ""),
            "last_seen": str(evt_d.get("last_seen") or ""),
        },
        "mentions": mentions_out,
        "participants": parts_out,
        "relations": rels_out,
    }


def _coerce_obs_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    field = str(item.get("field") or "").strip()
    value_text = str(item.get("value_text") or "").strip()
    value_json = item.get("value_json")
    if not isinstance(value_json, dict):
        value_json = {}
    evidence = item.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence_out = []
    for e in evidence:
        if not isinstance(e, dict):
            continue
        mention_id = str(e.get("mention_id") or "").strip()
        quote = str(e.get("quote") or "").strip()
        if mention_id and quote:
            evidence_out.append({"mention_id": mention_id, "quote": quote})
    return {"field": field, "value_text": value_text, "value_json": value_json, "evidence_json": evidence_out}


@register_tool(
    name="build_event_observations_llm",
    description="用 LLM 构建事件观测层：生成 start_time/description/color/credibility 并入库",
    category="Storage",
)
def build_event_observations_llm(
    event_id: str = "",
    limit: int = 50,
    max_mentions: int = 25,
    max_mention_chars: int = 360,
    timeout: int = 55,
) -> Dict[str, Any]:
    store = get_store()
    eid = str(event_id or "").strip()

    def _build_one(eid_one: str) -> Dict[str, Any]:
        ctx = _fetch_event_context(store, eid_one, int(max_mentions), int(max_mention_chars))
        if not ctx:
            return {"event_id": eid_one, "status": "skip", "reason": "event_not_found"}

        payload = {"input": ctx}
        prompt = _build_event_observations_prompt(payload)

        from ...infra import extract_json_from_llm_response
        from ...ports.llm_client import LLMCallConfig
        from ...adapters.llm.pool import get_llm_pool

        llm_pool = get_llm_pool()
        resp = llm_pool.call(
            prompt=prompt,
            config=LLMCallConfig(max_tokens=2200, timeout_seconds=int(timeout), retries=2),
        )
        if not resp.success:
            return {"event_id": eid_one, "status": "error", "error": resp.error or "llm_failed"}

        try:
            data = extract_json_from_llm_response(resp.content)
        except Exception as e:
            return {"event_id": eid_one, "status": "error", "error": f"json_parse_failed: {e}"}

        obs_in = data.get("observations")
        if not isinstance(obs_in, list) or not obs_in:
            return {"event_id": eid_one, "status": "skip", "reason": "empty_observations"}

        now = datetime.now(timezone.utc).isoformat()
        allowed = {"start_time", "description", "color", "credibility"}
        to_write: List[Dict[str, Any]] = []
        for raw in obs_in:
            item = _coerce_obs_item(raw)
            field = str(item.get("field") or "").strip()
            if field not in allowed:
                continue
            if field in {"start_time", "description"} and not str(item.get("value_text") or "").strip():
                continue
            to_write.append(
                {
                    "event_id": eid_one,
                    "field": field,
                    "value_text": item.get("value_text") or "",
                    "value_json": item.get("value_json") or {},
                    "evidence_json": item.get("evidence_json") or [],
                    "model_name": str(resp.model or "llm"),
                    "model_version": "",
                    "algorithm": "llm_observation_builder_v1",
                    "revision": 1,
                    "is_default": 1,
                    "created_at": now,
                    "updated_at": now,
                }
            )

        if not to_write:
            return {"event_id": eid_one, "status": "skip", "reason": "no_valid_observations"}

        wrote = store.upsert_event_observations(to_write)
        return {"event_id": eid_one, "status": "ok", "observations_written": wrote, "model": resp.model}

    if eid:
        store.seed_event_observations(eid)
        r = _build_one(eid)
        return {"status": "ok" if r.get("status") == "ok" else "error", "mode": "single", "result": r}

    limit_n = int(limit) if int(limit) > 0 else 50
    with store._lock:
        conn = store._connect()
        try:
            rows = conn.execute(
                """
                SELECT e.event_id
                FROM events e
                LEFT JOIN (
                    SELECT DISTINCT event_id AS event_id
                    FROM event_observations
                    WHERE algorithm='llm_observation_builder_v1'
                ) o ON o.event_id = e.event_id
                WHERE o.event_id IS NULL
                ORDER BY e.first_seen ASC
                LIMIT ?
                """,
                (limit_n,),
            ).fetchall()
            ids = [str(r["event_id"]) for r in rows or []]
        finally:
            conn.close()

    results: List[Dict[str, Any]] = []
    wrote_total = 0
    for x in ids:
        store.seed_event_observations(x)
        r = _build_one(x)
        results.append(r)
        if r.get("status") == "ok":
            wrote_total += int(r.get("observations_written") or 0)

    return {
        "status": "ok",
        "mode": "batch_missing_llm_only",
        "events_considered": len(ids),
        "observations_written": wrote_total,
        "results": results,
    }


def _looks_like_sha1_hex(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip().lower()
    if len(t) != 40:
        return False
    for ch in t:
        if ch not in "0123456789abcdef":
            return False
    return True


@register_tool(
    name="build_relation_states",
    description="离线构建关系状态层：由 relations 生成 relation_states（规则版）",
    category="Storage",
)
def build_relation_states(
    subject: str = "",
    predicate: str = "",
    object: str = "",
    limit: int = 200,
    algorithm: str = "rule_relation_states_v1",
) -> Dict[str, Any]:
    from ...adapters.sqlite.store import canonical_entity_id

    store = get_store()
    s_in = str(subject or "").strip()
    p_in = str(predicate or "").strip()
    o_in = str(object or "").strip()

    if (s_in or p_in or o_in) and not (s_in and p_in and o_in):
        return {"status": "error", "message": "subject/predicate/object 必须同时提供或都不提供"}

    if s_in and p_in and o_in:
        s_id = s_in if _looks_like_sha1_hex(s_in) else canonical_entity_id(s_in)
        o_id = o_in if _looks_like_sha1_hex(o_in) else canonical_entity_id(o_in)
        wrote = store.rebuild_relation_states_for_triple(s_id, p_in, o_id, algorithm=str(algorithm or "rule_relation_states_v1"))
        return {
            "status": "ok",
            "mode": "single",
            "triple": {"subject_entity_id": s_id, "predicate": p_in, "object_entity_id": o_id},
            "relation_states_written": wrote,
        }

    limit_n = int(limit) if int(limit) > 0 else 200
    with store._lock:
        conn = store._connect()
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT r.subject_entity_id AS subject_entity_id, r.predicate AS predicate, r.object_entity_id AS object_entity_id
                FROM relations r
                LEFT JOIN (
                    SELECT DISTINCT subject_entity_id, predicate, object_entity_id
                    FROM relation_states
                    WHERE algorithm=?
                ) rs
                ON rs.subject_entity_id=r.subject_entity_id AND rs.predicate=r.predicate AND rs.object_entity_id=r.object_entity_id
                WHERE rs.subject_entity_id IS NULL
                ORDER BY r.time ASC
                LIMIT ?
                """,
                (str(algorithm or "rule_relation_states_v1"), limit_n),
            ).fetchall()
            triples = [dict(r) for r in rows or []]
        finally:
            conn.close()

    wrote_total = 0
    for t in triples:
        s_id = str(t.get("subject_entity_id") or "").strip()
        p = str(t.get("predicate") or "").strip()
        o_id = str(t.get("object_entity_id") or "").strip()
        if not s_id or not p or not o_id:
            continue
        wrote_total += int(
            store.rebuild_relation_states_for_triple(s_id, p, o_id, algorithm=str(algorithm or "rule_relation_states_v1")) or 0
        )

    return {
        "status": "ok",
        "mode": "batch_missing_only",
        "triples_considered": len(triples),
        "relation_states_written": wrote_total,
    }


@register_tool(
    name="get_relation_timeline",
    description="查询关系状态时间线：返回 relation_states 列表",
    category="Storage",
)
def get_relation_timeline(
    subject: str,
    predicate: str,
    object: str,
    limit: int = 200,
) -> Dict[str, Any]:
    from ...adapters.sqlite.store import canonical_entity_id

    store = get_store()
    s_in = str(subject or "").strip()
    p_in = str(predicate or "").strip()
    o_in = str(object or "").strip()
    if not s_in or not p_in or not o_in:
        return {"status": "error", "message": "subject/predicate/object 不能为空"}
    s_id = s_in if _looks_like_sha1_hex(s_in) else canonical_entity_id(s_in)
    o_id = o_in if _looks_like_sha1_hex(o_in) else canonical_entity_id(o_in)
    timeline = store.list_relation_states(s_id, p_in, o_id, limit=int(limit))
    return {
        "status": "ok",
        "triple": {"subject_entity_id": s_id, "predicate": p_in, "object_entity_id": o_id},
        "timeline": timeline,
    }



