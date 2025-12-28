from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...infra.paths import tools as Tools


_tools = Tools()


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def canonical_entity_id(entity_name: str) -> str:
    name = (entity_name or "").strip()
    return _sha1_text(f"ent:{name}")


def canonical_event_id(abstract: str) -> str:
    a = (abstract or "").strip()
    return _sha1_text(f"evt:{a}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_iso_ts(val: Any) -> str:
    if not val:
        return ""
    if isinstance(val, str):
        return val.strip()
    try:
        return str(val)
    except Exception:
        return ""


def _norm_source_list(sources: Any) -> List[Dict[str, str]]:
    """
    统一 sources 为 list[dict]：
    - 允许历史格式 list[str]
    - 允许 list[dict]（包含 id/name/url 任意子集）
    """
    out: List[Dict[str, str]] = []

    def add_one(src: Any):
        if isinstance(src, str):
            name = src.strip()
            if not name:
                return
            out.append({"id": _sha1_text(f"src:{name}"), "name": name, "url": ""})
            return
        if isinstance(src, dict):
            _id = str(src.get("id") or "").strip()
            name = str(src.get("name") or "").strip()
            url = str(src.get("url") or "").strip()
            if not name and _id:
                name = _id
            if not _id and name:
                _id = _sha1_text(f"src:{name}")
            if not name:
                return
            out.append({"id": _id, "name": name, "url": url})

    if isinstance(sources, list):
        for s in sources:
            add_one(s)

    # 去重（按 id 优先，其次 name+url）
    seen = set()
    dedup: List[Dict[str, str]] = []
    for s in out:
        key = s.get("id") or f'{s.get("name","")}::{s.get("url","")}'
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)
    return dedup


def _choose_event_time(event_start_time: str, reported_at: str, first_seen: str) -> str:
    """
    每个元组都必须有 time：
    - 优先 event_start_time
    - 其次 reported_at
    - 再其次 first_seen
    - 再不行就是当前时间（兜底）
    """
    t = (event_start_time or "").strip() or (reported_at or "").strip() or (first_seen or "").strip()
    return t or _utc_now_iso()


@dataclass(frozen=True)
class SQLiteStoreConfig:
    db_path: Path


class SQLiteStore:
    """
    SQLite 主存储：
    - entities/events 为主表
    - participants/relations 为“带时间的元组”（强制 time 字段非空）
    """

    SCHEMA_VERSION = "2"

    def __init__(self, config: Optional[SQLiteStoreConfig] = None):
        self.config = config or SQLiteStoreConfig(db_path=_tools.SQLITE_DB_FILE)
        self._lock = threading.RLock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.config.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # 更适合并发读写的 WAL
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _ensure_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS entities (
                        entity_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        sources_json TEXT NOT NULL,
                        original_forms_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        abstract TEXT NOT NULL UNIQUE,
                        event_summary TEXT NOT NULL,
                        event_types_json TEXT NOT NULL,
                        event_start_time TEXT NOT NULL,
                        event_start_time_text TEXT NOT NULL,
                        event_start_time_precision TEXT NOT NULL,
                        reported_at TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        sources_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS participants (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        roles_json TEXT NOT NULL,
                        time TEXT NOT NULL,
                        reported_at TEXT NOT NULL,
                        UNIQUE(event_id, entity_id),
                        FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                        FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS relations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        subject_entity_id TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object_entity_id TEXT NOT NULL,
                        time TEXT NOT NULL,
                        reported_at TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        UNIQUE(event_id, subject_entity_id, predicate, object_entity_id),
                        FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                        FOREIGN KEY(subject_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
                        FOREIGN KEY(object_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_participants_time ON participants(time);
                    CREATE INDEX IF NOT EXISTS idx_relations_time ON relations(time);
                    CREATE INDEX IF NOT EXISTS idx_events_first_seen ON events(first_seen);

                    -- =========================
                    -- Mention-first (审计层：先落 mention，再 resolve 到 canonical)
                    -- 说明：mentions 不加外键约束（避免 merge 删除导致历史丢失/约束冲突）
                    -- =========================
                    CREATE TABLE IF NOT EXISTS entity_mentions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mention_id TEXT NOT NULL UNIQUE,
                        name_text TEXT NOT NULL,
                        reported_at TEXT NOT NULL,
                        source_json TEXT NOT NULL,
                        resolved_entity_id TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 1.0,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_entity_mentions_reported_at ON entity_mentions(reported_at);
                    CREATE INDEX IF NOT EXISTS idx_entity_mentions_resolved ON entity_mentions(resolved_entity_id);

                    CREATE TABLE IF NOT EXISTS event_mentions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mention_id TEXT NOT NULL UNIQUE,
                        abstract_text TEXT NOT NULL,
                        reported_at TEXT NOT NULL,
                        source_json TEXT NOT NULL,
                        resolved_event_id TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 1.0,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_event_mentions_reported_at ON event_mentions(reported_at);
                    CREATE INDEX IF NOT EXISTS idx_event_mentions_resolved ON event_mentions(resolved_event_id);

                    -- =========================
                    -- Review Layer (LLM 审查层)
                    -- =========================
                    CREATE TABLE IF NOT EXISTS review_tasks (
                        task_id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        status TEXT NOT NULL, -- pending | running | done | failed | cancelled
                        priority INTEGER NOT NULL DEFAULT 0,
                        input_hash TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        output_json TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL DEFAULT '',
                        prompt_version TEXT NOT NULL DEFAULT '',
                        error TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_review_tasks_status ON review_tasks(status);
                    CREATE INDEX IF NOT EXISTS idx_review_tasks_type ON review_tasks(type);
                    CREATE INDEX IF NOT EXISTS idx_review_tasks_priority ON review_tasks(priority);

                    CREATE TABLE IF NOT EXISTS merge_decisions (
                        decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        type TEXT NOT NULL,
                        input_hash TEXT NOT NULL UNIQUE,
                        output_json TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_merge_decisions_type ON merge_decisions(type);

                    -- alias 与 redirect（实体收敛的可回放机制）
                    CREATE TABLE IF NOT EXISTS entity_aliases (
                        alias TEXT PRIMARY KEY,
                        entity_id TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 1.0,
                        decision_input_hash TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_id);

                    CREATE TABLE IF NOT EXISTS entity_redirects (
                        from_entity_id TEXT PRIMARY KEY,
                        to_entity_id TEXT NOT NULL,
                        reason TEXT NOT NULL DEFAULT '',
                        decision_input_hash TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(from_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
                        FOREIGN KEY(to_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
                    );

                    -- =========================
                    -- Event merge / evolution
                    -- =========================
                    CREATE TABLE IF NOT EXISTS event_aliases (
                        abstract TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 1.0,
                        decision_input_hash TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_event_aliases_event ON event_aliases(event_id);

                    CREATE TABLE IF NOT EXISTS event_redirects (
                        from_event_id TEXT PRIMARY KEY,
                        to_event_id TEXT NOT NULL,
                        reason TEXT NOT NULL DEFAULT '',
                        decision_input_hash TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(from_event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                        FOREIGN KEY(to_event_id) REFERENCES events(event_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS event_edges (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        from_event_id TEXT NOT NULL,
                        to_event_id TEXT NOT NULL,
                        edge_type TEXT NOT NULL, -- follows/responds_to/escalates/causes/related
                        time TEXT NOT NULL,
                        reported_at TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 0.0,
                        evidence_json TEXT NOT NULL DEFAULT '[]',
                        decision_input_hash TEXT NOT NULL DEFAULT '',
                        UNIQUE(from_event_id, to_event_id, edge_type),
                        FOREIGN KEY(from_event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                        FOREIGN KEY(to_event_id) REFERENCES events(event_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_event_edges_time ON event_edges(time);

                    -- =========================
                    -- Schema Migrations (版本管理)
                    -- =========================
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        description TEXT NOT NULL DEFAULT '',
                        applied_at TEXT NOT NULL,
                        success INTEGER NOT NULL DEFAULT 1
                    );

                    -- =========================
                    -- Processed IDs (已处理新闻ID)
                    -- =========================
                    CREATE TABLE IF NOT EXISTS processed_ids (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        global_id TEXT NOT NULL UNIQUE,
                        source TEXT NOT NULL,
                        news_id TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_processed_ids_global_id ON processed_ids(global_id);
                    CREATE INDEX IF NOT EXISTS idx_processed_ids_source ON processed_ids(source);
                    CREATE INDEX IF NOT EXISTS idx_processed_ids_created_at ON processed_ids(created_at);
                    
                    -- =========================
                    -- News to Events Mapping (新闻ID到事件ID映射)
                    -- =========================
                    CREATE TABLE IF NOT EXISTS news_event_mapping (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        news_global_id TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(news_global_id, event_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_news_event_mapping_news ON news_event_mapping(news_global_id);
                    CREATE INDEX IF NOT EXISTS idx_news_event_mapping_event ON news_event_mapping(event_id);
                    CREATE INDEX IF NOT EXISTS idx_news_event_mapping_created_at ON news_event_mapping(created_at);
                    """
                )
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("schema_version", self.SCHEMA_VERSION),
                )
                conn.commit()
            finally:
                conn.close()

    # -------------------------
    # Review APIs
    # -------------------------

    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        try:
            s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            s = str(payload)
        return _sha1_text(f"payload:{s}")

    def enqueue_review_task(self, task_type: str, payload: Dict[str, Any], *, priority: int = 0) -> str:
        """
        入队审查任务（幂等）：同一 input_hash 只会保留一条 pending/running 任务。
        """
        from uuid import uuid4

        now = _utc_now_iso()
        input_hash = self._hash_payload({"type": task_type, "payload": payload})
        task_id = uuid4().hex

        with self._lock:
            conn = self._connect()
            try:
                # 若已有同 input_hash 的 pending/running，直接复用
                row = conn.execute(
                    "SELECT task_id FROM review_tasks WHERE input_hash=? AND status IN ('pending','running')",
                    (input_hash,),
                ).fetchone()
                if row is not None:
                    return str(row["task_id"])

                conn.execute(
                    """
                    INSERT INTO review_tasks(task_id, type, status, priority, input_hash, payload_json, created_at, updated_at)
                    VALUES(?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (task_id, task_type, int(priority), input_hash, json.dumps(payload, ensure_ascii=False), now, now),
                )
                conn.commit()
                return task_id
            finally:
                conn.close()

    def claim_next_review_task(self, *, task_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        领取一个待审查任务（按 priority DESC, created_at ASC）。
        """
        # 先把“卡死的 running 任务”回收到 pending（常见于用户中断/进程崩溃）
        try:
            self.requeue_stale_review_tasks(max_age_minutes=10)
        except Exception:
            pass
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                if task_type:
                    row = conn.execute(
                        """
                        SELECT task_id, type, input_hash, payload_json
                        FROM review_tasks
                        WHERE status='pending' AND type=?
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1
                        """,
                        (task_type,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT task_id, type, input_hash, payload_json
                        FROM review_tasks
                        WHERE status='pending'
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1
                        """
                    ).fetchone()
                if row is None:
                    return None
                task_id = str(row["task_id"])
                conn.execute(
                    "UPDATE review_tasks SET status='running', updated_at=? WHERE task_id=?",
                    (now, task_id),
                )
                conn.commit()
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except Exception:
                    payload = {}
                return {"task_id": task_id, "type": str(row["type"]), "input_hash": str(row["input_hash"]), "payload": payload}
            finally:
                conn.close()

    def requeue_stale_review_tasks(self, *, max_age_minutes: int = 10) -> int:
        """
        将超时的 running 任务重新入队 pending（避免因 Ctrl+C/kill 导致 worker 永久“取不到任务”）。
        判定依据：updated_at < now - max_age_minutes
        """
        try:
            minutes = int(max_age_minutes)
        except Exception:
            minutes = 10
        if minutes <= 0:
            minutes = 1

        cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(minutes=minutes)).isoformat()
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE review_tasks
                    SET status='pending', updated_at=?
                    WHERE status='running' AND updated_at < ?
                    """,
                    (now, cutoff),
                )
                conn.commit()
                return int(cur.rowcount or 0)
            finally:
                conn.close()

    def complete_review_task(
        self,
        task_id: str,
        *,
        status: str,
        output: Optional[Dict[str, Any]] = None,
        model: str = "",
        prompt_version: str = "",
        error: str = "",
    ) -> None:
        now = _utc_now_iso()
        out_json = json.dumps(output or {}, ensure_ascii=False) if output is not None else ""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE review_tasks
                    SET status=?, output_json=?, model=?, prompt_version=?, error=?, updated_at=?
                    WHERE task_id=?
                    """,
                    (status, out_json, model or "", prompt_version or "", error or "", now, task_id),
                )
                conn.commit()
            finally:
                conn.close()

    def upsert_merge_decision(self, decision_type: str, input_hash: str, output: Dict[str, Any], *, model: str, prompt_version: str) -> None:
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO merge_decisions(type, input_hash, output_json, model, prompt_version, created_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(input_hash) DO UPDATE SET
                        output_json=excluded.output_json,
                        model=excluded.model,
                        prompt_version=excluded.prompt_version
                    """,
                    (decision_type, input_hash, json.dumps(output, ensure_ascii=False), model, prompt_version, now),
                )
                conn.commit()
            finally:
                conn.close()

    # -------------------------
    # UPSERT APIs
    # -------------------------

    def upsert_entities(
        self,
        entities: List[str],
        entities_original: List[str],
        *,
        source: Any,
        reported_at: Optional[str],
    ) -> None:
        if len(entities) != len(entities_original):
            return

        base_ts = _norm_iso_ts(reported_at) or _utc_now_iso()
        sources = _norm_source_list([source])

        with self._lock:
            conn = self._connect()
            try:
                for name, orig in zip(entities, entities_original):
                    n = (name or "").strip()
                    o = (orig or "").strip()
                    if not n:
                        continue
                    ent_id = canonical_entity_id(n)

                    row = conn.execute(
                        "SELECT first_seen, last_seen, sources_json, original_forms_json FROM entities WHERE entity_id=?",
                        (ent_id,),
                    ).fetchone()

                    if row is None:
                        conn.execute(
                            """
                            INSERT INTO entities(entity_id, name, first_seen, last_seen, sources_json, original_forms_json)
                            VALUES(?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ent_id,
                                n,
                                base_ts,
                                base_ts,
                                json.dumps(sources, ensure_ascii=False),
                                json.dumps([o or n], ensure_ascii=False),
                            ),
                        )
                    else:
                        first_seen = str(row["first_seen"] or "")
                        last_seen = str(row["last_seen"] or "")
                        if first_seen and base_ts and base_ts < first_seen:
                            first_seen = base_ts
                        if last_seen and base_ts and base_ts > last_seen:
                            last_seen = base_ts
                        if not first_seen:
                            first_seen = base_ts
                        if not last_seen:
                            last_seen = base_ts

                        try:
                            old_sources = _norm_source_list(json.loads(row["sources_json"] or "[]"))
                        except Exception:
                            old_sources = []
                        merged_sources = _norm_source_list(old_sources + sources)

                        try:
                            old_forms = json.loads(row["original_forms_json"] or "[]")
                            if not isinstance(old_forms, list):
                                old_forms = []
                        except Exception:
                            old_forms = []
                        forms = [x for x in old_forms if isinstance(x, str) and x.strip()]
                        if o and o not in forms:
                            forms.append(o)

                        conn.execute(
                            """
                            UPDATE entities
                            SET name=?, first_seen=?, last_seen=?, sources_json=?, original_forms_json=?
                            WHERE entity_id=?
                            """,
                            (
                                n,
                                first_seen,
                                last_seen,
                                json.dumps(merged_sources, ensure_ascii=False),
                                json.dumps(forms or [n], ensure_ascii=False),
                                ent_id,
                            ),
                        )

                    # mention-first：记录每次出现（审计/回放）
                    try:
                        src_json = json.dumps(sources, ensure_ascii=False, sort_keys=True)
                    except Exception:
                        src_json = json.dumps(sources, ensure_ascii=False)
                    mention_id = _sha1_text(f"ent_mention:{n}|{base_ts}|{src_json}")
                    conn.execute(
                        """
                        INSERT INTO entity_mentions(mention_id, name_text, reported_at, source_json, resolved_entity_id, confidence, created_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(mention_id) DO NOTHING
                        """,
                        (mention_id, n, base_ts, src_json, ent_id, 1.0, base_ts),
                    )

                conn.commit()
            finally:
                conn.close()

    def upsert_events(self, extracted_events: List[Dict[str, Any]], *, source: Any, reported_at: Optional[str]) -> None:
        """
        extracted_events: 兼容当前抽取输出（字段可能缺失）
        强制：participants/relations 写入时必须带 time
        """
        base_ts = _norm_iso_ts(reported_at) or _utc_now_iso()
        sources = _norm_source_list([source])

        with self._lock:
            conn = self._connect()
            try:
                for item in extracted_events or []:
                    if not isinstance(item, dict):
                        continue
                    abstract = str(item.get("abstract") or "").strip()
                    if not abstract:
                        continue

                    event_id = canonical_event_id(abstract)
                    event_summary = str(item.get("event_summary") or "").strip()
                    event_types = item.get("event_types") if isinstance(item.get("event_types"), list) else []
                    event_types = [x for x in event_types if isinstance(x, str) and x.strip()]

                    entity_roles = item.get("entity_roles") if isinstance(item.get("entity_roles"), dict) else {}
                    relations = item.get("relations") if isinstance(item.get("relations"), list) else []
                    entities = item.get("entities") if isinstance(item.get("entities"), list) else []
                    entities = [x for x in entities if isinstance(x, str) and x.strip()]

                    event_start_time = _norm_iso_ts(item.get("event_start_time"))
                    event_start_time_text = str(item.get("event_start_time_text") or "").strip()
                    event_start_time_precision = str(item.get("event_start_time_precision") or "unknown").strip() or "unknown"

                    # 事件时间：用于所有元组 time
                    first_seen = base_ts
                    row = conn.execute(
                        "SELECT first_seen, last_seen, sources_json, event_types_json, event_start_time, event_start_time_text, event_start_time_precision, reported_at FROM events WHERE event_id=?",
                        (event_id,),
                    ).fetchone()

                    if row is None:
                        conn.execute(
                            """
                            INSERT INTO events(
                                event_id, abstract, event_summary, event_types_json,
                                event_start_time, event_start_time_text, event_start_time_precision,
                                reported_at, first_seen, last_seen, sources_json
                            )
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                event_id,
                                abstract,
                                event_summary,
                                json.dumps(event_types, ensure_ascii=False),
                                event_start_time,
                                event_start_time_text,
                                event_start_time_precision,
                                base_ts,
                                base_ts,
                                base_ts,
                                json.dumps(sources, ensure_ascii=False),
                            ),
                        )
                    else:
                        # 合并：first_seen 取更早，last_seen 取更晚；event_types union；sources union
                        old_first = str(row["first_seen"] or "")
                        old_last = str(row["last_seen"] or "")
                        first_seen = old_first or base_ts
                        if base_ts and first_seen and base_ts < first_seen:
                            first_seen = base_ts
                        last_seen = old_last or base_ts
                        if base_ts and last_seen and base_ts > last_seen:
                            last_seen = base_ts

                        try:
                            old_types = json.loads(row["event_types_json"] or "[]")
                            if not isinstance(old_types, list):
                                old_types = []
                        except Exception:
                            old_types = []
                        merged_types = []
                        seen_t = set()
                        for t in [*old_types, *event_types]:
                            if isinstance(t, str) and t.strip() and t not in seen_t:
                                seen_t.add(t)
                                merged_types.append(t)

                        try:
                            old_sources = _norm_source_list(json.loads(row["sources_json"] or "[]"))
                        except Exception:
                            old_sources = []
                        merged_sources = _norm_source_list(old_sources + sources)

                        # event_start_time：优先保留更精确/更早（这里先做简单规则：已有为空才补）
                        old_est = str(row["event_start_time"] or "")
                        old_est_txt = str(row["event_start_time_text"] or "")
                        old_est_p = str(row["event_start_time_precision"] or "unknown") or "unknown"
                        if not old_est and event_start_time:
                            old_est = event_start_time
                            old_est_txt = event_start_time_text
                            old_est_p = event_start_time_precision

                        old_reported = str(row["reported_at"] or "")
                        if old_reported and base_ts and base_ts < old_reported:
                            old_reported = base_ts
                        if not old_reported:
                            old_reported = base_ts

                        # event_summary：保留更长的（更信息密度）
                        old_summary = str(conn.execute("SELECT event_summary FROM events WHERE event_id=?", (event_id,)).fetchone()["event_summary"])
                        if event_summary and len(event_summary) > len(old_summary or ""):
                            old_summary = event_summary

                        conn.execute(
                            """
                            UPDATE events
                            SET event_summary=?, event_types_json=?, event_start_time=?, event_start_time_text=?, event_start_time_precision=?,
                                reported_at=?, first_seen=?, last_seen=?, sources_json=?
                            WHERE event_id=?
                            """,
                            (
                                old_summary or "",
                                json.dumps(merged_types, ensure_ascii=False),
                                old_est or "",
                                old_est_txt or "",
                                old_est_p or "unknown",
                                old_reported,
                                first_seen,
                                last_seen,
                                json.dumps(merged_sources, ensure_ascii=False),
                                event_id,
                            ),
                        )

                    # mention-first：记录每次事件摘要出现（审计/回放）
                    try:
                        src_json = json.dumps(sources, ensure_ascii=False, sort_keys=True)
                    except Exception:
                        src_json = json.dumps(sources, ensure_ascii=False)
                    mention_id = _sha1_text(f"evt_mention:{abstract}|{base_ts}|{src_json}")
                    conn.execute(
                        """
                        INSERT INTO event_mentions(mention_id, abstract_text, reported_at, source_json, resolved_event_id, confidence, created_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(mention_id) DO NOTHING
                        """,
                        (mention_id, abstract, base_ts, src_json, event_id, 1.0, base_ts),
                    )

                    # participants：entity_roles + entities（roles 可空，但元组必须有 time）
                    # 先确保 entities 都存在
                    for ent in entities:
                        ent_id = canonical_entity_id(ent)
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO entities(entity_id, name, first_seen, last_seen, sources_json, original_forms_json)
                            VALUES(?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ent_id,
                                ent,
                                base_ts,
                                base_ts,
                                json.dumps(sources, ensure_ascii=False),
                                json.dumps([ent], ensure_ascii=False),
                            ),
                        )

                    # 获取事件最终时间信息（用于元组 time）
                    evt_row = conn.execute(
                        "SELECT event_start_time, reported_at, first_seen FROM events WHERE event_id=?",
                        (event_id,),
                    ).fetchone()
                    evt_time = _choose_event_time(
                        str(evt_row["event_start_time"] or ""),
                        str(evt_row["reported_at"] or ""),
                        str(evt_row["first_seen"] or ""),
                    )

                    # participants upsert
                    for ent in entities:
                        roles = []
                        if isinstance(entity_roles, dict):
                            r = entity_roles.get(ent, [])
                            if isinstance(r, str):
                                roles = [r]
                            elif isinstance(r, list):
                                roles = [x for x in r if isinstance(x, str) and x.strip()]
                        conn.execute(
                            """
                            INSERT INTO participants(event_id, entity_id, roles_json, time, reported_at)
                            VALUES(?, ?, ?, ?, ?)
                            ON CONFLICT(event_id, entity_id) DO UPDATE SET
                                roles_json=excluded.roles_json,
                                time=excluded.time,
                                reported_at=excluded.reported_at
                            """,
                            (
                                event_id,
                                canonical_entity_id(ent),
                                json.dumps(roles, ensure_ascii=False),
                                evt_time,
                                base_ts,
                            ),
                        )

                    # relations upsert
                    for rel in relations:
                        if not isinstance(rel, dict):
                            continue
                        s = str(rel.get("subject") or "").strip()
                        p = str(rel.get("predicate") or "").strip()
                        o = str(rel.get("object") or "").strip()
                        if not s or not p or not o:
                            continue
                        # 关系两端实体可能不在该事件 entities 列表里（历史数据/抽取不一致），但为了满足外键与“每条元组带时间”，
                        # 这里保证 subject/object 至少在 entities 表中存在（最小记录）。
                        for name in (s, o):
                            ent_id = canonical_entity_id(name)
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO entities(entity_id, name, first_seen, last_seen, sources_json, original_forms_json)
                                VALUES(?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    ent_id,
                                    name,
                                    base_ts,
                                    base_ts,
                                    json.dumps(sources, ensure_ascii=False),
                                    json.dumps([name], ensure_ascii=False),
                                ),
                            )
                        # 关系证据统一为 list[str]
                        ev = rel.get("evidence", [])
                        if isinstance(ev, str):
                            ev_list = [ev.strip()] if ev.strip() else []
                        elif isinstance(ev, list):
                            ev_list = [x.strip() for x in ev if isinstance(x, str) and x.strip()]
                        else:
                            ev_list = []
                        conn.execute(
                            """
                            INSERT INTO relations(event_id, subject_entity_id, predicate, object_entity_id, time, reported_at, evidence_json)
                            VALUES(?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(event_id, subject_entity_id, predicate, object_entity_id) DO UPDATE SET
                                time=excluded.time,
                                reported_at=excluded.reported_at,
                                evidence_json=excluded.evidence_json
                            """,
                            (
                                event_id,
                                canonical_entity_id(s),
                                p,
                                canonical_entity_id(o),
                                evt_time,
                                base_ts,
                                json.dumps(ev_list, ensure_ascii=False),
                            ),
                        )

                conn.commit()
            finally:
                conn.close()

    # -------------------------
    # EXPORT (compat JSON)
    # -------------------------

    def export_entities_json(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                out: Dict[str, Any] = {}
                rows = conn.execute(
                    """
                    SELECT
                        e.name AS name,
                        e.first_seen AS first_seen,
                        e.sources_json AS sources_json,
                        e.original_forms_json AS original_forms_json,
                        COUNT(DISTINCT p.event_id) AS count
                    FROM entities e
                    LEFT JOIN participants p ON p.entity_id = e.entity_id
                    GROUP BY e.entity_id
                    """
                ).fetchall()
                for r in rows:
                    name = str(r["name"])
                    try:
                        sources = _norm_source_list(json.loads(r["sources_json"] or "[]"))
                    except Exception:
                        sources = []
                    try:
                        forms = json.loads(r["original_forms_json"] or "[]")
                        if not isinstance(forms, list):
                            forms = []
                    except Exception:
                        forms = []
                    out[name] = {
                        "first_seen": str(r["first_seen"] or ""),
                        "sources": sources,
                        "original_forms": [x for x in forms if isinstance(x, str) and x.strip()],
                        "count": int(r["count"] or 0),
                    }
                return out
            finally:
                conn.close()

    def get_entity_record_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        n = (name or "").strip()
        if not n:
            return None
        ent_id = canonical_entity_id(n)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT name, first_seen, last_seen, sources_json, original_forms_json FROM entities WHERE entity_id=?",
                    (ent_id,),
                ).fetchone()
                if row is None:
                    return None
                try:
                    sources = _norm_source_list(json.loads(row["sources_json"] or "[]"))
                except Exception:
                    sources = []
                try:
                    forms = json.loads(row["original_forms_json"] or "[]")
                    if not isinstance(forms, list):
                        forms = []
                except Exception:
                    forms = []
                return {
                    "name": str(row["name"] or n),
                    "first_seen": str(row["first_seen"] or ""),
                    "last_seen": str(row["last_seen"] or ""),
                    "sources": sources,
                    "original_forms": [x for x in forms if isinstance(x, str) and x.strip()],
                }
            finally:
                conn.close()

    def get_entity_event_samples(self, name: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        """
        给 LLM 审查提供上下文：该实体参与过的事件样本（含 time/summary）
        """
        n = (name or "").strip()
        if not n:
            return []
        ent_id = canonical_entity_id(n)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT p.time AS time, e.abstract AS abstract, e.event_summary AS event_summary
                    FROM participants p
                    JOIN events e ON e.event_id = p.event_id
                    WHERE p.entity_id=?
                    ORDER BY p.time DESC
                    LIMIT ?
                    """,
                    (ent_id, int(limit)),
                ).fetchall()
                out = []
                for r in rows:
                    out.append(
                        {
                            "time": str(r["time"] or ""),
                            "abstract": str(r["abstract"] or ""),
                            "event_summary": str(r["event_summary"] or ""),
                        }
                    )
                return out
            finally:
                conn.close()

    def export_abstract_map_json(self) -> Dict[str, Any]:
        """
        兼容 `data/abstract_to_event_map.json` 的结构：
        - 仍以 abstract 作为 key（便于旧 UI/逻辑）
        - 但内部会补充 event_id，并保证 relations 每条都有 time
        """
        with self._lock:
            conn = self._connect()
            try:
                out: Dict[str, Any] = {}
                events = conn.execute(
                    """
                    SELECT event_id, abstract, event_summary, event_types_json,
                           event_start_time, event_start_time_text, event_start_time_precision,
                           reported_at, first_seen, sources_json
                    FROM events
                    """
                ).fetchall()

                # alias abstracts（同一个 event_id 可能对应多个 abstract）
                alias_rows = conn.execute(
                    "SELECT abstract, event_id FROM event_aliases"
                ).fetchall()
                alias_to_evt = {str(r["abstract"]): str(r["event_id"]) for r in alias_rows}

                # 预取 participants/relations
                parts = conn.execute(
                    "SELECT event_id, entity_id, roles_json, time, reported_at FROM participants"
                ).fetchall()
                rels = conn.execute(
                    "SELECT event_id, subject_entity_id, predicate, object_entity_id, time, reported_at, evidence_json FROM relations"
                ).fetchall()
                # entity_id -> name
                ent_name = {
                    str(r["entity_id"]): str(r["name"])
                    for r in conn.execute("SELECT entity_id, name FROM entities").fetchall()
                }

                parts_by_evt: Dict[str, List[sqlite3.Row]] = {}
                for p in parts:
                    parts_by_evt.setdefault(str(p["event_id"]), []).append(p)
                rels_by_evt: Dict[str, List[sqlite3.Row]] = {}
                for r in rels:
                    rels_by_evt.setdefault(str(r["event_id"]), []).append(r)

                for e in events:
                    evt_id = str(e["event_id"])
                    abstract = str(e["abstract"])
                    try:
                        types = json.loads(e["event_types_json"] or "[]")
                        if not isinstance(types, list):
                            types = []
                    except Exception:
                        types = []
                    try:
                        sources = _norm_source_list(json.loads(e["sources_json"] or "[]"))
                    except Exception:
                        sources = []

                    # participants -> entities + entity_roles
                    entities: List[str] = []
                    roles_map: Dict[str, List[str]] = {}
                    for p in parts_by_evt.get(evt_id, []):
                        name = ent_name.get(str(p["entity_id"]), "")
                        if not name:
                            continue
                        entities.append(name)
                        try:
                            roles = json.loads(p["roles_json"] or "[]")
                            if not isinstance(roles, list):
                                roles = []
                        except Exception:
                            roles = []
                        roles_map[name] = [x for x in roles if isinstance(x, str) and x.strip()]

                    # relations -> triples with time
                    relations_out: List[Dict[str, Any]] = []
                    for r in rels_by_evt.get(evt_id, []):
                        s = ent_name.get(str(r["subject_entity_id"]), "")
                        o = ent_name.get(str(r["object_entity_id"]), "")
                        if not s or not o:
                            continue
                        try:
                            ev = json.loads(r["evidence_json"] or "[]")
                            if not isinstance(ev, list):
                                ev = []
                        except Exception:
                            ev = []
                        relations_out.append(
                            {
                                "subject": s,
                                "predicate": str(r["predicate"] or ""),
                                "object": o,
                                "evidence": [x for x in ev if isinstance(x, str) and x.strip()],
                                "time": str(r["time"] or ""),
                                "reported_at": str(r["reported_at"] or ""),
                            }
                        )

                    out[abstract] = {
                        "event_id": evt_id,
                        "entities": entities,
                        "event_summary": str(e["event_summary"] or ""),
                        "event_types": [x for x in types if isinstance(x, str) and x.strip()],
                        "entity_roles": roles_map,
                        "relations": relations_out,
                        "event_start_time": str(e["event_start_time"] or ""),
                        "event_start_time_text": str(e["event_start_time_text"] or ""),
                        "event_start_time_precision": str(e["event_start_time_precision"] or "unknown") or "unknown",
                        "reported_at": str(e["reported_at"] or ""),
                        "sources": sources,
                        "first_seen": str(e["first_seen"] or ""),
                    }

                # 额外输出 alias abstracts：让旧 abstract 也能映射到同一 event_id（兼容旧 UI/链接）
                for abs2, evt_id in alias_to_evt.items():
                    if abs2 in out:
                        continue
                    # 从 canonical event 复制一份（key 不同但 event_id 相同）
                    row = conn.execute(
                        """
                        SELECT event_id, abstract, event_summary, event_types_json,
                               event_start_time, event_start_time_text, event_start_time_precision,
                               reported_at, first_seen, sources_json
                        FROM events
                        WHERE event_id=?
                        """,
                        (evt_id,),
                    ).fetchone()
                    if row is None:
                        continue
                    # 递归复用：通过临时构造 events 列表的单条路径，避免重复实现过多逻辑
                    tmp_evt = dict(row)
                    tmp_evt["abstract"] = abs2
                    # 复用上面的构造逻辑：最简单是直接走已有 out[row["abstract"]] 的复制
                    canonical_abs = str(row["abstract"] or "")
                    base = out.get(canonical_abs)
                    if isinstance(base, dict):
                        out[abs2] = dict(base)
                        out[abs2]["event_id"] = evt_id
                    else:
                        # 极端兜底：只放最小字段
                        out[abs2] = {"event_id": evt_id, "entities": [], "event_summary": "", "event_types": [], "entity_roles": {}, "relations": [], "event_start_time": "", "event_start_time_text": "", "event_start_time_precision": "unknown", "reported_at": "", "sources": [], "first_seen": ""}
                return out
            finally:
                conn.close()

    def export_compat_json_files(self) -> None:
        entities = self.export_entities_json()
        abstract_map = self.export_abstract_map_json()
        _tools.ENTITIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        _tools.ABSTRACT_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        _tools.ENTITIES_FILE.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
        _tools.ABSTRACT_MAP_FILE.write_text(json.dumps(abstract_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------------------------
    # Processed IDs APIs
    # -------------------------

    def get_processed_ids(self) -> Set[str]:
        """
        获取所有已处理的新闻ID
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT global_id FROM processed_ids").fetchall()
                return {str(row["global_id"]) for row in rows}
            finally:
                conn.close()

    def add_processed_id(self, global_id: str, source: str, news_id: str) -> bool:
        """
        添加已处理的新闻ID
        """
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO processed_ids(global_id, source, news_id, created_at) VALUES(?, ?, ?, ?)",
                    (global_id, source, news_id, now)
                )
                conn.commit()
                return True
            except Exception as e:
                print(f"添加已处理ID失败: {e}")
                return False
            finally:
                conn.close()

    def add_processed_ids(self, ids: List[Tuple[str, str, str]]) -> int:
        """
        批量添加已处理的新闻ID
        ids: [(global_id, source, news_id), ...]
        返回实际插入的行数
        """
        if not ids:
            return 0
            
        now = _utc_now_iso()
        inserted_count = 0
        with self._lock:
            conn = self._connect()
            try:
                for global_id, source, news_id in ids:
                    cursor = conn.execute(
                        "INSERT OR IGNORE INTO processed_ids(global_id, source, news_id, created_at) VALUES(?, ?, ?, ?)",
                        (global_id, source, news_id, now)
                    )
                    # 如果插入了一行，则计数增加
                    if cursor.rowcount > 0:
                        inserted_count += cursor.rowcount
                conn.commit()
                return inserted_count
            except Exception as e:
                print(f"批量添加已处理ID失败: {e}")
                return 0
            finally:
                conn.close()

    # -------------------------
    # Apply Decisions (实体合并执行器)
    # -------------------------

    def _merge_json_lists_unique(self, a: Any, b: Any) -> List[str]:
        out: List[str] = []
        for x in (a or []):
            if isinstance(x, str) and x.strip() and x not in out:
                out.append(x.strip())
        for x in (b or []):
            if isinstance(x, str) and x.strip() and x not in out:
                out.append(x.strip())
        return out

    def merge_entities(self, from_entity_id: str, to_entity_id: str, *, reason: str = "", decision_input_hash: str = "") -> Dict[str, Any]:
        """
        合并实体（from -> to）：
        - 更新 participants/relations 外键
        - 合并 to 实体的 sources/original_forms/时间边界
        - 记录 entity_redirects + entity_aliases
        - 删除 from 实体
        """
        if not from_entity_id or not to_entity_id or from_entity_id == to_entity_id:
            return {"status": "skip"}

        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                src = conn.execute("SELECT * FROM entities WHERE entity_id=?", (from_entity_id,)).fetchone()
                dst = conn.execute("SELECT * FROM entities WHERE entity_id=?", (to_entity_id,)).fetchone()
                if src is None or dst is None:
                    return {"status": "missing"}

                src_name = str(src["name"])
                dst_name = str(dst["name"])

                # 合并实体属性（sources/forms/first_seen/last_seen）
                try:
                    src_sources = _norm_source_list(json.loads(src["sources_json"] or "[]"))
                except Exception:
                    src_sources = []
                try:
                    dst_sources = _norm_source_list(json.loads(dst["sources_json"] or "[]"))
                except Exception:
                    dst_sources = []
                merged_sources = _norm_source_list(dst_sources + src_sources)

                try:
                    src_forms = json.loads(src["original_forms_json"] or "[]")
                except Exception:
                    src_forms = []
                try:
                    dst_forms = json.loads(dst["original_forms_json"] or "[]")
                except Exception:
                    dst_forms = []
                merged_forms = self._merge_json_lists_unique(dst_forms, src_forms)
                # 保证 src_name 作为 alias 不丢
                if src_name and src_name not in merged_forms:
                    merged_forms.append(src_name)

                first_seen = min(str(src["first_seen"]), str(dst["first_seen"])) if src["first_seen"] and dst["first_seen"] else (str(dst["first_seen"]) or str(src["first_seen"]) or now)
                last_seen = max(str(src["last_seen"]), str(dst["last_seen"])) if src["last_seen"] and dst["last_seen"] else (str(dst["last_seen"]) or str(src["last_seen"]) or now)

                conn.execute(
                    """
                    UPDATE entities
                    SET first_seen=?, last_seen=?, sources_json=?, original_forms_json=?
                    WHERE entity_id=?
                    """,
                    (
                        first_seen,
                        last_seen,
                        json.dumps(merged_sources, ensure_ascii=False),
                        json.dumps(merged_forms, ensure_ascii=False),
                        to_entity_id,
                    ),
                )

                # participants：from -> to（可能产生 UNIQUE(event_id,entity_id) 冲突，需合并 roles）
                rows = conn.execute(
                    "SELECT event_id, roles_json, time, reported_at FROM participants WHERE entity_id=?",
                    (from_entity_id,),
                ).fetchall()
                for r in rows:
                    event_id = str(r["event_id"])
                    try:
                        roles_from = json.loads(r["roles_json"] or "[]")
                        if not isinstance(roles_from, list):
                            roles_from = []
                    except Exception:
                        roles_from = []
                    existing = conn.execute(
                        "SELECT roles_json, time, reported_at FROM participants WHERE event_id=? AND entity_id=?",
                        (event_id, to_entity_id),
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            "UPDATE participants SET entity_id=? WHERE event_id=? AND entity_id=?",
                            (to_entity_id, event_id, from_entity_id),
                        )
                    else:
                        try:
                            roles_to = json.loads(existing["roles_json"] or "[]")
                            if not isinstance(roles_to, list):
                                roles_to = []
                        except Exception:
                            roles_to = []
                        merged_roles = self._merge_json_lists_unique(roles_to, roles_from)
                        # time/reported_at：取较早的（更接近起点）
                        time_val = min(str(existing["time"]), str(r["time"])) if existing["time"] and r["time"] else (str(existing["time"]) or str(r["time"]) or now)
                        rep_val = min(str(existing["reported_at"]), str(r["reported_at"])) if existing["reported_at"] and r["reported_at"] else (str(existing["reported_at"]) or str(r["reported_at"]) or now)
                        conn.execute(
                            """
                            UPDATE participants
                            SET roles_json=?, time=?, reported_at=?
                            WHERE event_id=? AND entity_id=?
                            """,
                            (json.dumps(merged_roles, ensure_ascii=False), time_val, rep_val, event_id, to_entity_id),
                        )
                        conn.execute(
                            "DELETE FROM participants WHERE event_id=? AND entity_id=?",
                            (event_id, from_entity_id),
                        )

                # relations：subject/object 替换（可能产生 UNIQUE 冲突，需合并 evidence）
                rel_rows = conn.execute(
                    """
                    SELECT id, event_id, subject_entity_id, predicate, object_entity_id, time, reported_at, evidence_json
                    FROM relations
                    WHERE subject_entity_id=? OR object_entity_id=?
                    """,
                    (from_entity_id, from_entity_id),
                ).fetchall()

                for rr in rel_rows:
                    rid = int(rr["id"])
                    event_id = str(rr["event_id"])
                    subj = to_entity_id if str(rr["subject_entity_id"]) == from_entity_id else str(rr["subject_entity_id"])
                    obj = to_entity_id if str(rr["object_entity_id"]) == from_entity_id else str(rr["object_entity_id"])
                    pred = str(rr["predicate"])
                    tval = str(rr["time"] or now)
                    rval = str(rr["reported_at"] or now)
                    try:
                        ev_from = json.loads(rr["evidence_json"] or "[]")
                        if not isinstance(ev_from, list):
                            ev_from = []
                    except Exception:
                        ev_from = []

                    exist2 = conn.execute(
                        """
                        SELECT id, evidence_json, time, reported_at
                        FROM relations
                        WHERE event_id=? AND subject_entity_id=? AND predicate=? AND object_entity_id=?
                        """,
                        (event_id, subj, pred, obj),
                    ).fetchone()
                    if exist2 is None:
                        conn.execute(
                            """
                            UPDATE relations
                            SET subject_entity_id=?, object_entity_id=?
                            WHERE id=?
                            """,
                            (subj, obj, rid),
                        )
                    else:
                        try:
                            ev_to = json.loads(exist2["evidence_json"] or "[]")
                            if not isinstance(ev_to, list):
                                ev_to = []
                        except Exception:
                            ev_to = []
                        merged_ev = self._merge_json_lists_unique(ev_to, ev_from)
                        time2 = min(str(exist2["time"]), tval) if exist2["time"] and tval else (str(exist2["time"]) or tval or now)
                        rep2 = min(str(exist2["reported_at"]), rval) if exist2["reported_at"] and rval else (str(exist2["reported_at"]) or rval or now)
                        conn.execute(
                            "UPDATE relations SET evidence_json=?, time=?, reported_at=? WHERE id=?",
                            (json.dumps(merged_ev, ensure_ascii=False), time2, rep2, int(exist2["id"])),
                        )
                        conn.execute("DELETE FROM relations WHERE id=?", (rid,))

                # 记录 alias 与 redirect
                if src_name:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO entity_aliases(alias, entity_id, confidence, decision_input_hash, created_at)
                        VALUES(?, ?, ?, ?, ?)
                        """,
                        (src_name, to_entity_id, 1.0, decision_input_hash or "", now),
                    )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO entity_redirects(from_entity_id, to_entity_id, reason, decision_input_hash, created_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (from_entity_id, to_entity_id, reason or "", decision_input_hash or "", now),
                )

                # mentions：from -> to（保留历史但更新 resolved 指向，便于后续查询/投影）
                try:
                    conn.execute(
                        "UPDATE entity_mentions SET resolved_entity_id=? WHERE resolved_entity_id=?",
                        (to_entity_id, from_entity_id),
                    )
                except Exception:
                    pass

                # 删除源实体
                conn.execute("DELETE FROM entities WHERE entity_id=?", (from_entity_id,))

                conn.commit()
                return {"status": "merged", "from": src_name, "to": dst_name}
            finally:
                conn.close()

    def merge_events(self, from_event_id: str, to_event_id: str, *, reason: str = "", decision_input_hash: str = "") -> Dict[str, Any]:
        """
        合并事件（from -> to）：
        - participants/relations/event_edges 的 event_id 迁移
        - 合并 event_types/sources/first_seen/last_seen/summary
        - 记录 event_redirects + event_aliases（保留旧 abstract 作为 alias）
        - 删除 from 事件
        """
        if not from_event_id or not to_event_id or from_event_id == to_event_id:
            return {"status": "skip"}
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                src = conn.execute("SELECT * FROM events WHERE event_id=?", (from_event_id,)).fetchone()
                dst = conn.execute("SELECT * FROM events WHERE event_id=?", (to_event_id,)).fetchone()
                if src is None or dst is None:
                    return {"status": "missing"}

                src_abs = str(src["abstract"] or "")
                dst_abs = str(dst["abstract"] or "")

                # 合并 types
                try:
                    src_types = json.loads(src["event_types_json"] or "[]")
                    if not isinstance(src_types, list):
                        src_types = []
                except Exception:
                    src_types = []
                try:
                    dst_types = json.loads(dst["event_types_json"] or "[]")
                    if not isinstance(dst_types, list):
                        dst_types = []
                except Exception:
                    dst_types = []
                merged_types = self._merge_json_lists_unique(dst_types, src_types)

                # 合并 sources
                try:
                    src_sources = _norm_source_list(json.loads(src["sources_json"] or "[]"))
                except Exception:
                    src_sources = []
                try:
                    dst_sources = _norm_source_list(json.loads(dst["sources_json"] or "[]"))
                except Exception:
                    dst_sources = []
                merged_sources = _norm_source_list(dst_sources + src_sources)

                # summary：保留更长的
                src_summary = str(src["event_summary"] or "")
                dst_summary = str(dst["event_summary"] or "")
                best_summary = dst_summary
                if src_summary and len(src_summary) > len(dst_summary or ""):
                    best_summary = src_summary

                first_seen = min(str(src["first_seen"]), str(dst["first_seen"])) if src["first_seen"] and dst["first_seen"] else (str(dst["first_seen"]) or str(src["first_seen"]) or now)
                last_seen = max(str(src["last_seen"]), str(dst["last_seen"])) if src["last_seen"] and dst["last_seen"] else (str(dst["last_seen"]) or str(src["last_seen"]) or now)

                conn.execute(
                    """
                    UPDATE events
                    SET event_summary=?, event_types_json=?, sources_json=?, first_seen=?, last_seen=?
                    WHERE event_id=?
                    """,
                    (
                        best_summary or "",
                        json.dumps(merged_types, ensure_ascii=False),
                        json.dumps(merged_sources, ensure_ascii=False),
                        first_seen,
                        last_seen,
                        to_event_id,
                    ),
                )

                # participants/relations 迁移到新 event_id（冲突则合并）
                # participants UNIQUE(event_id,entity_id)
                p_rows = conn.execute(
                    "SELECT event_id, entity_id, roles_json, time, reported_at FROM participants WHERE event_id=?",
                    (from_event_id,),
                ).fetchall()
                for p in p_rows:
                    ent_id = str(p["entity_id"])
                    existing = conn.execute(
                        "SELECT roles_json, time, reported_at FROM participants WHERE event_id=? AND entity_id=?",
                        (to_event_id, ent_id),
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            "UPDATE participants SET event_id=? WHERE event_id=? AND entity_id=?",
                            (to_event_id, from_event_id, ent_id),
                        )
                    else:
                        try:
                            a_roles = json.loads(existing["roles_json"] or "[]")
                            if not isinstance(a_roles, list):
                                a_roles = []
                        except Exception:
                            a_roles = []
                        try:
                            b_roles = json.loads(p["roles_json"] or "[]")
                            if not isinstance(b_roles, list):
                                b_roles = []
                        except Exception:
                            b_roles = []
                        merged_roles = self._merge_json_lists_unique(a_roles, b_roles)
                        time_val = min(str(existing["time"]), str(p["time"])) if existing["time"] and p["time"] else (str(existing["time"]) or str(p["time"]) or now)
                        rep_val = min(str(existing["reported_at"]), str(p["reported_at"])) if existing["reported_at"] and p["reported_at"] else (str(existing["reported_at"]) or str(p["reported_at"]) or now)
                        conn.execute(
                            "UPDATE participants SET roles_json=?, time=?, reported_at=? WHERE event_id=? AND entity_id=?",
                            (json.dumps(merged_roles, ensure_ascii=False), time_val, rep_val, to_event_id, ent_id),
                        )
                        conn.execute(
                            "DELETE FROM participants WHERE event_id=? AND entity_id=?",
                            (from_event_id, ent_id),
                        )

                # relations UNIQUE(event_id, subj, pred, obj)
                r_rows = conn.execute(
                    "SELECT id, subject_entity_id, predicate, object_entity_id, time, reported_at, evidence_json FROM relations WHERE event_id=?",
                    (from_event_id,),
                ).fetchall()
                for rr in r_rows:
                    rid = int(rr["id"])
                    subj = str(rr["subject_entity_id"])
                    pred = str(rr["predicate"])
                    obj = str(rr["object_entity_id"])
                    tval = str(rr["time"] or now)
                    rval = str(rr["reported_at"] or now)
                    try:
                        ev_from = json.loads(rr["evidence_json"] or "[]")
                        if not isinstance(ev_from, list):
                            ev_from = []
                    except Exception:
                        ev_from = []
                    exist2 = conn.execute(
                        "SELECT id, evidence_json, time, reported_at FROM relations WHERE event_id=? AND subject_entity_id=? AND predicate=? AND object_entity_id=?",
                        (to_event_id, subj, pred, obj),
                    ).fetchone()
                    if exist2 is None:
                        conn.execute("UPDATE relations SET event_id=? WHERE id=?", (to_event_id, rid))
                    else:
                        try:
                            ev_to = json.loads(exist2["evidence_json"] or "[]")
                            if not isinstance(ev_to, list):
                                ev_to = []
                        except Exception:
                            ev_to = []
                        merged_ev = self._merge_json_lists_unique(ev_to, ev_from)
                        time2 = min(str(exist2["time"]), tval) if exist2["time"] and tval else (str(exist2["time"]) or tval or now)
                        rep2 = min(str(exist2["reported_at"]), rval) if exist2["reported_at"] and rval else (str(exist2["reported_at"]) or rval or now)
                        conn.execute(
                            "UPDATE relations SET evidence_json=?, time=?, reported_at=? WHERE id=?",
                            (json.dumps(merged_ev, ensure_ascii=False), time2, rep2, int(exist2["id"])),
                        )
                        conn.execute("DELETE FROM relations WHERE id=?", (rid,))

                # event_edges 迁移（from->to）
                conn.execute("UPDATE event_edges SET from_event_id=? WHERE from_event_id=?", (to_event_id, from_event_id))
                conn.execute("UPDATE event_edges SET to_event_id=? WHERE to_event_id=?", (to_event_id, from_event_id))

                # 记录 alias/redirect（保留 src_abs）
                if src_abs:
                    conn.execute(
                        "INSERT OR REPLACE INTO event_aliases(abstract, event_id, confidence, decision_input_hash, created_at) VALUES(?, ?, ?, ?, ?)",
                        (src_abs, to_event_id, 1.0, decision_input_hash or "", now),
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO event_redirects(from_event_id, to_event_id, reason, decision_input_hash, created_at) VALUES(?, ?, ?, ?, ?)",
                    (from_event_id, to_event_id, reason or "", decision_input_hash or "", now),
                )

                # 删除源事件
                conn.execute("DELETE FROM events WHERE event_id=?", (from_event_id,))

                # mentions：from -> to（保留历史但更新 resolved 指向，便于后续查询/投影）
                try:
                    conn.execute(
                        "UPDATE event_mentions SET resolved_event_id=? WHERE resolved_event_id=?",
                        (to_event_id, from_event_id),
                    )
                except Exception:
                    pass
                conn.commit()
                return {"status": "merged", "from": src_abs, "to": dst_abs}
            finally:
                conn.close()

    def _resolve_redirect_with_conn(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        from_col: str,
        to_col: str,
        start_id: str,
        max_hops: int,
    ) -> str:
        cur = (start_id or "").strip()
        if not cur:
            return ""
        seen = set()
        for _ in range(int(max_hops) if int(max_hops) > 0 else 20):
            if cur in seen:
                break
            seen.add(cur)
            row = conn.execute(
                f"SELECT {to_col} FROM {table} WHERE {from_col}=?",
                (cur,),
            ).fetchone()
            if row is None:
                break
            nxt = str(row[to_col] or "").strip()
            if not nxt or nxt == cur:
                break
            cur = nxt
        return cur

    def resolve_event_id(self, event_id: str, *, max_hops: int = 20) -> str:
        with self._lock:
            conn = self._connect()
            try:
                return self._resolve_redirect_with_conn(
                    conn,
                    table="event_redirects",
                    from_col="from_event_id",
                    to_col="to_event_id",
                    start_id=event_id,
                    max_hops=max_hops,
                )
            finally:
                conn.close()

    def resolve_event_id_by_abstract(self, abstract: str, *, max_hops: int = 20) -> str:
        a = (abstract or "").strip()
        if not a:
            return ""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT event_id FROM events WHERE abstract=?",
                    (a,),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        "SELECT event_id FROM event_aliases WHERE abstract=?",
                        (a,),
                    ).fetchone()
                if row is None:
                    return ""
                return self._resolve_redirect_with_conn(
                    conn,
                    table="event_redirects",
                    from_col="from_event_id",
                    to_col="to_event_id",
                    start_id=str(row["event_id"] or ""),
                    max_hops=max_hops,
                )
            finally:
                conn.close()

    # -------------------------
    # News-Event Mapping APIs
    # -------------------------

    def add_news_event_mapping(self, news_global_id: str, event_id: str) -> bool:
        """
        添加新闻ID到事件ID的映射
        """
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO news_event_mapping(news_global_id, event_id, created_at) VALUES(?, ?, ?)",
                    (news_global_id, event_id, now)
                )
                conn.commit()
                return True
            except Exception as e:
                print(f"添加新闻事件映射失败: {e}")
                return False
            finally:
                conn.close()

    def add_news_event_mappings(self, mappings: List[Tuple[str, str]]) -> int:
        """
        批量添加新闻ID到事件ID的映射
        mappings: [(news_global_id, event_id), ...]
        返回实际插入的行数
        """
        if not mappings:
            return 0
            
        now = _utc_now_iso()
        inserted_count = 0
        with self._lock:
            conn = self._connect()
            try:
                for news_global_id, event_id in mappings:
                    cursor = conn.execute(
                        "INSERT OR IGNORE INTO news_event_mapping(news_global_id, event_id, created_at) VALUES(?, ?, ?)",
                        (news_global_id, event_id, now)
                    )
                    # 如果插入了一行，则计数增加
                    if cursor.rowcount > 0:
                        inserted_count += cursor.rowcount
                conn.commit()
                return inserted_count
            except Exception as e:
                print(f"批量添加新闻事件映射失败: {e}")
                return 0
            finally:
                conn.close()

    def get_events_by_news_id(self, news_global_id: str) -> List[str]:
        """
        根据新闻ID获取关联的所有事件ID
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT event_id FROM news_event_mapping WHERE news_global_id=?",
                    (news_global_id,)
                ).fetchall()
                return [str(row["event_id"]) for row in rows]
            finally:
                conn.close()

    def get_news_by_event_id(self, event_id: str) -> List[str]:
        """
        根据事件ID获取关联的所有新闻ID
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT news_global_id FROM news_event_mapping WHERE event_id=?",
                    (event_id,)
                ).fetchall()
                return [str(row["news_global_id"]) for row in rows]
            finally:
                conn.close()


_store_singleton: Optional[SQLiteStore] = None
_store_lock = threading.Lock()


def get_store() -> SQLiteStore:
    global _store_singleton
    if _store_singleton is None:
        with _store_lock:
            if _store_singleton is None:
                _store_singleton = SQLiteStore()
    return _store_singleton

