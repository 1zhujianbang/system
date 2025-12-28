"""
适配器层 - Neo4j 图存储实现

实现 GraphStore 端口，提供 Neo4j 数据库访问。
"""
import logging
import os
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase, Driver

from ...ports.graph_store import GraphStore
from ...infra.config import get_config_manager

logger = logging.getLogger(__name__)

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
    from ...adapters.sqlite.store import _norm_source_list as _sqlite_norm_sources

    return _sqlite_norm_sources(sources)


def _choose_event_time(event_start_time: str, reported_at: str, first_seen: str) -> str:
    from ...adapters.sqlite.store import _choose_event_time as _sqlite_choose_time

    return _sqlite_choose_time(event_start_time, reported_at, first_seen)


def canonical_entity_id(entity_name: str) -> str:
    from ...adapters.sqlite.store import canonical_entity_id as _canonical_entity_id

    return _canonical_entity_id(entity_name)


def canonical_event_id(abstract: str) -> str:
    from ...adapters.sqlite.store import canonical_event_id as _canonical_event_id

    return _canonical_event_id(abstract)


class Neo4jAdapter(GraphStore):
    """Neo4j 图存储适配器"""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        初始化 Neo4j 适配器
        
        如果参数未提供，尝试从环境变量或配置管理器加载。
        """
        config = get_config_manager()
        
        self._uri = uri or os.getenv("NEO4J_URI") or config.get_config_value("neo4j.uri", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or config.get_config_value("neo4j.user", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD") or config.get_config_value("neo4j.password", "password")
        self._database = (
            (database or os.getenv("NEO4J_DATABASE") or config.get_config_value("neo4j.database", "")) or None
        )
        
        self._driver: Optional[Driver] = None
        self._lock = threading.RLock()
        self._connect()

    def _connect(self):
        """建立连接"""
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            # 验证连接
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self._uri}")
            self._ensure_schema()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {self._uri}: {e}")
            self._driver = None

    def is_available(self) -> bool:
        """检查图数据库是否可用"""
        if not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """关闭连接"""
        if self._driver:
            self._driver.close()
            logger.info("Closed Neo4j connection")

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行 Cypher 查询"""
        if not self._driver:
            self._connect()
            if not self._driver:
                raise ConnectionError("Neo4j driver is not available")

        params = params or {}
        try:
            session_kwargs = {"database": self._database} if self._database else {}
            with self._driver.session(**session_kwargs) as session:
                result = session.run(cypher, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}\nQuery: {cypher}\nParams: {params}")
            raise

    def execute_batch(self, operations: List[Dict[str, Any]]) -> None:
        """
        批量执行操作
        
        Args:
            operations: List of dicts with 'cypher' and 'params' keys
        """
        if not self._driver:
            self._connect()
            if not self._driver:
                raise ConnectionError("Neo4j driver is not available")
        
        if not operations:
            return

        try:
            session_kwargs = {"database": self._database} if self._database else {}
            with self._driver.session(**session_kwargs) as session:
                with session.begin_transaction() as tx:
                    for op in operations:
                        tx.run(op['cypher'], op.get('params', {}))
                    tx.commit()
            logger.info(f"Executed batch of {len(operations)} operations")
        except Exception as e:
            logger.error(f"Neo4j batch execution failed: {e}")
            raise

    def _ensure_schema(self) -> None:
        if not self._driver:
            return
        statements = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX event_abstract IF NOT EXISTS FOR (e:Event) ON (e.abstract)",
        ]
        try:
            session_kwargs = {"database": self._database} if self._database else {}
            with self._driver.session(**session_kwargs) as session:
                for s in statements:
                    session.run(s)
        except Exception:
            return

    def upsert_entities(
        self,
        entities: List[str],
        entities_original: List[str],
        *,
        source: Any,
        reported_at: Optional[str],
    ) -> None:
        base_ts = _norm_iso_ts(reported_at) or _utc_now_iso()
        sources = _norm_source_list([source])
        with self._lock:
            ops: List[Dict[str, Any]] = []
            for name, orig in zip(entities or [], entities_original or []):
                n = str(name or "").strip()
                if not n:
                    continue
                ent_id = canonical_entity_id(n)
                orig_s = str(orig or "").strip() or n
                ops.append(
                    {
                        "cypher": """
MERGE (e:Entity {entity_id: $entity_id})
ON CREATE SET
  e.name = $name,
  e.first_seen = $ts,
  e.last_seen = $ts,
  e.sources = $sources,
  e.original_forms = $original_forms
ON MATCH SET
  e.name = CASE WHEN $name <> '' THEN $name ELSE coalesce(e.name, '') END,
  e.first_seen = CASE WHEN coalesce(e.first_seen, '') = '' OR ($ts <> '' AND $ts < e.first_seen) THEN $ts ELSE e.first_seen END,
  e.last_seen = CASE WHEN coalesce(e.last_seen, '') = '' OR ($ts <> '' AND $ts > e.last_seen) THEN $ts ELSE e.last_seen END,
  e.sources = reduce(acc = coalesce(e.sources, []), x IN $sources | CASE WHEN any(y IN acc WHERE y.id = x.id) THEN acc ELSE acc + x END),
  e.original_forms = reduce(acc = coalesce(e.original_forms, []), x IN $original_forms | CASE WHEN x IN acc THEN acc ELSE acc + x END)
""",
                        "params": {
                            "entity_id": ent_id,
                            "name": n,
                            "ts": base_ts,
                            "sources": sources,
                            "original_forms": [orig_s],
                        },
                    }
                )
            self.execute_batch(ops)

    def upsert_events(self, extracted_events: List[Dict[str, Any]], *, source: Any, reported_at: Optional[str]) -> None:
        base_ts = _norm_iso_ts(reported_at) or _utc_now_iso()
        sources = _norm_source_list([source])
        with self._lock:
            ops: List[Dict[str, Any]] = []
            for item in extracted_events or []:
                if not isinstance(item, dict):
                    continue
                abstract = str(item.get("abstract") or "").strip()
                if not abstract:
                    continue

                event_id = canonical_event_id(abstract)
                event_summary = str(item.get("event_summary") or "").strip()
                event_types = item.get("event_types") if isinstance(item.get("event_types"), list) else []
                event_types = [x.strip() for x in event_types if isinstance(x, str) and x.strip()]

                entity_roles = item.get("entity_roles") if isinstance(item.get("entity_roles"), dict) else {}
                relations = item.get("relations") if isinstance(item.get("relations"), list) else []
                entities = item.get("entities") if isinstance(item.get("entities"), list) else []
                entities = [x for x in entities if isinstance(x, str) and x.strip()]

                event_start_time = _norm_iso_ts(item.get("event_start_time"))
                event_start_time_text = str(item.get("event_start_time_text") or "").strip()
                event_start_time_precision = str(item.get("event_start_time_precision") or "unknown").strip() or "unknown"

                evt_time = _choose_event_time(event_start_time, base_ts, base_ts)

                ops.append(
                    {
                        "cypher": """
MERGE (e:Event {event_id: $event_id})
ON CREATE SET
  e.abstract = $abstract,
  e.event_summary = $event_summary,
  e.event_types = $event_types,
  e.event_start_time = $event_start_time,
  e.event_start_time_text = $event_start_time_text,
  e.event_start_time_precision = $event_start_time_precision,
  e.reported_at = $reported_at,
  e.first_seen = $reported_at,
  e.last_seen = $reported_at,
  e.sources = $sources
ON MATCH SET
  e.abstract = CASE WHEN $abstract <> '' THEN $abstract ELSE coalesce(e.abstract, '') END,
  e.event_summary = CASE WHEN $event_summary <> '' AND (coalesce(e.event_summary, '') = '' OR size($event_summary) > size(e.event_summary)) THEN $event_summary ELSE coalesce(e.event_summary, '') END,
  e.event_types = CASE WHEN $event_types IS NOT NULL AND size($event_types) > 0 THEN $event_types ELSE coalesce(e.event_types, []) END,
  e.event_start_time = CASE WHEN coalesce(e.event_start_time, '') = '' AND $event_start_time <> '' THEN $event_start_time ELSE coalesce(e.event_start_time, '') END,
  e.event_start_time_text = CASE WHEN coalesce(e.event_start_time, '') = '' AND $event_start_time_text <> '' THEN $event_start_time_text ELSE coalesce(e.event_start_time_text, '') END,
  e.event_start_time_precision = CASE WHEN coalesce(e.event_start_time, '') = '' AND $event_start_time_precision <> '' THEN $event_start_time_precision ELSE coalesce(e.event_start_time_precision, 'unknown') END,
  e.reported_at = CASE WHEN coalesce(e.reported_at, '') = '' OR ($reported_at <> '' AND $reported_at < e.reported_at) THEN $reported_at ELSE e.reported_at END,
  e.first_seen = CASE WHEN coalesce(e.first_seen, '') = '' OR ($reported_at <> '' AND $reported_at < e.first_seen) THEN $reported_at ELSE e.first_seen END,
  e.last_seen = CASE WHEN coalesce(e.last_seen, '') = '' OR ($reported_at <> '' AND $reported_at > e.last_seen) THEN $reported_at ELSE e.last_seen END,
  e.sources = reduce(acc = coalesce(e.sources, []), x IN $sources | CASE WHEN any(y IN acc WHERE y.id = x.id) THEN acc ELSE acc + x END)
""",
                        "params": {
                            "event_id": event_id,
                            "abstract": abstract,
                            "event_summary": event_summary,
                            "event_types": event_types,
                            "event_start_time": event_start_time,
                            "event_start_time_text": event_start_time_text,
                            "event_start_time_precision": event_start_time_precision,
                            "reported_at": base_ts,
                            "sources": sources,
                        },
                    }
                )

                for ent in entities:
                    ent_id = canonical_entity_id(ent)
                    roles: List[str] = []
                    if isinstance(entity_roles, dict):
                        r = entity_roles.get(ent, [])
                        if isinstance(r, str) and r.strip():
                            roles = [r.strip()]
                        elif isinstance(r, list):
                            roles = [x.strip() for x in r if isinstance(x, str) and x.strip()]

                    ops.append(
                        {
                            "cypher": """
MERGE (e:Entity {entity_id: $entity_id})
ON CREATE SET
  e.name = $name,
  e.first_seen = $reported_at,
  e.last_seen = $reported_at,
  e.sources = $sources,
  e.original_forms = $original_forms
ON MATCH SET
  e.name = CASE WHEN $name <> '' THEN $name ELSE coalesce(e.name, '') END,
  e.first_seen = CASE WHEN coalesce(e.first_seen, '') = '' OR ($reported_at <> '' AND $reported_at < e.first_seen) THEN $reported_at ELSE e.first_seen END,
  e.last_seen = CASE WHEN coalesce(e.last_seen, '') = '' OR ($reported_at <> '' AND $reported_at > e.last_seen) THEN $reported_at ELSE e.last_seen END,
  e.sources = reduce(acc = coalesce(e.sources, []), x IN $sources | CASE WHEN any(y IN acc WHERE y.id = x.id) THEN acc ELSE acc + x END),
  e.original_forms = reduce(acc = coalesce(e.original_forms, []), x IN $original_forms | CASE WHEN x IN acc THEN acc ELSE acc + x END)
""",
                            "params": {
                                "entity_id": ent_id,
                                "name": ent,
                                "reported_at": base_ts,
                                "sources": sources,
                                "original_forms": [ent],
                            },
                        }
                    )

                    ops.append(
                        {
                            "cypher": """
MATCH (en:Entity {entity_id: $entity_id})
MATCH (ev:Event {event_id: $event_id})
MERGE (en)-[p:PARTICIPATED_IN]->(ev)
SET p.roles = $roles,
    p.time = $time,
    p.reported_at = $reported_at
""",
                            "params": {
                                "entity_id": ent_id,
                                "event_id": event_id,
                                "roles": roles,
                                "time": evt_time,
                                "reported_at": base_ts,
                            },
                        }
                    )

                for rel in relations:
                    if not isinstance(rel, dict):
                        continue
                    s = str(rel.get("subject") or "").strip()
                    p = str(rel.get("predicate") or "").strip()
                    o = str(rel.get("object") or "").strip()
                    if not s or not p or not o:
                        continue
                    ev = rel.get("evidence", [])
                    if isinstance(ev, str):
                        ev_list = [ev.strip()] if ev.strip() else []
                    elif isinstance(ev, list):
                        ev_list = [x.strip() for x in ev if isinstance(x, str) and x.strip()]
                    else:
                        ev_list = []

                    sid = canonical_entity_id(s)
                    oid = canonical_entity_id(o)
                    for _id, name in ((sid, s), (oid, o)):
                        ops.append(
                            {
                                "cypher": """
MERGE (e:Entity {entity_id: $entity_id})
ON CREATE SET
  e.name = $name,
  e.first_seen = $reported_at,
  e.last_seen = $reported_at,
  e.sources = $sources,
  e.original_forms = $original_forms
ON MATCH SET
  e.name = CASE WHEN $name <> '' THEN $name ELSE coalesce(e.name, '') END,
  e.first_seen = CASE WHEN coalesce(e.first_seen, '') = '' OR ($reported_at <> '' AND $reported_at < e.first_seen) THEN $reported_at ELSE e.first_seen END,
  e.last_seen = CASE WHEN coalesce(e.last_seen, '') = '' OR ($reported_at <> '' AND $reported_at > e.last_seen) THEN $reported_at ELSE e.last_seen END,
  e.sources = reduce(acc = coalesce(e.sources, []), x IN $sources | CASE WHEN any(y IN acc WHERE y.id = x.id) THEN acc ELSE acc + x END),
  e.original_forms = reduce(acc = coalesce(e.original_forms, []), x IN $original_forms | CASE WHEN x IN acc THEN acc ELSE acc + x END)
""",
                                "params": {
                                    "entity_id": _id,
                                    "name": name,
                                    "reported_at": base_ts,
                                    "sources": sources,
                                    "original_forms": [name],
                                },
                            }
                        )

                    ops.append(
                        {
                            "cypher": """
MATCH (s:Entity {entity_id: $sid})
MATCH (o:Entity {entity_id: $oid})
MERGE (s)-[r:RELATION {predicate: $predicate, event_id: $event_id}]->(o)
SET r.time = $time,
    r.reported_at = $reported_at,
    r.evidence = $evidence
""",
                            "params": {
                                "sid": sid,
                                "oid": oid,
                                "predicate": p,
                                "event_id": event_id,
                                "time": evt_time,
                                "reported_at": base_ts,
                                "evidence": ev_list,
                            },
                        }
                    )

            self.execute_batch(ops)

    def export_entities_json(self) -> Dict[str, Any]:
        rows = self.query(
            """
MATCH (e:Entity)
OPTIONAL MATCH (e)-[:PARTICIPATED_IN]->(ev:Event)
WITH e, count(DISTINCT ev.event_id) AS cnt
RETURN e.name AS name, e.first_seen AS first_seen, e.sources AS sources, e.original_forms AS original_forms, cnt AS count
"""
        )
        out: Dict[str, Any] = {}
        for r in rows:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            sources = r.get("sources") if isinstance(r.get("sources"), list) else []
            forms = r.get("original_forms") if isinstance(r.get("original_forms"), list) else []
            out[name] = {
                "first_seen": str(r.get("first_seen") or ""),
                "sources": sources,
                "original_forms": [x for x in forms if isinstance(x, str) and x.strip()],
                "count": int(r.get("count") or 0),
            }
        return out

    def export_abstract_map_json(self) -> Dict[str, Any]:
        evt_rows = self.query(
            """
MATCH (e:Event)
RETURN e.event_id AS event_id,
       e.abstract AS abstract,
       e.event_summary AS event_summary,
       e.event_types AS event_types,
       e.event_start_time AS event_start_time,
       e.event_start_time_text AS event_start_time_text,
       e.event_start_time_precision AS event_start_time_precision,
       e.reported_at AS reported_at,
       e.first_seen AS first_seen,
       e.sources AS sources
"""
        )
        parts_rows = self.query(
            """
MATCH (en:Entity)-[p:PARTICIPATED_IN]->(ev:Event)
RETURN ev.event_id AS event_id, en.name AS entity_name, p.roles AS roles, p.time AS time, p.reported_at AS reported_at
"""
        )
        rel_rows = self.query(
            """
MATCH (s:Entity)-[r:RELATION]->(o:Entity)
RETURN r.event_id AS event_id, s.name AS subject, r.predicate AS predicate, o.name AS object, r.time AS time, r.reported_at AS reported_at, r.evidence AS evidence
"""
        )

        parts_by_evt: Dict[str, List[Dict[str, Any]]] = {}
        for r in parts_rows:
            eid = str(r.get("event_id") or "")
            parts_by_evt.setdefault(eid, []).append(r)

        rels_by_evt: Dict[str, List[Dict[str, Any]]] = {}
        for r in rel_rows:
            eid = str(r.get("event_id") or "")
            rels_by_evt.setdefault(eid, []).append(r)

        out: Dict[str, Any] = {}
        for e in evt_rows:
            evt_id = str(e.get("event_id") or "")
            abstract = str(e.get("abstract") or "")
            if not abstract or not evt_id:
                continue
            types = e.get("event_types") if isinstance(e.get("event_types"), list) else []
            sources = e.get("sources") if isinstance(e.get("sources"), list) else []

            entities: List[str] = []
            roles_map: Dict[str, List[str]] = {}
            for p in parts_by_evt.get(evt_id, []):
                name = str(p.get("entity_name") or "").strip()
                if not name:
                    continue
                entities.append(name)
                roles = p.get("roles") if isinstance(p.get("roles"), list) else []
                roles_map[name] = [x for x in roles if isinstance(x, str) and x.strip()]

            relations_out: List[Dict[str, Any]] = []
            for r in rels_by_evt.get(evt_id, []):
                s = str(r.get("subject") or "").strip()
                o = str(r.get("object") or "").strip()
                if not s or not o:
                    continue
                ev = r.get("evidence") if isinstance(r.get("evidence"), list) else []
                relations_out.append(
                    {
                        "subject": s,
                        "predicate": str(r.get("predicate") or ""),
                        "object": o,
                        "evidence": [x for x in ev if isinstance(x, str) and x.strip()],
                        "time": str(r.get("time") or ""),
                        "reported_at": str(r.get("reported_at") or ""),
                    }
                )

            out[abstract] = {
                "event_id": evt_id,
                "entities": entities,
                "event_summary": str(e.get("event_summary") or ""),
                "event_types": [x for x in types if isinstance(x, str) and x.strip()],
                "entity_roles": roles_map,
                "relations": relations_out,
                "event_start_time": str(e.get("event_start_time") or ""),
                "event_start_time_text": str(e.get("event_start_time_text") or ""),
                "event_start_time_precision": str(e.get("event_start_time_precision") or "unknown") or "unknown",
                "reported_at": str(e.get("reported_at") or ""),
                "sources": sources,
                "first_seen": str(e.get("first_seen") or ""),
            }
        return out

    def fetch_entities(self) -> List[Dict[str, Any]]:
        rows = self.query(
            """
MATCH (e:Entity)
RETURN e.entity_id AS entity_id, e.name AS name, e.first_seen AS first_seen, e.last_seen AS last_seen
"""
        )
        return [dict(r) for r in rows]

    def fetch_events(self) -> List[Dict[str, Any]]:
        rows = self.query(
            """
MATCH (e:Event)
RETURN e.event_id AS event_id,
       e.abstract AS abstract,
       e.event_summary AS event_summary,
       e.event_types AS event_types,
       e.event_start_time AS event_start_time,
       e.reported_at AS reported_at,
       e.first_seen AS first_seen
"""
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            types = r.get("event_types") if isinstance(r.get("event_types"), list) else []
            out.append(
                {
                    "event_id": str(r.get("event_id") or ""),
                    "abstract": str(r.get("abstract") or ""),
                    "event_summary": str(r.get("event_summary") or ""),
                    "event_types_json": json.dumps([x for x in types if isinstance(x, str) and x.strip()], ensure_ascii=False),
                    "event_start_time": str(r.get("event_start_time") or ""),
                    "reported_at": str(r.get("reported_at") or ""),
                    "first_seen": str(r.get("first_seen") or ""),
                }
            )
        return out

    def fetch_participants_with_events(self) -> List[Dict[str, Any]]:
        rows = self.query(
            """
MATCH (en:Entity)-[p:PARTICIPATED_IN]->(e:Event)
RETURN p.event_id AS event_id,
       e.abstract AS abstract,
       e.event_summary AS event_summary,
       e.event_start_time AS event_start_time,
       e.reported_at AS evt_reported_at,
       e.first_seen AS first_seen,
       en.entity_id AS entity_id,
       p.roles AS roles,
       p.time AS time,
       p.reported_at AS reported_at
"""
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            roles = r.get("roles") if isinstance(r.get("roles"), list) else []
            out.append(
                {
                    "event_id": str(r.get("event_id") or ""),
                    "entity_id": str(r.get("entity_id") or ""),
                    "roles_json": json.dumps([x for x in roles if isinstance(x, str) and x.strip()], ensure_ascii=False),
                    "time": str(r.get("time") or ""),
                    "reported_at": str(r.get("reported_at") or ""),
                    "abstract": str(r.get("abstract") or ""),
                    "event_summary": str(r.get("event_summary") or ""),
                    "event_start_time": str(r.get("event_start_time") or ""),
                    "evt_reported_at": str(r.get("evt_reported_at") or ""),
                    "first_seen": str(r.get("first_seen") or ""),
                }
            )
        return out

    def fetch_relations(self) -> List[Dict[str, Any]]:
        rows = self.query(
            """
MATCH (s:Entity)-[r:RELATION]->(o:Entity)
RETURN r.event_id AS event_id,
       s.entity_id AS subject_entity_id,
       r.predicate AS predicate,
       o.entity_id AS object_entity_id,
       r.time AS time,
       r.reported_at AS reported_at,
       r.evidence AS evidence
"""
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            ev = r.get("evidence") if isinstance(r.get("evidence"), list) else []
            out.append(
                {
                    "event_id": str(r.get("event_id") or ""),
                    "subject_entity_id": str(r.get("subject_entity_id") or ""),
                    "predicate": str(r.get("predicate") or ""),
                    "object_entity_id": str(r.get("object_entity_id") or ""),
                    "time": str(r.get("time") or ""),
                    "reported_at": str(r.get("reported_at") or ""),
                    "evidence_json": json.dumps([x for x in ev if isinstance(x, str) and x.strip()], ensure_ascii=False),
                }
            )
        return out

    def fetch_event_edges(self) -> List[Dict[str, Any]]:
        return []

    def fetch_entity_timeline(self, entity_name: str) -> List[Dict[str, Any]]:
        name = str(entity_name or "").strip()
        if not name:
            return []
        rows = self.query(
            """
MATCH (en:Entity {name: $name})-[p:PARTICIPATED_IN]->(e:Event)
RETURN e.event_id AS event_id,
       e.abstract AS abstract,
       e.event_summary AS event_summary,
       e.event_start_time AS event_start_time,
       e.reported_at AS reported_at,
       p.roles AS roles
ORDER BY coalesce(e.event_start_time, e.reported_at, e.first_seen) ASC
""",
            {"name": name},
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            roles = r.get("roles") if isinstance(r.get("roles"), list) else []
            out.append(
                {
                    "event_id": str(r.get("event_id") or ""),
                    "abstract": str(r.get("abstract") or ""),
                    "event_summary": str(r.get("event_summary") or ""),
                    "event_start_time": str(r.get("event_start_time") or ""),
                    "reported_at": str(r.get("reported_at") or ""),
                    "roles_json": json.dumps([x for x in roles if isinstance(x, str) and x.strip()], ensure_ascii=False),
                }
            )
        return out

    def fetch_entity_relations(self, min_co_occurrence: int = 2) -> List[Dict[str, Any]]:
        rows = self.query(
            """
MATCH (a:Entity)-[:PARTICIPATED_IN]->(e:Event)<-[:PARTICIPATED_IN]-(b:Entity)
WHERE a.entity_id < b.entity_id
WITH a, b, count(DISTINCT e.event_id) AS co_occurrence, collect(DISTINCT e.abstract) AS events
WHERE co_occurrence >= $min_co
RETURN a.name AS entity1, b.name AS entity2, co_occurrence, events
ORDER BY co_occurrence DESC
""",
            {"min_co": int(min_co_occurrence)},
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            evs = r.get("events") if isinstance(r.get("events"), list) else []
            out.append(
                {
                    "entity1": str(r.get("entity1") or ""),
                    "entity2": str(r.get("entity2") or ""),
                    "co_occurrence": int(r.get("co_occurrence") or 0),
                    "events": ",".join([str(x) for x in evs if isinstance(x, str) and x.strip()]),
                }
            )
        return out


_neo4j_singleton: Optional[Neo4jAdapter] = None
_neo4j_singleton_lock = threading.Lock()


def get_neo4j_store() -> Neo4jAdapter:
    global _neo4j_singleton
    if _neo4j_singleton is None:
        with _neo4j_singleton_lock:
            if _neo4j_singleton is None:
                _neo4j_singleton = Neo4jAdapter()
    return _neo4j_singleton
