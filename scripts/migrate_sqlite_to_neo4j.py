"""
迁移脚本：将 SQLite 数据迁移到 Neo4j
"""
import argparse
import sys
import os
import sqlite3
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# 添加项目根目录到 path
sys.path.append(str(Path(__file__).parent.parent))

from src.adapters.graph_store.neo4j_adapter import Neo4jAdapter
from src.core import get_config_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 1000

def load_env_file(env_file: str) -> None:
    if not env_file:
        return
    p = Path(env_file)
    if not p.exists():
        logger.warning(f"Neo4j env file not found: {env_file}")
        return
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read Neo4j env file: {e}")
        return

    for line in text.splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = (k or "").strip()
        v = (v or "").strip().strip('"').strip("'")
        if not k:
            continue

        if k not in os.environ:
            os.environ[k] = v
        if k == "NEO4J_USERNAME" and "NEO4J_USER" not in os.environ:
            os.environ["NEO4J_USER"] = v


def get_sqlite_path() -> str:
    """获取 SQLite 数据库路径"""
    # 假设在 data/store.sqlite
    return str(Path(__file__).parent.parent / "data" / "store.sqlite")

def _parse_json_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if isinstance(obj, list):
        return obj
    return []

def _parse_json_list_str(value: Any) -> List[str]:
    out: List[str] = []
    for x in _parse_json_list(value):
        if x is None:
            continue
        t = (x.strip() if isinstance(x, str) else str(x).strip())
        if t:
            out.append(t)
    return out

def _prepare_entities(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        raw = r.get("original_forms_json")
        r2 = dict(r)
        r2["original_forms_json_raw"] = "" if raw is None else str(raw)
        r2["original_forms"] = _parse_json_list_str(raw)
        out.append(r2)
    return out

def _prepare_events(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        raw = r.get("event_types_json")
        r2 = dict(r)
        r2["event_types_json_raw"] = "" if raw is None else str(raw)
        r2["types"] = _parse_json_list_str(raw)
        out.append(r2)
    return out

def _prepare_participants(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        raw = r.get("roles_json")
        r2 = dict(r)
        r2["roles_json_raw"] = "" if raw is None else str(raw)
        r2["roles"] = _parse_json_list_str(raw)
        out.append(r2)
    return out

def _prepare_relations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        raw = r.get("evidence_json")
        r2 = dict(r)
        r2["evidence_json_raw"] = "" if raw is None else str(raw)
        r2["evidence"] = _parse_json_list_str(raw)
        out.append(r2)
    return out

def _prepare_event_edges(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        raw = r.get("evidence_json")
        r2 = dict(r)
        r2["evidence_json_raw"] = "" if raw is None else str(raw)
        r2["evidence"] = _parse_json_list_str(raw)
        out.append(r2)
    return out

def fetch_sqlite_data(table: str) -> List[Dict[str, Any]]:
    """从 SQLite 读取所有数据"""
    db_path = get_sqlite_path()
    if not os.path.exists(db_path):
        logger.error(f"SQLite database not found at {db_path}")
        return []
        
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table}")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch data from {table}: {e}")
        return []

def migrate_entities(neo4j: Neo4jAdapter):
    """迁移实体"""
    logger.info("Migrating entities...")
    rows = fetch_sqlite_data("entities")
    if not rows:
        logger.info("No entities to migrate.")
        return
    rows = _prepare_entities(rows)

    query = """
    UNWIND $batch AS row
    MERGE (n:Entity {id: row.entity_id})
    SET n.name = row.name,
        n.first_seen = row.first_seen,
        n.last_seen = row.last_seen,
        n.original_forms = row.original_forms,
        n.original_forms_json = row.original_forms_json_raw
    """
    
    _batch_execute(neo4j, query, rows, "entities")

def migrate_events(neo4j: Neo4jAdapter):
    """迁移事件"""
    logger.info("Migrating events...")
    rows = fetch_sqlite_data("events")
    if not rows:
        logger.info("No events to migrate.")
        return
    rows = _prepare_events(rows)

    query = """
    UNWIND $batch AS row
    MERGE (n:Event {id: row.event_id})
    SET n.abstract = row.abstract,
        n.summary = row.event_summary,
        n.types = row.types,
        n.types_json = row.event_types_json_raw,
        n.start_time = row.event_start_time,
        n.reported_at = row.reported_at
    """
    
    _batch_execute(neo4j, query, rows, "events")

def migrate_participants(neo4j: Neo4jAdapter):
    """迁移参与关系"""
    logger.info("Migrating participants...")
    rows = fetch_sqlite_data("participants")
    if not rows:
        logger.info("No participants to migrate.")
        return
    rows = _prepare_participants(rows)

    query = """
    UNWIND $batch AS row
    MATCH (e:Entity {id: row.entity_id})
    MATCH (ev:Event {id: row.event_id})
    MERGE (e)-[r:PARTICIPATED_IN]->(ev)
    SET r.roles = row.roles,
        r.roles_json = row.roles_json_raw,
        r.time = row.time,
        r.reported_at = row.reported_at
    """
    
    _batch_execute(neo4j, query, rows, "participants")

def migrate_relations(neo4j: Neo4jAdapter):
    """迁移实体间关系"""
    logger.info("Migrating relations...")
    rows = fetch_sqlite_data("relations")
    if not rows:
        logger.info("No relations to migrate.")
        return
    rows = _prepare_relations(rows)

    query = """
    UNWIND $batch AS row
    MATCH (s:Entity {id: row.subject_entity_id})
    MATCH (o:Entity {id: row.object_entity_id})
    MERGE (s)-[r:RELATED_TO {event_id: row.event_id, predicate: row.predicate}]->(o)
    SET r.time = row.time,
        r.reported_at = row.reported_at,
        r.evidence = row.evidence,
        r.evidence_json = row.evidence_json_raw
    """
    
    _batch_execute(neo4j, query, rows, "relations")

def migrate_event_edges(neo4j: Neo4jAdapter):
    """迁移事件演化边"""
    logger.info("Migrating event edges...")
    rows = fetch_sqlite_data("event_edges")
    if not rows:
        logger.info("No event edges to migrate.")
        return
    rows = _prepare_event_edges(rows)

    query = """
    UNWIND $batch AS row
    MATCH (f:Event {id: row.from_event_id})
    MATCH (t:Event {id: row.to_event_id})
    MERGE (f)-[r:EVOLVED_TO {type: row.edge_type}]->(t)
    SET r.time = row.time,
        r.reported_at = row.reported_at,
        r.confidence = row.confidence,
        r.evidence = row.evidence,
        r.evidence_json = row.evidence_json_raw
    """
    
    _batch_execute(neo4j, query, rows, "event_edges")

def _batch_execute(neo4j: Neo4jAdapter, query: str, rows: List[Dict], label: str):
    """批量执行辅助函数"""
    total = len(rows)
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        try:
            neo4j.query(query, {"batch": batch})
            logger.info(f"Processed {min(i + BATCH_SIZE, total)}/{total} {label}")
        except Exception as e:
            logger.error(f"Failed to process batch for {label}: {e}")

def dry_run() -> None:
    ent = _prepare_entities(fetch_sqlite_data("entities"))
    evt = _prepare_events(fetch_sqlite_data("events"))
    par = _prepare_participants(fetch_sqlite_data("participants"))
    rel = _prepare_relations(fetch_sqlite_data("relations"))
    eed = _prepare_event_edges(fetch_sqlite_data("event_edges"))

    logger.info(
        "Dry-run counts: entities=%s, events=%s, participants=%s, relations=%s, event_edges=%s",
        len(ent),
        len(evt),
        len(par),
        len(rel),
        len(eed),
    )

    if ent:
        logger.info("Dry-run sample: entity original_forms_len=%s", len(ent[0].get("original_forms") or []))
    if evt:
        logger.info("Dry-run sample: event types_len=%s", len(evt[0].get("types") or []))
    if par:
        logger.info("Dry-run sample: participant roles_len=%s", len(par[0].get("roles") or []))
    if rel:
        logger.info("Dry-run sample: relation evidence_len=%s", len(rel[0].get("evidence") or []))
    if eed:
        logger.info("Dry-run sample: event_edge evidence_len=%s", len(eed[0].get("evidence") or []))

def main():
    logger.info("Starting migration from SQLite to Neo4j...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=os.getenv("NEO4J_ENV_FILE", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env_file(args.env_file)
    
    # 检查 SQLite 文件
    if not os.path.exists(get_sqlite_path()):
        logger.error("SQLite database file not found. Please ensure data/store.sqlite exists.")
        return

    if args.dry_run:
        dry_run()
        return

    # 初始化 Neo4j 适配器
    try:
        neo4j = Neo4jAdapter()
        if not neo4j.is_available():
            logger.error("Neo4j is not available. Please check your connection/configuration.")
            return
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j adapter: {e}")
        return

    # 创建约束（可选但推荐）
    try:
        neo4j.query("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
        neo4j.query("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE")
        logger.info("Constraints created.")
    except Exception as e:
        logger.warning(f"Failed to create constraints (might already exist or not supported in this edition): {e}")

    # 执行迁移
    migrate_entities(neo4j)
    migrate_events(neo4j)
    migrate_participants(neo4j)
    migrate_relations(neo4j)
    migrate_event_edges(neo4j)
    
    neo4j.close()
    logger.info("Migration completed successfully.")

if __name__ == "__main__":
    main()
