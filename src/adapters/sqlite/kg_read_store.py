from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...ports.kg_read_store import KGReadStore
from ...infra.paths import tools as Tools


_tools = Tools()


class SQLiteKGReadStore(KGReadStore):
    """SQLite 的 KGReadStore 实现（只读查询）。"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _tools.SQLITE_DB_FILE

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_entities(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT entity_id, name, first_seen FROM entities").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def fetch_events(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT event_id, abstract, event_summary, event_types_json,
                       event_start_time, reported_at, first_seen
                FROM events
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def fetch_participants_with_events(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT p.event_id, p.entity_id, p.roles_json, p.time AS time,
                       e.abstract, e.event_summary, e.event_start_time, e.reported_at AS evt_reported_at, e.first_seen
                FROM participants p
                JOIN events e ON e.event_id = p.event_id
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def fetch_relations(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT subject_entity_id, predicate, object_entity_id, time, evidence_json FROM relations"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def fetch_event_edges(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT from_event_id, to_event_id, edge_type, time, confidence, evidence_json FROM event_edges"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def fetch_entity_timeline(self, entity_name: str) -> List[Dict[str, Any]]:
        """获取实体的时序事件链。
        
        Args:
            entity_name: 实体名称
            
        Returns:
            按时间排序的事件列表
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT
                    e.event_id,
                    e.abstract,
                    e.event_summary,
                    e.event_start_time,
                    e.reported_at,
                    p.roles_json
                FROM events e
                JOIN participants p ON e.event_id = p.event_id
                JOIN entities ent ON p.entity_id = ent.entity_id
                WHERE ent.name = ?
                ORDER BY COALESCE(e.event_start_time, e.reported_at) ASC
                """,
                (entity_name,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def fetch_entity_relations(self, min_co_occurrence: int = 2) -> List[Dict[str, Any]]:
        """推断实体-实体关系（基于共现事件）。
        
        Args:
            min_co_occurrence: 最小共现次数
            
        Returns:
            实体关系列表
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT 
                    e1.name as entity1,
                    e2.name as entity2,
                    COUNT(DISTINCT p1.event_id) as co_occurrence,
                    GROUP_CONCAT(DISTINCT evt.abstract) as events
                FROM participants p1
                JOIN participants p2 ON p1.event_id = p2.event_id
                JOIN entities e1 ON p1.entity_id = e1.entity_id
                JOIN entities e2 ON p2.entity_id = e2.entity_id
                JOIN events evt ON p1.event_id = evt.event_id
                WHERE e1.entity_id < e2.entity_id
                GROUP BY e1.name, e2.name
                HAVING co_occurrence >= ?
                ORDER BY co_occurrence DESC
                """,
                (min_co_occurrence,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()






