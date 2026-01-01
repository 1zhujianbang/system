import sys
import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.adapters.sqlite.kg_read_store import SQLiteKGReadStore
from src.app.snapshot_service import SnapshotService
from src.ports.snapshot import SnapshotParams


def test_sqlite_kg_read_store_fetch_relation_states(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            CREATE TABLE relation_states (
                relation_state_id TEXT PRIMARY KEY,
                subject_entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id TEXT NOT NULL,
                relation_kind TEXT NOT NULL DEFAULT '',
                valid_from TEXT NOT NULL,
                valid_to TEXT NOT NULL DEFAULT '',
                state_text TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '[]',
                algorithm TEXT NOT NULL DEFAULT '',
                revision INTEGER NOT NULL DEFAULT 0,
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO relation_states(
                relation_state_id, subject_entity_id, predicate, object_entity_id,
                relation_kind, valid_from, valid_to, state_text, evidence_json,
                algorithm, revision, is_default, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rs_default",
                "E1",
                "knows",
                "E2",
                "state",
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                "active",
                json.dumps(["q1"], ensure_ascii=False),
                "algo",
                0,
                1,
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO relation_states(
                relation_state_id, subject_entity_id, predicate, object_entity_id,
                relation_kind, valid_from, valid_to, state_text, evidence_json,
                algorithm, revision, is_default, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rs_non_default",
                "E1",
                "knows",
                "E2",
                "state",
                "2025-03-01T00:00:00Z",
                "2025-04-01T00:00:00Z",
                "inactive",
                json.dumps(["q2"], ensure_ascii=False),
                "algo",
                1,
                0,
                "2025-03-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    store = SQLiteKGReadStore(db_path=db)
    rows = store.fetch_relation_states()
    assert len(rows) == 1
    assert rows[0].get("relation_state_id") == "rs_default"


def test_build_ee_evo_uses_relation_states() -> None:
    rows_entities = [
        {"entity_id": "E1ID", "name": "Alice", "first_seen": "2025-01-01T00:00:00Z"},
        {"entity_id": "E2ID", "name": "Bob", "first_seen": "2025-01-01T00:00:00Z"},
    ]
    rows_relation_states = [
        {
            "relation_state_id": "rs1",
            "subject_entity_id": "E1ID",
            "predicate": "ally",
            "object_entity_id": "E2ID",
            "relation_kind": "state",
            "valid_from": "2025-01-01T00:00:00Z",
            "valid_to": "2025-02-01T00:00:00Z",
            "state_text": "active",
            "evidence_json": json.dumps(["quote1", {"quote": "quote2"}], ensure_ascii=False),
        }
    ]
    params = SnapshotParams(top_entities=20, top_events=0, max_edges=50, days_window=0, gap_days=30)
    svc = SnapshotService(store=object())
    snap = svc.build_ee_evo(rows_entities, rows_relation_states, [], params)

    assert snap.get("meta", {}).get("graph_type") == "EE_EVO"
    rel_nodes = [n for n in (snap.get("nodes") or []) if isinstance(n, dict) and n.get("type") == "relation_state"]
    assert len(rel_nodes) == 1
    rn = rel_nodes[0]
    assert rn.get("id") == "rs1"
    assert rn.get("predicate") == "ally"
    assert rn.get("relation_kind") == "state"
    assert rn.get("interval_start") == "2025-01-01T00:00:00Z"
    assert rn.get("interval_end") == "2025-02-01T00:00:00Z"
    assert rn.get("state_text") == "active"
    assert "quote1" in (rn.get("evidence") or [])
    assert "quote2" in (rn.get("evidence") or [])

    edges = snap.get("edges") or []
    assert {"from": "Alice", "to": "rs1", "type": "rel_in", "title": "ally", "time": "2025-01-01T00:00:00Z"} in edges
    assert {"from": "rs1", "to": "Bob", "type": "rel_out", "title": "ally", "time": "2025-01-01T00:00:00Z"} in edges


def test_evo_timeline_deltas_indexing() -> None:
    from datetime import datetime, timezone

    from src.web.pages_impl.graph import EvolutionGraphRenderer

    frames = [
        datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 20, 0, 0, tzinfo=timezone.utc),
    ]

    rel_intervals = {
        "r1": (datetime(2025, 1, 5, 0, 0, tzinfo=timezone.utc), datetime(2025, 1, 15, 0, 0, tzinfo=timezone.utc)),
        "r2": (datetime(2024, 12, 25, 0, 0, tzinfo=timezone.utc), datetime(2025, 1, 5, 0, 0, tzinfo=timezone.utc)),
        "r3": (datetime(2025, 1, 12, 0, 0, tzinfo=timezone.utc), None),
        "r4": (datetime(2024, 12, 1, 0, 0, tzinfo=timezone.utc), datetime(2024, 12, 15, 0, 0, tzinfo=timezone.utc)),
    }

    deltas = EvolutionGraphRenderer._build_evo_deltas(frames, rel_intervals)
    assert len(deltas) == len(frames)

    assert set(deltas[0].get("enter") or []) == {"r2"}
    assert set(deltas[0].get("exit") or []) == set()

    assert set(deltas[1].get("enter") or []) == {"r1"}
    assert set(deltas[1].get("exit") or []) == {"r2"}

    assert set(deltas[2].get("enter") or []) == {"r3"}
    assert set(deltas[2].get("exit") or []) == {"r1"}
