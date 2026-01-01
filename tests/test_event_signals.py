import sys
from pathlib import Path
import sqlite3
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.adapters.sqlite.store import SQLiteStore, SQLiteStoreConfig, canonical_event_id


def test_event_signals_upsert_and_get(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    store = SQLiteStore(SQLiteStoreConfig(db_path=db))

    abstract = "A happened"
    store.upsert_events(
        [
            {
                "abstract": abstract,
                "event_summary": abstract,
                "event_types": ["Test"],
                "entities": ["E1"],
                "relations": [],
                "entity_roles": {"E1": ["主体"]},
                "event_start_time": "",
                "event_start_time_text": "",
                "event_start_time_precision": "unknown",
            }
        ],
        source="test",
        reported_at="2025-12-31T00:00:00Z",
    )

    eid = canonical_event_id(abstract)
    wrote = store.upsert_event_signals(
        [
            {
                "event_id": eid,
                "sql_date": "20251231",
                "goldstein_scale": 1.5,
                "num_mentions": 12,
                "event_code": "010",
                "quad_class": 1,
                "avg_tone": -0.2,
                "source_json": {"provider": "gdelt", "table": "events"},
                "confidence": 0.8,
            }
        ]
    )
    assert wrote == 1

    got = store.get_event_signals(eid)
    assert got is not None
    assert got["event_id"] == eid
    assert got["sql_date"] == "20251231"
    assert got["num_mentions"] == 12
    assert isinstance(got.get("source"), dict)
    assert got["source"].get("provider") == "gdelt"


def test_event_signals_print_first_5_rows(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    store = SQLiteStore(SQLiteStoreConfig(db_path=db))

    events = []
    for i in range(1, 6):
        abstract = f"Event {i}"
        events.append(
            {
                "abstract": abstract,
                "event_summary": abstract,
                "event_types": ["Test"],
                "entities": [f"E{i}"],
                "relations": [],
                "entity_roles": {f"E{i}": ["主体"]},
                "event_start_time": "",
                "event_start_time_text": "",
                "event_start_time_precision": "unknown",
            }
        )
    store.upsert_events(events, source="test", reported_at="2025-12-31T00:00:00Z")

    signals = []
    for i in range(1, 6):
        abstract = f"Event {i}"
        eid = canonical_event_id(abstract)
        signals.append(
            {
                "event_id": eid,
                "sql_date": f"202512{25 + i:02d}",
                "goldstein_scale": float(i) * 0.5,
                "num_mentions": i * 10,
                "event_code": f"{i:03d}",
                "quad_class": (i % 4) + 1,
                "avg_tone": -0.1 * i,
                "source_json": {"provider": "gdelt", "table": "events"},
                "confidence": 0.7,
            }
        )
    wrote = store.upsert_event_signals(signals)
    assert wrote == 5

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT event_id, sql_date, goldstein_scale, num_mentions, event_code, quad_class, avg_tone, confidence, updated_at
            FROM event_signals
            ORDER BY sql_date ASC
            LIMIT 5
            """
        ).fetchall()
        out = [dict(r) for r in rows]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        assert len(out) == 5
    finally:
        conn.close()
