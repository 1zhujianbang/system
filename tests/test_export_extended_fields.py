import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.adapters.sqlite.store import SQLiteStore, SQLiteStoreConfig, canonical_event_id


def test_export_entities_and_events_extended_fields(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    store = SQLiteStore(SQLiteStoreConfig(db_path=db))

    abstract = "Export extended fields event"
    store.upsert_events(
        [
            {
                "abstract": abstract,
                "event_summary": "Summary v1",
                "event_types": ["Test"],
                "entities": ["E1", "E2"],
                "relations": [{"subject": "E1", "predicate": "合作", "object": "E2", "evidence": ["x"]}],
                "entity_roles": {"E1": ["主体"], "E2": ["客体"]},
                "event_start_time": "2025-12-31T00:00:00Z",
                "event_start_time_text": "2025-12-31",
                "event_start_time_precision": "date",
            }
        ],
        source="test",
        reported_at="2025-12-31T01:00:00Z",
    )

    eid = canonical_event_id(abstract)
    store.seed_event_observations(eid)
    store.upsert_event_observations(
        [
            {
                "event_id": eid,
                "field": "description",
                "value_text": "Summary v2",
                "value_json": {},
                "evidence_json": [],
                "model_name": "test",
                "model_version": "v",
                "algorithm": "override",
                "revision": 1,
                "is_default": 1,
            }
        ]
    )

    ents = store.export_entities_json()
    assert "E1" in ents
    assert "entity_id" in ents["E1"]
    assert "last_seen" in ents["E1"]
    assert "source_count" in ents["E1"]
    assert "internal_num_mentions" in ents["E1"]
    assert "count" in ents["E1"]

    am = store.export_abstract_map_json()
    assert abstract in am
    ev = am[abstract]
    assert ev.get("event_id") == eid
    assert ev.get("event_summary") == "Summary v2"
    assert ev.get("last_seen") is not None
    assert ev.get("entity_count") == 2
    assert ev.get("relation_count") == 1
    assert isinstance(ev.get("internal_num_mentions"), int)
    rels = ev.get("relations") or []
    assert isinstance(rels, list) and len(rels) == 1
    assert rels[0].get("relation_kind") == "event"
