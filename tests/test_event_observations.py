import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.adapters.sqlite.store import SQLiteStore, SQLiteStoreConfig, canonical_event_id, canonical_entity_id


def test_event_observations_seed_projection_and_validation(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    store = SQLiteStore(SQLiteStoreConfig(db_path=db))

    abstract = "Observation seed event"
    store.upsert_events(
        [
            {
                "abstract": abstract,
                "event_summary": "A process happened",
                "event_types": ["Test"],
                "entities": ["E1"],
                "relations": [],
                "entity_roles": {"E1": ["主体"]},
                "event_start_time": "2025-12-31T12:34:56Z",
                "event_start_time_text": "2025-12-31 12:34:56 UTC",
                "event_start_time_precision": "second",
            }
        ],
        source="test",
        reported_at="2025-12-31T13:00:00Z",
    )

    eid = canonical_event_id(abstract)
    wrote = store.seed_event_observations(eid)
    assert wrote >= 1

    obs = store.list_event_observations(eid)
    fields = {o.get("field") for o in obs}
    assert "start_time" in fields
    assert "description" in fields

    proj = store.get_default_event_projection(eid)
    assert proj.get("event_id") == eid
    assert proj.get("event_start_time") == "2025-12-31T12:34:56Z"
    assert proj.get("event_start_time_precision") == "second"
    assert proj.get("event_summary") == "A process happened"

    store.upsert_event_signals(
        [
            {
                "event_id": eid,
                "sql_date": "20251231",
                "num_mentions": 1,
                "event_code": "010",
                "source_json": {"provider": "test"},
                "confidence": 1.0,
            }
        ]
    )

    report = store.validate_event_against_signals(eid)
    assert report.get("status") == "ok"
    checks = report.get("checks") or []
    st = {c.get("field"): c.get("status") for c in checks}
    assert st.get("start_date") == "match"
    assert st.get("num_mentions") in {"match", "unknown"}


def test_build_event_observations_llm_overrides_default_projection(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "test.sqlite"
    store = SQLiteStore(SQLiteStoreConfig(db_path=db))

    abstract = "LLM observation event"
    store.upsert_events(
        [
            {
                "abstract": abstract,
                "event_summary": "Seed summary",
                "event_types": ["Test"],
                "entities": ["E1"],
                "relations": [],
                "entity_roles": {"E1": ["主体"]},
                "event_start_time": "2025-12-30T00:00:00Z",
                "event_start_time_text": "2025-12-30",
                "event_start_time_precision": "day",
            }
        ],
        source="test",
        reported_at="2025-12-31T13:00:00Z",
    )

    eid = canonical_event_id(abstract)
    store.seed_event_observations(eid)
    seeded = store.list_event_observations(eid)
    any_evidence = (seeded[0].get("evidence") or []) if seeded else []
    mention_id = str((any_evidence[0].get("mention_id") if any_evidence else "") or "")
    assert mention_id

    from src.ports.llm_client import LLMResponse

    class FakePool:
        def call(self, prompt: str, config=None, preferred_provider=None):
            content = {
                "observations": [
                    {
                        "field": "start_time",
                        "value_text": "2025-12-31 12:34:56 UTC",
                        "value_json": {
                            "time": "2025-12-31T12:34:56Z",
                            "time_text": "2025-12-31 12:34:56 UTC",
                            "precision": "second",
                            "candidates": [
                                {
                                    "time": "2025-12-31T12:34:56Z",
                                    "time_text": "2025-12-31 12:34:56 UTC",
                                    "precision": "second",
                                    "evidence": [{"mention_id": mention_id, "quote": "time evidence"}],
                                    "reason": "explicit timestamp",
                                }
                            ],
                            "selection_reason": "best supported by evidence",
                        },
                        "evidence": [{"mention_id": mention_id, "quote": "time evidence"}],
                    },
                    {
                        "field": "description",
                        "value_text": "LLM neutral description",
                        "value_json": {"seed_from": "llm"},
                        "evidence": [{"mention_id": mention_id, "quote": "desc evidence"}],
                    },
                    {
                        "field": "color",
                        "value_text": "中性过程性描述，不含数值",
                        "value_json": {"seed_from": "llm"},
                        "evidence": [{"mention_id": mention_id, "quote": "color evidence"}],
                    },
                    {
                        "field": "credibility",
                        "value_text": "基于现有证据的文字陈述，包含不确定性",
                        "value_json": {"seed_from": "llm"},
                        "evidence": [{"mention_id": mention_id, "quote": "cred evidence"}],
                    },
                ]
            }
            import json as _json

            return LLMResponse(content=_json.dumps(content, ensure_ascii=False), model="fake-model")

    import src.interfaces.tools.migration as migration_mod
    import src.adapters.llm.pool as pool_mod

    monkeypatch.setattr(migration_mod, "get_store", lambda: store)
    monkeypatch.setattr(pool_mod, "get_llm_pool", lambda: FakePool())

    out = migration_mod.build_event_observations_llm(event_id=eid)
    assert out.get("mode") == "single"
    res = out.get("result") or {}
    assert res.get("status") == "ok"
    assert int(res.get("observations_written") or 0) >= 2

    proj = store.get_default_event_projection(eid)
    assert proj.get("event_start_time") == "2025-12-31T12:34:56Z"
    assert proj.get("event_summary") == "LLM neutral description"


def test_relation_states_rebuild_and_timeline(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    store = SQLiteStore(SQLiteStoreConfig(db_path=db))

    store.upsert_events(
        [
            {
                "abstract": "RelState Event 1",
                "event_summary": "Event 1 summary",
                "event_types": ["Test"],
                "entities": ["E1", "E2"],
                "relations": [{"subject": "E1", "predicate": "关联", "object": "E2", "evidence": "ev1"}],
                "entity_roles": {"E1": ["主体"], "E2": ["客体"]},
                "event_start_time": "2025-12-31T12:00:00Z",
                "event_start_time_text": "2025-12-31 12:00:00 UTC",
                "event_start_time_precision": "second",
            },
            {
                "abstract": "RelState Event 2",
                "event_summary": "Event 2 summary",
                "event_types": ["Test"],
                "entities": ["E1", "E2"],
                "relations": [{"subject": "E1", "predicate": "关联", "object": "E2", "evidence": "ev2"}],
                "entity_roles": {"E1": ["主体"], "E2": ["客体"]},
                "event_start_time": "2025-12-31T13:00:00Z",
                "event_start_time_text": "2025-12-31 13:00:00 UTC",
                "event_start_time_precision": "second",
            },
        ],
        source="test",
        reported_at="2025-12-31T14:00:00Z",
    )

    wrote = store.rebuild_relation_states_for_triple(canonical_entity_id("E1"), "关联", canonical_entity_id("E2"))
    assert wrote == 2

    timeline = store.query_relation_timeline(canonical_entity_id("E1"), "关联", canonical_entity_id("E2"))
    assert len(timeline) == 2
    assert timeline[0].get("valid_from") == "2025-12-31T12:00:00Z"
    assert timeline[0].get("valid_to") == "2025-12-31T13:00:00Z"
    assert timeline[1].get("valid_from") == "2025-12-31T13:00:00Z"
    assert timeline[1].get("valid_to") in {"", None}
    assert "Event 1 summary" in str(timeline[0].get("state_text") or "")


def test_build_relation_states_tool_batch_missing(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "test.sqlite"
    store = SQLiteStore(SQLiteStoreConfig(db_path=db))

    store.upsert_events(
        [
            {
                "abstract": "RelState Tool Event",
                "event_summary": "Tool event summary",
                "event_types": ["Test"],
                "entities": ["E1", "E2"],
                "relations": [{"subject": "E1", "predicate": "合作", "object": "E2", "evidence": ["quote"]}],
                "entity_roles": {"E1": ["主体"], "E2": ["客体"]},
                "event_start_time": "2025-12-31T10:00:00Z",
                "event_start_time_text": "2025-12-31 10:00:00 UTC",
                "event_start_time_precision": "second",
            }
        ],
        source="test",
        reported_at="2025-12-31T10:01:00Z",
    )

    import src.interfaces.tools.migration as migration_mod

    monkeypatch.setattr(migration_mod, "get_store", lambda: store)
    with store._lock:
        conn = store._connect()
        try:
            conn.execute("DELETE FROM relation_states")
            conn.commit()
        finally:
            conn.close()

    out = migration_mod.build_relation_states(limit=50)
    assert out.get("status") == "ok"
    assert int(out.get("relation_states_written") or 0) >= 1

    tl = store.list_relation_states(canonical_entity_id("E1"), "合作", canonical_entity_id("E2"))
    assert len(tl) >= 1


def test_relation_states_incremental_on_upsert_events(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    store = SQLiteStore(SQLiteStoreConfig(db_path=db))

    store.upsert_events(
        [
            {
                "abstract": "RelState Inc Event 1",
                "event_summary": "Inc 1",
                "event_types": ["Test"],
                "entities": ["E1", "E2"],
                "relations": [{"subject": "E1", "predicate": "关联", "object": "E2", "evidence": "x"}],
                "entity_roles": {"E1": ["主体"], "E2": ["客体"]},
                "event_start_time": "2025-12-31T12:00:00Z",
                "event_start_time_text": "2025-12-31 12:00:00 UTC",
                "event_start_time_precision": "second",
            },
            {
                "abstract": "RelState Inc Event 2",
                "event_summary": "Inc 2",
                "event_types": ["Test"],
                "entities": ["E1", "E2"],
                "relations": [{"subject": "E1", "predicate": "关联", "object": "E2", "evidence": "y"}],
                "entity_roles": {"E1": ["主体"], "E2": ["客体"]},
                "event_start_time": "2025-12-31T13:00:00Z",
                "event_start_time_text": "2025-12-31 13:00:00 UTC",
                "event_start_time_precision": "second",
            },
        ],
        source="test",
        reported_at="2025-12-31T14:00:00Z",
    )

    timeline = store.query_relation_timeline(canonical_entity_id("E1"), "关联", canonical_entity_id("E2"))
    assert len(timeline) == 2
