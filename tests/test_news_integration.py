import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import asyncio
import json

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.news.api_manager import get_news_manager
from src.adapters.news.fetch_utils import fetch_from_collector, fetch_from_multiple_sources
from src.adapters.llm.pool import get_llm_pool
from src.app.business.extraction import batch_process_news


def _integration_enabled() -> bool:
    return os.getenv("RUN_INTEGRATION_TESTS", "").strip() == "1"


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture()
def storage_side_effects_disabled(monkeypatch):
    from src.app.business import extraction as extraction_mod
    import src.adapters.sqlite.store as store_mod

    class FakeConfigManager:
        def get_concurrency_limit(self, _name: str) -> int:
            return 1

        def get_rate_limit(self, _name: str) -> float:
            return 1.0

    class FakeStore:
        def add_processed_ids(self, _ids):
            return 0

        def export_compat_json_files(self):
            return None

    monkeypatch.setattr(extraction_mod, "get_config_manager", lambda: FakeConfigManager())
    monkeypatch.setattr(extraction_mod, "update_entities", lambda *args, **kwargs: True)
    monkeypatch.setattr(extraction_mod, "update_abstract_map", lambda *args, **kwargs: True)
    monkeypatch.setattr(store_mod, "get_store", lambda: FakeStore())
    return extraction_mod


def test_integration_fetch_gdelt_real_api() -> None:
    if not _integration_enabled():
        pytest.skip("RUN_INTEGRATION_TESTS!=1")

    mgr = get_news_manager()
    collector = mgr.get_collector("GDELT")

    now = datetime.now(timezone.utc)
    from_ = _iso_z(now - timedelta(days=2))
    to = _iso_z(now)

    news = asyncio.run(
        fetch_from_collector(
            collector=collector,
            source_name="GDELT",
            query="finance",
            limit=3,
            from_=from_,
            to=to,
        )
    )
    if not news:
        pytest.skip("GDELT returned empty result")
    print(json.dumps(news[:2], ensure_ascii=False, indent=2))
    assert isinstance(news, list)
    assert {"id", "title", "content", "source", "url", "datetime"}.issubset(set(news[0].keys()))


def test_integration_fetch_gnews_real_api() -> None:
    if not _integration_enabled():
        pytest.skip("RUN_INTEGRATION_TESTS!=1")

    mgr = get_news_manager()
    available = [s for s in mgr.list_available_sources() if s.startswith("GNews-")]
    if not available:
        pytest.skip("No available GNews sources (missing API key)")

    source_name = "GNews-us" if "GNews-us" in available else available[0]
    collector = mgr.get_collector(source_name)

    news = asyncio.run(
        fetch_from_collector(
            collector=collector,
            source_name=source_name,
            category="general",
            limit=3,
        )
    )
    if not news:
        pytest.skip("GNews returned empty result")
    print(json.dumps(news[:2], ensure_ascii=False, indent=2))
    assert isinstance(news, list)
    assert {"id", "title", "content", "source", "url", "datetime"}.issubset(set(news[0].keys()))


def test_integration_business_flow_fetch_and_extract(storage_side_effects_disabled) -> None:
    if not _integration_enabled():
        pytest.skip("RUN_INTEGRATION_TESTS!=1")

    llm_pool = get_llm_pool()
    client = getattr(llm_pool, "get_available_client", None)
    has_client = bool(client() if callable(client) else getattr(llm_pool, "clients", []))
    if not has_client:
        pytest.skip("No available LLM clients (missing AGENT1_LLM_APIS)")

    mgr = get_news_manager()
    available = mgr.list_available_sources()
    sources = ["GDELT"]
    if "GNews-us" in available:
        sources.append("GNews-us")
    elif any(s.startswith("GNews-") for s in available):
        sources.append([s for s in available if s.startswith("GNews-")][0])

    now = datetime.now(timezone.utc)
    from_ = _iso_z(now - timedelta(days=2))
    to = _iso_z(now)

    raw_news = asyncio.run(
        fetch_from_multiple_sources(
            api_pool=mgr,
            source_names=sources,
            concurrency_limit=2,
            query="OpenAI",
            limit=2,
            from_=from_,
            to=to,
        )
    )
    if not raw_news:
        pytest.skip("No news fetched from selected sources")

    events = asyncio.run(batch_process_news(raw_news, limit=1))
    print(json.dumps(events[:2] if isinstance(events, list) else events, ensure_ascii=False, indent=2))
    assert isinstance(events, list)
    if events:
        assert isinstance(events[0], dict)
        assert "abstract" in events[0]

