import sys
from pathlib import Path
from datetime import datetime, timezone
import asyncio
import json

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.news.gdelt_adapter import GDELTAdapter
from src.adapters.news.api_manager import GNewsAdapter
from src.adapters.news.fetch_utils import fetch_from_collector, fetch_from_multiple_sources
from src.app.business.extraction import batch_process_news
from src.ports.extraction import FetchResult, NewsItem


def test_fetch_utils_pass_extra_into_fetch_config() -> None:
    got = {}

    class FakeCollector:
        async def fetch(self, config):
            got["extra"] = config.extra
            return FetchResult(
                items=[
                    NewsItem(
                        id="1",
                        title="t",
                        content="c",
                        source_name="S",
                        source_url="https://example.com",
                    )
                ],
                total_fetched=1,
                success=True,
            )

    out = asyncio.run(
        fetch_from_collector(
            collector=FakeCollector(),
            source_name="S",
            query="x y",
            limit=1,
            extra={"gdelt": {"fulltext": {"enabled": True}}},
        )
    )
    assert out and isinstance(out, list)
    assert got["extra"] == {"gdelt": {"fulltext": {"enabled": True}}}


def test_fetch_from_collector_returns_standard_shape_gdelt() -> None:
    adapter = GDELTAdapter()

    async def fake_fetch(config):
        assert config.max_items == 2
        return FetchResult(
            items=[
                NewsItem(
                    id="https://gdelt.example/item1",
                    title="GDELT Title 1",
                    content="GDELT Content 1",
                    source_name="GDELT",
                    source_url="https://gdelt.example/item1",
                    published_at=datetime(2025, 12, 27, 10, 0, 0, tzinfo=timezone.utc),
                    author=None,
                    category=None,
                    language="en",
                    raw_data={"provider": "gdelt"},
                )
            ],
            total_fetched=1,
            success=True,
        )

    adapter.fetch = fake_fetch  # type: ignore[assignment]

    news = asyncio.run(
        fetch_from_collector(
            collector=adapter,
            source_name="GDELT",
            query="china trade",
            limit=2,
            from_="2025-12-26T00:00:00Z",
            to="2025-12-27T00:00:00Z",
            extra={"gdelt": {"page_strategy": "time_intervals", "interval_minutes": 60}},
        )
    )

    assert isinstance(news, list) and len(news) == 1
    assert {"id", "title", "content", "source", "url", "datetime"}.issubset(set(news[0].keys()))
    print(news[0])


def test_fetch_from_collector_returns_standard_shape_gnews() -> None:
    adapter = GNewsAdapter(api_key="dummy", language="en", name="GNews-us")

    async def fake_fetch(config):
        assert config.max_items == 2
        return FetchResult(
            items=[
                NewsItem(
                    id="https://gnews.example/item1",
                    title="GNews Title 1",
                    content="GNews Content 1",
                    source_name="GNews-us",
                    source_url="https://gnews.example/item1",
                    published_at=datetime(2025, 12, 27, 9, 0, 0, tzinfo=timezone.utc),
                    author="someone",
                    category=config.category,
                    language="en",
                    raw_data={"provider": "gnews"},
                )
            ],
            total_fetched=1,
            success=True,
        )

    adapter.fetch = fake_fetch  # type: ignore[assignment]

    news = asyncio.run(
        fetch_from_collector(
            collector=adapter,
            source_name="GNews-us",
            query="ai chips",
            category="technology",
            limit=2,
        )
    )

    assert isinstance(news, list) and len(news) == 1
    assert {"id", "title", "content", "source", "url", "datetime"}.issubset(set(news[0].keys()))
    print(news[0])


def test_fetch_from_multiple_sources_sorts_by_datetime() -> None:
    gdelt = GDELTAdapter()
    gnews = GNewsAdapter(api_key="dummy", language="en", name="GNews-us")

    async def fake_gdelt_fetch(config):
        return FetchResult(
            items=[
                NewsItem(
                    id="gdelt-1",
                    title="GDELT Combined Title",
                    content="GDELT Combined Content",
                    source_name="GDELT",
                    source_url="https://gdelt.example/gdelt-1",
                    published_at=datetime(2025, 12, 27, 11, 0, 0, tzinfo=timezone.utc),
                    language="en",
                )
            ],
            total_fetched=1,
            success=True,
        )

    async def fake_gnews_fetch(config):
        return FetchResult(
            items=[
                NewsItem(
                    id="gnews-1",
                    title="GNews Combined Title",
                    content="GNews Combined Content",
                    source_name="GNews-us",
                    source_url="https://gnews.example/gnews-1",
                    published_at=datetime(2025, 12, 27, 10, 0, 0, tzinfo=timezone.utc),
                    language="en",
                )
            ],
            total_fetched=1,
            success=True,
        )

    gdelt.fetch = fake_gdelt_fetch  # type: ignore[assignment]
    gnews.fetch = fake_gnews_fetch  # type: ignore[assignment]

    class FakePool:
        def get_collector(self, name: str):
            if name == "GDELT":
                return gdelt
            if name == "GNews-us":
                return gnews
            raise KeyError(name)

    raw_news = asyncio.run(
        fetch_from_multiple_sources(
            api_pool=FakePool(),
            source_names=["GDELT", "GNews-us"],
            concurrency_limit=2,
            query="test",
            limit=1,
        )
    )
    assert len(raw_news) == 2
    assert raw_news[0]["datetime"] >= raw_news[1]["datetime"]


@pytest.fixture()
def extraction_side_effects_disabled(monkeypatch):
    from src.app.business import extraction as extraction_mod
    import src.adapters.sqlite.store as store_mod

    class FakeConfigManager:
        def get_concurrency_limit(self, _name: str) -> int:
            return 1

        def get_rate_limit(self, _name: str) -> float:
            return 100000.0

    class FakeStore:
        def add_processed_ids(self, _ids):
            return 0

        def export_compat_json_files(self):
            return None

    monkeypatch.setattr(extraction_mod, "get_config_manager", lambda: FakeConfigManager())
    monkeypatch.setattr(extraction_mod, "get_llm_pool", lambda: object())
    monkeypatch.setattr(extraction_mod, "update_entities", lambda *args, **kwargs: True)
    monkeypatch.setattr(extraction_mod, "update_abstract_map", lambda *args, **kwargs: True)
    monkeypatch.setattr(store_mod, "get_store", lambda: FakeStore())
    return extraction_mod


def test_batch_process_news_smoke_print_result(monkeypatch, extraction_side_effects_disabled) -> None:
    def fake_llm_extract_events(title, content, api_pool, max_retries: int = 2, reported_at=None):
        return [
            {
                "abstract": f"{title} | {content[:50]}",
                "entities": ["OpenAI"],
                "entities_original": ["OpenAI"],
                "entity_roles": {"OpenAI": ["主体"]},
                "event_types": ["DemoEvent"],
                "relations": [],
                "event_start_time": "",
                "event_start_time_text": "",
                "event_start_time_precision": "unknown",
            }
        ]

    monkeypatch.setattr(extraction_side_effects_disabled, "llm_extract_events", fake_llm_extract_events)

    sample_news = [
        {
            "id": "sample-1",
            "title": "Sample Title",
            "content": "Sample Content that is long enough to be processed by the pipeline.",
            "source": "GDELT",
            "url": "https://example.com/sample-1",
            "datetime": "2025-12-27T00:00:00+00:00",
        }
    ]

    events = asyncio.run(batch_process_news(sample_news, limit=1))
    print(json.dumps(events, ensure_ascii=False, indent=2))
    assert isinstance(events, list) and len(events) == 1
    assert events[0].get("source") == "GDELT"
    assert events[0].get("news_id") == "sample-1"


def test_fetch_and_extract_multi_sources_smoke_print_result(monkeypatch, extraction_side_effects_disabled) -> None:
    def fake_llm_extract_events(title, content, api_pool, max_retries: int = 2, reported_at=None):
        return [
            {
                "abstract": f"{title} | {content[:50]}",
                "entities": ["EntityA"],
                "entities_original": ["EntityA"],
                "entity_roles": {"EntityA": ["主体"]},
                "event_types": ["DemoEvent"],
                "relations": [],
                "event_start_time": "",
                "event_start_time_text": "",
                "event_start_time_precision": "unknown",
            }
        ]

    monkeypatch.setattr(extraction_side_effects_disabled, "llm_extract_events", fake_llm_extract_events)

    gdelt = GDELTAdapter()
    gnews = GNewsAdapter(api_key="dummy", language="en", name="GNews-us")

    async def fake_gdelt_fetch(config):
        return FetchResult(
            items=[
                NewsItem(
                    id="gdelt-1",
                    title="GDELT Combined Title",
                    content="GDELT Combined Content",
                    source_name="GDELT",
                    source_url="https://gdelt.example/gdelt-1",
                    published_at=datetime(2025, 12, 27, 11, 0, 0, tzinfo=timezone.utc),
                    language="en",
                )
            ],
            total_fetched=1,
            success=True,
        )

    async def fake_gnews_fetch(config):
        return FetchResult(
            items=[
                NewsItem(
                    id="gnews-1",
                    title="GNews Combined Title",
                    content="GNews Combined Content",
                    source_name="GNews-us",
                    source_url="https://gnews.example/gnews-1",
                    published_at=datetime(2025, 12, 27, 10, 0, 0, tzinfo=timezone.utc),
                    language="en",
                )
            ],
            total_fetched=1,
            success=True,
        )

    gdelt.fetch = fake_gdelt_fetch  # type: ignore[assignment]
    gnews.fetch = fake_gnews_fetch  # type: ignore[assignment]

    class FakePool:
        def get_collector(self, name: str):
            if name == "GDELT":
                return gdelt
            if name == "GNews-us":
                return gnews
            raise KeyError(name)

    raw_news = asyncio.run(
        fetch_from_multiple_sources(
            api_pool=FakePool(),
            source_names=["GDELT", "GNews-us"],
            concurrency_limit=2,
            query="test",
            limit=1,
        )
    )
    assert len(raw_news) == 2
    assert raw_news[0]["datetime"] >= raw_news[1]["datetime"]

    events = asyncio.run(batch_process_news(raw_news, limit=2))
    print(json.dumps(events, ensure_ascii=False, indent=2))
    assert isinstance(events, list) and len(events) == 2

