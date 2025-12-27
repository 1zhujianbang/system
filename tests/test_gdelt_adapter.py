import sys
from pathlib import Path
import asyncio
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.news.gdelt_adapter import GDELTAdapter
from src.ports.extraction import FetchConfig


def test_gdelt_build_query_keywords() -> None:
    adapter = GDELTAdapter()
    query = adapter._build_query(FetchConfig(keywords=["China", "trade war"]))
    assert query == 'China "trade war"'


def test_gdelt_convert_v2_doc_dedup_and_fields() -> None:
    adapter = GDELTAdapter()
    data = {
        "articles": [
            {
                "url": "https://example.com/a",
                "title": "Title A",
                "snippet": "Snippet A",
                "seendate": "2025-12-27 10:11:12",
            },
            {
                "url": "https://example.com/a",
                "title": "Title A2",
                "snippet": "Snippet A2",
                "seendate": "2025-12-27 10:11:12",
            },
            {
                "url": "https://example.com/b",
                "title": "Title B",
                "seendate": "20251227101112",
            },
        ]
    }
    items = adapter._convert_v2_doc_to_news_items(data, config=FetchConfig(max_items=10, language="en"))
    assert [it.id for it in items] == ["https://example.com/a", "https://example.com/b"]
    assert items[0].title == "Title A"
    assert items[0].content == "Snippet A"
    assert items[0].published_at is not None
    assert items[0].published_at.tzinfo is not None
    assert items[0].language == "en"
    assert items[1].content == "Title B"


def test_gdelt_fetch_happy_path_without_network() -> None:
    adapter = GDELTAdapter()
    sample = {
        "articles": [
            {
                "url": "https://example.com/c",
                "title": "Title C",
                "snippet": "Snippet C",
                "seendate": "2025-12-27 10:11:12",
            }
        ]
    }

    async def fake_fetch_json(url: str, params: dict, timeout: float):
        assert "query" in params
        return sample

    adapter._fetch_json = fake_fetch_json  # type: ignore[assignment]
    cfg = FetchConfig(
        max_items=5,
        language="en",
        keywords=["test"],
        from_date=datetime(2025, 12, 26, tzinfo=timezone.utc),
        to_date=datetime(2025, 12, 27, tzinfo=timezone.utc),
    )
    result = asyncio.run(adapter.fetch(cfg))
    assert result.success is True
    assert result.total_fetched == 1
    assert result.items[0].id == "https://example.com/c"


def test_gdelt_fetch_time_intervals_without_network() -> None:
    adapter = GDELTAdapter()
    calls = []

    async def fake_fetch_json(url: str, params: dict, timeout: float):
        calls.append(params)
        start_dt = params.get("startdatetime") or "NA"
        return {
            "articles": [
                {
                    "url": f"https://example.com/{start_dt}",
                    "title": f"Title {start_dt}",
                    "snippet": f"Snippet {start_dt}",
                    "seendate": "2025-12-27 10:11:12",
                }
            ]
        }

    adapter._fetch_json = fake_fetch_json  # type: ignore[assignment]

    now = datetime(2025, 12, 27, 12, 0, 0, tzinfo=timezone.utc)
    cfg = FetchConfig(
        max_items=2,
        language="en",
        keywords=["test"],
        from_date=now - timedelta(hours=2),
        to_date=now,
        extra={"gdelt": {"page_strategy": "time_intervals", "interval_minutes": 60, "per_page": 1}},
    )
    result = asyncio.run(adapter.fetch(cfg))
    assert result.success is True
    assert result.total_fetched == 2
    assert len(calls) >= 2


def test_gdelt_extract_text_from_html_basic() -> None:
    adapter = GDELTAdapter()
    html_doc = """
    <html>
      <head>
        <title>ignored</title>
        <style>.x{color:red}</style>
        <script>var secret = 1;</script>
      </head>
      <body>
        <nav>Home | About</nav>
        <article>
          <h1>Real Title</h1>
          <p>This is the first paragraph of the article, with enough length to keep.</p>
          <p>This is the second paragraph of the article, also long enough to keep.</p>
        </article>
        <footer>Copyright</footer>
      </body>
    </html>
    """
    text = adapter._extract_text_from_html(html_doc, max_chars=2000)
    assert "Real Title" in text
    assert "first paragraph" in text
    assert "var secret" not in text


def test_gdelt_enrich_items_with_fulltext_without_network() -> None:
    adapter = GDELTAdapter()

    async def fake_fetch_html(url: str, *, timeout: float) -> str:
        return "<html><body><article><p>This is a long body paragraph with enough length to keep.</p></article></body></html>"

    adapter._fetch_html = fake_fetch_html  # type: ignore[assignment]
    item = adapter._convert_v2_doc_to_news_items(
        {"articles": [{"url": "https://example.com/x", "title": "T", "seendate": "2025-12-27 10:11:12"}]},
        config=FetchConfig(max_items=1, language="en"),
    )[0]
    assert item.content == "T"
    enriched = asyncio.run(
        adapter._enrich_items_with_fulltext(
            [item],
            timeout=5.0,
            max_concurrency=2,
            min_chars=50,
            max_chars=2000,
        )
    )
    assert enriched == 1
    assert "long body paragraph" in item.content
