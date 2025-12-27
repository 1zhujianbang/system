from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ...infra import get_logger
from ...ports.extraction import FetchConfig, FetchResult, NewsItem, NewsSource, NewsSourceType


class _MainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._buf: List[str] = []
        self._skip_depth = 0
        self._skip_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside"}
        self._block_tags = {
            "article",
            "main",
            "section",
            "div",
            "p",
            "br",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "blockquote",
            "pre",
            "hr",
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        t = (tag or "").lower()
        if t in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if t in self._block_tags:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = (tag or "").lower()
        if t in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if t in self._block_tags:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if data:
            self._buf.append(data)

    def get_text(self) -> str:
        return "".join(self._buf)


class GDELTAdapter(NewsSource):
    BASE_URL_V2_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        name: str = "GDELT",
        version: int = 2,
        table: str = "doc",
        coverage: bool = False,
        translation: bool = False,
        timeout: float = 30.0,
        language: str = "zh",
    ):
        self._name = name
        self._version = int(version)
        self._table = str(table or "doc").strip().lower()
        self._coverage = bool(coverage)
        self._translation = bool(translation)
        self._timeout = float(timeout)
        self._language = str(language or "zh")
        self._logger = get_logger(__name__)

    @property
    def source_type(self) -> NewsSourceType:
        return NewsSourceType.GDELT

    @property
    def source_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    async def fetch(self, config: Optional[FetchConfig] = None) -> FetchResult:
        config = config or FetchConfig()

        if self._version != 2:
            return FetchResult(
                items=[],
                total_fetched=0,
                success=False,
                error=f"Unsupported GDELT version: {self._version}",
                fetch_time=datetime.now(timezone.utc),
                metadata={"version": self._version, "table": self._table},
            )

        if self._table != "doc":
            return FetchResult(
                items=[],
                total_fetched=0,
                success=False,
                error=f"Unsupported GDELT table: {self._table}",
                fetch_time=datetime.now(timezone.utc),
                metadata={"version": self._version, "table": self._table},
            )

        try:
            max_items = max(int(config.max_items or 0), 1)
            gdelt_extra = self._gdelt_extra(config)
            per_page = int(gdelt_extra.get("per_page") or 250)
            per_page = min(max(per_page, 1), 250)
            max_pages = int(gdelt_extra.get("max_pages") or 10)
            max_pages = min(max(max_pages, 1), 200)
            startrecord = int(gdelt_extra.get("startrecord") or 1)
            startrecord = max(startrecord, 1)
            page_strategy = str(gdelt_extra.get("page_strategy") or "auto").strip().lower()

            timeout = float(config.timeout_seconds or self._timeout)

            items: List[NewsItem] = []
            seen: set[str] = set()
            pages = 0

            if self._should_use_time_intervals(config, per_page=per_page, max_pages=max_pages, page_strategy=page_strategy):
                interval_minutes = int(gdelt_extra.get("interval_minutes") or gdelt_extra.get("timeslice_minutes") or 60)
                interval_minutes = min(max(interval_minutes, 1), 24 * 60)
                max_intervals = int(gdelt_extra.get("max_intervals") or 80)
                max_intervals = min(max(max_intervals, 1), 2000)
                interval_items, intervals_used = await self._fetch_v2_doc_by_time_intervals(
                    config,
                    max_items=max_items,
                    per_page=per_page,
                    timeout=timeout,
                    interval_minutes=interval_minutes,
                    max_intervals=max_intervals,
                )
                items.extend(interval_items)
                seen.update([it.id for it in interval_items])
                pages = 0
                metadata_extra = {
                    "page_strategy": "time_intervals",
                    "interval_minutes": interval_minutes,
                    "intervals_used": intervals_used,
                }
            else:
                metadata_extra = {"page_strategy": "startrecord"}
                stalled_pages = 0
                while len(items) < max_items and pages < max_pages:
                    remaining = max_items - len(items)
                    page_size = min(per_page, remaining)
                    params = self._build_v2_doc_params(config, maxrecords=page_size, startrecord=startrecord)
                    data = await self._fetch_json(self.BASE_URL_V2_DOC, params=params, timeout=timeout)
                    articles = self._extract_articles(data)
                    if not articles:
                        break
                    page_items = self._convert_v2_doc_to_news_items({"articles": articles}, config=config)

                    before = len(items)
                    for it in page_items:
                        if it.id in seen:
                            continue
                        seen.add(it.id)
                        items.append(it)
                        if len(items) >= max_items:
                            break

                    if len(items) == before:
                        stalled_pages += 1
                    else:
                        stalled_pages = 0

                    startrecord += max(len(articles), 1)
                    pages += 1

                    if stalled_pages >= 1 and page_strategy in {"auto", "timeslice", "time_intervals"}:
                        interval_minutes = int(gdelt_extra.get("interval_minutes") or gdelt_extra.get("timeslice_minutes") or 60)
                        interval_minutes = min(max(interval_minutes, 1), 24 * 60)
                        max_intervals = int(gdelt_extra.get("max_intervals") or 80)
                        max_intervals = min(max(max_intervals, 1), 2000)
                        remaining2 = max_items - len(items)
                        if remaining2 > 0 and self._has_time_range(config):
                            interval_cfg = FetchConfig(
                                max_items=remaining2,
                                language=config.language,
                                category=config.category,
                                keywords=config.keywords,
                                from_date=config.from_date,
                                to_date=config.to_date,
                                timeout_seconds=config.timeout_seconds,
                                extra=config.extra,
                            )
                            interval_items, intervals_used = await self._fetch_v2_doc_by_time_intervals(
                                interval_cfg,
                                max_items=remaining2,
                                per_page=per_page,
                                timeout=timeout,
                                interval_minutes=interval_minutes,
                                max_intervals=max_intervals,
                                already_seen=seen,
                            )
                            for it in interval_items:
                                if it.id in seen:
                                    continue
                                seen.add(it.id)
                                items.append(it)
                                if len(items) >= max_items:
                                    break
                            metadata_extra = {
                                "page_strategy": "fallback_time_intervals",
                                "interval_minutes": interval_minutes,
                                "intervals_used": intervals_used,
                                "startrecord_pages": pages,
                            }
                        break

            fulltext_cfg = gdelt_extra.get("fulltext")
            fulltext_cfg = fulltext_cfg if isinstance(fulltext_cfg, dict) else {}
            fulltext_enabled = bool(fulltext_cfg.get("enabled") or False)
            enriched = 0
            if fulltext_enabled and items:
                max_concurrency = int(fulltext_cfg.get("max_concurrency") or 6)
                max_concurrency = min(max(max_concurrency, 1), 50)
                min_chars = int(fulltext_cfg.get("min_chars") or 200)
                min_chars = min(max(min_chars, 1), 5000)
                max_chars = int(fulltext_cfg.get("max_chars") or 2000)
                max_chars = min(max(max_chars, 200), 20000)
                enriched = await self._enrich_items_with_fulltext(
                    items,
                    timeout=timeout,
                    max_concurrency=max_concurrency,
                    min_chars=min_chars,
                    max_chars=max_chars,
                )

            return FetchResult(
                items=items,
                total_fetched=len(items),
                success=True,
                fetch_time=datetime.now(timezone.utc),
                metadata={
                    "version": self._version,
                    "table": self._table,
                    "endpoint": self.BASE_URL_V2_DOC,
                    "returned": len(items),
                    "pages_fetched": pages,
                    "per_page": per_page,
                    "fulltext_enabled": fulltext_enabled,
                    "fulltext_enriched": enriched,
                    **metadata_extra,
                },
            )
        except Exception as e:
            self._logger.error(f"GDELT fetch error: {e}")
            return FetchResult(
                items=[],
                total_fetched=0,
                success=False,
                error=str(e),
                fetch_time=datetime.now(timezone.utc),
                metadata={"version": self._version, "table": self._table},
            )

    async def fetch_stream(self, config: Optional[FetchConfig] = None) -> AsyncIterator[NewsItem]:
        config = config or FetchConfig()
        if self._version != 2 or self._table != "doc":
            return

        max_items = max(int(config.max_items or 0), 1)
        gdelt_extra = self._gdelt_extra(config)
        per_page = int(gdelt_extra.get("per_page") or 250)
        per_page = min(max(per_page, 1), 250)
        max_pages = int(gdelt_extra.get("max_pages") or 10)
        max_pages = min(max(max_pages, 1), 200)
        startrecord = int(gdelt_extra.get("startrecord") or 1)
        startrecord = max(startrecord, 1)
        page_strategy = str(gdelt_extra.get("page_strategy") or "auto").strip().lower()
        timeout = float(config.timeout_seconds or self._timeout)

        yielded = 0
        seen: set[str] = set()
        pages = 0

        if self._should_use_time_intervals(config, per_page=per_page, max_pages=max_pages, page_strategy=page_strategy):
            interval_minutes = int(gdelt_extra.get("interval_minutes") or gdelt_extra.get("timeslice_minutes") or 60)
            interval_minutes = min(max(interval_minutes, 1), 24 * 60)
            max_intervals = int(gdelt_extra.get("max_intervals") or 80)
            max_intervals = min(max(max_intervals, 1), 2000)
            async for it in self._fetch_v2_doc_stream_by_time_intervals(
                config,
                max_items=max_items,
                per_page=per_page,
                timeout=timeout,
                interval_minutes=interval_minutes,
                max_intervals=max_intervals,
            ):
                yield it
        else:
            stalled_pages = 0
            while yielded < max_items and pages < max_pages:
                remaining = max_items - yielded
                page_size = min(per_page, remaining)
                params = self._build_v2_doc_params(config, maxrecords=page_size, startrecord=startrecord)
                data = await self._fetch_json(self.BASE_URL_V2_DOC, params=params, timeout=timeout)
                articles = self._extract_articles(data)
                if not articles:
                    break
                page_items = self._convert_v2_doc_to_news_items({"articles": articles}, config=config)

                before = yielded
                for it in page_items:
                    if it.id in seen:
                        continue
                    seen.add(it.id)
                    yield it
                    yielded += 1
                    if yielded >= max_items:
                        break

                if yielded == before:
                    stalled_pages += 1
                else:
                    stalled_pages = 0

                startrecord += max(len(articles), 1)
                pages += 1

                if stalled_pages >= 1 and page_strategy in {"auto", "timeslice", "time_intervals"} and self._has_time_range(config):
                    interval_minutes = int(gdelt_extra.get("interval_minutes") or gdelt_extra.get("timeslice_minutes") or 60)
                    interval_minutes = min(max(interval_minutes, 1), 24 * 60)
                    max_intervals = int(gdelt_extra.get("max_intervals") or 80)
                    max_intervals = min(max(max_intervals, 1), 2000)
                    interval_cfg = FetchConfig(
                        max_items=max_items - yielded,
                        language=config.language,
                        category=config.category,
                        keywords=config.keywords,
                        from_date=config.from_date,
                        to_date=config.to_date,
                        timeout_seconds=config.timeout_seconds,
                        extra=config.extra,
                    )
                    async for it in self._fetch_v2_doc_stream_by_time_intervals(
                        interval_cfg,
                        max_items=max_items - yielded,
                        per_page=per_page,
                        timeout=timeout,
                        interval_minutes=interval_minutes,
                        max_intervals=max_intervals,
                        already_seen=seen,
                    ):
                        yield it
                        yielded += 1
                        if yielded >= max_items:
                            break
                    break

    def _gdelt_extra(self, config: FetchConfig) -> Dict[str, Any]:
        if not isinstance(config.extra, dict):
            return {}
        gdelt = config.extra.get("gdelt")
        return gdelt if isinstance(gdelt, dict) else {}

    def _has_time_range(self, config: FetchConfig) -> bool:
        return bool(config.from_date or config.to_date)

    def _should_use_time_intervals(
        self,
        config: FetchConfig,
        *,
        per_page: int,
        max_pages: int,
        page_strategy: str,
    ) -> bool:
        if page_strategy in {"time_intervals", "timeslice"}:
            return True
        if page_strategy in {"startrecord"}:
            return False
        if not self._has_time_range(config):
            return False
        if max_pages <= 1:
            return False
        if int(config.max_items or 0) > int(per_page or 0):
            return True
        return False

    def _build_v2_doc_params(self, config: FetchConfig, *, maxrecords: int, startrecord: int) -> Dict[str, str]:
        query = self._build_query(config)
        gdelt_extra = self._gdelt_extra(config)

        params: Dict[str, str] = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(min(max(int(maxrecords), 1), 250)),
            "startrecord": str(max(int(startrecord), 1)),
            "formatdatetime": "1",
        }

        start_dt, end_dt = self._normalize_time_range(config.from_date, config.to_date)
        if start_dt:
            params["startdatetime"] = start_dt
        if end_dt:
            params["enddatetime"] = end_dt

        sort = gdelt_extra.get("sort")
        if isinstance(sort, str) and sort.strip():
            params["sort"] = sort.strip()

        return params

    def _build_query(self, config: FetchConfig) -> str:
        gdelt_extra = self._gdelt_extra(config)
        override_query = gdelt_extra.get("query")
        if isinstance(override_query, str) and override_query.strip():
            return override_query.strip()

        parts: List[str] = []
        if config.keywords:
            for kw in config.keywords:
                kw = str(kw or "").strip()
                if not kw:
                    continue
                if any(ch.isspace() for ch in kw):
                    parts.append(f"\"{kw}\"")
                else:
                    parts.append(kw)

        if not parts:
            return "a"

        return " ".join(parts)

    def _to_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _with_time_range(
        self,
        config: FetchConfig,
        *,
        from_date: datetime,
        to_date: datetime,
    ) -> FetchConfig:
        return FetchConfig(
            max_items=config.max_items,
            language=config.language,
            category=config.category,
            keywords=config.keywords,
            from_date=from_date,
            to_date=to_date,
            timeout_seconds=config.timeout_seconds,
            extra=config.extra,
        )

    async def _fetch_v2_doc_by_time_intervals(
        self,
        config: FetchConfig,
        *,
        max_items: int,
        per_page: int,
        timeout: float,
        interval_minutes: int,
        max_intervals: int,
        already_seen: Optional[set[str]] = None,
    ) -> Tuple[List[NewsItem], int]:
        start_dt, end_dt = self._default_time_range(config)
        if not start_dt or not end_dt:
            return ([], 0)

        interval_minutes = min(max(int(interval_minutes), 1), 24 * 60)
        min_interval = timedelta(minutes=1)

        dur = timedelta(minutes=interval_minutes)
        stack: List[Tuple[datetime, datetime]] = []
        cur = start_dt
        while cur < end_dt:
            nxt = min(cur + dur, end_dt)
            stack.append((cur, nxt))
            cur = nxt
        items: List[NewsItem] = []
        seen = already_seen if already_seen is not None else set()
        intervals_used = 0

        while stack and len(items) < max_items and intervals_used < max_intervals:
            a, b = stack.pop()
            if b <= a:
                continue

            remaining = max_items - len(items)
            page_size = min(per_page, remaining)
            cfg2 = self._with_time_range(config, from_date=a, to_date=b)
            params = self._build_v2_doc_params(cfg2, maxrecords=page_size, startrecord=1)
            data = await self._fetch_json(self.BASE_URL_V2_DOC, params=params, timeout=timeout)
            articles = self._extract_articles(data)
            if not articles:
                intervals_used += 1
                continue

            page_items = self._convert_v2_doc_to_news_items({"articles": articles}, config=cfg2)
            new_count = 0
            for it in page_items:
                if it.id in seen:
                    continue
                seen.add(it.id)
                items.append(it)
                new_count += 1
                if len(items) >= max_items:
                    break

            intervals_used += 1

            if len(articles) >= per_page and (b - a) > min_interval and len(items) < max_items:
                mid = a + (b - a) / 2
                if mid > a and mid < b:
                    stack.append((a, mid))
                    stack.append((mid, b))
                continue

            if new_count == 0 and (b - a) > min_interval and len(items) < max_items:
                mid = a + (b - a) / 2
                if mid > a and mid < b:
                    stack.append((a, mid))
                    stack.append((mid, b))

        return (items, intervals_used)

    async def _fetch_v2_doc_stream_by_time_intervals(
        self,
        config: FetchConfig,
        *,
        max_items: int,
        per_page: int,
        timeout: float,
        interval_minutes: int,
        max_intervals: int,
        already_seen: Optional[set[str]] = None,
    ) -> AsyncIterator[NewsItem]:
        start_dt, end_dt = self._default_time_range(config)
        if not start_dt or not end_dt:
            return

        interval_minutes = min(max(int(interval_minutes), 1), 24 * 60)
        min_interval = timedelta(minutes=1)
        dur = timedelta(minutes=interval_minutes)
        stack: List[Tuple[datetime, datetime]] = []
        cur = start_dt
        while cur < end_dt:
            nxt = min(cur + dur, end_dt)
            stack.append((cur, nxt))
            cur = nxt
        seen = already_seen if already_seen is not None else set()
        yielded = 0
        intervals_used = 0

        while stack and yielded < max_items and intervals_used < max_intervals:
            a, b = stack.pop()
            if b <= a:
                continue

            remaining = max_items - yielded
            page_size = min(per_page, remaining)
            cfg2 = self._with_time_range(config, from_date=a, to_date=b)
            params = self._build_v2_doc_params(cfg2, maxrecords=page_size, startrecord=1)
            data = await self._fetch_json(self.BASE_URL_V2_DOC, params=params, timeout=timeout)
            articles = self._extract_articles(data)
            if not articles:
                intervals_used += 1
                continue

            page_items = self._convert_v2_doc_to_news_items({"articles": articles}, config=cfg2)
            new_count = 0
            for it in page_items:
                if it.id in seen:
                    continue
                seen.add(it.id)
                yield it
                yielded += 1
                new_count += 1
                if yielded >= max_items:
                    break

            intervals_used += 1

            if len(articles) >= per_page and (b - a) > min_interval and yielded < max_items:
                mid = a + (b - a) / 2
                if mid > a and mid < b:
                    stack.append((a, mid))
                    stack.append((mid, b))
                continue

            if new_count == 0 and (b - a) > min_interval and yielded < max_items:
                mid = a + (b - a) / 2
                if mid > a and mid < b:
                    stack.append((a, mid))
                    stack.append((mid, b))

    def _default_time_range(self, config: FetchConfig) -> Tuple[Optional[datetime], Optional[datetime]]:
        gdelt_extra = self._gdelt_extra(config)
        lookback_hours = gdelt_extra.get("lookback_hours")
        try:
            lookback_hours_i = int(lookback_hours) if lookback_hours is not None else 24
        except Exception:
            lookback_hours_i = 24
        lookback_hours_i = min(max(lookback_hours_i, 1), 24 * 30)

        end_dt = self._to_utc(config.to_date) if config.to_date else None
        start_dt = self._to_utc(config.from_date) if config.from_date else None

        if end_dt is None and start_dt is None:
            return (None, None)

        if end_dt is None and start_dt is not None:
            end_dt = datetime.now(timezone.utc)

        if start_dt is None and end_dt is not None:
            start_dt = end_dt - timedelta(hours=lookback_hours_i)

        if start_dt and end_dt and end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt

        return (start_dt, end_dt)

    def _extract_articles(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(data.get("articles"), list):
            return [a for a in data["articles"] if isinstance(a, dict)]
        if isinstance(data.get("data"), dict) and isinstance(data["data"].get("articles"), list):
            return [a for a in data["data"]["articles"] if isinstance(a, dict)]
        return []

    def _extract_text_from_html(self, html_text: str, *, max_chars: int) -> str:
        max_chars = min(max(int(max_chars), 200), 20000)
        parser = _MainTextHTMLParser()
        try:
            parser.feed(html_text or "")
            parser.close()
            raw = parser.get_text()
        except Exception:
            raw = html_text or ""

        raw = html.unescape(raw)
        raw = raw.replace("\u00a0", " ")
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        lines = [ln.strip() for ln in raw.split("\n")]
        kept: List[str] = []
        short_kept = 0
        for ln in lines:
            if not ln:
                continue
            if len(ln) < 30 and not any(ch in ln for ch in ("。", "！", "？", ".", "!", "?")):
                if not kept and short_kept < 2 and len(ln) >= 8:
                    kept.append(ln)
                    short_kept += 1
                continue
            kept.append(ln)
            if sum(len(x) for x in kept) >= max_chars:
                break
        text = "\n".join(kept).strip()
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def _normalize_time_range(
        self,
        from_date: Optional[datetime],
        to_date: Optional[datetime],
    ) -> Tuple[Optional[str], Optional[str]]:
        def _to_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        start = _to_utc(from_date) if from_date else None
        end = _to_utc(to_date) if to_date else None

        def _fmt(dt: datetime) -> str:
            return dt.strftime("%Y%m%d%H%M%S")

        return (_fmt(start) if start else None, _fmt(end) if end else None)

    async def _fetch_json(self, url: str, params: Dict[str, str], timeout: float) -> Dict[str, Any]:
        import aiohttp

        try:
            from aiohttp.resolver import ThreadedResolver

            resolver = ThreadedResolver()
        except Exception:
            resolver = None

        connector = aiohttp.TCPConnector(resolver=resolver)
        client_timeout = aiohttp.ClientTimeout(total=float(timeout))

        async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"GDELT API error: {resp.status} - {text}")
                return await resp.json()

    async def _fetch_html(self, url: str, *, timeout: float) -> str:
        import aiohttp

        try:
            from aiohttp.resolver import ThreadedResolver

            resolver = ThreadedResolver()
        except Exception:
            resolver = None

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; NewsAgent/1.0; +https://example.local)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        connector = aiohttp.TCPConnector(resolver=resolver)
        client_timeout = aiohttp.ClientTimeout(total=float(timeout))
        async with aiohttp.ClientSession(connector=connector, timeout=client_timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    text = await resp.text(errors="ignore")
                    raise RuntimeError(f"HTML fetch error: {resp.status} - {text[:200]}")
                return await resp.text(errors="ignore")

    async def _enrich_items_with_fulltext(
        self,
        items: List[NewsItem],
        *,
        timeout: float,
        max_concurrency: int,
        min_chars: int,
        max_chars: int,
    ) -> int:
        sem = asyncio.Semaphore(min(max(int(max_concurrency), 1), 50))
        min_chars = min(max(int(min_chars), 1), 5000)
        max_chars = min(max(int(max_chars), 200), 20000)

        async def enrich_one(it: NewsItem) -> bool:
            url = str(it.source_url or it.id or "").strip()
            if not url or not url.startswith(("http://", "https://")):
                return False
            cur = str(it.content or "").strip()
            if len(cur) >= min_chars and cur != (it.title or ""):
                return False
            async with sem:
                try:
                    html_text = await self._fetch_html(url, timeout=timeout)
                    text = self._extract_text_from_html(html_text, max_chars=max_chars)
                    text = text.strip()
                    if not text:
                        return False
                    if len(text) <= len(cur):
                        return False
                    it.content = text
                    return True
                except Exception:
                    return False

        results = await asyncio.gather(*(enrich_one(it) for it in items), return_exceptions=False)
        return sum(1 for r in results if r)

    def _convert_v2_doc_to_news_items(self, data: Dict[str, Any], config: FetchConfig) -> List[NewsItem]:
        articles: List[Dict[str, Any]] = []
        if isinstance(data.get("articles"), list):
            articles = data["articles"]
        elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("articles"), list):
            articles = data["data"]["articles"]

        out: List[NewsItem] = []
        seen: set[str] = set()

        for a in articles:
            if not isinstance(a, dict):
                continue
            url = str(a.get("url") or "").strip()
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)

            title = str(a.get("title") or "").strip()
            snippet = str(a.get("snippet") or a.get("description") or "").strip()
            content = snippet or title

            published_at = self._parse_any_datetime(a.get("seendate") or a.get("seenDate") or a.get("datetime"))

            out.append(
                NewsItem(
                    id=url,
                    title=title or url,
                    content=content,
                    source_name=self._name,
                    source_url=url,
                    published_at=published_at,
                    author=None,
                    category=config.category,
                    language=(config.language or self._language),
                    raw_data=a,
                )
            )

        return out

    def _parse_any_datetime(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value

        s = str(value).strip()
        if not s:
            return None

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y%m%d%H%M%S"):
            try:
                dt = datetime.strptime(s, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass

        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

