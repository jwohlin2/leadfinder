"""Brave Search API provider.

Uses the official Web Search API (https://api.search.brave.com/res/v1/web/search)
with the free tier's 1 QPS limit enforced client-side so a 4-worker benchmark
does not hammer the API and burn the monthly quota on HTTP 429s.
"""

from __future__ import annotations

import asyncio

import httpx

from leadfinder.search.base import SearchProvider, SearchResult, classify_source_type

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchAPIProvider(SearchProvider):
    name = "brave"

    def __init__(
        self,
        api_key: str = "",
        timeout: float = 20.0,
        max_queries_per_run: int = 400,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_queries = max_queries_per_run
        self._queries_used = 0
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            follow_redirects=True,
        )
        # The free plan rate-limits to ~1 request/second and ~20/minute.
        self._throttle = asyncio.Lock()
        self._last_sent = 0.0
        self._period = 1.0

    async def available(self) -> bool:
        if not self.api_key:
            return False
        if self._queries_used >= self.max_queries:
            return False
        try:
            response = await self._client.get(
                _ENDPOINT,
                params={"q": "leadfinder", "count": 1},
                timeout=min(self.timeout, 10.0),
            )
            return response.status_code == 200
        except Exception:
            return False

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        if not self.api_key:
            return []
        if self._queries_used >= self.max_queries:
            return []

        await self._rate_limit()
        try:
            response = await self._client.get(
                _ENDPOINT,
                params={"q": query, "count": min(max_results * 3, 20), "safesearch": "off"},
            )
        except Exception:
            return []

        self._queries_used += 1
        if response.status_code != 200:
            return []

        try:
            payload = response.json()
        except ValueError:
            return []

        results: list[SearchResult] = []
        for position, item in enumerate(payload.get("web", {}).get("results") or [], start=1):
            url = (item.get("url") or "").strip()
            if not url or not url.startswith("http"):
                continue
            title = (item.get("title") or "").strip()
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=(item.get("description") or "").strip(),
                    engine="brave",
                    position=position,
                    source_type=classify_source_type(url, title),
                )
            )
            if len(results) >= max_results * 2:
                break
        return results

    async def _rate_limit(self) -> None:
        """Space requests by ~1s to stay inside the free tier's QPS cap."""
        async with self._throttle:
            now = asyncio.get_event_loop().time()
            wait = self._last_sent + self._period - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_sent = asyncio.get_event_loop().time()

    async def close(self) -> None:
        await self._client.aclose()