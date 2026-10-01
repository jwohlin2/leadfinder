"""Free, zero-cost HTTP search fallbacks.

Used when SearXNG is not up yet. These are deliberately simple scrapes of the
public HTML endpoints - no API keys, no paid service. Treat them as a
temporary bridge, not the long-term answer: SearXNG is the real provider.
"""

from __future__ import annotations

import html as html_lib
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

from leadfinder.search.base import (
    SearchProvider,
    SearchResult,
    classify_source_type,
    score_result,
)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _unwrap_ddg(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


class DuckDuckGoProvider(SearchProvider):
    """Public HTML endpoint at html.duckduckgo.com - no key required."""

    name = "duckduckgo"

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

    async def available(self) -> bool:
        try:
            response = await self._client.post(
                "https://html.duckduckgo.com/html/", data={"q": "leadfinder"}
            )
            return response.status_code == 200 and "result" in response.text
        except Exception:
            return False

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        try:
            response = await self._client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "wt-wt"},
            )
        except Exception:
            return []
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "lxml")
        results: list[SearchResult] = []
        for position, node in enumerate(soup.select("div.result, div.web-result"), start=1):
            anchor = node.select_one("a.result__a") or node.select_one("a[href]")
            if anchor is None:
                continue
            url = _unwrap_ddg(anchor.get("href", "").strip())
            if not url.startswith("http"):
                continue
            title = html_lib.unescape(anchor.get_text(" ", strip=True))
            snippet_node = node.select_one(".result__snippet")
            snippet = html_lib.unescape(snippet_node.get_text(" ", strip=True)) if snippet_node else ""
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    engine="duckduckgo",
                    position=position,
                    source_type=classify_source_type(url, title),
                )
            )
            if len(results) >= max_results * 2:
                break
        return results

    async def close(self) -> None:
        await self._client.aclose()


class BingProvider(SearchProvider):
    """Public HTML results page."""

    name = "bing"

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
        )

    async def available(self) -> bool:
        try:
            response = await self._client.get("https://www.bing.com/search?q=leadfinder")
            return response.status_code == 200 and "b_algo" in response.text
        except Exception:
            return False

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        try:
            response = await self._client.get(
                "https://www.bing.com/search",
                params={"q": query, "count": max_results * 2, "setlang": "en"},
            )
        except Exception:
            return []
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "lxml")
        results: list[SearchResult] = []
        for position, node in enumerate(soup.select("li.b_algo"), start=1):
            anchor = node.select_one("h2 a[href]") or node.select_one("a[href]")
            if anchor is None:
                continue
            url = anchor.get("href", "").strip()
            if not url.startswith("http"):
                continue
            title = html_lib.unescape(anchor.get_text(" ", strip=True))
            caption = node.select_one(".b_caption p") or node.select_one("p")
            snippet = html_lib.unescape(caption.get_text(" ", strip=True)) if caption else ""
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    engine="bing",
                    position=position,
                    source_type=classify_source_type(url, title),
                )
            )
            if len(results) >= max_results * 2:
                break
        return results

    async def close(self) -> None:
        await self._client.aclose()


class SearchClient:
    """Facade over one primary provider plus ordered fallbacks, with caching.

    A provider that returns nothing is treated as unavailable for the rest of
    the process, so a run degrades once rather than paying the timeout on every
    single query.
    """

    def __init__(
        self,
        primary: SearchProvider,
        fallbacks: list[SearchProvider] | None = None,
        cache=None,
        max_results: int = 12,
    ) -> None:
        self.providers = [primary, *(fallbacks or [])]
        self.cache = cache
        self.max_results = max_results
        self.active: list[SearchProvider] = []
        self.stats = {"searches": 0, "cache_hits": 0, "failovers": 0}
        self._degraded: set[str] = set()

    async def probe(self) -> None:
        self.active = []
        for provider in self.providers:
            if provider.name in self._degraded:
                continue
            if await provider.available():
                self.active.append(provider)
        if self.active and self.active[0] is not self.providers[0]:
            self.stats["failovers"] += 1

    @property
    def provider_name(self) -> str:
        return self.active[0].name if self.active else "none"

    async def search(self, query: str, terms: list[str] | None = None) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []
        if not self.active:
            await self.probe()

        cached = self.cache.get(self.provider_name, query) if self.cache else None
        if cached is not None:
            self.stats["cache_hits"] += 1
            self.stats["searches"] += 1
            return self._rank([SearchResult.from_dict(item) for item in cached], terms or [])

        for provider in self.active:
            results = await provider.search(query, max_results=self.max_results)
            if results:
                self.stats["searches"] += 1
                if self.cache:
                    self.cache.put(provider.name, query, [r.to_dict() for r in results])
                return self._rank(results, terms or [])
            self._degraded.add(provider.name)

        self.stats["searches"] += 1
        return []

    @staticmethod
    def _rank(results: list[SearchResult], terms: list[str]) -> list[SearchResult]:
        for result in results:
            result.score = score_result(result, terms)
        results.sort(key=lambda item: (-item.score, item.position))
        return results

    async def close(self) -> None:
        for provider in self.providers:
            await provider.close()


__all__ = [
    "BingProvider",
    "DuckDuckGoProvider",
    "SearchClient",
]
