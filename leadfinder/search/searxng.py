"""SearXNG search provider (JSON API)."""

from __future__ import annotations

import httpx

from leadfinder.search.base import SearchProvider, SearchResult, classify_source_type


class SearxngProvider(SearchProvider):
    name = "searxng"

    def __init__(
        self,
        base_url: str,
        timeout: float = 20.0,
        language: str = "auto",
        response_format: str = "json",
        engines: list[str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.language = language
        # SearXNG only serves JSON when "json" is listed in search.formats.
        self.response_format = response_format or "json"
        self.engines = [e for e in (engines or []) if e]
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Accept": "application/json", "User-Agent": "leadfinder/0.1"},
            follow_redirects=True,
        )

    async def available(self) -> bool:
        try:
            response = await self._client.get(
                f"{self.base_url}/search",
                params={
                    "q": "leadfinder health check",
                    "format": self.response_format,
                },
                timeout=min(self.timeout, 10.0),
            )
            return response.status_code == 200
        except Exception:
            return False

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        params = {
            "q": query,
            "format": self.response_format,
            "language": self.language,
            "safesearch": 0,
        }
        if self.engines:
            params["engines"] = ",".join(self.engines)
        try:
            response = await self._client.get(f"{self.base_url}/search", params=params)
        except Exception:
            return []
        if response.status_code != 200:
            return []
        try:
            payload = response.json()
        except ValueError:
            return []
        raw_results = payload.get("results") or []
        results: list[SearchResult] = []
        for position, item in enumerate(raw_results, start=1):
            url = (item.get("url") or "").strip()
            if not url or not url.startswith("http"):
                continue
            title = (item.get("title") or "").strip()
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=(item.get("content") or item.get("title") or "").strip(),
                    engine=(item.get("engine") or "searxng").strip(),
                    position=position,
                    score=float(item.get("score") or 0.0),
                    source_type=classify_source_type(url, title),
                )
            )
            if len(results) >= max_results * 2:
                break
        return results

    async def close(self) -> None:
        await self._client.aclose()
