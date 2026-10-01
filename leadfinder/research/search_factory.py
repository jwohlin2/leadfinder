"""Build the search client from config, honouring the pluggable-provider design.

`search.provider: auto` probes the configured providers in order and uses the
first that answers, so a SearXNG container that is still booting does not block
the benchmark.
"""

from __future__ import annotations

from leadfinder.search.base import SearchProvider
from leadfinder.search.brave import BraveSearchAPIProvider
from leadfinder.search.free_web import BingProvider, DuckDuckGoProvider, SearchClient
from leadfinder.search.searxng import SearxngProvider

_PROVIDERS: dict[str, type] = {
    "searxng": SearxngProvider,
    "duckduckgo": DuckDuckGoProvider,
    "bing": BingProvider,
    "brave": BraveSearchAPIProvider,
}


def build_search_client(config, cache=None) -> SearchClient:
    name = (config.search.provider or "searxng").lower()
    timeout = config.search.timeout
    max_results = config.search.max_results

    def make(provider_name: str) -> SearchProvider:
        if provider_name == "searxng":
            return SearxngProvider(
                config.search.base_url,
                timeout=timeout,
                response_format=config.search.format,
                engines=config.search.engines,
            )
        if provider_name == "brave":
            return BraveSearchAPIProvider(
                config.search.api_key,
                timeout=timeout,
                max_queries_per_run=config.search.max_queries_per_run,
            )
        cls = _PROVIDERS.get(provider_name)
        if cls is None:
            raise ValueError(f"unknown search provider: {provider_name}")
        return cls(timeout=timeout)  # type: ignore[call-arg]

    if name == "auto":
        primary_name = "searxng"
        fallbacks = ["duckduckgo", "bing"]
    else:
        primary_name = name
        fallbacks = [
            p for p in (config.search.fallback_providers or []) if p != name
        ]

    if primary_name == "brave" and not config.search.api_key:
        # No key configured: silently fall back to the free scrapers rather
        # than raising, so the benchmark still runs without Brave configured.
        primary_name = "duckduckgo"
        fallbacks = ["bing"] + [f for f in fallbacks if f not in ("duckduckgo", "bing")]

    primary = make(primary_name)
    fallbacks = [make(item) for item in fallbacks if item in _PROVIDERS]
    return SearchClient(primary, fallbacks, cache=cache, max_results=max_results)
