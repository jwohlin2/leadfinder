"""Configuration loading: config.yaml + environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class QwenConfig(BaseModel):
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    provider: str = "openai_compatible"
    timeout: float = 120.0
    max_tokens: int = 2048
    temperature: float = 0.1
    max_retries: int = 2
    max_evidence_chars: int = 24000

    @property
    def enabled(self) -> bool:
        return bool(self.base_url.strip() and self.model.strip())


class ResearchConfig(BaseModel):
    max_search_queries: int = 20
    max_pages: int = 30
    max_llm_rounds: int = 4
    workers: int = 4
    early_stop: bool = True
    company_confidence_target: float = 0.7
    person_confidence_target: float = 0.6
    company_timeout: float = 420.0


class CrawlerConfig(BaseModel):
    timeout: float = 20.0
    max_redirects: int = 5
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    respect_robots: bool = False
    playwright_fallback: bool = True
    playwright_min_chars: int = 600


class SearchConfig(BaseModel):
    provider: str = "searxng"
    base_url: str = "http://localhost:8888"
    format: str = "json"
    fallback_providers: list[str] = Field(default_factory=lambda: ["duckduckgo"])
    timeout: float = 20.0
    max_results: int = 12
    fetch_top_n: int = 6
    # Brave Search API (free tier: 1 QPS, ~2k queries/month). `api_key` comes
    # from LEADFINDER_BRAVE_API_KEY. `max_queries_per_run` is a safety cap so a
    # 98-row benchmark cannot silently drain the monthly quota.
    api_key: str = ""
    max_queries_per_run: int = 400
    # Which upstream engines SearXNG should use. Free engines rate-limit
    # aggressively from a single IP, so pin the ones that actually answer
    # instead of letting SearXNG fan out over ~50 that mostly 429.
    engines: list[str] = Field(default_factory=lambda: ["bing", "mwmbl", "yep"])


class StorageConfig(BaseModel):
    database: str = "data/output/leadfinder.db"
    cache_ttl_hours: int = 336


class OutputConfig(BaseModel):
    directory: str = "data/output"
    write_html_report: bool = True
    write_json: bool = True
    write_csv: bool = True


class Config(BaseModel):
    qwen: QwenConfig = Field(default_factory=QwenConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    root: Path = REPO_ROOT

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (self.root / path)


_ENV_MAP: dict[tuple[str, str], str] = {
    ("qwen", "base_url"): "LEADFINDER_QWEN_BASE_URL",
    ("qwen", "model"): "LEADFINDER_QWEN_MODEL",
    ("qwen", "api_key"): "LEADFINDER_QWEN_API_KEY",
    ("qwen", "provider"): "LEADFINDER_QWEN_PROVIDER",
    ("search", "base_url"): "LEADFINDER_SEARCH_BASE_URL",
    ("search", "provider"): "LEADFINDER_SEARCH_PROVIDER",
    ("search", "format"): "LEADFINDER_SEARCH_FORMAT",
    ("search", "engines"): "LEADFINDER_SEARCH_ENGINES",
    ("search", "fallback_providers"): "LEADFINDER_SEARCH_FALLBACK_PROVIDERS",
    ("search", "api_key"): "LEADFINDER_BRAVE_API_KEY",
    ("search", "max_queries_per_run"): "LEADFINDER_BRAVE_MAX_QUERIES",
    ("research", "workers"): "LEADFINDER_WORKERS",
    ("research", "max_search_queries"): "LEADFINDER_MAX_SEARCH_QUERIES",
    ("research", "max_pages"): "LEADFINDER_MAX_PAGES",
    ("research", "max_llm_rounds"): "LEADFINDER_MAX_LLM_ROUNDS",
}

_ENV_CASTS: dict[tuple[str, str], str] = {
    ("research", "workers"): "int",
    ("research", "max_search_queries"): "int",
    ("research", "max_pages"): "int",
    ("research", "max_llm_rounds"): "int",
    ("search", "engines"): "list",
    ("search", "fallback_providers"): "list",
    ("search", "max_queries_per_run"): "int",
}


def _load_dotenv() -> None:
    """Minimal .env loader - avoids a python-dotenv dependency."""
    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> Config:
    _load_dotenv()

    raw: dict[str, Any] = {}
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded

    for (section, key), env_name in _ENV_MAP.items():
        env_value = os.environ.get(env_name)
        if env_value is None or env_value.strip() == "":
            continue
        cast = _ENV_CASTS.get((section, key))
        if cast == "int":
            try:
                env_value = int(env_value)
            except ValueError:
                continue
        elif cast == "list":
            env_value = [item.strip() for item in env_value.split(",") if item.strip()]
        raw.setdefault(section, {})
        raw[section][key] = env_value

    raw.pop("root", None)
    return Config(root=REPO_ROOT, **raw)
