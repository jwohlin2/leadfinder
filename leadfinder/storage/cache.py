"""Search and page caches.

Implemented immediately: a 98-company benchmark re-hits the same corporate
pages constantly (registry listings, Product Hunt, partner pages), and a
second run should be nearly free.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from leadfinder.models.evidence import utcnow
from leadfinder.storage.db import Database

_QUERY_KEY_SALT = "lf1"


def _hash(value: str) -> str:
    return hashlib.sha256(f"{_QUERY_KEY_SALT}:{value}".encode("utf-8")).hexdigest()[:32]


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class SearchCache:
    def __init__(self, db: Database, ttl_hours: int = 336) -> None:
        self.db = db
        self.ttl = timedelta(hours=ttl_hours)
        self.hits = 0
        self.misses = 0

    def _fresh(self, fetched_at: str | None) -> bool:
        stamp = _parse_ts(fetched_at)
        return stamp is not None and datetime.now(timezone.utc) - stamp <= self.ttl

    def get(self, provider: str, query: str) -> list[dict[str, Any]] | None:
        row = self.db.query_one(
            "SELECT results_json, fetched_at FROM search_cache "
            "WHERE provider = ? AND query_key = ?",
            (provider, _hash(query)),
        )
        if row is None:
            self.misses += 1
            return None
        if not self._fresh(row["fetched_at"]):
            self.misses += 1
            return None
        try:
            payload = json.loads(row["results_json"] or "[]")
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list):
            return None
        self.hits += 1
        return payload

    def put(self, provider: str, query: str, results: list[dict[str, Any]]) -> None:
        payload = json.dumps(results, ensure_ascii=False)
        self.db.execute(
            "INSERT INTO search_cache "
            "(provider, query_key, query, results_json, result_count, created_at, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(provider, query_key) DO UPDATE SET "
            "results_json = excluded.results_json, "
            "result_count = excluded.result_count, "
            "fetched_at = excluded.fetched_at",
            (provider, _hash(query), query, payload, len(results), utcnow(), utcnow()),
        )


class PageCache:
    def __init__(self, db: Database, ttl_hours: int = 336) -> None:
        self.db = db
        self.ttl = timedelta(hours=ttl_hours)
        self.hits = 0
        self.misses = 0

    def get(self, url: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT url, final_url, status_code, title, text, html, source_type, error, fetched_at "
            "FROM page_cache WHERE url_key = ?",
            (_hash(url),),
        )
        if row is None:
            self.misses += 1
            return None
        stamp = _parse_ts(row["fetched_at"])
        if stamp is None or datetime.now(timezone.utc) - stamp > self.ttl:
            self.misses += 1
            return None
        self.hits += 1
        return dict(row)

    def put(self, url: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO page_cache "
            "(url_key, url, final_url, status_code, title, text, html, source_type, error, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(url_key) DO UPDATE SET "
            "final_url = excluded.final_url, status_code = excluded.status_code, "
            "title = excluded.title, text = excluded.text, html = excluded.html, "
            "source_type = excluded.source_type, error = excluded.error, "
            "fetched_at = excluded.fetched_at",
            (
                _hash(url),
                url,
                payload.get("final_url") or "",
                payload.get("status_code"),
                payload.get("title") or "",
                payload.get("text") or "",
                payload.get("html") or "",
                payload.get("source_type") or "other",
                payload.get("error") or "",
                utcnow(),
            ),
        )

    @property
    def total_hits(self) -> int:
        return self.hits
