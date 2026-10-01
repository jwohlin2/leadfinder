"""Search provider interface.

Pluggable by design: SearXNG is the target, but a $0 free HTTP fallback keeps
the benchmark runnable while a container is still starting up.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urlparse

from leadfinder.models.evidence import SourceType

# Hosts that are almost never worth the crawl budget for lead research.
_LOW_VALUE_HOSTS = {
    "pinterest.com", "quora.com", "reddit.com", "youtube.com", "facebook.com",
    "instagram.com", "twitter.com", "x.com", "tiktok.com", "linkedin.com",
    "wikipedia.org", "wikidata.org", "amazon.com", "ebay.com", "alibaba.com",
    "indeed.com", "glassdoor.com", "yelp.com", "tripadvisor.com", "foursquare.com",
    "crunchbase.com", "medium.com", "blogspot.com", "wordpress.com",
    "translate.google.com", "webcache.googleusercontent.com",
}

_HIGH_VALUE_HOST_SUFFIXES = (
    "producthunt.com", "github.com", "gitlab.com", "prnewswire.com",
    "businesswire.com", "globenewswire.com", "europa.eu", "gov.uk", "go.jp",
    "lg.jp", ".go.jp", ".gob.jp", "unido.org", "wipo.int", "sec.gov",
    "find-and-update.company-information.service.gov.uk", "opencorporates.com",
    "pitchbook.com", "crunchbase.com", "techcrunch.com", "prnewswire.com",
    "linkedin.com", "medium.com", "devpost.com", "ycombinator.com",
    "kiep.uzh.ch", "dealroom.co", "zoominfo.com", "signalhire.com",
)

_GOOD_TLD_SUFFIXES = (".com", ".co", ".io", ".ai", ".jp", ".cn", ".de", ".uk",
                      ".it", ".es", ".nl", ".fr", ".se", ".ch", ".at", ".pt", ".be",
                      ".pl", ".hr", ".cz", ".kr", ".in", ".ca", ".au", ".nz", ".br",
                      ".mx", ".za", ".mt", ".ad", ".cm", ".eu", ".org", ".net")


@dataclass(slots=True)
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    engine: str = ""
    position: int = 0
    score: float = 0.0
    source_type: SourceType = "other"

    @property
    def host(self) -> str:
        return (urlparse(self.url).hostname or "").lower().removeprefix("www.")

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "engine": self.engine,
            "position": self.position,
            "score": self.score,
            "source_type": self.source_type,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SearchResult":
        return cls(
            url=payload.get("url", ""),
            title=payload.get("title", ""),
            snippet=payload.get("snippet", ""),
            engine=payload.get("engine", ""),
            position=int(payload.get("position") or 0),
            score=float(payload.get("score") or 0.0),
            source_type=payload.get("source_type", "other"),
        )


def classify_source_type(url: str, title: str = "") -> SourceType:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    blob = f"{host} {title}".lower()
    rules: list[tuple[SourceType, tuple[str, ...]]] = [
        ("product_hunt", ("producthunt.com",)),
        ("github", ("github.com", "gitlab.com", "bitbucket.org", "gitee.com")),
        ("registry", (
            "opencorporates.com", "company-information.service.gov.uk",
            "national-tax-agency", "houjin.jp", "gBizINFO", "gsxt.gov.cn",
            "unido.org", "europa.eu", "sec.gov", "registers", "commercial-register",
            "germany", "northdata.de", "companyhouse", "or.jp", "registry",
        )),
        ("press_release", (
            "prnewswire.com", "businesswire.com", "globenewswire.com",
            "einpresswire.com", "newsfilecorp.com", "accesswire.com",
        )),
        ("partner_announcement", (
            "partners", "partner.", "integrations", "alliance",
        )),
        ("conference", ("conference", "congress", "summit", "expo", "webinar",
                        "meetup.com", "eventbrite", "mktplace", "tradeshow")),
        ("interview", ("interview", "podcast", "youtube.com", "youtu.be")),
        ("pdf", (".pdf",)),
        ("news", ("news", "times", "post", "herald", "journal", "gazette",
                  "reuters", "bloomberg", "forbes", "techcrunch", "venturebeat")),
        ("social_profile", ("linkedin.com", "x.com", "twitter.com", "facebook.com",
                            "instagram.com", "medium.com", "substack", "note.com")),
        ("job_board", ("linkedin.com/jobs", "indeed", "glassdoor", "wellfound",
                       "angel.co", "otta.com", "wellfound.com")),
        ("directory", ("crunchbase", "pitchbook", "zoominfo", "apollo.io",
                       "clay.com", "signalhire", "leadiq", "apollo", "f6s.com",
                       "producthunt.com/topics", "clutch.co", "g2.com", "capterra")),
    ]
    for source_type, needles in rules:
        if any(needle in blob for needle in needles):
            return source_type
    return "company_website" if _looks_like_company_site(host) else "other"


def _looks_like_company_site(host: str) -> bool:
    if not host:
        return False
    if "." not in host:
        return False
    tld = "." + host.rsplit(".", 1)[-1]
    if host.endswith(_GOOD_TLD_SUFFIXES) or tld in _GOOD_TLD_SUFFIXES:
        return True
    parts = host.split(".")
    # Handle multi-part public suffixes roughly (co.uk, com.au, co.jp).
    if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}" in _GOOD_TLD_SUFFIXES:
        return True
    return False


_SITE_QUERY = re.compile(r"\bsite:([a-z0-9.\-]+)", re.IGNORECASE)


def score_result(result: SearchResult, query_terms: list[str]) -> float:
    """Rank results by expected usefulness for lead research.

    Deliberately simple and inspectable: host trust, title/snippet term overlap,
    and a penalty for social/aggregator noise. The point is to fetch fewer,
    better pages, not to be clever.
    """
    host = result.host
    if not host or host in _LOW_VALUE_HOSTS:
        return 0.0

    score = 1.0
    for suffix in _HIGH_VALUE_HOST_SUFFIXES:
        if host == suffix or host.endswith("." + suffix) or suffix in host:
            score += 3.0
            break

    title_blob = f"{result.title} {result.snippet}".lower()
    matched = sum(1 for term in query_terms if term and term in title_blob)
    score += matched * 1.4

    if _looks_like_company_site(host):
        score += 1.2
    if result.source_type in {
        "company_website", "registry", "press_release", "partner_announcement",
    }:
        score += 1.0
    if result.source_type in {"directory", "job_board", "forum", "social_profile"}:
        score -= 1.5
    if title_blob:
        first = result.title.lower()
        terms = [t for t in query_terms if t]
        if terms and any(first.startswith(t) for t in terms):
            score += 0.8
    if result.position:
        score += max(0.0, 1.5 - result.position * 0.1)
    if host.endswith(".pdf"):
        score -= 0.3
    return round(score, 2)


class SearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        """Return ranked results, or [] if this provider cannot answer."""

    async def available(self) -> bool:
        return True

    async def close(self) -> None:  # pragma: no cover - optional
        return None
