"""HTTP crawler with SQLite-backed page cache.

Stage 1 of the research strategy: try the supplied domain first, discover
relevant internal links from the homepage, and only then fetch what looks
useful. Never assume /about or /team exist.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from leadfinder.crawl.extract import (
    PageFacts,
    extract_facts,
    is_contact_page,
)
from leadfinder.models.evidence import utcnow
from leadfinder.search.base import classify_source_type

# Link text / href fragments that suggest a page carries identity, people or
# contact information. Weighted, not a hard whitelist.
_LINK_WEIGHTS: list[tuple[tuple[str, ...], float]] = [
    (("about", "über-uns", "ueber-uns", "quem-somos", "chi-siamo", "a-propos",
      "会社概要", "公司介绍", "企業情報", "about-us", "over-ons", "sur-nous",
      "acerca-de", "sobre-nos", "o-nas", "о нас", "om-os", "aboutus"), 3.0),
    (("team", "people", "mitarbeiter", "equipo", "équipe", "staff", "leadership",
      "management", "our-people", "chi-siamo", "中国社会", "スタッフ", "社員",
      "経営陣", "团队", "组织", "团队", " 멤버", "τομάς", "керів", "management",
      "board", "directory", "member", "membership", "faculty", "tutor",
      "doctors", "teachers", "speakers", "partners", "founders"), 2.6),
    (("contact", "kontakt", "contacto", "contatti", "contato", "get-in-touch",
      "inquiry", "enquiry", "reach-us", "iletisim", "お問い合わせ", "名录",
      "联系", "聯絡", "-contact", "support", "help", "reach", "contactez",
      "napisz", "porcontacto", "fale-conosco"), 2.4),
    (("impressum", "mentions-legales", "mentions-légales", "legal-notice",
      "legal", "colofon", "attribution", "特定商取引法", "会社情報", "약관",
      "qui-sommes-nous", "aviso-legal", "informazioni", "informazioni-legali",
      "regeln", "mentions-legales", "note-legali", "yousa-privacy", "legal-notice"), 1.8),
    (("privacy", "terms", "conditions", "tos", "agb", "cookie", "gdpr", "disclaimer",
      " recruiting", "careers", "jobs", "work-with-us", "採用情報", "recruit",
      "採用", "招聘", "privacy-policy", "privacy-notice"), 1.4),
    (("company", "corporate", "organization", "organisation", "profile",
      "company-profile", "corporate-profile", "wir-ueber-uns", "unternehmen",
      "azienda", "societe", "société", "empresa", "会社概要", "会社情報"), 2.2),
    (("press", "news", "media", "blog", "newsroom", "news-room", "presse",
      "actualites", "noticias", "ニュース", "お知らせ", "新闻", "media-centre",
      "resources", "case-stud", "customers", "client", "kunde", "clients"), 1.0),
]


@dataclass(slots=True)
class Page:
    url: str
    final_url: str = ""
    status_code: int | None = None
    title: str = ""
    text: str = ""
    html: str = ""
    source_type: str = "other"
    via: str = "http"
    error: str = ""
    retrieved_at: str = field(default_factory=utcnow)
    links: list[str] = field(default_factory=list)
    facts: PageFacts = field(default_factory=PageFacts)
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 400 and not self.error

    @property
    def host(self) -> str:
        return (urlparse(self.final_url or self.url).hostname or "").lower().removeprefix("www.")

    @property
    def length(self) -> int:
        return len(self.text or "")

    def to_cache_payload(self) -> dict:
        return {
            "final_url": self.final_url,
            "status_code": self.status_code,
            "title": self.title,
            "text": self.text,
            "html": self.html,
            "source_type": self.source_type,
            "error": self.error,
        }

    @classmethod
    def from_cache_payload(cls, url: str, payload: dict) -> "Page":
        page = cls(
            url=url,
            final_url=payload.get("final_url") or url,
            status_code=payload.get("status_code"),
            title=payload.get("title") or "",
            text=payload.get("text") or "",
            html=payload.get("html") or "",
            source_type=payload.get("source_type") or "other",
            error=payload.get("error") or "",
            via="cache",
            from_cache=True,
        )
        page.links = _extract_links(url, page.final_url, page.html)
        page.facts = extract_facts(page.final_url or url, page.html, page.text)
        return page


def _extract_links(base_url: str, final_url: str, html: str) -> list[str]:
    if not html:
        return []
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.lower().startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        target = urljoin(final_url or base_url, href)
        parsed = urlparse(target)
        if parsed.scheme not in {"http", "https"}:
            continue
        cleaned = target.split("#")[0]
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _clean_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.split("#")[0]


def _link_weight(url: str, anchor_text: str = "") -> float:
    blob = f"{url} {anchor_text}".lower()
    weight = 0.0
    for needles, value in _LINK_WEIGHTS:
        if any(needle in blob for needle in needles):
            weight = max(weight, value)
            break
    return weight


def _extract_text(html: str, url: str) -> tuple[str, str]:
    """Return (title, main text). Falls back to raw tag stripping."""
    from bs4 import BeautifulSoup

    title = ""
    text = ""
    if not html:
        return title, text
    try:
        import trafilatura

        text = (
            trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                favor_precision=False,
                no_fallback=False,
            )
            or ""
        ).strip()
    except Exception:
        text = ""

    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = str(og["content"]).strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)

    if len(text) < 200:
        for node in soup(["script", "style", "noscript", "svg", "template"]):
            node.decompose()
        fallback = re.sub(r"[ \t]+", " ", soup.get_text("\n", strip=True))
        if len(fallback) > len(text):
            text = fallback
    return title, text


class Crawler:
    def __init__(self, config, cache=None) -> None:
        self.config = config
        self.cache = cache
        self.stats = {"fetched": 0, "cache_hits": 0, "errors": 0, "fallbacks": 0}
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(4)
        self._robots: dict[str, RobotFileParser | None] = {}
        self._playwright = None
        self._links_by_page: dict[str, list[tuple[str, str]]] = {}

    # -- lifecycle -------------------------------------------------------
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.crawler.timeout,
                follow_redirects=True,
                max_redirects=self.config.crawler.max_redirects,
                headers={
                    "User-Agent": self.config.crawler.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9,ja;q=0.8,zh-CN;q=0.8,de;q=0.7,it;q=0.7,es;q=0.7",
                },
                http2=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    # -- robots ----------------------------------------------------------
    async def _allowed(self, url: str) -> bool:
        if not self.config.crawler.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = RobotFileParser()
            try:
                response = await self._get_client().get(f"{origin}/robots.txt", timeout=8.0)
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    parser = None
            except Exception:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.config.crawler.user_agent, url)

    # -- single fetch ----------------------------------------------------
    async def fetch(self, url: str, *, source_type: str | None = None) -> Page:
        url = _clean_url(url)
        if not url:
            return Page(url=url, error="empty url")

        cached = self.cache.get(url) if self.cache else None
        if cached is not None:
            self.stats["cache_hits"] += 1
            page = Page.from_cache_payload(url, cached)
            page.via = "cache"
            return page

        if not await self._allowed(url):
            return Page(url=url, error="blocked by robots.txt", via="http")

        page = await self._http_fetch(url, source_type)
        if (
            not page.ok
            or len(page.text) < self.config.crawler.playwright_min_chars
        ) and self.config.crawler.playwright_fallback:
            from leadfinder.crawl.playwright_fallback import render_with_playwright

            rendered = await render_with_playwright(url, self)
            if rendered is not None and len(rendered.text) > len(page.text):
                self.stats["fallbacks"] += 1
                page = rendered

        if self.cache is not None and (page.ok or page.error):
            self.cache.put(url, page.to_cache_payload())
        if not page.ok:
            self.stats["errors"] += 1
        else:
            self.stats["fetched"] += 1
        return page

    async def _http_fetch(self, url: str, source_type: str | None) -> Page:
        async with self._semaphore:
            try:
                response = await self._get_client().get(url)
            except Exception as exc:
                return Page(url=url, error=f"{type(exc).__name__}: {exc}"[:300], via="http")

        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type:
            text = await self._pdf_text(response.content)
            page = Page(
                url=url,
                final_url=str(response.url),
                status_code=response.status_code,
                title=url.rsplit("/", 1)[-1],
                text=text,
                source_type="pdf",
                via="http",
            )
            page.facts = extract_facts(page.final_url, "", text)
            return page

        try:
            html = response.text
        except Exception as exc:
            return Page(url=url, status_code=response.status_code,
                        error=f"decode: {type(exc).__name__}", via="http")

        final_url = str(response.url)
        title, text = _clean_text(html, final_url)
        page = Page(
            url=url,
            final_url=final_url,
            status_code=response.status_code,
            title=title,
            text=text,
            html=html,
            source_type=source_type or classify_source_type(final_url, title),
            via="http",
        )
        page.links = _extract_links(url, final_url, html)
        page.facts = extract_facts(final_url, html, text)
        self._remember_links(final_url, html)
        return page

    @staticmethod
    async def _pdf_text(content: bytes) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
            import io

            reader = PdfReader(io.BytesIO(content))
            return "\n".join((page.extract_text() or "") for page in reader.pages)[:200000]
        except Exception:
            return ""

    def _remember_links(self, base_url: str, html: str) -> None:
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return
        pairs: list[tuple[str, str]] = []
        for anchor in soup.find_all("a", href=True):
            href = str(anchor["href"]).strip()
            if not href or href.lower().startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            pairs.append((urljoin(base_url, href), anchor.get_text(" ", strip=True)[:120]))
        self._links_by_page[base_url] = pairs

    # -- stage 1: crawl the supplied domain ------------------------------
    def discover_internal(
        self,
        home_page: Page,
        domain: str,
        *,
        limit: int = 8,
    ) -> list[tuple[str, float]]:
        """Rank same-host links by how likely they are to carry useful signal."""
        root = domain.lower().removeprefix("www.")
        pairs = self._links_by_page.get(home_page.final_url or home_page.url) or []
        if not pairs and home_page.html:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(home_page.html, "lxml")
            base = home_page.final_url or home_page.url
            pairs = [
                (urljoin(base, a["href"]), a.get_text(" ", strip=True)[:120])
                for a in soup.find_all("a", href=True)
                if a.get("href") and not str(a["href"]).lower().startswith(
                    ("javascript:", "mailto:", "tel:", "#")
                )
            ]

        scored: dict[str, float] = {}
        for url, anchor_text in pairs:
            clean = url.split("#")[0]
            host = (urlparse(clean).hostname or "").lower().removeprefix("www.")
            if host != root:
                continue
            if re.search(r"\.(pdf|zip|jpg|jpeg|png|gif|svg|webp|mp4|mp3|docx?|xlsx?)$",
                         clean, re.I):
                continue
            if clean in {home_page.final_url, home_page.url}:
                continue
            weight = _link_weight(clean, anchor_text)
            if weight <= 0:
                continue
            # Slight preference for shallower paths and short hrefs (menu links).
            depth = clean.count("/") - 2
            weight += max(0.0, 0.6 - depth * 0.2)
            weight += 0.4 if anchor_text and len(anchor_text) < 40 else 0.0
            scored[clean] = max(scored.get(clean, 0.0), weight)

        ranked = sorted(scored.items(), key=lambda item: -item[1])
        return ranked[:limit]

    async def crawl_domain(
        self,
        domain: str,
        *,
        max_pages: int = 8,
    ) -> list[Page]:
        """Homepage first, then the highest-signal internal pages."""
        home = await self.fetch(f"https://{domain}")
        if not home.ok:
            home = await self.fetch(f"http://{domain}")
        pages = [home]
        if not home.ok:
            return pages

        candidates = self.discover_internal(home, domain, limit=max(0, max_pages - 1))
        if candidates:
            fetched = await asyncio.gather(
                *(self.fetch(url) for url, _ in candidates),
                return_exceptions=True,
            )
            for page in fetched:
                if isinstance(page, Page) and page.ok:
                    pages.append(page)
        return pages


def _clean_text(html: str, url: str) -> tuple[str, str]:
    return _extract_text(html, url)


def page_is_contact(page: Page) -> bool:
    return is_contact_page(page.final_url or page.url, page.title, page.text[:2000])


def crawl_time(page: Page) -> float:
    return time.time()
