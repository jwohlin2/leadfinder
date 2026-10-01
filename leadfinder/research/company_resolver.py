"""Stage 1-3: crawl the domain, search, and resolve the real company.

The general failure family this exists to solve is *brand -> legal entity*
mismatch: the supplied name is a product, project or local trade name, and the
company that actually employs the decision-makers sits behind a different name,
often in a different country. Nothing here is company-specific.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from leadfinder.crawl.crawler import Crawler, Page
from leadfinder.llm import prompts as P
from leadfinder.llm.qwen import LLMCallFailed, LLMUnavailable
from leadfinder.models.company import CompanyIdentity, CompanyInput, name_is_degraded
from leadfinder.models.evidence import Evidence, Source
from leadfinder.models.person import infer_size_hint
from leadfinder.research.budget import Budget
from leadfinder.search.base import SearchResult, classify_source_type

# Deterministic Stage-2 queries. No LLM required.
_BASE_QUERY_TEMPLATES = [
    '"{name}"',
    '"{name}" founder',
    '"{name}" CEO',
    '"{name}" owner',
    '"{name}" director',
    '"{name}" team',
    '"{name}" contact',
    '"{domain}"',
    "site:{domain} about",
    "site:{domain} team",
    "site:{domain} contact",
    "site:{domain} impressum",
    "site:{domain} legal",
]

# Ordered so a truncated budget still covers the highest-value angles first.
_PRIORITY_ROLES = ("founder", "ceo", "owner", "president", "director")


def build_initial_queries(item: CompanyInput) -> list[str]:
    """Deterministic first-round queries.

    A name mangled by the CSV export ('FlexIN???????') is not searchable, so the
    domain and role angles are promoted in that case instead of querying the
    literal junk.
    """
    name = item.company_name.strip()
    domain = item.domain.lower().removeprefix("www.")
    degraded = name_is_degraded(name)
    clean = re.sub(r"\s+", " ", name).strip()

    queries: list[str] = []
    if clean and not degraded:
        queries.append(f'"{clean}"')
        for role in _PRIORITY_ROLES:
            queries.append(f'"{clean}" {role}')
        queries.append(f'"{clean}" team')
    if domain:
        queries.append(f'"{domain}"')
        for suffix in ("about", "team", "contact", "impressum", "legal", "privacy"):
            queries.append(f"site:{domain} {suffix}")
    if degraded and domain:
        # The name is unusable; lean on the domain plus generic role language.
        for role in _PRIORITY_ROLES:
            queries.append(f"{domain} {role}")
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


def query_terms(item: CompanyInput, resolved: CompanyIdentity) -> list[str]:
    """Tokens used to rank search results for this company."""
    terms: list[str] = []
    for value in (
        resolved.canonical_name,
        resolved.legal_name,
        item.company_name,
        item.domain,
        *(resolved.aliases or []),
    ):
        cleaned = re.sub(r"[^\w\s.\-]", " ", (value or "").lower())
        for token in cleaned.split():
            if len(token) > 2 and token not in terms:
                terms.append(token)
    return terms[:10]


@dataclass(slots=True)
class EntityReport:
    identity: CompanyIdentity = field(default_factory=CompanyIdentity)
    pages: list[Page] = field(default_factory=list)
    snippets: list[tuple[str, str]] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    domain_ok: bool = False
    domain_error: str = ""
    llm_used: bool = False
    llm_note: str = ""


# Tokens that carry no identifying signal inside a company name. A legal name
# must share a *distinctive* token to be attributed to the target company.
_LEGAL_NOISE = {
    "ltd", "limited", "inc", "llc", "corp", "corporation", "gmbh", "ag", "sa",
    "sas", "sarl", "sl", "bv", "nv", "plc", "pty", "pte", "ab", "oy", "aps",
    "sro", "srl", "doo", "sp", "zoo", "the", "and", "of", "company", "co",
    "group", "holdings", "holding", "international", "services", "service",
    "solutions", "technologies", "software", "systems", "consulting", "lab",
    "labs", "ltda", "sau",
}


def _distinctive_tokens(*values: str) -> set[str]:
    out: set[str] = set()
    for value in values:
        for token in re.split(r"[^\w]+", (value or "").lower()):
            if len(token) >= 3 and token not in _LEGAL_NOISE:
                out.add(token)
    return out


def _shares_distinctive_token(candidate: str, item: "CompanyInput") -> bool:
    """True when a legal name overlaps the company name, domain or aliases."""
    target = _distinctive_tokens(item.company_name, item.domain)
    # A registrable domain is often the brand name (availroom.com -> availroom).
    for part in re.split(r"[.\-]", item.domain or ""):
        if len(part) >= 3:
            target.add(part.lower())
    return bool(_distinctive_tokens(candidate) & target)


def _same_registrable_domain(url: str, domain: str) -> bool:
    host = (urlparse(url or "").hostname or "").lower().removeprefix("www.")
    target = (domain or "").lower().removeprefix("www.")
    if not host or not target:
        return False
    return host == target or host.endswith("." + target) or target.endswith("." + host)


def _source_for(page: Page) -> Source:
    url = page.final_url or page.url
    return Source(
        url=url,
        title=page.title or url,
        source_type=classify_source_type(url, page.title),
        retrieved_at=page.retrieved_at,
    )


# Search engines return large multi-tenant pages (a partner directory with 200
# companies, a language-mirrored app listing) that cost a page-budget slot and
# contribute almost nothing. They crowd out the pages that actually name the
# company or its people.
_LOW_VALUE_URL = re.compile(
    r"(/tag/|/tags/|/category/|/categories/|/topics?/|/search\?|/page/\d|"
    r"/blog/\d|/archive|\.pdf($|\?)|/print/?$)",
    re.I,
)


def _worth_fetching(url: str, item: "CompanyInput") -> bool:
    """Cheap guard against spending the page budget on low-yield URLs."""
    if _LOW_VALUE_URL.search(url or ""):
        return False
    host = (urlparse(url or "").hostname or "").lower()
    if not host:
        return False
    # Google's Play listings are mirrored per locale, so one query returns a
    # dozen near-identical ?hl= variants of the same app page.
    if "play.google.com" in host:
        return False
    return True


def _deterministic_identity(
    item: CompanyInput, pages: list[Page], snippets: list[tuple[str, str]]
) -> CompanyIdentity:
    """Fallback / floor when the LLM is unavailable or unconfident.

    Collects the deterministic signals that are still worth asserting on their
    own: legal names found in page text, the copyright entity, the reachable
    host, and an active/inactive read from HTTP status.
    """
    identity = CompanyIdentity(
        canonical_name=item.company_name,
        domain=item.domain,
        status="unknown",
    )
    all_text = "\n".join(page.text for page in pages)
    home = pages[0] if pages else None

    legal_names: list[str] = []
    for page in pages:
        on_domain = _same_registrable_domain(page.final_url or page.url, item.domain)
        for name in page.facts.legal_names:
            # A legal name is only evidence about *this* company if it shares a
            # distinctive token with the target. Searching "TravelBy Software"
            # surfaces unrelated registry filings ("A1 TAXI SERVICES LTD"),
            # and a bare suffix match is not enough to claim ownership.
            if not on_domain and not _shares_distinctive_token(name, item):
                continue
            if name not in legal_names:
                legal_names.append(name)
    if legal_names:
        # Longest legal suffix wins: "TravelBy Software S.L." over "Software S.L."
        best = max(legal_names, key=len)
        identity.legal_name = best
        if len(legal_names) > 1:
            identity.aliases = legal_names[:5]

    if home is not None and home.ok:
        identity.status = "active"
        identity.domain = (urlparse(home.final_url or home.url).hostname or item.domain).lower()

    size_hint = infer_size_hint(all_text)
    if size_hint:
        identity.size_hint = size_hint

    if identity.legal_name and identity.legal_name != identity.canonical_name:
        # A legal entity distinct from the input name is the brand->entity case.
        identity.product_or_company = "brand"
        identity.confidence = 0.55
        identity.resolution_note = (
            f"legal entity '{identity.legal_name}' found in retrieved text; "
            "no LLM resolution available"
        )
    elif identity.status == "active":
        identity.product_or_company = "unknown"
        identity.confidence = 0.4
        identity.resolution_note = "domain reachable; identity unconfirmed by a second source"
    return identity


def _apply_llm_identity(
    identity: CompanyIdentity,
    resolved: P.ResolvedEntity,
    known_urls: set[str],
) -> CompanyIdentity:
    """Overlay LLM output, keeping only claims that cite a retrieved page."""
    valid = [
        url
        for url in resolved.source_urls
        if any(url.strip().rstrip("/") == known.rstrip("/") for known in known_urls)
    ]
    if not valid:
        identity.resolution_note = (
            (identity.resolution_note + "; " if identity.resolution_note else "")
            + "model proposed a resolution without a retrievable source; downgraded"
        )
        identity.confidence = min(identity.confidence, 0.3)
        return identity

    identity.canonical_name = resolved.canonical_name or identity.canonical_name
    identity.legal_name = resolved.legal_name or identity.legal_name
    identity.domain = (resolved.domain or identity.domain).lower().removeprefix("www.")
    identity.country = resolved.country or identity.country
    identity.language = resolved.language or identity.language
    identity.company_type = resolved.company_type or identity.company_type
    identity.product_or_company = resolved.product_or_company or identity.product_or_company
    identity.parent_company = resolved.parent_company
    identity.status = resolved.status if resolved.status != "unknown" else identity.status
    identity.industry = resolved.industry or identity.industry
    identity.size_hint = resolved.size_hint or identity.size_hint
    identity.confidence = max(0.0, min(1.0, resolved.confidence))
    identity.evidence_urls = valid[:8]
    identity.resolution_note = resolved.note or identity.resolution_note
    if resolved.aliases:
        merged = list(identity.aliases)
        for alias in resolved.aliases:
            cleaned = alias.strip()
            if cleaned and cleaned.lower() not in {m.lower() for m in merged}:
                merged.append(cleaned)
        identity.aliases = merged[:8]
    return identity


def identity_evidence(
    identity: CompanyIdentity, item: CompanyInput, pages: list[Page]
) -> list[Evidence]:
    """Evidence rows for the entity claims, each tied to a retrieved page."""
    evidence: list[Evidence] = []
    by_url = {p.final_url or p.url: p for p in pages}

    def add(claim_type: str, value: str, url: str, quote: str, confidence: float) -> None:
        if not value or not url:
            return
        page = by_url.get(url)
        snippet = quote or (page.text[:300] if page else "")
        evidence.append(
            Evidence(
                claim_type=claim_type,  # type: ignore[arg-type]
                claim_value=value,
                source_url=url,
                source_title=page.title if page else "",
                source_text=re.sub(r"\s+", " ", snippet)[:600],
                confidence=round(confidence, 2),
                source_type=classify_source_type(url, page.title if page else ""),
            )
        )

    for url in identity.evidence_urls:
        page = by_url.get(url)
        if page is None:
            continue
        if identity.canonical_name:
            add("company_name", identity.canonical_name, url,
                f"resolved canonical name for {item.domain}", identity.confidence)
        if identity.legal_name:
            quoted = ""
            for name in page.facts.legal_names:
                if name == identity.legal_name:
                    index = page.text.find(name)
                    if index != -1:
                        quoted = page.text[max(0, index - 120) : index + 240]
                    break
            add("legal_name", identity.legal_name, url, quoted, identity.confidence)
        if identity.country:
            add("country", identity.country, url,
                f"country per resolved identity ({identity.language or 'n/a'})",
                identity.confidence * 0.8)
        if identity.status and identity.status != "unknown":
            add("status", identity.status, url, "", 0.6)
        break  # one strong page is enough; extra pages would be noise

    if not evidence and pages:
        home = pages[0]
        add("domain", item.domain, home.final_url or home.url, home.text[:200], 0.5)
    return evidence


async def resolve_entity(
    *,
    item: CompanyInput,
    crawler: Crawler,
    search,
    llm,
    budget: Budget,
    site_page_budget: int = 8,
) -> EntityReport:
    """Stages 1-3 for one company."""
    report = EntityReport()

    # --- Stage 1: the supplied domain -----------------------------------
    pages = await crawler.crawl_domain(item.domain, max_pages=site_page_budget)
    report.pages = pages
    home = pages[0] if pages else None
    report.domain_ok = bool(home and home.ok)
    report.domain_error = home.error if home and not home.ok else ""
    budget.spend_pages(len(pages))

    for page in pages:
        report.sources.append(_source_for(page))

    # --- Stage 2: deterministic searches --------------------------------
    snippets: list[tuple[str, str]] = []
    for query in build_initial_queries(item):
        if budget.exhausted():
            break
        results = await search.search(query, terms=query_terms(item, CompanyIdentity()))
        budget.spend_query(query)
        for result in results[: budget.search_results_per_query]:
            snippets.append((result.url, f"{result.title} - {result.snippet}".strip(" -")))

    # Fetch the most promising off-site results, but keep a reserve of page
    # budget for the adaptive rounds.
    reserve = max(2, int(budget.max_pages * 0.35))
    fetchable = [
        result
        for result in _dedupe_results(snippets)
        if _is_fetchable(result[0], item.domain) and _worth_fetching(result[0], item)
    ]
    offsite = fetchable[: max(0, min(4, budget.remaining_pages - reserve))]
    for url, _ in offsite:
        if budget.exhausted():
            break
        page = await crawler.fetch(url)
        budget.spend_pages(1)
        if page.ok:
            report.pages.append(page)
            report.sources.append(_source_for(page))
            snippets.append((page.final_url or page.url, f"{page.title} - {page.text[:200]}"))

    report.snippets = snippets

    # --- Stage 3: entity resolution -------------------------------------
    identity = _deterministic_identity(item, report.pages, snippets)
    known_urls = {p.final_url or p.url for p in report.pages if p.ok}

    if llm.enabled and report.pages:
        page_payload = [
            {
                "url": p.final_url or p.url,
                "title": p.title,
                "text": p.text,
            }
            for p in report.pages
            if p.ok
        ]
        user = P.build_entity_prompt(
            input_company=item.company_name,
            input_domain=item.domain,
            pages=page_payload,
            search_snippets=[f"{url} :: {text}" for url, text in snippets[:12]],
            char_budget=llm.config.max_evidence_chars,
        )
        try:
            resolved = llm.structured(P.ENTITY_SYSTEM, user, P.ResolvedEntity)
            identity = _apply_llm_identity(identity, resolved, known_urls)
            report.llm_used = True
        except (LLMCallFailed, LLMUnavailable) as exc:
            report.llm_note = str(exc)[:200]

    report.identity = identity
    report.evidence = identity_evidence(identity, item, report.pages)
    return report


def _dedupe_results(snippets: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for url, text in snippets:
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append((url, text))
    return out


def _is_fetchable(url: str, domain: str) -> bool:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host or host == domain.lower().removeprefix("www."):
        return False
    if host.endswith(("facebook.com", "instagram.com", "pinterest.com", "x.com", "twitter.com")):
        return False
    return True


def rank_offsite(results: list[SearchResult], terms: list[str]) -> list[SearchResult]:
    from leadfinder.search.base import score_result

    for result in results:
        result.score = score_result(result, terms)
    results.sort(key=lambda item: (-item.score, item.position))
    return results
