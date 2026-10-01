"""Person discovery, reconciliation and ranking."""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from leadfinder.crawl.crawler import Page
from leadfinder.crawl.extract import (
    _COMPANY_SUFFIX_TOKENS,
    _looks_like_person_name,
    _looks_like_title,
    _NON_PERSON_TOKENS,
    _PRODUCT_TOKENS,
    _ROLE_TOKENS,
    _TITLE_RE,
)
from leadfinder.llm import prompts as P
from leadfinder.llm.qwen import LLMCallFailed, LLMUnavailable
from leadfinder.models.company import CompanyIdentity
from leadfinder.models.evidence import Evidence
from leadfinder.models.person import (
    Person,
    classify_role,
    normalize_person_name,
    person_name_conflict,
    person_name_key,
    score_person,
)
from leadfinder.search.base import classify_source_type

_DEPRECATED = re.compile(r"\b(former|previously|ex-|past co-|until 20\d\d|left in)\b", re.I)

# App stores and directories describe products, not employees: a store listing
# yields "Travel By Taxi | App Store - Apple - The developer", which looks like
# a person with a job title and is neither.
# Tech-stack analyzers (listsignal, builtwith, …) list "Google Fonts", "Cloudflare",
# "Webflow" etc. — product names that pass the name regex. Wikipedia pages are
# reliable people sources for the *subject* of the article, but the harvest scans
# the whole page including discography / credits tables ("Other Performer |
# Director | Album"), which pollutes candidates with non-people rows.
_NON_EMPLOYER_SOURCES = re.compile(
    r"(apps\.apple\.com|play\.google\.com|chromewebstore|apps\.microsoft\.com"
    r"|/app/[^/]+/id\d+|appsumo\.|g2\.com|capterra\.|softwareadvice\.|"
    r"crunchbase\.com|zoominfo\.com|apollo\.io|clay\.com|signalhire\."
    # Programmatic directory pages, where the "name" is a product or the
    # "title" is a UI label ("connectivity partner | ... | Google for Developers").
    r"|developers\.google\.com|developer\.apple\.com|/docs?/|/api/|"
    r"/developers/|/partners?/|/integration"
    # Tech-stack / website-analyzer pages: names are product labels.
    r"|listsignal\.|builtwith\.|wappalyzer\.|hyperping\.|whatcms\.|"
    r"statscrop\.|websiteseocheckers\.|sitechecker\.|browserstack\.|similarweb\.|"
    r"semrush\.com/website|ahrefs\.com/site|spyfu\.com"
    # Encyclopedia-style "credits" pages where table rows are not people.
    r"|wikipedia\.org/wiki/.*(?:discography|chart|charts|awards|category:))",
    re.I,
)


def _is_people_source(url: str, title: str = "") -> bool:
    return not _NON_EMPLOYER_SOURCES.search(f"{url or ''} {title or ''}")


# Department / function words that appear after "head of", "director of", etc.
# Capture "Head of Marketing at Acme" -> org "Marketing at Acme" is a dept
# heading, not a company attribution, so it must not sink a legit employee.
_DEPARTMENT_TOKENS = {
    "marketing", "sales", "operations", "engineering", "product", "products",
    "finance", "financial", "human", "resources", "technology", "information",
    "legal", "strategy", "strategies", "design", "designs", "growth", "revenue",
    "development", "business", "customer", "customers", "account", "accounts",
    "partnership", "partnerships", "people", "operations", "support", "security",
    "data", "delivery", "procurement", "quality", "research", "communications",
    "public", "relations", "brand", "content", "creative", "it", "services",
    "service", "solutions", "analytics", "acquisition", "insights", "sales",
    "supply", "chain", "logistics", "compliance", "risk", "procurement",
}


# A title such as "Managing Director of VOILÀ Hotel Rewards" attributes the
# person to a *different* company than the one being researched (often a partner
# or competitor quoted on the target's own page). Match transitive-attribution
# phrases like "of X", "at X", and reject the person when X is not an expected
# term for the target company. Department names ("Head of Marketing") are single
# role words and are not attributions.
_ATTRIBUTION_NOT = re.compile(
    r"\b(?:co-?founder|founder|ceo|owner|director|managing director|president|"
    r"head|partner|chief executive officer|vice president|general manager)\b"
    r"[^.!?;]*?\b(?:of|at)\b\s+([\w’'&][\w’'&.\-]*"
    r"(?:[\s-][\w’'&][\w’'&.\-]*){1,3})",
    re.I,
)


def _attributed_other_company(text: str, company_terms: list[str]) -> bool:
    """True if the title/context attaches the person to an organisation that is
    clearly not the target company (e.g. a partner's executive quoted on a press
    release). Department headings ("Head of Marketing") never match because the
    captured organisation needs >=2 capitalised proper nouns that are not role,
    product or department vocabulary."""
    if not text or not company_terms:
        return False
    expected = _target_terms(company_terms)
    for match in _ATTRIBUTION_NOT.finditer(text):
        org = match.group(1).strip()
        if not org or org in ("Board", "Division", "Operations", "Product"):
            continue
        lowered = org.lower()
        if any(
            token.lower() in expected for token in org.replace("-", " ").split()
        ):
            continue
        if any(term in lowered for term in expected):
            continue
        pieces = [t.strip(".'’\"") for t in org.replace("-", " ").split()]
        if not pieces:
            continue
        lowered_pieces = {p.lower() for p in pieces}
        # A legal-suffix token ("Inc", "Co", "GmbH") means this is corporate —
        # a genuine company attribution, never a department heading.
        if lowered_pieces & _COMPANY_SUFFIX_TOKENS:
            return True
        # A lowercase function word ("Head of Marketing at Acme" -> org "at")
        # means the regex swallowed a preposition inside a department heading.
        # A real organisation name does not contain "at", "of", "and" etc.
        if any(p.islower() for p in pieces) and not (
            lowered_pieces <= (_NON_PERSON_TOKENS | _ROLE_TOKENS
                               | _PRODUCT_TOKENS | _DEPARTMENT_TOKENS)
        ):
            continue
        # Otherwise it is a department heading only if every word is ordinary
        # role/product/department vocabulary ("Head of Product Marketing").
        if lowered_pieces <= (
            _NON_PERSON_TOKENS | _ROLE_TOKENS | _PRODUCT_TOKENS | _DEPARTMENT_TOKENS
        ):
            continue
        return True
    return False


def _target_terms(company_terms: list[str]) -> set[str]:
    """Lowercased distinctive terms for the target company, excluding legal
    suffixes and other noise tokens so a partner "Duetto Inc" cannot match the
    target purely on a shared "inc"."

    Tokens that carry no identity signal (inc, ltd, co, hotels, group, ...) are
    dropped; what remains is ``[^.!?;]``-safe distinctive vocabulary.
    """
    out: set[str] = set()
    for term in (company_terms or []):
        cleaned = (term or "").strip().lower()
        if len(cleaned) < 3:
            continue
        if cleaned in _COMPANY_SUFFIX_TOKENS:
            continue
        for token in re.split(r"[^\w]+", cleaned):
            if len(token) >= 3 and token not in _COMPANY_SUFFIX_TOKENS:
                out.add(token)
    return out


def _company_matches_full_name(
    current_company: str, distinctive: set[str], full_names: list[str]
) -> bool:
    """True if `current_company` is the target, judged on name similarity.

    Token overlap alone cannot separate "World Web Technologies" from "World
    Wide Technology": both share ``world``/``technologies``. The reliable
    signal is (a) near-identical whole-name similarity to one of the target's
    full names, or (b) a *distinctive* brand token — never just a generic
    legal-name token shared with unrelated firms.
    """
    if not current_company:
        return True
    lowered = current_company.lower()
    for name in full_names:
        cleaned = " ".join((name or "").lower().replace("-", " ").split())
        if not cleaned:
            continue
        if SequenceMatcher(None, lowered, cleaned).ratio() >= 0.85:
            return True
    tokens = {t.strip(".'’\"") for t in lowered.replace("-", " ").split()}
    return bool(tokens & distinctive)


def _identity_names(
    identity: CompanyIdentity | None, company_terms: list[str] | None
) -> tuple[set[str], list[str]]:
    """Distinctive brand tokens and full candidate names for the target.

    The distinctive set is built only from the user-supplied brand/domain
    (canonical name, domain, aliases), never from the resolved legal name or
    the flattened term list: "World Web Technologies" is a generic legal name
    whose tokens are shared with totally unrelated firms. `full_names` keeps
    every known spelling for the whole-name similarity check.
    """
    full_names: list[str] = []
    if identity is not None:
        for value in (
            identity.canonical_name,
            identity.legal_name,
            identity.domain,
            *(identity.aliases or []),
        ):
            if value and value not in full_names:
                full_names.append(value)
    for term in (company_terms or []):
        if term and term not in full_names:
            full_names.append(term)

    brand_tokens: set[str] = set()
    if identity is not None:
        for value in (
            identity.canonical_name,
            identity.domain,
            *(identity.aliases or []),
        ):
            brand_tokens.update(_target_terms([value]))
    return brand_tokens, [n for n in full_names if n]


@dataclass(slots=True)
class CandidateBundle:
    """Everything gathered about a candidate before ranking."""

    people: list[Person] = field(default_factory=list)
    llm_persons: list[P.ConfirmedPerson] = field(default_factory=list)
    best_person: str = ""
    best_reason: str = ""
    rejected: list[str] = field(default_factory=list)


def _evidence(
    person: Person,
    claim_type: str,
    value: str,
    url: str,
    title: str,
    quote: str,
    confidence: float,
) -> None:
    item = Evidence(
        claim_type=claim_type,  # type: ignore[arg-type]
        claim_value=value,
        source_url=url,
        source_title=title,
        source_text=re.sub(r"\s+", " ", quote or "")[:600],
        confidence=round(confidence, 2),
        person_id=person.person_id,
        source_type=classify_source_type(url, title),
    )
    for existing in person.evidence:
        if existing.claim_type == claim_type and existing.source_url == url:
            if len(item.source_text) > len(existing.source_text):
                existing.source_text = item.source_text
                existing.confidence = max(existing.confidence, item.confidence)
            return
    person.evidence.append(item)
    if url not in person.sources:
        person.sources.append(url)


def merge_person(existing: Person, incoming: Person) -> Person:
    """Fold a second sighting of the same person into the first."""
    if not existing.title and incoming.title:
        existing.title = incoming.title
        existing.role_category = incoming.role_category
    elif incoming.title:
        # Prefer the more senior title; on equal seniority prefer the
        # well-formed one ("Founder & President" beats snippet chrome such as
        # "WebRezPro - WebRezPro Team Frank Verhagen Founder & ...").
        incoming_clean = _looks_like_title(incoming.title)
        existing_clean = _looks_like_title(existing.title)
        incoming_rank = _title_rank(incoming.title)
        existing_rank = _title_rank(existing.title)
        better = incoming_rank > existing_rank
        if not better and incoming_rank == existing_rank:
            if incoming_clean and not existing_clean:
                better = True
            elif incoming_clean == existing_clean and len(incoming.title) > len(
                existing.title or ""
            ):
                better = True
        if better:
            existing.title = incoming.title
            existing.role_category = incoming.role_category
    if not existing.current_company and incoming.current_company:
        existing.current_company = incoming.current_company
    if not existing.linkedin and incoming.linkedin:
        existing.linkedin = incoming.linkedin
    if not existing.location and incoming.location:
        existing.location = incoming.location
    if not incoming.is_current:
        existing.is_current = False
    for item in incoming.evidence:
        _evidence(
            existing,
            item.claim_type,
            item.claim_value,
            item.source_url,
            item.source_title,
            item.source_text,
            item.confidence,
        )
    return existing


_TITLE_SENIORITY = (
    "founder", "owner", "ceo", "chief executive", "president", "managing director",
    "representative director", "coo", "cto", "cfo", "cmo", "vp", "vice president",
    "head of", "general manager", "director",
)


def _title_rank(title: str) -> int:
    lowered = (title or "").lower()
    for index, needle in enumerate(_TITLE_SENIORITY):
        if needle in lowered:
            return len(_TITLE_SENIORITY) - index
    return 0


def harvest_from_pages(
    pages: list[Page],
    *,
    company_domains: list[str],
    company_terms: list[str] | None = None,
) -> list[Person]:
    """People found in fetched pages, with the page as the evidence."""
    buckets: dict[str, Person] = {}
    terms = company_terms or []

    for page in pages:
        url = page.final_url or page.url
        on_domain = any(d and d in url.lower() for d in company_domains)
        if not _is_people_source(url, page.title):
            continue
        for mention in page.facts.people:
            name = mention.name.strip()
            if not _looks_like_person_name(name):
                continue
            if not name or len(name) < 3:
                continue
            title = mention.title.strip()
            if _attributed_other_company(f"{title} {mention.context}", terms):
                continue
            key = person_name_key(name)
            if not key:
                continue
            role, _ = classify_role(title)
            person = buckets.get(key)
            if person is None:
                person = Person(name=name, title=title, role_category=role)
                buckets[key] = person
            elif person_name_conflict(person.name, name) and len(name) < len(person.name):
                continue
            else:
                person = merge_person(person, Person(name=name, title=title, role_category=role))
                buckets[key] = person

            confidence = 0.7 if on_domain else 0.45
            _evidence(
                person,
                "person_name",
                name,
                url,
                page.title,
                mention.context or f"{name} {title}",
                confidence,
            )
            if title:
                _evidence(person, "person_title", title, url, page.title, mention.context,
                          0.75 if on_domain else 0.5)
            if not person.is_current and _DEPRECATED.search(f"{title} {mention.context}"):
                person.is_current = False
            if page.facts.socials.get("linkedin_profile"):
                person.linkedin = page.facts.socials["linkedin_profile"]

    return [person for person in buckets.values() if person.evidence]


_SNIPPET_PERSON = re.compile(
    r"([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){1,3})\s*[,–—\-:|]\s*"
    r"([^\n,;|]{3,60}?(?:founder|co-founder|ceo|president|owner|director|"
    r"manager|head of|partnerships|developer|engineer)[^\n,;|]{0,30})",
    re.I,
)


def _snippet_title_ok(title: str, person_name: str = "") -> bool:
    """Snippet titles are page titles, often fronted by company/product chrome
    ("WebRezPro Team Founder & President", "Acme CEO & Director"), so accept a
    title whose *tail* from the first role word onwards looks like a title,
    provided any leading chrome is short and contains no sentence punctuation.
    The person's own name is allowed to appear (and is stripped) in the chrome:
    directory snippets echo it e.g. "Team Frank Verhagen Founder & President".
    """
    value = (title or "").strip()
    if _looks_like_title(value):
        return True
    match = _TITLE_RE.search(value)
    if not match:
        return False
    prefix = value[: match.start()].strip(" ,;:()|·-–—")
    if person_name:
        for token in person_name.split():
            prefix = re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", " ", prefix, flags=re.I)
        prefix = re.sub(r"\s{2,}", " ", prefix).strip(" ,;:()|·-–—")
    tail = value[match.start() :].strip(" ,;:()|·-–—")
    if len(prefix) > 40 or len(prefix) < 2:
        return False
    tokens = prefix.split()
    if not any(t and t[0].isupper() and not t.isupper() for t in tokens):
        return False
    if not _looks_like_title(tail):
        return False
    return not any(ch in prefix for ch in ".,!?;:")


def harvest_from_snippets(
    snippets: list[tuple[str, str]],
    *,
    company_terms: list[str],
) -> list[Person]:
    """People visible in search-result snippets (weak evidence, but free)."""
    buckets: dict[str, Person] = {}
    for url, snippet in snippets:
        url = html_lib.unescape(url)
        snippet = html_lib.unescape(snippet)
        lowered = f"{url} {snippet}".lower()
        if not any(term and term in lowered for term in company_terms):
            continue
        if not _is_people_source(url, snippet):
            continue
        for match in _SNIPPET_PERSON.finditer(snippet):
            name, title = match.group(1).strip(), match.group(2).strip()
            if not _looks_like_person_name(name) or not _snippet_title_ok(title, name):
                continue
            if _attributed_other_company(f"{title} {snippet}", company_terms):
                continue
            if not name:
                continue
            key = person_name_key(name)
            if not key or key in buckets:
                continue
            role, _ = classify_role(title)
            person = Person(name=name, title=title, role_category=role)
            _evidence(person, "person_name", name, url, "", snippet, 0.35)
            _evidence(person, "person_title", title, url, "", snippet, 0.35)
            buckets[key] = person
    return list(buckets.values())


def rank_people(
    people: list[Person],
    *,
    company_terms: list[str],
    company_domains: list[str],
    size_hint: str = "",
) -> list[Person]:
    """Score and sort. Deterministic, so a grade can be reproduced."""
    for person in people:
        person.score = score_person(
            person,
            company_terms=company_terms,
            company_domains=company_domains,
            size_hint=size_hint,
        )
        person.confidence = _confidence_from_score(person, company_domains)
    people.sort(key=lambda p: (-p.score.total, p.name))
    return people


def _confidence_from_score(person: Person, company_domains: list[str]) -> float:
    """Confidence in the person-company link, not in the seniority.

    A well-evidenced junior is more *confident* than a well-titled guess, so
    this reads evidence and domain-match rather than the role.
    """
    if not person.evidence:
        return 0.0
    domains = [d.lower() for d in company_domains if d]
    best = 0.0
    for item in person.evidence:
        on_domain = any(d in item.source_url.lower() for d in domains)
        value = item.confidence + (0.15 if on_domain else 0.0)
        if item.claim_type == "person_title" and item.source_text:
            value += 0.1
        best = max(best, value)
    corroboration = min(0.15, 0.05 * (len(person.evidence) - 1))
    return round(min(0.98, best + corroboration), 2)


def _filter_supported(
    people: list[Person],
    allowed_urls: set[str],
) -> tuple[list[Person], list[str]]:
    """Drop anyone whose only evidence is a URL we never actually retrieved.

    This is the code-level enforcement of the handoff's "no source, no fact"
    rule: a model that invents a person plus a plausible-looking URL loses that
    person here.
    """
    kept: list[Person] = []
    dropped: list[str] = []
    for person in people:
        supported = [
            item
            for item in person.evidence
            if item.source_url in allowed_urls
            or any(item.source_url in url for url in allowed_urls)
        ]
        if not supported:
            dropped.append(person.name)
            continue
        person.evidence = supported
        person.sources = list(dict.fromkeys(item.source_url for item in supported))
        kept.append(person)
    return kept, dropped


def reconcile_with_llm(
    client,
    *,
    input_company: str,
    resolved_name: str,
    legal_name: str,
    country: str,
    size_hint: str,
    people: list[Person],
    char_budget: int,
) -> CandidateBundle:
    """Ask Qwen to verify/merge/rank the gathered candidates."""
    bundle = CandidateBundle(people=people)
    if not client.enabled or not people:
        return bundle

    candidates = []
    for person in people[:24]:
        candidates.append(
            {
                "name": person.name,
                "title": person.title,
                "context_company": person.current_company,
                "evidence": [
                    {"url": item.source_url, "quote": item.source_text}
                    for item in person.evidence[:3]
                ],
            }
        )
    user = P.build_person_prompt(
        input_company=input_company,
        resolved_name=resolved_name,
        legal_name=legal_name,
        country=country,
        size_hint=size_hint,
        candidates=candidates,
        char_budget=char_budget,
    )
    try:
        result = client.structured(P.PERSON_SYSTEM, user, P.PersonReconciliation)
    except (LLMCallFailed, LLMUnavailable):
        return bundle

    bundle.llm_persons = result.people
    bundle.best_person = result.best_person
    bundle.best_reason = result.reason
    bundle.rejected = result.unresolved
    return bundle


def apply_llm_verification(
    people: list[Person],
    llm_persons: list[P.ConfirmedPerson],
    *,
    allowed_urls: set[str],
    company_terms: list[str] | None = None,
    identity: CompanyIdentity | None = None,
) -> list[Person]:
    """Filter to LLM-confirmed people that also survive source verification."""
    if not llm_persons:
        return people
    confirmed: list[Person] = []
    by_key = {person_name_key(p.name): p for p in people}
    expected = _target_terms(company_terms)
    distinctive, full_names = _identity_names(identity, company_terms)

    for entry in llm_persons:
        key = person_name_key(entry.name)
        person = by_key.get(key)
        if person is None:
            continue
        # The model fills current_company from the evidence; if it names an
        # organisation that is clearly not the target, drop the person. This is
        # what kills partner executives quoted on press-release pages and
        # look-alike companies ("World Web Technologies" vs "World Wide
        # Technology"), where raw token overlap is not discriminating.
        if entry.current_company and not _company_matches_full_name(
            entry.current_company, distinctive, full_names
        ):
            continue
        supported = [
            url
            for url in entry.source_urls
            if url in allowed_urls or any(url in known for known in allowed_urls)
        ]
        if not supported:
            # The model believes it but cannot point at a retrieved page.
            continue
        if entry.title:
            person.title = entry.title
            person.role_category = classify_role(entry.title)[0]
        if entry.current_company:
            person.current_company = entry.current_company
        if entry.location:
            person.location = entry.location
        person.is_current = entry.is_current
        person.notes = entry.note
        person.confidence = max(person.confidence, 0.6)
        confirmed.append(person)

    return confirmed or people


def select_best(people: list[Person]) -> Person | None:
    return people[0] if people else None


def person_evidence(people: list[Person]) -> list[Evidence]:
    out: list[Evidence] = []
    for person in people:
        out.extend(person.evidence)
    return out
