"""Contact discovery and grading.

V0 starts with what is already published on fetched pages - that is where a
$0 pipeline beats a paid database. Inference is permitted but must be visibly
lower confidence than a published address, and the method is always recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from leadfinder.crawl.crawler import Page
from leadfinder.crawl.extract import classify_email, normalize_email, normalize_local
from leadfinder.models.evidence import Evidence
from leadfinder.models.result import ContactInfo
from leadfinder.search.base import classify_source_type

# Priority order from the handoff.
PRIORITY = [
    "direct_published",
    "direct_inferred",
    "general_email",
    "contact_form",
    "phone",
    "other",
]

_OTHER_CHANNEL_LABELS = (
    "whatsapp", "telegram", "wechat", "weixin", "line", "viber", "signal",
    "skype", "chat", "live chat", "livechat", "discord", "zulip", "kakao",
    "zalo", "messenger", "sms", "callback", "contact form", "お問い合わせ",
)


@dataclass(slots=True)
class ContactCandidate:
    kind: str            # direct_email | general_email | phone | contact_form | other
    value: str
    source_url: str
    source_title: str = ""
    quote: str = ""
    method: str = "published"   # published | inferred
    confidence: float = 0.0
    person_name: str = ""

    @property
    def priority(self) -> int:
        if self.kind == "direct_email":
            return 0 if self.method == "published" else 1
        if self.kind == "general_email":
            return 2
        if self.kind == "contact_form":
            return 3
        if self.kind == "phone":
            return 4
        return 5


@dataclass(slots=True)
class ContactReport:
    info: ContactInfo = field(default_factory=ContactInfo)
    candidates: list[ContactCandidate] = field(default_factory=list)
    email_pattern: str = ""     # observed local-part pattern on the company domain
    pattern_confidence: float = 0.0


def _domain_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _is_official(email: str, company_domains: list[str]) -> bool:
    domain = email.partition("@")[2].lower()
    for candidate in company_domains:
        if not candidate:
            continue
        if domain == candidate or domain.endswith("." + candidate):
            return True
    return False


def _quote_around(text: str, needle: str, width: int = 220) -> str:
    if not text or not needle:
        return ""
    index = text.lower().find(needle.lower())
    if index == -1:
        return ""
    start = max(0, index - width // 3)
    return re.sub(r"\s+", " ", text[start : index + width]).strip()


def observe_email_pattern(emails: list[tuple[str, str]]) -> tuple[str, float]:
    """Infer the company's own local-part convention from published addresses.

    Returns e.g. ('first.last', 0.8) when the site consistently publishes
    jane.doe@..., or ('first', 0.5) for jane@....
    """
    shapes: dict[str, int] = {}
    for email, domain in emails:
        local = email.partition("@")[0]
        if not local or "." not in domain:
            continue
        if local in {"info", "contact", "hello", "support", "sales", "admin",
                     "office", "enquiries", "mail", "team", "help", "hi"}:
            continue
        tokens = re.split(r"[._-]", local)
        tokens = [t for t in tokens if t]
        if len(tokens) == 1:
            shape = "first"
        elif len(tokens) == 2:
            shape = "first.last"
        elif len(tokens) >= 3:
            shape = "f.last"
        else:
            continue
        shapes[shape] = shapes.get(shape, 0) + 1
    if not shapes:
        return "", 0.0
    best, count = max(shapes.items(), key=lambda item: item[1])
    confidence = 0.5 if count == 1 else min(0.85, 0.5 + 0.12 * count)
    if len(shapes) > 1:
        confidence -= 0.1
    return best, max(0.0, confidence)


def build_inferred_email(
    person_name: str,
    domain: str,
    pattern: str,
    pattern_confidence: float,
) -> tuple[str, float]:
    """Apply an observed domain convention to a verified person's name."""
    local = normalize_local(person_name)
    tokens = [t for t in re.split(r"[\s.]+", local) if t]
    if not tokens or not domain or not pattern:
        return "", 0.0
    if len(tokens) == 1:
        return "", 0.0
    first, last = tokens[0], tokens[-1]
    if pattern == "first.last":
        candidate = f"{first}.{last}"
    elif pattern == "f.last":
        candidate = f"{first[0]}.{last}"
    else:  # "first"
        candidate = first
    email = f"{candidate}@{domain}"
    # Never present a guess as certain.
    return email, min(0.55, pattern_confidence * 0.6)


def collect_candidates(
    pages: list[Page],
    *,
    company_domains: list[str],
    person_names: list[str],
) -> list[ContactCandidate]:
    """Pull every contact channel that appears in already-fetched pages."""
    found: list[ContactCandidate] = []
    seen: set[tuple[str, str]] = set()

    def add(candidate: ContactCandidate) -> None:
        key = (candidate.kind, candidate.value.lower())
        if not candidate.value or key in seen:
            return
        seen.add(key)
        found.append(candidate)

    for page in pages:
        facts = page.facts
        url = page.final_url or page.url

        for email in facts.emails:
            kind, base_confidence = classify_email(email, person_names)
            if not kind:
                continue
            official = _is_official(email, company_domains)
            if not official and kind == "general":
                # A role mailbox on a third-party domain is rarely the path.
                continue
            if kind == "direct" and not official:
                # A personal-provider address published by the company is
                # usually a founder's, so it stays usable.
                base_confidence = min(base_confidence, 0.5)
            elif official:
                base_confidence = min(0.95, base_confidence + 0.2)
            matched_name = _match_person(email, person_names)
            add(
                ContactCandidate(
                    kind="direct_email" if kind == "direct" else "general_email",
                    value=email,
                    source_url=url,
                    source_title=page.title,
                    quote=_quote_around(page.text, email) or _quote_around(page.html, email),
                    method="published",
                    confidence=round(base_confidence, 2),
                    person_name=matched_name,
                )
            )

        for phone in facts.phones:
            add(
                ContactCandidate(
                    kind="phone",
                    value=phone,
                    source_url=url,
                    source_title=page.title,
                    quote=_quote_around(page.text, phone),
                    method="published",
                    confidence=0.5,
                )
            )

        for form_url in facts.contact_forms:
            add(
                ContactCandidate(
                    kind="contact_form",
                    value=form_url,
                    source_url=url,
                    source_title=page.title,
                    quote="contact form",
                    method="published",
                    confidence=0.6,
                )
            )

        if facts.wechat:
            add(
                ContactCandidate(
                    kind="other",
                    value=f"WeChat: {facts.wechat}",
                    source_url=url,
                    source_title=page.title,
                    quote=_quote_around(page.text, facts.wechat),
                    method="published",
                    confidence=0.6,
                )
            )

        blob = f"{page.title} {page.text[:1500]}".lower()
        if any(label in blob for label in _OTHER_CHANNEL_LABELS):
            for label, url_value in facts.socials.items():
                if label in {"wechat", "telegram", "whatsapp"}:
                    add(
                        ContactCandidate(
                            kind="other",
                            value=f"{label}: {url_value}",
                            source_url=url,
                            source_title=page.title,
                            quote=url_value,
                            method="published",
                            confidence=0.45,
                        )
                    )
    return found


def _match_person(email: str, person_names: list[str]) -> str:
    local = email.partition("@")[0]
    best, best_score = "", 0.0
    for name in person_names:
        tokens = [t for t in re.split(r"[\s.]+", normalize_local(name)) if len(t) > 1]
        if not tokens:
            continue
        score = 0.0
        if local == ".".join(tokens):
            score = 1.0
        elif local == tokens[0]:
            score = 0.7
        elif local == tokens[-1]:
            score = 0.7
        elif all(token in local for token in tokens):
            score = 0.6
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= 0.6 else ""


def resolve_contacts(
    pages: list[Page],
    *,
    company_domains: list[str],
    person_names: list[str],
    best_person: str = "",
) -> ContactReport:
    """Turn collected candidates into a prioritised ContactInfo."""
    report = ContactReport()
    report.candidates = collect_candidates(
        pages, company_domains=company_domains, person_names=person_names
    )

    # Observed convention on the company's own domain, used only for inference.
    official_emails = [
        (c.value, c.value.partition("@")[2])
        for c in report.candidates
        if c.kind in {"direct_email", "general_email"}
        and _is_official(c.value, company_domains)
        and c.method == "published"
    ]
    pattern, pattern_confidence = observe_email_pattern(official_emails)
    report.email_pattern = pattern
    report.pattern_confidence = pattern_confidence

    if pattern and best_person and company_domains:
        for domain in company_domains:
            email, confidence = build_inferred_email(best_person, domain, pattern, pattern_confidence)
            if not email:
                continue
            if any(c.value == email for c in report.candidates):
                break
            report.candidates.append(
                ContactCandidate(
                    kind="direct_email",
                    value=email,
                    source_url=f"inferred:{domain}",
                    source_title="inferred from observed domain convention",
                    quote=f"pattern {pattern}@ observed on the company domain",
                    method="inferred",
                    confidence=round(confidence, 2),
                    person_name=best_person,
                )
            )
            break

    report.candidates.sort(key=lambda c: (c.priority, -c.confidence))
    info = ContactInfo()

    for candidate in report.candidates:
        if candidate.kind == "direct_email" and not info.direct_email:
            info.direct_email = candidate.value
            info.email_confidence = candidate.confidence
            info.email_method = candidate.method
        elif candidate.kind == "general_email" and not info.general_email:
            info.general_email = candidate.value
        elif candidate.kind == "contact_form" and not info.contact_form:
            info.contact_form = candidate.value
            info.contact_page = candidate.value
        elif candidate.kind == "phone" and not info.phone:
            info.phone = candidate.value
        elif candidate.kind == "other" and not info.other:
            info.other = candidate.value

    if not info.email_confidence and info.general_email:
        for candidate in report.candidates:
            if candidate.value == info.general_email:
                info.email_confidence = candidate.confidence * 0.6
                info.email_method = "published"
                break

    if not info.wechat:
        for candidate in report.candidates:
            if candidate.value.lower().startswith("wechat:"):
                info.wechat = candidate.value.split(":", 1)[1].strip()
                break

    info.evidence_urls = list(
        dict.fromkeys(c.source_url for c in report.candidates if not c.source_url.startswith("inferred:"))
    )[:8]
    report.info = info
    return report


def contacts_evidence(report: ContactReport) -> list[Evidence]:
    evidence: list[Evidence] = []
    for candidate in report.candidates:
        claim_type = {
            "direct_email": "direct_email",
            "general_email": "general_email",
            "phone": "phone",
            "contact_form": "contact_form",
        }.get(candidate.kind, "other_contact")
        if candidate.source_url.startswith("inferred:"):
            # Inferred values are recorded with their method, not as retrieved
            # source text, so a reviewer can never mistake them for a quote.
            evidence.append(
                Evidence(
                    claim_type=claim_type,
                    claim_value=candidate.value,
                    source_url="inferred",
                    source_title=candidate.source_title,
                    source_text=candidate.quote,
                    confidence=candidate.confidence,
                )
            )
            continue
        evidence.append(
            Evidence(
                claim_type=claim_type,
                claim_value=candidate.value,
                source_url=candidate.source_url,
                source_title=candidate.source_title,
                source_text=candidate.quote,
                confidence=candidate.confidence,
                source_type=classify_source_type(
                    candidate.source_url, candidate.source_title
                ),
            )
        )
    return evidence


def sanitize_email(value: str) -> str:
    return normalize_email(value or "")
