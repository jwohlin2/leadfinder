"""Short, structured Qwen prompts and their Pydantic output schemas.

Four call types only:
  1. entity resolution
  2. query planning (including native-language vocabulary)
  3. person reconciliation / ranking
  4. final lead synthesis

Hard rule, enforced in code as well as in the prompt: a claim without a source
URL is dropped before it can reach the output.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- schemas


class ResolvedEntity(BaseModel):
    canonical_name: str = ""
    legal_name: str = ""
    domain: str = ""
    country: str = ""
    language: str = ""
    company_type: str = ""
    product_or_company: Literal["brand", "legal_company", "unknown"] = "unknown"
    parent_company: str = ""
    status: Literal["active", "inactive", "acquired", "defunct", "unknown"] = "unknown"
    aliases: list[str] = Field(default_factory=list)
    industry: str = ""
    size_hint: str = ""
    confidence: float = 0.0
    source_urls: list[str] = Field(default_factory=list)
    note: str = ""

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value or 0.0)))


class QueryPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)
    native_terms: list[str] = Field(default_factory=list)
    stop: bool = False
    reason: str = ""

    @field_validator("queries", "native_terms", mode="before")
    @classmethod
    def _listify(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value if str(item).strip()]


class ConfirmedPerson(BaseModel):
    name: str
    title: str = ""
    current_company: str = ""
    location: str = ""
    is_current: bool = True
    decision_maker: bool = True
    source_urls: list[str] = Field(default_factory=list)
    note: str = ""


class PersonReconciliation(BaseModel):
    people: list[ConfirmedPerson] = Field(default_factory=list)
    best_person: str = ""
    reason: str = ""
    unresolved: list[str] = Field(default_factory=list)


class ContactDecision(BaseModel):
    direct_email: str = ""
    email_method: Literal["published", "inferred", "none"] = "none"
    email_confidence: float = 0.0
    general_email: str = ""
    contact_form: str = ""
    phone: str = ""
    other: str = ""
    grade: Literal["A", "B", "C", "D", "E", "FAIL"] = "FAIL"
    grade_reason: str = ""
    status: Literal[
        "SUCCESS", "PARTIAL", "FAILED_DOMAIN", "FAILED_ENTITY_RESOLUTION",
        "FAILED_PERSON_DISCOVERY", "FAILED_CONTACT_DISCOVERY", "ERROR",
    ] = "PARTIAL"
    failure_category: str = ""
    summary: str = ""

    @field_validator("email_confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value or 0.0)))


# ------------------------------------------------------------ system prompts

JSON_RULE = (
    "Reply with a single JSON object and nothing else. "
    "No markdown fences, no commentary."
)

EVIDENCE_RULE = (
    "EVIDENCE RULE: you may only assert a fact if it appears in the evidence "
    "below. Every person and every company relationship must cite at least one "
    "source_url copied verbatim from the evidence. If you have no source for a "
    "fact, leave that field empty. Never guess a name, a title, an email "
    "address or a company."
)

ENTITY_SYSTEM = f"""You are a company entity-resolution engine.
Given a messy company name, a domain and retrieved evidence, identify the real
company behind it.

This matters most when the supplied name is a brand, product, project or local
trade name and the legal entity has a different name and a different country.

Report: canonical name, legal entity name, country, primary language, whether
the input is a brand or the legal company, parent company if any, whether the
company still looks active, and useful aliases.
{EVIDENCE_RULE}
Confidence: 0.9+ only if several independent sources agree; 0.5 if plausible
but single-sourced; 0 if you cannot tell.
{JSON_RULE}"""

PLANNER_SYSTEM = f"""You are a web-research query planner.
Given what is already known about a company, propose the next few search
queries most likely to reveal its decision-makers and a usable contact path.

If the company is outside the English-speaking world, include native-language
search terms in `native_terms`: job titles, "about us", "company profile",
"contact us" and "partnerships" equivalents in the local language. Titles come
first because they are what attach a name to a company.

Do not repeat queries that were already run. Prefer specific, high-yield
queries over broad ones. Return at most 5 queries.
Set `stop` to true if the current evidence is already enough.
{JSON_RULE}"""

PERSON_SYSTEM = f"""You are a people-verification engine.
You are given candidate names and titles gathered from several sources about
one company. Keep only candidates you can support with a cited source, merge
duplicates, and pick the single best decision-maker to contact.

REJECT a candidate if the evidence shows they belong to a DIFFERENT company:
partners' executives quoted on the target's press-release pages, competitors'
executives on comparison pages, suppliers, investors or advisors. A person's
`current_company` must be the target company itself. If a candidate's own
evidence names another organisation as their employer, exclude them.

Authority depends on company size: for a very small company the founder or
owner is usually right; for a larger organisation a head of partnerships,
business development or a product/commercial leader may be the better target.
Prefer someone who is current (not "former"/"previously").
{EVIDENCE_RULE}
Return 2-5 people when that many are supported.
{JSON_RULE}"""

SYNTHESIS_SYSTEM = f"""You are a lead-delivery engine.
Given a resolved company, verified decision-makers and every contact detail
found, choose the single most usable contact path and grade the lead:

A = named decision-maker + verified or directly published personal contact
B = named decision-maker + highly probable direct email/contact
C = named decision-maker + official company contact method
D = named legal representative/executive + official company contact method
E = company resolved but only generic contact information found
FAIL = company not confidently resolved, or no usable contact path

Never invent an email address. A contact form counts as a usable contact path.
{EVIDENCE_RULE}
{JSON_RULE}"""


# ------------------------------------------------------------ message builders

def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


def _numbered_sources(pages: list[dict], limit: int) -> str:
    """Render retrieved pages as a numbered evidence block.

    Numbering lets the model cite a short handle; the orchestrator maps handles
    back to full URLs so every stored claim keeps a real source.
    """
    lines: list[str] = []
    for index, page in enumerate(pages[:limit], start=1):
        url = page.get("url", "")
        title = page.get("title", "") or url
        body = _clip(page.get("text", ""), 1800)
        lines.append(f"[{index}] {title}\nURL: {url}\n{body}\n")
    return "\n".join(lines) if lines else "(no pages retrieved)"


def build_entity_prompt(
    *,
    input_company: str,
    input_domain: str,
    pages: list[dict],
    search_snippets: list[str],
    char_budget: int,
) -> str:
    per_page = max(700, char_budget // max(1, min(len(pages), 8)))
    rendered = _numbered_sources(pages, 8)
    if len(rendered) > char_budget:
        rendered = _clip(rendered, char_budget)
    snippets = "\n".join(f"- {_clip(s, 260)}" for s in search_snippets[:14]) or "(none)"
    return f"""INPUT COMPANY NAME: {input_company}
INPUT DOMAIN: {input_domain}

SEARCH SNIPPETS (weak evidence, must be corroborated by a page):
{snippets}

RETRIEVED PAGES (primary evidence):
{rendered}

Return the resolved entity. Cite the page numbers whose URL you used in
`source_urls` (use the full URL, not the number). If the evidence does not
establish who the real company is, set confidence to 0 and leave names empty."""


def build_planner_prompt(
    *,
    input_company: str,
    resolved_name: str,
    legal_name: str,
    country: str,
    language: str,
    known: str,
    already_run: list[str],
    contact_gap: str,
    people_gap: str,
) -> str:
    ran = "\n".join(f"- {q}" for q in already_run) or "(none)"
    return f"""TARGET COMPANY: {resolved_name or input_company}
LEGAL ENTITY: {legal_name or "(unknown)"}
COUNTRY: {country or "(unknown)"}    LANGUAGE: {language or "(unknown)"}

WHAT WE ALREADY KNOW:
{known or "(little)"}

WHAT IS STILL MISSING:
- decision-maker: {people_gap or "none found yet"}
- contact path: {contact_gap or "none found yet"}

QUERIES ALREADY RUN (do not repeat them):
{ran}

Propose the next queries, and any native-language vocabulary if this company
is outside the English-speaking world."""


def build_person_prompt(
    *,
    input_company: str,
    resolved_name: str,
    legal_name: str,
    country: str,
    size_hint: str,
    candidates: list[dict],
    char_budget: int,
) -> str:
    blocks: list[str] = []
    for candidate in candidates[:24]:
        evidence = "\n".join(
            f"    * {item.get('quote', '')[:300]}  <{item.get('url', '')}>"
            for item in candidate.get("evidence", [])[:3]
        )
        blocks.append(
            f"- {candidate.get('name', '?')}"
            f" | title: {candidate.get('title', '') or '(unknown)'}"
            f" | at: {candidate.get('context_company', '') or '(unknown)'}"
            f"\n{evidence or '    (no evidence yet)'}"
        )
    listing = "\n".join(blocks) or "(no candidates yet)"
    if len(listing) > char_budget:
        listing = _clip(listing, char_budget)
    return f"""INPUT: {input_company} ({legal_name or resolved_name or "unresolved"})
COUNTRY: {country or "(unknown)"}    COMPANY SIZE: {size_hint or "(unknown)"}

CANDIDATE PEOPLE AND THE EVIDENCE BEHIND THEM:
{listing}

Verify, merge duplicates, and choose the best decision-maker to contact."""


def build_synthesis_prompt(
    *,
    input_company: str,
    company_block: str,
    people_block: str,
    contact_block: str,
    sources: list[str],
) -> str:
    source_list = "\n".join(f"- {url}" for url in sources[:20]) or "(none)"
    return f"""INPUT: {input_company}

RESOLVED COMPANY:
{company_block}

VERIFIED DECISION-MAKERS:
{people_block}

CONTACT DETAILS FOUND (verbatim from pages - never add to this list):
{contact_block}

AVAILABLE SOURCE URLS:
{source_list}

Choose the best contact path, assign the grade, and explain in one sentence
why this is the right person and grade."""


_WS = re.compile(r"\s+")


def tidy(value: str) -> str:
    return _WS.sub(" ", (value or "").strip())
