"""Stage 4: adaptive research planning.

Qwen proposes the next searches; the orchestrator executes them. Qwen never
browses. The planner also emits native-language vocabulary, which is how the
system handles Japanese, Chinese and other non-English companies generically
rather than with a country switch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from leadfinder.llm import prompts as P
from leadfinder.llm.qwen import LLMCallFailed, LLMUnavailable
from leadfinder.models.company import CompanyIdentity, CompanyInput

# Fallback vocabulary per language, used when the LLM is unavailable. This is
# a language table, not a company table - it contains no company-specific logic.
_FALLBACK_VOCAB: dict[str, dict[str, list[str]]] = {
    "japanese": {
        "titles": ["代表取締役", "社長", "創業者", "代表者", "代表取締役社長", "会長"],
        "org": ["会社概要", "企業情報", "会社情報", "経営陣", "チーム"],
        "contact": ["お問い合わせ", "連絡先", "お問い合わせフォーム", "営業"],
        "commercial": ["提携", "事業開発", "パートナー", "営業担当"],
    },
    "chinese": {
        "titles": ["创始人", "法定代表人", "负责人", "董事长", "总经理", "总裁"],
        "org": ["公司简介", "公司介绍", "关于我们", "团队"],
        "contact": ["联系方式", "联系我们", "商务合作", "微信"],
        "commercial": ["商务合作", "渠道合作", "代理"],
    },
    "german": {
        "titles": ["Geschäftsführer", "Inhaber", "Gründer", "Vertreter", "Vorstand"],
        "org": ["Über uns", "Unternehmen", "Team", "Impressum"],
        "contact": ["Kontakt", "Kontaktformular", "Anfrage"],
        "commercial": ["Partnerschaft", "Vertrieb", "Geschäftsentwicklung"],
    },
    "spanish": {
        "titles": ["fundador", "director general", "CEO", "representante", "socio fundador"],
        "org": ["quiénes somos", "equipo", "sobre nosotros", "aviso legal"],
        "contact": ["contacto", "formulario de contacto", "atención al cliente"],
        "commercial": ["colaboración", "alianzas", "desarrollo de negocio"],
    },
    "italian": {
        "titles": ["fondatore", "amministratore delegato", "titolare", "socio fondatore", "CEO"],
        "org": ["chi siamo", "azienda", "team", "note legali"],
        "contact": ["contatti", "contattaci", "formulario di contatto"],
        "commercial": ["partnership", "vendite", "business development"],
    },
    "french": {
        "titles": ["fondateur", "PDG", "dirigeant", "gérant", "cofondateur"],
        "org": ["à propos", "notre équipe", "mentions légales", "qui sommes-nous"],
        "contact": ["contact", "nous contacter", "formulaire de contact"],
        "commercial": ["partenariat", "développement commercial"],
    },
    "portuguese": {
        "titles": ["fundador", "administrador", "gerente", "sócio fundador", "CEO"],
        "org": ["quem somos", "equipa", "empresa", "avisos legais"],
        "contact": ["contactos", "contacte-nos", "formulário de contacto"],
        "commercial": ["parceria", "desenvolvimento de negócios"],
    },
    "dutch": {
        "titles": ["oprichter", "directeur", "eigenaar", "algemeen directeur", "MD"],
        "org": ["over ons", "team", "bedrijf", "colofon"],
        "contact": ["contact", "contactformulier", "vragen"],
        "commercial": ["samenwerking", "partnerschap", "zakelijke ontwikkeling"],
    },
}

_LANGUAGE_KEYS = {
    "japanese": ("japanese", "ja", "jp", "日本語", "日本"),
    "chinese": ("chinese", "zh", "cn", "中文", "中国", "简体", "繁體"),
    "german": ("german", "de", "deutsch", "Deutschland"),
    "spanish": ("spanish", "es", "español", "españa", "espana", "latinoamérica"),
    "italian": ("italian", "it", "italiano", "italia"),
    "french": ("french", "fr", "français", "france"),
    "portuguese": ("portuguese", "pt", "português", "portugues", "portugal", "brasil"),
    "dutch": ("dutch", "nl", "nederlands", "nederland"),
}


def detect_language_vocabulary(country: str, language: str) -> dict[str, list[str]]:
    """Pick the vocabulary table for a country/language pair, if we have one."""
    blob = f"{country} {language}".lower()
    for key, needles in _LANGUAGE_KEYS.items():
        if any(needle in blob for needle in needles):
            return _FALLBACK_VOCAB[key]
    return {}


def vocabulary_queries(terms: list[str], company_name: str, legal_name: str) -> list[str]:
    """Turn a vocabulary table into concrete search queries."""
    subject = legal_name or company_name
    if not subject or not terms:
        return []
    quoted = f'"{subject}"'
    out: list[str] = []
    for term in terms[:4]:
        out.append(f"{quoted} {term}")
    return out


# Path segments that mark a page as a people/team/about page - the pages where
# decision-makers are actually published. Fetch prioritisation uses this so a
# limited page budget is spent on team pages before product pages.
_PEOPLE_PAGE_SEGMENTS = {
    "team", "our-team", "meet-the-team", "meet-our-team", "team-members",
    "about", "about-us", "aboutus", "who-we-are",
    "leadership", "leaders", "leadership-team", "our-leadership",
    "management", "management-team", "executive", "executives",
    "executive-team", "board", "board-of-directors", "directors",
    "founders", "founder", "staff", "people", "our-people", "key-people",
    "officers", "principals", "partners", "directory", "employees",
    "equipo", "equipe", "uber-uns",
    "ueber-uns", "sobre-nosotros", "chi-siamo", "quem-somos", "over-ons",
    "nosotros", "azienda", "unternehmen", "会社概要", "経営陣", "团队",
}


def is_people_page_url(url: str) -> bool:
    """True when a URL path looks like a team/about/leadership page."""
    try:
        path = urlparse(url or "").path or ""
    except ValueError:
        return False
    for segment in path.split("/"):
        cleaned = segment.strip().lower()
        for suffix in (".html", ".htm", ".php", ".aspx"):
            cleaned = cleaned.removesuffix(suffix)
        cleaned = cleaned.replace("_", "-")
        if cleaned in _PEOPLE_PAGE_SEGMENTS:
            return True
    return False


# The page vocabulary that surfaces people pages on the company's own site.
_PEOPLE_TERMS = ("team", "about", "leadership", "founders", "staff", "people")


def people_search_queries(
    identity: CompanyIdentity,
    item: CompanyInput,
    domains: list[str],
    already_run: list[str] | None = None,
) -> list[str]:
    """Deterministic site-scoped queries aimed at the company's people pages.

    Decision-makers live on /team, /about and /leadership pages that generic
    role queries rarely surface. This is the dedicated people pass: bounded,
    deterministic, no LLM round spent.
    """
    queries: list[str] = []
    for domain in domains[:2]:
        for term in _PEOPLE_TERMS:
            queries.append(f"site:{domain} {term}")
        queries.append(f'site:{domain} "meet the team" OR "our team"')
    vocab = detect_language_vocabulary(identity.country, identity.language)
    native = [t for t in (vocab.get("titles", []) + vocab.get("org", [])) if t]
    if native:
        for term in native[:4]:
            if domains:
                queries.append(f"site:{domains[0]} {term}")
            elif item.company_name:
                queries.append(f'"{item.company_name}" {term}')
    return _dedupe(queries, already_run or [])


@dataclass(slots=True)
class PlanResult:
    queries: list[str] = field(default_factory=list)
    native_terms: list[str] = field(default_factory=list)
    stop: bool = False
    reason: str = ""
    used_llm: bool = False
    fallback_used: bool = False


def _gaps(
    identity: CompanyIdentity,
    people_found: int,
    contact_ok: bool,
) -> tuple[str, str]:
    if identity.confidence < 0.6 or not identity.canonical_name:
        people_gap = "no confirmed decision-maker yet"
    elif people_found == 0:
        people_gap = "no confirmed decision-maker yet"
    elif people_found == 1:
        people_gap = "only one candidate; a second would corroborate"
    else:
        people_gap = ""
    if contact_ok:
        contact_gap = ""
    else:
        contact_gap = "no email, contact form or phone found yet"
    return people_gap, contact_gap


def plan_round(
    *,
    llm,
    item: CompanyInput,
    identity: CompanyIdentity,
    pages,
    people: list,
    contact_ok: bool,
    already_run: list[str],
) -> PlanResult:
    """One planning round. Never raises: planning is best-effort."""
    result = PlanResult()
    people_gap, contact_gap = _gaps(identity, len(people), contact_ok)

    known_parts: list[str] = []
    if identity.canonical_name:
        known_parts.append(f"canonical name: {identity.canonical_name}")
    if identity.legal_name:
        known_parts.append(f"legal entity: {identity.legal_name}")
    if identity.country:
        known_parts.append(f"country: {identity.country}")
    if identity.company_type:
        known_parts.append(f"type: {identity.company_type}")
    if identity.industry:
        known_parts.append(f"industry: {identity.industry}")
    if people:
        names = ", ".join(f"{p.name} ({p.title or 'title unknown'})" for p in people[:5])
        known_parts.append(f"candidate people: {names}")
    known = "\n".join(known_parts) or "(little)"

    if llm.enabled:
        user = P.build_planner_prompt(
            input_company=item.company_name,
            resolved_name=identity.canonical_name,
            legal_name=identity.legal_name,
            country=identity.country,
            language=identity.language,
            known=known,
            already_run=already_run,
            contact_gap=contact_gap,
            people_gap=people_gap,
        )
        try:
            plan = llm.structured(P.PLANNER_SYSTEM, user, P.QueryPlan)
            result.queries = plan.queries[:5]
            result.native_terms = plan.native_terms[:12]
            result.stop = plan.stop
            result.reason = plan.reason
            result.used_llm = True
        except (LLMCallFailed, LLMUnavailable):
            result.used_llm = False

    if not result.queries:
        result.queries = _fallback_queries(identity, item, people, contact_ok, already_run)
        result.fallback_used = True
        result.reason = result.reason or "deterministic gap-filling plan"

    if not result.native_terms:
        vocab = detect_language_vocabulary(identity.country, identity.language)
        flat = [t for group in vocab.values() for t in group]
        result.native_terms = flat[:12]

    result.queries = _dedupe(result.queries, already_run)
    return result


def _dedupe(queries: list[str], already_run: list[str]) -> list[str]:
    seen = {q.strip().lower() for q in already_run if q}
    out: list[str] = []
    for query in queries:
        cleaned = re.sub(r"\s+", " ", (query or "")).strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        out.append(cleaned)
    return out


def _fallback_queries(
    identity: CompanyIdentity,
    item: CompanyInput,
    people: list,
    contact_ok: bool,
    already_run: list[str],
) -> list[str]:
    """Gap-filling plan for when the LLM is unavailable.

    Mirrors the same reasoning the model is asked for: target the specific
    missing artifact (person or contact) and, for non-English companies, search
    the local vocabulary.
    """
    subject = identity.legal_name or identity.canonical_name or item.company_name
    out: list[str] = []

    if len(people) == 0:
        for term in ("founder", "CEO", "owner", "managing director", "president"):
            out.append(f'"{subject}" {term}')
    if not contact_ok:
        for term in ("contact", "email", "impressum", "contacto", "contatti"):
            out.append(f'"{subject}" {term}')
    if identity.product_or_company == "brand" and identity.legal_name:
        out.append(f'"{item.company_name}" "{identity.legal_name}"')
        out.append(f'"{subject}" company profile')
    if not out:
        out.append(f'"{subject}" about')
        out.append(f'"{subject}" press release')

    vocab = detect_language_vocabulary(identity.country, identity.language)
    native = vocabulary_queries(
        vocab.get("titles", []) + vocab.get("org", []), subject, identity.legal_name
    )
    out.extend(native[:3])
    return _dedupe(out[:6], already_run)


def success_reached(
    *,
    identity: CompanyIdentity,
    people: list,
    contact_ok: bool,
    config,
) -> tuple[bool, str]:
    """The handoff's stop condition, stated literally.

    high-confidence company identity + high-confidence decision maker +
    usable contact path. Remaining budget is not a reason to keep going.
    """
    if (
        identity.confidence >= config.research.company_confidence_target
        and bool(identity.canonical_name)
        and people
        and people[0].confidence >= config.research.person_confidence_target
        and contact_ok
    ):
        return True, "success condition reached (identity + decision maker + contact path)"
    return False, ""


def needs_another_round(
    plan: PlanResult,
    *,
    identity: CompanyIdentity,
    people: list,
    contact_ok: bool,
    config,
) -> tuple[bool, str]:
    """Combine the success condition with the planner's own stop signal."""
    done, reason = success_reached(
        identity=identity, people=people, contact_ok=contact_ok, config=config
    )
    if done:
        return False, reason
    if plan.stop and plan.used_llm:
        return False, "planner reported sufficient evidence"
    if not plan.queries:
        return False, "no new queries proposed"
    return True, ""
