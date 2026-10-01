"""Best-lead selection and A-F grading.

The LLM proposes the grade; this module owns the deterministic floor and
overrides the model when the stated evidence cannot support what it claims.
That keeps grades reproducible and stops a hallucinated email becoming an A.
"""

from __future__ import annotations

from dataclasses import dataclass

from leadfinder.llm import prompts as P
from leadfinder.llm.qwen import LLMCallFailed, LLMUnavailable
from leadfinder.models.company import CompanyIdentity
from leadfinder.models.person import Person
from leadfinder.models.result import BestLead, ContactInfo, LeadGrade, RunStatus

# Thresholds for a direct address to count as verified rather than guessed.
_VERIFIED_EMAIL = 0.8
_PROBABLE_EMAIL = 0.5
_STRONG_PERSON = 0.55


@dataclass(slots=True)
class GradeOutcome:
    grade: LeadGrade = "FAIL"
    reason: str = ""
    status: RunStatus = "PARTIAL"
    failure_category: str = ""


def _is_decision_maker(person: Person) -> bool:
    return person.role_category in {
        "founder", "owner", "chief_executive", "chief_operating", "executive",
        "senior_leader", "commercial_leader", "product_leader",
    }


def choose_best(
    people: list[Person],
    *,
    llm_reason: str = "",
) -> BestLead:
    if not people:
        return BestLead()
    best = people[0]
    components = best.score.as_dict()
    return BestLead(
        person_id=best.person_id,
        name=best.name,
        title=best.title,
        role_category=best.role_category,
        reason=llm_reason or _default_reason(best),
        confidence=best.confidence,
        score=best.score.total,
        score_components=components,
        sources=list(best.sources)[:6],
    )


def _default_reason(person: Person) -> str:
    parts = [f"{person.role_category.replace('_', ' ')} at this company"]
    if person.title:
        parts.append(f"titled '{person.title}'")
    source_count = len({item.source_url for item in person.evidence})
    if source_count:
        parts.append(f"supported by {source_count} source{'s' if source_count > 1 else ''}")
    if person.score.total >= 70:
        parts.append("high overall authority score")
    elif person.score.total >= 45:
        parts.append("moderate overall authority score")
    else:
        parts.append("weak evidence; verify before outreach")
    return "; ".join(parts[:4])


def grade(
    *,
    identity: CompanyIdentity,
    people: list[Person],
    contact: ContactInfo,
    domain_ok: bool,
    llm=None,
    llm_summary: str = "",
) -> GradeOutcome:
    """Deterministic grade with an LLM cross-check that cannot inflate it."""
    outcome = _deterministic_grade(
        identity=identity, people=people, contact=contact, domain_ok=domain_ok
    )

    if llm is not None and llm.enabled:
        try:
            decision = _ask_llm(llm, identity, people, contact, outcome)
        except (LLMCallFailed, LLMUnavailable):
            decision = None
        if decision is not None:
            # The model may lower or explain the grade but never raise it above
            # what the evidence supports.
            order = ["FAIL", "E", "D", "C", "B", "A"]
            proposed = decision.grade
            if order.index(proposed) < order.index(outcome.grade):
                outcome.grade = proposed
            if decision.grade_reason:
                outcome.reason = decision.grade_reason
            if decision.summary:
                outcome.reason = f"{decision.summary} | {outcome.reason}"
            if decision.failure_category:
                outcome.failure_category = decision.failure_category
    return outcome


def _ask_llm(
    llm,
    identity: CompanyIdentity,
    people: list[Person],
    contact: ContactInfo,
    outcome: GradeOutcome,
) -> P.ContactDecision | None:
    company_block = "\n".join(
        f"- {field}: {value}"
        for field, value in (
            ("canonical name", identity.canonical_name),
            ("legal name", identity.legal_name),
            ("domain", identity.domain),
            ("country", identity.country),
            ("type", identity.product_or_company),
            ("status", identity.status),
        )
        if value
    )
    people_block = "\n".join(
        f"- {p.name} | {p.title or 'title unknown'} | confidence {p.confidence} | "
        f"sources: {', '.join(p.sources[:3])}"
        for p in people[:5]
    ) or "(none)"
    contact_block = "\n".join(
        f"- {field}: {value}"
        for field, value in (
            ("direct email", contact.direct_email),
            ("email method", contact.email_method),
            ("email confidence", contact.email_confidence),
            ("general email", contact.general_email),
            ("contact form", contact.contact_form),
            ("phone", contact.phone),
            ("other", contact.other),
        )
        if value not in ("", 0.0, None)
    ) or "(none)"
    sources = list(dict.fromkeys(contact.evidence_urls + [s for p in people for s in p.sources]))

    user = P.build_synthesis_prompt(
        input_company=identity.canonical_name or "unknown",
        company_block=company_block or "(unresolved)",
        people_block=people_block,
        contact_block=contact_block,
        sources=sources,
    )
    return llm.structured(P.SYNTHESIS_SYSTEM, user, P.ContactDecision)


def _deterministic_grade(
    *,
    identity: CompanyIdentity,
    people: list[Person],
    contact: ContactInfo,
    domain_ok: bool,
) -> GradeOutcome:
    """The floor: what the retrieved evidence alone justifies."""
    decision_makers = [p for p in people if _is_decision_maker(p) and p.is_current]
    named = decision_makers or [p for p in people if p.is_current]

    identity_ok = bool(identity.canonical_name) and identity.confidence >= 0.5

    if not identity_ok:
        category = "wrong/invalid domain" if not domain_ok else "entity failure"
        return GradeOutcome(
            grade="FAIL",
            status="FAILED_ENTITY_RESOLUTION",
            failure_category=category,
            reason=(
                "domain did not resolve to a confident company identity"
                if not domain_ok
                else "company identity could not be confirmed from retrieved sources"
            ),
        )

    if not named:
        return GradeOutcome(
            grade="E" if contact.has_path else "FAIL",
            status="FAILED_PERSON_DISCOVERY" if not contact.has_path else "PARTIAL",
            failure_category="person failure",
            reason=(
                "company resolved but no named decision-maker could be evidenced"
                if not contact.has_path
                else "company resolved with generic contact only; no decision-maker found"
            ),
        )

    strong = [p for p in named if p.confidence >= _STRONG_PERSON]
    representative_only = bool(named) and not strong and named[0].confidence < 0.4

    email = contact.direct_email
    email_is_verified = bool(email) and contact.email_confidence >= _VERIFIED_EMAIL
    email_is_probable = bool(email) and contact.email_confidence >= _PROBABLE_EMAIL

    if email_is_verified and strong:
        grade, reason = "A", (
            f"{named[0].name} ({named[0].title or named[0].role_category}) with a "
            f"published/verified address on {email}"
        )
    elif email_is_probable and strong:
        grade, reason = "B", (
            f"{named[0].name} ({named[0].title or named[0].role_category}) with a "
            f"probable direct address ({contact.email_method})"
        )
    elif email and strong:
        grade, reason = "B", (
            f"{named[0].name} ({named[0].title or named[0].role_category}); direct "
            f"address present but only {contact.email_method} (confidence "
            f"{contact.email_confidence:.2f})"
        )
    elif named and contact.general_email:
        grade, reason = "C", (
            f"{named[0].name} ({named[0].title or named[0].role_category}) reachable "
            f"via official mailbox {contact.general_email}"
        )
    elif named and contact.contact_form:
        grade, reason = "C", (
            f"{named[0].name} ({named[0].title or named[0].role_category}) reachable "
            f"via official contact form {contact.contact_form}"
        )
    elif named and contact.phone:
        grade, reason = "C", (
            f"{named[0].name} ({named[0].title or named[0].role_category}) reachable "
            f"via published phone {contact.phone}"
        )
    elif named and contact.other:
        grade, reason = "C", (
            f"{named[0].name} reachable via official channel {contact.other}"
        )
    elif representative_only and contact.has_path:
        grade, reason = "D", (
            f"named representative {named[0].name} with weak role evidence; official "
            f"contact path available"
        )
    elif named and contact.has_path:
        grade, reason = "C", (
            f"{named[0].name} ({named[0].title or 'title unconfirmed'}) with an "
            f"official contact path"
        )
    elif contact.has_path:
        grade, reason = "E", "company resolved; only generic contact information found"
    else:
        return GradeOutcome(
            grade="FAIL",
            status="FAILED_CONTACT_DISCOVERY",
            failure_category="contact failure",
            reason="no usable contact path found in any retrieved page",
        )

    status = "SUCCESS" if grade in {"A", "B"} else "PARTIAL"
    if grade == "C" and identity.confidence < 0.7:
        status = "PARTIAL"
    return GradeOutcome(grade=grade, reason=reason, status=status)  # type: ignore[arg-type]


def summarize(identity: CompanyIdentity, best: BestLead, contact: ContactInfo) -> str:
    """One line a human can read in the CSV."""
    if not identity.canonical_name:
        return "unresolved company"
    bits = [identity.canonical_name]
    if identity.legal_name and identity.legal_name != identity.canonical_name:
        bits.append(f"({identity.legal_name})")
    if identity.country:
        bits.append(f"[{identity.country}]")
    if best.name:
        bits.append(f"- {best.name} {best.title or ''}".rstrip())
    if contact.direct_email:
        bits.append(f"<{contact.direct_email}>")
    elif contact.general_email:
        bits.append(f"<{contact.general_email}>")
    elif contact.contact_form:
        bits.append(contact.contact_form)
    return " ".join(bits)
