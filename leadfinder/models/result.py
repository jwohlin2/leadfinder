"""Output result models: contacts, best lead, grades, per-company research."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from leadfinder.models.company import CompanyIdentity
from leadfinder.models.evidence import Evidence, Source
from leadfinder.models.person import Person

LeadGrade = Literal["A", "B", "C", "D", "E", "FAIL"]

RunStatus = Literal[
    "SUCCESS",
    "PARTIAL",
    "FAILED_DOMAIN",
    "FAILED_ENTITY_RESOLUTION",
    "FAILED_PERSON_DISCOVERY",
    "FAILED_CONTACT_DISCOVERY",
    "ERROR",
]


class ContactInfo(BaseModel):
    direct_email: str = ""
    email_confidence: float = 0.0
    email_method: str = ""        # published | inferred | openenrich | none
    general_email: str = ""
    contact_form: str = ""
    phone: str = ""
    other: str = ""
    wechat: str = ""
    contact_page: str = ""
    evidence_urls: list[str] = Field(default_factory=list)

    @property
    def has_path(self) -> bool:
        return bool(
            self.direct_email
            or self.general_email
            or self.contact_form
            or self.phone
            or self.other
            or self.wechat
        )


class BestLead(BaseModel):
    person_id: int | None = None
    name: str = ""
    title: str = ""
    role_category: str = ""
    reason: str = ""
    confidence: float = 0.0
    score: float = 0.0
    score_components: dict[str, float] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


class ResearchStats(BaseModel):
    queries: list[str] = Field(default_factory=list)
    searches_performed: int = 0
    pages_fetched: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    cache_hits: int = 0
    duration_seconds: float = 0.0
    rounds: int = 0
    stop_reason: str = ""
    error: str = ""
    # Set when writing results to SQLite failed. Never fatal, but a benchmark
    # with a persist_error has grades without stored evidence behind them.
    persist_error: str = ""


class LeadResult(BaseModel):
    input_company: str
    input_domain: str
    company: CompanyIdentity = Field(default_factory=CompanyIdentity)
    people: list[Person] = Field(default_factory=list, repr=False)
    best_lead: BestLead = Field(default_factory=BestLead)
    contact: ContactInfo = Field(default_factory=ContactInfo)
    lead_grade: LeadGrade = "FAIL"
    grade_reason: str = ""
    status: RunStatus = "ERROR"
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list, repr=False)
    research: ResearchStats = Field(default_factory=ResearchStats)
    aliases: list[str] = Field(default_factory=list)
    related_entities: list[dict[str, str]] = Field(default_factory=list)
    failure_category: str = ""

    def to_json_dict(self, *, include_evidence: bool = True) -> dict:
        """Shape matches the handoff contract."""
        payload = {
            "input_company": self.input_company,
            "input_domain": self.input_domain,
            "company": self.company.model_dump(),
            "people": [person.model_dump(exclude={"evidence"}) for person in self.people],
            "best_lead": self.best_lead.model_dump(),
            "contact": self.contact.model_dump(),
            "lead_grade": self.lead_grade,
            "grade_reason": self.grade_reason,
            "status": self.status,
            "sources": [source.model_dump() for source in self.sources],
            "research": self.research.model_dump(),
            "aliases": self.aliases,
            "related_entities": self.related_entities,
            "failure_category": self.failure_category,
        }
        if include_evidence:
            payload["evidence"] = [item.model_dump(exclude={"person_id"}) for item in self.evidence]
        return payload
