"""Person candidates, role taxonomy and deterministic candidate scoring."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from leadfinder.models.evidence import Evidence

# --- Role taxonomy -------------------------------------------------------
# Ordered roughly by decision authority for a small company. Matching is done
# on normalised substrings, so it also catches non-English titles
# (e.g. "代表取締役" contains the "representative director" concept).
ROLE_TIERS: list[tuple[list[str], str, int]] = [
    (["founder", "founder & ceo", "co-founder", "cofounder", "創業者", "创设者", "创始人"],
     "founder", 25),
    (["owner", "proprietor", "sole proprietor", "part owner"],
     "owner", 24),
    (["chief executive", "ceo", "ceo & founder", " managing director",
      "ceo/md", "geschäftsführer", "代表執行役", "代表取締役", "代表者", "社长", "社長",
      "法定代表人", "总经理", "執行長", "대표이사"],
     "chief_executive", 23),
    (["coo", "chief operating", "co-founder & coo"],
     "chief_operating", 20),
    (["president", "président", "cto", "chief technology", "chief product",
      "chief financial", "cfo", "chief information"],
     "executive", 19),
    (["founding partner", "managing partner", "partner", "head of",
      "head designer", "head developer", "head engineer", "vp ", "vice president",
      "vice-president", "general manager", "gm", "geschäftsführung", "本部長", "室長",
      "部长", "负责人", "head of "],
     "senior_leader", 16),
    (["business development", "partnerships", "partnership", "bdm", "sales director",
      "head of sales", "account executive", " partnerships", "商务合作", "事業開発",
      "営業", "提携"],
     "commercial_leader", 14),
    (["product manager", "product lead", "head of product", "product owner",
      "产品经理", "プロダクト"],
     "product_leader", 12),
    (["operations", "ops lead", "head of ops", "general operations"],
     "operations_leader", 11),
    (["marketing", "growth", "head of marketing", "cmc"],
     "marketing_leader", 9),
    (["investor", "advisor", "board", "director", "non-executive", "supervisor"],
     "non_executive", 5),
    (["engineer", "developer", "designer", "consultant", "employee", "staff",
      "intern", "coordinator", "specialist", "staff engineer"],
     "individual_contributor", 3),
]

# Titles that indicate the person is not currently in a buying role.
_DEPRECATED_HINTS = [
    "former", "previously", "ex-", "past ", "until 20", "2019", "2018", "2017",
    "joined ", "left ", "deceased",
]

# Company-size-sensitive authority adjustment. A Head of Partnerships can be
# the right lead for a large org and a poor one for a 3-person shop, and vice
# versa (founder of a 500-person company may not own the budget).
SIZE_PREFERENCE: dict[str, dict[str, int]] = {
    "small": {"founder": 4, "owner": 4, "chief_executive": 3, "senior_leader": 0,
              "commercial_leader": -1, "non_executive": -3, "individual_contributor": -4},
    "medium": {"founder": 2, "owner": 2, "chief_executive": 3, "executive": 2,
               "senior_leader": 2, "commercial_leader": 2, "non_executive": -2,
               "individual_contributor": -3},
    "large": {"founder": 0, "owner": 0, "chief_executive": 3, "executive": 3,
              "senior_leader": 3, "commercial_leader": 3, "product_leader": 1,
              "non_executive": -1, "individual_contributor": -4},
}

# Host-level trust. An executive title on the company's own domain is the
# strongest possible signal; a directory listing is weak.
HOST_TRUST: dict[str, float] = {
    "company_site": 1.0,
    "registry": 0.95,
    "press_release": 0.85,
    "partner_announcement": 0.8,
    "news": 0.7,
    "interview": 0.7,
    "product_hunt": 0.7,
    "conference": 0.65,
    "social_profile": 0.6,
    "github": 0.55,
    "pdf": 0.55,
    "directory": 0.45,
    "job_board": 0.4,
    "forum": 0.25,
    "other": 0.4,
}


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def classify_role(title: str) -> tuple[str, int]:
    """Return (role_category, authority_0_25) for a job title."""
    norm = normalize_title(title)
    if not norm:
        return "unknown", 0
    for keywords, category, authority in ROLE_TIERS:
        for keyword in keywords:
            if keyword.strip() and keyword in norm:
                return category, authority
    if re.search(r"\bchief\b|\bceo\b|\bpresident\b", norm):
        return "chief_executive", 22
    return "unknown", 4


def infer_size_hint(text: str) -> str:
    """Best-effort company size band from any gathered text."""
    blob = (text or "").lower()
    if re.search(r"\b(\d{1,3})\s*[-+]?\s*(employees|staff|people)\b", blob):
        match = re.search(r"\b(\d{1,4})\s*[-+]?\s*(employees|staff|people)\b", blob)
        if match:
            try:
                count = int(match.group(1))
            except ValueError:
                count = 0
            if count >= 250:
                return "large"
            if count >= 25:
                return "medium"
    if re.search(r"\b(solo founder|one-person|2[- ]person|tiny team|small team|indie hacker)\b", blob):
        return "small"
    if re.search(r"\b(series [a-e]\b|funding round|raised \$|unicorn|ipo|nasdaq|fortune 500)\b", blob):
        return "large"
    if re.search(r"\b(saas platform|enterprise software|global team|offices in)\b", blob):
        return "medium"
    return ""


class CandidateScore(BaseModel):
    """Component scores, stored individually so a grade can be audited."""

    company_match: float = 0.0       # 0-30
    decision_authority: float = 0.0  # 0-25
    role_relevance: float = 0.0      # 0-20
    evidence_quality: float = 0.0    # 0-15
    evidence_recency: float = 0.0    # 0-10

    @property
    def total(self) -> float:
        return round(
            self.company_match
            + self.decision_authority
            + self.role_relevance
            + self.evidence_quality
            + self.evidence_recency,
            1,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "company_match": round(self.company_match, 1),
            "decision_authority": round(self.decision_authority, 1),
            "role_relevance": round(self.role_relevance, 1),
            "evidence_quality": round(self.evidence_quality, 1),
            "evidence_recency": round(self.evidence_recency, 1),
            "total": self.total,
        }


class Person(BaseModel):
    """A candidate decision-maker found for a company."""

    name: str
    title: str = ""
    role_category: str = "unknown"
    current_company: str = ""
    linkedin: str = ""
    location: str = ""
    confidence: float = 0.0
    score: CandidateScore = Field(default_factory=CandidateScore)
    sources: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list, repr=False)
    is_current: bool = True
    notes: str = ""
    person_id: int | None = None

    @property
    def display(self) -> str:
        return f"{self.name} - {self.title}" if self.title else self.name


def _recency_score(evidence: list[Evidence]) -> float:
    """Fresher evidence scores higher; undated evidence gets a neutral 4/10."""
    if not evidence:
        return 0.0
    years = []
    for item in evidence:
        match = re.search(r"\b(20[12]\d)\b", item.source_text or item.claim_value or "")
        if match:
            years.append(int(match.group(1)))
    if not years:
        return 4.0
    newest = max(years)
    age = max(0, datetime.now(timezone.utc).year - newest)
    if age <= 1:
        return 10.0
    if age == 2:
        return 8.0
    if age <= 4:
        return 6.0
    return 3.0


def score_person(
    person: Person,
    *,
    company_terms: list[str],
    company_domains: list[str],
    size_hint: str = "",
) -> CandidateScore:
    """Deterministic 0-100 candidate score from the five defined components."""
    text = " ".join(
        [person.name, person.title, person.current_company, person.location]
        + [item.claim_value for item in person.evidence]
    ).lower()

    # 1. company_match (0-30): does the evidence tie them to *this* company?
    if person.evidence:
        tied = 0
        for item in person.evidence:
            blob = f"{item.claim_value} {item.source_text} {item.source_url}".lower()
            host_match = any(d and d in item.source_url.lower() for d in company_domains)
            term_match = any(t and t in blob for t in company_terms)
            if host_match and term_match:
                tied += 30
            elif host_match or term_match:
                tied += 18
            else:
                tied += 6
        company_match = tied / len(person.evidence)
    else:
        company_match = 0.0
    if person.current_company and any(
        t and t in person.current_company.lower() for t in company_terms
    ):
        company_match = min(30.0, company_match + 6)

    # 2. decision_authority (0-25)
    _, base_authority = classify_role(person.title)
    if re.search("|".join(re.escape(h) for h in _DEPRECATED_HINTS), text):
        base_authority = max(0, int(base_authority * 0.3))
        if not person.is_current:
            base_authority = max(0, base_authority - 5)
    adjustments = SIZE_PREFERENCE.get(size_hint or "medium", SIZE_PREFERENCE["medium"])
    decision_authority = max(0.0, min(25.0, base_authority + adjustments.get(person.role_category, 0)))

    # 3. role_relevance (0-20): commercial/product authority is what a lead
    #    programme cares about; pure IC or board roles are not.
    relevance_by_category = {
        "founder": 20, "owner": 20, "chief_executive": 19, "chief_operating": 16,
        "executive": 16, "senior_leader": 15, "commercial_leader": 18,
        "product_leader": 12, "operations_leader": 10, "marketing_leader": 9,
        "non_executive": 4, "individual_contributor": 2, "unknown": 4,
    }
    role_relevance = float(relevance_by_category.get(person.role_category, 4))
    if not person.is_current:
        role_relevance *= 0.4

    # 4. evidence_quality (0-15): source trust x corroboration count.
    if person.evidence:
        trust = [HOST_TRUST.get(item.source_type, 0.4) for item in person.evidence]
        corroboration = min(1.0, len(person.evidence) / 3.0)
        evidence_quality = round(sum(trust) / len(trust) * 15 * (0.55 + 0.45 * corroboration), 1)
    else:
        evidence_quality = 0.0

    # 5. evidence_recency (0-10)
    evidence_recency = _recency_score(person.evidence)

    return CandidateScore(
        company_match=round(min(30.0, company_match), 1),
        decision_authority=round(decision_authority, 1),
        role_relevance=round(role_relevance, 1),
        evidence_quality=round(evidence_quality, 1),
        evidence_recency=round(evidence_recency, 1),
    )


# --- Name normalisation --------------------------------------------------

_NAME_NOISE = re.compile(
    r"\b(ceo|cfo|coo|cto|cio|cmo|vp|svp|evp|president|director|manager|founder|owner|"
    r"mr|mrs|ms|dr|prof|jr|sr|ii|iii)\b"
)


def normalize_person_name(raw: str) -> str:
    """Collapse a name to comparable tokens, dropping titles and punctuation."""
    text = (raw or "").strip()
    text = re.sub(r"[^\w\s.'-]", " ", text, flags=re.UNICODE)
    text = _NAME_NOISE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def person_name_key(raw: str) -> str:
    """Surname + first-initial key used to merge duplicate mentions."""
    tokens = normalize_person_name(raw).split()
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[-1]}|{tokens[0][0]}"


_NAME_TOKEN_STOPWORDS = {"the", "and", "of", "for", "ltd", "llc", "inc", "gmbh", "co"}


def person_name_conflict(a: str, b: str) -> bool:
    """True when two names are definitely different people."""
    key_a, key_b = person_name_key(a), person_name_key(b)
    if not key_a or not key_b:
        return False
    if key_a == key_b:
        return False
    tokens_a = {t for t in normalize_person_name(a).split() if t not in _NAME_TOKEN_STOPWORDS}
    tokens_b = {t for t in normalize_person_name(b).split() if t not in _NAME_TOKEN_STOPWORDS}
    if not tokens_a or not tokens_b:
        return False
    return not (tokens_a & tokens_b)
