"""Evidence records.

Every important claim in the output must be traceable to at least one
Evidence row. LLM conclusions are *not* evidence - only retrieved text is.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal[
    "company_name",
    "legal_name",
    "domain",
    "country",
    "language",
    "company_type",
    "product_or_company",
    "parent_company",
    "status",
    "person_name",
    "person_title",
    "person_company",
    "person_relationship",
    "direct_email",
    "general_email",
    "phone",
    "contact_form",
    "other_contact",
    "source_type",
]

SourceType = Literal[
    "company_website",
    "press_release",
    "product_hunt",
    "github",
    "conference",
    "partner_announcement",
    "registry",
    "news",
    "interview",
    "pdf",
    "social_profile",
    "directory",
    "job_board",
    "forum",
    "other",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Source(BaseModel):
    """A URL that contributed to a result."""

    url: str
    title: str = ""
    source_type: SourceType = "other"
    retrieved_at: str = Field(default_factory=utcnow)

    @property
    def host(self) -> str:
        from urllib.parse import urlparse

        return (urlparse(self.url).hostname or "").lower()


class Evidence(BaseModel):
    """One retrieved snippet that supports one claim.

    `quote` is the literal text from the page, not a paraphrase, so a human
    reviewing a lead can see exactly why the system believes it.
    """

    claim_type: ClaimType
    claim_value: str
    source_url: str
    source_title: str = ""
    source_text: str = ""
    retrieved_at: str = Field(default_factory=utcnow)
    confidence: float = 0.5
    person_id: int | None = None
    # Inherited from the page this came from; drives host trust when scoring.
    source_type: SourceType = "other"

    @property
    def host(self) -> str:
        from urllib.parse import urlparse

        return (urlparse(self.source_url).hostname or "").lower()
