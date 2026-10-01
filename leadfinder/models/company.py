"""Company identity models and input-row loading."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Characters that indicate a name arrived mangled (mojibake, truncation,
# spreadsheet damage). A name made only of these is unusable as a query.
_JUNK_CHARS = re.compile(r"[\?\uFFFD\u0000-\u001F]")


def clean_name(raw: str) -> str:
    """Strip junk characters and collapse whitespace.

    Frozen benchmark lists routinely contain names where non-Latin characters
    were replaced by '?' on export. We keep the file untouched but treat such
    a name as low-value for search purposes rather than querying "??" literally.
    """
    text = (raw or "").replace("\u00a0", " ")
    text = _JUNK_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_is_degraded(name: str) -> bool:
    """True when the name lost its non-ASCII characters on export."""
    original = (name or "").strip()
    if not original:
        return True
    if any(ch in original for ch in "??"):
        return True
    return False


class CompanyIdentity(BaseModel):
    """Resolved entity behind the supplied name/domain."""

    canonical_name: str = ""
    legal_name: str = ""
    domain: str = ""
    country: str = ""
    language: str = ""
    company_type: str = ""
    product_or_company: Literal["brand", "legal_company", "unknown"] = "unknown"
    parent_company: str = ""
    status: str = "unknown"
    aliases: list[str] = Field(default_factory=list)
    industry: str = ""
    size_hint: str = ""
    confidence: float = 0.0
    evidence_urls: list[str] = Field(default_factory=list)
    resolution_note: str = ""


class CompanyInput(BaseModel):
    """One row of the input CSV. Extra columns are preserved in `meta`."""

    company_name: str
    domain: str
    row_index: int = 0
    meta: dict[str, str] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.company_name or self.domain

    @property
    def degraded_name(self) -> bool:
        return name_is_degraded(self.company_name)


_NAME_KEYS = ("company_name", "company", "name", "company_nme", "organisation")
_DOMAIN_KEYS = ("domain", "company_domain", "website", "url", "site", "domain_name")

RESERVED_META = {"company_name", "domain"}


def _pick(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key]:
            return row[key].strip()
    return ""


def load_company_csv(path: str | Path) -> list[CompanyInput]:
    """Load the benchmark CSV.

    Column names are matched case-insensitively and tolerantly so the frozen
    file can be used verbatim. Any additional column (e.g. Apollo's verdict and
    person count) is preserved in `meta` for benchmark comparison, and the file
    itself is never rewritten.
    """
    rows: list[CompanyInput] = []
    lowered: dict[str, str] = {}

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows
        for original in reader.fieldnames:
            if original is None:
                continue
            lowered[original.strip().lower()] = original

        for index, raw_row in enumerate(reader):
            if raw_row is None:
                continue
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw_row.items()}
            company = _pick(row, _NAME_KEYS)
            domain = _pick(row, _DOMAIN_KEYS)
            if not company and not domain:
                continue
            company = clean_name(company) or domain
            domain = domain.lower()
            if domain.startswith(("http://", "https://")):
                domain = domain.split("://", 1)[1].strip("/").split("/")[0]
            domain = domain.removeprefix("www.")
            if not company or not domain:
                continue

            meta = {
                lowered_key: (raw_row.get(original) or "").strip()
                for lowered_key, original in lowered.items()
                if lowered_key not in RESERVED_META
            }
            rows.append(
                CompanyInput(company_name=company, domain=domain, row_index=index, meta=meta)
            )
    return rows
