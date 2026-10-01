"""Benchmark metrics.

The headline number is the A-D rate. Everything else exists to explain it: the
grade histogram, the failure-category breakdown, and the per-company cost of
searches, pages and Qwen calls.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from leadfinder.models.result import LeadResult

GRADES = ["A", "B", "C", "D", "E", "FAIL"]

FAILURE_CATEGORIES = [
    "entity failure",
    "person failure",
    "contact failure",
    "wrong/invalid domain",
    "insufficient evidence",
    "technical failure",
]

_STATUS_TO_CATEGORY = {
    "FAILED_DOMAIN": "wrong/invalid domain",
    "FAILED_ENTITY_RESOLUTION": "entity failure",
    "FAILED_PERSON_DISCOVERY": "person failure",
    "FAILED_CONTACT_DISCOVERY": "contact failure",
    "ERROR": "technical failure",
}


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


@dataclass(slots=True)
class Metrics:
    total: int = 0
    entity_resolved_pct: float = 0.0
    active_identified_pct: float = 0.0
    person_found_pct: float = 0.0
    decision_maker_found_pct: float = 0.0
    two_decision_makers_pct: float = 0.0
    grade_pct: dict[str, float] = field(default_factory=dict)
    grade_counts: dict[str, int] = field(default_factory=dict)
    a_d_pct: float = 0.0
    a_e_pct: float = 0.0
    avg_searches: float = 0.0
    avg_pages: float = 0.0
    avg_llm_calls: float = 0.0
    median_duration: float = 0.0
    failure_categories: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)
    comparison: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "total_companies": self.total,
            "entity_resolved_pct": self.entity_resolved_pct,
            "active_company_identified_pct": self.active_identified_pct,
            "person_found_pct": self.person_found_pct,
            "decision_maker_found_pct": self.decision_maker_found_pct,
            "two_or_more_decision_makers_pct": self.two_decision_makers_pct,
            "grade_pct": self.grade_pct,
            "grade_counts": self.grade_counts,
            "a_d_pct": self.a_d_pct,
            "a_e_pct": self.a_e_pct,
            "avg_searches_per_company": self.avg_searches,
            "avg_pages_per_company": self.avg_pages,
            "avg_qwen_calls_per_company": self.avg_llm_calls,
            "median_wall_clock_seconds": self.median_duration,
            "failure_categories": self.failure_categories,
            "status_counts": self.status_counts,
            "apollo_comparison": self.comparison,
        }


_DECISION_CATEGORIES = {
    "founder", "owner", "chief_executive", "chief_operating", "executive",
    "senior_leader", "commercial_leader", "product_leader",
}


def _is_decision_maker(result: LeadResult) -> list:
    return [
        person
        for person in result.people
        if person.role_category in _DECISION_CATEGORIES and person.is_current
    ]


def compute_metrics(
    results: list[LeadResult],
    *,
    inputs: list | None = None,
) -> Metrics:
    metrics = Metrics(total=len(results))
    if not results:
        return metrics

    grades = Counter(result.lead_grade for result in results)
    metrics.grade_counts = {grade: grades.get(grade, 0) for grade in GRADES}
    metrics.grade_pct = {
        grade: _pct(grades.get(grade, 0), len(results)) for grade in GRADES
    }

    resolved = [r for r in results if r.company.canonical_name and r.company.confidence >= 0.5]
    active = [r for r in results if r.company.status == "active"]
    with_person = [r for r in results if r.people]
    with_dm = [r for r in results if _is_decision_maker(r)]
    with_two_dm = [r for r in results if len(_is_decision_maker(r)) >= 2]

    metrics.entity_resolved_pct = _pct(len(resolved), len(results))
    metrics.active_identified_pct = _pct(len(active), len(results))
    metrics.person_found_pct = _pct(len(with_person), len(results))
    metrics.decision_maker_found_pct = _pct(len(with_dm), len(results))
    metrics.two_decision_makers_pct = _pct(len(with_two_dm), len(results))

    a_d = sum(grades.get(grade, 0) for grade in ("A", "B", "C", "D"))
    a_e = a_d + grades.get("E", 0)
    metrics.a_d_pct = _pct(a_d, len(results))
    metrics.a_e_pct = _pct(a_e, len(results))

    metrics.avg_searches = _mean([float(r.research.searches_performed) for r in results])
    metrics.avg_pages = _mean([float(r.research.pages_fetched) for r in results])
    metrics.avg_llm_calls = _mean([float(r.research.llm_calls) for r in results])
    metrics.median_duration = _median([r.research.duration_seconds for r in results])

    failures: Counter[str] = Counter()
    for result in results:
        if result.lead_grade in {"A", "B", "C", "D", "E"}:
            continue
        category = (
            result.failure_category
            or _STATUS_TO_CATEGORY.get(result.status, "insufficient evidence")
        )
        failures[category] += 1
    metrics.failure_categories = {cat: failures.get(cat, 0) for cat in FAILURE_CATEGORIES}
    for key in failures:
        metrics.failure_categories.setdefault(key, failures[key])

    metrics.status_counts = dict(Counter(result.status for result in results))

    if inputs:
        metrics.comparison = _compare_with_baseline(results, inputs)
    return metrics


def _compare_with_baseline(results: list[LeadResult], inputs: list) -> dict[str, int]:
    """Side-by-side with whatever the input CSV recorded as the prior result.

    The benchmark file ships with an 'Apollo said' and a 'People found' column.
    They are kept verbatim and reported alongside, which is what makes this a
    benchmark rather than just a run.

    The counts are kept like-for-like: 'with_people' compares companies where
    each side found at least one person, and 'people' compares total people -
    not decision makers, which is a stricter subset on this side only.
    """
    by_domain = {item.domain.lower(): item for item in inputs}
    counts = {
        "apollo_completed": 0,
        "apollo_with_people": 0,
        "apollo_people": 0,
        "leadfinder_leads": 0,
        "leadfinder_with_people": 0,
        "leadfinder_people": 0,
        "leadfinder_decision_makers": 0,
        "both": 0,
        "leadfinder_only": 0,
        "apollo_only": 0,
    }
    total = 0
    for result in results:
        item = by_domain.get(result.input_domain.lower())
        if item is None:
            continue
        total += 1
        meta = {k.lower(): v for k, v in item.meta.items()}
        apollo_verdict = (meta.get("apollo said") or "").strip().lower()
        try:
            apollo_people = int((meta.get("people found") or "0").strip() or 0)
        except ValueError:
            apollo_people = 0
        apollo_ok = apollo_verdict in {"completed", "complete", "done"}
        lf_ok = result.lead_grade in {"A", "B", "C", "D"}
        lf_people = len(result.people)
        lf_dms = len(_is_decision_maker(result))

        counts["apollo_completed"] += int(apollo_ok)
        counts["apollo_with_people"] += int(apollo_people > 0)
        counts["apollo_people"] += apollo_people
        counts["leadfinder_leads"] += int(lf_ok)
        counts["leadfinder_with_people"] += int(lf_people > 0)
        counts["leadfinder_people"] += lf_people
        counts["leadfinder_decision_makers"] += lf_dms
        if apollo_ok and lf_ok:
            counts["both"] += 1
        elif lf_ok:
            counts["leadfinder_only"] += 1
        elif apollo_ok:
            counts["apollo_only"] += 1
    if total:
        counts["total"] = total
    return counts


def missing_inputs(results: list[LeadResult], inputs: list) -> list[str]:
    """Guard against silently dropping rows."""
    seen = {result.input_domain.lower() for result in results}
    return [item.domain for item in inputs if item.domain.lower() not in seen]
