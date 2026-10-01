"""Benchmark metric regressions.

The Apollo comparison must stay like-for-like: companies where each side found
at least one person, and total people - not this side's stricter
decision-maker subset against Apollo's whole people column.
"""

from leadfinder.benchmark.metrics import _compare_with_baseline
from leadfinder.models.company import CompanyInput
from leadfinder.models.person import Person
from leadfinder.models.result import LeadResult


def _result(domain: str, grade: str, people: list[Person]) -> LeadResult:
    return LeadResult(
        input_company=domain,
        input_domain=domain,
        lead_grade=grade,
        people=people,
    )


def test_comparison_counts_are_like_for_like():
    inputs = [
        CompanyInput(
            company_name="A",
            domain="a.com",
            meta={"Apollo said": "completed", "People found": "3"},
        ),
        CompanyInput(
            company_name="B",
            domain="b.com",
            meta={"Apollo said": "thin", "People found": "0"},
        ),
    ]
    results = [
        _result(
            "a.com",
            "C",
            [
                Person(name="Ann A", role_category="founder"),
                Person(name="Bob B", role_category="individual_contributor"),
            ],
        ),
        _result("b.com", "C", [Person(name="Cara C", role_category="founder")]),
    ]

    comparison = _compare_with_baseline(results, inputs)

    assert comparison["apollo_completed"] == 1
    assert comparison["apollo_with_people"] == 1
    assert comparison["apollo_people"] == 3
    assert comparison["leadfinder_leads"] == 2
    assert comparison["leadfinder_with_people"] == 2
    assert comparison["leadfinder_people"] == 3
    assert comparison["leadfinder_decision_makers"] == 2
    assert comparison["both"] == 1
    assert comparison["leadfinder_only"] == 1
    assert comparison["apollo_only"] == 0
