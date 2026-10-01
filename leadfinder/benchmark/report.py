"""Output writers: leads.csv, results.json, benchmark_summary.md, report.html."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from leadfinder.benchmark.metrics import Metrics, compute_metrics, missing_inputs
from leadfinder.models.result import LeadResult

CSV_COLUMNS = [
    "input_company",
    "input_domain",
    "canonical_company",
    "legal_company",
    "country",
    "language",
    "company_type",
    "parent_company",
    "company_status",
    "best_person",
    "best_person_title",
    "best_person_role",
    "person_confidence",
    "candidate_score",
    "direct_email",
    "email_confidence",
    "email_method",
    "general_email",
    "contact_form",
    "phone",
    "other_contact",
    "lead_grade",
    "grade_reason",
    "company_confidence",
    "source_count",
    "person_count",
    "status",
    "failure_category",
    "searches",
    "pages",
    "llm_calls",
    "duration_seconds",
    "stop_reason",
    "summary",
    "source_urls",
]

LEADS_CSV = "leads.csv"
RESULTS_JSON = "results.json"
BENCHMARK_JSON = "benchmark_results.json"
BENCHMARK_CSV = "benchmark_results.csv"
SUMMARY_MD = "benchmark_summary.md"
REPORT_HTML = "report.html"


def _row(result: LeadResult) -> dict[str, object]:
    best = result.best_lead
    return {
        "input_company": result.input_company,
        "input_domain": result.input_domain,
        "canonical_company": result.company.canonical_name,
        "legal_company": result.company.legal_name,
        "country": result.company.country,
        "language": result.company.language,
        "company_type": result.company.product_or_company,
        "parent_company": result.company.parent_company,
        "company_status": result.company.status,
        "best_person": best.name,
        "best_person_title": best.title,
        "best_person_role": best.role_category,
        "person_confidence": round(best.confidence, 2),
        "candidate_score": best.score,
        "direct_email": result.contact.direct_email,
        "email_confidence": round(result.contact.email_confidence, 2),
        "email_method": result.contact.email_method,
        "general_email": result.contact.general_email,
        "contact_form": result.contact.contact_form,
        "phone": result.contact.phone,
        "other_contact": result.contact.other,
        "lead_grade": result.lead_grade,
        "grade_reason": result.grade_reason,
        "company_confidence": round(result.company.confidence, 2),
        "source_count": len(result.sources),
        "person_count": len(result.people),
        "status": result.status,
        "failure_category": result.failure_category,
        "searches": result.research.searches_performed,
        "pages": result.research.pages_fetched,
        "llm_calls": result.research.llm_calls,
        "duration_seconds": result.research.duration_seconds,
        "stop_reason": result.research.stop_reason,
        "summary": _summary(result),
        "source_urls": " | ".join(source.url for source in result.sources[:8]),
    }


def _summary(result: LeadResult) -> str:
    from leadfinder.research.ranker import summarize

    return summarize(result.company, result.best_lead, result.contact)


def write_csv(results: list[LeadResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(_row(result))
    return path


def write_json(results: list[LeadResult], path: Path, *, include_evidence: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(results),
        "results": [result.to_json_dict(include_evidence=include_evidence) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_summary(
    results: list[LeadResult],
    metrics: Metrics,
    path: Path,
    *,
    inputs: list | None = None,
    notes: list[str] | None = None,
) -> Path:
    lines: list[str] = []
    lines.append("# Lead Discovery Benchmark Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Total companies | {metrics.total} |")
    lines.append(f"| **A-D (primary metric)** | **{metrics.a_d_pct}%** |")
    lines.append(f"| A-E | {metrics.a_e_pct}% |")
    lines.append(f"| Entity resolved | {metrics.entity_resolved_pct}% |")
    lines.append(f"| Active company identified | {metrics.active_identified_pct}% |")
    lines.append("")

    lines.append("## Lead grades")
    lines.append("")
    lines.append("| Grade | Count | % |")
    lines.append("| --- | ---: | ---: |")
    for grade, count in metrics.grade_counts.items():
        lines.append(f"| {grade} | {count} | {metrics.grade_pct[grade]}% |")
    lines.append("")

    lines.append("## Discovery")
    lines.append("")
    lines.append("| Metric | % |")
    lines.append("| --- | ---: |")
    lines.append(f"| >=1 person found | {metrics.person_found_pct}% |")
    lines.append(f"| >=1 decision maker | {metrics.decision_maker_found_pct}% |")
    lines.append(f"| >=2 decision makers | {metrics.two_decision_makers_pct}% |")
    lines.append("")

    lines.append("## Cost per company")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Avg searches | {metrics.avg_searches} |")
    lines.append(f"| Avg pages | {metrics.avg_pages} |")
    lines.append(f"| Avg Qwen calls | {metrics.avg_llm_calls} |")
    lines.append(f"| Median wall-clock (s) | {metrics.median_duration} |")
    lines.append("")

    lines.append("## Failure categories")
    lines.append("")
    total_failures = sum(metrics.failure_categories.values())
    lines.append("| Category | Count |")
    lines.append("| --- | ---: |")
    for category, count in sorted(metrics.failure_categories.items(), key=lambda i: -i[1]):
        lines.append(f"| {category} | {count} |")
    if total_failures == 0:
        lines.append("| _(none)_ | 0 |")
    lines.append("")

    if metrics.comparison:
        lines.append("## Comparison with the recorded baseline")
        lines.append("")
        comparison = metrics.comparison
        total = comparison.get("total", 0)
        lines.append(f"Baseline rows compared: {total}")
        lines.append("")
        lines.append("| Metric | Apollo baseline | This run |")
        lines.append("| --- | ---: | ---: |")
        lines.append(
            f"| Companies with a lead (Apollo completed / this run A-D) "
            f"| {comparison.get('apollo_completed', 0)} "
            f"({_pct_of(comparison.get('apollo_completed', 0), total)}%) "
            f"| {comparison.get('leadfinder_leads', 0)} "
            f"({_pct_of(comparison.get('leadfinder_leads', 0), total)}%) |"
        )
        lines.append(
            f"| Companies with >=1 person "
            f"| {comparison.get('apollo_with_people', 0)} "
            f"({_pct_of(comparison.get('apollo_with_people', 0), total)}%) "
            f"| {comparison.get('leadfinder_with_people', 0)} "
            f"({_pct_of(comparison.get('leadfinder_with_people', 0), total)}%) |"
        )
        lines.append(
            f"| People found (total) | {comparison.get('apollo_people', 0)} "
            f"| {comparison.get('leadfinder_people', 0)} |"
        )
        lines.append(
            f"| Decision makers (this run only) | - "
            f"| {comparison.get('leadfinder_decision_makers', 0)} |"
        )
        lines.append(f"| Leads by both | {comparison.get('both', 0)} | - |")
        lines.append(f"| Leads only by Apollo | {comparison.get('apollo_only', 0)} | - |")
        lines.append(
            f"| Leads only by this run | - | {comparison.get('leadfinder_only', 0)} |"
        )
        lines.append("")

    lines.append("## Per-company results")
    lines.append("")
    lines.append(
        "| Input | Domain | Resolved entity | Best lead | Contact | Grade | Conf | Sources |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: |")
    for result in sorted(results, key=lambda r: (r.lead_grade, r.input_company)):
        contact = (
            result.contact.direct_email
            or result.contact.general_email
            or result.contact.contact_form
            or result.contact.phone
            or result.contact.other
            or "-"
        )
        entity = result.company.legal_name or result.company.canonical_name or "-"
        if result.company.canonical_name and entity != result.company.canonical_name:
            entity = f"{result.company.canonical_name} ({entity})"
        person = (
            f"{result.best_lead.name} - {result.best_lead.title}"
            if result.best_lead.name
            else "-"
        )
        lines.append(
            f"| {result.input_company} | {result.input_domain} | {entity} | {person} "
            f"| {contact} | **{result.lead_grade}** "
            f"| {result.best_lead.confidence:.2f} | {len(result.sources)} |"
        )
    lines.append("")

    if notes:
        lines.append("## Run notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    if inputs:
        dropped = missing_inputs(results, inputs)
        if dropped:
            lines.append("## Rows not accounted for")
            lines.append("")
            for domain in dropped:
                lines.append(f"- {domain}")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _pct_of(value: int, total: int) -> str:
    if not total:
        return "0.0"
    return f"{100.0 * value / total:.1f}"


def write_benchmark_files(
    results: list[LeadResult],
    directory: Path,
    *,
    inputs: list | None = None,
    notes: list[str] | None = None,
) -> dict[str, Metrics]:
    metrics = compute_metrics(results, inputs=inputs)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(results, directory / BENCHMARK_CSV)
    write_json(results, directory / BENCHMARK_JSON)
    write_summary(results, metrics, directory / SUMMARY_MD, inputs=inputs, notes=notes)
    return {"metrics": metrics}
