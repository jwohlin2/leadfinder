"""Benchmark orchestration: run the frozen list, then write the artefacts."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from leadfinder.benchmark.metrics import Metrics, compute_metrics, missing_inputs
from leadfinder.benchmark.report import (
    BENCHMARK_CSV,
    BENCHMARK_JSON,
    LEADS_CSV,
    RESULTS_JSON,
    SUMMARY_MD,
    write_csv,
    write_json,
    write_summary,
)
from leadfinder.models.company import CompanyInput
from leadfinder.models.result import LeadResult
from leadfinder.research.orchestrator import run_companies
from leadfinder.storage.db import Database

ProgressFn = Callable[[LeadResult, int, int], None]


@dataclass(slots=True)
class BenchmarkOutcome:
    results: list[LeadResult] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    inputs: list[CompanyInput] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    duration: float = 0.0
    artefacts: dict[str, str] = field(default_factory=dict)
    # Run conditions (LLM presence, budget, cache size). Recorded so a report
    # can be interpreted correctly: a deterministic-only run is a floor, not a
    # measurement of what the tool can do.
    notes: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        return (
            f"{self.metrics.total} companies | A-D {self.metrics.a_d_pct}% | "
            f"A-E {self.metrics.a_e_pct}% | entity resolved "
            f"{self.metrics.entity_resolved_pct}% | {self.duration:.0f}s"
        )


async def run_benchmark(
    *,
    items: list[CompanyInput],
    config,
    db: Database | None = None,
    workers: int | None = None,
    on_progress: ProgressFn | None = None,
) -> BenchmarkOutcome:
    """Run every supplied company exactly once, then produce all artefacts.

    The list is never modified based on what is discovered.
    """
    output_dir = config.resolve(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    database = db or Database(config.resolve(config.storage.database))

    total = len(items)
    completed = 0
    counter_lock = asyncio.Lock()

    def progress(result: LeadResult, done: int, count: int) -> None:
        if on_progress is not None:
            on_progress(result, done, count)

    def on_result(result: LeadResult) -> None:
        nonlocal completed
        completed += 1
        progress(result, completed, total)

    started = time.monotonic()
    results = await run_companies(
        items=items, config=config, db=database, workers=workers, on_result=on_result
    )
    duration = time.monotonic() - started

    # Never drop a row: anything that produced no result is emitted as a FAIL
    # row so the CSV always accounts for the full input list.
    seen = {result.input_domain.lower() for result in results}
    for item in items:
        if item.domain.lower() not in seen:
            fallback = LeadResult(input_company=item.company_name, input_domain=item.domain)
            fallback.status = "ERROR"
            fallback.lead_grade = "FAIL"
            fallback.failure_category = "technical failure"
            fallback.grade_reason = "no result produced for this input row"
            results.append(fallback)

    metrics = compute_metrics(results, inputs=items)
    outcome = BenchmarkOutcome(
        results=results,
        metrics=metrics,
        inputs=items,
        missing=missing_inputs(results, items),
        duration=duration,
    )

    notes = _run_notes(config, database)
    if outcome.missing:
        notes.append(f"{len(outcome.missing)} input row(s) produced no result")
    outcome.notes = notes

    artefacts: dict[str, str] = {}
    write_failures: list[str] = []

    def _attempt(name: str, writer) -> None:
        # One locked file (an open spreadsheet, a sync client) must not cost
        # the whole report; the remaining artefacts are still written.
        try:
            artefacts[name] = str(writer())
        except Exception as exc:
            write_failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if config.output.write_csv:
        _attempt(LEADS_CSV, lambda: write_csv(results, output_dir / LEADS_CSV))
        _attempt(BENCHMARK_CSV, lambda: write_csv(results, output_dir / BENCHMARK_CSV))
    if config.output.write_json:
        _attempt(RESULTS_JSON, lambda: write_json(results, output_dir / RESULTS_JSON))
        _attempt(BENCHMARK_JSON, lambda: write_json(results, output_dir / BENCHMARK_JSON))
    _attempt(
        SUMMARY_MD,
        lambda: write_summary(
            results, metrics, output_dir / SUMMARY_MD, inputs=items, notes=notes
        ),
    )
    if config.output.write_html_report:
        from leadfinder.benchmark.html_report import write_html_report

        _attempt(
            "report.html",
            lambda: write_html_report(
                results, metrics, output_dir / "report.html", inputs=items
            ),
        )
    if write_failures:
        notes.extend(f"artifact write failed: {failure}" for failure in write_failures)
        outcome.notes = notes
    artefacts["database"] = str(database.path)
    outcome.artefacts = artefacts
    return outcome


def _run_notes(config, database: Database) -> list[str]:
    notes: list[str] = []
    if not config.qwen.enabled:
        notes.append(
            "Qwen was NOT configured for this run, so entity resolution and grading "
            "fell back to deterministic extraction only. Grades are therefore a "
            "lower bound."
        )
    else:
        notes.append(
            f"Qwen model: {config.qwen.model} via {config.qwen.base_url} "
            f"(provider={config.qwen.provider})"
        )
    notes.append(
        f"Budget per company: {config.research.max_search_queries} searches, "
        f"{config.research.max_pages} pages, {config.research.max_llm_rounds} LLM rounds, "
        f"{config.research.workers} workers"
    )
    try:
        pages_cached = database.scalar("SELECT COUNT(*) FROM page_cache") or 0
        searches_cached = database.scalar("SELECT COUNT(*) FROM search_cache") or 0
        notes.append(f"Cache: {pages_cached} pages, {searches_cached} queries stored")
    except Exception:
        pass
    return notes


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
