"""Command line interface.

    python -m leadfinder doctor
    python -m leadfinder run data/input/companies.csv
    python -m leadfinder run data/input/companies.csv --limit 10
    python -m leadfinder company "Availroom" "availroom.com"
    python -m leadfinder report
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from leadfinder.benchmark.metrics import compute_metrics
from leadfinder.benchmark.report import (
    LEADS_CSV,
    RESULTS_JSON,
    SUMMARY_MD,
    write_csv,
    write_json,
    write_summary,
)
from leadfinder.benchmark.runner import run_benchmark
from leadfinder.config import DEFAULT_CONFIG_PATH, load_config
from leadfinder.llm.qwen import QwenClient
from leadfinder.models.company import CompanyInput, load_company_csv
from leadfinder.models.result import LeadResult
from leadfinder.storage.db import Database

GRADE_COLORS = {
    "A": "\033[92m", "B": "\033[92m", "C": "\033[94m", "D": "\033[94m",
    "E": "\033[93m", "FAIL": "\033[91m",
}
RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _grade(grade: str) -> str:
    if not _supports_color():
        return grade
    return f"{GRADE_COLORS.get(grade, '')}{grade}{RESET}"


def _print_result(result: LeadResult, index: int, total: int) -> None:
    company = result.company
    identity = company.legal_name or company.canonical_name or "unresolved"
    prefix = f"[{index}/{total}]"
    print(
        f"{prefix} {result.input_company} ({result.input_domain}) -> "
        f"{identity} | {company.country or '?'} | "
        f"lead: {result.best_lead.name or '-'}"
        f"{' (' + result.best_lead.title + ')' if result.best_lead.title else ''} | "
        f"{_grade(result.lead_grade)} | {result.status}"
    )
    if result.grade_reason:
        print(f"        why: {result.grade_reason[:180]}")


def _print_single(result: LeadResult) -> None:
    """The inspectable lead view the handoff asks for."""
    company = result.company
    print()
    print(f"Input company      : {result.input_company} ({result.input_domain})")
    print(f"Resolved company   : {company.canonical_name or '(unresolved)'}")
    if company.legal_name:
        print(f"Legal entity       : {company.legal_name}")
    if company.parent_company:
        print(f"Parent company     : {company.parent_company}")
    print(
        f"Country / language : {company.country or '?'} / {company.language or '?'}  "
        f"({company.product_or_company}, {company.status}, "
        f"confidence {company.confidence:.2f})"
    )
    if company.resolution_note:
        print(f"Resolution note    : {company.resolution_note}")
    print()

    best = result.best_lead
    if best.name:
        print(f"Best lead          : {best.name} - {best.title or '(title unknown)'}")
        print(f"  role / score     : {best.role_category} / {best.score}")
        print(f"  score components : {best.score_components}")
    else:
        print("Best lead          : (none found)")
    if best.reason:
        print(f"  why              : {best.reason}")
    print()

    contact = result.contact
    print("Contact:")
    if contact.direct_email:
        print(f"  direct email     : {contact.direct_email} "
              f"[{contact.email_method}, confidence {contact.email_confidence:.2f}]")
    if contact.general_email:
        print(f"  general email    : {contact.general_email}")
    if contact.contact_form:
        print(f"  contact form     : {contact.contact_form}")
    if contact.phone:
        print(f"  phone            : {contact.phone}")
    if contact.other:
        print(f"  other            : {contact.other}")
    if contact.wechat:
        print(f"  wechat           : {contact.wechat}")
    if not contact.has_path:
        print("  (no contact path found)")
    print()

    print(f"Grade              : {_grade(result.lead_grade)}  - {result.grade_reason}")
    print(f"Status             : {result.status}"
          + (f" ({result.failure_category})" if result.failure_category else ""))
    print()

    if result.people:
        print(f"People ({len(result.people)}):")
        for person in result.people[:6]:
            print(f"  - {person.name} | {person.title or '(unknown)'} | "
                  f"{person.role_category} | score {person.score.total} | "
                  f"confidence {person.confidence:.2f}")
        print()

    if result.sources:
        print("Sources:")
        for number, source in enumerate(result.sources[:15], start=1):
            print(f"  {number}. {source.title[:80] or source.url}  ({source.source_type})")
            print(f"     {source.url}")
        if len(result.sources) > 15:
            print(f"  ... {len(result.sources) - 15} more")
    else:
        print("Sources: (none)")

    quotes = [
        item
        for item in result.evidence
        if item.claim_type in {"person_title", "legal_name", "company_name", "direct_email"}
    ][:8]
    if quotes:
        print()
        print("Key evidence quotes:")
        for item in quotes:
            print(f"  [{item.claim_type}] {item.claim_value}  <- {item.source_url}")
            if item.source_text:
                print(f"      \"{item.source_text[:200]}\"")
    print()
    print(
        f"Research: {result.research.searches_performed} searches, "
        f"{result.research.pages_fetched} pages, {result.research.llm_calls} Qwen calls, "
        f"{result.research.cache_hits} cache hits, {result.research.duration_seconds}s"
        + (f"  [{result.research.stop_reason}]" if result.research.stop_reason else "")
    )


# ------------------------------------------------------------------ commands

def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.workers:
        config.research.workers = args.workers
    if args.max_queries is not None:
        config.research.max_search_queries = args.max_queries
    if args.max_pages is not None:
        config.research.max_pages = args.max_pages
    if args.max_rounds is not None:
        config.research.max_llm_rounds = args.max_rounds

    source = Path(args.input)
    if not source.exists():
        print(f"error: input file not found: {source}", file=sys.stderr)
        return 2

    items = load_company_csv(source)
    if not items:
        print(f"error: no company rows parsed from {source}", file=sys.stderr)
        return 2
    if args.limit:
        items = items[: args.limit]
    if args.offset:
        items = items[args.offset :]

    print(f"Loaded {len(items)} companies from {source}")
    if not config.qwen.enabled:
        print("WARNING: Qwen is not configured. Running deterministic-only;")
        print("         grades will be a lower bound. See: python -m leadfinder doctor")
    print()

    def progress(result: LeadResult, done: int, total: int) -> None:
        _print_result(result, done, total)

    outcome = asyncio.run(
        run_benchmark(items=items, config=config, on_progress=progress)
    )

    print()
    print("=" * 70)
    print(outcome.summary_line())
    print("=" * 70)
    for name, path in outcome.artefacts.items():
        print(f"  {name:20s} {path}")
    for note in outcome.notes:
        if note.startswith("artifact write failed"):
            print(f"  WARNING: {note}")
    if outcome.missing:
        print(f"  rows not accounted for: {len(outcome.missing)}")
    return 0


def cmd_company(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    item = CompanyInput(company_name=args.name, domain=args.domain.lower())
    from leadfinder.research.orchestrator import run_companies
    from leadfinder.storage.db import Database

    def progress(result: LeadResult) -> None:
        return None

    database = Database(config.resolve(config.storage.database))
    results = asyncio.run(
        run_companies(items=[item], config=config, db=database, on_result=progress)
    )
    if not results:
        print("error: no result produced", file=sys.stderr)
        return 1
    result = results[0]
    _print_single(result)

    if args.json:
        import json

        output = config.resolve(config.output.directory)
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{item.domain}.json"
        write_json([result], path)
        print(f"JSON written to {path}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ok = True

    print("leadfinder doctor")
    print("=" * 70)
    print(f"config file        : {args.config or DEFAULT_CONFIG_PATH}")
    print(f"repo root          : {config.root}")
    print()

    print("Qwen / LLM")
    client = QwenClient(config.qwen)
    if not config.qwen.enabled:
        print("  [FAIL] not configured")
        print("         set LEADFINDER_QWEN_BASE_URL and LEADFINDER_QWEN_MODEL")
        print("         in .env (see .env.example), or fill config.yaml -> qwen:")
        ok = False
    else:
        reachable, detail = client.health()
        print(f"  base_url          : {config.qwen.base_url}")
        print(f"  model             : {config.qwen.model}")
        print(f"  provider          : {config.qwen.provider}")
        print(f"  [{'OK  ' if reachable else 'FAIL'}] {detail}")
        if not reachable:
            print("         the run will continue without LLM reasoning (lower-bound grades)")
    client.close()
    print()

    print("Search")
    from leadfinder.research.search_factory import build_search_client
    from leadfinder.storage.cache import SearchCache

    db = Database(config.resolve(config.storage.database))

    async def probe_search() -> list:
        client = build_search_client(
            config, cache=SearchCache(db, config.storage.cache_ttl_hours)
        )
        try:
            await client.probe()
            # Close inside the same event loop that opened the connections.
            return [p.name for p in client.providers if p in client.active]
        finally:
            await client.close()

    try:
        active = asyncio.run(probe_search())
    except Exception as exc:
        active = []
        print(f"  [WARN] probe failed: {type(exc).__name__}: {exc}")
    for name in ("searxng", "duckduckgo", "bing"):
        if name in active:
            print(f"  [OK  ] {name}")
    if not active:
        print("  [WARN] no search provider reachable; results will be empty")
    else:
        print(f"  active provider: {active[0]}")
    print()

    print("Storage")
    print(f"  database          : {config.resolve(config.storage.database)}")
    print(f"  output directory  : {config.resolve(config.output.directory)}")
    print(f"  input list        : {config.resolve('data/input/companies.csv')}")
    print()

    print("Budget")
    print(f"  searches/company  : {config.research.max_search_queries}")
    print(f"  pages/company     : {config.research.max_pages}")
    print(f"  llm rounds/company: {config.research.max_llm_rounds}")
    print(f"  workers           : {config.research.workers}")
    print()
    print("ready" if ok else "NOT READY: fix the [FAIL] items above to enable LLM reasoning")
    return 0 if ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    source = Path(args.input) if args.input else config.resolve(
        config.output.directory
    ) / RESULTS_JSON
    if source.is_dir():
        source = source / RESULTS_JSON
    if not source.exists():
        print(f"error: {source} not found. Run a benchmark first.", file=sys.stderr)
        return 2

    import json

    from leadfinder.models.company import CompanyIdentity
    from leadfinder.models.result import BestLead, ContactInfo, ResearchStats

    payload = json.loads(source.read_text(encoding="utf-8"))
    results = [
        LeadResult(
            input_company=item["input_company"],
            input_domain=item["input_domain"],
            company=CompanyIdentity(**item["company"]),
            best_lead=BestLead(**item["best_lead"]),
            contact=ContactInfo(**item["contact"]),
            lead_grade=item["lead_grade"],
            grade_reason=item.get("grade_reason", ""),
            status=item["status"],
            aliases=item.get("aliases", []),
            related_entities=item.get("related_entities", []),
            failure_category=item.get("failure_category", ""),
            research=ResearchStats(**item["research"]),
        )
        for item in payload["results"]
    ]
    for result in results:
        print(f"{result.lead_grade:4s} {result.input_company:24s} {result.company.canonical_name}")

    output = config.resolve(config.output.directory)
    metrics = compute_metrics(results)
    summary = write_summary(results, metrics, output / SUMMARY_MD)
    html = None
    if config.output.write_html_report:
        from leadfinder.benchmark.html_report import write_html_report

        html = write_html_report(results, metrics, output / "report.html")
    print()
    print(f"summary : {summary}")
    if html:
        print(f"report  : {html}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m leadfinder",
        description="Zero-cost lead discovery for obscure companies",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the benchmark over a CSV of companies")
    run.add_argument("input", help="CSV with company_name,domain columns")
    run.add_argument("--limit", type=int, default=None, help="only run the first N rows")
    run.add_argument("--offset", type=int, default=0, help="skip the first N rows")
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--max-queries", type=int, default=None)
    run.add_argument("--max-pages", type=int, default=None)
    run.add_argument("--max-rounds", type=int, default=None)
    run.set_defaults(func=cmd_run)

    company = sub.add_parser("company", help="research a single company")
    company.add_argument("name")
    company.add_argument("domain")
    company.add_argument("--json", action="store_true", help="also write per-company JSON")
    company.set_defaults(func=cmd_company)

    doctor = sub.add_parser("doctor", help="check LLM, search and storage wiring")
    doctor.set_defaults(func=cmd_doctor)

    report = sub.add_parser("report", help="rebuild summary/report from results.json")
    report.add_argument("--input", default=None, help="path to results.json")
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
