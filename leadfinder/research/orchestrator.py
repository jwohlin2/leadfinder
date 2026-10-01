"""The research worker: one isolated pipeline per company.

Flow:
  Stage 1  crawl the supplied domain
  Stage 2  deterministic searches
  Stage 3  Qwen entity resolution
  Stage 4  adaptive loop: plan -> search -> fetch -> reconcile -> re-grade
  final    Qwen synthesis, deterministic grading, persistence

Stops as soon as identity + decision maker + contact path are in hand. Budget
left over is not a reason to keep searching.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field

from leadfinder.crawl.crawler import Crawler, Page
from leadfinder.llm.qwen import QwenClient
from leadfinder.models.company import CompanyIdentity, CompanyInput
from leadfinder.models.evidence import Evidence, Source, utcnow
from leadfinder.models.person import Person, infer_size_hint
from leadfinder.models.result import LeadResult, ResearchStats
from leadfinder.research import company_resolver as cr
from leadfinder.research import person_resolver as pr
from leadfinder.research import query_planner as qp
from leadfinder.research import ranker
from leadfinder.research.budget import Budget
from leadfinder.research.contact_resolver import contacts_evidence, resolve_contacts
from leadfinder.search.base import SearchResult, classify_source_type
from leadfinder.search.free_web import SearchClient
from leadfinder.storage.cache import PageCache, SearchCache
from leadfinder.storage.db import Database


@dataclass(slots=True)
class WorkerContext:
    """Per-company isolated state. Nothing here is shared between companies."""

    item: CompanyInput
    config: object
    db: Database
    search: SearchClient
    crawler: Crawler
    llm: QwenClient
    budget: Budget
    pages: list[Page] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    snippets: list[tuple[str, str]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    people: list[Person] = field(default_factory=list)
    identity: CompanyIdentity = field(default_factory=CompanyIdentity)
    known_urls: set[str] = field(default_factory=set)
    run_id: int | None = None
    company_id: int | None = None
    query_rows: list[tuple[int, str, int, int]] = field(default_factory=list)

    def absorb(self, pages: list[Page], sources: list[Source]) -> None:
        existing = {p.final_url or p.url for p in self.pages}
        for page in pages:
            url = page.final_url or page.url
            if page.ok and url not in existing:
                self.pages.append(page)
                existing.add(url)
                self.known_urls.add(url)
        for source in sources:
            if source.url not in {s.url for s in self.sources}:
                self.sources.append(source)


class ResearchWorker:
    def __init__(
        self,
        *,
        config,
        db: Database,
        search: SearchClient,
        llm: QwenClient | None = None,
    ) -> None:
        self.config = config
        self.db = db
        # One search client for the whole batch: a single connection pool and a
        # single event loop, closed once at the end of the run.
        self.search = search
        self.llm = llm or QwenClient(config.qwen)

    async def run_one(self, item: CompanyInput) -> LeadResult:
        started = time.monotonic()
        result = LeadResult(
            input_company=item.company_name,
            input_domain=item.domain,
        )
        page_cache = PageCache(self.db, self.config.storage.cache_ttl_hours)
        crawler = Crawler(self.config, cache=page_cache)
        search = self.search

        budget = Budget(
            max_search_queries=self.config.research.max_search_queries,
            max_pages=self.config.research.max_pages,
            max_llm_rounds=self.config.research.max_llm_rounds,
            timeout=self.config.research.company_timeout,
        )
        ctx = WorkerContext(
            item=item,
            config=self.config,
            db=self.db,
            search=search,
            crawler=crawler,
            llm=self.llm,
            budget=budget,
        )

        try:
            await self._execute(ctx, result)
        except asyncio.CancelledError:
            # Wall-clock cancellation: grade what is already in hand instead
            # of losing a company that had already found its people.
            ctx.budget.note_stop("wall-clock limit")
            self._finalize_partial(ctx, result)
        except Exception as exc:  # never let one company kill the batch
            result.status = "ERROR"
            result.lead_grade = "FAIL"
            result.grade_reason = f"unhandled error: {type(exc).__name__}: {exc}"[:300]
            result.research.error = traceback.format_exc()[-1200:]
            result.failure_category = "technical failure"
        finally:
            await crawler.close()
            result.research.duration_seconds = round(time.monotonic() - started, 2)
            result.research.queries = list(budget.queries)
            result.research.searches_performed = search.stats["searches"]
            result.research.pages_fetched = crawler.stats["fetched"]
            result.research.cache_hits = crawler.stats["cache_hits"] + search.stats["cache_hits"]
            result.research.llm_calls = self.llm.calls
            result.research.llm_failures = self.llm.failures
            result.research.rounds = budget.llm_rounds
            result.research.stop_reason = budget.stop_reason
            result.sources = ctx.sources[:40]
            result.evidence = ctx.evidence[:400]
            try:
                self._persist(ctx, result)
            except Exception:
                # Persistence must never fail a company, but a silent failure
                # means the benchmark reports grades with no stored evidence.
                result.research.persist_error = traceback.format_exc()[-600:]

        return result

    # -- pipeline --------------------------------------------------------
    async def _execute(self, ctx: WorkerContext, result: LeadResult) -> None:
        item = ctx.item

        report = await cr.resolve_entity(
            item=item,
            crawler=ctx.crawler,
            search=ctx.search,
            llm=ctx.llm,
            budget=ctx.budget,
        )
        ctx.absorb(report.pages, report.sources)
        ctx.identity = report.identity
        ctx.evidence.extend(report.evidence)
        ctx.snippets.extend(report.snippets)
        if report.llm_note:
            result.research.error = report.llm_note

        if not report.domain_ok:
            result.status = "FAILED_DOMAIN"
            result.lead_grade = "FAIL"
            result.failure_category = "wrong/invalid domain"
            result.grade_reason = (
                f"supplied domain {item.domain} did not return a usable page"
                + (f" ({report.domain_error})" if report.domain_error else "")
            )
            return

        domains = self._company_domains(item, ctx.identity)
        terms = cr.query_terms(item, ctx.identity)

        # --- Stage 3.5: people-targeted pass ------------------------------
        # Decision-makers live on the company's own /team, /about and
        # /leadership pages, which generic role queries rarely surface. Run a
        # bounded, deterministic set of site-scoped queries and fetch the
        # people pages they return before the adaptive loop can spend its
        # budget on contact/identity gaps. Two queries are kept in reserve for
        # the loop so contact discovery still has room.
        people_queries = qp.people_search_queries(
            ctx.identity, item, domains, already_run=ctx.budget.queries
        )
        # Fetch people pages the resolver's searches already surfaced: those
        # URLs are known and cost no query budget. Then run the remaining
        # people queries for snippets only - their pages are fetched only if
        # they look like team/about pages.
        await self._fetch_people_pages(ctx, domains)
        allowance = max(0, min(len(people_queries), ctx.budget.remaining_queries - 2))
        if allowance:
            await self._run_queries(
                ctx, people_queries[:allowance], terms, fetch=False
            )
            await self._fetch_people_pages(ctx, domains)
        self._collect_people(ctx, terms, domains)

        # --- Stage 4: adaptive loop -------------------------------------
        while True:
            contact = self._resolve_contacts(ctx)
            done, reason = qp.success_reached(
                identity=ctx.identity,
                people=ctx.people,
                contact_ok=contact.has_path,
                config=self.config,
            )
            if done and self.config.research.early_stop:
                ctx.budget.note_stop(reason)
                break
            if ctx.budget.exhausted():
                ctx.budget.note_stop("budget exhausted")
                break
            if not ctx.budget.spend_round():
                ctx.budget.note_stop("llm round budget exhausted")
                break

            plan = qp.plan_round(
                llm=ctx.llm,
                item=item,
                identity=ctx.identity,
                pages=ctx.pages,
                people=ctx.people,
                contact_ok=contact.has_path,
                already_run=ctx.budget.queries,
            )
            if not plan.queries:
                ctx.budget.note_stop("planner produced no new queries")
                break
            if plan.stop:
                ctx.budget.note_stop(plan.reason or "planner requested stop")
                break

            await self._run_queries(ctx, plan.queries, terms)
            await self._fetch_people_pages(ctx, domains)
            self._collect_people(ctx, terms, domains)

        # --- final grading ----------------------------------------------
        contact = self._resolve_contacts(ctx)
        self._finalize_people(ctx, domains, contact)
        contact = self._resolve_contacts(ctx, person_names=[p.name for p in ctx.people])
        self._finalize_people(ctx, domains, contact)

        best = ranker.choose_best(ctx.people, llm_reason="")
        outcome = ranker.grade(
            identity=ctx.identity,
            people=ctx.people,
            contact=contact,
            domain_ok=report.domain_ok,
            llm=ctx.llm,
        )
        result.company = ctx.identity
        result.people = ctx.people[:8]
        result.best_lead = best
        result.contact = contact
        result.lead_grade = outcome.grade
        result.grade_reason = outcome.reason
        result.status = outcome.status
        result.failure_category = outcome.failure_category
        result.aliases = ctx.identity.aliases[:8]
        result.related_entities = self._related_entities(ctx.identity, item)

    async def _run_queries(
        self,
        ctx: WorkerContext,
        queries: list[str],
        terms: list[str],
        fetch: bool = True,
    ) -> None:
        for query in queries:
            if ctx.budget.remaining_queries <= 0 or ctx.budget.timed_out:
                break
            if not ctx.budget.spend_query(query):
                break
            results = await ctx.search.search(query, terms=terms)
            results = [r for r in results if r.url]
            ctx.query_rows.append((len(ctx.budget.queries), query, 0, len(results)))
            if not results:
                continue
            # Snippets are free (the search already returned them) and are a
            # people source in their own right. The people pass reads every
            # result; other rounds read only what they may fetch.
            snippet_slice = results if not fetch else results[
                : max(1, self.config.search.fetch_top_n)
            ]
            for result in snippet_slice:
                ctx.snippets.append(
                    (result.url, f"{result.title} - {result.snippet}".strip(" -"))
                )
            if not fetch:
                continue
            candidates = results[: max(1, self.config.search.fetch_top_n)]
            for result in candidates:
                if ctx.budget.remaining_pages <= 1 or ctx.budget.timed_out:
                    break
                url = result.url
                if url in ctx.known_urls:
                    continue
                page = await ctx.crawler.fetch(url)
                ctx.budget.spend_pages(1)
                if not page.ok:
                    continue
                ctx.absorb(
                    [page],
                    [
                        Source(
                            url=page.final_url or page.url,
                            title=page.title,
                            source_type=classify_source_type(page.final_url or page.url,
                                                              page.title),
                            retrieved_at=page.retrieved_at,
                        )
                    ],
                )
                ctx.snippets.append((page.final_url or page.url, f"{page.title} - {page.text[:200]}"))

    async def _fetch_people_pages(self, ctx: WorkerContext, domains: list[str]) -> None:
        """Fetch team/about pages already visible in snippets, people first.

        The deterministic searches surface /team, /about and /directory pages
        whose snippets never make it into the fetch set (they sit below the
        top-N cut, or the resolver spends its offsite budget on identity
        pages). This spends the remaining page budget on those URLs directly,
        on the company's own domain only.
        """
        candidates: list[str] = []
        for url, _ in ctx.snippets:
            if not url or url in ctx.known_urls or url in candidates:
                continue
            if not qp.is_people_page_url(url):
                continue
            if not any(cr._same_registrable_domain(url, d) for d in domains):
                continue
            if not pr._is_people_source(url):
                continue
            if not cr._worth_fetching(url, ctx.item):
                continue
            candidates.append(url)
        for url in candidates:
            if ctx.budget.remaining_pages <= 1 or ctx.budget.timed_out:
                break
            page = await ctx.crawler.fetch(url)
            ctx.budget.spend_pages(1)
            if not page.ok:
                continue
            ctx.absorb(
                [page],
                [
                    Source(
                        url=page.final_url or page.url,
                        title=page.title,
                        source_type=classify_source_type(
                            page.final_url or page.url, page.title
                        ),
                        retrieved_at=page.retrieved_at,
                    )
                ],
            )
            ctx.snippets.append(
                (page.final_url or page.url, f"{page.title} - {page.text[:200]}")
            )

    def _company_domains(self, item: CompanyInput, identity: CompanyIdentity) -> list[str]:
        domains = [item.domain.lower().removeprefix("www.")]
        if identity.domain:
            resolved = identity.domain.lower().removeprefix("www.")
            if resolved and resolved not in domains:
                domains.append(resolved)
        return [d for d in domains if d]

    def _collect_people(
        self, ctx: WorkerContext, terms: list[str], domains: list[str]
    ) -> None:
        """Gather candidates from pages and snippets, then merge into ctx.people."""
        found = pr.harvest_from_pages(ctx.pages, company_domains=domains, company_terms=terms)
        found += pr.harvest_from_snippets(ctx.snippets, company_terms=terms)
        if not found:
            return

        merged: dict[str, Person] = {pr.person_name_key(p.name): p for p in ctx.people}
        for person in found:
            key = pr.person_name_key(person.name)
            if not key:
                continue
            existing = merged.get(key)
            if existing is not None:
                pr.merge_person(existing, person)
            else:
                merged[key] = person
        ctx.people = list(merged.values())
        self._rank_people(ctx, terms, domains)

    def _finalize_people(
        self, ctx: WorkerContext, domains: list[str], contact
    ) -> None:
        terms = cr.query_terms(ctx.item, ctx.identity)
        allowed = set(ctx.known_urls)
        allowed |= {url for url, _ in ctx.snippets}
        kept, _dropped = pr._filter_supported(ctx.people, allowed)
        if kept:
            ctx.people = kept

        bundle = pr.reconcile_with_llm(
            ctx.llm,
            input_company=ctx.item.company_name,
            resolved_name=ctx.identity.canonical_name,
            legal_name=ctx.identity.legal_name,
            country=ctx.identity.country,
            size_hint=ctx.identity.size_hint or infer_size_hint(
                " ".join(p.text[:800] for p in ctx.pages)
            ),
            people=ctx.people,
            char_budget=self.config.qwen.max_evidence_chars,
        )
        if bundle.llm_persons:
            verified = pr.apply_llm_verification(
                ctx.people,
                bundle.llm_persons,
                allowed_urls=allowed,
                company_terms=terms,
                identity=ctx.identity,
            )
            if verified:
                ctx.people = verified
            if bundle.best_person:
                for person in ctx.people:
                    if pr.person_name_key(person.name) == pr.person_name_key(bundle.best_person):
                        person.notes = bundle.best_reason
                        break
        self._rank_people(ctx, terms, domains)

    def _rank_people(self, ctx: WorkerContext, terms: list[str], domains: list[str]) -> None:
        size_hint = ctx.identity.size_hint or infer_size_hint(
            " ".join(p.text[:800] for p in ctx.pages)
        )
        if size_hint:
            ctx.identity.size_hint = size_hint
        ctx.people = pr.rank_people(
            ctx.people,
            company_terms=terms or [ctx.item.domain],
            company_domains=domains,
            size_hint=size_hint,
        )

    def _finalize_partial(self, ctx: WorkerContext, result: LeadResult) -> None:
        """Grade the state gathered before the wall clock ran out.

        No LLM and no network here: the task is already being cancelled, and a
        company that found its decision-maker should not be reported as a
        technical failure just because the contact search ran long.
        """
        terms = cr.query_terms(ctx.item, ctx.identity)
        domains = self._company_domains(ctx.item, ctx.identity)
        contact = self._resolve_contacts(ctx)
        allowed = set(ctx.known_urls) | {url for url, _ in ctx.snippets}
        kept, _dropped = pr._filter_supported(ctx.people, allowed)
        if kept:
            ctx.people = kept
        self._rank_people(ctx, terms, domains)

        result.company = ctx.identity
        result.people = ctx.people[:8]
        result.best_lead = ranker.choose_best(ctx.people, llm_reason="")
        outcome = ranker._deterministic_grade(
            identity=ctx.identity,
            people=ctx.people,
            contact=contact,
            domain_ok=bool(ctx.identity.canonical_name),
        )
        result.contact = contact
        result.lead_grade = outcome.grade
        result.grade_reason = (
            f"{outcome.reason} (wall-clock limit; graded from partial evidence)"
        )
        result.status = outcome.status
        result.failure_category = outcome.failure_category
        result.aliases = ctx.identity.aliases[:8]

    def _resolve_contacts(self, ctx: WorkerContext, person_names: list[str] | None = None):
        domains = self._company_domains(ctx.item, ctx.identity)
        names = person_names if person_names is not None else [p.name for p in ctx.people]
        report = resolve_contacts(
            ctx.pages,
            company_domains=domains,
            person_names=names,
            best_person=ctx.people[0].name if ctx.people else "",
        )
        return report.info

    def _related_entities(
        self, identity: CompanyIdentity, item: CompanyInput
    ) -> list[dict[str, str]]:
        """Brand <-> legal-company links so later runs can reuse the record."""
        out: list[dict[str, str]] = []
        if identity.legal_name and identity.legal_name != identity.canonical_name:
            out.append(
                {
                    "name": identity.legal_name,
                    "domain": identity.domain,
                    "relation": "legal_entity_of",
                }
            )
        if identity.parent_company:
            out.append({"name": identity.parent_company, "domain": "", "relation": "parent_of"})
        for alias in identity.aliases[:4]:
            if alias and alias.lower() not in {item.company_name.lower()}:
                out.append({"name": alias, "domain": "", "relation": "alias_of"})
        return out

    # -- persistence -----------------------------------------------------
    def _persist(self, ctx: WorkerContext, result: LeadResult) -> None:
        db = ctx.db
        now = utcnow()

        company_id = db.scalar("SELECT id FROM companies WHERE domain = ?", (result.input_domain,))
        if company_id is None:
            company_id = db.insert(
                "INSERT INTO companies (domain, input_company, canonical_name, legal_name, "
                "country, language, company_type, product_or_company, parent_company, status, "
                "industry, size_hint, confidence, aliases_json, resolution_note, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.input_domain,
                    result.input_company,
                    result.company.canonical_name,
                    result.company.legal_name,
                    result.company.country,
                    result.company.language,
                    result.company.company_type,
                    result.company.product_or_company,
                    result.company.parent_company,
                    result.company.status,
                    result.company.industry,
                    result.company.size_hint,
                    result.company.confidence,
                    __import__("json").dumps(result.company.aliases, ensure_ascii=False),
                    result.company.resolution_note,
                    now,
                    now,
                ),
            )
        else:
            db.execute(
                "UPDATE companies SET canonical_name=?, legal_name=?, country=?, language=?, "
                "company_type=?, product_or_company=?, parent_company=?, status=?, industry=?, "
                "size_hint=?, confidence=?, aliases_json=?, resolution_note=?, updated_at=? "
                "WHERE id=?",
                (
                    result.company.canonical_name,
                    result.company.legal_name,
                    result.company.country,
                    result.company.language,
                    result.company.company_type,
                    result.company.product_or_company,
                    result.company.parent_company,
                    result.company.status,
                    result.company.industry,
                    result.company.size_hint,
                    result.company.confidence,
                    __import__("json").dumps(result.company.aliases, ensure_ascii=False),
                    result.company.resolution_note,
                    now,
                    company_id,
                ),
            )
        ctx.company_id = company_id

        run_id = db.insert(
            "INSERT INTO research_runs (company_id, input_company, input_domain, status, "
            "lead_grade, failure_category, searches, pages, llm_calls, cache_hits, duration, "
            "stop_reason, error, started_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                company_id,
                result.input_company,
                result.input_domain,
                result.status,
                result.lead_grade,
                result.failure_category,
                result.research.searches_performed,
                result.research.pages_fetched,
                result.research.llm_calls,
                result.research.cache_hits,
                result.research.duration_seconds,
                result.research.stop_reason,
                result.research.error,
                now,
                now,
            ),
        )
        ctx.run_id = run_id

        for offset, (position, query, _engine, count) in enumerate(ctx.query_rows, start=1):
            db.execute(
                "INSERT INTO search_queries (run_id, provider, query, round, result_count, "
                "created_at) VALUES (?,?,?,?,?,?)",
                (run_id, ctx.search.provider_name, query, position, count, now),
            )

        for page in ctx.pages[:60]:
            db.execute(
                "INSERT OR IGNORE INTO pages (url, final_url, status_code, title, text, html, "
                "source_type, via, error, retrieved_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    page.url,
                    page.final_url,
                    page.status_code,
                    page.title[:500],
                    page.text[:60000],
                    page.html[:200000],
                    page.source_type,
                    page.via,
                    page.error,
                    page.retrieved_at,
                ),
            )

        evidence_ids: dict[int, int] = {}
        for item in ctx.evidence:
            evidence_id = db.insert(
                "INSERT INTO evidence (company_id, person_id, claim_type, claim_value, "
                "source_url, source_title, source_text, retrieved_at, confidence) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    company_id,
                    None,
                    item.claim_type,
                    item.claim_value[:500],
                    item.source_url,
                    item.source_title[:500],
                    item.source_text[:2000],
                    item.retrieved_at,
                    item.confidence,
                ),
            )
            evidence_ids[id(item)] = evidence_id

        for person in result.people:
            person_id = db.insert(
                "INSERT OR IGNORE INTO people (company_id, name, normalized_name, name_key, "
                "title, role_category, current_company, linkedin, location, confidence, "
                "score_total, score_components, is_current, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    company_id,
                    person.name,
                    pr.normalize_person_name(person.name),
                    pr.person_name_key(person.name),
                    person.title,
                    person.role_category,
                    person.current_company,
                    person.linkedin,
                    person.location,
                    person.confidence,
                    person.score.total,
                    __import__("json").dumps(person.score.as_dict()),
                    1 if person.is_current else 0,
                    person.notes,
                ),
            )
            if not person_id:
                person_id = db.scalar(
                    "SELECT id FROM people WHERE company_id=? AND name_key=?",
                    (company_id, pr.person_name_key(person.name)),
                )
            person.person_id = person_id
            for evidence_item in person.evidence:
                eid = db.insert(
                    "INSERT INTO evidence (company_id, person_id, claim_type, claim_value, "
                    "source_url, source_title, source_text, retrieved_at, confidence) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        company_id,
                        person_id,
                        evidence_item.claim_type,
                        evidence_item.claim_value[:500],
                        evidence_item.source_url,
                        evidence_item.source_title[:500],
                        evidence_item.source_text[:2000],
                        evidence_item.retrieved_at,
                        evidence_item.confidence,
                    ),
                )
                db.execute(
                    "INSERT OR IGNORE INTO person_evidence (person_id, evidence_id) VALUES (?,?)",
                    (person_id, eid),
                )

        contact = result.contact
        for kind, value, confidence in (
            ("direct_email", contact.direct_email, contact.email_confidence),
            ("general_email", contact.general_email, 0.5),
            ("contact_form", contact.contact_form, 0.6),
            ("phone", contact.phone, 0.4),
            ("other", contact.other, 0.4),
            ("wechat", contact.wechat, 0.4),
        ):
            if not value:
                continue
            db.execute(
                "INSERT OR IGNORE INTO contacts (company_id, person_id, kind, value, method, "
                "confidence, source_url) VALUES (?,?,?,?,?,?,?)",
                (
                    company_id,
                    result.best_lead.person_id,
                    kind,
                    value[:500],
                    contact.email_method if kind == "direct_email" else "published",
                    confidence,
                    (contact.evidence_urls or [""])[0],
                ),
            )

        for entity in result.related_entities:
            db.execute(
                "INSERT OR IGNORE INTO related_entities (company_id, name, domain, relation) "
                "VALUES (?,?,?,?)",
                (company_id, entity.get("name", ""), entity.get("domain", ""),
                 entity.get("relation", "")),
            )


async def run_companies(
    *,
    items: list[CompanyInput],
    config,
    db: Database,
    workers: int | None = None,
    on_result=None,
) -> list[LeadResult]:
    """Run a batch with bounded concurrency. Every input gets a result."""
    from leadfinder.research.search_factory import build_search_client

    worker_count = max(1, workers or config.research.workers)
    search_cache = SearchCache(db, config.storage.cache_ttl_hours)
    llm = QwenClient(config.qwen)
    search_client = build_search_client(config, cache=search_cache)
    worker = ResearchWorker(
        config=config, db=db, search=search_client, llm=llm
    )
    semaphore = asyncio.Semaphore(worker_count)

    async def guarded(item: CompanyInput) -> LeadResult:
        async with semaphore:
            task = asyncio.create_task(worker.run_one(item))
            timed_out = False
            try:
                done, _pending = await asyncio.wait(
                    {task}, timeout=config.research.company_timeout + 90
                )
                if not done:
                    # run_one suppresses the cancellation, finalises the
                    # partial state and returns it, so a company that already
                    # found its people is not thrown away.
                    timed_out = True
                    task.cancel()
                result = await task
            except asyncio.CancelledError:
                if timed_out and task.cancelled():
                    result = LeadResult(
                        input_company=item.company_name, input_domain=item.domain
                    )
                    result.status = "ERROR"
                    result.lead_grade = "FAIL"
                    result.failure_category = "technical failure"
                    result.grade_reason = "company exceeded the wall-clock limit"
                    result.research.error = "wall-clock cancellation during cleanup"
                else:
                    raise
            except Exception as exc:
                result = LeadResult(input_company=item.company_name, input_domain=item.domain)
                result.status = "ERROR"
                result.lead_grade = "FAIL"
                result.failure_category = "technical failure"
                result.grade_reason = f"{type(exc).__name__}: {exc}"[:300]
            if on_result is not None:
                on_result(result)
            return result

    try:
        results = await asyncio.gather(*(guarded(item) for item in items))
    finally:
        llm.close()
        await search_client.close()
    return list(results)
