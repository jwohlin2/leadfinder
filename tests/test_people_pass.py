"""Tests for the people-targeted pass.

Decision-makers live on /team, /about, /leadership and /directory pages. These
tests pin the URL classifier and the deterministic query builder the pass uses
to reach them before the adaptive loop spends its budget elsewhere.
"""

from leadfinder.config import Config
from leadfinder.crawl.crawler import Page
from leadfinder.crawl.extract import PageFacts
from leadfinder.models.company import CompanyIdentity, CompanyInput
from leadfinder.models.evidence import Evidence
from leadfinder.models.person import Person
from leadfinder.models.result import LeadResult
from leadfinder.research.budget import Budget
from leadfinder.research.orchestrator import ResearchWorker, WorkerContext
from leadfinder.research.query_planner import (
    is_people_page_url,
    people_search_queries,
)

ITEM = CompanyInput(company_name="WebRezPro", domain="webrezpro.com")
IDENT = CompanyIdentity(
    canonical_name="WebRezPro",
    legal_name="World Web Technologies Inc",
    domain="webrezpro.com",
)


def test_people_page_urls_are_recognised():
    assert is_people_page_url("https://webrezpro.com/directory/frank-verhagen/")
    assert is_people_page_url("https://webrezpro.com/team")
    assert is_people_page_url("https://availroom.com/about-us/")
    assert is_people_page_url("https://availroom.com/equipo")
    assert is_people_page_url("https://x.com/leadership-team.html")
    assert is_people_page_url("https://x.com/About_Us")


def test_non_people_pages_are_not_prioritised():
    assert not is_people_page_url("https://x.com/blog/post-1")
    assert not is_people_page_url("https://x.com/pricing")
    assert not is_people_page_url("https://x.com/products/team-sync")
    assert not is_people_page_url("")
    assert not is_people_page_url("https://x.com/contact")


def test_people_queries_are_site_scoped_and_deduped():
    queries = people_search_queries(IDENT, ITEM, ["webrezpro.com"])
    assert "site:webrezpro.com team" in queries
    assert "site:webrezpro.com leadership" in queries
    assert "site:webrezpro.com founders" in queries
    again = people_search_queries(
        IDENT, ITEM, ["webrezpro.com"], already_run=["site:webrezpro.com team"]
    )
    assert "site:webrezpro.com team" not in again


def test_people_queries_use_native_vocabulary():
    ident = CompanyIdentity(
        canonical_name="株式会社チャットボット",
        domain="chatbot.co.jp",
        country="Japan",
        language="Japanese",
    )
    item = CompanyInput(company_name="株式会社チャットボット", domain="chatbot.co.jp")
    queries = people_search_queries(ident, item, ["chatbot.co.jp"])
    assert any("代表取締役" in query for query in queries)


def test_partial_finalize_keeps_people_found_before_timeout():
    # A wall-clock cancellation must not throw away a decision-maker that was
    # already evidenced: the company is graded from the partial state instead
    # of being reported as a technical failure.
    config = Config()
    worker = ResearchWorker(config=config, db=None, search=None, llm=object())
    item = CompanyInput(company_name="Zenya", domain="zenya.com")
    ctx = WorkerContext(
        item=item,
        config=config,
        db=None,
        search=None,
        crawler=None,
        llm=object(),
        budget=Budget(),
    )
    ctx.identity = CompanyIdentity(
        canonical_name="Zenya", domain="zenya.com", confidence=0.9
    )
    ctx.known_urls.add("https://zenya.com/team")
    contact_page = Page(
        url="https://zenya.com/contact",
        status_code=200,
        title="Contact",
        text="Reach us at info@zenya.com",
    )
    contact_page.facts = PageFacts(emails=["info@zenya.com"])
    ctx.pages = [contact_page]
    person = Person(name="Jesal Sangani", title="Founder")
    person.evidence.append(
        Evidence(
            claim_type="person_name",
            claim_value="Jesal Sangani",
            source_url="https://zenya.com/team",
            source_title="Team",
            source_text="Jesal Sangani Founder",
            confidence=0.7,
        )
    )
    ctx.people = [person]

    result = LeadResult(input_company="Zenya", input_domain="zenya.com")
    worker._finalize_partial(ctx, result)

    assert result.best_lead.name == "Jesal Sangani"
    assert result.people and result.people[0].name == "Jesal Sangani"
    assert result.lead_grade in {"A", "B", "C", "D"}
    assert "wall-clock" in result.grade_reason
