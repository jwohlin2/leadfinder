"""Regressions for false positives found while running real companies.

Each test here corresponds to a real output that looked plausible but was
wrong: an app-store title read as an employee, an unrelated UK registry filing
adopted as the legal entity, a font URL parsed as a contact address, and page
chrome read as a job title.
"""

from leadfinder.crawl.extract import (
    extract_emails,
    extract_legal_names,
    normalize_email,
)
from leadfinder.models.company import CompanyIdentity, CompanyInput
from leadfinder.models.person import Person
from leadfinder.research.company_resolver import (
    _shares_distinctive_token,
    _worth_fetching,
)
from leadfinder.research.person_resolver import (
    _company_matches_full_name,
    _identity_names,
    _is_people_source,
    _snippet_title_ok,
    harvest_from_snippets,
    merge_person,
)

ITEM = CompanyInput(company_name="TravelBy Software", domain="availroom.com")


def test_generic_uk_company_is_not_the_legal_entity():
    # Searching the brand surfaces unrelated Companies House filings.
    assert not _shares_distinctive_token("A1 TAXI SERVICES LTD", ITEM)
    assert not _shares_distinctive_token("Taxi Services Ltd", ITEM)
    # The brand token or the domain stem is enough.
    assert _shares_distinctive_token("AvailRoom S.L", ITEM)
    assert _shares_distinctive_token("TravelBy Software S.L.", ITEM)


def test_app_store_listings_are_not_people_sources():
    assert not _is_people_source("https://apps.apple.com/us/app/travel-by-taxi/id6504042591")
    assert not _is_people_source("https://play.google.com/store/apps/details?id=com.availroom.app2")
    # A partner directory names hundreds of companies; its "entries" are not staff.
    assert not _is_people_source("https://developers.google.com/hotels/connectivity-partners")
    # A real team page is still a valid source.
    assert _is_people_source("https://www.availroom.com/equipo")
    assert _is_people_source("https://www.availroom.com/law/legal")


def test_url_collapsed_into_a_local_part_is_not_an_email():
    # A table of partner sites with normalised whitespace produced these.
    for junk in (
        "www.ireserv@ion.it",
        "www.rev@o.com",
        "www.asi@ech.in",
        "www.equ@or.travel",
    ):
        assert normalize_email(junk) == ""
    assert normalize_email("info@availroom.com") == "info@availroom.com"


def test_email_local_part_must_not_start_like_a_url():
    assert extract_emails("visit https://example.com/en/ now") == []
    assert extract_emails("write to Info@Availroom.com") == ["info@availroom.com"]


def test_legal_name_after_a_trailing_period_is_still_found():
    # "S.L.," has no word boundary after the final dot, which broke \b.
    names = extract_legal_names(
        "AvailRoom is a product of TravelBy Software S.L., registered in Spain."
    )
    assert any("TravelBy Software" in name for name in names)


def test_low_value_urls_do_not_consume_the_page_budget():
    assert not _worth_fetching("https://x.com/blog/page/2", ITEM)
    assert not _worth_fetching("https://x.com/search?q=availroom", ITEM)
    assert not _worth_fetching("https://play.google.com/store/apps/details?id=x&hl=ky", ITEM)
    assert _worth_fetching("https://www.availroom.com/legal", ITEM)
    assert _worth_fetching("https://fr.linkedin.com/company/availroom", ITEM)


# Cross-company people surfaced through press-release snippets must not be
# adopted as leads; the company's own decision-makers must survive.
WEBSREZ_TERMS = ["webrezpro", "world", "web", "technologies", "inc"]
PARTNER_SNIPPETS = [
    (
        "https://webrezpro.com/press-releases/webrezpro-integrates-duetto-revenue-strategy-solutions/",
        "Duetto Co - Chief Executive Officer of Duetto Inc Business Editor",
    ),
    (
        "https://webrezpro.com/press-releases/webrezpro-integrates-voila-hotel-rewards/",
        "Peter Gorla - Managing Director of VOILA Hotel Rewards",
    ),
]
OWN_SNIPPETS = [
    (
        "https://webrezpro.com/directory/frank-verhagen/",
        "Frank Verhagen - WebRezPro Team Founder & President 403-777-9300 ext 207 frank@18.236.170.203",
    ),
]
GARBAGE_SNIPPETS = [
    (
        "https://webrezpro.com/",
        "WebRezPro Owner Login Page Hotel ID: Own",
    ),
    (
        "https://webrezpro.com/",
        "Google Fonts",
    ),
    (
        "https://webrezpro.com/",
        "Spymaster Cis Coalition Businessesaroenet",
    ),
]


def test_partner_executives_from_press_releases_are_rejected():
    found = harvest_from_snippets(PARTNER_SNIPPETS, company_terms=WEBSREZ_TERMS)
    assert all(p.name != "Duetto Co" for p in found)
    assert all(p.name != "Peter Gorla" for p in found)


def test_own_company_directory_snippet_survives_title_chrome():
    found = harvest_from_snippets(OWN_SNIPPETS, company_terms=WEBSREZ_TERMS)
    assert any(p.name == "Frank Verhagen" for p in found)
    for p in found:
        if p.name == "Frank Verhagen":
            # The page-chrome prefix is tolerated; the title must be role-based.
            assert "Founder" in p.title or "founder" in p.title


def test_snippet_title_must_not_be_chrome_alone():
    # Directory chrome, UI labels and tech-stack product names are not people.
    for url, title in GARBAGE_SNIPPETS:
        found = harvest_from_snippets([(url, title)], company_terms=WEBSREZ_TERMS)
        assert found == [], f"{title!r} should not produce a person"


def test_snippet_allows_html_entities_in_title():
    # Brave returns raw JSON snippets with &amp; et al. The &amp; must not make
    # the title look like prose or break the role-word tail.
    url = "https://webrezpro.com/directory/frank-verhagen/"
    snippet = (
        "Frank Verhagen - WebRezPro - WebRezPro Team Frank Verhagen "
        "Founder &amp; President Direct: 403-777-9300 ext 207Toll Free: "
        "1-800-221-3429 ext 207 Email: frank@18.236.170.203 Get to know Frank"
    )
    found = harvest_from_snippets([(url, snippet)], company_terms=WEBSREZ_TERMS)
    assert [p.name for p in found] == ["Frank Verhagen"]
    p = found[0]
    assert p.evidence[0].source_url == url
    assert "&amp;" not in " ".join(e.source_text for e in p.evidence)


def test_clean_page_title_beats_snippet_chrome():
    # The fetched directory page yields "Founder & President"; the snippet
    # yields page chrome with the same role. The clean title must win in both
    # merge orders.
    page_person = Person(name="Frank Verhagen", title="Founder & President")
    snippet_person = Person(
        name="Frank Verhagen",
        title="WebRezPro - WebRezPro Team Frank Verhagen Founder & President "
        "Direct: 403-777-9300 ext 207",
    )
    assert merge_person(page_person, snippet_person).title == "Founder & President"
    assert merge_person(snippet_person, page_person).title == "Founder & President"


def test_lookalike_company_names_do_not_match_web_resolution():
    # "World Web Technologies" (WebRezPro's owner) and "World Wide
    # Technology" (wwt.com, an unrelated firm) share 'world'+'technologies'.
    # Raw token overlap accepted the wrong firm's executives; whole-name
    # similarity plus the distinctive brand term must not.
    ident = CompanyIdentity(
        canonical_name="WebRezPro",
        legal_name="World Web Technologies Inc",
        domain="webrezpro.com",
    )
    distinctive, full_names = _identity_names(
        ident, ["webrezpro", "world", "web", "technologies", "inc"]
    )
    assert _company_matches_full_name("World Web Technologies", distinctive, full_names)
    assert _company_matches_full_name("World Web Technologies Inc", distinctive, full_names)
    assert _company_matches_full_name("WebRezPro", distinctive, full_names)
    assert _company_matches_full_name("World Wide Technology", distinctive, full_names) is False
    assert _company_matches_full_name("WWT", distinctive, full_names) is False
    assert _company_matches_full_name("Duetto Inc", distinctive, full_names) is False
