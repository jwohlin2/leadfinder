from leadfinder.crawl.extract import (
    classify_email,
    extract_emails_from_html,
    extract_facts,
    extract_legal_names,
    extract_people,
    find_contact_forms,
    normalize_email,
)

TEAM_HTML = """
<html><body>
<table>
  <tr><td>Jane Smith</td><td>Founder &amp; CEO</td></tr>
  <tr><td>Marco Rossi</td><td>Head of Partnerships</td></tr>
</table>
<h3>Kenji Tanaka</h3><p>Managing Director</p>
<h3>Channel Manager</h3><p>All your listings synced centrally</p>
<h3>Property Management System</h3><p>The complete platform</p>
<p>Elena Duarte, COO</p>
<p>Reach us: Founder: Ana Ruiz</p>
<div>Vice President - David Okafor</div>
</body></html>
"""

PEOPLE = extract_people(TEAM_HTML, "")


def test_extracts_real_people():
    names = {p.name for p in PEOPLE}
    assert "Jane Smith" in names
    assert "Marco Rossi" in names
    assert "Kenji Tanaka" in names
    assert "Elena Duarte" in names
    assert "David Okafor" in names


def test_rejects_product_names_as_people():
    names = {p.name for p in PEOPLE}
    assert "Channel Manager" not in names
    assert "Property Management System" not in names


def test_titles_are_job_titles_not_prose():
    by_name = {p.name: p.title for p in PEOPLE}
    assert by_name["Jane Smith"] == "Founder & CEO"
    assert by_name["Marco Rossi"] == "Head of Partnerships"
    assert by_name["Kenji Tanaka"] == "Managing Director"
    for title in by_name.values():
        assert len(title) <= 60


HTML_WITH_CSS = """
<html><body>
<style>@font-face{src:url(fonts.gst@ic.com/x.woff2)}</style>
<a href="mailto:info@availroom.com">Write to us</a>
<footer class="site-footer"><p>AvailRoom, info@availroom.com, +34 919 03 01 00</p></footer>
<form id="cookie-preferences" class="fs-cc-prefs_form">
  <input type="checkbox" name="analytics">
</form>
<form class="newsletter-signup">
  <input name="email" type="email"><button>Subscribe</button>
</form>
<form class="contact-form" action="/contact">
  <input name="your name"><input name="email" type="email">
  <textarea name="your message"></textarea>
  <button>Send</button>
</form>
</body></html>
"""


def test_emails_come_from_contacts_not_stylesheets():
    emails = extract_emails_from_html(HTML_WITH_CSS)
    assert "info@availroom.com" in emails
    assert "fonts.gst@ic.com" not in emails


def test_cookie_and_newsletter_forms_are_not_contact_forms():
    forms = find_contact_forms(HTML_WITH_CSS, "https://example.com/")
    assert forms == ["https://example.com/contact"]


def test_normalize_email_rejects_junk():
    assert normalize_email("info@example.com") == ""
    assert normalize_email("a..b@x.com") == ""
    assert normalize_email("@x.com") == ""
    assert normalize_email("no-at-sign") == ""
    assert normalize_email("Support@Company.com") == "support@company.com"


def test_classify_email_splits_direct_and_general():
    assert classify_email("jane.smith@acme.com", ["Jane Smith"])[0] == "direct"
    assert classify_email("info@acme.com", ["Jane Smith"])[0] == "general"
    assert classify_email("jane@gmail.com", ["Jane Smith"])[0] == "direct"


LEGAL_HTML = """
<p>AvailRoom is a product of TravelBy Software S.L., registered in Spain.</p>
<p>株式会社チャットボットは東京に本社を置く</p>
<p>© 2024 TravelBy Software S.L.</p>
"""


def test_legal_names_include_latin_and_japanese():
    names = extract_legal_names(LEGAL_HTML)
    assert any("Software" in n for n in names)
    assert any("株式会社" in n for n in names)


def test_extract_facts_bundles_everything():
    facts = extract_facts("https://x.com/", HTML_WITH_CSS, "Reach us at info@availroom.com")
    assert "info@availroom.com" in facts.emails
    assert facts.phones
    assert facts.contact_forms == ["https://x.com/contact"]


DIRECTORY_TEXT = """
WebRezPro Team
Frank Verhagen
Founder & President
Direct: 403-777-9300 ext 207
Toll Free: 1-800-221-3429 ext 207
Sarah Duguay
Director of Marketing
"""


def test_adjacent_lines_name_then_title():
    # Div-grid team pages extract as a name line above a title line; the
    # same-line rules cannot see this shape.
    by_name = {p.name: p.title for p in extract_people("", DIRECTORY_TEXT)}
    assert by_name.get("Frank Verhagen") == "Founder & President"
    assert by_name.get("Sarah Duguay") == "Director of Marketing"


def test_adjacent_nav_labels_are_not_people():
    assert extract_people("", "About Us\nOur Team\nContact\nBook Now") == []
