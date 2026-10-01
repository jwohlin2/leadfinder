"""Minimal static HTML report.

Secondary per the handoff, but cheap: a single self-contained file that makes
the overall metrics and each lead's evidence easy to inspect.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from leadfinder.benchmark.metrics import Metrics
from leadfinder.models.result import LeadResult

_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "FAIL": 5}

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0; padding: 24px; background: #0f1115; color: #e6e8ee; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 12px; }
.sub { color: #98a2b3; margin-bottom: 20px; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 24px; }
.card { background: #171a21; border: 1px solid #262b36; border-radius: 8px;
        padding: 12px 16px; min-width: 130px; }
.card .k { color: #98a2b3; font-size: 11px; text-transform: uppercase;
           letter-spacing: .04em; }
.card .v { font-size: 22px; font-weight: 600; margin-top: 2px; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #232833;
         vertical-align: top; }
th { color: #98a2b3; font-weight: 600; font-size: 11px; text-transform: uppercase;
     letter-spacing: .04em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.grade { display: inline-block; min-width: 20px; text-align: center; border-radius: 4px;
         padding: 1px 6px; font-weight: 700; font-size: 12px; }
.g-A { background: #065f46; color: #a7f3d0; }
.g-B { background: #065f46; color: #a7f3d0; }
.g-C { background: #1e3a8a; color: #bfdbfe; }
.g-D { background: #1e40af; color: #bfdbfe; }
.g-E { background: #713f12; color: #fde68a; }
.g-FAIL { background: #7f1d1d; color: #fecaca; }
tr.row { cursor: pointer; }
tr.row:hover td { background: #1b1f28; }
tr.detail td { background: #12151c; padding: 0; }
.detail-inner { padding: 14px 16px; display: grid; gap: 14px;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.block h3 { margin: 0 0 6px; font-size: 12px; text-transform: uppercase;
            letter-spacing: .04em; color: #98a2b3; }
.block ul { margin: 0; padding-left: 18px; }
.block li { margin-bottom: 5px; }
a { color: #7dd3fc; }
code { background: #1c2029; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
.quote { color: #b6bccb; font-size: 12px; border-left: 2px solid #333a48;
         padding-left: 8px; margin-top: 3px; }
.muted { color: #98a2b3; }
.hide { display: none; }
"""


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _cards(metrics: Metrics) -> str:
    items = [
        ("A-D (primary)", f"{metrics.a_d_pct}%"),
        ("A-E", f"{metrics.a_e_pct}%"),
        ("Entity resolved", f"{metrics.entity_resolved_pct}%"),
        (">=1 decision maker", f"{metrics.decision_maker_found_pct}%"),
        ("Avg searches", f"{metrics.avg_searches}"),
        ("Avg pages", f"{metrics.avg_pages}"),
        ("Avg Qwen calls", f"{metrics.avg_llm_calls}"),
        ("Median seconds", f"{metrics.median_duration}"),
    ]
    cells = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v">{_esc(v)}</div></div>'
        for k, v in items
    )
    return f'<div class="cards">{cells}</div>'


def _detail(result: LeadResult, index: int) -> str:
    contact = result.contact
    contact_bits = []
    for label, value in (
        ("Direct email", f"{contact.direct_email} ({contact.email_method}, "
                         f"conf {contact.email_confidence:.2f})" if contact.direct_email else ""),
        ("General email", contact.general_email),
        ("Contact form", contact.contact_form),
        ("Phone", contact.phone),
        ("Other", contact.other),
        ("WeChat", contact.wechat),
    ):
        if value:
            contact_bits.append(f"<li><strong>{_esc(label)}:</strong> {_esc(value)}</li>")

    people_items = []
    for person in result.people[:5]:
        sources = "".join(
            f'<li><a href="{_esc(url)}" target="_blank" rel="noreferrer noopener">'
            f"{_esc(url[:110])}</a></li>"
            for url in person.sources[:3]
        )
        quotes = "".join(
            f'<div class="quote">{_esc(item.source_text[:260])}</div>'
            for item in person.evidence[:2]
            if item.source_text
        )
        people_items.append(
            f"<li><strong>{_esc(person.name)}</strong> - {_esc(person.title or 'title unknown')} "
            f"({_esc(person.role_category)}, score {person.score.total}, "
            f"conf {person.confidence:.2f}){quotes}<ul>{sources}</ul></li>"
        )

    source_items = "".join(
        f'<li><a href="{_esc(source.url)}" target="_blank" rel="noreferrer noopener">'
        f"{_esc(source.title[:90] or source.url)}</a> "
        f'<span class="muted">[{_esc(source.source_type)}]</span></li>'
        for source in result.sources[:10]
    )

    return f"""
<tr class="detail hide" data-detail="{index}"><td colspan="8">
  <div class="detail-inner">
    <div class="block">
      <h3>Resolved company</h3>
      <ul>
        <li>Canonical: <code>{_esc(result.company.canonical_name or '-')}</code></li>
        <li>Legal: <code>{_esc(result.company.legal_name or '-')}</code></li>
        <li>Domain: <code>{_esc(result.company.domain or result.input_domain)}</code></li>
        <li>Country / language: {_esc(result.company.country or '-')} /
            {_esc(result.company.language or '-')}</li>
        <li>Brand vs legal: {_esc(result.company.product_or_company)}</li>
        <li>Status: {_esc(result.company.status)} (conf {result.company.confidence:.2f})</li>
        <li>Parent: {_esc(result.company.parent_company or '-')}</li>
      </ul>
      <p class="muted">{_esc(result.company.resolution_note)}</p>
    </div>
    <div class="block">
      <h3>Best lead</h3>
      <p><strong>{_esc(result.best_lead.name or '-')}</strong>
         - {_esc(result.best_lead.title or 'title unknown')}</p>
      <p class="muted">{_esc(result.grade_reason)}</p>
      <p class="muted">Score components:
         <code>{_esc(json.dumps(result.best_lead.score_components))}</code></p>
    </div>
    <div class="block">
      <h3>Contact</h3>
      <ul>{''.join(contact_bits) or '<li class="muted">none found</li>'}</ul>
    </div>
    <div class="block">
      <h3>People ({len(result.people)})</h3>
      <ul>{''.join(people_items) or '<li class="muted">none found</li>'}</ul>
    </div>
    <div class="block">
      <h3>Sources ({len(result.sources)})</h3>
      <ul>{source_items or '<li class="muted">none</li>'}</ul>
    </div>
    <div class="block">
      <h3>Run</h3>
      <ul>
        <li>Status: {_esc(result.status)}</li>
        <li>Grade: <span class="grade g-{_esc(result.lead_grade)}">{_esc(result.lead_grade)}</span></li>
        <li>Searches / pages / Qwen: {result.research.searches_performed} /
            {result.research.pages_fetched} / {result.research.llm_calls}</li>
        <li>Cache hits: {result.research.cache_hits}</li>
        <li>Duration: {result.research.duration_seconds}s</li>
        <li>Stop reason: {_esc(result.research.stop_reason or '-')}</li>
      </ul>
    </div>
  </div>
</td></tr>"""


def _rows(results: list[LeadResult]) -> str:
    out: list[str] = []
    for index, result in enumerate(sorted(results, key=lambda r: (_GRADE_ORDER.get(r.lead_grade, 9),
                                                                  r.input_company))):
        contact = (
            result.contact.direct_email
            or result.contact.general_email
            or result.contact.contact_form
            or result.contact.phone
            or result.contact.other
            or "-"
        )
        out.append(
            f'<tr class="row" data-target="{index}">'
            f"<td>{_esc(result.input_company)}</td>"
            f"<td><code>{_esc(result.input_domain)}</code></td>"
            f"<td>{_esc(result.company.canonical_name or '-')}"
            + (
                f'<br><span class="muted">{_esc(result.company.legal_name)}</span>'
                if result.company.legal_name
                and result.company.legal_name != result.company.canonical_name
                else ""
            )
            + "</td>"
            f"<td>{_esc(result.best_lead.name or '-')}"
            + (f'<br><span class="muted">{_esc(result.best_lead.title)}</span>'
               if result.best_lead.title else "")
            + "</td>"
            f"<td>{_esc(contact)}</td>"
            f'<td class="num"><span class="grade g-{_esc(result.lead_grade)}">'
            f"{_esc(result.lead_grade)}</span></td>"
            f'<td class="num">{result.best_lead.confidence:.2f}</td>'
            f'<td class="num">{len(result.sources)}</td>'
            "</tr>"
        )
        out.append(_detail(result, index))
    return "".join(out)


_JS = """
document.querySelectorAll('tr.row').forEach(function (row) {
  row.addEventListener('click', function () {
    var target = document.querySelector('[data-detail="' + row.dataset.target + '"]');
    if (!target) return;
    target.classList.toggle('hide');
  });
});
"""


def write_html_report(
    results: list[LeadResult],
    metrics: Metrics,
    path: Path,
    *,
    inputs: list | None = None,
) -> Path:
    grades = " ".join(
        f'<span class="grade g-{grade}">{grade} {count}</span>'
        for grade, count in metrics.grade_counts.items()
    )
    failures = " ".join(
        f"<code>{_esc(category)} {count}</code>"
        for category, count in metrics.failure_categories.items()
        if count
    ) or '<span class="muted">none</span>'

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lead discovery benchmark</title>
<style>{_CSS}</style></head>
<body>
<h1>Lead discovery benchmark</h1>
<div class="sub">{metrics.total} companies &middot;
generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} &middot;
click any row to see its evidence</div>
{_cards(metrics)}
<p class="muted">{grades} &nbsp;&nbsp; failures: {failures}</p>
<table>
<thead><tr>
<th>Input</th><th>Domain</th><th>Resolved entity</th><th>Best lead</th>
<th>Contact</th><th>Grade</th><th>Conf</th><th>Sources</th>
</tr></thead>
<tbody>{_rows(results)}</tbody>
</table>
<script>{_JS}</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
