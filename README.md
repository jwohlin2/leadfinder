# leadfinder

Zero-cost lead discovery for obscure companies: given a company name and a
domain, resolve the real legal entity, find a current decision-maker, and
produce a contact path (email, form or phone) — with a retrieved source for
every claim.

No paid data providers. Search runs on the Brave Search API free tier with
DuckDuckGo/Bing fallbacks (self-hosted SearXNG also supported); the LLM is any
OpenAI-compatible endpoint you supply (vLLM, llama.cpp, LM Studio, Ollama...).

## Benchmark

Frozen 98-company list (`data/input/companies.csv`), same input every run:

| Metric | Result |
| --- | --- |
| A–D leads (primary) | **16.3%** |
| A–E | 56.1% |
| Entity resolved | 69.4% |
| ≥1 person found | 27.6% |
| ≥1 decision maker | 22.4% |
| ≥2 decision makers | 14.3% |
| Technical failures | 0 |

Against the Apollo columns shipped in the CSV: Apollo marks 17 companies
completed and lists people for 24 of them (69 people total). This run found at
least one person for 27 companies (81 people) and graded 16 as A–D leads
(person + usable contact path); 13 of those A–D leads are companies Apollo did
not mark completed. Per-company detail is in `data/output/benchmark_summary.md`
and `report.html`.

## How a company is resolved

1. **Crawl the supplied domain** and extract contacts, legal names, people and
   contact forms from the homepage plus the highest-signal internal links.
2. **Search deterministically** for the brand, the domain and role terms, then
   resolve the entity with the LLM — distinguishing brand from legal entity and
   only attributing a legal name that shares a distinctive token or appears on
   the target's own domain.
3. **People pass**: site-scoped queries (`site:domain team|about|leadership|
   founders|staff|people`, plus native-language vocabulary) find the pages
   where decision-makers are actually published; those pages are fetched
   preferentially, ahead of the adaptive loop's own budget.
4. **Adaptive loop**: the planner proposes gap-filling searches (contact,
   corroboration) until the success condition — confident identity + decision
   maker + contact path — or the per-company budget runs out.
5. **Score and grade A–FAIL**, with the deterministic scorer as a floor: the
   LLM can lower or explain a grade but never promote a claim past its
   evidence.

People are harvested from team tables, heading/text adjacency (a name line
directly above a title line), inline pairs, prose, and search snippets —
rejecting app stores, partner directories and tech-stack listings, which
describe products rather than staff.

## Quick start

```powershell
# 1. Install (Python 3.11+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # add .[playwright] for JS-rendered sites

# 2. Configure: copy .env.example to .env and fill in the Qwen + Brave values
Copy-Item .env.example .env

# 3. Check wiring
python -m leadfinder doctor

# 4. Research one company
python -m leadfinder company "TravelBy Software" "availroom.com"

# 5. Run the full 98-company benchmark
python -m leadfinder run data/input/companies.csv --workers 4
```

Optional self-hosted SearXNG instead of Brave:

```powershell
docker compose -f docker/docker-compose.yml up -d
# then set LEADFINDER_SEARCH_PROVIDER=searxng in .env
```

Without an LLM configured the pipeline still runs end to end on deterministic
signals only; grades are conservative (typically E/FAIL) because no
decision-maker can be verified. `doctor` probes `/v1/models` and reports the
connection status.

## Usage

```powershell
python -m leadfinder run data/input/companies.csv [--limit N] [--offset N]
                     [--workers 4] [--max-queries 20] [--max-pages 30] [--max-rounds 4]
python -m leadfinder company "Name" "domain.com" [--json]
python -m leadfinder doctor
python -m leadfinder report        # rebuild summary/report from results.json
```

Outputs land in `data/output/`:

| File | Contents |
| --- | --- |
| `leads.csv` / `benchmark_results.csv` | one row per company: entity, best lead, contact, grade, reason, sources |
| `benchmark_summary.md` | headline metrics, grade mix, failure categories, per-company table |
| `report.html` | self-contained HTML report |
| `results.json` / `benchmark_results.json` | full results including evidence |
| `leadfinder.db` | SQLite database: companies, runs, people, evidence, page/search caches |

## Configuration

`config.yaml` holds the defaults; every value can be overridden by an
environment variable (see `.env.example`). Key knobs:

| Setting | Default | Purpose |
| --- | --- | --- |
| `research.max_search_queries` | 20 | hard search budget per company |
| `research.max_pages` | 30 | hard page-fetch budget per company |
| `research.max_llm_rounds` | 4 | planner rounds per company |
| `research.company_timeout` | 420 | wall-clock ceiling (s); partial results are still graded |
| `research.workers` | 4 | concurrent companies |
| `search.fetch_top_n` | 6 | results fetched per query |
| `search.max_queries_per_run` | 400 | Brave quota safety cap |
| `crawler.playwright_fallback` | true | render JS-heavy pages when text is thin |

## Evidence philosophy

Every person, title, company name and contact path must point at a page that
was actually retrieved — the pipeline enforces "no source, no fact" in code,
not just in the prompt. Cross-company attributions (partner executives quoted
on a press release, look-alike company names) are rejected by whole-name
similarity plus distinctive-token checks, and the LLM verifier cannot rescue a
candidate whose source URL was never fetched.

## Project layout

```
leadfinder/
  cli.py              command line entry points
  config.py           config.yaml + environment overrides
  crawl/              HTTP crawler, extraction, Playwright fallback
  llm/                OpenAI-compatible client + prompt builders
  models/             pydantic models: company, person, evidence, result
  research/           entity resolution, people pass, planner, grading, contacts
  search/             Brave, SearXNG, DuckDuckGo/Bing providers + cache
  storage/            SQLite schema, database, page/search caches
  benchmark/          metrics, CSV/Markdown/HTML report writers
tests/                unit tests (extraction, false positives, people pass)
docker/               optional SearXNG compose setup
data/input/           frozen benchmark CSV
```

## Tests

```powershell
pip install -e ".[dev]"
pytest
```

## Data handling

The tool reads public web pages and stores retrieved URLs and quotes as
evidence. `data/output/` contains personal contact data and is gitignored —
do not commit it. Respect each site's terms and applicable privacy law when
using the output.

## License

Private repository — all rights reserved.
