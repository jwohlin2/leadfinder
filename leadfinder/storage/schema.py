"""SQLite schema.

Deliberately plain relational tables. No graph layer in V0: brand/parent
relationships are stored as normalised `related_entities` rows so later
research can reuse an existing company record instead of re-researching it.
"""

from __future__ import annotations

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    domain            TEXT UNIQUE,
    input_company     TEXT,
    canonical_name    TEXT,
    legal_name        TEXT,
    country           TEXT,
    language          TEXT,
    company_type      TEXT,
    product_or_company TEXT,
    parent_company    TEXT,
    status            TEXT,
    industry          TEXT,
    size_hint         TEXT,
    confidence        REAL DEFAULT 0,
    aliases_json      TEXT DEFAULT '[]',
    resolution_note   TEXT,
    created_at        TEXT,
    updated_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_companies_canonical ON companies(canonical_name);

CREATE TABLE IF NOT EXISTS related_entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    name        TEXT,
    domain      TEXT,
    relation    TEXT,
    UNIQUE(company_id, name, relation)
);

CREATE TABLE IF NOT EXISTS research_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    input_company TEXT,
    input_domain  TEXT,
    status        TEXT,
    lead_grade    TEXT,
    failure_category TEXT,
    searches      INTEGER DEFAULT 0,
    pages         INTEGER DEFAULT 0,
    llm_calls     INTEGER DEFAULT 0,
    cache_hits    INTEGER DEFAULT 0,
    duration      REAL DEFAULT 0,
    stop_reason   TEXT,
    error         TEXT,
    started_at    TEXT,
    finished_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_domain ON research_runs(input_domain);

CREATE TABLE IF NOT EXISTS search_queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES research_runs(id) ON DELETE CASCADE,
    provider    TEXT,
    query       TEXT,
    round       INTEGER,
    result_count INTEGER,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS search_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id    INTEGER REFERENCES search_queries(id) ON DELETE CASCADE,
    position    INTEGER,
    url         TEXT,
    title       TEXT,
    snippet     TEXT,
    engine      TEXT,
    score       REAL,
    fetched     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_results_query ON search_results(query_id);

CREATE TABLE IF NOT EXISTS pages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    url          TEXT,
    final_url    TEXT,
    status_code  INTEGER,
    title        TEXT,
    text         TEXT,
    html         TEXT,
    source_type  TEXT,
    via          TEXT,
    error        TEXT,
    retrieved_at TEXT,
    UNIQUE(url, via)
);
CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);

CREATE TABLE IF NOT EXISTS evidence (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id   INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    person_id    INTEGER,
    claim_type   TEXT,
    claim_value  TEXT,
    source_url   TEXT,
    source_title TEXT,
    source_text  TEXT,
    retrieved_at TEXT,
    confidence   REAL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_evidence_company ON evidence(company_id);
CREATE INDEX IF NOT EXISTS idx_evidence_person ON evidence(person_id);

CREATE TABLE IF NOT EXISTS people (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id       INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    name             TEXT,
    normalized_name  TEXT,
    name_key         TEXT,
    title            TEXT,
    role_category    TEXT,
    current_company  TEXT,
    linkedin         TEXT,
    location         TEXT,
    confidence       REAL DEFAULT 0,
    score_total      REAL DEFAULT 0,
    score_components TEXT DEFAULT '{}',
    is_current       INTEGER DEFAULT 1,
    notes            TEXT,
    UNIQUE(company_id, name_key)
);
CREATE INDEX IF NOT EXISTS idx_people_name_key ON people(name_key);

CREATE TABLE IF NOT EXISTS person_evidence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    evidence_id INTEGER REFERENCES evidence(id) ON DELETE CASCADE,
    UNIQUE(person_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS contacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    person_id   INTEGER,
    kind        TEXT,
    value       TEXT,
    method      TEXT,
    confidence  REAL DEFAULT 0,
    source_url  TEXT,
    UNIQUE(company_id, kind, value)
);

-- ------------------------------- caches ---------------------------------
CREATE TABLE IF NOT EXISTS search_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT,
    query_key   TEXT,
    query       TEXT,
    results_json TEXT,
    result_count INTEGER,
    created_at  TEXT,
    fetched_at  TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_search_cache_key ON search_cache(provider, query_key);

CREATE TABLE IF NOT EXISTS page_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    url_key      TEXT UNIQUE,
    url          TEXT,
    final_url    TEXT,
    status_code  INTEGER,
    title        TEXT,
    text         TEXT,
    html         TEXT,
    source_type  TEXT,
    error        TEXT,
    fetched_at   TEXT
);
"""
