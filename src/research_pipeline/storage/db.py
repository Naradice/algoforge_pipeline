"""Database connection and DDL migration for SQLite."""

from __future__ import annotations

import logging
from pathlib import Path
import aiosqlite

logger = logging.getLogger(__name__)

DDL_STATEMENTS = """
-- 1. Logical sources table
CREATE TABLE IF NOT EXISTS sources (
    source_id               TEXT PRIMARY KEY,
    source_type             TEXT NOT NULL CHECK (source_type IN ('paper', 'oss')),
    canonical_identifier    TEXT NOT NULL,
    primary_name            TEXT NOT NULL,
    primary_url             TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_sources_type_canonical ON sources(source_type, canonical_identifier);

-- 2. Source aliases table (e.g. GitHub rename tracking)
CREATE TABLE IF NOT EXISTS source_aliases (
    alias_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id               TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    alias_name              TEXT NOT NULL,
    is_current              INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    detected_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CONSTRAINT uq_source_alias UNIQUE (source_id, alias_name)
);

CREATE INDEX IF NOT EXISTS idx_aliases_name ON source_aliases(alias_name);

-- 3. Source versions table
CREATE TABLE IF NOT EXISTS source_versions (
    version_id              TEXT PRIMARY KEY,
    source_id               TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    upstream_version        TEXT NOT NULL,
    requested_version       TEXT,
    content_scope           TEXT NOT NULL CHECK (content_scope IN ('abstract_only', 'readme_only', 'metadata_only', 'full_text')),
    raw_content_hash        TEXT NOT NULL,
    extracted_text_hash     TEXT,
    extractor_version       TEXT NOT NULL,
    storage_rel_path        TEXT NOT NULL,
    metadata_json           TEXT NOT NULL,
    published_at            TEXT,
    status                  TEXT NOT NULL CHECK (status IN ('active', 'missing_file', 'corrupted', 'content_diverged')),
    collected_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CONSTRAINT uq_source_upstream UNIQUE (source_id, upstream_version)
);

CREATE INDEX IF NOT EXISTS idx_source_versions_source_id ON source_versions(source_id);
CREATE INDEX IF NOT EXISTS idx_source_versions_text_hash ON source_versions(extracted_text_hash);

-- 4. Discovery runs table
CREATE TABLE IF NOT EXISTS discovery_runs (
    run_id                  TEXT PRIMARY KEY,
    mode                    TEXT NOT NULL CHECK (mode IN ('continuous', 'problem_driven')),
    collector_type          TEXT NOT NULL CHECK (collector_type IN ('arxiv', 'github')),
    collector_version       TEXT NOT NULL,
    search_parameters_json  TEXT NOT NULL,
    agent_id                TEXT,
    status                  TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'partial')),
    search_results_json     TEXT,
    items_found             INTEGER NOT NULL DEFAULT 0,
    items_collected         INTEGER NOT NULL DEFAULT 0,
    items_skipped           INTEGER NOT NULL DEFAULT 0,
    error_message           TEXT,
    started_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at            TEXT
);

-- 5. Collection attempts table
CREATE TABLE IF NOT EXISTS collection_attempts (
    attempt_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL REFERENCES discovery_runs(run_id) ON DELETE CASCADE,
    target_identifier       TEXT NOT NULL,
    resolved_source_id      TEXT,
    resolved_version        TEXT,
    attempt_number          INTEGER NOT NULL DEFAULT 1,
    status                  TEXT NOT NULL CHECK (status IN ('success', 'skipped_duplicate', 'failed')),
    error_category          TEXT CHECK (error_category IN ('transient', 'rate_limit', 'permanent', 'validation', 'version_mismatch', 'content_diverged')),
    error_message           TEXT,
    http_status             INTEGER,
    duration_ms             INTEGER,
    attempted_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_attempts_run_id ON collection_attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_attempts_target ON collection_attempts(target_identifier);
"""


async def init_db(db_path: Path) -> aiosqlite.Connection:
    """Initialize SQLite database with required pragmas and tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))

    # Apply performance and integrity pragmas
    await conn.execute("PRAGMA journal_mode = WAL;")
    await conn.execute("PRAGMA foreign_keys = ON;")
    await conn.execute("PRAGMA synchronous = NORMAL;")
    await conn.execute("PRAGMA busy_timeout = 5000;")

    # Apply DDL
    await conn.executescript(DDL_STATEMENTS)
    await conn.commit()
    return conn
