"""Unit tests for StorageReconciler."""

import os
import time
import pytest
from pathlib import Path
import aiosqlite

from research_pipeline.storage.content_store import ContentStore, compute_sha256
from research_pipeline.storage.reconciler import StorageReconciler
from research_pipeline.storage.repository import Repository
from research_pipeline.storage.db import init_db


@pytest.mark.asyncio
async def test_reconciler_cleans_stale_tmp(tmp_path: Path):
    storage_dir = tmp_path / "storage"
    store = ContentStore(storage_dir)
    db_path = tmp_path / "hde.db"
    await init_db(db_path)

    # Create one old tmp dir and one fresh tmp dir
    old_tmp = store.tmp_dir / "old-session"
    old_tmp.mkdir()
    (old_tmp / "junk.txt").write_text("junk")
    # Set mtime back by 7200 seconds (2 hours)
    old_time = time.time() - 7200
    os.utime(old_tmp, (old_time, old_time))

    fresh_tmp = store.tmp_dir / "fresh-session"
    fresh_tmp.mkdir()
    (fresh_tmp / "active.txt").write_text("active")

    reconciler = StorageReconciler(
        store, db_path, tmp_path / "hde.lock", tmp_max_age_seconds=3600
    )
    report = await reconciler.reconcile()

    assert report.tmp_cleaned == 1
    assert not old_tmp.exists()
    assert fresh_tmp.exists()


@pytest.mark.asyncio
async def test_reconciler_moves_orphan_directory(tmp_path: Path):
    storage_dir = tmp_path / "storage"
    store = ContentStore(storage_dir)
    db_path = tmp_path / "hde.db"
    await init_db(db_path)

    # Create an orphan directory in storage that is not in DB
    orphan_dir = storage_dir / "papers" / "arxiv_2301.99999" / "v1"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "abstract.txt").write_text("Orphan content")

    reconciler = StorageReconciler(store, db_path, tmp_path / "hde.lock")
    report = await reconciler.reconcile()

    assert len(report.orphans_moved) == 1
    assert not orphan_dir.exists()
    orphans = list(store.orphan_dir.iterdir())
    assert len(orphans) >= 1
    assert any("orphan" in o.name for o in orphans)


@pytest.mark.asyncio
async def test_reconciler_detects_missing_and_corrupted(tmp_path: Path):
    storage_dir = tmp_path / "storage"
    store = ContentStore(storage_dir)
    db_path = tmp_path / "hde.db"
    conn = await init_db(db_path)
    repo = Repository(conn)

    # 1. Setup a valid record, but tamper with its file (corrupted)
    valid_dir = "papers/arxiv_corrupted/v1"
    files = {"abstract.txt": "Original text"}
    orig_hash = compute_sha256("Original text")
    artifact = store.stage_and_commit(valid_dir, files, raw_content_hash="raw", extracted_text_hash=orig_hash)
    await repo.upsert_source("arxiv:arxiv_corrupted", "paper", "arxiv_corrupted", "Corrupted Paper", "http://example.com")
    await repo.create_source_version(
        version_id="arxiv:arxiv_corrupted:v1",
        source_id="arxiv:arxiv_corrupted",
        upstream_version="v1",
        requested_version=None,
        content_scope="abstract_only",
        raw_content_hash="raw",
        extracted_text_hash=orig_hash,
        extractor_version="v1",
        storage_rel_path=valid_dir,
        metadata_json={},
        published_at=None,
        status="active",
    )
    await conn.commit()

    # Tamper with file
    (artifact.final_abs_dir / "abstract.txt").write_text("TAMPERED CONTENT")

    # 2. Setup a record whose file is completely missing
    missing_dir = "papers/arxiv_missing/v1"
    await repo.upsert_source("arxiv:arxiv_missing", "paper", "arxiv_missing", "Missing Paper", "http://example.com")
    await repo.create_source_version(
        version_id="arxiv:arxiv_missing:v1",
        source_id="arxiv:arxiv_missing",
        upstream_version="v1",
        requested_version=None,
        content_scope="abstract_only",
        raw_content_hash="raw",
        extracted_text_hash="somehash",
        extractor_version="v1",
        storage_rel_path=missing_dir,
        metadata_json={},
        published_at=None,
        status="active",
    )
    await conn.commit()
    await conn.close()

    reconciler = StorageReconciler(store, db_path, tmp_path / "hde.lock")
    report = await reconciler.reconcile()

    assert "arxiv:arxiv_corrupted:v1" in report.corrupted_marked
    assert "arxiv:arxiv_missing:v1" in report.missing_marked

    # Verify DB statuses
    async with aiosqlite.connect(str(db_path)) as c:
        r = Repository(c)
        v_corr = await r.get_source_version("arxiv:arxiv_corrupted", "v1")
        assert v_corr["status"] == "corrupted"
        v_miss = await r.get_source_version("arxiv:arxiv_missing", "v1")
        assert v_miss["status"] == "missing_file"
