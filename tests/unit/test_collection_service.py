"""Unit tests for CollectionService orchestration, idempotency, and error handling."""

import pytest
import httpx
from pathlib import Path
import aiosqlite

from research_pipeline.config import Config
from research_pipeline.collectors.arxiv import ArxivCollector
from research_pipeline.collectors.github import GithubCollector
from research_pipeline.collectors.models import CandidateRef, CollectedContent
from research_pipeline.collectors.rate_limiter import RateLimiter
from research_pipeline.service.collection_service import CollectionService
from research_pipeline.storage.content_store import ContentStore, compute_sha256
from research_pipeline.storage.repository import Repository
from research_pipeline.storage.db import init_db

from tests.unit.test_arxiv_collector import MockTransport
from tests.unit.test_github_collector import MockGithubTransport


@pytest.mark.asyncio
async def test_collection_service_arxiv_discovery(
    test_config: Config, test_db: aiosqlite.Connection, arxiv_atom_xml: str
):
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)
    store = ContentStore(test_config.storage_dir)
    service = CollectionService(test_config, store, collector)

    docs = await service.run_discovery("market simulation", max_results=3)
    assert len(docs) == 3

    # Check contract model properties
    doc1 = docs[0]
    assert doc1.source_id == "arxiv:2301.12345"
    assert doc1.upstream_version == "v3"
    assert doc1.content_scope == "abstract_only"
    assert Path(doc1.local_artifact_dir).exists()
    assert (Path(doc1.local_artifact_dir) / "abstract.txt").is_file()

    # Check DB state
    repo = Repository(test_db)
    v1 = await repo.get_source_version("arxiv:2301.12345", "v3")
    assert v1 is not None
    assert v1["status"] == "active"


@pytest.mark.asyncio
async def test_collection_service_fixed_set_idempotency(
    test_config: Config, test_db: aiosqlite.Connection, arxiv_atom_xml: str
):
    """
    Fixed candidate set idempotency acceptance test:
    Run 1 collects 3 items. Run 2 runs exact same set, 0 new items collected, 3 skipped duplicates.
    """
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)
    store = ContentStore(test_config.storage_dir)
    service = CollectionService(test_config, store, collector)

    # Run 1
    docs1 = await service.run_discovery("fixed set query", max_results=3)
    assert len(docs1) == 3

    # Capture storage bytes and DB rows after Run 1
    storage_bytes_1 = store.get_current_storage_bytes()
    repo = Repository(test_db)
    versions_1 = await repo.list_all_versions()
    assert len(versions_1) == 3

    # Run 2: exact same
    docs2 = await service.run_discovery("fixed set query", max_results=3)
    assert len(docs2) == 0  # No new docs created

    # Verification: storage and versions unchanged
    storage_bytes_2 = store.get_current_storage_bytes()
    assert storage_bytes_2 == storage_bytes_1
    versions_2 = await repo.list_all_versions()
    assert len(versions_2) == 3

    # Verify discovery run records
    async with test_db.execute(
        "SELECT status, items_collected, items_skipped FROM discovery_runs ORDER BY started_at;"
    ) as cursor:
        rows = await cursor.fetchall()
        assert len(rows) == 2
        # Run 1
        assert rows[0][0] == "completed"
        assert rows[0][1] == 3
        assert rows[0][2] == 0
        # Run 2
        assert rows[1][0] == "completed"
        assert rows[1][1] == 0
        assert rows[1][2] == 3


@pytest.mark.asyncio
async def test_collection_service_github_alias_and_rename(
    test_config: Config, test_db: aiosqlite.Connection, github_samples: dict
):
    client = httpx.AsyncClient(transport=MockGithubTransport(github_samples))
    collector = GithubCollector(test_config, RateLimiter(), client=client)
    store = ContentStore(test_config.storage_dir)
    service = CollectionService(test_config, store, collector)

    # 1. Collect openai/gym
    docs1 = await service.run_discovery("gym", max_results=1)
    # 2. Collect renamed farama-foundation/gymnasium (same repo_id 28457823)
    cand_rename = CandidateRef(
        source_type="oss",
        canonical_identifier="farama-foundation/gymnasium",
        title="farama-foundation/gymnasium",
        primary_url="https://github.com/farama-foundation/gymnasium",
        raw_reference="farama-foundation/gymnasium",
    )
    async def mock_search(q):
        return [cand_rename]

    collector.search = mock_search  # type: ignore
    docs2 = await service.run_discovery("gymnasium", max_results=1)

    repo = Repository(test_db)
    # Check aliases table
    async with test_db.execute(
        "SELECT alias_name, is_current FROM source_aliases WHERE source_id = 'github:repo:28457823' ORDER BY detected_at;"
    ) as cursor:
        aliases = await cursor.fetchall()
        assert len(aliases) == 2
        assert aliases[0][0] == "openai/gym"
        assert aliases[0][1] == 0  # old alias superseded
        assert aliases[1][0] == "farama-foundation/gymnasium"
        assert aliases[1][1] == 1  # current alias


@pytest.mark.asyncio
async def test_collection_service_content_diverged(
    test_config: Config, test_db: aiosqlite.Connection, arxiv_atom_xml: str
):
    """Test requirement 3: Upstream content divergence without version bump is caught."""
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)
    store = ContentStore(test_config.storage_dir)
    service = CollectionService(test_config, store, collector)

    # Pre-populate DB with identical version (v3) but DIFFERENT raw_content_hash
    repo = Repository(test_db)
    await repo.upsert_source("arxiv:2301.12345", "paper", "2301.12345", "Paper Title", "http://example.com")
    await repo.create_source_version(
        version_id="arxiv:2301.12345:v3",
        source_id="arxiv:2301.12345",
        upstream_version="v3",
        requested_version=None,
        content_scope="abstract_only",
        raw_content_hash="old_diverged_hash_value",
        extracted_text_hash="old_text_hash",
        extractor_version="arxiv:1.0.0",
        storage_rel_path="papers/arxiv_2301_12345/v3",
        metadata_json={},
        published_at=None,
        status="active",
    )
    await test_db.commit()

    # Mock collector search to return candidate with requested_version=None (requesting latest)
    cand_latest = CandidateRef(
        source_type="paper",
        canonical_identifier="2301.12345",
        title="Sample",
        primary_url="https://arxiv.org/abs/2301.12345",
        requested_version=None,
    )
    async def mock_search_latest(q):
        return [cand_latest]

    collector.search = mock_search_latest  # type: ignore

    # Now run discovery which fetches the actual v3 from mock XML
    docs = await service.run_discovery("query", max_results=1)
    # Must NOT overwrite!
    assert len(docs) == 0

    # Verify attempt recorded as content_diverged
    async with test_db.execute(
        "SELECT status, error_category FROM collection_attempts WHERE resolved_source_id = 'arxiv:2301.12345';"
    ) as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "failed"
        assert row[1] == "content_diverged"
