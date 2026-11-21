"""Integration test with live GitHub API (Opt-in via -m external)."""

import pytest
from pathlib import Path
import aiosqlite

from research_pipeline.config import Config
from research_pipeline.collectors.github import GithubCollector
from research_pipeline.collectors.models import CandidateRef
from research_pipeline.collectors.rate_limiter import RateLimiter
from research_pipeline.service.collection_service import CollectionService
from research_pipeline.storage.content_store import ContentStore


@pytest.mark.external
@pytest.mark.asyncio
async def test_live_github_fetch_one(test_config: Config, test_db: aiosqlite.Connection):
    """Fetch 1 known public repository from live GitHub API and verify schema compliance."""
    collector = GithubCollector(test_config, RateLimiter())
    store = ContentStore(test_config.storage_dir)
    service = CollectionService(test_config, store, collector, db_path=test_config.db_path)

    candidate = CandidateRef(
        source_type="oss",
        canonical_identifier="octocat/Hello-World",
        title="octocat/Hello-World",
        primary_url="https://github.com/octocat/Hello-World",
        raw_reference="octocat/Hello-World",
    )

    async def mock_search(q):
        return [candidate]
    collector.search = mock_search  # type: ignore

    docs = await service.run_discovery("Hello-World", max_results=1)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_id.startswith("github:repo:")
    assert len(doc.upstream_version) == 40  # 40-char commit SHA
    assert Path(doc.local_artifact_dir).exists()
    assert (Path(doc.local_artifact_dir) / "raw_repo.json").is_file()
