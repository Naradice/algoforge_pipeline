"""Integration test with live arXiv API (Opt-in via -m external)."""

import pytest
from pathlib import Path
import aiosqlite

from research_pipeline.config import Config
from research_pipeline.collectors.arxiv import ArxivCollector
from research_pipeline.collectors.models import CandidateRef
from research_pipeline.collectors.rate_limiter import RateLimiter
from research_pipeline.service.collection_service import CollectionService
from research_pipeline.storage.content_store import ContentStore


@pytest.mark.external
@pytest.mark.asyncio
async def test_live_arxiv_fetch_one(test_config: Config, test_db: aiosqlite.Connection):
    """Fetch 1 known paper from live arXiv API and verify schema compliance."""
    collector = ArxivCollector(test_config, RateLimiter())
    store = ContentStore(test_config.storage_dir)
    service = CollectionService(test_config, store, collector, db_path=test_config.db_path)

    # 1706.03762: Attention Is All You Need
    candidate = CandidateRef(
        source_type="paper",
        canonical_identifier="1706.03762",
        title="Attention Is All You Need",
        primary_url="https://arxiv.org/abs/1706.03762",
        requested_version=None,
    )

    collector.search = lambda q: [candidate]  # type: ignore
    async def mock_search(q):
        return [candidate]
    collector.search = mock_search  # type: ignore

    docs = await service.run_discovery("attention is all you need", max_results=1)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_id == "arxiv:1706.03762"
    assert doc.upstream_version.startswith("v")
    assert "Transformer" in doc.content_text or "attention" in doc.content_text.lower()
    assert Path(doc.local_artifact_dir).exists()
    assert (Path(doc.local_artifact_dir) / "abstract.txt").is_file()
