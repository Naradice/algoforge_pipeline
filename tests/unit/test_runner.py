"""Unit tests for HDE runner (one-shot, daemon loop, lock concurrency, restart idempotency)."""

import asyncio
import pytest
import httpx
from pathlib import Path
import aiosqlite

from research_pipeline.config import Config
from research_pipeline.collectors.arxiv import ArxivCollector
from research_pipeline.collectors.rate_limiter import RateLimiter
from research_pipeline.scheduler.runner import execute_pass, daemon_loop
from research_pipeline.storage.lock import StorageLock
from research_pipeline.storage.repository import Repository
from tests.unit.test_arxiv_collector import MockTransport


@pytest.mark.asyncio
async def test_runner_once_mode(test_config: Config, arxiv_atom_xml: str):
    """Verify --mode once completes a pass and records collected documents."""
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)

    collected_count = await execute_pass(
        collector_type="arxiv",
        query="market simulation",
        limit=3,
        config_override=test_config,
        collector_override=collector,
    )
    assert collected_count == 3

    # Verify DB records created
    async with aiosqlite.connect(str(test_config.db_path)) as conn:
        repo = Repository(conn)
        versions = await repo.list_all_versions()
        assert len(versions) == 3


@pytest.mark.asyncio
async def test_runner_daemon_loop_intervals(test_config: Config, arxiv_atom_xml: str):
    """Verify --mode daemon executes repeatedly at specified intervals until stopped."""
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)

    stop_event = asyncio.Event()

    # Schedule stop_event to trigger after 0.25 seconds
    async def _stopper():
        await asyncio.sleep(0.25)
        stop_event.set()

    stopper_task = asyncio.create_task(_stopper())

    passes = await daemon_loop(
        collector_type="arxiv",
        query="market simulation",
        limit=3,
        interval_seconds=0.05,  # Short interval for testing
        stop_event=stop_event,
        config_override=test_config,
        collector_override=collector,
    )
    await stopper_task

    # Should have run multiple passes within 0.25 seconds
    assert passes >= 2


@pytest.mark.asyncio
async def test_runner_prevents_duplicate_execution(test_config: Config, arxiv_atom_xml: str):
    """Verify runner respects StorageLock and safely skips execution if another job is running."""
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)

    # Acquire the lock externally to simulate a concurrently running process
    external_lock = StorageLock(test_config.lock_file_path, operation="external_worker")
    external_lock.acquire()

    try:
        # execute_pass should detect the busy lock and return 0 (skipped)
        collected_count = await execute_pass(
            collector_type="arxiv",
            query="market simulation",
            limit=3,
            config_override=test_config,
            collector_override=collector,
        )
        assert collected_count == 0
    finally:
        external_lock.release()

    # After lock is released, execution succeeds
    collected_count = await execute_pass(
        collector_type="arxiv",
        query="market simulation",
        limit=3,
        config_override=test_config,
        collector_override=collector,
    )
    assert collected_count == 3


@pytest.mark.asyncio
async def test_runner_restart_idempotency(test_config: Config, arxiv_atom_xml: str):
    """Verify that after stopping and restarting, already collected sources are not re-registered."""
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)

    # Pass 1 (first run)
    count1 = await execute_pass(
        collector_type="arxiv",
        query="market simulation",
        limit=3,
        config_override=test_config,
        collector_override=collector,
    )
    assert count1 == 3

    # Pass 2 (restart simulation with same query)
    count2 = await execute_pass(
        collector_type="arxiv",
        query="market simulation",
        limit=3,
        config_override=test_config,
        collector_override=collector,
    )
    # No new documents collected
    assert count2 == 0

    # Total versions in DB remain 3
    async with aiosqlite.connect(str(test_config.db_path)) as conn:
        repo = Repository(conn)
        versions = await repo.list_all_versions()
        assert len(versions) == 3
