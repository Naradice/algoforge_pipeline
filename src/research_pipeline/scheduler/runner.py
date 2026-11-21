"""HDE collection runner supporting one-shot execution and lightweight daemon loop."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

from research_pipeline.config import load_config
from research_pipeline.collectors.arxiv import ArxivCollector
from research_pipeline.collectors.github import GithubCollector
from research_pipeline.collectors.rate_limiter import RateLimiter
from research_pipeline.service.collection_service import CollectionService
from research_pipeline.storage.content_store import ContentStore
from research_pipeline.storage.db import init_db
from research_pipeline.storage.lock import StorageLock, StorageLockBusyError
from research_pipeline.storage.reconciler import StorageReconciler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hde.runner")


async def execute_pass(
    collector_type: str,
    query: str,
    limit: int = 10,
    mode: str = "continuous",
    config_override: Optional[Config] = None,
    collector_override: Optional[BaseCollector] = None,
) -> int:
    config = config_override or load_config()
    content_store = ContentStore(config.storage_dir, max_storage_bytes=config.max_storage_bytes)

    # 1. Initialize DB
    await init_db(config.db_path)

    # 2. Run Reconciler to ensure storage integrity before collection
    reconciler = StorageReconciler(content_store, config.db_path, config.lock_file_path)
    try:
        report = await reconciler.reconcile()
        if (
            report.tmp_cleaned > 0
            or report.orphans_moved
            or report.missing_marked
            or report.corrupted_marked
        ):
            logger.info(
                f"[runner] Reconciliation complete: tmp_cleaned={report.tmp_cleaned}, "
                f"orphans={len(report.orphans_moved)}, missing={len(report.missing_marked)}, "
                f"corrupted={len(report.corrupted_marked)}"
            )
    except StorageLockBusyError:
        logger.warning("[runner] Could not acquire lock for reconciliation; another job is running. Skipping pass.")
        return 0

    # 3. Instantiate collector
    if collector_override:
        collector = collector_override
    else:
        rate_limiter = RateLimiter(max_wait_seconds=config.max_rate_limit_wait_seconds)
        collector = (
            ArxivCollector(config, rate_limiter)
            if collector_type == "arxiv"
            else GithubCollector(config, rate_limiter)
        )

    # 4. Run discovery
    service = CollectionService(config, content_store, collector)
    logger.info(f"[runner] Starting {collector_type} discovery: query={query!r}, limit={limit}")
    docs = await service.run_discovery(query_text=query, mode=mode, max_results=limit, agent_id="cli_runner")
    logger.info(f"[runner] Pass complete: collected {len(docs)} documents.")
    return len(docs)


async def daemon_loop(
    collector_type: str,
    query: str,
    limit: int,
    interval_seconds: float,
    stop_event: asyncio.Event,
    config_override: Optional[Config] = None,
    collector_override: Optional[BaseCollector] = None,
) -> int:
    """Run passes at interval_seconds until stop_event is set. Returns number of passes executed."""
    logger.info(f"[runner] Starting daemon loop every {interval_seconds}s for query {query!r}")
    passes_executed = 0
    while not stop_event.is_set():
        try:
            await execute_pass(
                collector_type,
                query,
                limit=limit,
                mode="continuous",
                config_override=config_override,
                collector_override=collector_override,
            )
            passes_executed += 1
        except Exception as e:
            logger.exception(f"[runner] Error during pass: {e}")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass
    return passes_executed


def main() -> None:
    parser = argparse.ArgumentParser(description="HDE Collection Runner")
    parser.add_argument("--mode", choices=["once", "daemon"], default="once", help="Execution mode")
    parser.add_argument("--collector", choices=["arxiv", "github"], default="arxiv", help="Target source collector")
    parser.add_argument("--query", type=str, default="agent-based market simulation", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum items to search/collect")
    parser.add_argument("--interval", type=float, default=3600.0, help="Daemon interval in seconds")

    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()

    def _sig_handler(*_):
        logger.info("[runner] Shutdown signal received. Stopping...")
        stop_event.set()

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, _sig_handler)
        loop.add_signal_handler(signal.SIGTERM, _sig_handler)

    try:
        if args.mode == "once":
            loop.run_until_complete(
                execute_pass(args.collector, args.query, limit=args.limit, mode="continuous")
            )
        else:
            loop.run_until_complete(
                daemon_loop(args.collector, args.query, args.limit, args.interval, stop_event)
            )
    except KeyboardInterrupt:
        logger.info("[runner] KeyboardInterrupt received. Exiting.")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
