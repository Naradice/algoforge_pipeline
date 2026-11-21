"""Collection service orchestrating discovery, rate limits, lock scopes, and persistence."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import aiosqlite

from research_pipeline.config import Config
from research_pipeline.collectors.base import BaseCollector
from research_pipeline.collectors.models import (
    SearchQuery,
    CandidateRef,
    CollectedContent,
    CollectorError,
    TransientError,
    RateLimitExceededError,
    PermanentError,
    VersionMismatchError,
    ValidationCollectorError,
)
from research_pipeline.service.contracts import ResearchDocument
from research_pipeline.storage.content_store import ContentStore, StoredArtifact
from research_pipeline.storage.lock import StorageLock
from research_pipeline.storage.repository import Repository

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CollectionService:
    def __init__(
        self,
        config: Config,
        content_store: ContentStore,
        collector: BaseCollector,
        db_path: Path | None = None,
    ):
        self.config = config
        self.content_store = content_store
        self.collector = collector
        self.db_path = db_path or config.db_path

    async def run_discovery(
        self,
        query_text: str,
        mode: str = "continuous",
        max_results: int = 10,
        agent_id: Optional[str] = None,
    ) -> List[ResearchDocument]:
        """
        Execute discovery run:
        1. Create run record in DB
        2. Perform search (outside StorageLock)
        3. For each candidate:
           - Fetch details (outside StorageLock)
           - Enter StorageLock critical section to stage files and commit DB transaction
        4. Complete run and return valid ResearchDocument list
        """
        run_id = str(uuid.uuid4())
        search_params = {
            "query_text": query_text,
            "max_results": max_results,
            "collector_type": "arxiv" if self.collector.source_type == "paper" else "github",
        }

        async with aiosqlite.connect(str(self.db_path)) as conn:
            repo = Repository(conn)
            await repo.create_discovery_run(
                run_id=run_id,
                mode=mode,
                collector_type="arxiv" if self.collector.source_type == "paper" else "github",
                collector_version=self.config.collector_version,
                search_parameters=search_params,
                agent_id=agent_id,
            )
            await conn.commit()

        # Step 2: Search (outside StorageLock)
        candidates: List[CandidateRef] = []
        try:
            candidates = await self.collector.search(
                SearchQuery(query_text=query_text, max_results=max_results)
            )
        except Exception as e:
            logger.exception(f"[service] Search failed for query {query_text!r}: {e}")
            async with aiosqlite.connect(str(self.db_path)) as conn:
                repo = Repository(conn)
                await repo.update_discovery_run(
                    run_id=run_id,
                    status="failed",
                    items_found=0,
                    items_collected=0,
                    items_skipped=0,
                    error_message=f"Search failed: {e}",
                )
                await conn.commit()
            return []

        search_results_summary = [
            {"rank": idx + 1, "identifier": c.canonical_identifier, "title": c.title}
            for idx, c in enumerate(candidates)
        ]

        collected_docs: List[ResearchDocument] = []
        items_collected = 0
        items_skipped = 0
        has_partial_failures = False
        abort_run = False

        # Step 3: Fetch and process each candidate
        for candidate in candidates:
            if abort_run:
                break

            target_id = candidate.raw_reference or candidate.canonical_identifier
            start_time = time.time()

            # Pre-check in DB: already collected with exact version?
            async with aiosqlite.connect(str(self.db_path)) as conn:
                repo = Repository(conn)
                # If requested_version is known, check if already active
                if candidate.requested_version:
                    source_id = (
                        f"arxiv:{candidate.canonical_identifier}"
                        if self.collector.source_type == "paper"
                        else f"github:repo:{candidate.canonical_identifier}"
                    )
                    existing = await repo.get_source_version(source_id, candidate.requested_version)
                    if existing and existing["status"] == "active":
                        logger.info(f"[service] Skipping known duplicate {source_id}:{candidate.requested_version}")
                        await repo.record_attempt(
                            run_id=run_id,
                            target_identifier=target_id,
                            status="skipped_duplicate",
                            resolved_source_id=source_id,
                            resolved_version=candidate.requested_version,
                        )
                        await conn.commit()
                        items_skipped += 1
                        continue

            # Fetch content with transient retry (outside StorageLock)
            content: Optional[CollectedContent] = None
            fetch_error_category: Optional[str] = None
            fetch_error_msg: Optional[str] = None

            for attempt in range(1, 4):
                try:
                    content = await self.collector.fetch(candidate)
                    break
                except RateLimitExceededError as e:
                    logger.warning(f"[service] Rate limit exceeded on {target_id}: {e}")
                    fetch_error_category = "rate_limit"
                    fetch_error_msg = str(e)
                    abort_run = True
                    has_partial_failures = True
                    break
                except VersionMismatchError as e:
                    logger.warning(f"[service] Version mismatch on {target_id}: {e}")
                    fetch_error_category = "version_mismatch"
                    fetch_error_msg = str(e)
                    has_partial_failures = True
                    break
                except PermanentError as e:
                    logger.warning(f"[service] Permanent error on {target_id}: {e}")
                    fetch_error_category = "permanent"
                    fetch_error_msg = str(e)
                    has_partial_failures = True
                    break
                except ValidationCollectorError as e:
                    logger.warning(f"[service] Validation error on {target_id}: {e}")
                    fetch_error_category = "validation"
                    fetch_error_msg = str(e)
                    has_partial_failures = True
                    break
                except TransientError as e:
                    logger.warning(f"[service] Transient error on {target_id} (attempt {attempt}/3): {e}")
                    if attempt == 3:
                        fetch_error_category = "transient"
                        fetch_error_msg = str(e)
                        has_partial_failures = True
                    else:
                        await asyncio.sleep(1.0 * attempt)

            if content is None:
                duration_ms = int((time.time() - start_time) * 1000)
                async with aiosqlite.connect(str(self.db_path)) as conn:
                    repo = Repository(conn)
                    await repo.record_attempt(
                        run_id=run_id,
                        target_identifier=target_id,
                        status="failed",
                        error_category=fetch_error_category,
                        error_message=fetch_error_msg,
                        duration_ms=duration_ms,
                    )
                    await conn.commit()
                continue

            # Step 4: Critical Section (StorageLock held for FS write & DB commit)
            async with StorageLock(self.config.lock_file_path, operation="collection"):
                async with aiosqlite.connect(str(self.db_path)) as conn:
                    repo = Repository(conn)

                    # Double-check existing version in DB
                    existing = await repo.get_source_version(content.source_id, content.upstream_version)
                    if existing:
                        # Check requirement 3: Upstream content divergence check
                        if existing["raw_content_hash"] != content.raw_content_hash:
                            logger.error(
                                f"[service] Upstream content divergence detected for {content.source_id}:{content.upstream_version}! "
                                f"Expected {existing['raw_content_hash']}, but got {content.raw_content_hash}."
                            )
                            await repo.record_attempt(
                                run_id=run_id,
                                target_identifier=target_id,
                                status="failed",
                                resolved_source_id=content.source_id,
                                resolved_version=content.upstream_version,
                                error_category="content_diverged",
                                error_message="Upstream API content changed for identical version tag without version bump",
                            )
                            await conn.commit()
                            has_partial_failures = True
                            continue

                        logger.info(f"[service] Skipping verified duplicate {content.source_id}:{content.upstream_version}")
                        await repo.record_attempt(
                            run_id=run_id,
                            target_identifier=target_id,
                            status="skipped_duplicate",
                            resolved_source_id=content.source_id,
                            resolved_version=content.upstream_version,
                        )
                        await conn.commit()
                        items_skipped += 1
                        continue

                    # Safe relative path in storage
                    sub_type = "papers" if self.collector.source_type == "paper" else "oss"
                    safe_source = content.source_id.replace(":", "_").replace("/", "_").replace(".", "_")
                    safe_version = content.upstream_version.replace(":", "_").replace("/", "_")
                    rel_dir = f"{sub_type}/{safe_source}/{safe_version}"

                    # Stage files atomically
                    artifact: StoredArtifact
                    try:
                        artifact = self.content_store.stage_and_commit(
                            rel_target_dir=rel_dir,
                            files=content.files_to_store,
                            raw_content_hash=content.raw_content_hash,
                            extracted_text_hash=content.extracted_text_hash,
                        )
                    except Exception as e:
                        logger.exception(f"[service] Staging failed for {content.source_id}: {e}")
                        await repo.record_attempt(
                            run_id=run_id,
                            target_identifier=target_id,
                            status="failed",
                            error_category="validation",
                            error_message=f"File staging failed: {e}",
                        )
                        await conn.commit()
                        has_partial_failures = True
                        continue

                    # DB Commit (Transaction)
                    version_id = f"{content.source_id}:{content.upstream_version}"
                    try:
                        await repo.upsert_source(
                            source_id=content.source_id,
                            source_type=self.collector.source_type,
                            canonical_identifier=candidate.canonical_identifier,
                            primary_name=candidate.title,
                            primary_url=candidate.primary_url,
                        )
                        if content.alias_name:
                            await repo.record_alias(content.source_id, content.alias_name)

                        await repo.create_source_version(
                            version_id=version_id,
                            source_id=content.source_id,
                            upstream_version=content.upstream_version,
                            requested_version=candidate.requested_version,
                            content_scope=content.content_scope,
                            raw_content_hash=content.raw_content_hash,
                            extracted_text_hash=content.extracted_text_hash,
                            extractor_version=content.extractor_version,
                            storage_rel_path=artifact.final_rel_path,
                            metadata_json=content.metadata,
                            published_at=content.published_at,
                            status="active",
                        )
                        duration_ms = int((time.time() - start_time) * 1000)
                        await repo.record_attempt(
                            run_id=run_id,
                            target_identifier=target_id,
                            status="success",
                            resolved_source_id=content.source_id,
                            resolved_version=content.upstream_version,
                            duration_ms=duration_ms,
                        )
                        await conn.commit()
                        items_collected += 1

                        # Create contract document
                        doc = ResearchDocument(
                            source_id=content.source_id,
                            version_id=version_id,
                            source_type=self.collector.source_type,
                            canonical_identifier=candidate.canonical_identifier,
                            primary_name=candidate.title,
                            primary_url=candidate.primary_url,
                            upstream_version=content.upstream_version,
                            requested_version=candidate.requested_version,
                            content_scope=content.content_scope,
                            content_text=content.content_text,
                            raw_content_hash=content.raw_content_hash,
                            extracted_text_hash=content.extracted_text_hash,
                            extractor_version=content.extractor_version,
                            published_at=content.published_at,
                            metadata=content.metadata,
                            local_artifact_dir=str(artifact.final_abs_dir),
                            collected_at=_now_iso(),
                        )
                        collected_docs.append(doc)
                    except Exception as e:
                        logger.exception(f"[service] DB commit failed for {content.source_id}. Rolling back staged files: {e}")
                        self.content_store.rollback_staged(artifact)
                        await conn.rollback()
                        has_partial_failures = True

        # Step 5: Finalize run record
        final_status = "completed"
        if abort_run or has_partial_failures:
            final_status = "partial" if (items_collected > 0 or items_skipped > 0) else "failed"

        async with aiosqlite.connect(str(self.db_path)) as conn:
            repo = Repository(conn)
            await repo.update_discovery_run(
                run_id=run_id,
                status=final_status,
                items_found=len(candidates),
                items_collected=items_collected,
                items_skipped=items_skipped,
                search_results=search_results_summary,
            )
            await conn.commit()

        return collected_docs
