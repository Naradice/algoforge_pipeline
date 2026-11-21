"""Storage and database integrity reconciler with double-check locking."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import aiosqlite

from research_pipeline.storage.content_store import ContentStore, compute_sha256
from research_pipeline.storage.lock import StorageLock
from research_pipeline.storage.repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationReport:
    tmp_cleaned: int = 0
    orphans_moved: List[str] = field(default_factory=list)
    missing_marked: List[str] = field(default_factory=list)
    corrupted_marked: List[str] = field(default_factory=list)


class StorageReconciler:
    def __init__(
        self,
        content_store: ContentStore,
        db_path: Path,
        lock_file_path: Path,
        tmp_max_age_seconds: float = 3600.0,
    ):
        self.content_store = content_store
        self.db_path = db_path
        self.lock_file_path = lock_file_path
        self.tmp_max_age_seconds = tmp_max_age_seconds

    async def reconcile(self) -> ReconciliationReport:
        """Run double-check reconciliation under exclusive StorageLock."""
        report = ReconciliationReport()

        async with StorageLock(self.lock_file_path, operation="reconciliation"):
            # 1. Clean stale .tmp/ directories
            now = time.time()
            if self.content_store.tmp_dir.exists():
                for entry in self.content_store.tmp_dir.iterdir():
                    if entry.is_dir():
                        mtime = entry.stat().st_mtime
                        if now - mtime > self.tmp_max_age_seconds:
                            logger.info(f"[reconciler] cleaning stale tmp dir: {entry.name}")
                            shutil.rmtree(entry, ignore_errors=True)
                            report.tmp_cleaned += 1

            # Open DB connection
            async with aiosqlite.connect(str(self.db_path)) as conn:
                repo = Repository(conn)
                all_versions = await repo.list_all_versions()
                version_map = {v["storage_rel_path"]: v for v in all_versions}

                # 2. Check for missing or corrupted files for active DB records
                for v in all_versions:
                    if v["status"] != "active":
                        continue

                    rel_path = v["storage_rel_path"]
                    abs_dir = self.content_store.storage_dir / Path(*rel_path.split("/"))

                    if not abs_dir.exists():
                        logger.warning(f"[reconciler] missing directory for version {v['version_id']}: {rel_path}")
                        await repo.update_source_version_status(v["version_id"], "missing_file")
                        report.missing_marked.append(v["version_id"])
                        continue

                    # Verify extracted text hash if present
                    if v["extracted_text_hash"]:
                        text_file = abs_dir / ("abstract.txt" if "papers" in rel_path else "readme.md")
                        if not text_file.exists():
                            logger.warning(f"[reconciler] missing text file for {v['version_id']}: {text_file.name}")
                            await repo.update_source_version_status(v["version_id"], "missing_file")
                            report.missing_marked.append(v["version_id"])
                            continue

                        current_hash = compute_sha256(text_file.read_bytes())
                        if current_hash != v["extracted_text_hash"]:
                            logger.error(f"[reconciler] corrupted text file for {v['version_id']}: hash mismatch")
                            await repo.update_source_version_status(v["version_id"], "corrupted")
                            orphan_target = (
                                self.content_store.orphan_dir
                                / f"corrupted_{int(now)}_{abs_dir.name}"
                            )
                            os.replace(abs_dir, orphan_target)
                            report.corrupted_marked.append(v["version_id"])
                            continue

                # 3. Check for orphan directories in storage (present in FS but not in DB)
                for root_type in ("papers", "oss"):
                    root_dir = self.content_store.storage_dir / root_type
                    if not root_dir.exists():
                        continue

                    for source_dir in root_dir.iterdir():
                        if not source_dir.is_dir() or source_dir.name.startswith("."):
                            continue
                        for ver_dir in source_dir.iterdir():
                            if not ver_dir.is_dir() or ver_dir.name.startswith("."):
                                continue

                            rel_check = f"{root_type}/{source_dir.name}/{ver_dir.name}"
                            if rel_check not in version_map:
                                # Double-check re-query before moving to orphan
                                async with conn.execute(
                                    "SELECT version_id FROM source_versions WHERE storage_rel_path = ?;",
                                    (rel_check,),
                                ) as cursor:
                                    if await cursor.fetchone() is None:
                                        logger.warning(f"[reconciler] orphan directory detected: {rel_check}")
                                        orphan_target = (
                                            self.content_store.orphan_dir
                                            / f"orphan_{int(now)}_{source_dir.name}_{ver_dir.name}"
                                        )
                                        os.replace(ver_dir, orphan_target)
                                        report.orphans_moved.append(rel_check)

                await conn.commit()

        return report
