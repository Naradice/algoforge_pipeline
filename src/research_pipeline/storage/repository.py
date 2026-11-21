"""Database repository operations for HDE."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import aiosqlite


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def upsert_source(
        self,
        source_id: str,
        source_type: str,
        canonical_identifier: str,
        primary_name: str,
        primary_url: str,
    ) -> None:
        now = _now_iso()
        query = """
        INSERT INTO sources (source_id, source_type, canonical_identifier, primary_name, primary_url, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            primary_name = excluded.primary_name,
            primary_url = excluded.primary_url,
            updated_at = excluded.updated_at;
        """
        await self.conn.execute(
            query,
            (source_id, source_type, canonical_identifier, primary_name, primary_url, now, now),
        )

    async def record_alias(self, source_id: str, alias_name: str) -> None:
        # Mark all other aliases for this source as not current if new alias differs
        await self.conn.execute(
            "UPDATE source_aliases SET is_current = 0 WHERE source_id = ? AND alias_name != ?;",
            (source_id, alias_name),
        )
        query = """
        INSERT INTO source_aliases (source_id, alias_name, is_current, detected_at)
        VALUES (?, ?, 1, ?)
        ON CONFLICT(source_id, alias_name) DO UPDATE SET
            is_current = 1,
            detected_at = excluded.detected_at;
        """
        await self.conn.execute(query, (source_id, alias_name, _now_iso()))

    async def get_source_version(
        self, source_id: str, upstream_version: str
    ) -> Optional[Dict[str, Any]]:
        query = """
        SELECT version_id, source_id, upstream_version, requested_version, content_scope,
               raw_content_hash, extracted_text_hash, extractor_version, storage_rel_path,
               metadata_json, published_at, status, collected_at
        FROM source_versions
        WHERE source_id = ? AND upstream_version = ?;
        """
        async with self.conn.execute(query, (source_id, upstream_version)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "version_id": row[0],
                "source_id": row[1],
                "upstream_version": row[2],
                "requested_version": row[3],
                "content_scope": row[4],
                "raw_content_hash": row[5],
                "extracted_text_hash": row[6],
                "extractor_version": row[7],
                "storage_rel_path": row[8],
                "metadata_json": json.loads(row[9]),
                "published_at": row[10],
                "status": row[11],
                "collected_at": row[12],
            }

    async def create_source_version(
        self,
        version_id: str,
        source_id: str,
        upstream_version: str,
        requested_version: Optional[str],
        content_scope: str,
        raw_content_hash: str,
        extracted_text_hash: Optional[str],
        extractor_version: str,
        storage_rel_path: str,
        metadata_json: Dict[str, Any],
        published_at: Optional[str],
        status: str = "active",
    ) -> None:
        query = """
        INSERT INTO source_versions (
            version_id, source_id, upstream_version, requested_version, content_scope,
            raw_content_hash, extracted_text_hash, extractor_version, storage_rel_path,
            metadata_json, published_at, status, collected_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        await self.conn.execute(
            query,
            (
                version_id,
                source_id,
                upstream_version,
                requested_version,
                content_scope,
                raw_content_hash,
                extracted_text_hash,
                extractor_version,
                storage_rel_path,
                json.dumps(metadata_json, ensure_ascii=False),
                published_at,
                status,
                _now_iso(),
            ),
        )

    async def update_source_version_status(self, version_id: str, status: str) -> None:
        await self.conn.execute(
            "UPDATE source_versions SET status = ? WHERE version_id = ?;",
            (status, version_id),
        )

    async def list_all_versions(self) -> List[Dict[str, Any]]:
        query = """
        SELECT version_id, source_id, upstream_version, content_scope,
               raw_content_hash, extracted_text_hash, storage_rel_path, status
        FROM source_versions;
        """
        async with self.conn.execute(query) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "version_id": r[0],
                    "source_id": r[1],
                    "upstream_version": r[2],
                    "content_scope": r[3],
                    "raw_content_hash": r[4],
                    "extracted_text_hash": r[5],
                    "storage_rel_path": r[6],
                    "status": r[7],
                }
                for r in rows
            ]

    async def create_discovery_run(
        self,
        run_id: str,
        mode: str,
        collector_type: str,
        collector_version: str,
        search_parameters: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> None:
        query = """
        INSERT INTO discovery_runs (
            run_id, mode, collector_type, collector_version, search_parameters_json,
            agent_id, status, started_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 'running', ?);
        """
        await self.conn.execute(
            query,
            (
                run_id,
                mode,
                collector_type,
                collector_version,
                json.dumps(search_parameters, ensure_ascii=False),
                agent_id,
                _now_iso(),
            ),
        )

    async def update_discovery_run(
        self,
        run_id: str,
        status: str,
        items_found: int,
        items_collected: int,
        items_skipped: int,
        search_results: Optional[List[Dict[str, Any]]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        query = """
        UPDATE discovery_runs
        SET status = ?,
            items_found = ?,
            items_collected = ?,
            items_skipped = ?,
            search_results_json = ?,
            error_message = ?,
            completed_at = ?
        WHERE run_id = ?;
        """
        await self.conn.execute(
            query,
            (
                status,
                items_found,
                items_collected,
                items_skipped,
                json.dumps(search_results, ensure_ascii=False) if search_results else None,
                error_message,
                _now_iso(),
                run_id,
            ),
        )

    async def record_attempt(
        self,
        run_id: str,
        target_identifier: str,
        status: str,
        attempt_number: int = 1,
        resolved_source_id: Optional[str] = None,
        resolved_version: Optional[str] = None,
        error_category: Optional[str] = None,
        error_message: Optional[str] = None,
        http_status: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        query = """
        INSERT INTO collection_attempts (
            run_id, target_identifier, resolved_source_id, resolved_version,
            attempt_number, status, error_category, error_message, http_status,
            duration_ms, attempted_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        await self.conn.execute(
            query,
            (
                run_id,
                target_identifier,
                resolved_source_id,
                resolved_version,
                attempt_number,
                status,
                error_category,
                error_message,
                http_status,
                duration_ms,
                _now_iso(),
            ),
        )
