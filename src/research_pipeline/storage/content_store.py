"""Atomic content storage and integrity checking."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any


class StorageError(RuntimeError):
    """Base storage exception."""


class StorageQuotaExceededError(StorageError):
    """Raised when storage disk usage exceeds quota limit."""


class InvalidIdentifierError(ValueError):
    """Raised when an identifier fails path safety validation."""


PATH_SEGMENT_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.:]+$")


def validate_path_segment(segment: str) -> str:
    """Ensure a string can be safely used as a single path segment without traversal risks."""
    if not segment or not PATH_SEGMENT_REGEX.match(segment) or ".." in segment:
        raise InvalidIdentifierError(f"Invalid path segment: {segment!r}")
    if len(segment) > 128:
        raise InvalidIdentifierError(f"Path segment too long (>128): {segment!r}")
    return segment


def canonical_json_bytes(data: Dict[str, Any] | None) -> bytes:
    """Deterministic UTF-8 canonical JSON bytes with sorted keys."""
    if data is None:
        return b""
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_github_raw_composite_hash(raw_repo: dict, raw_readme: dict | None) -> str:
    """Deterministic composite hash for GitHub: repo JSON + separator + readme JSON."""
    repo_bytes = canonical_json_bytes(raw_repo)
    readme_bytes = canonical_json_bytes(raw_readme) if raw_readme else b""
    composite = repo_bytes + b"\n---HDE_RAW_SEP---\n" + readme_bytes
    return hashlib.sha256(composite).hexdigest()


@dataclass
class StoredArtifact:
    final_rel_path: str
    final_abs_dir: Path
    raw_content_hash: str
    extracted_text_hash: str | None


class ContentStore:
    def __init__(self, storage_dir: Path, max_storage_bytes: int = 5 * 1024 * 1024 * 1024):
        self.storage_dir = storage_dir
        self.tmp_dir = storage_dir / ".tmp"
        self.orphan_dir = storage_dir / ".orphan"
        self.max_storage_bytes = max_storage_bytes

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.orphan_dir.mkdir(parents=True, exist_ok=True)

    def get_current_storage_bytes(self) -> int:
        total = 0
        for entry in self.storage_dir.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    def check_storage_quota(self, additional_bytes: int = 0) -> None:
        current = self.get_current_storage_bytes()
        if current + additional_bytes > self.max_storage_bytes:
            raise StorageQuotaExceededError(
                f"Storage quota exceeded: {current + additional_bytes} bytes > {self.max_storage_bytes} bytes"
            )

    def stage_and_commit(
        self,
        rel_target_dir: str,
        files: Dict[str, bytes | str],
        raw_content_hash: str,
        extracted_text_hash: str | None = None,
    ) -> StoredArtifact:
        """
        Stage files in .tmp/{uuid}, compute hashes, fsync, and atomically move to final directory.
        rel_target_dir: e.g. "papers/arxiv_2301.12345/v3"
        files: filename -> content bytes or string
        """
        # Validate path segments
        segments = rel_target_dir.replace("\\", "/").strip("/").split("/")
        for seg in segments:
            validate_path_segment(seg)

        final_dir = self.storage_dir / Path(*segments)
        staging_dir = self.tmp_dir / str(uuid.uuid4())
        staging_dir.mkdir(parents=True, exist_ok=False)

        total_bytes = 0
        try:
            for filename, content in files.items():
                validate_path_segment(filename)
                target_file = staging_dir / filename
                content_bytes = content.encode("utf-8") if isinstance(content, str) else content
                total_bytes += len(content_bytes)
                with open(target_file, "wb") as f:
                    f.write(content_bytes)
                    f.flush()
                    os.fsync(f.fileno())

            self.check_storage_quota(total_bytes)

            # Atomic move
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                # If target already exists, backup to orphan before replace to ensure safe overwrites
                backup_orphan = self.orphan_dir / f"conflict_{uuid.uuid4()}_{final_dir.name}"
                os.replace(final_dir, backup_orphan)

            os.replace(staging_dir, final_dir)

            return StoredArtifact(
                final_rel_path=rel_target_dir,
                final_abs_dir=final_dir,
                raw_content_hash=raw_content_hash,
                extracted_text_hash=extracted_text_hash,
            )
        except Exception:
            # Clean up staging dir on any failure
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def rollback_staged(self, artifact: StoredArtifact) -> None:
        """Compensation action: move committed directory to .orphan if DB commit failed."""
        if artifact.final_abs_dir.exists():
            orphan_path = self.orphan_dir / f"rollback_{uuid.uuid4()}_{artifact.final_abs_dir.name}"
            try:
                os.replace(artifact.final_abs_dir, orphan_path)
            except Exception:
                shutil.rmtree(artifact.final_abs_dir, ignore_errors=True)
