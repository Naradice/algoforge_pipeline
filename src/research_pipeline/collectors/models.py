"""Domain models and exceptions for collectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


class CollectorError(Exception):
    """Base collector error."""


class TransientError(CollectorError):
    """Temporary network or server errors suitable for retry."""


class RateLimitExceededError(CollectorError):
    """API rate limit exceeded and wait time exceeds threshold."""
    def __init__(self, message: str, wait_seconds: float):
        super().__init__(message)
        self.wait_seconds = wait_seconds


class PermanentError(CollectorError):
    """404 Not Found, deleted resources, or unrecoverable bad requests."""


class VersionMismatchError(CollectorError):
    """Requested version does not match resolved version from upstream."""


class ValidationCollectorError(CollectorError):
    """Invalid payload, corrupted response, or missing required fields."""


@dataclass(frozen=True)
class SearchQuery:
    query_text: str
    max_results: int = 10
    start_offset: int = 0


@dataclass(frozen=True)
class CandidateRef:
    source_type: str              # 'paper' | 'oss'
    canonical_identifier: str     # '2301.12345' | '28457823'
    title: str                    # Paper title or owner/repo
    primary_url: str
    requested_version: Optional[str] = None  # Explicit version if requested
    raw_reference: Optional[str] = None      # Original query or input (e.g. "openai/gym")


@dataclass(frozen=True)
class CollectedContent:
    candidate: CandidateRef
    source_id: str                # e.g. "arxiv:2301.12345", "github:repo:28457823"
    upstream_version: str         # Confirmed version (e.g. "v3", 40-char SHA)
    content_scope: str            # 'abstract_only' | 'readme_only' | 'metadata_only'
    content_text: Optional[str]   # Abstract or README (None for metadata_only)
    raw_content_bytes: bytes      # Raw XML or composite JSON bytes
    raw_content_hash: str         # SHA-256 of raw_content_bytes
    extracted_text_hash: Optional[str]
    extractor_version: str
    published_at: Optional[str]   # ISO8601
    metadata: Dict[str, Any]
    files_to_store: Dict[str, bytes | str]
    alias_name: Optional[str] = None  # e.g. "openai/gym" for GitHub
