"""External contracts and schemas handed over to H1a."""

from __future__ import annotations

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class ResearchDocument(BaseModel):
    """Immutable research document handed over from H0 to H1a."""

    source_id: str = Field(..., description="Logical source ID, e.g. 'arxiv:2301.12345', 'github:repo:28457823'")
    version_id: str = Field(..., description="Specific version ID, e.g. 'arxiv:2301.12345:v3'")
    source_type: Literal["paper", "oss"] = Field(..., description="Source type")
    canonical_identifier: str = Field(..., description="Immutable upstream identifier ('2301.12345', '28457823')")
    primary_name: str = Field(..., description="Paper title or latest repository name (owner/repo)")
    primary_url: str = Field(..., description="Primary web URL")
    upstream_version: str = Field(..., description="Confirmed upstream version ('v3', 40-character commit SHA)")
    requested_version: Optional[str] = Field(None, description="Explicitly requested version if any")

    # Critical boundary for H1a evidence interpretation
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"] = Field(
        ...,
        description="Scope of text content. Abstract/README claims are treated as 'reported', never 'reproduced'."
    )

    content_text: Optional[str] = Field(None, description="Abstract or README body text (None for metadata_only)")
    raw_content_hash: str = Field(
        ...,
        description="SHA-256 hash of normalized upstream API response composite (deterministic canonical JSON bytes for reproducibility, not raw wire bytes)."
    )
    extracted_text_hash: Optional[str] = Field(None, description="SHA-256 hash of extracted text if present")
    extractor_version: str = Field(..., description="Extractor implementation version")

    published_at: Optional[str] = Field(None, description="Published/commit date in ISO8601")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Structured metadata dictionary")
    local_artifact_dir: str = Field(..., description="Absolute local path to stored artifact directory")
    collected_at: str = Field(..., description="Timestamp of collection completion in ISO8601")
