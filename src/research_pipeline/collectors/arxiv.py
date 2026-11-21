"""arXiv official API collector with version validation and legacy ID support."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse

import httpx

from research_pipeline.config import Config
from research_pipeline.collectors.base import BaseCollector
from research_pipeline.collectors.models import (
    SearchQuery,
    CandidateRef,
    CollectedContent,
    PermanentError,
    TransientError,
    ValidationCollectorError,
    VersionMismatchError,
)
from research_pipeline.collectors.rate_limiter import RateLimiter
from research_pipeline.storage.content_store import (
    compute_sha256,
    canonical_json_bytes,
    validate_path_segment,
)

logger = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Regex for modern: 2301.12345 or 2301.12345v2
MODERN_ARXIV_REGEX = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
# Regex for legacy: math/0309099 or math/0309099v2 or hep-th/9912012
LEGACY_ARXIV_REGEX = re.compile(r"^([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$")


def parse_arxiv_identifier(raw: str) -> Tuple[str, Optional[str]]:
    """
    Parse an arXiv URL or raw ID into (canonical_base_id, requested_version).
    Returns e.g. ("2301.12345", "v2") or ("math/0309099", None).
    """
    cleaned = raw.strip()
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        path = urlparse(cleaned).path.strip("/")
        # path is like "abs/2301.12345v2" or "pdf/2301.12345.pdf"
        for prefix in ("abs/", "pdf/"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        if path.endswith(".pdf"):
            path = path[:-4]
        cleaned = path

    # Check modern
    m_mod = MODERN_ARXIV_REGEX.match(cleaned)
    if m_mod:
        base_id, ver = m_mod.group(1), m_mod.group(2)
        validate_path_segment(base_id.replace(".", "_"))
        return base_id, ver

    # Check legacy
    m_leg = LEGACY_ARXIV_REGEX.match(cleaned)
    if m_leg:
        base_id, ver = m_leg.group(1), m_leg.group(2)
        validate_path_segment(base_id.replace("/", "_").replace(".", "_"))
        return base_id, ver

    raise ValidationCollectorError(f"Cannot parse arXiv identifier from: {raw!r}")


def _normalize_abstract(text: str | None) -> str:
    if not text:
        return ""
    # Replace sequences of whitespace/newlines with single spaces while preserving paragraph breaks if any
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return " ".join(lines)


class ArxivCollector(BaseCollector):
    API_URL = "https://export.arxiv.org/api/query"

    def __init__(self, config: Config, rate_limiter: RateLimiter, client: httpx.AsyncClient | None = None):
        self.config = config
        self.rate_limiter = rate_limiter
        self.rate_limiter.set_min_interval("arxiv.org", config.arxiv_min_interval_seconds)
        self._client = client

    @property
    def source_type(self) -> str:
        return "paper"

    async def _get(self, params: Dict[str, Any]) -> str:
        client = self._client or httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=30.0,
            follow_redirects=True,
        )
        try:
            await self.rate_limiter.throttle("arxiv.org")
            resp = await client.get(self.API_URL, params=params)
            await self.rate_limiter.handle_rate_limit_headers("arxiv.org", resp.headers, resp.status_code)

            if resp.status_code >= 500:
                raise TransientError(f"arXiv API server error (HTTP {resp.status_code})")
            if resp.status_code >= 400:
                raise PermanentError(f"arXiv API client error (HTTP {resp.status_code})")
            return resp.text
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as e:
            raise TransientError(f"arXiv connection failed: {e}") from e
        finally:
            if not self._client:
                await client.aclose()

    async def search(self, query: SearchQuery) -> List[CandidateRef]:
        params = {
            "search_query": query.query_text,
            "start": query.start_offset,
            "max_results": query.max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        xml_text = await self._get(params)
        return self._parse_feed_candidates(xml_text)

    def _parse_feed_candidates(self, xml_text: str) -> List[CandidateRef]:
        candidates: List[CandidateRef] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ValidationCollectorError(f"Failed to parse arXiv Atom XML: {e}") from e

        for entry in root.findall("atom:entry", ATOM_NS):
            id_elem = entry.find("atom:id", ATOM_NS)
            title_elem = entry.find("atom:title", ATOM_NS)
            if id_elem is None or not id_elem.text:
                continue

            entry_id_url = id_elem.text.strip()
            base_id, version = parse_arxiv_identifier(entry_id_url)
            title = _normalize_abstract(title_elem.text) if title_elem is not None else base_id

            candidate = CandidateRef(
                source_type=self.source_type,
                canonical_identifier=base_id,
                title=title,
                primary_url=f"https://arxiv.org/abs/{base_id}",
                requested_version=version,
                raw_reference=entry_id_url,
            )
            candidates.append(candidate)
        return candidates

    async def fetch(self, candidate: CandidateRef) -> CollectedContent:
        params = {"id_list": candidate.canonical_identifier}
        xml_text = await self._get(params)

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ValidationCollectorError(f"Failed to parse arXiv Atom XML: {e}") from e

        entry = root.find("atom:entry", ATOM_NS)
        if entry is None:
            raise PermanentError(f"arXiv document not found for ID: {candidate.canonical_identifier}")

        id_elem = entry.find("atom:id", ATOM_NS)
        if id_elem is None or not id_elem.text:
            raise ValidationCollectorError(f"Missing <id> in arXiv entry for: {candidate.canonical_identifier}")

        # Extract resolved upstream version
        _, resolved_version = parse_arxiv_identifier(id_elem.text.strip())
        if not resolved_version:
            # If API did not return version in ID, default to v1
            resolved_version = "v1"

        # R1 Validation: Check requested vs resolved version
        if candidate.requested_version and candidate.requested_version != resolved_version:
            raise VersionMismatchError(
                f"Requested arXiv version {candidate.requested_version!r} but upstream API returned {resolved_version!r}. "
                "Cannot substitute with unverified latest version."
            )

        title_elem = entry.find("atom:title", ATOM_NS)
        summary_elem = entry.find("atom:summary", ATOM_NS)
        published_elem = entry.find("atom:published", ATOM_NS)
        updated_elem = entry.find("atom:updated", ATOM_NS)

        title = _normalize_abstract(title_elem.text) if title_elem is not None else candidate.title
        abstract_text = _normalize_abstract(summary_elem.text) if summary_elem is not None else ""
        published_at = published_elem.text.strip() if published_elem is not None and published_elem.text else None
        updated_at = updated_elem.text.strip() if updated_elem is not None and updated_elem.text else None

        authors = [
            a.find("atom:name", ATOM_NS).text.strip()
            for a in entry.findall("atom:author", ATOM_NS)
            if a.find("atom:name", ATOM_NS) is not None and a.find("atom:name", ATOM_NS).text
        ]
        categories = [
            c.attrib.get("term")
            for c in entry.findall("atom:category", ATOM_NS)
            if c.attrib.get("term")
        ]

        metadata = {
            "title": title,
            "authors": authors,
            "categories": categories,
            "published_at": published_at,
            "updated_at": updated_at,
            "canonical_identifier": candidate.canonical_identifier,
            "resolved_version": resolved_version,
            "requested_version": candidate.requested_version,
            "primary_url": candidate.primary_url,
        }

        # Serialized entry XML for raw content hash
        raw_entry_bytes = ET.tostring(entry, encoding="utf-8")
        raw_content_hash = compute_sha256(raw_entry_bytes)
        extracted_text_hash = compute_sha256(abstract_text) if abstract_text else None

        # Build clean directory identifier
        safe_canonical = candidate.canonical_identifier.replace("/", "_").replace(".", "_")
        source_id = f"arxiv:{candidate.canonical_identifier}"

        files_to_store: Dict[str, bytes | str] = {
            "raw_entry.xml": raw_entry_bytes,
            "abstract.txt": abstract_text,
            "metadata.json": canonical_json_bytes(metadata).decode("utf-8"),
        }

        return CollectedContent(
            candidate=candidate,
            source_id=source_id,
            upstream_version=resolved_version,
            content_scope="abstract_only",
            content_text=abstract_text,
            raw_content_bytes=raw_entry_bytes,
            raw_content_hash=raw_content_hash,
            extracted_text_hash=extracted_text_hash,
            extractor_version=self.config.arxiv_extractor_version,
            published_at=published_at,
            metadata=metadata,
            files_to_store=files_to_store,
        )
