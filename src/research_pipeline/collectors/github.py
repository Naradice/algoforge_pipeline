"""GitHub REST API collector with repo_id tracking, commit SHA ref locking, and composite hashing."""

from __future__ import annotations

import base64
import logging
import re
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
)
from research_pipeline.collectors.rate_limiter import RateLimiter
from research_pipeline.storage.content_store import (
    compute_sha256,
    compute_github_raw_composite_hash,
    canonical_json_bytes,
    validate_path_segment,
)

logger = logging.getLogger(__name__)

GITHUB_REPO_REGEX = re.compile(r"^([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)$")


def parse_github_repo(raw: str) -> Tuple[str, str]:
    """Parse a GitHub URL or 'owner/repo' string into (owner, repo)."""
    cleaned = raw.strip()
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        parsed = urlparse(cleaned)
        path = parsed.path.strip("/")
        cleaned = path

    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    # Split into owner and repo
    parts = cleaned.split("/")
    if len(parts) >= 2:
        owner = parts[0].strip()
        repo = parts[1].strip()
        if ".." not in owner and ".." not in repo and GITHUB_REPO_REGEX.match(f"{owner}/{repo}"):
            return owner, repo

    raise ValidationCollectorError(f"Cannot parse GitHub repository from: {raw!r}")


class GithubCollector(BaseCollector):
    BASE_URL = "https://api.github.com"

    def __init__(self, config: Config, rate_limiter: RateLimiter, client: httpx.AsyncClient | None = None):
        self.config = config
        self.rate_limiter = rate_limiter
        interval = (
            config.github_auth_min_interval_seconds
            if config.github_token
            else config.github_unauth_min_interval_seconds
        )
        self.rate_limiter.set_min_interval("api.github.com", interval)
        self._client = client

    @property
    def source_type(self) -> str:
        return "oss"

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": self.config.user_agent,
        }
        if self.config.github_token:
            headers["Authorization"] = f"token {self.config.github_token}"
        return headers

    async def _request(self, method: str, path: str, params: Dict[str, Any] | None = None) -> httpx.Response:
        client = self._client or httpx.AsyncClient(headers=self._headers(), timeout=30.0, follow_redirects=True)
        url = f"{self.BASE_URL}{path}" if path.startswith("/") else path
        try:
            await self.rate_limiter.throttle("api.github.com")
            resp = await client.request(method, url, params=params)
            await self.rate_limiter.handle_rate_limit_headers("api.github.com", resp.headers, resp.status_code)

            if resp.status_code >= 500:
                raise TransientError(f"GitHub API server error (HTTP {resp.status_code})")
            if resp.status_code in (403, 429):
                # Secondary rate limit or abuse detection
                raise TransientError(f"GitHub rate limit encountered (HTTP {resp.status_code}): {resp.text}")
            if resp.status_code == 404:
                raise PermanentError(f"GitHub resource not found: {url}")
            if resp.status_code >= 400:
                raise PermanentError(f"GitHub API client error (HTTP {resp.status_code}): {resp.text}")
            return resp
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as e:
            raise TransientError(f"GitHub connection failed: {e}") from e
        finally:
            if not self._client:
                await client.aclose()

    async def search(self, query: SearchQuery) -> List[CandidateRef]:
        # GitHub search repositories API
        path = "/search/repositories"
        params = {
            "q": query.query_text,
            "per_page": query.max_results,
            "page": (query.start_offset // query.max_results) + 1,
        }
        resp = await self._request("GET", path, params=params)
        data = resp.json()
        items = data.get("items", [])

        candidates: List[CandidateRef] = []
        for item in items:
            repo_id = str(item["id"])
            full_name = item["full_name"]
            html_url = item["html_url"]

            candidate = CandidateRef(
                source_type=self.source_type,
                canonical_identifier=repo_id,
                title=full_name,
                primary_url=html_url,
                raw_reference=full_name,
            )
            candidates.append(candidate)
        return candidates

    async def fetch(self, candidate: CandidateRef) -> CollectedContent:
        # Determine owner/repo from candidate reference
        owner, repo = parse_github_repo(candidate.raw_reference or candidate.title)

        # 1. Fetch Repository Details
        repo_resp = await self._request("GET", f"/repos/{owner}/{repo}")
        raw_repo_json = repo_resp.json()
        repo_id = str(raw_repo_json["id"])
        validate_path_segment(repo_id)

        default_branch = raw_repo_json.get("default_branch", "main")
        full_name = raw_repo_json.get("full_name", f"{owner}/{repo}")
        primary_url = raw_repo_json.get("html_url", candidate.primary_url)

        # 2. Pin to fixed Commit SHA on default branch
        commits_resp = await self._request("GET", f"/repos/{owner}/{repo}/commits/{default_branch}")
        commit_data = commits_resp.json()
        commit_sha = commit_data.get("sha")
        if not commit_sha or len(commit_sha) != 40:
            raise ValidationCollectorError(f"Invalid commit SHA returned for {full_name}: {commit_sha!r}")
        validate_path_segment(commit_sha)

        commit_date = (
            commit_data.get("commit", {}).get("committer", {}).get("date")
            or raw_repo_json.get("pushed_at")
        )

        # 3. Fetch README with explicit ref=<commit_sha> (R2)
        raw_readme_json: Optional[Dict[str, Any]] = None
        readme_text: Optional[str] = None
        content_scope = "readme_only"

        try:
            readme_resp = await self._request("GET", f"/repos/{owner}/{repo}/readme", params={"ref": commit_sha})
            raw_readme_json = readme_resp.json()
            # Decode base64
            encoded_content = raw_readme_json.get("content", "")
            try:
                readme_text = base64.b64decode(encoded_content).decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"[github] Failed to decode README base64 for {full_name}: {e}")
                readme_text = ""
        except PermanentError as e:
            if "not found" in str(e).lower() or "404" in str(e):
                logger.info(f"[github] No README found for {full_name} at commit {commit_sha} -> metadata_only")
                content_scope = "metadata_only"
                raw_readme_json = None
                readme_text = None
            else:
                raise

        # 4. Deterministic composite hash (R2)
        raw_composite_hash = compute_github_raw_composite_hash(raw_repo_json, raw_readme_json)
        extracted_text_hash = compute_sha256(readme_text) if readme_text is not None else None

        source_id = f"github:repo:{repo_id}"

        metadata = {
            "repository_id": repo_id,
            "full_name": full_name,
            "owner": owner,
            "repo": repo,
            "description": raw_repo_json.get("description"),
            "primary_url": primary_url,
            "commit_sha": commit_sha,
            "default_branch": default_branch,
            "stargazers_count": raw_repo_json.get("stargazers_count", 0),
            "license": raw_repo_json.get("license"),
            "topics": raw_repo_json.get("topics", []),
            "published_at": commit_date,
            "has_readme": (content_scope != "metadata_only"),
        }

        files_to_store: Dict[str, bytes | str] = {
            "raw_repo.json": canonical_json_bytes(raw_repo_json),
            "metadata.json": canonical_json_bytes(metadata).decode("utf-8"),
        }
        if raw_readme_json is not None:
            files_to_store["raw_readme.json"] = canonical_json_bytes(raw_readme_json)
        if readme_text is not None:
            files_to_store["readme.md"] = readme_text

        # Raw composite bytes representation for CollectedContent
        raw_composite_bytes = (
            canonical_json_bytes(raw_repo_json)
            + b"\n---HDE_RAW_SEP---\n"
            + canonical_json_bytes(raw_readme_json)
        )

        return CollectedContent(
            candidate=candidate,
            source_id=source_id,
            upstream_version=commit_sha,
            content_scope=content_scope,
            content_text=readme_text,
            raw_content_bytes=raw_composite_bytes,
            raw_content_hash=raw_composite_hash,
            extracted_text_hash=extracted_text_hash,
            extractor_version=self.config.github_extractor_version,
            published_at=commit_date,
            metadata=metadata,
            files_to_store=files_to_store,
            alias_name=full_name,
        )
