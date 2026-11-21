"""Unit tests for GithubCollector with mocks."""

import pytest
import httpx
from urllib.parse import parse_qs, urlparse

from research_pipeline.config import Config
from research_pipeline.collectors.github import GithubCollector
from research_pipeline.collectors.models import (
    CandidateRef,
)
from research_pipeline.collectors.rate_limiter import RateLimiter


class MockGithubTransport(httpx.AsyncBaseTransport):
    def __init__(self, samples: dict):
        self.samples = samples

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        parsed_url = urlparse(str(request.url))
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # 0. /search/repositories
        if "/search/repositories" in path:
            items = list(self.samples["repos"].values())
            return httpx.Response(200, json={"items": items, "total_count": len(items)}, request=request)

        # 1. /repos/{owner}/{repo}/commits/{branch}
        if "/commits/" in path:
            for name, commit in self.samples["commits"].items():
                if name in path:
                    return httpx.Response(200, json=commit, request=request)
            return httpx.Response(404, json={"message": "Not Found"}, request=request)

        # 2. /repos/{owner}/{repo}/readme
        if path.endswith("/readme"):
            ref = query.get("ref", [None])[0]
            # Ensure ref is passed!
            if not ref:
                return httpx.Response(400, json={"message": "ref is required"}, request=request)

            for name, readme in self.samples["readmes"].items():
                if name in path:
                    return httpx.Response(200, json=readme, request=request)
            # 404 for missing readme
            return httpx.Response(404, json={"message": "Not Found"}, request=request)

        # 3. /repos/{owner}/{repo}
        for name, repo in self.samples["repos"].items():
            if path.endswith(f"/repos/{name}"):
                return httpx.Response(200, json=repo, request=request)

        return httpx.Response(404, json={"message": "Not Found"}, request=request)


@pytest.mark.asyncio
async def test_github_fetch_success(test_config: Config, github_samples: dict):
    client = httpx.AsyncClient(transport=MockGithubTransport(github_samples))
    collector = GithubCollector(test_config, RateLimiter(), client=client)

    candidate = CandidateRef(
        source_type="oss",
        canonical_identifier="openai/gym",
        title="openai/gym",
        primary_url="https://github.com/openai/gym",
        raw_reference="openai/gym",
    )
    content = await collector.fetch(candidate)
    assert content.source_id == "github:repo:28457823"
    assert content.upstream_version == "7f8b2c4e5d6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c"
    assert content.content_scope == "readme_only"
    assert "OpenAI Gym" in content.content_text
    assert "raw_repo.json" in content.files_to_store
    assert "raw_readme.json" in content.files_to_store
    assert "readme.md" in content.files_to_store
    assert "metadata.json" in content.files_to_store
    assert content.alias_name == "openai/gym"


@pytest.mark.asyncio
async def test_github_fetch_metadata_only_when_no_readme(test_config: Config, github_samples: dict):
    client = httpx.AsyncClient(transport=MockGithubTransport(github_samples))
    collector = GithubCollector(test_config, RateLimiter(), client=client)

    candidate = CandidateRef(
        source_type="oss",
        canonical_identifier="test/no-readme-repo",
        title="test/no-readme-repo",
        primary_url="https://github.com/test/no-readme-repo",
        raw_reference="test/no-readme-repo",
    )
    content = await collector.fetch(candidate)
    assert content.source_id == "github:repo:99999999"
    assert content.content_scope == "metadata_only"
    assert content.content_text is None
    assert content.extracted_text_hash is None
    assert "raw_repo.json" in content.files_to_store
    assert "metadata.json" in content.files_to_store
    assert "readme.md" not in content.files_to_store
