"""Unit tests for identifier parsing and path safety."""

import pytest

from research_pipeline.collectors.arxiv import parse_arxiv_identifier
from research_pipeline.collectors.github import parse_github_repo
from research_pipeline.collectors.models import ValidationCollectorError
from research_pipeline.storage.content_store import validate_path_segment, InvalidIdentifierError


def test_parse_arxiv_identifier():
    # Modern format with explicit version
    base, ver = parse_arxiv_identifier("2301.12345v2")
    assert base == "2301.12345"
    assert ver == "v2"

    # Modern format without version
    base, ver = parse_arxiv_identifier("2301.12345")
    assert base == "2301.12345"
    assert ver is None

    # Full URL
    base, ver = parse_arxiv_identifier("https://arxiv.org/abs/2301.12345v3")
    assert base == "2301.12345"
    assert ver == "v3"

    # PDF URL
    base, ver = parse_arxiv_identifier("https://arxiv.org/pdf/2301.12345v1.pdf")
    assert base == "2301.12345"
    assert ver == "v1"

    # Legacy format with version
    base, ver = parse_arxiv_identifier("math/0309099v2")
    assert base == "math/0309099"
    assert ver == "v2"

    # Legacy format without version
    base, ver = parse_arxiv_identifier("hep-th/9912012")
    assert base == "hep-th/9912012"
    assert ver is None

    # Legacy URL
    base, ver = parse_arxiv_identifier("https://arxiv.org/abs/math/0309099")
    assert base == "math/0309099"
    assert ver is None

    # Invalid IDs
    with pytest.raises(ValidationCollectorError):
        parse_arxiv_identifier("not-an-arxiv-id")

    with pytest.raises(ValidationCollectorError):
        parse_arxiv_identifier("../../etc/passwd")


def test_parse_github_repo():
    owner, repo = parse_github_repo("openai/gym")
    assert owner == "openai"
    assert repo == "gym"

    owner, repo = parse_github_repo("https://github.com/farama-foundation/gymnasium.git")
    assert owner == "farama-foundation"
    assert repo == "gymnasium"

    owner, repo = parse_github_repo("https://github.com/test-org/repo_name/")
    assert owner == "test-org"
    assert repo == "repo_name"

    with pytest.raises(ValidationCollectorError):
        parse_github_repo("invalid_format_without_slash")

    with pytest.raises(ValidationCollectorError):
        parse_github_repo("../attack/repo")
