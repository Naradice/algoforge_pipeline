"""Unit tests for ContentStore and path safety."""

import pytest
from pathlib import Path

from research_pipeline.storage.content_store import (
    ContentStore,
    validate_path_segment,
    InvalidIdentifierError,
    StorageQuotaExceededError,
    compute_github_raw_composite_hash,
    compute_sha256,
)


def test_validate_path_segment():
    assert validate_path_segment("arxiv_2301.12345") == "arxiv_2301.12345"
    assert validate_path_segment("github:repo:28457823") == "github:repo:28457823"
    assert validate_path_segment("v1") == "v1"

    with pytest.raises(InvalidIdentifierError):
        validate_path_segment("../secret")

    with pytest.raises(InvalidIdentifierError):
        validate_path_segment("dir/subdir")

    with pytest.raises(InvalidIdentifierError):
        validate_path_segment("dir\\subdir")

    with pytest.raises(InvalidIdentifierError):
        validate_path_segment("")


def test_github_composite_hash_deterministic():
    raw_repo = {"name": "test", "id": 12345, "stars": 100}
    raw_readme = {"content": "dGVzdA==", "encoding": "base64"}

    hash1 = compute_github_raw_composite_hash(raw_repo, raw_readme)
    hash2 = compute_github_raw_composite_hash(raw_repo, raw_readme)
    assert hash1 == hash2

    # Different key order must produce same hash
    raw_repo_reordered = {"stars": 100, "name": "test", "id": 12345}
    hash3 = compute_github_raw_composite_hash(raw_repo_reordered, raw_readme)
    assert hash1 == hash3

    # README missing (None) produces deterministic hash
    hash_no_readme = compute_github_raw_composite_hash(raw_repo, None)
    assert hash_no_readme != hash1


def test_atomic_stage_and_commit(tmp_path: Path):
    store = ContentStore(tmp_path / "storage")
    files = {
        "metadata.json": '{"title": "Test Paper"}',
        "abstract.txt": "This is a test abstract.",
    }
    raw_hash = compute_sha256('{"title": "Test Paper"}')
    text_hash = compute_sha256("This is a test abstract.")

    artifact = store.stage_and_commit(
        rel_target_dir="papers/arxiv_2301.12345/v1",
        files=files,
        raw_content_hash=raw_hash,
        extracted_text_hash=text_hash,
    )

    assert artifact.final_abs_dir.exists()
    assert (artifact.final_abs_dir / "metadata.json").is_file()
    assert (artifact.final_abs_dir / "abstract.txt").is_file()
    # Staging dir must be empty/clean
    assert len(list(store.tmp_dir.iterdir())) == 0


def test_rollback_staged(tmp_path: Path):
    store = ContentStore(tmp_path / "storage")
    files = {"test.txt": "hello"}
    artifact = store.stage_and_commit(
        "papers/test/v1", files, raw_content_hash="h1"
    )
    assert artifact.final_abs_dir.exists()

    store.rollback_staged(artifact)
    assert not artifact.final_abs_dir.exists()
    # Moved to orphan
    orphans = list(store.orphan_dir.iterdir())
    assert len(orphans) == 1
    assert "rollback" in orphans[0].name


def test_storage_quota_exceeded(tmp_path: Path):
    store = ContentStore(tmp_path / "storage", max_storage_bytes=100)
    # Write 150 bytes
    large_files = {"big.bin": b"x" * 150}
    with pytest.raises(StorageQuotaExceededError):
        store.stage_and_commit("papers/overflow/v1", large_files, "hash")
