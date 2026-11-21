"""Cross-Split Isolation and Integrity Tests for Gold Standard Dataset (Section 6 & User Instruction 2).

Verifies:
1. Exact count: 10 Development, 12 Holdout Pilot, 8 Challenge (Total = 30)
2. Cross-split isolation: Zero overlap of arXiv Base IDs
3. GitHub Fork isolation: Zero overlap of repository_id and root_parent_repo_id across splits
4. Content hash isolation: Zero overlap of README text hashes across splits
5. Verbatim quote integrity: Every entity quote strictly matches canonical text slice
6. Target field hash integrity: Every target_field_hash strictly matches SHA-256 of text
7. Scope constraint: Every README text is <= 8,000 characters
"""

import json
from pathlib import Path
import pytest

from research_pipeline.extractors.models import compute_canonical_hash


@pytest.fixture
def gold_dataset():
    path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json")
    assert path.exists(), "Gold standard ground truth file must exist"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_split_counts(gold_dataset):
    assert len(gold_dataset) == 30

    dev_docs = [d for d in gold_dataset if d["split"] == "development"]
    holdout_docs = [d for d in gold_dataset if d["split"] == "holdout_pilot"]
    challenge_docs = [d for d in gold_dataset if d["split"] == "challenge"]

    assert len(dev_docs) == 10
    assert len(holdout_docs) == 12
    assert len(challenge_docs) == 8


def test_cross_split_arxiv_isolation(gold_dataset):
    dev_arxiv = {d["arxiv_base_id"] for d in gold_dataset if d["split"] == "development" and d["arxiv_base_id"]}
    holdout_arxiv = {d["arxiv_base_id"] for d in gold_dataset if d["split"] == "holdout_pilot" and d["arxiv_base_id"]}
    challenge_arxiv = {d["arxiv_base_id"] for d in gold_dataset if d["split"] == "challenge" and d["arxiv_base_id"]}

    assert len(dev_arxiv.intersection(holdout_arxiv)) == 0, "arXiv Base IDs must not leak between dev and holdout"
    assert len(dev_arxiv.intersection(challenge_arxiv)) == 0, "arXiv Base IDs must not leak between dev and challenge"
    assert len(holdout_arxiv.intersection(challenge_arxiv)) == 0, "arXiv Base IDs must not leak between holdout and challenge"


def test_cross_split_github_fork_isolation(gold_dataset):
    dev_repos = {d["repository_id"] for d in gold_dataset if d["split"] == "development" and d["repository_id"]}
    holdout_repos = {d["repository_id"] for d in gold_dataset if d["split"] == "holdout_pilot" and d["repository_id"]}
    challenge_repos = {d["repository_id"] for d in gold_dataset if d["split"] == "challenge" and d["repository_id"]}

    assert len(dev_repos.intersection(holdout_repos)) == 0, "GitHub repository IDs must not overlap between dev and holdout"
    assert len(dev_repos.intersection(challenge_repos)) == 0, "GitHub repository IDs must not overlap between dev and challenge"
    assert len(holdout_repos.intersection(challenge_repos)) == 0, "GitHub repository IDs must not overlap between holdout and challenge"

    # Root parent repo ID check (prevents fork leakage)
    dev_roots = {d["root_parent_repo_id"] for d in gold_dataset if d["split"] == "development" and d["root_parent_repo_id"]}
    holdout_roots = {d["root_parent_repo_id"] for d in gold_dataset if d["split"] == "holdout_pilot" and d["root_parent_repo_id"]}
    challenge_roots = {d["root_parent_repo_id"] for d in gold_dataset if d["split"] == "challenge" and d["root_parent_repo_id"]}

    assert len(dev_roots.intersection(holdout_roots)) == 0, "Root parent repository IDs must not overlap between dev and holdout"
    assert len(dev_roots.intersection(challenge_roots)) == 0, "Root parent repository IDs must not overlap between dev and challenge"
    assert len(holdout_roots.intersection(challenge_roots)) == 0, "Root parent repository IDs must not overlap between holdout and challenge"


def test_readme_scope_and_verbatim_integrity(gold_dataset):
    for doc in gold_dataset:
        fields = doc["target_fields"]
        hashes = doc["target_field_hashes"]

        # Scope constraint (User Instruction 2)
        readme = fields.get("readme_text", "")
        assert len(readme) <= 8000, f"README for {doc['source_id']} exceeds 8000 char limit"

        # Hash integrity
        for field_name, text in fields.items():
            if text:
                expected_hash = compute_canonical_hash(text)
                assert hashes[field_name] == expected_hash, f"Hash mismatch in {doc['source_id']} field {field_name}"

        # Entity verbatim quote integrity
        for ent in doc.get("entities", []):
            ev = ent["evidence"]
            fn = ev["target_field_name"]
            txt = fields[fn]
            quote = ev["verbatim_quote"]
            start = ev["start_char"]
            end = ev["end_char"]

            assert end <= len(txt), f"End offset exceeds text length in {doc['source_id']}"
            assert txt[start:end] == quote, f"Slice [{start}:{end}] in {doc['source_id']} does not match verbatim quote '{quote}'"
            assert ev["target_field_hash"] == hashes[fn], f"Evidence target_field_hash mismatch in {doc['source_id']}"
