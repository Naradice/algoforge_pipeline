"""Unit tests for Evidence Validator and LocalRelation integrity (User Instruction 1 & R9)."""

import pytest

from research_pipeline.extractors.models import (
    EvidenceSpan,
    LocalRelation,
    canonicalize_text,
    compute_canonical_hash,
)
from research_pipeline.extractors.validator import (
    resolve_evidence_offset,
    validate_evidence,
    validate_local_relation,
)


@pytest.fixture
def sample_text():
    raw = "We introduce Neural Posterior Estimation (NPE) to calibrate an agent-based model. NPE provides robust estimation."
    return canonicalize_text(raw)


@pytest.fixture
def sample_hash(sample_text):
    return compute_canonical_hash(sample_text)


def test_offset_resolution_unchanged(sample_text, sample_hash):
    # Quote: "Neural Posterior Estimation (NPE)"
    quote = "Neural Posterior Estimation (NPE)"
    start = sample_text.index(quote)
    end = start + len(quote)

    evidence = EvidenceSpan(
        target_field_name="abstract_text",
        target_field_hash=sample_hash,
        verbatim_quote=quote,
        start_char=start,
        end_char=end,
    )

    resolved = resolve_evidence_offset(evidence, sample_text, sample_hash)
    assert resolved.raw_offset_valid is True
    assert resolved.resolved_offset_valid is True
    assert resolved.offset_resolution_status == "unchanged"
    assert resolved.start_char == start
    assert resolved.end_char == end


def test_offset_resolution_unique_match_repaired(sample_text, sample_hash):
    # Quote: "agent-based model" (appears once)
    quote = "agent-based model"
    true_start = sample_text.index(quote)
    true_end = true_start + len(quote)

    # Erroneous raw LLM offset
    evidence = EvidenceSpan(
        target_field_name="abstract_text",
        target_field_hash=sample_hash,
        verbatim_quote=quote,
        start_char=0,  # Wrong start
        end_char=10,  # Wrong end
    )

    resolved = resolve_evidence_offset(evidence, sample_text, sample_hash)
    assert resolved.raw_offset_valid is False
    assert resolved.resolved_offset_valid is True
    assert resolved.offset_resolution_status == "unique_match"
    assert resolved.start_char == true_start
    assert resolved.end_char == true_end
    assert resolved.raw_start_char == 0
    assert resolved.raw_end_char == 10


def test_offset_resolution_ambiguous_multiple_matches(sample_text, sample_hash):
    # Quote: "NPE" (appears twice in sample_text)
    quote = "NPE"
    assert sample_text.count(quote) == 2

    # Wrong raw offset
    evidence = EvidenceSpan(
        target_field_name="abstract_text",
        target_field_hash=sample_hash,
        verbatim_quote=quote,
        start_char=0,
        end_char=5,
    )

    resolved = resolve_evidence_offset(evidence, sample_text, sample_hash)
    assert resolved.raw_offset_valid is False
    assert resolved.resolved_offset_valid is False
    assert resolved.offset_resolution_status == "ambiguous"

    # validate_evidence must reject ambiguous quote without auto-acceptance
    valid, reason, _ = validate_evidence(evidence, {"abstract_text": sample_text})
    assert valid is False
    assert "ambiguous match" in reason


def test_offset_resolution_not_found(sample_text, sample_hash):
    quote = "Completely hallucinated quote not in text"
    evidence = EvidenceSpan(
        target_field_name="abstract_text",
        target_field_hash=sample_hash,
        verbatim_quote=quote,
        start_char=0,
        end_char=10,
    )

    valid, reason, resolved = validate_evidence(evidence, {"abstract_text": sample_text})
    assert valid is False
    assert resolved.offset_resolution_status == "not_found"
    assert "not found" in reason


def test_local_relation_validation_integrity_r9(sample_text, sample_hash):
    canonical_fields = {"abstract_text": sample_text}
    quote = "calibrate an agent-based model"
    start = sample_text.index(quote)
    end = start + len(quote)

    ev = EvidenceSpan(
        target_field_name="abstract_text",
        target_field_hash=sample_hash,
        verbatim_quote=quote,
        start_char=start,
        end_char=end,
    )

    accepted_entities = {
        "met_01": "method",
        "mod_01": "model",
        "phe_01": "phenomenon",
    }

    # 1. Valid APPLIES_TO: method -> model
    rel_valid = LocalRelation(
        relation_id="rel_01",
        src_local_id="met_01",
        dst_local_id="mod_01",
        relation_type="APPLIES_TO",
        evidence=ev,
    )
    is_valid, msg, res_rel = validate_local_relation(rel_valid, accepted_entities, canonical_fields)
    assert is_valid is True
    assert res_rel is not None

    # 2. Invalid: dangling reference (dst entity rejected)
    rel_dangling = LocalRelation(
        relation_id="rel_02",
        src_local_id="met_01",
        dst_local_id="mod_rejected_99",
        relation_type="APPLIES_TO",
        evidence=ev,
    )
    is_valid, msg, _ = validate_local_relation(rel_dangling, accepted_entities, canonical_fields)
    assert is_valid is False
    assert "dangling reference" in msg

    # 3. Invalid: self-reference
    rel_self = LocalRelation(
        relation_id="rel_03",
        src_local_id="met_01",
        dst_local_id="met_01",
        relation_type="APPLIES_TO",
        evidence=ev,
    )
    is_valid, msg, _ = validate_local_relation(rel_self, accepted_entities, canonical_fields)
    assert is_valid is False
    assert "Self-referential" in msg

    # 4. Invalid endpoints: APPLIES_TO requires method -> model, but got phenomenon -> model
    rel_wrong_endpoints = LocalRelation(
        relation_id="rel_04",
        src_local_id="phe_01",
        dst_local_id="mod_01",
        relation_type="APPLIES_TO",
        evidence=ev,
    )
    is_valid, msg, _ = validate_local_relation(rel_wrong_endpoints, accepted_entities, canonical_fields)
    assert is_valid is False
    assert "Invalid endpoints" in msg
