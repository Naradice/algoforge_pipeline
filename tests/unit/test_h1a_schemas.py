"""Unit tests for H1a Pydantic schemas, controlled vocabulary, and state consistency."""

import pytest
from pydantic import ValidationError

from research_pipeline.extractors.models import (
    EvidenceSpan,
    CapabilityIO,
    PhenomenonEntity,
    ModelEntity,
    MethodEntity,
    CapabilityEntity,
    AssumptionItem,
    LimitationItem,
    LocalRelation,
    ExtractedKnowledgeRecord,
    canonicalize_text,
    compute_canonical_hash,
)
from research_pipeline.extractors.vocab import is_valid_l1_type, is_known_l2_type


def test_text_canonicalization_and_hash():
    raw_text = "  First line with CRLF\r\nSecond line with spaces   "
    canonical = canonicalize_text(raw_text)
    assert canonical == "First line with CRLF\nSecond line with spaces"
    h = compute_canonical_hash(canonical)
    assert len(h) == 64
    assert h == compute_canonical_hash(canonical)  # Deterministic


def test_evidence_span_validation():
    # Valid span
    span = EvidenceSpan(
        target_field_name="abstract_text",
        target_field_hash="a" * 64,
        verbatim_quote="Minority Game",
        start_char=10,
        end_char=23,
    )
    assert span.verbatim_quote == "Minority Game"

    # Invalid span: end_char <= start_char
    with pytest.raises(ValidationError):
        EvidenceSpan(
            target_field_name="abstract_text",
            target_field_hash="a" * 64,
            verbatim_quote="Invalid",
            start_char=20,
            end_char=20,
        )

    # Invalid hash format
    with pytest.raises(ValidationError):
        EvidenceSpan(
            target_field_name="abstract_text",
            target_field_hash="not_a_sha256",
            verbatim_quote="Invalid",
            start_char=0,
            end_char=5,
        )


def test_capability_io_state_consistency_r8():
    span = EvidenceSpan(
        target_field_name="abstract_text",
        target_field_hash="b" * 64,
        verbatim_quote="price series",
        start_char=0,
        end_char=12,
    )

    # Valid explicit IO
    io_explicit = CapabilityIO(
        data_type_l1="timeseries_1d",
        data_type_l2="price_timeseries",
        is_unspecified=False,
        evidence=span,
    )
    assert io_explicit.data_type_l1 == "timeseries_1d"
    assert io_explicit.evidence is not None

    # Invalid explicit IO: missing evidence
    with pytest.raises(ValidationError, match="explicit IO requires evidence"):
        CapabilityIO(
            data_type_l1="timeseries_1d",
            is_unspecified=False,
            evidence=None,
        )

    # Invalid explicit IO: data_type_l1 cannot be unknown
    with pytest.raises(ValidationError, match="data_type_l1 cannot be 'unknown'"):
        CapabilityIO(
            data_type_l1="unknown",
            is_unspecified=False,
            evidence=span,
        )

    # Valid unspecified IO
    io_unspecified = CapabilityIO(
        data_type_l1="unknown",
        is_unspecified=True,
    )
    assert io_unspecified.is_unspecified is True
    assert io_unspecified.evidence is None
    assert io_unspecified.data_type_l2 is None

    # Invalid unspecified IO: non-unknown L1 type
    with pytest.raises(ValidationError, match="data_type_l1 must be 'unknown'"):
        CapabilityIO(
            data_type_l1="timeseries_1d",
            is_unspecified=True,
        )

    # Invalid unspecified IO: fabricated evidence
    with pytest.raises(ValidationError, match="evidence must not be fabricated"):
        CapabilityIO(
            data_type_l1="unknown",
            is_unspecified=True,
            evidence=span,
        )

    # Invalid unspecified IO: L2 must be absent
    with pytest.raises(ValidationError, match="data_type_l2 must be None"):
        CapabilityIO(
            data_type_l1="unknown",
            data_type_l2="price_timeseries",
            is_unspecified=True,
        )


def test_local_relation_and_candidate_ids_r10():
    span = EvidenceSpan(
        target_field_name="abstract_text",
        target_field_hash="c" * 64,
        verbatim_quote="We use NPE to fit ABM",
        start_char=5,
        end_char=26,
    )

    rel = LocalRelation(
        relation_id="rel_01",
        src_local_id="met_01",
        dst_local_id="mod_01",
        relation_type="APPLIES_TO",
        evidence=span,
    )
    assert rel.relation_type == "APPLIES_TO"

    record = ExtractedKnowledgeRecord(
        source_id="arxiv_2301_00001",
        version_id="ver_01",
        version_hash="d" * 64,
        extraction_run_id="run_c4_trial1",
        content_scope="abstract_only",
        local_relations=[rel],
    )

    candidate_id = record.generate_candidate_entity_id("method", "met_01")
    assert candidate_id == f"arxiv_2301_00001:{'d'*64}:run_c4_trial1:method:met_01"

    edge_id = record.generate_candidate_edge_id(candidate_id, "APPLIES_TO", "dst_id")
    assert edge_id == f"{candidate_id}->APPLIES_TO->dst_id"
