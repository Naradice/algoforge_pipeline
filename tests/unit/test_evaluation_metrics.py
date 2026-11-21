"""Unit tests for Matching, Evaluation Metrics, and Cluster Bootstrap."""

import pytest

from research_pipeline.extractors.raw_artifact import (
    EvaluationCandidate,
    RawExtractionArtifact,
    ExtractionTrioBundle,
)
from research_pipeline.extractors.models import (
    EvidenceSpan,
    ExtractedKnowledgeRecord,
    CapabilityEntity,
    CapabilityIO,
)
from research_pipeline.evaluation.matching import (
    match_entities_bipartite,
    are_semantically_equivalent,
)
from research_pipeline.evaluation.metrics import evaluate_run
from research_pipeline.evaluation.bootstrap import (
    calculate_wilson_score_interval,
    run_cluster_bootstrap,
    calculate_multilabel_cohens_kappa,
)


def test_semantic_equivalence_matching():
    # Exact and synonym equivalence
    assert are_semantically_equivalent("Neural Posterior Estimation", ["NPE", "SNPE"], "NPE") is True
    assert are_semantically_equivalent("Minority Game", ["El Farol"], "Minority Game Model") is True
    assert are_semantically_equivalent("GARCH(1,1)", [], "GARCH") is True
    assert are_semantically_equivalent("Transformer", [], "Convolutional Network") is False


def test_bipartite_matching_hungarian():
    gold = [
        {"local_id": "g1", "entity_type": "model", "name": "Minority Game", "synonyms": []},
        {"local_id": "g2", "entity_type": "method", "name": "Neural Posterior Estimation", "synonyms": ["NPE"]},
    ]

    preds = [
        EvaluationCandidate(
            candidate_local_id="p1",
            entity_type="model",
            name_or_statement="Minority Game",
            condition_id="C4",
            extraction_run_id="run1",
        ),
        EvaluationCandidate(
            candidate_local_id="p2",
            entity_type="method",
            name_or_statement="NPE",
            condition_id="C4",
            extraction_run_id="run1",
        ),
        EvaluationCandidate(
            candidate_local_id="p3",
            entity_type="phenomenon",
            name_or_statement="Spurious Phenomenon",
            condition_id="C4",
            extraction_run_id="run1",
        ),
    ]

    res = match_entities_bipartite(gold, preds)
    assert res.true_positives == 2
    assert res.false_positives == 1
    assert res.false_negatives == 0
    assert res.precision == round(2 / 3, 4)
    assert res.recall == 1.0


def test_wilson_score_interval():
    # 0 successes in 12 trials
    lo, hi = calculate_wilson_score_interval(0, 12)
    assert lo == 0.0
    assert 0.20 <= hi <= 0.30  # Rule of three upper bound is ~25%

    # 10 successes in 10 trials
    lo, hi = calculate_wilson_score_interval(10, 10)
    assert lo >= 0.70
    assert hi == 1.0


def test_multilabel_cohens_kappa():
    # Perfect agreement
    r1 = [{"model": 1, "method": 0}, {"model": 0, "method": 1}]
    r2 = [{"model": 1, "method": 0}, {"model": 0, "method": 1}]
    kappa = calculate_multilabel_cohens_kappa(r1, r2, classes=["model", "method"])
    assert kappa["model"] == 1.0
    assert kappa["method"] == 1.0
    assert kappa["macro_average"] == 1.0


def test_full_evaluation_report_and_cluster_bootstrap():
    gold_records = [
        {
            "source_id": "doc1",
            "entities": [
                {"local_id": "g1", "entity_type": "capability", "name": "Simulate Price", "synonyms": []},
                {"local_id": "g2", "entity_type": "model", "name": "Hawkes Process", "synonyms": []},
            ],
        },
        {
            "source_id": "doc2",
            "entities": [
                {"local_id": "g3", "entity_type": "method", "name": "NPE", "synonyms": []},
            ],
        },
    ]

    # Mock extraction bundle for doc1
    cand_doc1 = [
        EvaluationCandidate(
            candidate_local_id="p1",
            entity_type="capability",
            name_or_statement="Simulate Price",
            condition_id="C4",
            extraction_run_id="run1",
            evidence=EvidenceSpan(
                target_field_name="abstract_text",
                target_field_hash="a" * 64,
                verbatim_quote="simulate price",
                start_char=0,
                end_char=14,
                raw_offset_valid=True,
                resolved_offset_valid=True,
            ),
        ),
        EvaluationCandidate(
            candidate_local_id="p2",
            entity_type="model",
            name_or_statement="Hawkes Process",
            condition_id="C4",
            extraction_run_id="run1",
            evidence=EvidenceSpan(
                target_field_name="abstract_text",
                target_field_hash="a" * 64,
                verbatim_quote="Hawkes Process",
                start_char=15,
                end_char=29,
                raw_offset_valid=True,
                resolved_offset_valid=True,
            ),
        ),
    ]

    bundle1 = ExtractionTrioBundle(
        extraction_run_id="run1",
        source_id="doc1",
        version_id="v1",
        raw_artifact=RawExtractionArtifact(
            extraction_run_id="run1",
            source_id="doc1",
            version_id="v1",
            condition_id="C4",
            raw_response_text="{}",
            executed_at="2026-11-15T00:00:00Z",
            model_checkpoint="gemini-2.5-flash",
            prompt_hash="dummy",
            json_parse_status="success",
        ),
        evaluation_candidates=cand_doc1,
        accepted_record=ExtractedKnowledgeRecord(
            source_id="doc1",
            version_id="v1",
            version_hash="a" * 64,
            extraction_run_id="run1",
            content_scope="abstract_only",
            capabilities=[
                CapabilityEntity(
                    local_id="p1",
                    name="Simulate Price",
                    operation="generate",
                    inputs=[CapabilityIO(data_type_l1="unknown", is_unspecified=True)],
                    outputs=[CapabilityIO(data_type_l1="unknown", is_unspecified=True)],
                    evidence=cand_doc1[0].evidence,
                )
            ],
        ),
    )

    # Mock extraction bundle for doc2
    bundle2 = ExtractionTrioBundle(
        extraction_run_id="run1",
        source_id="doc2",
        version_id="v1",
        raw_artifact=RawExtractionArtifact(
            extraction_run_id="run1",
            source_id="doc2",
            version_id="v1",
            condition_id="C4",
            raw_response_text="{}",
            executed_at="2026-11-15T00:00:00Z",
            model_checkpoint="gemini-2.5-flash",
            prompt_hash="dummy",
            json_parse_status="success",
        ),
        evaluation_candidates=[
            EvaluationCandidate(
                candidate_local_id="p3",
                entity_type="method",
                name_or_statement="NPE",
                condition_id="C4",
                extraction_run_id="run1",
            )
        ],
    )

    report = evaluate_run("C4", gold_records, [bundle1, bundle2])
    assert report.overall_f1 == 1.0
    assert report.class_metrics["capability"].f1 == 1.0
    assert report.class_metrics["model"].f1 == 1.0
    assert report.class_metrics["method"].f1 == 1.0

    # Cluster Bootstrap (run 50 iterations for quick unit test)
    ci = run_cluster_bootstrap("C4", gold_records, [bundle1, bundle2], n_iterations=50)
    assert "overall_f1" in ci
    assert ci["overall_f1"].point_estimate == 1.0
