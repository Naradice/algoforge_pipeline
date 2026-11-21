"""Unit tests for BudgetTracker and StandardizationConverter (Step 4 & User Instruction 4)."""

import pytest
import json

from research_pipeline.extractors.budget_tracker import BudgetTracker, BudgetExceededError
from research_pipeline.extractors.converter import StandardizationConverter
from research_pipeline.extractors.raw_artifact import RawExtractionArtifact
from research_pipeline.extractors.llm_extractor import ExtractionService, MockLLMTransport


def test_budget_tracker_reservation_and_settlement():
    tracker = BudgetTracker(max_usd=0.05, max_tokens=10_000)

    # Initial summary
    s0 = tracker.get_summary()
    assert s0["spent_usd"] == 0.0
    assert s0["reserved_usd"] == 0.0

    # Phase 1: Reserve
    cost = tracker.reserve(
        reservation_id="res_01",
        model_id="gemini-2.5-flash",
        estimated_input_tokens=1000,
        max_output_tokens=1000,
    )
    assert cost > 0
    s1 = tracker.get_summary()
    assert s1["reserved_usd"] > 0
    assert s1["spent_usd"] == 0.0

    # Phase 2: Settle
    actual_cost = tracker.settle(
        reservation_id="res_01",
        actual_input_tokens=800,
        actual_output_tokens=300,
    )
    assert actual_cost > 0
    s2 = tracker.get_summary()
    assert s2["reserved_usd"] == 0.0
    assert s2["spent_usd"] == round(actual_cost, 6)
    assert s2["total_calls_settled"] == 1
    assert s2["total_ledger_entries"] == 1

    # Verify per-request ledger
    ledger = tracker.get_ledger()
    assert len(ledger) == 1
    entry = ledger[0]
    assert entry.call_id == "res_01"
    assert entry.input_tokens == 800
    assert entry.output_tokens == 300
    assert entry.total_tokens == 1100
    assert entry.cost_usd == round(actual_cost, 6)
    assert entry.remaining_budget_usd < 0.05

    # Verify table format
    table = tracker.format_ledger_table()
    assert "res_01" in table
    assert "gemini-2.5-flash" in table


def test_budget_tracker_exceeded_hard_stop():
    # Very small budget
    tracker = BudgetTracker(max_usd=0.001, max_tokens=100)

    with pytest.raises(BudgetExceededError, match="budget exceeded"):
        tracker.reserve(
            reservation_id="res_fail",
            model_id="gemini-2.5-pro",
            estimated_input_tokens=2000,
            max_output_tokens=1000,
        )


def test_converter_c1_free_text():
    raw_c1 = (
        "### Phenomenon\n"
        "- Volatility Clustering: periods of high volatility cluster together\n\n"
        "### Model\n"
        "- Minority Game: an agent-based simulation\n"
    )

    artifact = RawExtractionArtifact(
        extraction_run_id="run_c1",
        source_id="src_01",
        version_id="v_01",
        condition_id="C1",
        raw_response_text=raw_c1,
        executed_at="2026-11-15T00:00:00Z",
        model_checkpoint="gemini-2.5-flash",
        prompt_hash="dummy",
        json_parse_status="not_applicable",
    )

    candidates, success, reason = StandardizationConverter.convert_c1_free_text(artifact)
    assert success is True
    assert len(candidates) == 2
    types = [c.entity_type for c in candidates]
    assert "phenomenon" in types
    assert "model" in types
    assert candidates[0].condition_id == "C1"
    assert candidates[0].evidence is None


def test_converter_c1_failure_reporting():
    # Malformed text without headers
    artifact = RawExtractionArtifact(
        extraction_run_id="run_c1_fail",
        source_id="src_01",
        version_id="v_01",
        condition_id="C1",
        raw_response_text="I am an AI and I cannot extract anything from this text.",
        executed_at="2026-11-15T00:00:00Z",
        model_checkpoint="gemini-2.5-flash",
        prompt_hash="dummy",
        json_parse_status="not_applicable",
    )

    candidates, success, reason = StandardizationConverter.convert_c1_free_text(artifact)
    assert success is False
    assert len(candidates) == 0
    assert "Failed to parse" in reason


def test_extraction_service_c0_and_llm():
    tracker = BudgetTracker()
    service = ExtractionService(budget_tracker=tracker)

    canonical_fields = {
        "abstract_text": "We investigate Hawkes Process and Minority Game to model Volatility Clustering using Neural Posterior Estimation.",
    }

    # C0 Rule-based execution
    artifact_c0 = service.run_c0_rule_based("src_01", "v_01", canonical_fields, "run_c0")
    assert artifact_c0.condition_id == "C0"
    assert artifact_c0.json_parse_status == "success"
    c0_candidates = StandardizationConverter.convert_c0_rule_based(artifact_c0)
    assert len(c0_candidates) >= 3

    # C4 LLM execution (with mock transport)
    artifact_c4 = service.run_llm_condition(
        condition_id="C4",
        source_id="src_01",
        version_id="v_01",
        canonical_fields=canonical_fields,
        extraction_run_id="run_c4",
        prompt_template="Extract knowledge from {{DOCUMENT_TEXT}}",
        system_instruction="Strict JSON output.",
    )
    assert artifact_c4.condition_id == "C4"
    assert artifact_c4.json_parse_status == "success"
    c4_candidates, rejections = StandardizationConverter.convert_c3_c4_with_evidence(artifact_c4)
    assert len(c4_candidates) > 0
    assert tracker.state.total_calls_settled == 1
