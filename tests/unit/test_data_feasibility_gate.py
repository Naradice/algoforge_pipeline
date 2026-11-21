"""Unit tests for HDE Step H2-D3 Data Feasibility & Semantic Fidelity Gate."""

import json
from pathlib import Path
import pytest
from research_pipeline.discovery.models import HypothesisCandidate
from research_pipeline.discovery.data_gate_models import ProxySpecification
from research_pipeline.discovery.data_feasibility_gate import evaluate_data_and_semantic_gate


@pytest.fixture
def conformance_cases():
    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "gold_standard" / "h2_d3_conformance_cases.json"
    )
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)


def test_h2_d3_specification_conformance_16_cases(conformance_cases):
    """Verify 100% exact match across all 16 specification conformance cases."""
    assert len(conformance_cases) == 16

    for item in conformance_cases:
        cid = item["case_id"]
        cand = HypothesisCandidate(**item["candidate"])
        gold = item["gold_judgments"]
        proxy_spec = ProxySpecification(**item["proxy_spec"]) if item.get("proxy_spec") else None

        res = evaluate_data_and_semantic_gate(cand, proxy_spec=proxy_spec, case_id=cid)

        assert res.data_feasibility == gold["data_feasibility"], f"Data feasibility mismatch in {cid}"
        assert res.semantic_fidelity == gold["semantic_fidelity"], f"Semantic fidelity mismatch in {cid}"
        assert res.estimation_spec == gold["estimation_spec"], f"Estimation spec mismatch in {cid}"
        assert res.identifiability == gold["identifiability"], f"Identifiability mismatch in {cid}"
        assert res.temporal_validity == gold["temporal_validity"], f"Temporal validity mismatch in {cid}"
        assert res.gate_decision == gold["gate_decision"], f"Gate decision mismatch in {cid}"


def test_gate_blocks_lookahead_leak_while_accepting_supervised_label():
    """Verify that forward target in label is accepted while forward close in features is blocked."""
    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "gold_standard" / "h2_d3_conformance_cases.json"
    )
    with open(fixture_path, encoding="utf-8") as f:
        cases = {c["case_id"]: c for c in json.load(f)}

    # Case 08: Supervised target Y_t = r_{t+1} -> PASS
    c08 = cases["conf_08_supervised_label_sound"]
    cand08 = HypothesisCandidate(**c08["candidate"])
    res08 = evaluate_data_and_semantic_gate(cand08, case_id="conf_08")
    assert res08.temporal_validity == "VALID"
    assert res08.gate_decision == "PASS"

    # Case 09: Feature leak X_t contains C_{t+1} -> BLOCK
    c09 = cases["conf_09_feature_leak_block"]
    cand09 = HypothesisCandidate(**c09["candidate"])
    res09 = evaluate_data_and_semantic_gate(cand09, case_id="conf_09")
    assert res09.temporal_validity == "LOOKAHEAD_LEAK"
    assert res09.semantic_fidelity == "INVALID"
    assert res09.gate_decision == "BLOCK"


def test_gate_orthogonality_unsupported_data_with_exact_semantics():
    """Verify that L2 order book imbalance is UNSUPPORTED x EXACT -> BLOCK (not INVALID)."""
    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "gold_standard" / "h2_d3_conformance_cases.json"
    )
    with open(fixture_path, encoding="utf-8") as f:
        cases = {c["case_id"]: c for c in json.load(f)}

    c05 = cases["conf_05_l2_orderbook_block"]
    cand05 = HypothesisCandidate(**c05["candidate"])
    res05 = evaluate_data_and_semantic_gate(cand05, case_id="conf_05")
    assert res05.data_feasibility == "UNSUPPORTED"
    assert res05.semantic_fidelity == "EXACT"
    assert res05.gate_decision == "BLOCK"
