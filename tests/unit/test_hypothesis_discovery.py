import pytest
from pathlib import Path

from research_pipeline.storage.knowledge_graph import KnowledgeGraphStore
from research_pipeline.extractors.models import CanonicalEntity, RelationEdge, RelationEvidence
from research_pipeline.discovery.models import (
    GraphPath,
    PathHop,
    HypothesisCandidate,
    AlgoForgeBacktestSpec,
)
from research_pipeline.discovery.path_finder import MultiHopPathFinder
from research_pipeline.discovery.hypothesis_generator import (
    TASKS,
    build_task_prompt,
)
from research_pipeline.discovery.evaluator import evaluate_hypothesis_blind


@pytest.fixture
def mock_kg(tmp_path: Path):
    db_file = tmp_path / "test_h2_kg.sqlite"
    store = KnowledgeGraphStore(db_file)

    # 3 entities: Tool -> Model -> Phenomenon
    e1 = CanonicalEntity(entity_id="soft_cvxpy", canonical_name="CVXPY", entity_type="software", origin_type="extracted")
    e2 = CanonicalEntity(entity_id="mod_hjb", canonical_name="HJB Equation", entity_type="model", origin_type="extracted")
    e3 = CanonicalEntity(entity_id="phe_crash", canonical_name="Flash Crash", entity_type="phenomenon", origin_type="extracted")
    store.insert_entity(e1)
    store.insert_entity(e2)
    store.insert_entity(e3)

    edge1 = RelationEdge(
        edge_id="soft_cvxpy->APPLIES_TO->mod_hjb",
        src_entity_id="soft_cvxpy",
        dst_entity_id="mod_hjb",
        relation_type="APPLIES_TO",
        origin_type="text_supported",
        evidences=[
            RelationEvidence(
                evidence_id="ev_01",
                edge_id="soft_cvxpy->APPLIES_TO->mod_hjb",
                source_id="paper_1",
                version_hash="a" * 64,
                verbatim_quote="CVXPY applies to HJB",
                start_char=0,
                end_char=20,
            )
        ]
    )
    edge2 = RelationEdge(
        edge_id="mod_hjb->EXPLAINS->phe_crash",
        src_entity_id="mod_hjb",
        dst_entity_id="phe_crash",
        relation_type="EXPLAINS",
        origin_type="text_supported",
        evidences=[
            RelationEvidence(
                evidence_id="ev_02",
                edge_id="mod_hjb->EXPLAINS->phe_crash",
                source_id="paper_2",
                version_hash="b" * 64,
                verbatim_quote="HJB explains flash crashes",
                start_char=0,
                end_char=26,
            )
        ]
    )
    store.insert_relation_edge(edge1)
    store.insert_relation_edge(edge2)
    return store


def test_path_finder_multi_hop(mock_kg: KnowledgeGraphStore):
    finder = MultiHopPathFinder(mock_kg)
    paths = finder.find_paths(start_entity_type="software", target_entity_type="phenomenon", min_hops=2, max_hops=2)
    assert len(paths) == 1
    p = paths[0]
    assert p.source_concept == "CVXPY"
    assert p.target_concept == "Flash Crash"
    assert p.hop_count == 2
    assert len(p.hops) == 2
    assert p.hops[0].relation_type == "APPLIES_TO"
    assert p.hops[1].relation_type == "EXPLAINS"


def test_hypothesis_generator_tasks_and_prompts():
    assert len(TASKS) == 3
    for tid, task in TASKS.items():
        assert len(task.doc_ids) == 3
        assert len(task.facts_nl) >= 3
        assert len(task.facts_kg) >= 3

        mock_docs = [{"source_id": sid, "source_type": "paper", "target_fields": {"abstract_text": "sample text"}} for sid in task.doc_ids]
        
        p0 = build_task_prompt(task, "H2-C0", mock_docs)
        p1 = build_task_prompt(task, "H2-C1", mock_docs)
        p2 = build_task_prompt(task, "H2-C2", mock_docs)
        p3 = build_task_prompt(task, "H2-C3", mock_docs)

        assert "TARGET GOAL:" in p0
        assert "PRIMARY LITERATURE SOURCES:" in p0
        assert "EXTRACTED DOMAIN FACTS (Unstructured Natural Language)" in p1
        assert "STRUCTURED KNOWLEDGE GRAPH PATHS (Relational Graph Structure)" in p2
        assert "STRUCTURED KNOWLEDGE GRAPH PATHS & VERBATIM EVIDENCE" in p3


def test_evaluator_rubric_and_blinding():
    cand = HypothesisCandidate(
        hypothesis_id="hyp_001",
        condition_id="H2-C2",
        task_id="task_1",
        title="Hawkes Regime-Adaptive Execution",
        target_phenomenon="Flash crash liquidation cascades in foreign exchange limit order books",
        proposed_mechanism="Calibrate multivariate Hawkes self-excitation parameters in real time to detect branching cascades",
        analogical_source="Point process branching models from seismology and high-frequency equity trading",
        claim="Detecting endogenous order cancellation feedback prevents execution slippage during rapid cascade regimes",
        grounding_rationale="Derived from multivariate Hawkes point process models and invariant risk minimization under distribution shifts",
        testable_falsification_condition="Out-of-sample Sharpe ratio does not exceed GARCH baseline with p-value < 0.05",
        backtest_spec=AlgoForgeBacktestSpec(
            target_asset="USDJPY",
            timeframe="M5",
            feature_engineering="Multivariate Hawkes cross-excitation intensity lambda(t) and order flow imbalance",
            entry_condition="Enter short when Hawkes baseline excitation exceeds 95th percentile historical threshold",
            exit_condition="Trailing stop at 1.5x ATR or 30-minute time-stop",
            risk_management="Volatility-parity position sizing capped at 2% portfolio NAV",
            benchmark_baseline="Standard TWAP liquidation baseline",
            falsification_metric="T-statistic < 2.0 on risk-adjusted alpha",
        )
    )
    res = evaluate_hypothesis_blind(cand)
    assert not res.is_disqualified
    assert res.testability_score >= 0.8
    assert res.operational_concreteness >= 0.9
    assert res.grounding_validity_score >= 0.7
    assert res.novelty_score >= 0.7
    assert res.overall_score >= 0.7


def test_evaluator_disqualification_gate():
    # Candidate with unfalsifiable tautology
    cand_bad = HypothesisCandidate(
        hypothesis_id="hyp_bad",
        condition_id="H2-C0",
        task_id="task_1",
        title="Vague Profit Hypothesis",
        target_phenomenon="Price changes",
        proposed_mechanism="Buy low and sell high using market signals",
        analogical_source="Trading intuition",
        claim="This will produce positive returns consistently",
        grounding_rationale="Common sense and basic charting",
        testable_falsification_condition="The strategy fails to make a profit",  # No quantitative metric or threshold
        backtest_spec=AlgoForgeBacktestSpec(
            target_asset="USDJPY",
            timeframe="M5",
            feature_engineering="Moving average",
            entry_condition="Buy when price rises",
            exit_condition="Sell when price falls",
            risk_management="None",
            benchmark_baseline="Buy and hold",
            falsification_metric="Negative return",
        )
    )
    res = evaluate_hypothesis_blind(cand_bad)
    assert res.is_disqualified
    assert "Disqualified at Gate A" in (res.evaluator_feedback or "")
    assert res.overall_score == 0.0
