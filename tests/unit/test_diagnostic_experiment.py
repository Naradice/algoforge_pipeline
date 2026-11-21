import json
import pytest
from pathlib import Path
from research_pipeline.discovery.models import HypothesisCandidate, AlgoForgeBacktestSpec
from research_pipeline.discovery.diagnostic_generator import CONCEPTS, build_diagnostic_prompt
from research_pipeline.discovery.diagnostic_gold import GOLD_SPECIFICATIONS
from research_pipeline.discovery.diagnostic_evaluator import evaluate_diagnostic_hypothesis


def test_concepts_fact_symmetry():
    """Verify that C0 (natural language) and C1 (KG) have exactly identical facts."""
    assert len(CONCEPTS) == 4
    for cid, concept in CONCEPTS.items():
        assert len(concept.facts_nl) == len(concept.facts_kg), f"Mismatch in count for {cid}"
        assert len(concept.facts_nl) >= 2


def test_evaluator_catches_almgren_chriss_math_misconception():
    """Verify that confusing inventory trajectory with price trajectory triggers E_math."""
    cand = HypothesisCandidate(
        hypothesis_id="test_cand_01",
        condition_id="D1-C1",
        task_id="d1_almgren_chriss",
        title="Flawed Execution Strategy",
        target_phenomenon="Slippage",
        proposed_mechanism="Almgren-Chriss liquidation",
        analogical_source="Almgren-Chriss",
        claim="Minimizes cost",
        grounding_rationale="Grounded in optimal liquidation",
        testable_falsification_condition="Sharpe ratio < 1.0",
        backtest_spec=AlgoForgeBacktestSpec(
            target_asset="USDJPY",
            timeframe="M5",
            feature_engineering="Transient price impact",
            entry_condition="Initiate liquidation when the price deviates from the optimal trajectory by 10 pips",
            exit_condition="Cease liquidation when target volume is reached",
            risk_management="Stop-loss 1%",
            benchmark_baseline="TWAP",
            falsification_metric="Sharpe < 1.0",
        )
    )
    res = evaluate_diagnostic_hypothesis(cand, "almgren_chriss")
    assert res.e_math is True
    assert "conflated with spot price trajectory" in res.error_evidence.get("E_math", "")
    assert res.is_operable is False


def test_evaluator_catches_almgren_chriss_role_misconception():
    """Verify that using Almgren-Chriss for directional entry triggers E_role."""
    cand = HypothesisCandidate(
        hypothesis_id="test_cand_02",
        condition_id="D1-C0",
        task_id="d1_almgren_chriss",
        title="Almgren Alpha Strategy",
        target_phenomenon="Trend prediction",
        proposed_mechanism="Predicting price movements using liquidation slopes",
        analogical_source="Almgren-Chriss",
        claim="Generates excess returns",
        grounding_rationale="Literature shows liquidation trajectories balance risk",
        testable_falsification_condition="Sharpe ratio fails to exceed 1.5",
        backtest_spec=AlgoForgeBacktestSpec(
            target_asset="USDJPY",
            timeframe="M5",
            feature_engineering="Optimal liquidation trajectory slope",
            entry_condition="Enter a trade when the predicted optimal liquidation trajectory indicates a favorable price movement",
            exit_condition="Exit trade on trailing stop",
            risk_management="Max drawdown 2%",
            benchmark_baseline="Moving average",
            falsification_metric="Sharpe < 1.5",
        )
    )
    res = evaluate_diagnostic_hypothesis(cand, "almgren_chriss")
    assert res.e_role is True
    assert "repurposed as speculative directional trade entry signal" in res.error_evidence.get("E_role", "")
    assert res.is_operable is False


def test_evaluator_clean_candidate():
    """Verify that a correctly formulated hypothesis passes with all 5 flags False under M5 OHLCV contract."""
    cand = HypothesisCandidate(
        hypothesis_id="test_cand_clean",
        condition_id="D1-C2",
        task_id="d1_almgren_chriss",
        title="Adaptive Almgren-Chriss Liquidation Engine",
        target_phenomenon="Execution slippage under transient market impact",
        proposed_mechanism="Liquidate target inventory X_0 along optimal discrete trajectory x_k",
        analogical_source="Almgren-Chriss Transient Impact Model",
        claim="Following discrete schedule x_k reduces total execution slippage compared to TWAP",
        grounding_rationale="Grounded in optimal control minimizing expected cost and variance",
        testable_falsification_condition="Total execution cost fails to achieve 15% reduction vs TWAP with p < 0.05",
        backtest_spec=AlgoForgeBacktestSpec(
            target_asset="USDJPY",
            timeframe="M5",
            feature_engineering="Historical Garman-Klass volatility from 20-day M5 OHLC bars and linear temporary impact parameter estimated from in-sample regression",
            entry_condition="Initiate scheduled liquidation of initial quantity X_0 = 1,000,000 USDJPY units at market open t_0 under fixed horizon T = 12 M5 intervals",
            exit_condition="Liquidate units according to schedule; complete execution when remaining inventory x_k reaches zero at terminal step K = 12",
            risk_management="Dynamic participation rate ceiling capped at 10% of 5-minute volume",
            benchmark_baseline="Standard TWAP execution baseline",
            falsification_metric="Execution slippage exceeds TWAP baseline",
        )
    )
    res = evaluate_diagnostic_hypothesis(cand, "almgren_chriss")
    assert res.e_role is False
    assert res.e_math is False
    assert res.e_data is False
    assert res.e_leap is False
    assert res.e_inop is False
    assert res.feasibility_status == "SUPPORTED"
    assert res.is_operable is True


def test_dev36_gold_benchmark():
    """Run full Dev36 benchmark to enforce 100% exact match and 0 UNKNOWN->SUPPORTED promotions."""
    fix_path = Path(__file__).resolve().parent.parent / "fixtures" / "gold_standard" / "h2_d2b_dev36_gold.json"
    with open(fix_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for item in cases:
        concept_id = item["concept_id"]
        gold_flags = item["gold_flags"]
        gold_feas = item["gold_feasibility_status"]
        cand = HypothesisCandidate.model_validate(item["candidate"])
        res = evaluate_diagnostic_hypothesis(cand, concept_id)

        pred_flags = {
            "E_role": int(res.e_role),
            "E_math": int(res.e_math),
            "E_data": int(res.e_data),
            "E_leap": int(res.e_leap),
            "E_inop": int(res.e_inop),
        }
        assert pred_flags == gold_flags, f"Mismatch in flags for {item['spec_id']}: {pred_flags} vs {gold_flags}"
        assert res.feasibility_status == gold_feas, f"Mismatch in feasibility for {item['spec_id']}: {res.feasibility_status} vs {gold_feas}"
