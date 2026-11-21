"""Standardized Evaluation Rubric for Step H2 Generated Hypotheses with Condition Blinding and Disqualification Gates."""

from typing import Dict, Any, List, Optional
import re
from research_pipeline.discovery.models import HypothesisCandidate, HypothesisEvaluationResult


def evaluate_hypothesis_blind(
    candidate: HypothesisCandidate,
    reference_facts: Optional[List[str]] = None,
) -> HypothesisEvaluationResult:
    """Evaluate a hypothesis candidate with Condition ID blinded, applying DQ gates and independent T/G/N/O scoring."""
    
    # 0. Condition ID Blinding Verification
    blinded_candidate = candidate.model_copy()
    blinded_candidate.condition_id = None  # Ensure evaluator cannot condition on C0-C3 label

    cand = blinded_candidate
    spec = cand.backtest_spec

    # 1. Disqualification (DQ) Gate Checks
    # Gate A: Unfalsifiable or missing quantitative threshold
    falsif = (cand.testable_falsification_condition or "").strip().lower()
    has_metric = any(kw in falsif for kw in [
        "sharpe", "drawdown", "p-value", "p <", "p<", "t-stat", "basis point", "bps",
        "slippage", "return", "ic", "information coefficient", "%", "alpha"
    ])
    has_operator = any(op in falsif for op in ["<", ">", "<=", ">=", "=", "exceed", "fall", "degrade", "fail"])
    
    if len(falsif) < 15 or not has_metric or not has_operator:
        return HypothesisEvaluationResult(
            hypothesis_id=cand.hypothesis_id,
            is_disqualified=True,
            disqualification_reason="DQ: Testable falsification condition lacks quantitative metric threshold or clear operational falsification boundary.",
            testability_score=0.1,
            grounding_validity_score=0.2,
            novelty_score=0.2,
            operational_concreteness=0.2,
            overall_score=0.0,
            evaluator_feedback="Disqualified at Gate A: Unfalsifiable or non-quantitative failure criteria."
        )

    # Gate B: Empty or vacuous backtest spec
    if spec is None or not spec.entry_condition or len(spec.entry_condition.strip()) < 10 or not spec.exit_condition or len(spec.exit_condition.strip()) < 10:
        return HypothesisEvaluationResult(
            hypothesis_id=cand.hypothesis_id,
            is_disqualified=True,
            disqualification_reason="DQ: Backtest specification lacks actionable entry or exit logic for AlgoForge execution.",
            testability_score=0.3,
            grounding_validity_score=0.3,
            novelty_score=0.3,
            operational_concreteness=0.0,
            overall_score=0.0,
            evaluator_feedback="Disqualified at Gate B: Incomplete AlgoForge backtest specification."
        )

    # Gate C: Hallucination or contradictory grounding
    rationale = (cand.grounding_rationale or "").strip().lower()
    mech = (cand.proposed_mechanism or "").strip().lower()
    combined_grounding = f"{rationale} {mech}"
    if len(combined_grounding) < 25:
        return HypothesisEvaluationResult(
            hypothesis_id=cand.hypothesis_id,
            is_disqualified=True,
            disqualification_reason="DQ: Insufficient or vacuous grounding rationale supporting the mechanism.",
            testability_score=0.4,
            grounding_validity_score=0.0,
            novelty_score=0.2,
            operational_concreteness=0.4,
            overall_score=0.0,
            evaluator_feedback="Disqualified at Gate C: Vacuous grounding rationale."
        )

    # 2. Independent Multi-Dimensional Scoring
    # Dimension 1: Testability (T, 0.0 ~ 1.0)
    testability = 0.0
    if has_metric:
        testability += 0.4
    if has_operator and len(falsif) > 25:
        testability += 0.3
    if spec and spec.benchmark_baseline and len(spec.benchmark_baseline) > 5:
        testability += 0.15
    if spec and spec.falsification_metric and len(spec.falsification_metric) > 5:
        testability += 0.15
    testability = min(1.0, testability)

    # Dimension 2: Grounding Validity (G, 0.0 ~ 1.0)
    # Grounding is evaluated against reference facts if provided, or semantic coherence of literature mechanism
    grounding = 0.0
    if len(rationale) > 40:
        grounding += 0.3
    if len(cand.proposed_mechanism) > 40:
        grounding += 0.3
    
    # Check alignment with literature concepts
    literature_keywords = [
        "hawkes", "point process", "intensity", "branching", "cascade",
        "invariant risk minimization", "irm", "distribution shift", "order flow",
        "jump diffusion", "regime-switching", "latent", "poisson",
        "deep equilibrium", "implicit layer", "root-finding", "volatility surface",
        "almgren-chriss", "transient impact", "liquidation", "execution",
        "hamilton-jacobi-bellman", "hjb", "pinn", "physics-informed",
        "garch", "arch", "convex optimization", "cvxpy", "quadratic programming"
    ]
    matched_kws = sum(1 for kw in literature_keywords if kw in combined_grounding)
    if matched_kws >= 2:
        grounding += 0.3
    elif matched_kws == 1:
        grounding += 0.15

    if reference_facts:
        ref_text = " ".join(reference_facts).lower()
        if any(kw in ref_text and kw in combined_grounding for kw in literature_keywords):
            grounding += 0.1
    grounding = min(1.0, grounding)

    # Dimension 3: Novelty (N, 0.0 ~ 1.0)
    novelty = 0.0
    # Cross-domain / cross-concept integration
    if cand.analogical_source and len(cand.analogical_source.strip()) > 5:
        novelty += 0.35
    if cand.target_phenomenon and len(cand.target_phenomenon.strip()) > 10:
        novelty += 0.25
    # Synthesis depth: combining 2 or more distinct methodological paradigms
    if matched_kws >= 3:
        novelty += 0.40
    elif matched_kws >= 2:
        novelty += 0.25
    else:
        novelty += 0.10
    novelty = min(1.0, novelty)

    # Dimension 4: Operational Concreteness (O, 0.0 ~ 1.0)
    operational = 0.0
    if spec:
        if spec.target_asset and spec.target_asset.strip():
            operational += 0.10
        if spec.timeframe and spec.timeframe.strip():
            operational += 0.10
        if spec.feature_engineering and len(spec.feature_engineering.strip()) > 15:
            operational += 0.25
        if spec.entry_condition and len(spec.entry_condition.strip()) > 20:
            operational += 0.25
        if spec.exit_condition and len(spec.exit_condition.strip()) > 20:
            operational += 0.15
        if spec.risk_management and len(spec.risk_management.strip()) > 10:
            operational += 0.15
    operational = min(1.0, operational)

    # 3. Secondary Overall Score (Weighted combination)
    overall = round(0.30 * testability + 0.25 * grounding + 0.25 * novelty + 0.20 * operational, 4)

    feedback = (
        f"Testability: {testability:.2f} (falsification={has_metric}), "
        f"Grounding: {grounding:.2f} (matched_concepts={matched_kws}), "
        f"Novelty: {novelty:.2f} (analogical_source='{cand.analogical_source}'), "
        f"AlgoForge Concreteness: {operational:.2f}."
    )

    return HypothesisEvaluationResult(
        hypothesis_id=cand.hypothesis_id,
        is_disqualified=False,
        disqualification_reason=None,
        testability_score=round(testability, 4),
        grounding_validity_score=round(grounding, 4),
        novelty_score=round(novelty, 4),
        operational_concreteness=round(operational, 4),
        overall_score=overall,
        evaluator_feedback=feedback,
    )
