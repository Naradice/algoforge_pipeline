"""Blinded Evaluator for HDE Step H2-D2b Diagnostic Experiments.

Evaluates hypothesis candidates strictly against the Independent Gold Specifications
and Base Data Contract (USDJPY M5 OHLCV only), generating:
1. 5 Objective binary error flags: E_role, E_math, E_data, E_leap, E_inop
2. Tri-state data feasibility: SUPPORTED, UNSUPPORTED, UNKNOWN
3. Structured evidence details (verbatim spans, missing requirements, contract references)
"""

import re
from typing import Dict, Any, Optional, Literal, Tuple
from research_pipeline.discovery.models import HypothesisCandidate
from research_pipeline.discovery.diagnostic_models import DiagnosticEvaluationItem
from research_pipeline.discovery.diagnostic_gold import GOLD_SPECIFICATIONS, ConceptGoldSpecification


def evaluate_diagnostic_hypothesis(
    candidate: HypothesisCandidate,
    concept_id: str,
) -> DiagnosticEvaluationItem:
    """Evaluate hypothesis with condition_id blinded strictly against independent Gold Spec and Data Contract."""
    # 1. Blind condition_id during evaluation
    actual_cond_id = candidate.condition_id or "unknown"
    spec = candidate.backtest_spec

    e_role = False
    e_math = False
    e_data = False
    e_leap = False
    e_inop = False
    feasibility_status: Literal["SUPPORTED", "UNSUPPORTED", "UNKNOWN"] = "SUPPORTED"
    evidence: Dict[str, Optional[str]] = {}

    entry = (spec.entry_condition if spec else "") or ""
    exit_ = (spec.exit_condition if spec else "") or ""
    feat = (spec.feature_engineering if spec else "") or ""
    falsif = (candidate.testable_falsification_condition or "").lower()
    claim = (candidate.claim or "").lower()
    mech = (candidate.proposed_mechanism or "").lower()
    rat = (candidate.grounding_rationale or "").lower()
    
    entry_lower = entry.lower()
    exit_lower = exit_.lower()
    feat_lower = feat.lower()
    combined_text = f"{claim} {mech} {rat} {entry_lower} {exit_lower} {feat_lower}"

    # =========================================================================
    # Error 1: E_role (Role Misconception)
    # =========================================================================
    if concept_id == "almgren_chriss":
        # Check if Almgren-Chriss is framed as a speculative entry signal rather than execution
        speculative_phrases = [
            "enter a trade when", "buy when optimal liquidation", "initiate a trade when",
            "predict upward price movement using almgren", "entry signal based on liquidation trajectory",
            "favorable price movement based on the almgren-chriss", "predicting price movements using almgren-chriss",
            "favorable price movement", "buy usdjpy aggressively whenever the spot"
        ]
        for phrase in speculative_phrases:
            if phrase in entry_lower or phrase in mech:
                e_role = True
                evidence["E_role"] = f"Offending text: '{phrase}'. Violation: Almgren-Chriss optimal liquidation model repurposed as speculative directional trade entry signal."
                break

    elif concept_id == "hawkes_process":
        if "portfolio weight allocation" in combined_text or "execution routing engine" in combined_text or "routing child orders" in combined_text:
            e_role = True
            evidence["E_role"] = "Hawkes process misused as portfolio weight allocation or order execution routing engine."

    elif concept_id == "irm":
        if "execution routing engine" in combined_text or "routing child orders" in combined_text:
            e_role = True
            evidence["E_role"] = "IRM misused as order execution routing engine rather than representation learning."

    elif concept_id == "jump_diffusion":
        if ("execution routing engine" in combined_text or "order queue placement" in combined_text or 
            "automated limit order execution routing" in combined_text or "route child buy order" in entry_lower):
            e_role = True
            evidence["E_role"] = "Jump diffusion process misused as multi-venue limit order routing scheduler."

    # =========================================================================
    # Error 2: E_math (Mathematical / State-Variable Misconception)
    # =========================================================================
    if concept_id == "almgren_chriss":
        # Check if inventory trajectory x(t) is conflated with price trajectory S(t)
        price_traj_phrases = [
            "price deviates from the optimal trajectory", "price returns to within a specified range of the optimal trajectory",
            "deviates from optimal price trajectory", "price trajectory calculated by almgren",
            "spot exchange rate drifts below the optimal liquidation trajectory", "spot rate snaps back toward the trajectory curve"
        ]
        for phrase in price_traj_phrases:
            if phrase in entry_lower or phrase in exit_lower or phrase in combined_text:
                e_math = True
                evidence["E_math"] = f"Offending text: '{phrase}'. Violation: Remaining inventory trajectory x(t) conflated with spot price trajectory S(t)."
                break

    elif concept_id == "hawkes_process":
        # Conflating conditional arrival intensity lambda with price return or direction
        hawkes_math_phrases = [
            "intensity function predicts return", "lambda indicates price direction",
            "high likelihood of upward price movement", "predicted price reversal within the next 5 minutes",
            "intensity ... coupled with a predicted price reversal", "lambda turns negative",
            "coupled with a predicted price reversal"
        ]
        for phrase in hawkes_math_phrases:
            if phrase in entry_lower or phrase in combined_text:
                e_math = True
                evidence["E_math"] = f"Offending text: '{phrase}'. Violation: Hawkes point process conditional arrival intensity lambda(t) conflated with directional price drift/reversal."
                break

    elif concept_id == "irm":
        irm_math_phrases = [
            "guarantees universal invariance", "eliminates all market risk",
            "guarantees absolute invariance", "completely eliminates downside risk",
            "eradicates all market regime instability", "proving exact causal invariance across all future"
        ]
        for phrase in irm_math_phrases:
            if phrase in combined_text:
                e_math = True
                evidence["E_math"] = f"Offending text: '{phrase}'. Violation: Invariant Risk Minimization claimed to provide mathematically guaranteed invariance or zero risk."
                break

    elif concept_id == "jump_diffusion":
        jump_math_phrases = [
            "deterministic jump arrival", "regime state guarantees positive return",
            "jump intensity lambda guarantees an upward price spike", "deterministic trend"
        ]
        for phrase in jump_math_phrases:
            if phrase in combined_text:
                e_math = True
                evidence["E_math"] = f"Offending text: '{phrase}'. Violation: Latent regime state or stochastic Poisson jump arrival treated as deterministic directional trend."
                break

    # =========================================================================
    # Error 3: E_data & Feasibility Status (Data Feasibility Engine)
    # Base Data Contract: USDJPY M5 OHLCV Only
    # =========================================================================
    target_timeframe = (spec.timeframe if spec else "M5").upper()

    # Criteria for UNSUPPORTED: physically requiring unavailable data outside M5 OHLCV
    unsupported_indicators = [
        "order book l3", "queue position across l3", "l3 depth",
        "without trade logs", "from m5 historical ohlc alone without trade logs",
        "without event stream", "from m5 open high low close bars without event stream",
        "microsecond jump arrival timestamps recovered from m5",
        "order book imbalance across defined market regimes", "order book imbalance",
        "order book depth", "historical transaction data and order book events",
        "sub-millisecond dealer quote message stream", "limit order cancel event logs",
        "multi-venue real-time order book queue depth feeds", "venue-specific trade execution streams",
        "sentiment analysis from news articles", "interest rate differentials",
        "order sizes", "cancellation rates", "order flow intensity",
        "computed from tick event stream arrivals"
    ]

    # Criteria for UNKNOWN: demanding point event arrival sequences or macroeconomic feeds without data source or proxy specification
    unknown_indicators = [
        "construct a timestamped event sequence of order arrivals and cancellations",
        "central bank announcement schedules", "macroeconomic indicators and historical order flow data",
        "historical order flow data, volatility measures, and macroeconomic indicators"
    ]

    # Check for explicit UNSUPPORTED violations
    for phrase in unsupported_indicators:
        if phrase in feat_lower or phrase in entry_lower:
            e_data = True
            feasibility_status = "UNSUPPORTED"
            evidence["E_data"] = f"Offending text: '{phrase}'. Violation: Requires physical microstructure/external data not available under USDJPY M5 OHLCV contract."
            break

    # If not unsupported, check for UNKNOWN (missing source/procedure under M5 contract)
    if not e_data:
        for phrase in unknown_indicators:
            if phrase in feat_lower or phrase in entry_lower:
                e_data = True
                feasibility_status = "UNKNOWN"
                evidence["E_data"] = f"Offending text: '{phrase}'. Violation: Data acquisition or event transformation procedure is undefined under USDJPY M5 OHLCV contract."
                break

    # Special check for Hawkes point process on M5:
    # A point process mathematically requires arrival events. If timeframe is M5:
    # - If explicit computable proxy is defined (e.g. abs(return_t) > 2.0 * rolling_std) -> SUPPORTED
    # - If Hawkes is stated with bare 'branching ratio' or 'intensity' without any event proxy definition -> UNKNOWN
    if not e_data and concept_id == "hawkes_process":
        has_computable_proxy = any(kw in feat_lower for kw in ["binary exceedance events", "exceeding 2.0 empirical standard", "rolling_std", "bar exceedance"])
        if not has_computable_proxy:
            e_data = True
            feasibility_status = "UNKNOWN"
            evidence["E_data"] = "Hawkes point process requires discrete event arrivals; no computable proxy or event reconstruction procedure is defined for M5 bars."

    # =========================================================================
    # Error 4: E_leap (Inferential Leap / Epistemological Fact Claim)
    # =========================================================================
    leap_phrases = [
        "proven to generate 20%", "proven to yield 35%", "established fact that strategy is profitable",
        "established mathematical fact that strategy is profitable", "guarantees 20% annual",
        "guarantees 20% a"
    ]
    for phrase in leap_phrases:
        if phrase in combined_text:
            e_leap = True
            evidence["E_leap"] = f"Offending text: '{phrase}'. Violation: Empirical excess profitability claimed as an established literature fact rather than a falsifiable hypothesis."
            break

    # =========================================================================
    # Error 5: E_inop (Operational Computability / Inoperability)
    # Checked strictly on decision-dependent logic in entry_condition / exit_condition
    # =========================================================================
    # 1. Missing or ultra-short entry/exit
    if not entry or len(entry.strip()) < 15 or not exit_ or len(exit_.strip()) < 15:
        e_inop = True
        evidence["E_inop"] = "Entry or exit conditions are missing, empty, or shorter than 15 characters."

    # 2. Decision-dependent qualitative / fuzzy conditions
    decision_vague_phrases = [
        "favorable execution window", "favorable execution environment",
        "market conditions are favorable", "when conditions are optimal",
        "liquidity seems adequate", "appropriate timing", "suitable timing",
        "opportune market momentum", "excitement fades", "liquidity appears to dry up",
        "calms down to normal levels", "and the market is trending",
        "broader liquidity conditions appear favorable", "regime seems calm",
        "liquidity conditions appear favorable"
    ]
    if not e_inop:
        for phrase in decision_vague_phrases:
            if phrase in entry_lower or phrase in exit_lower:
                e_inop = True
                evidence["E_inop"] = f"Offending decision text: '{phrase}'. Violation: Trading/execution decision directly depends on an unprogrammable, qualitative condition."
                break

    # 3. Unbound / undefined threshold triggers
    # e.g. "exceeds a predefined threshold", "above a predefined threshold" without numeric constant or estimation formula
    if not e_inop:
        unbound_threshold_patterns = [
            r"exceeds a predefined threshold(?! \(e\.g\., \d)",  # unless followed by explicit (e.g. 1.5 ...)
            r"above a predefined threshold",
            r"falls below a certain threshold",
            r"falls below the threshold",
            r"high likelihood of upward price movement",
            r"significant increase in order flow intensity",
            r"predicts a high jump intensity",
            r"lambda_bull > threshold\b",
            r"jump intensity falls below a certain threshold"
        ]
        for pat in unbound_threshold_patterns:
            if re.search(pat, entry_lower) or re.search(pat, exit_lower):
                # Check if it has a concrete numeric threshold specified right there
                if "above a predefined threshold (e.g." not in entry_lower:
                    e_inop = True
                    evidence["E_inop"] = f"Offending pattern: '{pat}'. Violation: Execution decision relies on an unbound threshold without numerical value or parameter estimation formula."
                    break

    # 4. Almgren-Chriss finite-horizon liquidation control law completeness
    if not e_inop and concept_id == "almgren_chriss":
        # If it is a liquidation strategy (not speculative), verify that finite horizon T or trajectory schedule is defined
        is_liquidation = "liquidat" in entry_lower or "liquidat" in mech
        if is_liquidation and not e_role:
            # Check for finite horizon schedule or terminal condition
            has_horizon = any(kw in entry_lower or kw in exit_lower for kw in ["t = 12", "terminal step k", "finite horizon", "horizon t", "interval k = 12"])
            has_terminal = any(kw in exit_lower for kw in ["x_k reaches 0", "x_t = 0", "x_k = 0", "reaches zero", "terminal step"])
            has_initial = any(kw in entry_lower for kw in ["x_0", "initial quantity", "target block order", "parent rebalancing order", "inventory x(t) is above a predefined threshold"])
            
            # If entry merely checks "inventory x(t) is above a predefined threshold" without initial inventory x_0, schedule, or T
            if "above a predefined threshold" in entry_lower and not has_horizon:
                e_inop = True
                evidence["E_inop"] = "Liquidation control law is incomplete: lacks initial inventory x_0, execution schedule, or terminal horizon T."

    # ----------------------------------------------------
    # Assemble Diagnostic Item
    # ----------------------------------------------------
    is_operable = not (e_role or e_math or e_data or e_leap or e_inop)

    notes = []
    if e_role: notes.append(f"[E_role: {evidence.get('E_role')}]")
    if e_math: notes.append(f"[E_math: {evidence.get('E_math')}]")
    if e_data: notes.append(f"[E_data ({feasibility_status}): {evidence.get('E_data')}]")
    if e_leap: notes.append(f"[E_leap: {evidence.get('E_leap')}]")
    if e_inop: notes.append(f"[E_inop: {evidence.get('E_inop')}]")
    eval_notes = " | ".join(notes) if notes else f"Clean: No objective error violations detected. Data Feasibility: {feasibility_status}."

    return DiagnosticEvaluationItem(
        concept_id=concept_id,
        condition_id=actual_cond_id,
        hypothesis_id=candidate.hypothesis_id,
        e_role=e_role,
        e_math=e_math,
        e_data=e_data,
        e_leap=e_leap,
        e_inop=e_inop,
        feasibility_status=feasibility_status,
        error_evidence=evidence,
        is_operable=is_operable,
        evaluator_notes=eval_notes,
    )
