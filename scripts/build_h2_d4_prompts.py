"""Build and audit H2-D4 Prompts and Pre-Freeze Manifest with Strict Semantic Parity.

Adheres strictly to RP-HDE-H2-D4-PLAN-v0.2:
- 4 concepts: almgren_chriss, hawkes_process, irm, jump_diffusion
- 3 conditions: C0 (KG only), C1 (KG + Natural Language IO), C2 (KG + Structured YAML IO)
- Strict Semantic Parity: C1 prose is an exact isomorphic transcription of C2 YAML.
- Pre-freezes SHA256 hashes of evaluators, prompts, and intent criteria before generation.
"""

import json
import hashlib
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures" / "gold_standard"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_PATH = FIXTURES_DIR / "h2_d4_prompts.json"
INTENT_PATH = FIXTURES_DIR / "h2_d4_intent_criteria.json"
MANIFEST_PATH = FIXTURES_DIR / "h2_d4_audit_manifest.json"

EVALUATOR_D2B_PATH = BASE_DIR / "src" / "research_pipeline" / "discovery" / "diagnostic_evaluator.py"
GATE_D3_PATH = BASE_DIR / "src" / "research_pipeline" / "discovery" / "data_feasibility_gate.py"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ==============================================================================
# 1. 4 Concepts Knowledge Content & Isomorphic Capability IO Definitions
# ==============================================================================

CONCEPTS_D4 = {
    "almgren_chriss": {
        "concept_name": "Almgren-Chriss Transient Impact Model",
        "category": "execution_control",
        "kg_triples": [
            "[Almgren-Chriss Transient Impact Model (model)] --(EXPLAINS)--> [Transient Price Impact (phenomenon)]",
            "[Compute Optimal Liquidation Trajectories (capability)] --(PROVIDED_BY)--> [Almgren-Chriss Transient Impact Model (model)]",
            "[Almgren-Chriss Transient Impact Model (model)] --(VARIANT_OF)--> [Almgren-Chriss Liquidation Framework (model)]"
        ],
        "c1_natural_language_io": (
            "Capability IO Specification for Almgren-Chriss:\n"
            "The model computes optimal liquidation trajectories for pre-existing inventory.\n"
            "Input features: Initial parent inventory quantity X_0 (lots to liquidate), target execution horizon T (number of M5 intervals), and risk aversion parameter lambda.\n"
            "Outputs: Discrete remaining inventory trajectory x_k over intervals k in {0, ..., K} where x_0 = X_0 and terminal inventory x_K = 0, along with trade volume schedule v_k = x_{k-1} - x_k.\n"
            "Data Contract: Input data is strictly USDJPY M5 OHLCV bars. Transient impact decay parameters must be estimated via closed-form proxies or in-sample regression from historical M5 bars without assuming external order book depth.\n"
            "Control Variables & Boundary Conditions: Initial state x(0) = X_0, finite horizon T = K intervals, monotonic liquidation schedule v_k >= 0, and exact terminal boundary constraint x_K = 0.\n"
            "Identifiability & Estimation Assumptions: Temporary impact parameter eta is estimated via linear regression of bar returns on volume under full-rank volume variance; inventory trajectory x(t) is an inventory exhaustion schedule, NOT a price forecast S(t).\n"
            "Temporal Rules: Liquidation order v_k is evaluated using confirmed bar close C_{k-1} and executed at next bar open O_k."
        ),
        "c2_structured_yaml_io": (
            "capability_io_specification:\n"
            "  model_name: Almgren-Chriss Transient Impact Model\n"
            "  primary_capability: Compute Optimal Liquidation Trajectories\n"
            "  inputs:\n"
            "    features:\n"
            "      - name: initial_inventory_quantity\n"
            "        symbol: X_0\n"
            "        description: Total lots to liquidate from pre-existing inventory\n"
            "      - name: liquidation_horizon\n"
            "        symbol: T\n"
            "        description: Fixed finite number of M5 intervals K\n"
            "      - name: risk_aversion_parameter\n"
            "        symbol: lambda\n"
            "        description: Weight on execution variance vs expected cost\n"
            "  outputs:\n"
            "    targets:\n"
            "      - name: remaining_inventory_trajectory\n"
            "        symbol: x_k\n"
            "        domain: '[0, X_0]'\n"
            "        description: Scheduled remaining inventory at step k\n"
            "      - name: trading_volume_schedule\n"
            "        symbol: v_k\n"
            "        formula: x_{k-1} - x_k\n"
            "        description: Lots to liquidate in interval k\n"
            "  data_contract:\n"
            "    primary_feed: USDJPY M5 OHLCV bars only\n"
            "    external_feeds: NONE\n"
            "    proxy_requirement: Transient impact parameter eta estimated via in-sample regression on M5 bars without order book depth\n"
            "  control_laws:\n"
            "    variables:\n"
            "      initial_state: x_0 = X_0\n"
            "      terminal_horizon: T = K M5 intervals\n"
            "      monotonicity_condition: v_k >= 0\n"
            "      terminal_boundary_condition: x_K = 0 (complete liquidation)\n"
            "  identifiability:\n"
            "    assumptions:\n"
            "      - Impact parameter eta identified via linear regression under non-zero volume variance\n"
            "      - Trajectory x(t) is strictly an inventory liquidation schedule, NOT a directional price forecast S(t)\n"
            "  temporal_rules:\n"
            "    execution:\n"
            "      decision_time: Confirmed bar close C_{k-1}\n"
            "      execution_time: Next bar open O_k\n"
        )
    },
    "hawkes_process": {
        "concept_name": "Multivariate Hawkes Point Process",
        "category": "point_process",
        "kg_triples": [
            "[Multivariate Hawkes Point Process (model)] --(EXPLAINS)--> [High-Frequency Flash Crash Propagation (phenomenon)]",
            "[Detect Endogenous Cascade Risks (capability)] --(PROVIDED_BY)--> [Multivariate Hawkes Point Process (model)]",
            "[Multivariate Hawkes Point Process (model)] --(VARIANT_OF)--> [Hawkes Point Process (model)]"
        ],
        "c1_natural_language_io": (
            "Capability IO Specification for Hawkes Point Process:\n"
            "The model estimates self-exciting arrival intensities and detects endogenous cascade risks.\n"
            "Input features: Discrete event arrival sequence with timestamps tau_i. Under the USDJPY M5 OHLCV data contract where tick orders are unavailable, arrivals must be defined as an explicit M5 bar return exceedance proxy (such as abs(return_t) > 2.0 * rolling_std).\n"
            "Outputs: Conditional event arrival intensity lambda(t) and self-excitation branching ratio alpha/beta.\n"
            "Data Contract: Input data is strictly USDJPY M5 OHLCV bars with no tick trade logs or order book depth. Native order arrival and cancellation events cannot be directly observed on M5 bars.\n"
            "Control Variables & Boundary Conditions: Hawkes process is a statistical point process estimation model, not an order routing scheduler or speculative price direction forecaster.\n"
            "Identifiability & Estimation Assumptions: Parameters (mu, alpha, beta) are estimated via MLE or moment matching over a fixed lookback window under the strict stationarity stability condition alpha < beta with positive baseline mu > 0; the estimated intensity lambda(t) represents event occurrence frequency, NOT directional price return.\n"
            "Temporal Rules: Event exceedances are confirmed at bar close C_t; position de-risking executes at next bar open O_{t+1}."
        ),
        "c2_structured_yaml_io": (
            "capability_io_specification:\n"
            "  model_name: Multivariate Hawkes Point Process\n"
            "  primary_capability: Detect Endogenous Cascade Risks\n"
            "  inputs:\n"
            "    features:\n"
            "      - name: event_arrival_sequence\n"
            "        symbol: tau_i\n"
            "        proxy_definition: Timestamps of M5 absolute return exceedance abs(return_t) > 2.0 * rolling_std\n"
            "        description: Discrete point process events derived from M5 bar volatility jumps\n"
            "  outputs:\n"
            "    targets:\n"
            "      - name: conditional_intensity\n"
            "        symbol: lambda(t)\n"
            "        description: Instantaneous arrival rate of volatility jump events\n"
            "      - name: branching_ratio\n"
            "        symbol: alpha / beta\n"
            "        description: Expected number of child jump events triggered per parent event\n"
            "  data_contract:\n"
            "    primary_feed: USDJPY M5 OHLCV bars only\n"
            "    external_feeds: NONE\n"
            "    data_limitation: Native tick-level order arrivals and cancellations cannot be directly observed on M5 bars\n"
            "  control_laws:\n"
            "    variables:\n"
            "      model_type: Statistical point process estimation model (not an order routing scheduler or alpha directional signal)\n"
            "  identifiability:\n"
            "    assumptions:\n"
            "      - Parameters (mu, alpha, beta) estimated via MLE over fixed lookback window\n"
            "      - Subcritical stationarity constraint alpha < beta with mu > 0\n"
            "      - Intensity lambda(t) measures jump clustering frequency, NOT expected directional return\n"
            "  temporal_rules:\n"
            "    execution:\n"
            "      decision_time: Confirmed bar close C_t after exceedance evaluation\n"
            "      execution_time: Next bar open O_{t+1}\n"
        )
    },
    "irm": {
        "concept_name": "Invariant Risk Minimization (IRM)",
        "category": "robust_prediction",
        "kg_triples": [
            "[Invariant Risk Minimization (model)] --(ADDRESSES)--> [Out-of-Distribution Regime Shifts (phenomenon)]",
            "[Predict Non-Stationary Order Flow (capability)] --(PROVIDED_BY)--> [Invariant Risk Minimization (model)]",
            "[Invariant Risk Minimization (model)] --(VARIANT_OF)--> [Empirical Risk Minimization (model)]"
        ],
        "c1_natural_language_io": (
            "Capability IO Specification for Invariant Risk Minimization:\n"
            "The model learns invariant feature representations across heterogeneous market environments to mitigate out-of-distribution performance drop.\n"
            "Input features: Feature matrix X_t computed from M5 OHLCV (log returns, Garman-Klass volatility, RSI, and bar-direction signed volume sign(C_t - O_t)*V_t) partitioned into explicit market session environments E = {Asian, European, US}.\n"
            "Outputs: Invariant representation Phi(X_t) and directional classifier prediction probability P(y_{t+1} = 1 | X_t).\n"
            "Data Contract: Input data is strictly USDJPY M5 OHLCV bars only. Native L2/L3 order book imbalance is unavailable under this contract; volume features use bar direction proxy without claiming tick trade signs.\n"
            "Control Variables & Boundary Conditions: Penalty parameter lambda_IRM balances empirical loss across environments against gradient variance penalty; IRM is a predictive feature representation learner, not an execution router.\n"
            "Identifiability & Estimation Assumptions: Invariance is estimated via empirical gradient penalty across session environments under linear classifier assumption; IRM does NOT guarantee zero downside risk or universal causal invariance under arbitrary unobserved interventions.\n"
            "Temporal Rules: Supervisory target label Y_t = return_{t+1} is used strictly during training; decision is evaluated at bar close C_t and executed at next bar open O_{t+1}."
        ),
        "c2_structured_yaml_io": (
            "capability_io_specification:\n"
            "  model_name: Invariant Risk Minimization (IRM)\n"
            "  primary_capability: Learn Invariant Representations Across Regime Shifts\n"
            "  inputs:\n"
            "    features:\n"
            "      - name: market_features\n"
            "        formula: M5 log returns, Garman-Klass volatility, RSI, and signed bar volume sign(C_t - O_t)*V_t\n"
            "      - name: environment_partition\n"
            "        symbol: E\n"
            "        domains: Asian (UTC 0-8), European (UTC 8-16), US (UTC 16-24)\n"
            "  outputs:\n"
            "    targets:\n"
            "      - name: invariant_representation\n"
            "        symbol: Phi(X_t)\n"
            "      - name: directional_probability\n"
            "        symbol: P(y_{t+1} = 1 | X_t)\n"
            "  data_contract:\n"
            "    primary_feed: USDJPY M5 OHLCV bars only\n"
            "    external_feeds: NONE\n"
            "    data_limitation: L2/L3 order book queue imbalance is unavailable; signed volume uses M5 bar direction\n"
            "  control_laws:\n"
            "    variables:\n"
            "      penalty_multiplier: lambda_IRM on gradient norm penalty\n"
            "      model_type: Predictive representation learner (not an order execution routing engine)\n"
            "  identifiability:\n"
            "    assumptions:\n"
            "      - Simultaneous optimality across session environments under linear classifier\n"
            "      - IRM regularizes distribution shift but does NOT guarantee zero risk or universal invariance\n"
            "  temporal_rules:\n"
            "    execution:\n"
            "      target_label: Y_t = return_{t+1} used strictly for training\n"
            "      decision_time: Confirmed bar close C_t\n"
            "      execution_time: Next bar open O_{t+1}\n"
        )
    },
    "jump_diffusion": {
        "concept_name": "Regime-Switching Jump Diffusion Model",
        "category": "state_space_jump",
        "kg_triples": [
            "[Regime-Switching Jump Diffusion Model (model)] --(EXPLAINS)--> [Central Bank Rate Announcement Jumps (phenomenon)]",
            "[Infer Latent Jump Intensities (capability)] --(PROVIDED_BY)--> [Regime-Switching Jump Diffusion Model (model)]",
            "[Regime-Switching Jump Diffusion Model (model)] --(VARIANT_OF)--> [Merton Jump Diffusion Model (model)]"
        ],
        "c1_natural_language_io": (
            "Capability IO Specification for Regime-Switching Jump Diffusion:\n"
            "The model filters continuous Gaussian diffusion from discrete Poisson jumps across latent market regimes.\n"
            "Input features: M5 log-returns r_t, Realized Variance RV_t, and Bipower Variation BV_t computed over a rolling lookback window of 288 M5 bars.\n"
            "Outputs: Jump component test statistic J_t = max(0, RV_t - BV_t), filtered regime state probability P(S_t = tranquil | F_t), and filtered Poisson jump arrival intensity lambda_jump.\n"
            "Data Contract: Input data is strictly USDJPY M5 OHLCV bars without external macro news feeds or microsecond tick arrival logs. Announcement timing is inferred from return bipower jump metrics.\n"
            "Control Variables & Boundary Conditions: Regime-switching jump model estimates state transition probabilities for risk management, not for automated limit order placement into book queues.\n"
            "Identifiability & Estimation Assumptions: Continuous and jump variances are separated via threshold bipower variation; 2-state Markov chain transition matrix is estimated via Hamilton filter under state variance separability constraints sigma_crisis > sigma_tranquil; Poisson jumps are stochastic arrival events, NOT deterministic price surges.\n"
            "Temporal Rules: Bipower jump statistics are computed on completed bar close C_t; position adjustment executes at next bar open O_{t+1}."
        ),
        "c2_structured_yaml_io": (
            "capability_io_specification:\n"
            "  model_name: Regime-Switching Jump Diffusion Model\n"
            "  primary_capability: Infer Latent Jump Intensities and Regimes\n"
            "  inputs:\n"
            "    features:\n"
            "      - name: log_returns\n"
            "        symbol: r_t\n"
            "      - name: realized_variance\n"
            "        symbol: RV_t\n"
            "      - name: bipower_variation\n"
            "        symbol: BV_t\n"
            "        lookback_window: 288 M5 bars\n"
            "  outputs:\n"
            "    targets:\n"
            "      - name: jump_component_statistic\n"
            "        symbol: J_t = max(0, RV_t - BV_t)\n"
            "      - name: filtered_regime_probability\n"
            "        symbol: P(S_t = tranquil | F_t)\n"
            "      - name: filtered_jump_intensity\n"
            "        symbol: lambda_jump\n"
            "  data_contract:\n"
            "    primary_feed: USDJPY M5 OHLCV bars only\n"
            "    external_feeds: NONE\n"
            "    data_limitation: Microsecond jump arrival timestamps and external news feeds are unavailable\n"
            "  control_laws:\n"
            "    variables:\n"
            "      model_type: Regime filtering and tail risk model (not an automated limit order placement engine)\n"
            "  identifiability:\n"
            "    assumptions:\n"
            "      - Mancini threshold bipower variation separates jump from diffusive variance\n"
            "      - Hamilton filter updates regime probabilities under variance separability sigma_crisis > sigma_tranquil\n"
            "      - Jumps are stochastic Poisson arrivals, NOT deterministic directional trends\n"
            "  temporal_rules:\n"
            "    execution:\n"
            "      decision_time: Confirmed bar close C_t after bipower calculation\n"
            "      execution_time: Next bar open O_{t+1}\n"
        )
    }
}

# ==============================================================================
# 2. Intent & Transformation Criteria Definitions
# ==============================================================================

INTENT_CRITERIA = {
    "almgren_chriss": {
        "original_intent_definition": "Formulates optimal liquidation trajectory x_k of pre-existing parent inventory under market risk vs transient impact trade-off.",
        "valid_transformation_definition": "Transforms liquidation to an adaptive pacing or execution variance minimization proxy using M5 historical volatility and linear volume regression, specifying initial X_0, finite horizon T, and terminal condition x_T = 0.",
        "infeasible_recognition_definition": "Explicitly notes that order book depth and instant execution fills are required for true transient impact calibration, and states the limitations under M5 OHLCV.",
        "invalid_substitution_definition": "Repurposes liquidation trajectory as a speculative directional buy/sell entry signal, or conflates remaining inventory x(t) with price S(t)."
    },
    "hawkes_process": {
        "original_intent_definition": "Models endogenous feedback and clustering in order flows (order arrivals and cancellations) in the limit order book.",
        "valid_transformation_definition": "Explicitly formulates an M5 return exceedance proxy for price jump clustering, complete with the 5 proxy requirements (original target, proxy target, transformed hypothesis, permitted interpretations, prohibited interpretations).",
        "infeasible_recognition_definition": "Explicitly states that tick-level order book events cannot be directly observed on M5 bars and that order flow Hawkes cannot be directly estimated natively.",
        "invalid_substitution_definition": "Claims M5 price changes measure true order book cancellations, or treats arrival intensity lambda as directional price return."
    },
    "irm": {
        "original_intent_definition": "Learns invariant feature representations across multiple market environments to mitigate out-of-distribution regime shifts.",
        "valid_transformation_definition": "Defines M5 OHLCV native features partitioned into explicit session environments (Asian, European, US) with verifiable training objectives and bar-direction signed volume.",
        "infeasible_recognition_definition": "Explicitly notes that L2/L3 order book queue depth and microsecond trade signs are unavailable on M5 bars alone.",
        "invalid_substitution_definition": "Demands raw L3 order book imbalance, claims IRM eliminates all market risk, or uses IRM as an order execution routing scheduler."
    },
    "jump_diffusion": {
        "original_intent_definition": "Disentangles diffusive Brownian volatility from discrete Poisson jump intensities across macroeconomic regimes.",
        "valid_transformation_definition": "Calculates computable threshold bipower variation on M5 returns to infer regime states with complete estimation and variance separability specifications.",
        "infeasible_recognition_definition": "Explicitly states that microsecond jump timestamps and external central bank news feeds are unavailable under M5 OHLCV.",
        "invalid_substitution_definition": "Treats Poisson jumps as deterministic upward surges, uses unconfirmed daily close intraday, or repurposes model as multi-venue limit order placement engine."
    }
}


def build_prompt_text(concept_id: str, condition_id: str) -> str:
    """Build standardized prompt text for a given concept and condition."""
    c_data = CONCEPTS_D4[concept_id]
    triples_text = "\n".join(f"- {t}" for t in c_data["kg_triples"])

    base_header = (
        f"Target Concept: {c_data['concept_name']} ({c_data['category']})\n"
        f"Data Contract: USDJPY M5 OHLCV only (no external data feeds, no L2/L3 order book depth, no tick trade logs).\n\n"
        f"Knowledge Graph Relation Paths:\n{triples_text}\n\n"
    )

    if condition_id == "C0":
        prompt = (
            f"{base_header}"
            f"Task: Formulate a single mathematically sound, data-feasible, and testable algorithmic trading hypothesis for USDJPY execution in AlgoForge based strictly on the above Knowledge Graph facts.\n"
        )
    elif condition_id == "C1":
        prompt = (
            f"{base_header}"
            f"Capability IO Information (Natural Language Description):\n{c_data['c1_natural_language_io']}\n\n"
            f"Task: Formulate a single mathematically sound, data-feasible, and testable algorithmic trading hypothesis for USDJPY execution in AlgoForge based on the above Knowledge Graph facts and Capability IO description.\n"
        )
    elif condition_id == "C2":
        prompt = (
            f"{base_header}"
            f"Capability IO Information (Structured Schema):\n```yaml\n{c_data['c2_structured_yaml_io']}```\n\n"
            f"Task: Formulate a single mathematically sound, data-feasible, and testable algorithmic trading hypothesis for USDJPY execution in AlgoForge based on the above Knowledge Graph facts and Structured Capability IO schema.\n"
        )
    else:
        raise ValueError(f"Unknown condition_id: {condition_id}")

    return prompt


def main():
    # 1. Build all 12 prompt texts (4 concepts x 3 conditions)
    prompts_dict = {}
    for cid in CONCEPTS_D4:
        prompts_dict[cid] = {}
        for cond in ["C0", "C1", "C2"]:
            prompts_dict[cid][cond] = build_prompt_text(cid, cond)

    # Save Prompts Fixture
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts_dict, f, indent=2, ensure_ascii=False)
    prompts_sha = sha256_file(PROMPTS_PATH)
    print(f"Prompts fixture saved to {PROMPTS_PATH}")
    print(f"Prompts SHA256: {prompts_sha}")

    # Save Intent Criteria Fixture
    with open(INTENT_PATH, "w", encoding="utf-8") as f:
        json.dump(INTENT_CRITERIA, f, indent=2, ensure_ascii=False)
    intent_sha = sha256_file(INTENT_PATH)
    print(f"Intent Criteria saved to {INTENT_PATH}")
    print(f"Intent Criteria SHA256: {intent_sha}")

    # Check Evaluators SHA256
    d2b_sha = sha256_file(EVALUATOR_D2B_PATH)
    d3_sha = sha256_file(GATE_D3_PATH)
    print(f"Evaluator D2b SHA256: {d2b_sha}")
    print(f"Gate D3 SHA256:       {d3_sha}")

    # Write Manifest
    manifest = {
        "experiment_id": "RP-HDE-H2-D4",
        "protocol_version": "v0.2",
        "prompts_fixture_path": str(PROMPTS_PATH),
        "prompts_sha256": prompts_sha,
        "intent_criteria_path": str(INTENT_PATH),
        "intent_criteria_sha256": intent_sha,
        "evaluator_d2b_path": str(EVALUATOR_D2B_PATH),
        "evaluator_d2b_sha256": d2b_sha,
        "gate_d3_path": str(GATE_D3_PATH),
        "gate_d3_sha256": d3_sha,
        "seeds": [42, 43, 44],
        "concepts": list(CONCEPTS_D4.keys()),
        "conditions": ["C0", "C1", "C2"],
        "total_calls": 36,
        "model_id": "gpt-4o-mini-2024-07-18",
        "temperature": 0.2,
        "frozen_before_generation": True
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
