"""Build H2-D5 Prompts, Pre-Frozen Gold Parameter Fixtures, and Audit Manifest.

Adheres strictly to RP-HDE-H2-D5-PLAN-v0.2 with user review execution conditions:
- Target Concepts: Hawkes Point Process, Regime-Switching Jump Diffusion
- 3 Conditions: D0 (Natural Language IO Baseline), D1 (Parameter Enumeration), D2 (Grounded Disposition)
- Pre-freezes Essential Parameter Sets (P*_gold) across 4 layers to fix evaluation denominators before generation:
  1. Layer 1: Model Internal Parameters
  2. Layer 2: Decision Thresholds
  3. Layer 3: Risk & Sizing Constraints
  4. Layer 4: Execution Timing & Horizon Constraints
- Computes and records SHA256 hashes in h2_d5_audit_manifest.json.
"""

import json
import hashlib
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures" / "gold_standard"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_PATH = FIXTURES_DIR / "h2_d5_prompts.json"
GOLD_PARAMS_PATH = FIXTURES_DIR / "h2_d5_gold_parameters.json"
MANIFEST_PATH = FIXTURES_DIR / "h2_d5_audit_manifest.json"

EVALUATOR_D2B_PATH = BASE_DIR / "src" / "research_pipeline" / "discovery" / "diagnostic_evaluator.py"
GATE_D3_PATH = BASE_DIR / "src" / "research_pipeline" / "discovery" / "data_feasibility_gate.py"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ==============================================================================
# 1. Essential Gold Parameters (Fixed Denominators across 4 Layers)
# ==============================================================================

GOLD_PARAMETERS = {
    "hawkes_process": {
        "concept_name": "Multivariate Hawkes Point Process",
        "essential_parameters": {
            "layer1_model_parameters": [
                {
                    "param_id": "hawkes_baseline_mu",
                    "symbol": "mu",
                    "description": "Baseline background arrival rate",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "mu > 0"
                },
                {
                    "param_id": "hawkes_excitation_alpha",
                    "symbol": "alpha",
                    "description": "Excitation magnitude / jump reflexivity",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "alpha >= 0"
                },
                {
                    "param_id": "hawkes_decay_beta",
                    "symbol": "beta",
                    "description": "Exponential memory decay rate",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "beta > 0 and alpha < beta (subcritical stationarity)"
                }
            ],
            "layer2_decision_thresholds": [
                {
                    "param_id": "hawkes_branching_ratio_threshold",
                    "symbol": "(alpha/beta)*",
                    "description": "Critical cascade risk threshold triggering de-risking or position entry",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "0 < (alpha/beta)* < 1.0"
                },
                {
                    "param_id": "hawkes_intensity_threshold",
                    "symbol": "lambda*",
                    "description": "Absolute arrival intensity spike threshold",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "lambda* > mu"
                }
            ],
            "layer3_risk_constraints": [
                {
                    "param_id": "hawkes_risk_sizing_scale",
                    "symbol": "w_risk",
                    "description": "Position sizing scale factor inversely proportional to cascade intensity",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "0 <= w_risk <= 1.0"
                },
                {
                    "param_id": "hawkes_max_drawdown_stop",
                    "symbol": "max_dd_stop",
                    "description": "Maximum adverse excursion / stop loss threshold in basis points or pips",
                    "admissible_dispositions": ["EMPIRICAL_FIXED", "IN_SAMPLE_ESTIMATED"],
                    "theoretical_constraint": "max_dd_stop > 0"
                }
            ],
            "layer4_execution_constraints": [
                {
                    "param_id": "hawkes_execution_timing_rule",
                    "symbol": "t_exec",
                    "description": "Decision evaluated at confirmed bar close C_t and executed at open O_{t+1}",
                    "admissible_dispositions": ["THEORETICAL_CONSTRAINT", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "available_at(C_t) <= decision_time < execution_time(O_{t+1})"
                },
                {
                    "param_id": "hawkes_max_holding_horizon",
                    "symbol": "K_horizon",
                    "description": "Finite maximum holding duration in M5 bars (time-stop exit)",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "K_horizon >= 1"
                }
            ]
        }
    },
    "jump_diffusion": {
        "concept_name": "Regime-Switching Jump Diffusion Model",
        "essential_parameters": {
            "layer1_model_parameters": [
                {
                    "param_id": "jump_bipower_threshold_theta",
                    "symbol": "theta_BV",
                    "description": "Truncation threshold multiplier separating jump variance from diffusive variance",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "theta_BV > 0 (typically 3.0 * sigma_M5)"
                },
                {
                    "param_id": "jump_hamilton_transition_probs",
                    "symbol": "p_00, p_11",
                    "description": "Markov state transition persistence probabilities (tranquil vs crisis)",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "0 < p_00, p_11 < 1.0"
                },
                {
                    "param_id": "jump_regime_variance_ratio",
                    "symbol": "sigma_crisis / sigma_tranquil",
                    "description": "Variance separability scale between latent regimes",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "sigma_crisis > sigma_tranquil > 0"
                }
            ],
            "layer2_decision_thresholds": [
                {
                    "param_id": "jump_regime_posterior_threshold",
                    "symbol": "P(S_t = crisis)*",
                    "description": "Posterior regime probability threshold triggering protective de-risking or hedging",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "0.5 <= P(crisis)* < 1.0"
                },
                {
                    "param_id": "jump_component_size_threshold",
                    "symbol": "J_t*",
                    "description": "Minimum realized jump component J_t = max(0, RV_t - BV_t) triggering jump alert",
                    "admissible_dispositions": ["IN_SAMPLE_ESTIMATED", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "J_t* > 0"
                }
            ],
            "layer3_risk_constraints": [
                {
                    "param_id": "jump_regime_lot_multiplier",
                    "symbol": "lot_crisis",
                    "description": "Capital allocation fraction during crisis/jump regime",
                    "admissible_dispositions": ["EMPIRICAL_FIXED", "IN_SAMPLE_ESTIMATED"],
                    "theoretical_constraint": "0.0 <= lot_crisis < 1.0"
                },
                {
                    "param_id": "jump_trailing_stop_threshold",
                    "symbol": "trailing_stop_bps",
                    "description": "Dynamic volatility-scaled stop loss in bps based on tranquil diffusion sigma",
                    "admissible_dispositions": ["EMPIRICAL_FIXED", "IN_SAMPLE_ESTIMATED"],
                    "theoretical_constraint": "trailing_stop_bps > 0"
                }
            ],
            "layer4_execution_constraints": [
                {
                    "param_id": "jump_execution_timing_rule",
                    "symbol": "t_exec",
                    "description": "Bipower metrics evaluated on confirmed bar close C_t and executed at open O_{t+1}",
                    "admissible_dispositions": ["THEORETICAL_CONSTRAINT", "EMPIRICAL_FIXED"],
                    "theoretical_constraint": "available_at(C_t) <= decision_time < execution_time(O_{t+1})"
                },
                {
                    "param_id": "jump_regime_rebalance_interval",
                    "symbol": "K_rebalance",
                    "description": "Regime filtering update frequency (e.g. every M5 bar or every 12 bars)",
                    "admissible_dispositions": ["EMPIRICAL_FIXED", "IN_SAMPLE_ESTIMATED"],
                    "theoretical_constraint": "K_rebalance >= 1"
                }
            ]
        }
    }
}


# ==============================================================================
# 2. Prompts Building for D0, D1, D2
# ==============================================================================

BASE_IO_SPECS = {
    "hawkes_process": (
        "Capability IO Specification for Multivariate Hawkes Point Process:\n"
        "The model estimates self-exciting arrival intensities and detects endogenous cascade risks.\n"
        "Input features: Discrete event arrival sequence with timestamps tau_i. Under the USDJPY M5 OHLCV data contract where tick orders are unavailable, arrivals must be defined as an explicit M5 bar return exceedance proxy (such as abs(return_t) > 2.0 * rolling_std).\n"
        "Outputs: Conditional event arrival intensity lambda(t) and self-excitation branching ratio alpha/beta.\n"
        "Data Contract: Input data is strictly USDJPY M5 OHLCV bars with no tick trade logs or order book depth. Native order arrival and cancellation events cannot be directly observed on M5 bars.\n"
        "Control Variables & Boundary Conditions: Hawkes process is a statistical point process estimation model, not an order routing scheduler or speculative price direction forecaster.\n"
        "Identifiability & Estimation Assumptions: Parameters (mu, alpha, beta) are estimated via MLE or moment matching over a fixed lookback window under the strict stationarity stability condition alpha < beta with positive baseline mu > 0; the estimated intensity lambda(t) represents event occurrence frequency, NOT directional price return.\n"
        "Temporal Rules: Event exceedances are confirmed at bar close C_t; position de-risking executes at next bar open O_{t+1}."
    ),
    "jump_diffusion": (
        "Capability IO Specification for Regime-Switching Jump Diffusion:\n"
        "The model filters continuous Gaussian diffusion from discrete Poisson jumps across latent market regimes.\n"
        "Input features: M5 log-returns r_t, Realized Variance RV_t, and Bipower Variation BV_t computed over a rolling lookback window of 288 M5 bars.\n"
        "Outputs: Jump component test statistic J_t = max(0, RV_t - BV_t), filtered regime state probability P(S_t = tranquil | F_t), and filtered Poisson jump arrival intensity lambda_jump.\n"
        "Data Contract: Input data is strictly USDJPY M5 OHLCV bars without external macro news feeds or microsecond tick arrival logs. Announcement timing is inferred from return bipower jump metrics.\n"
        "Control Variables & Boundary Conditions: Regime-switching jump model estimates state transition probabilities for risk management, not for automated limit order placement into book queues.\n"
        "Identifiability & Estimation Assumptions: Continuous and jump variances are separated via threshold bipower variation; 2-state Markov chain transition matrix is estimated via Hamilton filter under state variance separability constraints sigma_crisis > sigma_tranquil; Poisson jumps are stochastic arrival events, NOT deterministic price surges.\n"
        "Temporal Rules: Bipower jump statistics are computed on completed bar close C_t; position adjustment executes at next bar open O_{t+1}."
    )
}

KG_TRIPLES = {
    "hawkes_process": [
        "[Multivariate Hawkes Point Process (model)] --(EXPLAINS)--> [High-Frequency Flash Crash Propagation (phenomenon)]",
        "[Detect Endogenous Cascade Risks (capability)] --(PROVIDED_BY)--> [Multivariate Hawkes Point Process (model)]",
        "[Multivariate Hawkes Point Process (model)] --(VARIANT_OF)--> [Hawkes Point Process (model)]"
    ],
    "jump_diffusion": [
        "[Regime-Switching Jump Diffusion Model (model)] --(EXPLAINS)--> [Central Bank Rate Announcement Jumps (phenomenon)]",
        "[Infer Latent Jump Intensities (capability)] --(PROVIDED_BY)--> [Regime-Switching Jump Diffusion Model (model)]",
        "[Regime-Switching Jump Diffusion Model (model)] --(VARIANT_OF)--> [Merton Jump Diffusion Model (model)]"
    ]
}


def build_h2_d5_prompt(concept_id: str, condition_id: str) -> str:
    triples_text = "\n".join(f"- {t}" for t in KG_TRIPLES[concept_id])
    c_name = GOLD_PARAMETERS[concept_id]["concept_name"]
    io_spec = BASE_IO_SPECS[concept_id]

    base_header = (
        f"Target Concept: {c_name}\n"
        f"Data Contract: USDJPY M5 OHLCV only (no external data feeds, no L2/L3 order book depth, no tick trade logs).\n\n"
        f"Knowledge Graph Relation Paths:\n{triples_text}\n\n"
        f"Capability IO Information (Natural Language Description):\n{io_spec}\n\n"
    )

    if condition_id == "D0":
        prompt = (
            f"{base_header}"
            f"Task: Formulate a single mathematically sound, data-feasible, and testable algorithmic trading hypothesis for USDJPY execution in AlgoForge based on the above Knowledge Graph facts and Capability IO description.\n"
        )
    elif condition_id == "D1":
        prompt = (
            f"{base_header}"
            f"INSTRUCTION ADDITION (Parameter Enumeration Mandate):\n"
            f"To convert this theoretical model into an operable AlgoForge trading strategy, you MUST explicitly identify and enumerate all undetermined parameters across the following 4 distinct layers in your backtest_spec:\n"
            f"  Layer 1: Model Internal Parameters (estimation parameters, kernel decay, variances)\n"
            f"  Layer 2: Decision Thresholds (entry/exit trigger cutoffs, probability thresholds)\n"
            f"  Layer 3: Risk & Sizing Constraints (position scaling fractions, stop loss boundaries)\n"
            f"  Layer 4: Execution Timing & Horizon Constraints (decision bar, execution bar, max holding duration)\n"
            f"Do not leave any operational parameters implicit or qualitative.\n\n"
            f"Task: Formulate a single mathematically sound, data-feasible, and testable algorithmic trading hypothesis for USDJPY execution in AlgoForge including the explicit 4-layer parameter enumeration.\n"
        )
    elif condition_id == "D2":
        prompt = (
            f"{base_header}"
            f"INSTRUCTION ADDITION (Grounded Disposition Protocol):\n"
            f"To convert this theoretical model into an operable AlgoForge trading strategy without ungrounded hallucinations, you MUST explicitly enumerate all undetermined parameters across the 4 layers (Model, Decision, Risk, Execution), AND for EACH parameter, you MUST assign one of the following 4 grounded disposition classes with complete justification:\n"
            f"  1. 'THEORETICAL_CONSTRAINT': Fixed by mathematical stability/probability bounds (e.g. alpha < beta). Specify the theoretical formula. Note: theoretical constraints alone CANNOT serve as trade decision trigger thresholds.\n"
            f"  2. 'EMPIRICAL_FIXED': Fixed to a specific numeric constant from literature. You MUST cite the literature/source and justify why this exact numeric constant is valid for USDJPY M5 bars.\n"
            f"  3. 'IN_SAMPLE_ESTIMATED': Estimated from training data. You MUST specify the 6-point estimation protocol: (a) search space bounds, (b) objective function/loss metric, (c) estimation lookback window, (d) in-sample validation split, (e) re-estimation frequency, and (f) lookahead protection rule.\n"
            f"  4. 'DECLARED_INFEASIBLE': Explicitly declared infeasible under the USDJPY M5 data contract. You MUST state the exact mathematical or data contract reason why this parameter cannot be soundly fixed or estimated on M5 bars.\n"
            f"CRITICAL RULE: DO NOT insert arbitrary numeric constants without empirical or theoretical proof. Fabricating ungrounded numbers (e.g. declaring threshold = 0.8 without proof) is strictly penalized.\n\n"
            f"Task: Formulate a single mathematically sound, data-feasible, and testable algorithmic trading hypothesis for USDJPY execution in AlgoForge strictly adhering to the Grounded Disposition Protocol.\n"
        )
    else:
        raise ValueError(f"Unknown condition_id: {condition_id}")

    return prompt


def main():
    prompts_db = {}
    for cid in ["hawkes_process", "jump_diffusion"]:
        prompts_db[cid] = {}
        for cond in ["D0", "D1", "D2"]:
            prompts_db[cid][cond] = build_h2_d5_prompt(cid, cond)

    # Save Prompts Fixture
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts_db, f, indent=2, ensure_ascii=False)
    prompts_sha = sha256_file(PROMPTS_PATH)
    print(f"Prompts fixture saved to: {PROMPTS_PATH}")
    print(f"Prompts SHA256: {prompts_sha}")

    # Save Gold Parameters Fixture
    with open(GOLD_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(GOLD_PARAMETERS, f, indent=2, ensure_ascii=False)
    gold_sha = sha256_file(GOLD_PARAMS_PATH)
    print(f"Gold Parameters fixture saved to: {GOLD_PARAMS_PATH}")
    print(f"Gold Parameters SHA256: {gold_sha}")

    # Check Evaluators Hashes (Strictly Frozen)
    d2b_sha = sha256_file(EVALUATOR_D2B_PATH)
    d3_sha = sha256_file(GATE_D3_PATH)
    print(f"Evaluator D2b SHA256: {d2b_sha}")
    print(f"Gate D3 SHA256:       {d3_sha}")

    # Write Manifest
    manifest = {
        "experiment_id": "RP-HDE-H2-D5",
        "protocol_version": "v0.2",
        "prompts_fixture_path": str(PROMPTS_PATH),
        "prompts_sha256": prompts_sha,
        "gold_parameters_path": str(GOLD_PARAMS_PATH),
        "gold_parameters_sha256": gold_sha,
        "evaluator_d2b_path": str(EVALUATOR_D2B_PATH),
        "evaluator_d2b_sha256": d2b_sha,
        "gate_d3_path": str(GATE_D3_PATH),
        "gate_d3_sha256": d3_sha,
        "concepts": ["hawkes_process", "jump_diffusion"],
        "conditions": ["D0", "D1", "D2"],
        "seeds": [42, 43, 44],
        "total_calls": 18,
        "model_id": "gpt-4o-mini-2024-07-18",
        "temperature": 0.2,
        "frozen_before_generation": True
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
