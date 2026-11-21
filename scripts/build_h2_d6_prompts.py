"""Build H2-D6 Prompts, Gold Parameter Necessity Fixtures, and Audit Manifest.

Adheres strictly to RP-HDE-H2-D6-PLAN-v0.2 with User Review Conditions:
1. Gold Necessity Classification across 4 tiers:
   - MODEL_REQUIRED: Essential for mathematical estimation/identification.
   - STRATEGY_REQUIRED: Essential for operational trade execution.
   - CONDITIONAL: Required only if specific sub-strategy is chosen.
   - OPTIONAL: Supplementary diagnostics.
2. Prompts for:
   - S0: Single-prompt baseline (simultaneous model + strategy)
   - S1 Stage 1: Mathematical model identification spec
   - S1 Stage 2: Trading strategy binding spec
   - S2a: Generic patch revision prompt
   - S2b: Deterministic non-leaking checklist patch prompt
3. Freezes SHA256 hashes in tests/fixtures/gold_standard/h2_d6_audit_manifest.json.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures" / "gold_standard"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_PATH = FIXTURES_DIR / "h2_d6_prompts.json"
GOLD_PARAMS_PATH = FIXTURES_DIR / "h2_d6_gold_parameters.json"
MANIFEST_PATH = FIXTURES_DIR / "h2_d6_audit_manifest.json"

EVALUATOR_D2B_PATH = BASE_DIR / "src" / "research_pipeline" / "discovery" / "diagnostic_evaluator.py"
GATE_D3_PATH = BASE_DIR / "src" / "research_pipeline" / "discovery" / "data_feasibility_gate.py"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ==============================================================================
# 1. Gold Parameters with 4-Tier Necessity Classification
# ==============================================================================

GOLD_PARAMETERS_D6 = {
    "hawkes_process": {
        "concept_name": "Multivariate Hawkes Point Process",
        "parameters": [
            {
                "param_id": "hawkes_baseline_mu",
                "symbol": "mu",
                "layer": "layer1_model",
                "necessity": "MODEL_REQUIRED",
                "description": "Baseline background arrival rate",
                "theoretical_rule": "mu > 0 strictly positive background rate under univariate exponential kernel"
            },
            {
                "param_id": "hawkes_excitation_alpha",
                "symbol": "alpha",
                "layer": "layer1_model",
                "necessity": "MODEL_REQUIRED",
                "description": "Excitation magnitude / jump reflexivity",
                "theoretical_rule": "alpha >= 0 under univariate exponential kernel"
            },
            {
                "param_id": "hawkes_decay_beta",
                "symbol": "beta",
                "layer": "layer1_model",
                "necessity": "MODEL_REQUIRED",
                "description": "Exponential memory decay rate",
                "theoretical_rule": "beta > 0 and alpha < beta (subcritical stationarity condition for univariate exponential kernel)"
            },
            {
                "param_id": "hawkes_intensity_threshold",
                "symbol": "lambda*",
                "layer": "layer2_decision",
                "necessity": "STRATEGY_REQUIRED",
                "description": "Arrival intensity spike cutoff triggering trade action",
                "theoretical_rule": "Requires empirical calibration or in-sample estimation; cannot be assumed arbitrarily"
            },
            {
                "param_id": "hawkes_execution_timing_rule",
                "symbol": "t_exec",
                "layer": "layer4_execution",
                "necessity": "STRATEGY_REQUIRED",
                "description": "Temporal execution ordering",
                "theoretical_rule": "Signal evaluated on confirmed bar close C_t; executed at next open O_{t+1} (idealized backtest assumption)"
            },
            {
                "param_id": "hawkes_max_drawdown_stop",
                "symbol": "max_dd",
                "layer": "layer3_risk",
                "necessity": "STRATEGY_REQUIRED",
                "description": "Risk stop loss / maximum adverse excursion threshold",
                "theoretical_rule": "Must be defined quantitatively in pips or basis points"
            },
            {
                "param_id": "hawkes_branching_ratio_threshold",
                "symbol": "(alpha/beta)*",
                "layer": "layer2_decision",
                "necessity": "CONDITIONAL",
                "description": "Critical cascade risk cutoff",
                "theoretical_rule": "Required only if branching ratio is used as active trade trigger. If strategy relies purely on lambda*, alpha/beta is model diagnostic only."
            },
            {
                "param_id": "hawkes_risk_sizing_scale",
                "symbol": "w_risk",
                "layer": "layer3_risk",
                "necessity": "CONDITIONAL",
                "description": "Volatility/intensity inversely scaled lot sizing",
                "theoretical_rule": "Required only if dynamic position sizing is chosen; optional if fixed lot sizing is used."
            },
            {
                "param_id": "hawkes_max_holding_horizon",
                "symbol": "K_horizon",
                "layer": "layer4_execution",
                "necessity": "OPTIONAL",
                "description": "Finite holding horizon / time stop exit in M5 bars",
                "theoretical_rule": "Optional if another valid exit rule (e.g. intensity drops below baseline) is fully defined."
            }
        ]
    },
    "jump_diffusion": {
        "concept_name": "Regime-Switching Jump Diffusion Model",
        "parameters": [
            {
                "param_id": "jump_bipower_threshold_theta",
                "symbol": "theta_BV",
                "layer": "layer1_model",
                "necessity": "MODEL_REQUIRED",
                "description": "Truncation threshold multiplier separating jump variance from diffusive variance",
                "theoretical_rule": "theta_BV > 0 over fixed rolling observation window (e.g. 288 M5 bars)"
            },
            {
                "param_id": "jump_hamilton_transition_probs",
                "symbol": "p_00, p_11",
                "layer": "layer1_model",
                "necessity": "MODEL_REQUIRED",
                "description": "Hamilton Markov regime transition probabilities",
                "theoretical_rule": "0 < p_00, p_11 < 1.0; state estimation must use filtered probability P(S_t | F_t) strictly, not smoothed probability P(S_t | F_T)"
            },
            {
                "param_id": "jump_regime_variance_ratio",
                "symbol": "sigma_ratio",
                "layer": "layer1_model",
                "necessity": "MODEL_REQUIRED",
                "description": "Variance separability scale between crisis and tranquil regimes",
                "theoretical_rule": "sigma_crisis > sigma_tranquil > 0"
            },
            {
                "param_id": "jump_execution_timing_rule",
                "symbol": "t_exec",
                "layer": "layer4_execution",
                "necessity": "STRATEGY_REQUIRED",
                "description": "Temporal execution ordering",
                "theoretical_rule": "Signal evaluated on confirmed bar close C_t; executed at next open O_{t+1} (idealized backtest assumption)"
            },
            {
                "param_id": "jump_trailing_stop_threshold",
                "symbol": "stop_bps",
                "layer": "layer3_risk",
                "necessity": "STRATEGY_REQUIRED",
                "description": "Dynamic volatility-scaled stop loss rule",
                "theoretical_rule": "Quantitative threshold in bps or ATR multiples"
            },
            {
                "param_id": "jump_regime_posterior_threshold",
                "symbol": "P(crisis)*",
                "layer": "layer2_decision",
                "necessity": "CONDITIONAL",
                "description": "Filtered posterior regime probability cutoff triggering defensive action",
                "theoretical_rule": "Required if regime switching is the primary trade trigger. Must strictly use filtered probability P(S_t | F_t)."
            },
            {
                "param_id": "jump_component_size_threshold",
                "symbol": "J_t*",
                "layer": "layer2_decision",
                "necessity": "CONDITIONAL",
                "description": "Realized jump component cutoff J_t = max(0, RV_t - BV_t)",
                "theoretical_rule": "Required if jump magnitude is used directly as trade trigger."
            },
            {
                "param_id": "jump_regime_lot_multiplier",
                "symbol": "lot_crisis",
                "layer": "layer3_risk",
                "necessity": "CONDITIONAL",
                "description": "Capital allocation fraction during crisis regime",
                "theoretical_rule": "Required only if regime-conditioned allocation sizing is adopted."
            },
            {
                "param_id": "jump_regime_rebalance_interval",
                "symbol": "K_rebal",
                "layer": "layer4_execution",
                "necessity": "OPTIONAL",
                "description": "Regime filtering parameter update frequency",
                "theoretical_rule": "Optional if updated per completed bar."
            }
        ]
    }
}


# ==============================================================================
# 2. Capability IO Specifications (Strictly Controlled across S0, S1, S2)
# ==============================================================================

BASE_IO_SPECS = {
    "hawkes_process": (
        "Capability IO Specification for Multivariate Hawkes Point Process:\n"
        "The model estimates self-exciting arrival intensities and detects endogenous cascade risks.\n"
        "Input features: Discrete event arrival sequence with timestamps tau_i. Under the USDJPY M5 OHLCV data contract where tick orders are unavailable, arrivals must be defined as an explicit M5 bar return exceedance proxy (such as abs(return_t) > 2.0 * rolling_std).\n"
        "Outputs: Conditional event arrival intensity lambda(t) and self-excitation branching ratio alpha/beta.\n"
        "Data Contract: Input data is strictly USDJPY M5 OHLCV bars with no tick trade logs or order book depth. Native order arrival and cancellation events cannot be directly observed on M5 bars.\n"
        "Control Variables & Boundary Conditions: Hawkes process is a statistical point process estimation model, not an order routing scheduler or speculative price direction forecaster.\n"
        "Identifiability & Estimation Assumptions: Parameters (mu, alpha, beta) are estimated via MLE or moment matching over a fixed lookback window under the strict stationarity stability condition alpha < beta with positive baseline mu > 0 (assuming univariate exponential kernel); the estimated intensity lambda(t) represents event occurrence frequency, NOT directional price return.\n"
        "Temporal Rules: Event exceedances are confirmed at bar close C_t; position execution takes place at next bar open O_{t+1} (idealized backtest assumption)."
    ),
    "jump_diffusion": (
        "Capability IO Specification for Regime-Switching Jump Diffusion:\n"
        "The model filters continuous Gaussian diffusion from discrete Poisson jumps across latent market regimes.\n"
        "Input features: M5 log-returns r_t, Realized Variance RV_t, and Bipower Variation BV_t computed over a rolling lookback window of 288 M5 bars.\n"
        "Outputs: Jump component test statistic J_t = max(0, RV_t - BV_t), filtered regime state probability P(S_t = tranquil | F_t), and filtered Poisson jump arrival intensity lambda_jump.\n"
        "Data Contract: Input data is strictly USDJPY M5 OHLCV bars without external macro news feeds or microsecond tick arrival logs. Announcement timing is inferred from return bipower jump metrics.\n"
        "Control Variables & Boundary Conditions: Regime-switching jump model estimates state transition probabilities for risk management, not for automated limit order placement into book queues.\n"
        "Identifiability & Estimation Assumptions: Continuous and jump variances are separated via threshold bipower variation; 2-state Markov chain transition matrix is estimated via Hamilton filter under state variance separability constraints sigma_crisis > sigma_tranquil; state probabilities for trading MUST strictly use filtered probabilities P(S_t | F_t) to prevent future leak; Poisson jumps are stochastic arrival events, NOT deterministic price surges.\n"
        "Temporal Rules: Bipower jump statistics are computed on completed bar close C_t; position adjustment executes at next bar open O_{t+1} (idealized backtest assumption)."
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


def build_s0_prompt(cid: str) -> str:
    triples = "\n".join(f"- {t}" for t in KG_TRIPLES[cid])
    cname = GOLD_PARAMETERS_D6[cid]["concept_name"]
    return (
        f"Target Concept: {cname}\n"
        f"Data Contract: USDJPY M5 OHLCV only (no external data feeds, no L2/L3 order book depth, no tick trade logs).\n\n"
        f"Knowledge Graph Relation Paths:\n{triples}\n\n"
        f"Capability IO Information (Natural Language Description):\n{BASE_IO_SPECS[cid]}\n\n"
        f"INSTRUCTION (Simultaneous Synthesis):\n"
        f"Formulate a complete, mathematically sound, data-feasible, and testable algorithmic trading hypothesis for USDJPY execution in AlgoForge.\n"
        f"Your output must specify both:\n"
        f"1. Mathematical Model Specification: Formal estimation protocol (search bounds, objective function, lookback window, validation split, update timing, lookahead protection) and proxy definition.\n"
        f"2. Complete AlgoForge Backtest Spec: Unambiguous entry condition, exit condition, risk management, benchmark baseline, and falsification metric.\n"
        f"Rule: DO NOT fabricate arbitrary numeric cutoffs without calibration or literature grounding.\n"
        f"Output must be a single valid JSON object matching the HypothesisCandidate schema.\n"
    )


def build_s1_stage1_prompt(cid: str) -> str:
    triples = "\n".join(f"- {t}" for t in KG_TRIPLES[cid])
    cname = GOLD_PARAMETERS_D6[cid]["concept_name"]
    return (
        f"Target Concept: {cname}\n"
        f"Data Contract: USDJPY M5 OHLCV only (no external data feeds, no L2/L3 order book depth, no tick trade logs).\n\n"
        f"Knowledge Graph Relation Paths:\n{triples}\n\n"
        f"Capability IO Information (Natural Language Description):\n{BASE_IO_SPECS[cid]}\n\n"
        f"STAGE 1 INSTRUCTION (Mathematical Model Identification):\n"
        f"Focus exclusively on specifying the mathematical estimation and state-space identification protocol for this model on USDJPY M5 data.\n"
        f"Specify:\n"
        f"- Target phenomenon and proposed mechanism\n"
        f"- Internal model parameters and theoretical boundaries\n"
        f"- Complete in-sample estimation protocol (search range, objective/likelihood, lookback window, validation split, update timing, lookahead protection)\n"
        f"- Filtered outputs (e.g. filtered intensity lambda(t) or filtered regime probability P(S_t|F_t))\n"
        f"Do NOT generate trade execution rules or backtest specs yet. Focus purely on mathematical and econometric identification.\n"
        f"Output must be a valid JSON object matching Stage 1 Model Spec.\n"
    )


def build_s1_stage2_prompt(cid: str, stage1_output_text: str) -> str:
    cname = GOLD_PARAMETERS_D6[cid]["concept_name"]
    return (
        f"Target Concept: {cname}\n"
        f"Data Contract: USDJPY M5 OHLCV only.\n\n"
        f"STAGE 1 MATHEMATICAL MODEL IDENTIFICATION SPECIFICATION (Confirmed Input):\n"
        f"```json\n{stage1_output_text}\n```\n\n"
        f"STAGE 2 INSTRUCTION (Trading Strategy Binding):\n"
        f"Using the mathematical model and filtered outputs established in Stage 1, formulate a complete, operable AlgoForge trading strategy.\n"
        f"You must determine:\n"
        f"- Concrete entry condition (linking filtered model outputs to trade action without ungrounded hallucinations)\n"
        f"- Clear exit condition (time-stop or signal reversal)\n"
        f"- Risk management (quantitative stop loss and sizing)\n"
        f"- Execution timing (C_t decision -> O_{{t+1}} execution)\n"
        f"- Benchmark baseline and falsification metric\n"
        f"Output must be a single valid JSON object matching the complete HypothesisCandidate schema.\n"
    )


def build_s2a_patch_prompt(s1_spec_text: str) -> str:
    return (
        f"CURRENT STRATEGY SPECIFICATION (Candidate under review):\n"
        f"```json\n{s1_spec_text}\n```\n\n"
        f"REVISION INSTRUCTION (Generic Specification Polish):\n"
        f"Carefully review the current specification. Check for any missing, qualitative, or incomplete operational parameters across model estimation, entry triggers, exit rules, and risk management.\n"
        f"Output a JSON patch containing modifications to make the strategy fully determined, operable, and grounded without arbitrary hallucinations.\n"
        f"Output format MUST be JSON:\n"
        f"{{\n"
        f'  "patch_summary": "Brief explanation of updates",\n'
        f'  "updates": {{\n'
        f'    "feature_engineering": "updated text or null if unchanged",\n'
        f'    "entry_condition": "updated text or null if unchanged",\n'
        f'    "exit_condition": "updated text or null if unchanged",\n'
        f'    "risk_management": "updated text or null if unchanged",\n'
        f'    "falsification_metric": "updated text or null if unchanged"\n'
        f'  }}\n'
        f"}}\n"
    )


def build_s2b_patch_prompt(s1_spec_text: str, missing_items_checklist: List[Dict[str, str]]) -> str:
    checklist_lines = "\n".join(
        f"- [{item['layer'].upper()}] Parameter '{item['symbol']}' ({item['param_id']}): {item['necessity']} — {item['description']}"
        for item in missing_items_checklist
    )
    return (
        f"CURRENT STRATEGY SPECIFICATION (Candidate under review):\n"
        f"```json\n{s1_spec_text}\n```\n\n"
        f"AUDIT FEEDBACK (Deterministic Non-Leaking Missing Parameter Checklist):\n"
        f"An automated completeness audit has identified that the following essential parameters for this strategy architecture are currently undetermined, qualitative, or missing estimation specifications:\n"
        f"{checklist_lines}\n\n"
        f"REVISION INSTRUCTION (Checklist-Guided Grounded Resolution):\n"
        f"Provide explicit in-sample estimation protocols (search bounds, window, objective) or literature-grounded justifications for the missing parameters listed above.\n"
        f"CRITICAL RULE: DO NOT insert arbitrary ungrounded numeric constants. Specify in-sample estimation procedures or theoretical boundary rules.\n"
        f"Output a JSON patch containing updates:\n"
        f"{{\n"
        f'  "patch_summary": "Brief explanation of how checklist items were resolved",\n'
        f'  "updates": {{\n'
        f'    "feature_engineering": "updated text or null if unchanged",\n'
        f'    "entry_condition": "updated text or null if unchanged",\n'
        f'    "exit_condition": "updated text or null if unchanged",\n'
        f'    "risk_management": "updated text or null if unchanged",\n'
        f'    "falsification_metric": "updated text or null if unchanged"\n'
        f'  }}\n'
        f"}}\n"
    )


def main():
    prompts_db = {
        "hawkes_process": {
            "S0": build_s0_prompt("hawkes_process"),
            "S1_Stage1": build_s1_stage1_prompt("hawkes_process"),
        },
        "jump_diffusion": {
            "S0": build_s0_prompt("jump_diffusion"),
            "S1_Stage1": build_s1_stage1_prompt("jump_diffusion"),
        }
    }

    # Save Prompts Fixture
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts_db, f, indent=2, ensure_ascii=False)
    prompts_sha = sha256_file(PROMPTS_PATH)
    print(f"Prompts fixture saved to: {PROMPTS_PATH}")
    print(f"Prompts SHA256: {prompts_sha}")

    # Save Gold Parameters Fixture
    with open(GOLD_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(GOLD_PARAMETERS_D6, f, indent=2, ensure_ascii=False)
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
        "experiment_id": "RP-HDE-H2-D6",
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
        "conditions": ["S0", "S1", "S2a", "S2b"],
        "seeds": [42, 43, 44],
        "total_calls": 30,
        "total_series": 24,
        "model_id": "gpt-4o-mini-2024-07-18",
        "temperature": 0.2,
        "frozen_before_generation": True
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
