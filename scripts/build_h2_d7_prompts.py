"""
Script: build_h2_d7_prompts.py
Constructs and freezes the exact prompts for H2-D7 2x2 Factorial Diagnostic Experiment:
- Factor 1: Capacity (Standard [1000/600] vs Expanded [2000/1200])
- Factor 2: Skeletal Guidance (None [C00, C10] vs Verified Skeleton [C01, C11])

Guarantees identical prompt text within each skeleton condition to completely isolate capacity from instruction.
Saves to tests/fixtures/gold_standard/h2_d7_prompts.json and records SHA256 hashes.
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR))
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures" / "gold_standard"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_PATH = FIXTURES_DIR / "h2_d7_prompts.json"
MANIFEST_PATH = FIXTURES_DIR / "h2_d7_manifest.json"

# Base Capability IO Specs (Same as P1 / S0)
from scripts.build_h2_d6_prompts import BASE_IO_SPECS, KG_TRIPLES, GOLD_PARAMETERS_D6

# ==============================================================================
# Verified Mathematical Specification Skeletons for C01 and C11
# ==============================================================================

HAWKES_SKELETON_TEXT = """VERIFIED ECONOMETRIC SPECIFICATION SKELETON (Discrete-Time Autoregressive Conditional Hazard Hawkes):
```json
{
  "mathematical_model": "Discrete-Time Recursive Hawkes Bernoulli-Hazard Model on USDJPY M5",
  "discrete_event_definition": {
    "formula": "y_t = I(|r_t| > 2.0 * sigma_roll_t)",
    "rolling_std_window": "Calculated on strictly past 20 M5 bars from t-20 to t-1 (bar t strictly excluded)",
    "return_definition": "r_t = log(C_t / C_{t-1})"
  },
  "hazard_state_equation": {
    "recursive_formula": "h_t = mu + rho * (h_{t-1} - mu) + gamma * (1.0 - rho) * y_{t-1}, with h_t > 0",
    "event_probability_link": "p_t = 1.0 - exp(-h_t) (Complementary log-log Bernoulli hazard link)",
    "dimensionless_parameters": {
      "mu": "Baseline bar-level expected hazard rate, mu in [1e-4, 0.05]",
      "rho": "Autoregressive memory persistence per 5-min step, rho in [0.01, 0.95]",
      "gamma": "Relative excitation ratio, gamma in [0.01, 0.95]",
      "implied_alpha": "alpha = gamma * (1.0 - rho), guaranteeing linear stability rho + alpha < 1"
    }
  },
  "estimation_protocol": {
    "initial_estimation_timing": "Initial estimation occurs at bar t=520 (first 20 bars initialize event rolling std; 500-bar rolling window accumulates from bar 21 to 520)",
    "warmup_period": "Bars 21 to 70 (first 50 bars of window) warm up recursive state h_t with initial state h_20 = mu; likelihood evaluated on bars 71 to 520",
    "log_likelihood_objective": "log L(mu, rho, gamma) = sum_{t=71}^{520} [y_t * log(1 - exp(-h_t)) - (1 - y_t) * h_t]",
    "solver": "L-BFGS-B with independent box constraints mu in [1e-4, 0.05], rho in [0.01, 0.95], gamma in [0.01, 0.95]",
    "initial_values": {"mu": 0.01, "rho": 0.50, "gamma": 0.20},
    "update_frequency": "Recalculated every 50 M5 bars (t = 520, 570, 620, ...)",
    "fallback_hierarchy": "If solver fails to converge within 100 iterations, fall back to initial values at t=520, or maintain previous estimates theta_{t-50} for t >= 570"
  }
}
```"""

JUMP_SKELETON_TEXT = """VERIFIED ECONOMETRIC SPECIFICATION SKELETON (Exploratory Local Jump Detector & Hamilton 2-State Log-RV Model):
```json
{
  "mathematical_model": "Decoupled Local-Standardized Return Jump Detector and Hamilton 2-State Log-RV Markov Switching",
  "module_a_intraday_jump_detector": {
    "local_continuous_volatility": "sigma_hat_t^2 = (pi / 2) * (1 / (K - 1)) * sum_{i=t-K+1}^{t-1} |r_i| * |r_{i-1}|, with K = 72 bars (6 hours, strictly excluding bar t, exactly 71 adjacent pairs)",
    "test_statistic": "L_t = |r_t| / sigma_hat_t if sigma_hat_t >= 1e-8 else 0.0 (fail-closed zero-variance handling)",
    "jump_threshold": "L* = 3.5 (standardized return exceeds 3.5 local continuous volatility)",
    "jump_signal": "I(L_t > 3.5) * sign(r_t)"
  },
  "module_b_hamilton_regime_model": {
    "daily_trading_boundary": "00:00 UTC to 23:55 UTC (288 M5 bars per trading day)",
    "daily_realized_variance": "RV_d = (288 / M_d) * sum_{i=1}^{M_d} r_{d,i}^2, where valid bars M_d >= 200 (days with M_d < 200 excluded as non-trading)",
    "observation_equation": "y_d = log(RV_d) ~ N(mu_{S_d}, sigma_{S_d}^2), with S_d in {0: tranquil, 1: crisis}",
    "state_identification_constraint": "mu_0 < mu_1 (strictly enforced via M-step label sorting to eliminate label switching)",
    "transition_matrix": "P = [[p00, 1-p00], [1-p11, p11]], with p00, p11 projected into [0.50, 0.99]",
    "estimation_protocol": {
      "estimation_window": "Rolling 60 trading days (initial estimation occurs at day 60 close at 00:00 UTC)",
      "em_initial_values": {"mu_0": "percentile(y, 25)", "mu_1": "percentile(y, 75)", "sigma_0": "std(y)/sqrt(2)", "sigma_1": "std(y)/sqrt(2)", "P": [[0.90, 0.10], [0.20, 0.80]]},
      "convergence_criteria": "Change in log-likelihood < 1e-6 or max 100 iterations. Aborts if likelihood decreases by > 1e-7",
      "causal_filtering": "Estimation completed at 00:00 UTC after day d closes; prior state probability xi_{d+1|d} = P^T * xi_{d|d} applied out-of-sample to M5 bars of day d+1",
      "fallback_hierarchy": "If EM fails to converge on day 60, use initial values; for day d > 60, maintain day d-1 estimates"
    }
  }
}
```"""


def build_c00_c10_initial_prompt(cid: str) -> str:
    """Prompt for C00 and C10 (No Skeleton). Identical to P1 A1 prompt."""
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


def build_c01_c11_initial_prompt(cid: str) -> str:
    """Prompt for C01 and C11 (With Skeleton). Augments C00 with verified mathematical skeleton."""
    triples = "\n".join(f"- {t}" for t in KG_TRIPLES[cid])
    cname = GOLD_PARAMETERS_D6[cid]["concept_name"]
    skeleton = HAWKES_SKELETON_TEXT if cid == "hawkes_process" else JUMP_SKELETON_TEXT

    return (
        f"Target Concept: {cname}\n"
        f"Data Contract: USDJPY M5 OHLCV only (no external data feeds, no L2/L3 order book depth, no tick trade logs).\n\n"
        f"Knowledge Graph Relation Paths:\n{triples}\n\n"
        f"Capability IO Information (Natural Language Description):\n{BASE_IO_SPECS[cid]}\n\n"
        f"{skeleton}\n\n"
        f"INSTRUCTION (Skeletal Guidance Synthesis):\n"
        f"Formulate a complete, mathematically sound, data-feasible, and testable algorithmic trading hypothesis for USDJPY execution in AlgoForge.\n"
        f"You MUST bind your strategy to the verified econometric specification skeleton provided above:\n"
        f"1. Mathematical Model Specification: Accurately incorporate the verified skeleton formulas, observation windows, parameter constraints, and estimation protocols into feature_engineering.\n"
        f"2. Complete AlgoForge Backtest Spec: Formulate an unambiguous entry condition (linking the model outputs to trade signals), exit condition, risk management (dynamic stop loss and position sizing), benchmark baseline, and falsification metric.\n"
        f"Rule: DO NOT fabricate arbitrary numeric cutoffs. Ensure execution timing follows confirmed bar close C_t -> next bar open O_{{t+1}}.\n"
        f"Output must be a single valid JSON object matching the HypothesisCandidate schema.\n"
    )


def build_checklist_patch_prompt(spec_text: str, missing_items_checklist: List[Dict[str, str]]) -> str:
    """Checklist patch prompt used across all conditions in H2-D7."""
    checklist_lines = "\n".join(
        f"- [{item['layer'].upper()}] Parameter '{item['symbol']}' ({item['param_id']}): {item['necessity']} — {item['description']}"
        for item in missing_items_checklist
    )
    return (
        f"CURRENT STRATEGY SPECIFICATION (Candidate under review):\n"
        f"```json\n{spec_text}\n```\n\n"
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
            "C00_C10_initial": build_c00_c10_initial_prompt("hawkes_process"),
            "C01_C11_initial": build_c01_c11_initial_prompt("hawkes_process"),
        },
        "jump_diffusion": {
            "C00_C10_initial": build_c00_c10_initial_prompt("jump_diffusion"),
            "C01_C11_initial": build_c01_c11_initial_prompt("jump_diffusion"),
        }
    }

    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts_db, f, indent=2, ensure_ascii=False)

    prompts_sha = hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest()
    print(f"H2-D7 Prompts saved to: {PROMPTS_PATH}")
    print(f"SHA256: {prompts_sha}")

    manifest = {
        "experiment_id": "H2-D7-MATH-SPEC-DIAG-05",
        "prompts_sha256": prompts_sha,
        "prompts_file": str(PROMPTS_PATH),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"H2-D7 Manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
