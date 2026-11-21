"""HDE Step H2-D6 Blind Automated Evaluation Runner.

Blinded Dual-Gate Evaluation across all 24 series candidates (6 base units x 4 conditions S0, S1, S2a, S2b):
1. Strips condition_id and seed prior to evaluation.
2. Evaluator 1 (Frozen D2b): evaluate_diagnostic_hypothesis -> 5-axis error flags (E_role, E_math, E_data, E_leap, E_inop).
3. Evaluator 2 (Frozen D3): evaluate_data_and_semantic_gate -> 2D Data Feasibility & Semantic Fidelity Gate (PASS / BLOCK / REVIEW).
4. Persists complete machine evaluation output to data/h2_d6_automated_eval_results.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR / "src"))

from research_pipeline.discovery.models import HypothesisCandidate
from research_pipeline.discovery.diagnostic_evaluator import evaluate_diagnostic_hypothesis
from research_pipeline.discovery.data_feasibility_gate import evaluate_data_and_semantic_gate

RAW_CANDIDATES_PATH = BASE_DIR / "data" / "h2_d6_raw_candidates.json"
OUTPUT_EVAL_PATH = BASE_DIR / "data" / "h2_d6_automated_eval_results.json"


def run_blind_evaluation():
    if not RAW_CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Raw candidates file not found: {RAW_CANDIDATES_PATH}")

    with open(RAW_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        units = json.load(f)

    eval_results = []
    print(f"Starting Blinded Dual-Gate Evaluation on 24 series candidates (6 units x 4 conditions)...")

    for u in units:
        base_unit_id = u["base_unit_id"]
        cid = u["concept_id"]
        seed = u["seed"]

        for cond_id in ["S0", "S1", "S2a", "S2b"]:
            c_info = u["candidates"][cond_id]
            call_id = c_info["call_id"]
            raw_cand = c_info["candidate"]

            # Blind condition_id
            blinded_dict = dict(raw_cand)
            blinded_dict["condition_id"] = None
            cand_obj = HypothesisCandidate(**blinded_dict)

            # Evaluator 1: Frozen Diagnostic Evaluator
            diag_item = evaluate_diagnostic_hypothesis(cand_obj, cid)

            # Evaluator 2: Frozen Data Feasibility Gate
            gate_item = evaluate_data_and_semantic_gate(cand_obj, case_id=call_id)

            has_5axis_error = (
                diag_item.e_role
                or diag_item.e_math
                or diag_item.e_data
                or diag_item.e_leap
                or diag_item.e_inop
            )
            dual_pass = (not has_5axis_error) and (gate_item.gate_decision == "PASS")

            item_result = {
                "base_unit_id": base_unit_id,
                "call_id": call_id,
                "concept_id": cid,
                "condition_id": cond_id,
                "seed": seed,
                "blinded_during_eval": True,
                "diagnostic_5axis": diag_item.model_dump(),
                "data_semantic_gate": gate_item.model_dump(),
                "dual_gate_pass": dual_pass,
                "candidate_summary": {
                    "title": cand_obj.title,
                    "target_phenomenon": cand_obj.target_phenomenon,
                    "proposed_mechanism": cand_obj.proposed_mechanism,
                    "entry_condition": cand_obj.backtest_spec.entry_condition if cand_obj.backtest_spec else "",
                    "exit_condition": cand_obj.backtest_spec.exit_condition if cand_obj.backtest_spec else "",
                    "feature_engineering": cand_obj.backtest_spec.feature_engineering if cand_obj.backtest_spec else "",
                    "risk_management": cand_obj.backtest_spec.risk_management if cand_obj.backtest_spec else "",
                }
            }
            eval_results.append(item_result)

    with open(OUTPUT_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"Blinded Evaluation complete! Evaluated {len(eval_results)} candidates.")
    print(f"Results saved to: {OUTPUT_EVAL_PATH}")


if __name__ == "__main__":
    run_blind_evaluation()
