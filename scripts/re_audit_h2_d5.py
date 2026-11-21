"""HDE Step H2-D5 Unified Re-Audit Runner.

Addresses Scientific Review Directives:
1. Applies Unified 3-Stage Executability Framework across all 18 cases:
   - SyntacticallyExecutable: Codeable syntax with explicit numbers/rules.
   - GroundedExecutable: Non-hallucinated, grounded in theory, literature, or complete in-sample estimation specs.
   - EmpiricallyExecutable: Verified on actual market backtest (N/A in H2-D5).
2. Re-audits Hawkes D1 (s42, s43) and Jump-Diffusion under strict, non-asymmetric grounding standards.
3. Classifies the 11 Automated Dual-Gate PASS vs Audit FAIL discrepancies into:
   - Category A: True Evaluator Misses within existing scope (e.g. E_inop / E_math false negatives).
   - Category B: Novel H2-D5 Completeness Requirements beyond frozen evaluator scope.
   - Category C: Criterion Interpretation Differences.
4. Outputs re-audit ledger to data/h2_d5_reaudit_results.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR / "src"))

RAW_PATH = BASE_DIR / "data" / "h2_d5_raw_candidates.json"
AUTO_PATH = BASE_DIR / "data" / "h2_d5_automated_eval_results.json"
GOLD_PARAMS_PATH = BASE_DIR / "tests" / "fixtures" / "gold_standard" / "h2_d5_gold_parameters.json"
OUTPUT_PATH = BASE_DIR / "data" / "h2_d5_reaudit_results.json"


def run_reaudit():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_candidates = json.load(f)
    with open(AUTO_PATH, "r", encoding="utf-8") as f:
        auto_evals = json.load(f)
    with open(GOLD_PARAMS_PATH, "r", encoding="utf-8") as f:
        gold_spec = json.load(f)

    auto_map = {r["call_id"]: r for r in auto_evals}

    reaudit_records = []
    discrepancy_categories = {"Category_A": 0, "Category_B": 0, "Category_C": 0}

    for r in raw_candidates:
        call_id = r["call_id"]
        cid = r["concept_id"]
        cond = r["condition_id"]
        seed = r["seed"]
        cand = r["candidate"]
        spec = cand.get("backtest_spec") or {}
        entry = (spec.get("entry_condition") or "").lower()
        exit_ = (spec.get("exit_condition") or "").lower()
        feat = (spec.get("feature_engineering") or "").lower()
        risk = (spec.get("risk_management") or "").lower()
        claim = (cand.get("claim") or "").lower()
        mech = (cand.get("proposed_mechanism") or "").lower()
        text = f"{claim} {mech} {entry} {exit_} {feat} {risk}"

        auto_item = auto_map[call_id]
        auto_dual_pass = auto_item["dual_gate_pass"]
        auto_gate_dec = auto_item["data_semantic_gate"]["gate_decision"]
        diag = auto_item["diagnostic_5axis"]

        # 1. Syntactic Executability
        # Does it contain explicit programmable rules or numbers (even if ungrounded)?
        has_entry_syntax = bool(entry and entry != "unspecified" and not any(w in entry for w in ["when intensity increases", "qualitative", "unspecified"]))
        has_exit_syntax = bool(exit_ and exit_ != "unspecified")
        has_risk_syntax = bool(risk and risk != "unspecified")
        has_feature_syntax = bool(feat and feat != "unspecified")
        syntactically_executable = bool(has_entry_syntax and has_exit_syntax and has_risk_syntax and has_feature_syntax)

        # 2. Grounded Executability
        # Strict unified criteria: Are parameters backed by empirical justification or complete in-sample estimation protocol?
        # Are numbers hallucinated?
        hallucinated_numbers = []
        is_grounded_executable = False
        unsubstantiated_reason = ""

        if cid == "hawkes_process":
            # Check s42, s43, s44
            if "0.5" in entry or "0.1" in entry or "0.8" in entry or "1.0" in entry:
                # Was there an in-sample estimation protocol or literature citation for USDJPY M5?
                has_calibration = any(w in text for w in ["calibrated via mle", "grid search", "in-sample estimation protocol", "historical sample"])
                if not has_calibration:
                    hallucinated_numbers.append("entry_threshold_0.5_or_0.1")
            if "alpha/beta > 1" in entry:
                hallucinated_numbers.append("explosive_branching_ratio_violation")
                unsubstantiated_reason = "Branching ratio alpha/beta > 1 violates subcritical stationarity."

        elif cid == "jump_diffusion":
            # Check if bipower cutoffs or regime probabilities are hallucinated
            if "15 pips" in text or "0.8" in text or "0.5" in text:
                has_calibration = any(w in text for w in ["calibrated via", "in-sample", "historical rolling"])
                if not has_calibration:
                    hallucinated_numbers.append("uncalibrated_jump_cutoff")

        # Complete In-Sample Estimation protocol check
        has_complete_in_sample = (
            any(w in text for w in ["mle", "grid search", "in_sample_estimated", "calibration"]) and
            any(w in text for w in ["lookback", "window", "bars"]) and
            any(w in text for w in ["objective", "likelihood", "sharpe", "sortino", "loss"])
        )

        has_justified_prior = any(w in text for w in ["literature", "prior empirical", "standard convention in"])

        if syntactically_executable and not hallucinated_numbers and (has_complete_in_sample or has_justified_prior):
            is_grounded_executable = True
        else:
            is_grounded_executable = False
            if not unsubstantiated_reason:
                if hallucinated_numbers:
                    unsubstantiated_reason = f"Arbitrary numbers without M5 calibration: {hallucinated_numbers}"
                elif not syntactically_executable:
                    unsubstantiated_reason = "Syntactically incomplete operational rules"
                else:
                    unsubstantiated_reason = "Incomplete in-sample estimation protocol or theoretical boundary used as trade trigger"

        # 3. Discrepancy Classification for Auto-PASS vs Audit-FAIL
        discrepancy_category = None
        discrepancy_explanation = ""

        if auto_dual_pass and not is_grounded_executable:
            # Why did auto PASS?
            # Check if frozen evaluator should have caught it (Category A)
            if hallucinated_numbers and ("explosive_branching_ratio_violation" in hallucinated_numbers):
                # E_math should have flagged alpha/beta > 1 stationarity violation!
                discrepancy_category = "Category_A"
                discrepancy_explanation = "Evaluator Miss: Frozen diagnostic evaluator missed stationarity violation alpha/beta > 1 (E_math false negative)."
                discrepancy_categories["Category_A"] += 1
            elif not syntactically_executable:
                # E_inop false negative
                discrepancy_category = "Category_A"
                discrepancy_explanation = "Evaluator Miss: Frozen evaluator missed qualitative inoperable syntax (E_inop false negative)."
                discrepancy_categories["Category_A"] += 1
            else:
                # Syntax was present, so frozen E_inop passed, but H2-D5 4-layer parameter completeness was unfulfilled
                discrepancy_category = "Category_B"
                discrepancy_explanation = "Novel Scope: Hypothesis satisfies basic 5-axis/data contracts, but fails H2-D5 4-layer parameter completeness and grounding standards."
                discrepancy_categories["Category_B"] += 1

        elif not auto_dual_pass and is_grounded_executable:
            discrepancy_category = "Category_C"
            discrepancy_explanation = "Interpretation Difference: Grounded specification blocked by conservative frozen evaluator."
            discrepancy_categories["Category_C"] += 1

        rec = {
            "call_id": call_id,
            "concept_id": cid,
            "condition_id": cond,
            "seed": seed,
            "syntactically_executable": syntactically_executable,
            "grounded_executable": is_grounded_executable,
            "empirically_executable": "NOT_TESTED_IN_H2_D5",
            "hallucinated_numbers": hallucinated_numbers,
            "unsubstantiated_reason": unsubstantiated_reason,
            "auto_dual_pass": auto_dual_pass,
            "is_discrepancy": (auto_dual_pass != is_grounded_executable),
            "discrepancy_category": discrepancy_category,
            "discrepancy_explanation": discrepancy_explanation,
        }
        reaudit_records.append(rec)

    # Condition Aggregations
    cond_summary = {}
    for cond in ["D0", "D1", "D2"]:
        crecs = [r for r in reaudit_records if r["condition_id"] == cond]
        n = len(crecs)
        syntactic_rate = sum(1 for r in crecs if r["syntactically_executable"]) / n
        grounded_rate = sum(1 for r in crecs if r["grounded_executable"]) / n
        auto_pass_rate = sum(1 for r in crecs if r["auto_dual_pass"]) / n
        halluc_rate = sum(1 for r in crecs if r["hallucinated_numbers"]) / n

        cond_summary[cond] = {
            "n_calls": n,
            "syntactically_executable_rate": round(syntactic_rate, 4),
            "grounded_executable_rate": round(grounded_rate, 4),
            "hallucinated_number_rate": round(halluc_rate, 4),
            "auto_pass_rate": round(auto_pass_rate, 4),
        }

    output_data = {
        "experiment_id": "RP-HDE-H2-D5-REAUDIT",
        "total_audited": len(reaudit_records),
        "discrepancy_breakdown": discrepancy_categories,
        "condition_summary": cond_summary,
        "reaudit_records": reaudit_records,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("=== H2-D5 UNIFIED RE-AUDIT COMPLETE ===")
    print(f"Total Audited: {len(reaudit_records)}")
    print(f"Discrepancy Breakdown: Category A (Evaluator Miss) = {discrepancy_categories['Category_A']}, Category B (Novel H2-D5 Scope) = {discrepancy_categories['Category_B']}, Category C = {discrepancy_categories['Category_C']}")
    print("\nCondition Summary:")
    for c, s in cond_summary.items():
        print(f"  {c}: Syntactic={s['syntactically_executable_rate']*100:.1f}%, Grounded={s['grounded_executable_rate']*100:.1f}%, Hallucinated={s['hallucinated_number_rate']*100:.1f}%, AutoPASS={s['auto_pass_rate']*100:.1f}%")


if __name__ == "__main__":
    run_reaudit()
