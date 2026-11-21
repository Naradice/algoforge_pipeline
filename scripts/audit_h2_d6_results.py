"""HDE Step H2-D6 Post-Generation Audit and Statistical Estimator.

Adheres strictly to RP-HDE-H2-D6-PLAN-v0.2 with user review execution conditions:
1. Evaluates all 24 series candidates across 4 conditions (S0, S1, S2a, S2b).
2. Decoupled Coverage Metrics:
   - GoldCoverageRaw: Denominator is fixed 9 Gold parameters (cross-experiment continuity).
   - RequiredCoverage: Denominator is strategy-specific required items (MODEL_REQUIRED + STRATEGY_REQUIRED + CONDITIONAL if active).
3. 3-Stage Executability:
   - R_GroundedExecutable: Complete codeable implementation specs (in-sample protocol or justified prior) without hallucinated numbers.
   - R_SyntacticallyExecutable: Explicit programmable rules/numbers (syntax level).
   - R_HallucinatedFixation: Arbitrary numbers asserted without empirical calibration.
4. Fidelity Decoupling:
   - R_StructuralFidelity: Preserved mathematical model architecture.
   - R_TargetFidelity: Preserved original microstructural physical phenomenon.
   - TrivializationDegeneration: Abandoned structure into plain heuristics.
5. Primary Paired Difference Estimators:
   - Delta_stage = S1 - S0 (Two-stage generation method effect)
   - Delta_revision = S2a - S1 (Generic revision opportunity & extra budget effect)
   - Delta_checklist = S2b - S2a (Pure informational value of missing parameter checklist effect)
6. Persists audit report to data/h2_d6_audit_results.json.
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR / "src"))

RAW_PATH = BASE_DIR / "data" / "h2_d6_raw_candidates.json"
AUTO_PATH = BASE_DIR / "data" / "h2_d6_automated_eval_results.json"
GOLD_PARAMS_PATH = BASE_DIR / "tests" / "fixtures" / "gold_standard" / "h2_d6_gold_parameters.json"
OUTPUT_AUDIT_PATH = BASE_DIR / "data" / "h2_d6_audit_results.json"

CONCEPTS = ["hawkes_process", "jump_diffusion"]
CONDITIONS = ["S0", "S1", "S2a", "S2b"]
SEEDS = [42, 43, 44]


def audit_candidate_specs(
    cid: str,
    cand: Dict[str, Any],
    gold_spec: Dict[str, Any],
) -> Dict[str, Any]:
    spec = cand.get("backtest_spec") or {}
    claim = (cand.get("claim") or "").lower()
    mech = (cand.get("proposed_mechanism") or "").lower()
    rat = (cand.get("grounding_rationale") or "").lower()
    entry = (spec.get("entry_condition") or "").lower()
    exit_ = (spec.get("exit_condition") or "").lower()
    feat = (spec.get("feature_engineering") or "").lower()
    risk = (spec.get("risk_management") or "").lower()
    falsif = (spec.get("falsification_metric") or "").lower()
    text = f"{claim} {mech} {rat} {entry} {exit_} {feat} {risk} {falsif}"

    params = gold_spec["parameters"]
    total_gold_raw = len(params)  # Exactly 9

    # Detect chosen strategy architecture
    is_hawkes_lambda_strategy = ("lambda" in text or "intensity" in text) and not ("alpha/beta >" in text)
    is_jump_bipower_strategy = ("bipower" in text or "j_t" in text or "jump" in text) and not ("p(crisis)" in text or "p(s_t" in text)

    raw_determined_count = 0
    required_total_count = 0
    required_determined_count = 0
    hallucinated_numbers = []

    for p in params:
        pid = p["param_id"]
        sym = p["symbol"].lower()
        nec = p["necessity"]

        # Check necessity for this specific candidate
        is_required = False
        if nec in ["MODEL_REQUIRED", "STRATEGY_REQUIRED"]:
            is_required = True
        elif nec == "CONDITIONAL":
            if pid == "hawkes_branching_ratio_threshold" and not is_hawkes_lambda_strategy:
                is_required = True
            elif pid == "jump_regime_posterior_threshold" and not is_jump_bipower_strategy:
                is_required = True
            elif pid == "jump_component_size_threshold" and is_jump_bipower_strategy:
                is_required = True
            elif "sizing" in p["description"].lower() and "scale" in text:
                is_required = True

        if is_required:
            required_total_count += 1

        # Check if determined with concrete estimation protocol or justified prior
        is_determined = False
        if sym.replace("*", "") in text or pid in text:
            # Check for concrete estimation protocol (bounds, window, objective)
            has_protocol = any(w in text for w in ["lookback", "window", "mle", "grid", "threshold", "estimated via", "bounds", "objective", "loss"])
            has_justification = any(w in text for w in ["literature", "prior empirical", "standard convention", "paper"])
            if (has_protocol or has_justification) and not any(w in text for w in ["unspecified", "qualitative"]):
                is_determined = True

        if is_determined:
            raw_determined_count += 1
            if is_required:
                required_determined_count += 1

    # Check for hallucinated numbers
    if cid == "hawkes_process":
        # Look for arbitrary numbers in entry condition
        matches = re.findall(r'(\d+\.\d+|\b\d+\b)', entry)
        for m in matches:
            if m not in ["0", "1", "2", "3"]:  # Standard offsets or powers
                # Was this number derived via in-sample protocol or cited?
                if not any(w in text for w in ["in-sample", "calibrated", "grid search", "mle", "literature", "rolling"]):
                    hallucinated_numbers.append(f"arbitrary_entry_{m}")
        if "alpha/beta > 1" in entry or "alpha/beta > 1.0" in entry:
            hallucinated_numbers.append("stationarity_violation_alpha_gt_beta")

    elif cid == "jump_diffusion":
        matches = re.findall(r'(\d+\.\d+|\b\d+\b)', entry)
        for m in matches:
            if m not in ["0", "1", "2", "288"]:
                if not any(w in text for w in ["in-sample", "calibrated", "grid search", "mle", "literature", "threshold"]):
                    hallucinated_numbers.append(f"arbitrary_entry_{m}")

    gold_coverage_raw = raw_determined_count / total_gold_raw
    required_coverage = (required_determined_count / required_total_count) if required_total_count > 0 else 0.0

    # Syntactic Executability
    has_entry_syntax = bool(entry and entry != "unspecified" and not any(w in entry for w in ["qualitative", "unspecified", "when intensity increases"]))
    has_exit_syntax = bool(exit_ and exit_ != "unspecified")
    has_risk_syntax = bool(risk and risk != "unspecified")
    has_feature_syntax = bool(feat and feat != "unspecified")
    syntactically_executable = bool(has_entry_syntax and has_exit_syntax and has_risk_syntax and has_feature_syntax)

    # Grounded Executability
    # Must have required coverage >= 0.75, syntactic rules, no hallucinations, and concrete in-sample or justified rules
    has_estimation_protocol = (
        any(w in text for w in ["mle", "grid search", "in_sample_estimated", "calibration"]) and
        any(w in text for w in ["lookback", "window", "bars"])
    )
    is_grounded_executable = bool(
        syntactically_executable and
        (required_coverage >= 0.75) and
        (not hallucinated_numbers) and
        (has_estimation_protocol or any(w in text for w in ["literature", "prior empirical"]))
    )

    # Structural & Target Fidelity
    structural_fidelity = False
    target_fidelity = False
    is_trivialized = False

    if cid == "hawkes_process":
        if ("hawkes" in text or "intensity" in text or "self-excit" in text or "branching ratio" in text) and ("lambda" in text or "alpha" in text or "beta" in text):
            structural_fidelity = True
        if ("order flow" in text or "order arrival" in text or "cancellation" in text) and not ("proxy" in text or "return exceedance" in text):
            target_fidelity = True
        if ("rsi" in entry or "moving average" in entry or "bollinger" in entry) and ("intensity" not in entry and "branching" not in entry):
            is_trivialized = True
            structural_fidelity = False

    elif cid == "jump_diffusion":
        if ("bipower" in text or "bv_t" in text or "realized variance" in text or "hamilton" in text or "regime-switching" in text) and ("jump" in text):
            structural_fidelity = True
        if ("central bank" in text or "announcement" in text or "macro news" in text) and not ("inferred" in text or "proxy" in text):
            target_fidelity = True
        if ("atr" in entry or "bollinger" in entry) and ("bipower" not in feat and "bv" not in feat and "regime" not in feat):
            is_trivialized = True
            structural_fidelity = False

    return {
        "gold_coverage_raw": round(gold_coverage_raw, 4),
        "required_coverage": round(required_coverage, 4),
        "required_total_count": required_total_count,
        "required_determined_count": required_determined_count,
        "syntactically_executable": syntactically_executable,
        "grounded_executable": is_grounded_executable,
        "hallucinated_numbers": hallucinated_numbers,
        "has_hallucinations": bool(hallucinated_numbers),
        "structural_fidelity": structural_fidelity,
        "target_fidelity": target_fidelity,
        "is_trivialized": is_trivialized,
    }


def run_post_generation_audit():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        units = json.load(f)
    with open(AUTO_PATH, "r", encoding="utf-8") as f:
        auto_evals = json.load(f)
    with open(GOLD_PARAMS_PATH, "r", encoding="utf-8") as f:
        gold_params_db = json.load(f)

    auto_map = {r["call_id"]: r for r in auto_evals}

    audit_records = []

    for u in units:
        base_unit_id = u["base_unit_id"]
        cid = u["concept_id"]
        seed = u["seed"]
        gold_spec = gold_params_db[cid]

        for cond_id in CONDITIONS:
            c_info = u["candidates"][cond_id]
            call_id = c_info["call_id"]
            cand = c_info["candidate"]
            auto_item = auto_map[call_id]

            audit_res = audit_candidate_specs(cid, cand, gold_spec)

            rec = {
                "base_unit_id": base_unit_id,
                "call_id": call_id,
                "concept_id": cid,
                "condition_id": cond_id,
                "seed": seed,
                "diagnostics": c_info.get("diagnostics", {}),
                "audit": audit_res,
                "auto_evaluation": {
                    "dual_gate_pass": auto_item["dual_gate_pass"],
                    "gate_decision": auto_item["data_semantic_gate"]["gate_decision"],
                    "5axis_errors": auto_item["diagnostic_5axis"],
                }
            }
            audit_records.append(rec)

    # Condition Statistics Across All Series (N=6 per condition: 2 concepts x 3 seeds)
    metrics_to_average = [
        "gold_coverage_raw",
        "required_coverage",
        "syntactically_executable",
        "grounded_executable",
        "has_hallucinations",
        "structural_fidelity",
        "target_fidelity",
        "is_trivialized",
    ]

    cond_stats = {}
    for cond in CONDITIONS:
        crecs = [r for r in audit_records if r["condition_id"] == cond]
        n = len(crecs)  # Exactly 6

        c_dict = {"n_series": n}
        for m in metrics_to_average:
            vals = []
            for r in crecs:
                v = r["audit"][m]
                vals.append(float(v) if isinstance(v, (int, float, bool)) else 0.0)
            c_dict[m] = round(sum(vals) / n, 4)

        c_dict["auto_pass_rate"] = round(sum(1 for r in crecs if r["auto_evaluation"]["dual_gate_pass"]) / n, 4)
        cond_stats[cond] = c_dict

    # Concept-Level Paired Differences
    concept_stats = {}
    for cid in CONCEPTS:
        concept_stats[cid] = {}
        for cond in CONDITIONS:
            crecs = [r for r in audit_records if r["concept_id"] == cid and r["condition_id"] == cond]
            n = len(crecs)  # Exactly 3 seeds

            c_dict = {"n_seeds": n}
            for m in metrics_to_average:
                vals = [float(r["audit"][m]) for r in crecs]
                c_dict[m] = round(sum(vals) / n, 4)
            c_dict["auto_pass_rate"] = round(sum(1 for r in crecs if r["auto_evaluation"]["dual_gate_pass"]) / n, 4)
            concept_stats[cid][cond] = c_dict

        # Paired Differences
        d_stage = {m: round(concept_stats[cid]["S1"][m] - concept_stats[cid]["S0"][m], 4) for m in metrics_to_average}
        d_rev   = {m: round(concept_stats[cid]["S2a"][m] - concept_stats[cid]["S1"][m], 4) for m in metrics_to_average}
        d_check = {m: round(concept_stats[cid]["S2b"][m] - concept_stats[cid]["S2a"][m], 4) for m in metrics_to_average}
        d_net   = {m: round(concept_stats[cid]["S2b"][m] - concept_stats[cid]["S0"][m], 4) for m in metrics_to_average}

        concept_stats[cid]["delta_stage_s1_minus_s0"] = d_stage
        concept_stats[cid]["delta_revision_s2a_minus_s1"] = d_rev
        concept_stats[cid]["delta_checklist_s2b_minus_s2a"] = d_check
        concept_stats[cid]["delta_net_s2b_minus_s0"] = d_net

    # Macro Paired Differences (across 2 concepts)
    macro_differences = {}
    for diff_key in ["delta_stage_s1_minus_s0", "delta_revision_s2a_minus_s1", "delta_checklist_s2b_minus_s2a", "delta_net_s2b_minus_s0"]:
        macro_differences[diff_key] = {}
        for m in metrics_to_average:
            avg_val = (concept_stats["hawkes_process"][diff_key][m] + concept_stats["jump_diffusion"][diff_key][m]) / 2.0
            macro_differences[diff_key][m] = round(avg_val, 4)

    audit_summary = {
        "experiment_id": "RP-HDE-H2-D6",
        "total_audited_series": len(audit_records),
        "condition_statistics": cond_stats,
        "concept_statistics_and_paired_differences": concept_stats,
        "macro_paired_differences": macro_differences,
        "audit_records": audit_records,
    }

    with open(OUTPUT_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    print("\n==================================================")
    print("H2-D6 Post-Generation Audit & Estimator Complete!")
    print(f"Total Audited Series: {len(audit_records)}")
    print("\nCondition Summary:")
    for cond, st in cond_stats.items():
        print(f"  {cond}: Grounded={st['grounded_executable']}, ReqCoverage={st['required_coverage']}, Halluc={st['has_hallucinations']}, StructFid={st['structural_fidelity']}, AutoPASS={st['auto_pass_rate']}")
    print("\nMacro Paired Differences:")
    print("  Delta_stage     (S1  - S0 ):", macro_differences["delta_stage_s1_minus_s0"])
    print("  Delta_revision  (S2a - S1 ):", macro_differences["delta_revision_s2a_minus_s1"])
    print("  Delta_checklist (S2b - S2a):", macro_differences["delta_checklist_s2b_minus_s2a"])
    print("  Delta_net       (S2b - S0 ):", macro_differences["delta_net_s2b_minus_s0"])
    print(f"Saved to: {OUTPUT_AUDIT_PATH}")
    print("==================================================")


if __name__ == "__main__":
    run_post_generation_audit()
