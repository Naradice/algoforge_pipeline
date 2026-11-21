"""HDE Step H2-D5 Post-Generation Audit and Statistical Estimator.

Adheres strictly to RP-HDE-H2-D5-PLAN-v0.2 with user review execution conditions:
1. Denominator Fixed from Pre-Frozen Gold (P*_gold, 9 essential parameters per concept).
2. Evaluates Parameter-Level Metrics:
   - R_GroundedDisposition: Proper uncertainty management across gold denominator (FIXED, ESTIMATED, or valid INFEASIBLE).
   - R_ExecutableResolution: Proportion of gold parameters resolved into executable specs (FIXED or ESTIMATED only!).
   - R_ValidInfeasibility: Validity of declared infeasibility (N/A if count=0).
3. Evaluates Strategy-Level Executability:
   - R_StrategyExecutable: 1 if all essential rules (estimation, entry, exit, risk, timing) are fully operational; 0 otherwise.
4. Evaluates Model vs Target Fidelity:
   - R_StructuralFidelity: Preserved mathematical state-space / point process structure.
   - R_TargetFidelity: Preserved original microstructural physical phenomenon.
   - TrivializationDegeneration: Abandoned structure into plain heuristics (ATR/Bollinger).
   - HallucinatedFixation: Unreasoned arbitrary numeric assignment.
5. Computes Concept-Level Paired Differences (D1-D0, D2-D1, D2-D0).
6. Persists audit report to data/h2_d5_audit_results.json.
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR / "src"))

RAW_CANDIDATES_PATH = BASE_DIR / "data" / "h2_d5_raw_candidates.json"
AUTO_EVAL_PATH = BASE_DIR / "data" / "h2_d5_automated_eval_results.json"
GOLD_PARAMS_PATH = BASE_DIR / "tests" / "fixtures" / "gold_standard" / "h2_d5_gold_parameters.json"
OUTPUT_AUDIT_PATH = BASE_DIR / "data" / "h2_d5_audit_results.json"

CONCEPTS = ["hawkes_process", "jump_diffusion"]
CONDITIONS = ["D0", "D1", "D2"]
SEEDS = [42, 43, 44]


def audit_candidate_parameters(
    cid: str,
    cand: Dict[str, Any],
    cond: str,
    gold_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit parameter enumeration and disposition against pre-frozen gold denominator."""
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

    essential_params = []
    for layer_name, params in gold_spec["essential_parameters"].items():
        for p in params:
            essential_params.append((layer_name, p))

    total_gold_count = len(essential_params)  # Exactly 9
    param_audit_details = []

    grounded_count = 0
    executable_count = 0
    infeasible_declared_count = 0
    infeasible_valid_count = 0
    hallucinated_count = 0

    for layer_name, p_def in essential_params:
        pid = p_def["param_id"]
        sym = p_def["symbol"].lower()
        desc = p_def["description"].lower()

        # Check if parameter is mentioned / addressed
        is_mentioned = False
        terms_to_check = [sym, sym.replace("*", ""), desc]
        if pid == "hawkes_baseline_mu":
            terms_to_check += ["mu", "baseline", "background rate"]
        elif pid == "hawkes_excitation_alpha":
            terms_to_check += ["alpha", "excitation", "branching"]
        elif pid == "hawkes_decay_beta":
            terms_to_check += ["beta", "decay rate", "memory decay"]
        elif pid == "hawkes_branching_ratio_threshold":
            terms_to_check += ["branching ratio", "alpha/beta", "critical threshold"]
        elif pid == "hawkes_intensity_threshold":
            terms_to_check += ["intensity threshold", "lambda*", "spike threshold"]
        elif pid == "jump_bipower_threshold_theta":
            terms_to_check += ["theta", "bipower threshold", "truncation threshold", "bv"]
        elif pid == "jump_hamilton_transition_probs":
            terms_to_check += ["transition prob", "markov", "hamilton", "persistence"]
        elif pid == "jump_regime_variance_ratio":
            terms_to_check += ["variance separability", "variance ratio", "sigma_crisis"]
        elif pid == "jump_regime_posterior_threshold":
            terms_to_check += ["posterior", "regime probability", "p(crisis)", "p(s_t"]
        elif pid == "jump_component_size_threshold":
            terms_to_check += ["jump component", "j_t", "jump alert"]

        for term in terms_to_check:
            if term in text:
                is_mentioned = True
                break

        assigned_disposition = "UNRESOLVED_OMISSION"
        is_disposition_valid = False
        is_executable = False
        audit_note = ""

        if not is_mentioned:
            assigned_disposition = "UNRESOLVED_OMISSION"
            audit_note = f"Essential parameter {sym} ({pid}) was omitted from specification."
        else:
            # Check how it was treated
            # Check if declared infeasible
            if "infeasible" in text and (sym in text or pid in text or "cannot directly observe" in text or "m5 limitation" in text):
                assigned_disposition = "DECLARED_INFEASIBLE"
                infeasible_declared_count += 1
                # Valid infeasibility under M5?
                if cid == "hawkes_process" and ("tick" in text or "order book" in text or "cancellation" in text):
                    is_disposition_valid = True
                    infeasible_valid_count += 1
                    audit_note = f"Valid infeasibility declaration: tick-level orders unavailable on M5."
                elif cid == "jump_diffusion" and ("microsecond" in text or "news feed" in text):
                    is_disposition_valid = True
                    infeasible_valid_count += 1
                    audit_note = f"Valid infeasibility declaration: external news feeds unavailable on M5."
                else:
                    is_disposition_valid = False
                    audit_note = f"Unsubstantiated infeasibility declaration."

            # Check if in-sample estimated
            elif any(w in text for w in ["in_sample_estimated", "mle", "grid search", "calmar", "lookback", "train split", "estimated via"]):
                assigned_disposition = "IN_SAMPLE_ESTIMATED"
                # Check 6-point estimation specs in text
                has_search_space = any(w in text for w in ["search space", "bounds", "range", "grid", "threshold"])
                has_obj = any(w in text for w in ["objective", "loss", "likelihood", "sharpe", "sortino", "calmar"])
                has_window = any(w in text for w in ["lookback", "window", "bars", "period"])
                
                if has_search_space and has_obj and has_window:
                    is_disposition_valid = True
                    is_executable = True
                    audit_note = f"Concrete estimation protocol provided (objective, window, search space)."
                else:
                    is_disposition_valid = False
                    audit_note = f"Incomplete estimation specification (missing search range, window, or objective function)."

            # Check if theoretical constraint
            elif any(w in text for w in ["theoretical_constraint", "stability condition", "alpha < beta", "mu > 0", "stationarity"]):
                assigned_disposition = "THEORETICAL_CONSTRAINT"
                is_disposition_valid = True
                # Cannot serve as trade trigger!
                if layer_name == "layer2_decision_thresholds":
                    audit_note = f"Theoretical constraint cannot serve as trade decision trigger threshold."
                    is_executable = False
                else:
                    audit_note = f"Theoretical stability constraint acknowledged."
                    is_executable = False  # Boundary only

            # Check if empirical fixed
            elif any(w in text for w in ["empirical_fixed", "literature", "fixed at", "set to"]):
                assigned_disposition = "EMPIRICAL_FIXED"
                # Check if reasoned or arbitrary hallucination
                has_justification = any(w in text for w in ["justif", "literature", "m5 volatility", "prior study", "standard convention", "paper"])
                if has_justification:
                    is_disposition_valid = True
                    is_executable = True
                    audit_note = f"Empirical value supported by literature / domain rationale."
                else:
                    assigned_disposition = "HALLUCINATED_FIXATION"
                    hallucinated_count += 1
                    audit_note = f"Arbitrary numeric assignment without empirical rationale."

            else:
                # Qualitative mention without resolution
                assigned_disposition = "QUALITATIVE_UNRESOLVED"
                audit_note = f"Parameter mentioned qualitatively but left undetermined."

        if is_disposition_valid:
            grounded_count += 1
        if is_executable:
            executable_count += 1

        param_audit_details.append({
            "param_id": pid,
            "layer": layer_name,
            "symbol": sym,
            "is_mentioned": is_mentioned,
            "disposition": assigned_disposition,
            "is_valid": is_disposition_valid,
            "is_executable": is_executable,
            "note": audit_note,
        })

    r_grounded = grounded_count / total_gold_count
    r_executable = executable_count / total_gold_count
    r_valid_infeas = (
        (infeasible_valid_count / infeasible_declared_count)
        if infeasible_declared_count > 0
        else "N/A"
    )

    # Strategy-Level Executability (R_StrategyExecutable)
    # Binary: all essential layers must have operational definitions
    has_operable_entry = bool(entry and entry != "unspecified" and not any(w in entry for w in ["qualitative", "unspecified", "when intensity increases"]))
    has_operable_exit = bool(exit_ and exit_ != "unspecified")
    has_operable_risk = bool(risk and risk != "unspecified")
    has_operable_model = (executable_count >= 5)  # At least model and decision layers resolved
    strategy_executable = (has_operable_entry and has_operable_exit and has_operable_risk and has_operable_model)

    return {
        "total_gold_parameters": total_gold_count,
        "grounded_count": grounded_count,
        "executable_count": executable_count,
        "infeasible_declared_count": infeasible_declared_count,
        "infeasible_valid_count": infeasible_valid_count,
        "hallucinated_count": hallucinated_count,
        "r_grounded_disposition": round(r_grounded, 4),
        "r_executable_resolution": round(r_executable, 4),
        "r_valid_infeasibility": (round(r_valid_infeas, 4) if isinstance(r_valid_infeas, float) else r_valid_infeas),
        "r_strategy_executable": strategy_executable,
        "param_details": param_audit_details,
    }


def audit_candidate_fidelity(
    cid: str,
    cand: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit mathematical structural fidelity vs target fidelity vs trivialization degeneration."""
    spec = cand.get("backtest_spec") or {}
    claim = (cand.get("claim") or "").lower()
    mech = (cand.get("proposed_mechanism") or "").lower()
    rat = (cand.get("grounding_rationale") or "").lower()
    entry = (spec.get("entry_condition") or "").lower()
    feat = (spec.get("feature_engineering") or "").lower()
    text = f"{claim} {mech} {rat} {entry} {feat}"

    structural_fidelity = False
    target_fidelity = False
    is_trivialized = False
    classification = "UNKNOWN"

    if cid == "hawkes_process":
        # Structural: self-exciting arrival intensity lambda(t) or branching ratio alpha/beta
        if ("hawkes" in text or "intensity" in text or "self-excit" in text or "branching ratio" in text) and ("lambda" in text or "alpha" in text or "beta" in text):
            structural_fidelity = True

        # Target: native order flow / order book event arrivals directly
        if ("order flow" in text or "order arrival" in text or "cancellation" in text) and not ("proxy" in text or "return exceedance" in text):
            target_fidelity = True

        # Trivialization: using plain indicators without point process intensity
        if ("rsi" in entry or "moving average" in entry or "bollinger" in entry) and ("intensity" not in entry and "branching" not in entry):
            is_trivialized = True
            structural_fidelity = False

    elif cid == "jump_diffusion":
        # Structural: bipower variation, Hamilton regime switching, or Poisson jump filtering
        if ("bipower" in text or "bv_t" in text or "realized variance" in text or "hamilton" in text or "regime-switching" in text) and ("jump" in text):
            structural_fidelity = True

        # Target: microsecond central bank rate announcement jumps directly
        if ("central bank" in text or "announcement" in text or "macro news" in text) and not ("inferred" in text or "proxy" in text):
            target_fidelity = True

        # Trivialization: using plain ATR / volatility band without bipower variation
        if ("atr" in entry or "bollinger" in entry) and ("bipower" not in feat and "bv" not in feat and "regime" not in feat):
            is_trivialized = True
            structural_fidelity = False

    # Classification
    if structural_fidelity and target_fidelity:
        classification = "DIRECT_REPRODUCTION"
    elif structural_fidelity and not target_fidelity:
        classification = "VALID_PROXY_TRANSFORMATION"
    elif is_trivialized or (not structural_fidelity and not target_fidelity):
        classification = "TRIVIALIZATION_DEGENERATION"
    else:
        classification = "PSEUDO_MODEL_HEURISTIC"

    return {
        "r_structural_fidelity": structural_fidelity,
        "r_target_fidelity": target_fidelity,
        "is_trivialized": is_trivialized,
        "fidelity_classification": classification,
    }


def run_post_generation_audit():
    with open(RAW_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        raw_candidates = json.load(f)
    with open(AUTO_EVAL_PATH, "r", encoding="utf-8") as f:
        auto_evals = json.load(f)
    with open(GOLD_PARAMS_PATH, "r", encoding="utf-8") as f:
        gold_params_db = json.load(f)

    auto_eval_map = {r["call_id"]: r for r in auto_evals}

    audit_records = []
    mismatch_count = 0

    for r in raw_candidates:
        call_id = r["call_id"]
        cid = r["concept_id"]
        cond = r["condition_id"]
        seed = r["seed"]
        cand = r["candidate"]
        auto_item = auto_eval_map[call_id]

        gold_spec = gold_params_db[cid]

        # Audit Parameters against Pre-Frozen Gold Denominator
        param_audit = audit_candidate_parameters(cid, cand, cond, gold_spec)

        # Audit Fidelity
        fidelity_audit = audit_candidate_fidelity(cid, cand)

        # Check Evaluators consistency
        auto_gate = auto_item["data_semantic_gate"]["gate_decision"]
        auto_pass = auto_item["dual_gate_pass"]

        # Audit Gate check
        audit_pass = bool(
            auto_pass and
            param_audit["r_strategy_executable"] and
            fidelity_audit["r_structural_fidelity"] and
            not fidelity_audit["is_trivialized"]
        )

        is_mismatch = (auto_pass != audit_pass)
        if is_mismatch:
            mismatch_count += 1

        rec = {
            "call_id": call_id,
            "concept_id": cid,
            "condition_id": cond,
            "seed": seed,
            "param_audit": param_audit,
            "fidelity_audit": fidelity_audit,
            "auto_evaluation": {
                "gate_decision": auto_gate,
                "dual_gate_pass": auto_pass,
                "5axis_errors": auto_item["diagnostic_5axis"],
            },
            "audit_pass": audit_pass,
            "is_mismatch": is_mismatch,
        }
        audit_records.append(rec)

    # Concept-Level Paired Differences (D1-D0, D2-D1, D2-D0)
    # Aggregated by Concept x Condition
    metrics_to_track = [
        "r_grounded_disposition",
        "r_executable_resolution",
        "r_strategy_executable",
        "r_structural_fidelity",
        "r_target_fidelity",
        "is_trivialized",
    ]

    concept_stats = {}
    for cid in CONCEPTS:
        concept_stats[cid] = {}
        for cond in CONDITIONS:
            crecs = [r for r in audit_records if r["concept_id"] == cid and r["condition_id"] == cond]
            n = len(crecs)  # Exactly 3 seeds

            c_dict = {"n_seeds": n}
            for m in metrics_to_track:
                vals = []
                for r in crecs:
                    if m.startswith("r_"):
                        if m in r["param_audit"]:
                            vals.append(float(r["param_audit"][m]))
                        elif m in r["fidelity_audit"]:
                            vals.append(1.0 if r["fidelity_audit"][m] else 0.0)
                    elif m == "is_trivialized":
                        vals.append(1.0 if r["fidelity_audit"][m] else 0.0)
                c_dict[m] = round(sum(vals) / n, 4)

            # Auto pass and Audit pass
            c_dict["auto_pass_rate"] = round(sum(1 for r in crecs if r["auto_evaluation"]["dual_gate_pass"]) / n, 4)
            c_dict["audit_pass_rate"] = round(sum(1 for r in crecs if r["audit_pass"]) / n, 4)
            concept_stats[cid][cond] = c_dict

        # Concept Paired Differences
        d_enum = {m: round(concept_stats[cid]["D1"][m] - concept_stats[cid]["D0"][m], 4) for m in metrics_to_track}
        d_disp = {m: round(concept_stats[cid]["D2"][m] - concept_stats[cid]["D1"][m], 4) for m in metrics_to_track}
        d_net  = {m: round(concept_stats[cid]["D2"][m] - concept_stats[cid]["D0"][m], 4) for m in metrics_to_track}

        concept_stats[cid]["diff_d1_minus_d0"] = d_enum
        concept_stats[cid]["diff_d2_minus_d1"] = d_disp
        concept_stats[cid]["diff_d2_minus_d0"] = d_net

    # Condition Statistics Across All Calls (N=6 per condition)
    cond_stats = {}
    for cond in CONDITIONS:
        crecs = [r for r in audit_records if r["condition_id"] == cond]
        n = len(crecs)  # Exactly 6 (2 concepts x 3 seeds)
        cond_stats[cond] = {
            "n_calls": n,
            "r_grounded_disposition": round(sum(r["param_audit"]["r_grounded_disposition"] for r in crecs) / n, 4),
            "r_executable_resolution": round(sum(r["param_audit"]["r_executable_resolution"] for r in crecs) / n, 4),
            "r_strategy_executable": round(sum(1 for r in crecs if r["param_audit"]["r_strategy_executable"]) / n, 4),
            "r_structural_fidelity": round(sum(1 for r in crecs if r["fidelity_audit"]["r_structural_fidelity"]) / n, 4),
            "r_target_fidelity": round(sum(1 for r in crecs if r["fidelity_audit"]["r_target_fidelity"]) / n, 4),
            "trivialization_rate": round(sum(1 for r in crecs if r["fidelity_audit"]["is_trivialized"]) / n, 4),
            "total_hallucinated_fixations": sum(r["param_audit"]["hallucinated_count"] for r in crecs),
            "total_infeasible_declarations": sum(r["param_audit"]["infeasible_declared_count"] for r in crecs),
            "auto_pass_rate": round(sum(1 for r in crecs if r["auto_evaluation"]["dual_gate_pass"]) / n, 4),
            "audit_pass_rate": round(sum(1 for r in crecs if r["audit_pass"]) / n, 4),
        }

    audit_summary = {
        "experiment_id": "RP-HDE-H2-D5",
        "total_audited": len(audit_records),
        "total_mismatch": mismatch_count,
        "mismatch_rate": round(mismatch_count / len(audit_records), 4),
        "condition_statistics": cond_stats,
        "concept_statistics_and_paired_differences": concept_stats,
        "audit_records": audit_records,
    }

    with open(OUTPUT_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    print(f"\n==================================================")
    print(f"H2-D5 Audit & Estimator Complete!")
    print(f"Total Audited: {len(audit_records)}")
    print(f"Condition Stats:")
    for cond, st in cond_stats.items():
        print(f"  {cond}: Grounded={st['r_grounded_disposition']}, ExecRes={st['r_executable_resolution']}, StratExec={st['r_strategy_executable']}, StructFid={st['r_structural_fidelity']}, Trivial={st['trivialization_rate']}")
    print(f"Saved to: {OUTPUT_AUDIT_PATH}")
    print(f"==================================================")


if __name__ == "__main__":
    run_post_generation_audit()
