"""HDE Step H2-D4 Post-Generation Audit and Statistical Estimator.

Adheres strictly to RP-HDE-H2-D4-PLAN-v0.2:
1. Conducts text-level independent audit across all 36 generated candidates.
2. Evaluates Research Intent metrics:
   - R_OriginalIntent: Preserved original research intent (e.g. inventory liquidation, point process cascades).
   - R_ValidTransformation: Formulated compliant M5 proxy specification with proxy targets and constraints.
   - R_InfeasibleRecognition: Explicitly acknowledged M5 data limitations (non-hallucinatory recognition).
   - R_InvalidSubstitution: Fallacious substitution (e.g. execution as alpha entry, treating intensity as price forecast).
3. Cross-audits Automated Dual-Gate (5-axis + 2D Gate) decisions and calculates Mismatch Rate.
4. Calculates Concept-Level Paired Differences d_i (Structure Effect: C2 - C1) and e_i (Information Effect: C1 - C0).
5. Computes Macro Estimators Delta_struct and Delta_info for both Primary (Automated) and Sensitivity (Audited) sets.
6. Persists audit report to data/h2_d4_audit_results.json.
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR / "src"))

from research_pipeline.discovery.models import HypothesisCandidate

RAW_CANDIDATES_PATH = BASE_DIR / "data" / "h2_d4_raw_candidates.json"
AUTO_EVAL_PATH = BASE_DIR / "data" / "h2_d4_automated_eval_results.json"
INTENT_PATH = BASE_DIR / "tests" / "fixtures" / "gold_standard" / "h2_d4_intent_criteria.json"
OUTPUT_AUDIT_PATH = BASE_DIR / "data" / "h2_d4_audit_results.json"

CONCEPTS = ["almgren_chriss", "hawkes_process", "irm", "jump_diffusion"]
CONDITIONS = ["C0", "C1", "C2"]
SEEDS = [42, 43, 44]


def audit_candidate_intent(
    cid: str,
    cand: Dict[str, Any],
    criteria: Dict[str, str],
) -> Dict[str, bool]:
    """Audit research intent preservation, valid transformation, and infeasible recognition."""
    claim = (cand.get("claim") or "").lower()
    mech = (cand.get("proposed_mechanism") or "").lower()
    rat = (cand.get("grounding_rationale") or "").lower()
    phenom = (cand.get("target_phenomenon") or "").lower()
    spec = cand.get("backtest_spec") or {}
    entry = (spec.get("entry_condition") or "").lower()
    exit_ = (spec.get("exit_condition") or "").lower()
    feat = (spec.get("feature_engineering") or "").lower()
    risk = (spec.get("risk_management") or "").lower()
    falsif = (spec.get("falsification_metric") or "").lower()

    text = f"{claim} {mech} {rat} {phenom} {entry} {exit_} {feat} {risk} {falsif}"

    orig_intent = False
    valid_trans = False
    infeas_rec = False
    invalid_sub = False

    if cid == "almgren_chriss":
        # Original Intent: Liquidation of pre-existing inventory / execution variance
        has_liquidation_terms = any(w in text for w in ["liquidation", "inventory", "liquidate", "execution slippage", "execution cost", "variance of cost"])
        is_speculative_entry = any(w in entry for w in ["enter a trade when", "buy when", "enter long", "enter short", "initiate a trade when", "buy usdjpy", "favorable price movement"])
        
        if has_liquidation_terms and not is_speculative_entry:
            orig_intent = True
        
        # Valid transformation: adaptive pacing / execution proxy on M5
        if ("schedule" in text or "trajectory" in text or "pacing" in text) and ("m5" in text or "bar" in text):
            valid_trans = True
            
        # Infeasible recognition: notes need for order book or fill logs
        if any(w in text for w in ["order book", "fill log", "limitations of m5", "ohlcv alone", "without order book", "l2", "depth"]):
            infeas_rec = True
            
        if is_speculative_entry or "speculative" in entry or "signal based on liquidation trajectory" in entry:
            invalid_sub = True

    elif cid == "hawkes_process":
        # Original Intent: Clustering of events, cascade risk, point process
        if any(w in text for w in ["cascade", "clustering", "intensity", "point process", "self-excit", "endogenous"]):
            orig_intent = True
            
        # Valid transformation: M5 return exceedance proxy
        if any(w in text for w in ["exceedance", "rolling_std", "return threshold", "volatility jump", "proxy", "jump clustering"]):
            valid_trans = True
            
        # Infeasible recognition
        if any(w in text for w in ["tick", "order book", "limitations of m5", "cannot directly observe", "native order", "ohlcv"]):
            infeas_rec = True
            
        # Invalid substitution: treating lambda as directional return or execution router
        if "directional price return" in text or "portfolio weight allocation" in text or "execution router" in text:
            invalid_sub = True

    elif cid == "irm":
        # Original Intent: Invariant representation across environments / regimes
        if any(w in text for w in ["invariant", "distribution shift", "regime shift", "environment", "out-of-distribution"]):
            orig_intent = True
            
        # Valid transformation: session partitioning (Asian/European/US) on M5
        if any(w in text for w in ["session", "asian", "european", "us", "partition", "environment"]):
            valid_trans = True
            
        # Infeasible recognition
        if any(w in text for w in ["order book", "l2", "l3", "queue depth", "tick trade signs", "limitations"]):
            infeas_rec = True
            
        # Invalid substitution: execution routing engine or claiming zero risk
        if "execution routing engine" in text or "eliminates all risk" in text or "routing child orders" in text:
            invalid_sub = True

    elif cid == "jump_diffusion":
        # Original Intent: Filtering Poisson jumps from Brownian diffusion
        if any(w in text for w in ["jump", "poisson", "diffusion", "regime", "bipower", "realized variance"]):
            orig_intent = True
            
        # Valid transformation: Bipower variation / realized variance on M5
        if any(w in text for w in ["bipower", "realized variance", "rv_t", "bv_t", "mancini", "hamilton"]):
            valid_trans = True
            
        # Infeasible recognition
        if any(w in text for w in ["microsecond", "external news", "central bank news feed", "limitations of m5"]):
            infeas_rec = True
            
        # Invalid substitution: deterministic price surges or order queue placement
        if "deterministic price surge" in text or "order queue placement" in text or "execution routing engine" in text:
            invalid_sub = True

    return {
        "is_original_intent": orig_intent,
        "is_valid_transformation": valid_trans,
        "is_infeasible_recognition": infeas_rec,
        "is_invalid_substitution": invalid_sub,
    }


def audit_dual_gate_accuracy(
    cand: Dict[str, Any],
    auto_eval: Dict[str, Any],
    cid: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Audit accuracy of automated dual-gate decision against verbatim candidate text."""
    spec = cand.get("backtest_spec") or {}
    claim = (cand.get("claim") or "").lower()
    mech = (cand.get("proposed_mechanism") or "").lower()
    rat = (cand.get("grounding_rationale") or "").lower()
    entry = (spec.get("entry_condition") or "").lower()
    exit_ = (spec.get("exit_condition") or "").lower()
    feat = (spec.get("feature_engineering") or "").lower()
    text = f"{claim} {mech} {rat} {entry} {exit_} {feat}"

    auto_gate_dec = auto_eval["data_semantic_gate"]["gate_decision"]
    auto_pass = auto_eval["dual_gate_pass"]
    diag = auto_eval["diagnostic_5axis"]

    # Audited evaluations
    audit_data_feas = "SUPPORTED"
    audit_semantic = "EXACT"
    audit_temporal = "VALID"
    audit_e_role = diag.get("e_role", False)
    audit_e_math = diag.get("e_math", False)
    audit_e_data = diag.get("e_data", False)
    audit_e_leap = diag.get("e_leap", False)
    audit_e_inop = diag.get("e_inop", False)

    # 1. Check Data Feasibility
    # If explicitly requires L2/L3 order book or tick data without proxy -> UNSUPPORTED
    unsupported_data_cues = ["raw l2 order book", "l3 queue depth", "microsecond tick feed", "level 2 book", "tick order arrival sequence"]
    if any(cue in feat or cue in entry for cue in unsupported_data_cues):
        if not ("proxy" in text or "derived from m5" in text):
            audit_data_feas = "UNSUPPORTED"
            audit_e_data = True

    # 2. Check Temporal Validity
    if "today's daily close" in text or "current day's close" in text or "retroactive execution at same-bar open" in text:
        audit_temporal = "LOOKAHEAD_LEAK"

    # 3. Check Role misconception
    if cid == "almgren_chriss":
        spec_entry = any(w in entry for w in ["enter a trade when", "buy when", "enter long", "initiate a trade when", "buy usdjpy aggressively"])
        if spec_entry:
            audit_e_role = True
            audit_semantic = "INVALID"

    # Audited Gate Decision
    if audit_data_feas == "UNSUPPORTED" or audit_semantic == "INVALID" or audit_temporal == "LOOKAHEAD_LEAK":
        audit_gate_dec = "BLOCK"
    elif audit_data_feas == "UNKNOWN" or audit_semantic == "UNKNOWN":
        audit_gate_dec = "REVIEW"
    else:
        audit_gate_dec = "PASS"

    has_5axis = audit_e_role or audit_e_math or audit_e_data or audit_e_leap or audit_e_inop
    audit_dual_pass = (not has_5axis) and (audit_gate_dec == "PASS")

    is_mismatch = (auto_gate_dec != audit_gate_dec) or (auto_pass != audit_dual_pass)
    mismatch_reason = ""
    if is_mismatch:
        mismatch_reason = f"Mismatch: Auto(gate={auto_gate_dec}, pass={auto_pass}) vs Audit(gate={audit_gate_dec}, pass={audit_dual_pass})"

    audit_details = {
        "audited_gate_decision": audit_gate_dec,
        "audited_dual_pass": audit_dual_pass,
        "audited_data_feasibility": audit_data_feas,
        "audited_semantic_fidelity": audit_semantic,
        "audited_temporal_validity": audit_temporal,
        "audited_e_role": audit_e_role,
        "audited_e_math": audit_e_math,
        "audited_e_data": audit_e_data,
        "audited_e_leap": audit_e_leap,
        "audited_e_inop": audit_e_inop,
    }

    return is_mismatch, mismatch_reason, audit_details


def run_post_generation_audit():
    with open(RAW_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        raw_candidates = json.load(f)
    with open(AUTO_EVAL_PATH, "r", encoding="utf-8") as f:
        auto_evals = json.load(f)
    with open(INTENT_PATH, "r", encoding="utf-8") as f:
        intent_criteria = json.load(f)

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

        # Audit Intent
        intent_res = audit_candidate_intent(cid, cand, intent_criteria.get(cid, {}))

        # Audit Gate Accuracy
        is_mismatch, mismatch_reason, audit_details = audit_dual_gate_accuracy(cand, auto_item, cid)
        if is_mismatch:
            mismatch_count += 1

        rec = {
            "call_id": call_id,
            "concept_id": cid,
            "condition_id": cond,
            "seed": seed,
            "intent_audit": intent_res,
            "auto_evaluation": {
                "gate_decision": auto_item["data_semantic_gate"]["gate_decision"],
                "dual_gate_pass": auto_item["dual_gate_pass"],
                "5axis_errors": auto_item["diagnostic_5axis"],
            },
            "audited_evaluation": audit_details,
            "is_mismatch": is_mismatch,
            "mismatch_reason": mismatch_reason,
        }
        audit_records.append(rec)

    # Statistical Aggregations: Concept-level paired differences
    # N=4 concepts x 3 seeds
    by_concept = {c: {"C0": [], "C1": [], "C2": []} for c in CONCEPTS}
    by_concept_audited = {c: {"C0": [], "C1": [], "C2": []} for c in CONCEPTS}

    for rec in audit_records:
        c = rec["concept_id"]
        cond = rec["condition_id"]
        auto_val = 1.0 if rec["auto_evaluation"]["dual_gate_pass"] else 0.0
        audit_val = 1.0 if rec["audited_evaluation"]["audited_dual_pass"] else 0.0
        by_concept[c][cond].append(auto_val)
        by_concept_audited[c][cond].append(audit_val)

    # Primary Estimators (Automated)
    d_struct_auto = {}  # C2 - C1
    e_info_auto = {}    # C1 - C0
    for c in CONCEPTS:
        mean_c0 = sum(by_concept[c]["C0"]) / len(by_concept[c]["C0"])
        mean_c1 = sum(by_concept[c]["C1"]) / len(by_concept[c]["C1"])
        mean_c2 = sum(by_concept[c]["C2"]) / len(by_concept[c]["C2"])
        d_struct_auto[c] = mean_c2 - mean_c1
        e_info_auto[c] = mean_c1 - mean_c0

    delta_struct_auto = sum(d_struct_auto.values()) / len(d_struct_auto)
    delta_info_auto = sum(e_info_auto.values()) / len(e_info_auto)

    # Sensitivity Estimators (Audited)
    d_struct_audited = {}
    e_info_audited = {}
    for c in CONCEPTS:
        mean_c0 = sum(by_concept_audited[c]["C0"]) / len(by_concept_audited[c]["C0"])
        mean_c1 = sum(by_concept_audited[c]["C1"]) / len(by_concept_audited[c]["C1"])
        mean_c2 = sum(by_concept_audited[c]["C2"]) / len(by_concept_audited[c]["C2"])
        d_struct_audited[c] = mean_c2 - mean_c1
        e_info_audited[c] = mean_c1 - mean_c0

    delta_struct_audited = sum(d_struct_audited.values()) / len(d_struct_audited)
    delta_info_audited = sum(e_info_audited.values()) / len(e_info_audited)

    # Rate per condition (N=12 each)
    cond_stats = {}
    for cond in CONDITIONS:
        crecs = [r for r in audit_records if r["condition_id"] == cond]
        n = len(crecs)
        pass_auto = sum(1 for r in crecs if r["auto_evaluation"]["dual_gate_pass"]) / n
        pass_audited = sum(1 for r in crecs if r["audited_evaluation"]["audited_dual_pass"]) / n
        orig_intent = sum(1 for r in crecs if r["intent_audit"]["is_original_intent"]) / n
        valid_trans = sum(1 for r in crecs if r["intent_audit"]["is_valid_transformation"]) / n
        infeas_rec = sum(1 for r in crecs if r["intent_audit"]["is_infeasible_recognition"]) / n
        invalid_sub = sum(1 for r in crecs if r["intent_audit"]["is_invalid_substitution"]) / n

        e_role_cnt = sum(1 for r in crecs if r["auto_evaluation"]["5axis_errors"]["e_role"])
        e_math_cnt = sum(1 for r in crecs if r["auto_evaluation"]["5axis_errors"]["e_math"])
        e_data_cnt = sum(1 for r in crecs if r["auto_evaluation"]["5axis_errors"]["e_data"])
        e_leap_cnt = sum(1 for r in crecs if r["auto_evaluation"]["5axis_errors"]["e_leap"])
        e_inop_cnt = sum(1 for r in crecs if r["auto_evaluation"]["5axis_errors"]["e_inop"])

        cond_stats[cond] = {
            "n_calls": n,
            "pass_rate_auto": round(pass_auto, 4),
            "pass_rate_audited": round(pass_audited, 4),
            "original_intent_rate": round(orig_intent, 4),
            "valid_transformation_rate": round(valid_trans, 4),
            "infeasible_recognition_rate": round(infeas_rec, 4),
            "invalid_substitution_rate": round(invalid_sub, 4),
            "5axis_error_counts": {
                "e_role": e_role_cnt,
                "e_math": e_math_cnt,
                "e_data": e_data_cnt,
                "e_leap": e_leap_cnt,
                "e_inop": e_inop_cnt,
            }
        }

    mismatch_rate = mismatch_count / len(audit_records)

    audit_summary = {
        "experiment_id": "RP-HDE-H2-D4",
        "total_audited": len(audit_records),
        "mismatch_count": mismatch_count,
        "mismatch_rate": round(mismatch_rate, 4),
        "primary_analysis_automated": {
            "d_struct_by_concept": {c: round(d_struct_auto[c], 4) for c in CONCEPTS},
            "e_info_by_concept": {c: round(e_info_auto[c], 4) for c in CONCEPTS},
            "macro_delta_struct": round(delta_struct_auto, 4),
            "macro_delta_info": round(delta_info_auto, 4),
        },
        "sensitivity_analysis_audited": {
            "d_struct_by_concept": {c: round(d_struct_audited[c], 4) for c in CONCEPTS},
            "e_info_by_concept": {c: round(e_info_audited[c], 4) for c in CONCEPTS},
            "macro_delta_struct": round(delta_struct_audited, 4),
            "macro_delta_info": round(delta_info_audited, 4),
        },
        "condition_statistics": cond_stats,
        "audit_records": audit_records,
    }

    with open(OUTPUT_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    print(f"\n==================================================")
    print(f"H2-D4 Audit & Statistical Analysis Complete!")
    print(f"Total Audited: {len(audit_records)}")
    print(f"Mismatch Count: {mismatch_count}/{len(audit_records)} ({mismatch_rate*100:.1f}%)")
    print(f"Primary Delta_struct (C2 - C1): {delta_struct_auto:+.4f}")
    print(f"Primary Delta_info   (C1 - C0): {delta_info_auto:+.4f}")
    print(f"Sensitivity Delta_struct (Audited): {delta_struct_audited:+.4f}")
    print(f"Sensitivity Delta_info   (Audited): {delta_info_audited:+.4f}")
    print(f"Saved to: {OUTPUT_AUDIT_PATH}")
    print(f"==================================================")


if __name__ == "__main__":
    run_post_generation_audit()
