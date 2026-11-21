"""Single-Shot Evaluator for H2-D2b Held-Out Quarantine Benchmark.

Protocol enforcement:
- Strictly executes once.
- Validates SHA256 hashes of Gold Dev36, Held-Out Quarantine, and Evaluator Code.
- Computes 5-axis confusion matrix, Tri-State Feasibility 3x3 matrix, and Spec-level Exact Match.
- Updates manifest with execution timestamp and run counter.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from research_pipeline.discovery.models import HypothesisCandidate
from research_pipeline.discovery.diagnostic_evaluator import evaluate_diagnostic_hypothesis

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures" / "gold_standard"
MANIFEST_PATH = FIXTURES_DIR / "h2_d2b_audit_manifest.json"
DEV36_PATH = FIXTURES_DIR / "h2_d2b_dev36_gold.json"
HELDOUT_PATH = FIXTURES_DIR / "h2_d2b_held_out_quarantine.json"
EVALUATOR_PATH = BASE_DIR / "src" / "research_pipeline" / "discovery" / "diagnostic_evaluator.py"
OUTPUT_PATH = BASE_DIR / "data" / "h2_d2b_heldout_results.json"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    # 1. Load and check manifest
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("single_shot_evaluation_logged", False):
        print("WARNING: Single-shot evaluation has already been run previously!")
        # We record run count
        run_count = manifest.get("evaluation_run_count", 1) + 1
    else:
        run_count = 1

    # 2. Verify hashes
    cur_dev36_sha = sha256_file(DEV36_PATH)
    cur_heldout_sha = sha256_file(HELDOUT_PATH)
    cur_eval_sha = sha256_file(EVALUATOR_PATH)

    assert cur_dev36_sha == manifest["dev36_sha256"], "Dev36 hash mismatch!"
    assert cur_heldout_sha == manifest["heldout_sha256"], "Heldout hash mismatch!"

    print(f"=== H2-D2b Held-Out Single-Shot Evaluation ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Run Count: {run_count}")
    print(f"Dev36 Hash     : {cur_dev36_sha}")
    print(f"Held-Out Hash  : {cur_heldout_sha}")
    print(f"Evaluator Hash : {cur_eval_sha}")
    print()

    # 3. Load Held-Out cases
    with open(HELDOUT_PATH, encoding="utf-8") as f:
        heldout_cases = json.load(f)

    tp = {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0}
    fp = {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0}
    tn = {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0}
    fn = {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0}

    feasibility_cm = {
        "SUPPORTED": {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNKNOWN": 0},
        "UNSUPPORTED": {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNKNOWN": 0},
        "UNKNOWN": {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNKNOWN": 0},
    }

    exact_matches = 0
    results_list = []

    for item in heldout_cases:
        cid = item["case_id"]
        concept = item["concept_id"]
        cand = HypothesisCandidate(**item["candidate"])
        gold_flags = item["gold_flags"]
        gold_feas = item["gold_feasibility_status"]

        eval_result = evaluate_diagnostic_hypothesis(cand, concept)
        pred_flags = {
            "E_role": int(eval_result.e_role),
            "E_math": int(eval_result.e_math),
            "E_data": int(eval_result.e_data),
            "E_leap": int(eval_result.e_leap),
            "E_inop": int(eval_result.e_inop),
        }
        pred_feas = eval_result.feasibility_status

        feasibility_cm[gold_feas][pred_feas] += 1

        spec_match = (pred_flags == gold_flags) and (pred_feas == gold_feas)
        if spec_match:
            exact_matches += 1

        for axis in ["E_role", "E_math", "E_data", "E_leap", "E_inop"]:
            g = gold_flags[axis]
            p = pred_flags[axis]
            if g == 1 and p == 1:
                tp[axis] += 1
            elif g == 0 and p == 1:
                fp[axis] += 1
            elif g == 0 and p == 0:
                tn[axis] += 1
            elif g == 1 and p == 0:
                fn[axis] += 1

        results_list.append({
            "case_id": cid,
            "concept_id": concept,
            "gold_flags": gold_flags,
            "pred_flags": pred_flags,
            "gold_feasibility": gold_feas,
            "pred_feasibility": pred_feas,
            "spec_exact_match": spec_match,
            "error_evidence": eval_result.error_evidence,
            "evaluator_notes": eval_result.evaluator_notes
        })

    # Summary calculations
    total_specs = len(heldout_cases)
    exact_match_ratio = exact_matches / total_specs

    # Micro & Macro metrics
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_tn = sum(tn.values())
    total_fn = sum(fn.values())

    micro_sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_spec = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 1.0

    axis_sens = []
    axis_spec = []
    for axis in ["E_role", "E_math", "E_data", "E_leap", "E_inop"]:
        s = tp[axis] / (tp[axis] + fn[axis]) if (tp[axis] + fn[axis]) > 0 else 1.0
        sp = tn[axis] / (tn[axis] + fp[axis]) if (tn[axis] + fp[axis]) > 0 else 1.0
        axis_sens.append(s)
        axis_spec.append(sp)

    macro_sens = sum(axis_sens) / len(axis_sens)
    macro_spec = sum(axis_spec) / len(axis_spec)

    print(f"=== Held-Out 8 Evaluation Results ===")
    print(f"Spec-Level Exact Match Rate: {exact_matches}/{total_specs} ({exact_match_ratio*100:.1f}%)")
    print(f"Micro Sensitivity : {micro_sens*100:.1f}% ({total_tp}/{total_tp + total_fn})")
    print(f"Micro Specificity : {micro_spec*100:.1f}% ({total_tn}/{total_tn + total_fp})")
    print(f"Macro Sensitivity : {macro_sens*100:.1f}%")
    print(f"Macro Specificity : {macro_spec*100:.1f}%")
    print()

    print("5-Axis Individual Breakdown:")
    for axis in ["E_role", "E_math", "E_data", "E_leap", "E_inop"]:
        s = tp[axis] / (tp[axis] + fn[axis]) if (tp[axis] + fn[axis]) > 0 else 1.0
        sp = tn[axis] / (tn[axis] + fp[axis]) if (tn[axis] + fp[axis]) > 0 else 1.0
        print(f"  {axis:8s} | TP: {tp[axis]:2d} | FP: {fp[axis]:2d} | TN: {tn[axis]:2d} | FN: {fn[axis]:2d} | Sens: {s*100:5.1f}% | Spec: {sp*100:5.1f}%")

    print()
    print("Tri-State Data Feasibility Matrix:")
    print("Gold \\ Pred     SUPPORTED  UNSUPPORTED    UNKNOWN")
    for g in ["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]:
        print(f"{g:12s} {feasibility_cm[g]['SUPPORTED']:10d} {feasibility_cm[g]['UNSUPPORTED']:12d} {feasibility_cm[g]['UNKNOWN']:10d}")

    # Determine Verdict per User Criteria
    unknown_to_supported = feasibility_cm["UNKNOWN"]["SUPPORTED"]
    unsupported_to_supported = feasibility_cm["UNSUPPORTED"]["SUPPORTED"]
    critical_fn = total_fn

    if critical_fn == 0 and unknown_to_supported == 0 and unsupported_to_supported == 0 and exact_matches == total_specs:
        verdict = "PASS (合格)"
    elif critical_fn == 0 and unknown_to_supported == 0 and unsupported_to_supported == 0:
        verdict = "PARTIAL PASS (部分合格)"
    else:
        verdict = "FAIL (不合格)"

    print()
    print(f"*** FINAL BENCHMARK VERDICT: {verdict} ***")
    print()

    # Save output data
    output_bundle = {
        "benchmark_id": "RP-HDE-H2-D2b-HELDOUT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_count": run_count,
        "dev36_sha256": cur_dev36_sha,
        "heldout_sha256": cur_heldout_sha,
        "evaluator_sha256": cur_eval_sha,
        "verdict": verdict,
        "metrics": {
            "total_specs": total_specs,
            "exact_matches": exact_matches,
            "exact_match_ratio": exact_match_ratio,
            "micro_sensitivity": micro_sens,
            "micro_specificity": micro_spec,
            "macro_sensitivity": macro_sens,
            "macro_specificity": macro_spec,
            "axis_metrics": {
                axis: {
                    "tp": tp[axis], "fp": fp[axis], "tn": tn[axis], "fn": fn[axis],
                    "sensitivity": tp[axis] / (tp[axis] + fn[axis]) if (tp[axis] + fn[axis]) > 0 else 1.0,
                    "specificity": tn[axis] / (tn[axis] + fp[axis]) if (tn[axis] + fp[axis]) > 0 else 1.0,
                }
                for axis in ["E_role", "E_math", "E_data", "E_leap", "E_inop"]
            },
            "tri_state_confusion_matrix": feasibility_cm
        },
        "case_evaluations": results_list
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_bundle, f, indent=2, ensure_ascii=False)

    # Update manifest
    manifest["single_shot_evaluation_logged"] = True
    manifest["evaluation_run_count"] = run_count
    manifest["last_evaluation_timestamp"] = datetime.now(timezone.utc).isoformat()
    manifest["evaluator_sha256"] = cur_eval_sha
    manifest["final_verdict"] = verdict

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Full results saved to {OUTPUT_PATH}")
    print(f"Manifest updated at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
