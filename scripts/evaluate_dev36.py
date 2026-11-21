"""Script to evaluate the 36 Development cases against the pre-fixed Dev36 Gold set."""

import json
from pathlib import Path
from research_pipeline.discovery.models import HypothesisCandidate
from research_pipeline.discovery.diagnostic_evaluator import evaluate_diagnostic_hypothesis

DEV36_PATH = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h2_d2b_dev36_gold.json")

with open(DEV36_PATH, encoding="utf-8") as f:
    dev36 = json.load(f)

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
failures = []

for item in dev36:
    sid = item["spec_id"]
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
    else:
        failures.append({
            "spec_id": sid,
            "concept_id": concept,
            "gold_flags": gold_flags,
            "pred_flags": pred_flags,
            "gold_feas": gold_feas,
            "pred_feas": pred_feas,
            "eval_notes": eval_result.evaluator_notes
        })

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

print(f"=== Dev36 Evaluation Summary ===")
print(f"Total Specs: {len(dev36)}")
print(f"Spec-level Exact Matches: {exact_matches}/{len(dev36)} ({exact_matches/len(dev36)*100:.1f}%)")
print()

for axis in ["E_role", "E_math", "E_data", "E_leap", "E_inop"]:
    sens = tp[axis] / (tp[axis] + fn[axis]) if (tp[axis] + fn[axis]) > 0 else 1.0
    spec = tn[axis] / (tn[axis] + fp[axis]) if (tn[axis] + fp[axis]) > 0 else 1.0
    print(f"{axis:8s} | TP: {tp[axis]:2d} | FP: {fp[axis]:2d} | TN: {tn[axis]:2d} | FN: {fn[axis]:2d} | Sens: {sens*100:5.1f}% | Spec: {spec*100:5.1f}%")

print()
print("=== Tri-State Data Feasibility Confusion Matrix ===")
print("Gold \\ Pred     SUPPORTED  UNSUPPORTED    UNKNOWN")
for g in ["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]:
    print(f"{g:12s} {feasibility_cm[g]['SUPPORTED']:10d} {feasibility_cm[g]['UNSUPPORTED']:12d} {feasibility_cm[g]['UNKNOWN']:10d}")

print()
if failures:
    print(f"--- Mismatches ({len(failures)}) ---")
    for f in failures:
        print(f["spec_id"], f["concept_id"])
        print("  Gold Flags:", f["gold_flags"], "Pred Flags:", f["pred_flags"])
        print("  Gold Feas :", f["gold_feas"], "Pred Feas :", f["pred_feas"])
        print("  Notes     :", f["eval_notes"])
else:
    print("ALL 36 DEV CASES MATCH GOLD PERFECTLY (36/36)!")
