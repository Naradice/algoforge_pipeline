"""Validation Benchmark Runner for HDE Step H2-D2.

Evaluates the hardened diagnostic evaluator on 24 independent paired test cases,
computing Sensitivity, Specificity, confusion matrices, and false negative analysis.
"""

import json
from pathlib import Path
from research_pipeline.discovery.models import HypothesisCandidate
from research_pipeline.discovery.diagnostic_evaluator import evaluate_diagnostic_hypothesis


def run_evaluator_validation():
    fixture_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/h2_d2_benchmark_cases.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"=== HDE Step H2-D2: Evaluator Validity Benchmark ({len(cases)} Cases) ===\n")

    error_keys = ["e_role", "e_math", "e_data", "e_leap", "e_inop"]
    stats = {k: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for k in error_keys}
    clean_stats = {"total_clean": 0, "clean_passed": 0, "clean_false_positive": 0}

    results = []

    for c in cases:
        case_id = c["case_id"]
        concept_id = c["concept_id"]
        ptype = c["perturbation_type"]
        gt = c["ground_truth_errors"]
        cand_dict = c["candidate"]
        cand = HypothesisCandidate.model_validate(cand_dict)

        eval_res = evaluate_diagnostic_hypothesis(cand, concept_id)
        pred_dict = {
            "e_role": eval_res.e_role,
            "e_math": eval_res.e_math,
            "e_data": eval_res.e_data,
            "e_leap": eval_res.e_leap,
            "e_inop": eval_res.e_inop,
        }

        # Track clean cases
        is_clean_gt = (ptype == "clean")
        if is_clean_gt:
            clean_stats["total_clean"] += 1
            if eval_res.is_operable:
                clean_stats["clean_passed"] += 1
            else:
                clean_stats["clean_false_positive"] += 1

        # Track per-error confusion
        for k in error_keys:
            actual = gt[k]
            pred = pred_dict[k]
            if actual and pred:
                stats[k]["TP"] += 1
            elif not actual and pred:
                stats[k]["FP"] += 1
            elif not actual and not pred:
                stats[k]["TN"] += 1
            elif actual and not pred:
                stats[k]["FN"] += 1

        match = (gt == pred_dict)
        results.append({
            "case_id": case_id,
            "concept_id": concept_id,
            "perturbation_type": ptype,
            "match": match,
            "ground_truth": gt,
            "predicted": pred_dict,
            "notes": eval_res.evaluator_notes
        })

    # Summary table
    print("| Error Category | TP | FP | TN | FN | Sensitivity | Specificity |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

    total_tp = sum(stats[k]["TP"] for k in error_keys)
    total_fp = sum(stats[k]["FP"] for k in error_keys)
    total_tn = sum(stats[k]["TN"] for k in error_keys)
    total_fn = sum(stats[k]["FN"] for k in error_keys)

    for k in error_keys:
        tp = stats[k]["TP"]
        fp = stats[k]["FP"]
        tn = stats[k]["TN"]
        fn = stats[k]["FN"]
        sens = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        print(f"| {k} | {tp} | {fp} | {tn} | {fn} | {sens * 100:.1f}% | {spec * 100:.1f}% |")

    overall_sens = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 1.0
    overall_spec = (total_tn / (total_tn + total_fp)) if (total_tn + total_fp) > 0 else 1.0

    print("=" * 70)
    print(f"Overall Error Sensitivity: {overall_sens * 100:.1f}% ({total_tp}/{total_tp + total_fn})")
    print(f"Overall Non-Error Specificity: {overall_spec * 100:.1f}% ({total_tn}/{total_tn + total_fp})")
    print(f"Clean Specification Accuracy: {clean_stats['clean_passed']}/{clean_stats['total_clean']} ({clean_stats['clean_passed'] / clean_stats['total_clean'] * 100:.1f}%)")

    # Inspect mismatches if any
    mismatches = [r for r in results if not r["match"]]
    if mismatches:
        print(f"\n[WARNING] Found {len(mismatches)} Mismatches:")
        for m in mismatches:
            print(f"  - Case [{m['case_id']}] ({m['concept_id']} | {m['perturbation_type']}):")
            print(f"    GT:   {m['ground_truth']}")
            print(f"    Pred: {m['predicted']}")
            print(f"    Notes: {m['notes']}")
    else:
        print("\n[SUCCESS] 100% Exact Match on All 24 Benchmark Test Cases! Zero False Negatives, Zero False Positives.")

    output_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d2_validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_sensitivity": overall_sens,
            "overall_specificity": overall_spec,
            "clean_accuracy": clean_stats["clean_passed"] / clean_stats["total_clean"],
            "category_stats": stats,
            "case_results": results
        }, f, indent=2)


if __name__ == "__main__":
    run_evaluator_validation()
