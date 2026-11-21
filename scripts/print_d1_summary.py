"""Script to parse and display H2-D1 diagnostic results."""

import json
from pathlib import Path

def print_diagnostic_summary():
    results_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d1_diagnostic_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=== HDE Step H2-D1 Diagnostic Experiment Summary ===")
    print("Total Calls:", data["total_calls"])
    print(f"Total Cost USD: ${data['total_cost_usd']:.6f}")
    print(f"Reconciled Cumulative Spent: ${data['reconciled_cumulative_spent_usd']:.6f}")
    print(f"Reconciled Remaining Budget: ${data['reconciled_remaining_budget_usd']:.6f}")
    print("=" * 80)

    # 4 concepts x 3 conditions table
    matrix = {}
    for r in data["results"]:
        cid = r["call_id"]
        c_id = r["concept_id"]
        cond = r["condition_id"]
        ev = r["evaluation"]
        cand = r["candidate"]
        spec = cand.get("backtest_spec", {})
        matrix.setdefault(c_id, {})[cond] = (cand, spec, ev)

    print("\n| Concept | Condition | IsOperable | E_role | E_math | E_data | E_leap | E_inop | Notes / Offending Snippet |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")

    for c_id, conds in matrix.items():
        for cond in ["D1-C0", "D1-C1", "D1-C2"]:
            cand, spec, ev = conds[cond]
            errs = []
            for k in ["E_role", "E_math", "E_data", "E_leap", "E_inop"]:
                if ev.get(k.lower()):
                    errs.append(f"{k}: {ev.get('error_evidence', {}).get(k, '')}")
            err_str = "; ".join(errs) if errs else "None (Clean)"
            print(f"| {c_id} | {cond} | {ev['is_operable']} | {ev['e_role']} | {ev['e_math']} | {ev['e_data']} | {ev['e_leap']} | {ev['e_inop']} | {err_str} |")

    print("\n=== Detailed Candidate Inspection ===")
    for c_id, conds in matrix.items():
        print(f"\n################### Concept: {c_id} ###################")
        for cond in ["D1-C0", "D1-C1", "D1-C2"]:
            cand, spec, ev = conds[cond]
            print(f"--- Condition: {cond} ---")
            print("Title:", cand.get("title"))
            print("Claim:", cand.get("claim"))
            print("Mechanism:", cand.get("proposed_mechanism"))
            print("Entry:", spec.get("entry_condition"))
            print("Exit:", spec.get("exit_condition"))
            print("Features:", spec.get("feature_engineering"))
            print("Falsification:", cand.get("testable_falsification_condition"))
            print("Evaluation Notes:", ev.get("evaluator_notes"))
            print()

if __name__ == "__main__":
    print_diagnostic_summary()
