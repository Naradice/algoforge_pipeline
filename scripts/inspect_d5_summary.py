import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
audit_file = BASE_DIR / "data" / "h2_d5_audit_results.json"
data = json.load(open(audit_file, encoding="utf-8"))

print("=== H2-D5 AUDIT SUMMARY BY CALL ===")
for r in data["audit_records"]:
    cid = r["concept_id"]
    cond = r["condition_id"]
    seed = r["seed"]
    pa = r["param_audit"]
    fa = r["fidelity_audit"]
    auto_pass = r["auto_evaluation"]["dual_gate_pass"]
    audit_pass = r["audit_pass"]
    print(f"{cid:15} | {cond:2} | s{seed} | auto_pass={str(auto_pass):5} | audit_pass={str(audit_pass):5} | Grounded={pa['r_grounded_disposition']:.2f} | ExecRes={pa['r_executable_resolution']:.2f} | StratExec={str(pa['r_strategy_executable']):5} | StructFid={str(fa['r_structural_fidelity']):5} | Trivial={str(fa['is_trivialized']):5} | Class={fa['fidelity_classification']}")

print("\n=== CONDITION STATISTICS ===")
for cond, st in data["condition_statistics"].items():
    print(f"\nCondition {cond}:")
    for k, v in st.items():
        print(f"  {k}: {v}")

print("\n=== CONCEPT-LEVEL STATISTICS AND PAIRED DIFFERENCES ===")
for cid, c_data in data["concept_statistics_and_paired_differences"].items():
    print(f"\nConcept: {cid}")
    print("  D0:", c_data["D0"])
    print("  D1:", c_data["D1"])
    print("  D2:", c_data["D2"])
    print("  D1 - D0 (Enumeration effect):", c_data["diff_d1_minus_d0"])
    print("  D2 - D1 (Disposition effect):", c_data["diff_d2_minus_d1"])
    print("  D2 - D0 (Net effect):        ", c_data["diff_d2_minus_d0"])
