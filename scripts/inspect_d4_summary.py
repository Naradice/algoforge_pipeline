import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
audit_file = BASE_DIR / "data" / "h2_d4_audit_results.json"
data = json.load(open(audit_file, encoding="utf-8"))

print("=== H2-D4 SUMMARY BY CALL ===")
for r in data["audit_records"]:
    cid = r["concept_id"]
    cond = r["condition_id"]
    seed = r["seed"]
    auto_pass = r["auto_evaluation"]["dual_gate_pass"]
    gate = r["auto_evaluation"]["gate_decision"]
    errs = r["auto_evaluation"]["5axis_errors"]
    err_str = ",".join([k for k, v in errs.items() if v and k.startswith("e_")])
    if not err_str:
        err_str = "CLEAN"
    intent = r["intent_audit"]
    intent_str = f"orig={intent['is_original_intent']}, valid_trans={intent['is_valid_transformation']}, infeas={intent['is_infeasible_recognition']}, invalid_sub={intent['is_invalid_substitution']}"
    print(f"{cid:15} | {cond:3} | s{seed} | pass={str(auto_pass):5} | gate={gate:5} | errs={err_str:15} | {intent_str}")

print("\n=== CONDITION STATISTICS ===")
for cond, stats in data["condition_statistics"].items():
    print(f"\nCondition {cond}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

print("\n=== MACRO ESTIMATORS ===")
print("Automated:")
print("  d_struct_by_concept:", data["primary_analysis_automated"]["d_struct_by_concept"])
print("  e_info_by_concept:  ", data["primary_analysis_automated"]["e_info_by_concept"])
print("  macro_delta_struct: ", data["primary_analysis_automated"]["macro_delta_struct"])
print("  macro_delta_info:   ", data["primary_analysis_automated"]["macro_delta_info"])
print("Audited (Sensitivity):")
print("  d_struct_by_concept:", data["sensitivity_analysis_audited"]["d_struct_by_concept"])
print("  e_info_by_concept:  ", data["sensitivity_analysis_audited"]["e_info_by_concept"])
print("  macro_delta_struct: ", data["sensitivity_analysis_audited"]["macro_delta_struct"])
print("  macro_delta_info:   ", data["sensitivity_analysis_audited"]["macro_delta_info"])
