import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
audit_file = BASE_DIR / "data" / "h2_d6_audit_results.json"
data = json.load(open(audit_file, encoding="utf-8"))

print("=== H2-D6 AUDIT SUMMARY BY SERIES ===")
for r in data["audit_records"]:
    cid = r["concept_id"]
    cond = r["condition_id"]
    seed = r["seed"]
    a = r["audit"]
    auto_pass = r["auto_evaluation"]["dual_gate_pass"]
    print(f"{cid:15} | {cond:3} | s{seed} | auto={str(auto_pass):5} | Grounded={str(a['grounded_executable']):5} | ReqCov={a['required_coverage']:.2f} | GoldRaw={a['gold_coverage_raw']:.2f} | Halluc={str(a['has_hallucinations']):5} | StructFid={str(a['structural_fidelity']):5}")

print("\n=== CONDITION STATISTICS ===")
for cond, st in data["condition_statistics"].items():
    print(f"\nCondition {cond}:")
    for k, v in st.items():
        print(f"  {k}: {v}")

print("\n=== CONCEPT-LEVEL STATISTICS & PAIRED DIFFERENCES ===")
for cid, c_data in data["concept_statistics_and_paired_differences"].items():
    print(f"\nConcept: {cid}")
    print("  S0:", c_data["S0"])
    print("  S1:", c_data["S1"])
    print("  S2a:", c_data["S2a"])
    print("  S2b:", c_data["S2b"])
    print("  Delta_stage (S1 - S0):       ", c_data["delta_stage_s1_minus_s0"])
    print("  Delta_revision (S2a - S1):   ", c_data["delta_revision_s2a_minus_s1"])
    print("  Delta_checklist (S2b - S2a): ", c_data["delta_checklist_s2b_minus_s2a"])
    print("  Delta_net (S2b - S0):        ", c_data["delta_net_s2b_minus_s0"])
