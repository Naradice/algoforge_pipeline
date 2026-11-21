import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
AUDIT_PATH = BASE_DIR / "data" / "h2_d7_audit_results.json"

with open(AUDIT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"{'Block ID':<30} | {'Concept':<16} | {'Seed':<5} | {'C00':<5} {'C10':<5} {'C01':<5} {'C11':<5} | {'d_cap':<7} {'d_skel':<7} {'d_inter':<7}")
print("-" * 95)
for b in data["blocks_analysis"]:
    bid = b["block_id"]
    cid = b["concept_id"]
    seed = b["seed"]
    y = b["y_exec"]
    d = b["deltas_exec"]
    print(f"{bid:<30} | {cid:<16} | {seed:<5} | {y['C00']:<5} {y['C10']:<5} {y['C01']:<5} {y['C11']:<5} | {d['d_cap']:<+7.2f} {d['d_skel']:<+7.2f} {d['d_inter']:<+7.2f}")

print("\nSecondary Metric: R_param (fraction of parameters Specified)")
print(f"{'Block ID':<30} | {'Concept':<16} | {'Seed':<5} | {'C00':<7} {'C10':<7} {'C01':<7} {'C11':<7} | {'d_cap':<7} {'d_skel':<7} {'d_inter':<7}")
print("-" * 105)
for b in data["blocks_analysis"]:
    bid = b["block_id"]
    cid = b["concept_id"]
    seed = b["seed"]
    y = b["y_param"]
    d = b["deltas_param"]
    print(f"{bid:<30} | {cid:<16} | {seed:<5} | {y['C00']:<7.2f} {y['C10']:<7.2f} {y['C01']:<7.2f} {y['C11']:<7.2f} | {d['d_cap']:<+7.2f} {d['d_skel']:<+7.2f} {d['d_inter']:<+7.2f}")
