"""
Script: update_audit_json_v1_3.py
Updates h2_d7_audit_results.json to reflect the exact R_executable_math = R_struct and R_param logic.
"""

import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
AUDIT_JSON_PATH = BASE_DIR / "data" / "h2_d7_audit_results.json"

with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Update R_executable_math in block_analysis
for b in data["block_analysis"]:
    bid = b["block_id"]
    y_struct = b["y_exec"] # structural binding
    y_param = b["y_param_approved"]
    
    y_math = {}
    for c in ["C00", "C10", "C01", "C11"]:
        y_math[c] = 1 if (y_struct[c] == 1 and y_param[c] == 1) else 0
    
    b["y_executable_math"] = y_math
    
    # Deltas for mathematical executability
    d_cap = ((y_math["C10"] - y_math["C00"]) + (y_math["C11"] - y_math["C01"])) / 2.0
    d_skel = ((y_math["C01"] - y_math["C00"]) + (y_math["C11"] - y_math["C10"])) / 2.0
    d_inter = (y_math["C11"] - y_math["C01"]) - (y_math["C10"] - y_math["C00"])
    b["deltas_executable_math"] = {"d_cap": d_cap, "d_skel": d_skel, "d_inter": d_inter}

# Update summary in track_b_approved_v0_5
conditions = ["C00", "C10", "C01", "C11"]
math_sum = {c: sum(b["y_executable_math"][c] for b in data["block_analysis"]) / 6.0 for c in conditions}
data["track_b_approved_v0_5"]["r_executable_math_by_condition"] = math_sum

with open(AUDIT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Updated h2_d7_audit_results.json successfully!")
print("Math executability:", math_sum)
