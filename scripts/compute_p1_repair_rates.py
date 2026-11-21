"""
Script: compute_p1_repair_rates.py
Calculates exact repair rates and corruption checks from initial generation (A_init, B_init)
to revised conditions (A0, A1, B0, B1).
"""

import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
DATA_DIR = BASE_DIR / "data"
RAW_CANDIDATES_PATH = DATA_DIR / "h2_d6_p1_raw_candidates.json"

with open(RAW_CANDIDATES_PATH, "r", encoding="utf-8") as f:
    raw_blocks = json.load(f)

# Inspect A_init and B_init ratings
# Hawkes: mu, alpha, beta, lambda*, t_exec, max_dd
# Jump: theta_bv, p_trans, sigma_ratio, t_exec, stop_bps, j_star

init_ratings = {
    # Hawkes s42
    "h2_d6_p1_hawkes_process_s42_A_init": {"mu": 0, "alpha": 0, "beta": 0, "lambda_star": 0, "t_exec": 1, "max_dd": 0},
    "h2_d6_p1_hawkes_process_s42_B_init": {"mu": 0, "alpha": 0, "beta": 0, "lambda_star": 0, "t_exec": 0, "max_dd": 1},
    # Hawkes s43
    "h2_d6_p1_hawkes_process_s43_A_init": {"mu": 0, "alpha": 0, "beta": 0, "lambda_star": 0, "t_exec": 1, "max_dd": 0},
    "h2_d6_p1_hawkes_process_s43_B_init": {"mu": 0, "alpha": 0, "beta": 0, "lambda_star": 0, "t_exec": 0, "max_dd": 0},
    # Hawkes s44
    "h2_d6_p1_hawkes_process_s44_A_init": {"mu": 0, "alpha": 0, "beta": 0, "lambda_star": 1, "t_exec": 1, "max_dd": 1},
    "h2_d6_p1_hawkes_process_s44_B_init": {"mu": 0, "alpha": 0, "beta": 0, "lambda_star": 0, "t_exec": 0, "max_dd": 0},
    # Jump s42
    "h2_d6_p1_jump_diffusion_s42_A_init": {"theta_bv": 0, "p_trans": 0, "sigma_ratio": 0, "t_exec": 0, "stop_bps": 0, "j_star": 0},
    "h2_d6_p1_jump_diffusion_s42_B_init": {"theta_bv": 0, "p_trans": 0, "sigma_ratio": 0, "t_exec": 0, "stop_bps": 1, "j_star": 0},
    # Jump s43
    "h2_d6_p1_jump_diffusion_s43_A_init": {"theta_bv": 0, "p_trans": 0, "sigma_ratio": 0, "t_exec": 0, "stop_bps": 0, "j_star": 0},
    "h2_d6_p1_jump_diffusion_s43_B_init": {"theta_bv": 0, "p_trans": 0, "sigma_ratio": 0, "t_exec": 1, "stop_bps": 1, "j_star": 0},
    # Jump s44
    "h2_d6_p1_jump_diffusion_s44_A_init": {"theta_bv": 0, "p_trans": 0, "sigma_ratio": 0, "t_exec": 0, "stop_bps": 0, "j_star": 0},
    "h2_d6_p1_jump_diffusion_s44_B_init": {"theta_bv": 0, "p_trans": 0, "sigma_ratio": 0, "t_exec": 0, "stop_bps": 1, "j_star": 0},
}

with open(DATA_DIR / "h2_d6_p1_audit_results.json", "r", encoding="utf-8") as f:
    audit_data = json.load(f)

series_dict = {s["series_id"]: s for s in audit_data["series_summary"]}

def compute_repairs(base_prefix, rev_cond):
    missing_count = 0
    repaired_count = 0
    dropped_count = 0
    repaired_items = []
    dropped_items = []

    for block in raw_blocks:
        bid = block["block_id"]
        init_id = f"{bid}_{base_prefix}"
        rev_id = f"{bid}_{rev_cond}"

        init_p = init_ratings[init_id]
        rev_p = series_dict[rev_id]["parameter_ratings"]

        for pname, init_val in init_p.items():
            rev_val = rev_p[pname]["S"]
            if init_val == 0:
                missing_count += 1
                if rev_val == 1:
                    repaired_count += 1
                    repaired_items.append((rev_id, pname, rev_p[pname]["rationale"]))
            elif init_val == 1:
                if rev_val == 0:
                    dropped_count += 1
                    dropped_items.append((rev_id, pname))

    repair_rate = repaired_count / missing_count if missing_count > 0 else 0.0
    return {
        "missing_count": missing_count,
        "repaired_count": repaired_count,
        "repair_rate": round(repair_rate, 4),
        "dropped_count": dropped_count,
        "repaired_items": repaired_items,
        "dropped_items": dropped_items,
    }

rep_a0 = compute_repairs("A_init", "A0")
rep_a1 = compute_repairs("A_init", "A1")
rep_b0 = compute_repairs("B_init", "B0")
rep_b1 = compute_repairs("B_init", "B1")

print("================================================================================")
print("P1 Repair Rates and Specification Corruption Summary")
print("================================================================================")
print(f"A_init -> A0 (Generic):    Repairs: {rep_a0['repaired_count']}/{rep_a0['missing_count']} ({rep_a0['repair_rate']*100:.2f}%) | Dropped: {rep_a0['dropped_count']}")
print(f"A_init -> A1 (Checklist):  Repairs: {rep_a1['repaired_count']}/{rep_a1['missing_count']} ({rep_a1['repair_rate']*100:.2f}%) | Dropped: {rep_a1['dropped_count']}")
print(f"B_init -> B0 (Generic):    Repairs: {rep_b0['repaired_count']}/{rep_b0['missing_count']} ({rep_b0['repair_rate']*100:.2f}%) | Dropped: {rep_b0['dropped_count']}")
print(f"B_init -> B1 (Checklist):  Repairs: {rep_b1['repaired_count']}/{rep_b1['missing_count']} ({rep_b1['repair_rate']*100:.2f}%) | Dropped: {rep_b1['dropped_count']}")
print("\n--- Detailed Repaired Items in A1 ---")
for item in rep_a1["repaired_items"]:
    print(f"  {item[0]} [{item[1]}]: {item[2]}")
print("\n--- Detailed Repaired Items in B1 ---")
for item in rep_b1["repaired_items"]:
    print(f"  {item[0]} [{item[1]}]: {item[2]}")
print("================================================================================")
