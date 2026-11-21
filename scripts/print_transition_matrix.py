import json
from pathlib import Path

DATA_PATH = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d6_p1_raw_candidates.json")
raw = json.load(open(DATA_PATH, encoding="utf-8"))

# Load audit data
AUDIT_PATH = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d6_p1_audit_results.json")
audit = json.load(open(AUDIT_PATH, encoding="utf-8"))
series_dict = {s["series_id"]: s for s in audit["series_summary"]}

# Initial ratings (Hawkes: mu, alpha, beta, lambda*, t_exec, max_dd; Jump: theta, p, sigma, t_exec, stop, j_star)
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

print("=========================================================================================================")
print("Series                     | Initial S | Cond | Revised S | Transition (0->1, 1->0, 0->0, 1->1)")
print("=========================================================================================================")

transitions = {"A0": {"0->1": 0, "1->0": 0, "0->0": 0, "1->1": 0},
               "A1": {"0->1": 0, "1->0": 0, "0->0": 0, "1->1": 0},
               "B0": {"0->1": 0, "1->0": 0, "0->0": 0, "1->1": 0},
               "B1": {"0->1": 0, "1->0": 0, "0->0": 0, "1->1": 0}}

for b in raw:
    bid = b["block_id"]
    for prefix, rev_cond in [("A_init", "A0"), ("A_init", "A1"), ("B_init", "B0"), ("B_init", "B1")]:
        init_id = f"{bid}_{prefix}"
        rev_id = f"{bid}_{rev_cond}"
        init_p = init_ratings[init_id]
        rev_p = series_dict[rev_id]["parameter_ratings"]
        
        init_str = "".join(str(init_p[p]) for p in init_p)
        rev_str = "".join(str(rev_p[p]["S"]) for p in init_p)
        
        t_list = []
        for p in init_p:
            iv = init_p[p]
            rv = rev_p[p]["S"]
            if iv == 0 and rv == 1:
                t_type = "0->1"
                t_list.append(f"{p}: 0->1 (REPAIRED)")
            elif iv == 1 and rv == 0:
                t_type = "1->0"
                t_list.append(f"{p}: 1->0 (DROPPED)")
            elif iv == 0 and rv == 0:
                t_type = "0->0"
            else:
                t_type = "1->1"
            transitions[rev_cond][t_type] += 1
            
        t_summary = ", ".join(t_list) if t_list else "no changes"
        print(f"{rev_id:26s} | {init_str}      | {rev_cond:4s} | {rev_str}     | {t_summary}")

print("\n=== TRANSITION MATRIX SUMMARY ===")
for cond, t_data in transitions.items():
    print(f"{cond}: 0->1 (Repairs)={t_data['0->1']}, 1->0 (Drops)={t_data['1->0']}, 1->1 (Maintained)={t_data['1->1']}, 0->0 (Unrepaired)={t_data['0->0']}")
