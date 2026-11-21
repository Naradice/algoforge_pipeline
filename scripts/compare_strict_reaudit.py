import json
from pathlib import Path

DATA_PATH = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d6_p1_audit_results.json")
audit = json.load(open(DATA_PATH, encoding="utf-8"))

# Let's inspect current S=1 items in audit
print("=== CURRENT S=1 ITEMS IN AUDIT ===")
current_s1 = {}
for s in audit["series_summary"]:
    sid = s["series_id"]
    cond = s["condition"]
    s_items = [k for k, v in s["parameter_ratings"].items() if v["S"] == 1]
    if s_items:
        current_s1[sid] = s_items
        print(f"{sid:35s} ({cond}): {s_items}")

# Now let's test Strict Re-audit:
# Deduct:
# 1. hawkes_s42_B1: t_exec (close of M5 bar)
# 2. hawkes_s43_B1: t_exec (close of M5 bar)
# 3. jump_s44_A1: t_exec (bar following announcement)
# 4. jump_s42_A1: theta_bv, j_star
# 5. jump_s44_A1: theta_bv
# 6. jump_s44_B1: theta_bv

print("\n=== STRICT RE-AUDIT COMPARISON ===")
conditions = ["A0", "A1", "B0", "B1"]

orig_counts = {c: audit["condition_aggregates"][c]["S"] for c in conditions}
strict_counts = dict(orig_counts)

# Apply deductions:
strict_counts["B1"] -= 2  # hawkes_s42_B1 t_exec, hawkes_s43_B1 t_exec
strict_counts["A1"] -= 1  # jump_s44_A1 t_exec
strict_counts["A1"] -= 2  # jump_s42_A1 theta_bv, j_star
strict_counts["A1"] -= 1  # jump_s44_A1 theta_bv
strict_counts["B1"] -= 1  # jump_s44_B1 theta_bv

print("Condition | Orig S (Rate)        | Strict S (Rate)       | Difference")
print("----------------------------------------------------------------------")
for c in conditions:
    o_s = orig_counts[c]
    s_s = strict_counts[c]
    o_pct = o_s / 36 * 100
    s_pct = s_s / 36 * 100
    print(f"{c:9s} | {o_s:2d}/36 ({o_pct:5.2f}%)       | {s_s:2d}/36 ({s_pct:5.2f}%)       | {s_s - o_s:+2d} items")

print("\nFactorial Effects under Strict Re-audit:")
d_check_single = (strict_counts["A1"] - strict_counts["A0"]) / 36
d_check_two = (strict_counts["B1"] - strict_counts["B0"]) / 36
d_inter = d_check_two - d_check_single
d_two_gen = (strict_counts["B0"] - strict_counts["A0"]) / 36
d_two_check = (strict_counts["B1"] - strict_counts["A1"]) / 36
d_two_all = ((strict_counts["B0"] + strict_counts["B1"]) / 2 - (strict_counts["A0"] + strict_counts["A1"]) / 2) / 36
d_check_all = ((strict_counts["A1"] + strict_counts["B1"]) / 2 - (strict_counts["A0"] + strict_counts["B0"]) / 2) / 36

print(f"  Checklist Main Effect in Single-Stage (A1 - A0): {d_check_single*100:+.2f}% ({strict_counts['A1']}/36 vs {strict_counts['A0']}/36)")
print(f"  Checklist Main Effect in Two-Stage    (B1 - B0): {d_check_two*100:+.2f}% ({strict_counts['B1']}/36 vs {strict_counts['B0']}/36)")
print(f"  Overall Checklist Effect:                        {d_check_all*100:+.2f}%")
print(f"  Two-Stage Main Effect in Generic      (B0 - A0): {d_two_gen*100:+.2f}%")
print(f"  Two-Stage Main Effect in Checklist    (B1 - A1): {d_two_check*100:+.2f}%")
print(f"  Overall Two-Stage Effect:                        {d_two_all*100:+.2f}%")
print(f"  Interaction Effect (Delta_two - Delta_single):   {d_inter*100:+.2f}%")
