import json
from pathlib import Path

DATA_PATH = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d6_p1_raw_candidates.json")
raw = json.load(open(DATA_PATH, encoding="utf-8"))

cond_tokens = {}
for b in raw:
    for cond, d in b["diagnostics"].items():
        if cond not in cond_tokens:
            cond_tokens[cond] = {"in": 0, "out": 0, "cost": 0.0}
        cond_tokens[cond]["in"] += d["in_tokens"]
        cond_tokens[cond]["out"] += d["out_tokens"]
        cond_tokens[cond]["cost"] += d["cost_usd"]

print("==========================================================================================")
print("Condition   | Calls | In Tokens | Out Tokens | Total Tokens | Cost USD     | Cost / Call")
print("==========================================================================================")
for cond in ["A_init", "B_stage1", "B_init", "A0", "A1", "B0", "B1"]:
    d = cond_tokens[cond]
    n_calls = 6
    tot = d["in"] + d["out"]
    c_per_call = d["cost"] / n_calls
    print(f"{cond:11s} | {n_calls:5d} | {d['in']:9d} | {d['out']:10d} | {tot:12d} | ${d['cost']:10.6f} | ${c_per_call:9.6f}")

print("==========================================================================================")
tot_in = sum(d["in"] for d in cond_tokens.values())
tot_out = sum(d["out"] for d in cond_tokens.values())
tot_all = tot_in + tot_out
tot_cost = sum(d["cost"] for d in cond_tokens.values())
print(f"TOTAL       |    42 | {tot_in:9d} | {tot_out:10d} | {tot_all:12d} | ${tot_cost:10.6f} | ${tot_cost/42:9.6f}")
print("==========================================================================================")

# Combined two-stage initial (B_stage1 + B_init)
b_init_in = cond_tokens["B_stage1"]["in"] + cond_tokens["B_init"]["in"]
b_init_out = cond_tokens["B_stage1"]["out"] + cond_tokens["B_init"]["out"]
b_init_cost = cond_tokens["B_stage1"]["cost"] + cond_tokens["B_init"]["cost"]
print(f"Two-Stage Initial (B_st1 + B_init, 12 calls): In={b_init_in}, Out={b_init_out}, Tot={b_init_in+b_init_out}, Cost=${b_init_cost:.6f}")
