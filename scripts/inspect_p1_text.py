import json
from pathlib import Path

DATA_PATH = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d6_p1_raw_candidates.json")
data = json.load(open(DATA_PATH, encoding="utf-8"))

for b in data:
    bid = b["block_id"]
    cid = b["concept_id"]
    seed = b["seed"]
    print(f"\n=======================================================")
    print(f"BLOCK: {bid} (Concept: {cid}, Seed: {seed})")
    print(f"=======================================================")
    for cond in ["A_init", "B_init", "A0", "A1", "B0", "B1"]:
        c = b["candidates"][cond]
        spec = c.get("backtest_spec", {}) or {}
        print(f"--- Cond: {cond} ---")
        print(f"  Title: {c.get('title', '')}")
        print(f"  Entry: {spec.get('entry_condition', '')}")
        print(f"  Exit:  {spec.get('exit_condition', '')}")
        print(f"  Feat:  {spec.get('feature_engineering', '')}")
        print(f"  Risk:  {spec.get('risk_management', '')}")
