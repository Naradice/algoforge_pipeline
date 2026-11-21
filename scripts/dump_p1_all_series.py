import json
from pathlib import Path

DATA_PATH = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d6_p1_raw_candidates.json")
data = json.load(open(DATA_PATH, encoding="utf-8"))

for b in data:
    bid = b["block_id"]
    cid = b["concept_id"]
    seed = b["seed"]
    for cond in ["A0", "A1", "B0", "B1"]:
        c = b["candidates"][cond]
        spec = c.get("backtest_spec", {}) or {}
        print(f"================================================================================")
        print(f"SERIES: {bid}_{cond} (Concept: {cid}, Seed: {seed}, Condition: {cond})")
        print(f"================================================================================")
        print(f"CLAIM: {c.get('claim', '')}")
        print(f"MECHANISM: {c.get('proposed_mechanism', '')}")
        print(f"RATIONALE: {c.get('grounding_rationale', '')}")
        print(f"FEAT: {spec.get('feature_engineering', '')}")
        print(f"ENTRY: {spec.get('entry_condition', '')}")
        print(f"EXIT: {spec.get('exit_condition', '')}")
        print(f"RISK: {spec.get('risk_management', '')}")
        print(f"FALSIF: {spec.get('falsification_metric', '')}")
        print(f"DIAG: {b['diagnostics'].get(cond, {})}")
        print()
