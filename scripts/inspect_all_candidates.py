import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
RAW_PATH = BASE_DIR / "data" / "h2_d7_raw_candidates.json"

with open(RAW_PATH, "r", encoding="utf-8") as f:
    blocks = json.load(f)

for b in blocks:
    bid = b["block_id"]
    cid = b["concept_id"]
    print(f"\n=======================================================")
    print(f"BLOCK: {bid} (Concept: {cid})")
    print(f"=======================================================")
    for cond in ["C00", "C10", "C01", "C11"]:
        c = b["candidates"][cond]
        bs = c.get("backtest_spec") or {}
        fe = bs.get("feature_engineering")
        entry = bs.get("entry_condition")
        exit_ = bs.get("exit_condition")
        risk = bs.get("risk_management")
        print(f"\n--- {cond} ---")
        print(f"  FE: {str(fe)[:200]}...")
        print(f"  Entry: {str(entry)[:120]}...")
        print(f"  Exit: {str(exit_)[:120]}...")
        print(f"  Risk: {str(risk)[:120]}...")
