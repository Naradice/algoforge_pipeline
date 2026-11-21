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
    print(f"BLOCK: {bid}")
    print(f"=======================================================")
    for cond in ["C00", "C10", "C01", "C11"]:
        c = b["candidates"][cond]
        bs = c.get("backtest_spec") or {}
        fe = bs.get("feature_engineering")
        print(f"\n[{cond}] FE:")
        print(json.dumps(fe, indent=2, ensure_ascii=False) if isinstance(fe, dict) else str(fe))
