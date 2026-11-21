import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
RAW_PATH = BASE_DIR / "data" / "h2_d7_raw_candidates.json"

with open(RAW_PATH, "r", encoding="utf-8") as f:
    blocks = json.load(f)

b0 = blocks[0]
for cond in ["C00", "C10", "C01", "C11"]:
    c = b0["candidates"][cond]
    bs = c.get("backtest_spec") or {}
    print(f"\n==========================================")
    print(f"Condition: {cond}")
    print(f"==========================================")
    print("Feature engineering:", bs.get("feature_engineering"))
    print("Entry condition:", bs.get("entry_condition"))
    print("Exit condition:", bs.get("exit_condition"))
    print("Risk management:", bs.get("risk_management"))
