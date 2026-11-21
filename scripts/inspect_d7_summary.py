import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
RAW_PATH = BASE_DIR / "data" / "h2_d7_raw_candidates.json"

with open(RAW_PATH, "r", encoding="utf-8") as f:
    blocks = json.load(f)

print(f"Total blocks loaded: {len(blocks)}")
for b in blocks:
    bid = b["block_id"]
    cost = b["total_block_cost_usd"]
    order = b["execution_order"]
    print(f"\nBlock: {bid} | Cost: ${cost:.6f} | Order: {order}")
    for c in ["C00", "C10", "C01", "C11"]:
        cand = b["candidates"][c]
        diag = b["diagnostics"][c]
        fin_init = diag["initial_call"]["finish_reason"]
        fin_patch = diag["patch_call"]["finish_reason"]
        out_tok_init = diag["initial_call"]["out_tokens"]
        out_tok_patch = diag["patch_call"]["out_tokens"]
        print(f"  [{c}] init_tok={out_tok_init} ({fin_init}), patch_tok={out_tok_patch} ({fin_patch})")
        print(f"       params: {list(cand.get('parameters', {}).keys())}")
        print(f"       math keys: {list(cand.get('mathematical_formulation', {}).keys())}")
