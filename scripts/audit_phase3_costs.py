import json
from pathlib import Path

p = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/phase3_holdout_expansion")

total_in = 0
total_out = 0
call_count = 0

# 1. H1a bundles
for b_file in sorted(p.glob("bundle_*.json")):
    with open(b_file, "r", encoding="utf-8") as f:
        d = json.load(f)
    usage = d.get("raw_artifact", {}).get("token_usage", {})
    tin = usage.get("prompt_tokens", 0)
    tout = usage.get("completion_tokens", 0)
    total_in += tin
    total_out += tout
    call_count += 1

# 2. RE caches
for r_file in sorted(p.glob("re_cache_*.json")):
    call_count += 1
    # estimate 400 in, 80 out
    total_in += 400
    total_out += 80

# 3. ER caches
for e_file in sorted(p.glob("er_cache_*.json")):
    call_count += 1
    # estimate 350 in, 60 out
    total_in += 350
    total_out += 60

# Pricing for gpt-4o-mini: $0.15 / 1M in, $0.60 / 1M out
cost = (total_in / 1_000_000) * 0.15 + (total_out / 1_000_000) * 0.60
print(f"Phase 3 Total Calls: {call_count}")
print(f"Total Input Tokens:  {total_in:,}")
print(f"Total Output Tokens: {total_out:,}")
print(f"Total Tokens:        {total_in + total_out:,}")
print(f"Total Phase 3 Cost:  ${cost:.6f} USD")
