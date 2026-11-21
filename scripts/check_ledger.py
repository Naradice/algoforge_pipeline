import json
from pathlib import Path

path = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/budget_ledger.jsonl")
lines = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]

print(f"Total entries: {len(lines)}")
total_cost = sum(l.get("cost_usd", 0.0) for l in lines)
print(f"Total spent across entire project: ${total_cost:.6f} USD")
print(f"Remaining project budget: ${10.0 - total_cost:.6f} USD")
if lines:
    print("Latest entry:", lines[-1])
