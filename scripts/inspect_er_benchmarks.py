import json
from pathlib import Path

path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1b_benchmarks.json")
with open(path, "r", encoding="utf-8") as f:
    b = json.load(f)

er = b["h1b_er_benchmark"]
for cat in ["same_as_pairs", "variant_of_pairs", "distinct_hard_negatives"]:
    items = er.get(cat, [])
    if items:
        print(f"Sample item from {cat}:", items[0])
