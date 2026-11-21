import json
from pathlib import Path

path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/holdout_expansion_ground_truth.json")
with open(path, "r", encoding="utf-8") as f:
    docs = json.load(f)

for d in docs:
    print(f"source_id: {d['source_id']}, version_id: {d['version_id']}, source_type: {d['source_type']}")
    print("  target_fields keys:", d.get("target_fields", {}).keys())
    print("  target_field_hashes keys:", d.get("target_field_hashes", {}).keys())
