import json
from pathlib import Path

p = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/phase3_holdout_expansion")
for f in p.glob("bundle_*.json"):
    with open(f, "r", encoding="utf-8") as b_f:
        d = json.load(b_f)
    for c in d.get("evaluation_candidates", []):
        nos = c.get("name_or_statement")
        if not isinstance(nos, str):
            print(f"{f.name} | type={c.get('entity_type')} | non-str name_or_statement: {type(nos)}: {nos}")
