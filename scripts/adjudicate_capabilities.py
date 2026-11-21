"""Adjudication script for Capability matches in C4 across 3 trials.

Inspects each extracted Capability candidate against the Gold Capability for that document.
Classifies each into:
- TRUE_EQUIVALENT (真の同値): Identical core operation verb + target object + compatible horizon
- PARTIAL_MATCH (部分一致): Overlaps in verb or object, but differs in scope/horizon
- NON_EQUIVALENT (非同値): Conceptually distinct capability
"""

import json
from pathlib import Path
from typing import List, Dict, Any

gt_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json")
with open(gt_path, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

dev_docs = {d["source_id"]: d for d in ground_truth if d.get("split") == "development"}
exp_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/dev_c0_c4")

# Collect unique pairs of (Gold Capability, Extracted Capability) in C4
observed_pairs = []

for sid, doc in dev_docs.items():
    gold_caps = [e for e in doc.get("entities", []) if e.get("entity_type") == "capability"]
    if not gold_caps:
        continue
    gold_cap = gold_caps[0]
    gold_name = gold_cap.get("name") or gold_cap.get("statement") or ""
    gold_quote = gold_cap.get("evidence", {}).get("verbatim_quote", "")
    
    for t in [1, 2, 3]:
        p = exp_dir / f"bundle_dev_C4_t{t}_{sid}.json"
        with open(p, "r", encoding="utf-8") as f:
            b = json.load(f)
        
        acc_caps = [c for c in b.get("evaluation_candidates", []) if c.get("entity_type") == "capability"]
        for ac in acc_caps:
            pred_name = ac.get("name_or_statement")
            pred_quote = ac.get("evidence", {}).get("verbatim_quote", "") if ac.get("evidence") else ""
            observed_pairs.append({
                "source_id": sid,
                "trial": t,
                "gold_name": gold_name,
                "gold_quote": gold_quote,
                "pred_name": pred_name,
                "pred_quote": pred_quote,
            })

# Deduplicate by (source_id, gold_name, pred_name)
unique_pairs = {}
for p in observed_pairs:
    k = (p["source_id"], p["gold_name"], p["pred_name"])
    if k not in unique_pairs:
        unique_pairs[k] = p
        unique_pairs[k]["count"] = 0
    unique_pairs[k]["count"] += 1

print(f"Total capability extraction instances in C4 (3 trials x 10 docs): {len(observed_pairs)}")
print(f"Unique (Gold, Pred) pairs observed: {len(unique_pairs)}")

print("\n" + "="*80)
print(f"{'Source ID':<18} | {'Gold Capability':<28} | {'Extracted Capability':<32} | Count")
print("-"*80)
for (sid, gname, pname), data in unique_pairs.items():
    print(f"{sid:<18} | {gname:<28} | {pname:<32} | {data['count']}")
