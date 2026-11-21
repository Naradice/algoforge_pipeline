import json
from pathlib import Path
from research_pipeline.evaluation.matching import are_semantically_equivalent

base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
gt_path = base_dir / "tests/fixtures/gold_standard/h1a_ground_truth.json"
with open(gt_path, "r", encoding="utf-8") as f:
    gt = json.load(f)

dev_docs = {d["source_id"]: d for d in gt if d.get("split") == "development"}

# Load C4 trial 1 bundles from step21
c4_dir = base_dir / "data/experiments/step21_h1a_r2"

total_gold_relations = 0
both_endpoints_detected = 0

print("=== End-to-End Relation Endpoint Co-availability Audit (C4 t1) ===")
for sid, doc in dev_docs.items():
    gold_ents = {e["local_id"]: e for e in doc.get("entities", [])}
    gold_rels = doc.get("relations", [])
    
    bundle_path = c4_dir / f"bundle_step21_C4_t1_{sid}.json"
    if not bundle_path.exists():
        continue
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    preds = bundle.get("evaluation_candidates", [])
    
    for r in gold_rels:
        total_gold_relations += 1
        src_gold = gold_ents.get(r["src_local_id"])
        dst_gold = gold_ents.get(r["dst_local_id"])
        
        if not src_gold or not dst_gold:
            continue
            
        src_found = any(
            p["entity_type"] == src_gold["entity_type"] and
            are_semantically_equivalent(src_gold.get("name") or src_gold.get("statement") or "", src_gold.get("synonyms", []), p["name_or_statement"])
            for p in preds
        )
        dst_found = any(
            p["entity_type"] == dst_gold["entity_type"] and
            are_semantically_equivalent(dst_gold.get("name") or dst_gold.get("statement") or "", dst_gold.get("synonyms", []), p["name_or_statement"])
            for p in preds
        )
        
        if src_found and dst_found:
            both_endpoints_detected += 1
            print(f"  [CO-AVAILABLE] {sid} {r['src_local_id']}->{r['relation_type']}->{r['dst_local_id']} (src={src_found}, dst={dst_found})")
        else:
            print(f"  [BROKEN ENDPOINT] {sid} {r['src_local_id']}->{r['relation_type']}->{r['dst_local_id']} (src={src_found}, dst={dst_found})")

r_endpoint = both_endpoints_detected / total_gold_relations if total_gold_relations > 0 else 0.0
print(f"\nTotal Gold Relations: {total_gold_relations}")
print(f"Relations with BOTH endpoints extracted: {both_endpoints_detected}")
print(f"R_endpoint = {r_endpoint:.4f} ({r_endpoint*100:.1f}%)")
