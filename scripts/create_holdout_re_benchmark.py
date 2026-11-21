import json
import hashlib
from pathlib import Path
from itertools import permutations

base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
gt_path = base_dir / "tests/fixtures/gold_standard/holdout_expansion_ground_truth.json"
out_path = base_dir / "tests/fixtures/gold_standard/holdout_expansion_re_benchmark.json"

with open(gt_path, "r", encoding="utf-8") as f:
    docs = json.load(f)

pos_pairs = []
neg_candidate_pairs = []

for d in docs:
    sid = d["source_id"]
    entities_by_lid = {e["local_id"]: e for e in d.get("entities", [])}
    gold_edges = set()
    for r in d.get("relations", []):
        s_lid = r["src_local_id"]
        d_lid = r["dst_local_id"]
        rtype = r["relation_type"]
        gold_edges.add((s_lid, d_lid))
        
        s_ent = entities_by_lid[s_lid]
        d_ent = entities_by_lid[d_lid]
        pos_pairs.append({
            "relation_id": f"re_exp_pos_{len(pos_pairs)+1:02d}",
            "source_id": sid,
            "src_local_id": s_lid,
            "src_name": s_ent["name"] or s_ent.get("statement"),
            "src_type": s_ent["entity_type"],
            "dst_local_id": d_lid,
            "dst_name": d_ent["name"] or d_ent.get("statement"),
            "dst_type": d_ent["entity_type"],
            "expected_relation": rtype,
            "evidence_quote": r["evidence"]["verbatim_quote"]
        })
        
    # Generate all other directed pairs within the document
    lids = list(entities_by_lid.keys())
    for s_lid, d_lid in permutations(lids, 2):
        if (s_lid, d_lid) not in gold_edges:
            s_ent = entities_by_lid[s_lid]
            d_ent = entities_by_lid[d_lid]
            neg_candidate_pairs.append({
                "source_id": sid,
                "src_local_id": s_lid,
                "src_name": s_ent["name"] or s_ent.get("statement"),
                "src_type": s_ent["entity_type"],
                "dst_local_id": d_lid,
                "dst_name": d_ent["name"] or d_ent.get("statement"),
                "dst_type": d_ent["entity_type"],
                "expected_relation": "NONE"
            })

print(f"Total positive relations: {len(pos_pairs)}")
print(f"Total non-connected intra-document pairs available: {len(neg_candidate_pairs)}")

# Deterministically sample exactly 19 negative pairs to balance 1:1
# We sort deterministically by (source_id, src_local_id, dst_local_id)
neg_candidate_pairs.sort(key=lambda p: (p["source_id"], p["src_local_id"], p["dst_local_id"]))
selected_negs = []
# Select evenly across documents
step = max(1, len(neg_candidate_pairs) // len(pos_pairs))
for i in range(len(pos_pairs)):
    idx = (i * step) % len(neg_candidate_pairs)
    p = neg_candidate_pairs[idx]
    # avoid duplicates
    if p in selected_negs:
        for alt in neg_candidate_pairs:
            if alt not in selected_negs:
                p = alt
                break
    selected_negs.append(p)
    p["relation_id"] = f"re_exp_neg_{len(selected_negs):02d}"

re_benchmark = {
    "description": "Sealed Holdout Expansion Relation Extraction Benchmark (19 Positives, 19 Negatives)",
    "positive_relations": pos_pairs,
    "negative_relations": selected_negs
}

content = json.dumps(re_benchmark, indent=2, ensure_ascii=False)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content)

sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
print(f"Created Holdout Expansion RE Benchmark: {out_path}")
print(f"Total positives: {len(pos_pairs)}, Total negatives: {len(selected_negs)}")
print(f"SHA-256: {sha}")
