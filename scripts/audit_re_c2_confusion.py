import json
from pathlib import Path

base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
bm_path = base_dir / "tests/fixtures/gold_standard/holdout_expansion_re_benchmark.json"
with open(bm_path, "r", encoding="utf-8") as f:
    bm = json.load(f)

all_pairs = bm["positive_relations"] + bm["negative_relations"]
exp_dir = base_dir / "data/experiments/phase3_holdout_expansion"

exact_matches = []
misclassified_types = []
missed_to_none = []
true_negatives = []
spurious_negatives = []

for idx, pair in enumerate(all_pairs):
    res_id = f"phase3_re_c2_pair_{idx+1:02d}"
    cache_file = exp_dir / f"re_cache_{res_id}.json"
    with open(cache_file, "r", encoding="utf-8") as f:
        pred_dict = json.load(f)
    
    pred_rel = pred_dict.get("relation_type", "NONE")
    gold_rel = pair.get("expected_relation", "NONE")
    
    item = {
        "pair_id": pair["relation_id"],
        "src": pair["src_name"],
        "dst": pair["dst_name"],
        "gold": gold_rel,
        "pred": pred_rel,
        "reason": pred_dict.get("brief_reasoning", "")
    }
    
    if gold_rel != "NONE":
        if pred_rel == gold_rel:
            exact_matches.append(item)
        elif pred_rel == "NONE":
            missed_to_none.append(item)
        else:
            misclassified_types.append(item)
    else:
        if pred_rel == "NONE":
            true_negatives.append(item)
        else:
            spurious_negatives.append(item)

print(f"Total pairs evaluated: {len(all_pairs)} (19 Positives, 19 Negatives)")
print(f"Exact Matches (TP):           {len(exact_matches)}")
print(f"Misclassified Types:          {len(misclassified_types)}")
print(f"Missed to NONE (FN):          {len(missed_to_none)}")
print(f"True Negatives (TN):          {len(true_negatives)}")
print(f"Spurious Relations on Neg (FP): {len(spurious_negatives)}")

print("\n--- Misclassified Types Details ---")
for m in misclassified_types:
    print(f"  [{m['pair_id']}] Gold: {m['gold']} -> Pred: {m['pred']} | '{m['src']}' -> '{m['dst']}' | {m['reason']}")

print("\n--- Missed to NONE Details ---")
for m in missed_to_none:
    print(f"  [{m['pair_id']}] Gold: {m['gold']} -> Pred: NONE | '{m['src']}' -> '{m['dst']}' | {m['reason']}")

print("\n--- Spurious on Negatives Sample ---")
for m in spurious_negatives[:5]:
    print(f"  [{m['pair_id']}] Pred: {m['pred']} | '{m['src']}' -> '{m['dst']}' | {m['reason']}")
