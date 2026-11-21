import json

with open('c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/step22_h1b_re/step22_re_results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

c0_preds = {p['pair_id']: p for p in res['detailed_predictions']['RE-C0']}
c2_preds = {p['pair_id']: p for p in res['detailed_predictions']['RE-C2']}
c3_preds = {p['pair_id']: p for p in res['detailed_predictions']['RE-C3']}

# Hybrid Architecture 1: RE-C0 as pre-filter -> If RE-C0 != NONE, use RE-C2
# If RE-C0 == NONE, output NONE
hybrid1_preds = []
for pid, p0 in c0_preds.items():
    p2 = c2_preds[pid]
    gold = p0['gold_rel']
    if p0['predicted_rel'] == 'NONE':
        final_rel = 'NONE'
    else:
        final_rel = p2['predicted_rel']
    hybrid1_preds.append({
        'pair_id': pid,
        'gold_rel': gold,
        'predicted_rel': final_rel,
        'is_correct': (final_rel == gold)
    })

# Hybrid Architecture 2: Agreement of RE-C0 and RE-C2 (Intersection)
hybrid2_preds = []
for pid, p0 in c0_preds.items():
    p2 = c2_preds[pid]
    gold = p0['gold_rel']
    if p0['predicted_rel'] == p2['predicted_rel']:
        final_rel = p0['predicted_rel']
    else:
        final_rel = 'NONE'
    hybrid2_preds.append({
        'pair_id': pid,
        'gold_rel': gold,
        'predicted_rel': final_rel,
        'is_correct': (final_rel == gold)
    })

def compute_metrics(plist):
    pos = [p for p in plist if p['gold_rel'] != 'NONE']
    neg = [p for p in plist if p['gold_rel'] == 'NONE']
    
    tp = sum(1 for p in pos if p['predicted_rel'] == p['gold_rel'])
    pos_misclass = sum(1 for p in pos if p['predicted_rel'] != 'NONE' and p['predicted_rel'] != p['gold_rel'])
    pos_missed = sum(1 for p in pos if p['predicted_rel'] == 'NONE')
    
    neg_tn = sum(1 for p in neg if p['predicted_rel'] == 'NONE')
    neg_spurious = sum(1 for p in neg if p['predicted_rel'] != 'NONE')
    
    fp = neg_spurious + pos_misclass
    fn = pos_missed + pos_misclass
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    
    return {
        'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': neg_tn,
        'exact': tp, 'misclass': pos_misclass, 'missed': pos_missed,
        'spurious': neg_spurious, 'tn_count': neg_tn
    }

print("=== Standalone RE-C0 ===")
print(compute_metrics(list(c0_preds.values())))

print("\n=== Hybrid 1 (RE-C0 Pre-filter -> RE-C2 Decision) ===")
print(compute_metrics(hybrid1_preds))

print("\n=== Hybrid 2 (RE-C0 & RE-C2 Strict Agreement) ===")
print(compute_metrics(hybrid2_preds))
