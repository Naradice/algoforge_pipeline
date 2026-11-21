import json

with open('c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/step22_h1b_re/step22_re_results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

for cond in ['RE-C0', 'RE-C1', 'RE-C2', 'RE-C3']:
    preds = res['detailed_predictions'][cond]
    pos = [p for p in preds if p['gold_rel'] != 'NONE']
    neg = [p for p in preds if p['gold_rel'] == 'NONE']
    
    # Correct multi-class & binary projections
    # Positive pairs (11):
    # - Exact Match: predicted_rel == gold_rel
    # - Relation Type Misclassification: predicted_rel != NONE and predicted_rel != gold_rel
    # - Missed / Under-detection: predicted_rel == NONE
    pos_exact = sum(1 for p in pos if p['predicted_rel'] == p['gold_rel'])
    pos_misclass = sum(1 for p in pos if p['predicted_rel'] != 'NONE' and p['predicted_rel'] != p['gold_rel'])
    pos_missed = sum(1 for p in pos if p['predicted_rel'] == 'NONE')
    
    # Negative pairs (11):
    # - True Negative: predicted_rel == NONE
    # - False Positive (Spurious Relation): predicted_rel != NONE
    neg_tn = sum(1 for p in neg if p['predicted_rel'] == 'NONE')
    neg_spurious = sum(1 for p in neg if p['predicted_rel'] != 'NONE')
    
    print(f"=== {cond} Audit ===")
    print(f"  Positive Pairs (Total 11): Exact={pos_exact}, Misclassified_Type={pos_misclass}, Missed(NONE)={pos_missed} | Sum = {pos_exact+pos_misclass+pos_missed}")
    print(f"  Negative Pairs (Total 11): True_Negative(NONE)={neg_tn}, Spurious_Relation={neg_spurious} | Sum = {neg_tn+neg_spurious}")
    
    # Binary Existence (Relation vs NONE):
    # TP = pos predicted as non-NONE (pos_exact + pos_misclass)
    # FN = pos predicted as NONE (pos_missed)
    # TN = neg predicted as NONE (neg_tn)
    # FP = neg predicted as non-NONE (neg_spurious)
    bin_tp = pos_exact + pos_misclass
    bin_fn = pos_missed
    bin_tn = neg_tn
    bin_fp = neg_spurious
    
    # Strict Typed Relation Metric:
    # TP_strict = pos_exact (exact relation type)
    # FP_strict = neg_spurious + pos_misclass (incorrect relations emitted)
    # FN_strict = pos_missed + pos_misclass (unrecovered gold relations)
    print(f"  Strict Typed: TP={pos_exact}, FP={neg_spurious + pos_misclass}, FN={pos_missed + pos_misclass}, TN={neg_tn}")
    print(f"  Binary Detection: TP={bin_tp}, FP={bin_fp}, FN={bin_fn}, TN={bin_tn} (Total = {bin_tp+bin_fp+bin_fn+bin_tn})\n")
