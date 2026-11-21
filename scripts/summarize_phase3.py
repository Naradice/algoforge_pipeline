import json

with open("c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/phase3_holdout_expansion/phase3_holdout_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

print("=== METADATA ===")
print(json.dumps(res["metadata"], indent=2))

print("\n=== H1A EXTRACTION (TRIAL 1) ===")
t1 = res["h1a_extraction"]["trial_1"]
print(f"Semantic Micro F1: {t1['semantic_micro_f1']} (P={t1['semantic_micro_precision']}, R={t1['semantic_micro_recall']})")
print(f"Strict Micro F1:   {t1['strict_micro_f1']}")
for cls, v in t1["per_class"].items():
    print(f"  [{cls:12s}] gold={v['gold_count']:2d} pred={v['pred_count']:2d} tp_sem={v['tp_semantic']:2d} F1={v['semantic_f1']:.4f}")

print("\n=== H1A EXTRACTION (TRIAL 2) ===")
t2 = res["h1a_extraction"]["trial_2"]
print(f"Semantic Micro F1: {t2['semantic_micro_f1']} (P={t2['semantic_micro_precision']}, R={t2['semantic_micro_recall']})")
print(f"Strict Micro F1:   {t2['strict_micro_f1']}")
for cls, v in t2["per_class"].items():
    print(f"  [{cls:12s}] gold={v['gold_count']:2d} pred={v['pred_count']:2d} tp_sem={v['tp_semantic']:2d} F1={v['semantic_f1']:.4f}")

print("\n=== H1B RELATION EXTRACTION (ORACLE 38 PAIRS) ===")
for m in ["oracle_re_c0", "oracle_re_c2", "oracle_re_hybrid"]:
    st = res["h1b_relation_extraction"][m]
    print(f"  [{m:18s}] P={st['precision']:.4f} R={st['recall']:.4f} F1={st['f1']:.4f} (TP={st['tp']}, FP={st['fp']}, FN={st['fn']}, TN={st['tn']})")

print("\n=== END-TO-END RELATION EXTRACTION ===")
e2e = res["h1b_relation_extraction"]["end_to_end"]
print(f"Gold Relations Total:         {e2e['gold_relations_total']}")
print(f"Endpoints Co-Available:       {e2e['endpoints_co_available']}")
print(f"R_endpoint:                   {e2e['r_endpoint']:.4f} ({e2e['endpoints_co_available']}/{e2e['gold_relations_total']})")
print(f"R_RE|endpoint (RE-C0):        {e2e['r_re_given_endpoint_c0']:.4f}")
print(f"R_RE|endpoint (RE-Hybrid):    {e2e['r_re_given_endpoint_hyb']:.4f}")
print(f"Exact R_E2E (RE-C0):          {e2e['r_e2e_c0']:.4f}")
print(f"Exact R_E2E (RE-Hybrid):      {e2e['r_e2e_hyb']:.4f}")

print("\n=== H1B ENTITY RESOLUTION (10 PAIRS) ===")
er_b = res["h1b_entity_resolution"]["baseline_er_c2"]
er_p = res["h1b_entity_resolution"]["proposed_er"]
print(f"Baseline ER-C2 (SAME_AS):    P={er_b['precision']:.4f} R={er_b['recall']:.4f} F1={er_b['f1']:.4f}")
print(f"Proposed ER 3-Way Accuracy:  {er_p['three_way_accuracy']:.4f} (10/10)")
print(f"Proposed VARIANT_OF Recall:  {er_p['variant_of_recall']:.4f} (3/3)")

print("\n=== KNOWLEDGE GRAPH DATABASE ===")
kg = res["knowledge_graph"]
print(f"Stats:        {kg['stats']}")
print(f"Orphan Edges: {kg['orphan_edges']}")
print(f"Self Loops:   {kg['self_loops']}")
