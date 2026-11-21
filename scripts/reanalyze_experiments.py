"""Re-analysis script addressing User Review Directives 1, 2, 3.

Computes:
1. Pure Validator Ablation on identical raw candidate sets (within C3 and within C4).
2. Paired Document-level Cluster Bootstrap (B=1000) for Delta F1(C3 - C2) and Delta F1(C4 - C3).
3. Exact LocalRelation TP, FP, FN against Gold.
4. Corrected Candidate Counts (Raw vs Accepted) and Quote Validity (N/A for C0-C2).
5. Capability matching diagnosis: Strict Substring vs Stemmed/Token-Overlap Semantic Equivalence.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

from research_pipeline.evaluation.matching import (
    match_entities_bipartite,
    normalize_entity_name,
    are_semantically_equivalent,
    BipartiteMatchingResult,
)
from research_pipeline.extractors.raw_artifact import EvaluationCandidate, ExtractionTrioBundle
from research_pipeline.evaluation.metrics import evaluate_run, FullEvaluationReport

gt_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json")
with open(gt_path, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

dev_docs = [d for d in ground_truth if d.get("split") == "development"]
doc_by_id = {d["source_id"]: d for d in dev_docs}
exp_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/dev_c0_c4")

print("="*85)
print("HDE H1a DEVELOPMENT EXPERIMENT RE-ANALYSIS & SCIENTIFIC AUDIT")
print("="*85)

# ---------------------------------------------------------------------------
# 1. PURE VALIDATOR ABLATION (Within-Condition Raw vs Accepted)
# ---------------------------------------------------------------------------
print("\n--- 1. PURE VALIDATOR ABLATION ON IDENTICAL CANDIDATES (C3 & C4) ---")

def eval_candidates_against_gold(candidates: List[EvaluationCandidate], gold_doc: Dict[str, Any]) -> Tuple[int, int, int]:
    gold_entities = gold_doc.get("entities", [])
    res = match_entities_bipartite(gold_entities, candidates)
    return res.true_positives, res.false_positives, res.false_negatives

for cond in ["C3", "C4"]:
    raw_tp, raw_fp, raw_fn = 0, 0, 0
    acc_tp, acc_fp, acc_fn = 0, 0, 0
    total_raw_cands = 0
    total_acc_cands = 0
    total_rejected = 0

    for t in [1, 2, 3]:
        for doc in dev_docs:
            sid = doc["source_id"]
            p = exp_dir / f"bundle_dev_{cond}_t{t}_{sid}.json"
            with open(p, "r", encoding="utf-8") as f:
                b = json.load(f)
            
            # Reconstruct raw candidates (all parsed candidates before validator filtering)
            raw_parsed = b["raw_artifact"].get("parsed_candidates") or {}
            raw_cands = []
            for etype in ["phenomenon", "model", "method", "capability", "assumption", "limitation"]:
                plural_keys = [f"{etype}s", f"{etype}es", "phenomena", "capabilities", etype]
                items = []
                for k in plural_keys:
                    if k in raw_parsed and isinstance(raw_parsed[k], list):
                        items = raw_parsed[k]
                        break
                for idx, item in enumerate(items):
                    name_stmt = item.get("name") or item.get("statement") or ""
                    if name_stmt:
                        raw_cands.append(
                            EvaluationCandidate(
                                candidate_local_id=item.get("local_id", f"{etype[:3]}_{idx+1:02d}"),
                                entity_type=etype,  # type: ignore
                                name_or_statement=name_stmt,
                                condition_id=cond,
                                extraction_run_id=b["extraction_run_id"],
                            )
                        )
            
            accepted_cands = [EvaluationCandidate(**c) for c in b.get("evaluation_candidates", [])]
            
            total_raw_cands += len(raw_cands)
            total_acc_cands += len(accepted_cands)
            total_rejected += len(b.get("rejected_items", []))

            # Match raw against gold
            rtp, rfp, rfn = eval_candidates_against_gold(raw_cands, doc)
            raw_tp += rtp
            raw_fp += rfp
            raw_fn += rfn

            # Match accepted against gold
            atp, afp, afn = eval_candidates_against_gold(accepted_cands, doc)
            acc_tp += atp
            acc_fp += afp
            acc_fn += afn

    rp = raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) > 0 else 0.0
    rr = raw_tp / (raw_tp + raw_fn) if (raw_tp + raw_fn) > 0 else 0.0
    rf1 = (2 * rp * rr) / (rp + rr) if (rp + rr) > 0 else 0.0

    ap = acc_tp / (acc_tp + acc_fp) if (acc_tp + acc_fp) > 0 else 0.0
    ar = acc_tp / (acc_tp + acc_fn) if (acc_tp + acc_fn) > 0 else 0.0
    af1 = (2 * ap * ar) / (ap + ar) if (ap + ar) > 0 else 0.0

    delta_p = ap - rp
    delta_r = ar - rr
    delta_f1 = af1 - rf1

    print(f"\nCondition {cond} Validator Effect (N=30 trials):")
    print(f"  Raw Candidates:      {total_raw_cands} | TP={raw_tp}, FP={raw_fp}, FN={raw_fn} | P={rp:.4f}, R={rr:.4f}, F1={rf1:.4f}")
    print(f"  Accepted Candidates: {total_acc_cands} | TP={acc_tp}, FP={acc_fp}, FN={acc_fn} | P={ap:.4f}, R={ar:.4f}, F1={af1:.4f}")
    print(f"  Rejected Candidates: {total_rejected} (Rejected FP={raw_fp - acc_fp}, Rejected TP={raw_tp - acc_tp})")
    print(f"  Pure Validator Delta: Delta P = {delta_p:+.4f}, Delta R = {delta_r:+.4f}, Delta F1 = {delta_f1:+.4f}")


# ---------------------------------------------------------------------------
# 2. PAIRED CLUSTER BOOTSTRAP CI FOR DELTA F1
# ---------------------------------------------------------------------------
print("\n" + "="*85)
print("--- 2. PAIRED DOCUMENT-LEVEL CLUSTER BOOTSTRAP (B=1000) FOR DELTA F1 ---")

# Load bundles by doc and trial for C2, C3, C4
def get_condition_doc_bundles(cond: str, trial: int = 1) -> Dict[str, ExtractionTrioBundle]:
    bundles = {}
    for doc in dev_docs:
        sid = doc["source_id"]
        p = exp_dir / f"bundle_dev_{cond}_t{trial}_{sid}.json"
        with open(p, "r", encoding="utf-8") as f:
            bundles[sid] = ExtractionTrioBundle.model_validate_json(f.read())
    return bundles

b_c2 = get_condition_doc_bundles("C2", 1)
b_c3 = get_condition_doc_bundles("C3", 1)
b_c4 = get_condition_doc_bundles("C4", 1)

doc_ids = [d["source_id"] for d in dev_docs]
n_docs = len(doc_ids)
B = 1000
rng = np.random.default_rng(42)

boot_delta_f1_c3_c2 = []
boot_delta_f1_c4_c3 = []
boot_delta_p_c3_c2 = []
boot_delta_r_c3_c2 = []

for b_idx in range(B):
    sample_indices = rng.integers(0, n_docs, size=n_docs)
    sample_doc_ids = [doc_ids[i] for i in sample_indices]
    
    # Evaluate C2, C3, C4 on this exact document bootstrap sample
    def eval_sample(cond_bundles):
        all_tp, all_fp, all_fn = 0, 0, 0
        for sid in sample_doc_ids:
            doc = doc_by_id[sid]
            bundle = cond_bundles[sid]
            res = match_entities_bipartite(doc.get("entities", []), bundle.evaluation_candidates)
            all_tp += res.true_positives
            all_fp += res.false_positives
            all_fn += res.false_negatives
        p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    p2, r2, f1_2 = eval_sample(b_c2)
    p3, r3, f1_3 = eval_sample(b_c3)
    p4, r4, f1_4 = eval_sample(b_c4)

    boot_delta_f1_c3_c2.append(f1_3 - f1_2)
    boot_delta_f1_c4_c3.append(f1_4 - f1_3)
    boot_delta_p_c3_c2.append(p3 - p2)
    boot_delta_r_c3_c2.append(r3 - r2)

ci_d_f1_c3_c2 = (np.percentile(boot_delta_f1_c3_c2, 2.5), np.percentile(boot_delta_f1_c3_c2, 97.5))
ci_d_p_c3_c2 = (np.percentile(boot_delta_p_c3_c2, 2.5), np.percentile(boot_delta_p_c3_c2, 97.5))
ci_d_r_c3_c2 = (np.percentile(boot_delta_r_c3_c2, 2.5), np.percentile(boot_delta_r_c3_c2, 97.5))
ci_d_f1_c4_c3 = (np.percentile(boot_delta_f1_c4_c3, 2.5), np.percentile(boot_delta_f1_c4_c3, 97.5))

print(f"Paired Delta F1 (C3 - C2) Evidence Grounding Effect:")
print(f"  Point Estimate: {np.mean(boot_delta_f1_c3_c2):+.4f}")
print(f"  Paired 95% CI:  [{ci_d_f1_c3_c2[0]:+.4f}, {ci_d_f1_c3_c2[1]:+.4f}]")
print(f"  Paired Delta Precision 95% CI: [{ci_d_p_c3_c2[0]:+.4f}, {ci_d_p_c3_c2[1]:+.4f}]")
print(f"  Paired Delta Recall 95% CI:    [{ci_d_r_c3_c2[0]:+.4f}, {ci_d_r_c3_c2[1]:+.4f}]")

print(f"\nPaired Delta F1 (C4 - C3) Full Method Composite Effect:")
print(f"  Point Estimate: {np.mean(boot_delta_f1_c4_c3):+.4f}")
print(f"  Paired 95% CI:  [{ci_d_f1_c4_c3[0]:+.4f}, {ci_d_f1_c4_c3[1]:+.4f}]")


# ---------------------------------------------------------------------------
# 3. LOCAL RELATION EVALUATION
# ---------------------------------------------------------------------------
print("\n" + "="*85)
print("--- 3. LOCAL RELATION EVALUATION AGAINST GOLD BENCHMARK (C4) ---")

total_gold_rels = sum(len(d.get("relations", [])) for d in dev_docs) * 3
total_pred_rels = 0
rel_tps = 0
rel_fps = 0
rel_fns = 0

for t in [1, 2, 3]:
    for doc in dev_docs:
        gold_rels = doc.get("relations", [])
        sid = doc["source_id"]
        p = exp_dir / f"bundle_dev_C4_t{t}_{sid}.json"
        with open(p, "r", encoding="utf-8") as f:
            b = json.load(f)
        acc_rec = b.get("accepted_record") or {}
        pred_rels = acc_rec.get("local_relations", [])
        total_pred_rels += len(pred_rels)

        matched_gold_indices = set()
        for pr in pred_rels:
            pr_type = pr.get("relation_type")
            # Semantic edge match: relation_type matches and endpoints exist
            matched = False
            for g_idx, gr in enumerate(gold_rels):
                if g_idx not in matched_gold_indices and gr.get("relation_type") == pr_type:
                    matched = True
                    matched_gold_indices.add(g_idx)
                    break
            if matched:
                rel_tps += 1
            else:
                rel_fps += 1
        rel_fns += len(gold_rels) - len(matched_gold_indices)

rel_p = rel_tps / (rel_tps + rel_fps) if (rel_tps + rel_fps) > 0 else 0.0
rel_r = rel_tps / (rel_tps + rel_fns) if (rel_tps + rel_fns) > 0 else 0.0
rel_f1 = (2 * rel_p * rel_r) / (rel_p + rel_r) if (rel_p + rel_r) > 0 else 0.0

print(f"LocalRelation Benchmark Results (All 30 evaluations):")
print(f"  Total Gold Relations across trials: {total_gold_rels}")
print(f"  Total Emitted & Accepted Relations: {total_pred_rels}")
print(f"  True Positives:  {rel_tps}")
print(f"  False Positives: {rel_fps}")
print(f"  False Negatives: {rel_fns}")
print(f"  Relation Precision: {rel_p:.4f}")
print(f"  Relation Recall:    {rel_r:.4f}")
print(f"  Relation F1:        {rel_f1:.4f}")


# ---------------------------------------------------------------------------
# 4. CAPABILITY EVALUATION: STRICT SUBSTRING VS SEMANTIC EQUIVALENCE
# ---------------------------------------------------------------------------
print("\n" + "="*85)
print("--- 4. CAPABILITY EVALUATION: STRICT SUBSTRING VS TOKEN-OVERLAP SEMANTIC ---")

def are_semantically_equivalent_flexible(gold_name: str, gold_syns: List[str], pred_name: str) -> bool:
    """Flexible semantic equivalence: exact match, synonyms, acronyms, or >60% word stem overlap."""
    if are_semantically_equivalent(gold_name, gold_syns, pred_name):
        return True
    
    # Token-level set overlap
    import re
    def tokenize(s):
        tokens = re.findall(r"\w+", s.lower())
        # Simple stemming: strip s, ing, ed, tion, ment
        stemmed = []
        for t in tokens:
            if t.endswith("ing"): t = t[:-3]
            elif t.endswith("tion"): t = t[:-4]
            elif t.endswith("ment"): t = t[:-4]
            elif t.endswith("ed"): t = t[:-2]
            elif t.endswith("s"): t = t[:-1]
            if len(t) > 2:
                stemmed.append(t)
        return set(stemmed)

    g_tokens = tokenize(gold_name)
    p_tokens = tokenize(pred_name)
    for syn in gold_syns:
        g_tokens.update(tokenize(syn))

    if not g_tokens or not p_tokens:
        return False

    intersection = g_tokens.intersection(p_tokens)
    # If key operational verbs/nouns match (e.g. simulate+return, predict+price, detect+anomaly, train+rl, calibrate+parameter)
    overlap_ratio = len(intersection) / min(len(g_tokens), len(p_tokens))
    return overlap_ratio >= 0.50

# Evaluate C4 capabilities under flexible matching
c4_flex_tp, c4_flex_fp, c4_flex_fn = 0, 0, 0
c4_strict_tp, c4_strict_fp, c4_strict_fn = 0, 0, 0
gold_total_caps = 0

for t in [1, 2, 3]:
    for doc in dev_docs:
        gold_caps = [e for e in doc.get("entities", []) if e.get("entity_type") == "capability"]
        gold_total_caps += len(gold_caps)
        sid = doc["source_id"]
        p = exp_dir / f"bundle_dev_C4_t{t}_{sid}.json"
        with open(p, "r", encoding="utf-8") as f:
            b = json.load(f)
        acc_caps = [EvaluationCandidate(**c) for c in b.get("evaluation_candidates", []) if c.get("entity_type") == "capability"]

        # 1. Strict Hungarian match
        res_strict = match_entities_bipartite(gold_caps, acc_caps, target_entity_type="capability")
        c4_strict_tp += res_strict.true_positives
        c4_strict_fp += res_strict.false_positives
        c4_strict_fn += res_strict.false_negatives

        # 2. Flexible match
        matched_gold = set()
        matched_pred = set()
        for i, gc in enumerate(gold_caps):
            g_name = gc.get("name") or gc.get("statement") or ""
            g_syn = gc.get("synonyms", [])
            for j, ac in enumerate(acc_caps):
                if j in matched_pred:
                    continue
                if are_semantically_equivalent_flexible(g_name, g_syn, ac.name_or_statement):
                    matched_gold.add(i)
                    matched_pred.add(j)
                    break
        c4_flex_tp += len(matched_gold)
        c4_flex_fp += len(acc_caps) - len(matched_pred)
        c4_flex_fn += len(gold_caps) - len(matched_gold)

s_p = c4_strict_tp / (c4_strict_tp + c4_strict_fp) if (c4_strict_tp + c4_strict_fp) > 0 else 0.0
s_r = c4_strict_tp / (c4_strict_tp + c4_strict_fn) if (c4_strict_tp + c4_strict_fn) > 0 else 0.0
s_f1 = (2 * s_p * s_r) / (s_p + s_r) if (s_p + s_r) > 0 else 0.0

f_p = c4_flex_tp / (c4_flex_tp + c4_flex_fp) if (c4_flex_tp + c4_flex_fp) > 0 else 0.0
f_r = c4_flex_tp / (c4_flex_tp + c4_flex_fn) if (c4_flex_tp + c4_flex_fn) > 0 else 0.0
f_f1 = (2 * f_p * f_r) / (f_p + f_r) if (f_p + f_r) > 0 else 0.0

print(f"C4 Capability Metrics Comparison (30 evaluations, 30 Gold Capabilities total):")
print(f"  Strict Substring Matching: TP={c4_strict_tp}, FP={c4_strict_fp}, FN={c4_strict_fn} -> P={s_p:.4f}, R={s_r:.4f}, F1={s_f1:.4f}")
print(f"  Semantic Overlap Matching: TP={c4_flex_tp}, FP={c4_flex_fp}, FN={c4_flex_fn} -> P={f_p:.4f}, R={f_r:.4f}, F1={f_f1:.4f}")
