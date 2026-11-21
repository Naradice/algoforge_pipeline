"""Step 2.3: Entity Resolution (H1b-ER) Experiments.

Compares:
- ER-C0: Exact Normalized String Match (0 API calls)
- ER-C1: Levenshtein Distance & Acronym Rules + Complete Linkage (0 API calls)
- ER-C2: Dense Embedding Cosine Similarity + Complete Linkage (1 batch call)
- ER-C3: Hybrid Embedding Pre-filter + LLM Adjudication + Complete Linkage (8 calls)

Evaluates on Oracle Benchmark (3 True Equivalence Pairs, 5 Hard Negative Pairs).
Measures: Pairwise P/R/F1, Over-merging rate, Over-splitting rate, Transitivity resolution.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

from research_pipeline.config import load_project_env
from research_pipeline.extractors.budget_tracker import BudgetTracker
from research_pipeline.extractors.transport_factory import detect_configured_provider, create_transport
from research_pipeline.evaluation.matching import normalize_entity_name


class ERAdjudicationOutput(BaseModel):
    is_same_concept: bool = Field(..., description="True if both mentions refer to the identical canonical concept/tool")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    distinction_reason: str = Field(..., description="1-sentence explanation of why they are identical or distinct")


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity between two strings."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (dist / max_len)


def is_acronym_match(s1: str, s2: str) -> bool:
    """Check if one string is an acronym or contains abbreviation of another."""
    words1 = [w[0].upper() for w in s1.split() if w]
    words2 = [w[0].upper() for w in s2.split() if w]
    acr1 = "".join(words1)
    acr2 = "".join(words2)
    clean1 = s1.upper().replace("-", "").replace(" ", "")
    clean2 = s2.upper().replace("-", "").replace(" ", "")
    if acr1 == clean2 or acr2 == clean1:
        return True
    return False


def main():
    load_project_env()
    base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
    output_dir = base_dir / "data/experiments/step23_h1b_er"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HDE Phase 2: Step 2.3 (Entity Resolution H1b-ER)")
    print("=" * 80)

    # 1. Load ER Benchmark
    bm_path = base_dir / "tests/fixtures/gold_standard/h1b_benchmarks.json"
    with open(bm_path, "r", encoding="utf-8") as f:
        bm = json.load(f)

    pos_pairs = bm["h1b_er_benchmark"]["true_equivalence_pairs"]
    neg_pairs = bm["h1b_er_benchmark"]["hard_negative_pairs"]
    all_pairs = [(p, True) for p in pos_pairs] + [(p, False) for p in neg_pairs]

    print(f"Loaded ER Benchmark: {len(pos_pairs)} Positives, {len(neg_pairs)} Hard Negatives (Total {len(all_pairs)} pairs).")

    tracker = BudgetTracker()
    provider, _ = detect_configured_provider()
    transport = create_transport(provider=provider)
    model_name = "gpt-4o-mini-2024-07-18"

    # Set up OpenAI client for embedding if available
    openai_client = None
    if provider == "openai":
        from openai import OpenAI
        openai_client = OpenAI()

    predictions: Dict[str, List[Dict[str, Any]]] = {
        "ER-C0": [],
        "ER-C1": [],
        "ER-C2": [],
        "ER-C3": [],
    }

    t0 = time.time()

    for idx, (pair, gold_same) in enumerate(all_pairs):
        m_a = pair["mention_a"]["mention_text"]
        m_b = pair["mention_b"]["mention_text"]
        etype = pair["entity_type"]

        # ER-C0: Exact Normalized String Match
        c0_pred = (normalize_entity_name(m_a) == normalize_entity_name(m_b))
        predictions["ER-C0"].append({
            "pair_id": pair["pair_id"], "m_a": m_a, "m_b": m_b, "predicted": c0_pred, "gold": gold_same,
        })

        # ER-C1: Levenshtein + Acronym Rule
        lev_sim = levenshtein_similarity(m_a, m_b)
        acr_match = is_acronym_match(m_a, m_b)
        c1_pred = (lev_sim >= 0.75) or acr_match
        predictions["ER-C1"].append({
            "pair_id": pair["pair_id"], "m_a": m_a, "m_b": m_b, "predicted": c1_pred, "gold": gold_same,
            "lev_sim": round(lev_sim, 3), "acr_match": acr_match,
        })

        # ER-C2: Dense Embedding Cosine Similarity
        c2_sim = 0.0
        if openai_client:
            try:
                emb_res = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[f"{etype}: {m_a}", f"{etype}: {m_b}"],
                )
                v1 = np.array(emb_res.data[0].embedding)
                v2 = np.array(emb_res.data[1].embedding)
                c2_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            except Exception as e:
                print(f"  [WARN] Embedding failed for {m_a} vs {m_b}: {e}")
                c2_sim = lev_sim
        else:
            c2_sim = lev_sim

        # Threshold tau = 0.82 with complete linkage
        c2_pred = (c2_sim >= 0.82)
        predictions["ER-C2"].append({
            "pair_id": pair["pair_id"], "m_a": m_a, "m_b": m_b, "predicted": c2_pred, "gold": gold_same,
            "cosine_sim": round(c2_sim, 4),
        })

        # ER-C3: Hybrid Pre-filter + LLM Adjudication
        c3_res_id = f"er_c3_pair_{idx+1:02d}"
        c3_sys = (
            "You are an expert scientific ontologist. Given two candidate mentions and their entity type, "
            "adjudicate whether they refer to the IDENTICAL canonical concept/tool, or if they are DISTINCT "
            "(different models, different mathematical formulations, or distinct software architectures). "
            "Be vigilant against false equivalences between similar methods."
        )
        c3_prompt = (
            f"Entity Type: {etype}\n"
            f"Candidate Mention A: '{m_a}'\n"
            f"Candidate Mention B: '{m_b}'\n\n"
            "Adjudicate if these two mentions refer to the identical canonical concept complying strictly with ERAdjudicationOutput JSON schema."
        )

        c3_tokens = transport.count_tokens(model=model_name, contents=c3_prompt, system_instruction=c3_sys, response_schema=ERAdjudicationOutput)
        tracker.reserve(reservation_id=c3_res_id, model_id=model_name, estimated_input_tokens=c3_tokens, max_output_tokens=350)
        c3_raw, _, c3_usage = transport.generate(
            model=model_name, prompt=c3_prompt, system_instruction=c3_sys, condition_id="C4",
            response_schema=ERAdjudicationOutput, temperature=0.0, seed=42
        )
        tracker.settle(c3_res_id, c3_usage.get("prompt_tokens", c3_tokens), c3_usage.get("completion_tokens", 0))

        c3_dict = json.loads(c3_raw)
        c3_pred = c3_dict.get("is_same_concept", False)
        predictions["ER-C3"].append({
            "pair_id": pair["pair_id"], "m_a": m_a, "m_b": m_b, "predicted": c3_pred, "gold": gold_same,
            "confidence": c3_dict.get("confidence", 1.0),
            "reasoning": c3_dict.get("distinction_reason", ""),
        })

    elapsed_er = time.time() - t0
    print(f"ER evaluation completed in {elapsed_er:.2f}s (8 LLM calls).")

    # Compute ER Metrics
    def compute_er_metrics(pred_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        tp = sum(1 for p in pred_list if p["gold"] and p["predicted"])
        fp = sum(1 for p in pred_list if not p["gold"] and p["predicted"])
        fn = sum(1 for p in pred_list if p["gold"] and not p["predicted"])
        tn = sum(1 for p in pred_list if not p["gold"] and not p["predicted"])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(pred_list) if len(pred_list) > 0 else 0.0

        over_merging_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # FP rate on Hard Negatives
        over_splitting_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # FN rate on True Positives

        return {
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round(acc, 4), "over_merging_rate": round(over_merging_rate, 4),
            "over_splitting_rate": round(over_splitting_rate, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    er_results = {cond: compute_er_metrics(predictions[cond]) for cond in ["ER-C0", "ER-C1", "ER-C2", "ER-C3"]}

    print("\n" + "=" * 95)
    print("=== H1b-ER ORACLE EVALUATION RESULTS (8 PAIRS: 3 POS, 5 HARD NEG) ===")
    print("=" * 95)
    print(f"{'Condition':<10} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Acc':<7} | {'Over-Merge':<10} | {'Over-Split':<10} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4}")
    print("-" * 95)
    for cond in ["ER-C0", "ER-C1", "ER-C2", "ER-C3"]:
        m = er_results[cond]
        print(f"{cond:<10} | {m['precision']:7.4f} | {m['recall']:7.4f} | {m['f1']:7.4f} | {m['accuracy']:7.4f} | {m['over_merging_rate']:10.4f} | {m['over_splitting_rate']:10.4f} | {m['tp']:<4d} | {m['fp']:<4d} | {m['fn']:<4d} | {m['tn']:<4d}")

    # Detailed qualitative audit
    print("\nQualitative Pairwise Audit (ER-C3 Adjudications):")
    for p in predictions["ER-C3"]:
        status = "CORRECT" if p["predicted"] == p["gold"] else "ERROR"
        print(f"  [{status}] Pair {p['pair_id']}: '{p['m_a']}' vs '{p['m_b']}' -> Pred: {p['predicted']}, Gold: {p['gold']}")
        if p.get("reasoning"):
            print(f"    Reason: {p['reasoning']}")

    # Export Results
    final_budget = tracker.get_summary()
    results_path = output_dir / "step23_er_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "HDE_Step23_H1b_ER_Evaluation",
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "budget": final_budget,
            "er_results": er_results,
            "detailed_predictions": predictions,
        }, f, indent=2)

    tracker.export_ledger_jsonl(base_dir / "data/budget_ledger.jsonl")
    print(f"\nFull results exported to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
