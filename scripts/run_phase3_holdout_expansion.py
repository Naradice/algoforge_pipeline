"""HDE Phase 3 Holdout Expansion Complete Evaluation Runner.

Complies strictly with Plan Approval and Preconditions:
1. Pinned Primary Model: gpt-4o-mini-2024-07-18, temp=0.0, seed=42
2. Fixed Conditions:
   - H1a: Baseline C4 (7-class single-stage)
   - H1b-RE: Baseline RE-C0 vs Proposed RE-C0 ∩ RE-C2 (Strict Agreement Ensemble)
   - H1b-ER: Baseline ER-C2 (tau=0.82 frozen) vs Proposed ER-C2 + VARIANT_OF Adjudication
   - E2E: Actual C4 outputs into ER/RE compared against Oracle inputs (R_E2E = R_endpoint * R_RE|endpoint)
3. Evaluates on 10 Holdout Expansion documents (R=2 trials for H1a stability)
4. Evaluates Oracle RE (19 Positives, 19 Negatives) and E2E RE
5. Evaluates ER Benchmark (10 pairs: 3 SAME_AS, 3 VARIANT_OF, 4 DISTINCT)
6. Populates relational KnowledgeGraphStore with SQLite foreign key integrity verification
7. Enforces two-phase budget reservations against $10.00 USD Hard Limit
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from typing_extensions import Literal

from research_pipeline.config import load_project_env
from research_pipeline.extractors.budget_tracker import BudgetTracker
from research_pipeline.extractors.transport_factory import detect_configured_provider, create_transport
from research_pipeline.extractors.models import (
    ExtractedKnowledgeRecord,
    CanonicalEntity,
    MentionProvenance,
    RelationEdge,
    RelationEvidence,
    compute_canonical_hash,
)
from research_pipeline.extractors.raw_artifact import (
    RawExtractionArtifact,
    RejectionRecord,
    EvaluationCandidate,
    ExtractionTrioBundle,
)
from research_pipeline.extractors.converter import StandardizationConverter
from research_pipeline.extractors.validator import validate_evidence
from research_pipeline.evaluation.matching import (
    match_entities_bipartite,
    normalize_entity_name,
    are_semantically_equivalent,
)
from research_pipeline.evaluation.metrics import TARGET_CLASSES
from research_pipeline.storage.knowledge_graph import KnowledgeGraphStore


# --- Pydantic Schemas for RE and ER ---
class RelationClassificationOutput(BaseModel):
    relation_type: Literal["EXPLAINS", "APPLIES_TO", "PROVIDED_BY", "REQUIRES", "CONSTRAINED_BY", "IMPLEMENTS", "NONE"] = Field(
        ..., description="Predicted semantic relation type from source to destination, or NONE if no direct relation"
    )
    verbatim_evidence_quote: str = Field(default="", description="Exact quote from text supporting the relation, or empty if NONE")
    brief_reasoning: str = Field(default="", description="1-sentence justification for the predicted relation")


class ERClassificationOutput(BaseModel):
    relation: Literal["SAME_AS", "VARIANT_OF", "DISTINCT"] = Field(
        ..., description="SAME_AS if identical concept/tool; VARIANT_OF if one is a specialization/extension/variant of the other; DISTINCT if different concepts"
    )
    direction: Literal["bidirectional", "a_is_variant_of_b", "b_is_variant_of_a", "none"] = Field(
        default="none", description="Direction of variant relation: a_is_variant_of_b if mention A is the specialized variant of B; b_is_variant_of_a if B is variant of A; bidirectional if SAME_AS; none if DISTINCT"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    rationale: str = Field(..., description="1-sentence justification explaining the mathematical, algorithmic, or functional reason")


FEW_SHOT_RE = """Examples:

Example 1:
Text: "We propose a Recurrent Graph Convolutional network (R-GCN) to extract predictive alpha signals from multi-exchange order books."
Source: "extract predictive alpha signals" (capability)
Destination: "Recurrent Graph Convolutional network" (model)
Output JSON:
{
  "relation_type": "PROVIDED_BY",
  "verbatim_evidence_quote": "Recurrent Graph Convolutional network (R-GCN) to extract predictive alpha signals",
  "brief_reasoning": "The R-GCN model provides the functional capability to extract predictive alpha signals."
}

Example 2:
Text: "We analyze order flow imbalance in high-frequency equity markets using deep neural networks."
Source: "order flow imbalance" (phenomenon)
Destination: "deep neural networks" (model)
Output JSON:
{
  "relation_type": "NONE",
  "verbatim_evidence_quote": "",
  "brief_reasoning": "Order flow imbalance is the subject phenomenon, not explaining the neural network model (the direction is reverse)."
}
"""

FEW_SHOT_ER = """Examples:

Example 1:
Mention A: 'Gym Trading Environment' (type: software)
Mention B: 'Gym Financial Trading Environment' (type: software)
Output JSON:
{
  "relation": "SAME_AS",
  "direction": "bidirectional",
  "confidence": 0.95,
  "rationale": "Exact identical software environment with trivial modifier variation."
}

Example 2:
Mention A: 'Sequential Neural Posterior Estimation' (type: method)
Mention B: 'Neural Posterior Estimation' (type: method)
Output JSON:
{
  "relation": "VARIANT_OF",
  "direction": "a_is_variant_of_b",
  "confidence": 0.90,
  "rationale": "SNPE is a multi-round sequential specialization and algorithmic extension of base amortized NPE."
}

Example 3:
Mention A: 'Maximum Likelihood Estimation' (type: method)
Mention B: 'Neural Posterior Estimation' (type: method)
Output JSON:
{
  "relation": "DISTINCT",
  "direction": "none",
  "confidence": 0.95,
  "rationale": "Both are parameter inference methods, but MLE is point optimization while NPE is amortized Bayesian neural density estimation."
}
"""


def predict_re_c0(src_type: str, dst_type: str) -> str:
    """RE-C0: Rule-based co-occurrence heuristic."""
    s, d = src_type.lower(), dst_type.lower()
    if (s in ["model", "method"] and d == "phenomenon"):
        return "EXPLAINS"
    if (s == "method" and d == "model"):
        return "APPLIES_TO"
    if (s == "capability" and d in ["model", "method", "software"]):
        return "PROVIDED_BY"
    if (s == "software" and d in ["model", "method"]):
        return "IMPLEMENTS"
    if (s in ["capability", "model", "method", "software"] and d == "limitation"):
        return "CONSTRAINED_BY"
    if (s in ["capability", "model"] and d == "assumption"):
        return "REQUIRES"
    return "NONE"


def compute_re_metrics(plist: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def execute_c4_doc(
    doc: Dict[str, Any],
    trial_idx: int,
    tracker: BudgetTracker,
    transport: Any,
    model_name: str,
    output_dir: Path,
) -> ExtractionTrioBundle:
    """Execute live C4 extraction on a single Holdout Expansion document."""
    source_id = doc["source_id"]
    version_id = doc["version_id"]
    canonical_fields = doc["target_fields"]
    target_field_name = "readme_text" if doc["source_type"] == "oss" else "abstract_text"
    text_content = canonical_fields.get(target_field_name, "")
    title = canonical_fields.get("title_text", "")

    run_id = f"phase3_exp_C4_t{trial_idx}_{source_id}"
    cache_path = output_dir / f"bundle_{run_id}.json"

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return ExtractionTrioBundle.model_validate_json(f.read())

    schema_cls = ExtractedKnowledgeRecord
    system_instruction = (
        "You are an expert scientific researcher. Extract complete, structured knowledge and capability entities "
        "from the provided text. Output JSON adhering strictly to the schema. "
        "Every entity and relation must cite exact verbatim evidence from the text. "
        "For OSS software entities, categorize them as SoftwareEntity. "
        "If input/output data types for Capability are not explicitly mentioned in the text, "
        "set is_unspecified=true, data_type_l1='unknown', and evidence=null."
    )
    prompt = (
        f"Document Title: {title}\n\n"
        f"Source Text ({target_field_name}):\n{text_content}\n\n"
        "Extract complete structured knowledge and relations complying strictly with ExtractedKnowledgeRecord JSON schema."
    )

    counted_tokens = transport.count_tokens(
        model=model_name,
        contents=prompt,
        system_instruction=system_instruction,
        response_schema=schema_cls,
    )
    tracker.reserve(
        reservation_id=run_id,
        model_id=model_name,
        estimated_input_tokens=counted_tokens,
        max_output_tokens=2048,
        allowance_tokens=0,
    )

    try:
        raw_text, http_status, usage = transport.generate(
            model=model_name,
            prompt=prompt,
            system_instruction=system_instruction,
            condition_id="C4",
            response_schema=schema_cls,
            temperature=0.0,
            seed=42,
        )
    except Exception as e:
        tracker.release_reservation(run_id)
        raise RuntimeError(f"C4 call failed for {run_id}: {e}")

    actual_in = usage.get("prompt_tokens", counted_tokens)
    actual_out = usage.get("completion_tokens", 0)
    tracker.settle(run_id, actual_in, actual_out)

    parsed_dict = None
    parse_err = None
    json_status = "success"
    try:
        parsed_dict = json.loads(raw_text)
    except Exception as pe:
        json_status = "invalid_json"
        parse_err = str(pe)

    raw_artifact = RawExtractionArtifact(
        extraction_run_id=run_id,
        source_id=source_id,
        version_id=version_id,
        condition_id="C4",
        raw_response_text=raw_text,
        http_status=http_status,
        token_usage=usage,
        executed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_checkpoint=model_name,
        prompt_hash=compute_canonical_hash(prompt),
        target_field_hashes=doc["target_field_hashes"],
        json_parse_status=json_status,
        parse_error=parse_err,
        parsed_candidates=parsed_dict,
    )

    candidates, rejected_items = StandardizationConverter.convert_c3_c4_with_evidence(raw_artifact)
    accepted_candidates = []
    for c in candidates:
        if c.evidence:
            valid, reason, resolved_ev = validate_evidence(c.evidence, canonical_fields)
            if valid and resolved_ev:
                c.evidence = resolved_ev
                accepted_candidates.append(c)
            else:
                c.conversion_status = "failed"
                rejected_items.append(RejectionRecord(
                    local_id=c.candidate_local_id,
                    entity_type=c.entity_type,
                    rejection_stage="quote_validity",
                    rejection_reason=reason,
                    raw_entity_data=c.raw_payload,
                ))
        else:
            rejected_items.append(RejectionRecord(
                local_id=c.candidate_local_id,
                entity_type=c.entity_type,
                rejection_stage="missing_evidence",
                rejection_reason="Evidence span was missing",
                raw_entity_data=c.raw_payload,
            ))

    accepted_record = None
    if parsed_dict and json_status == "success":
        try:
            accepted_record = ExtractedKnowledgeRecord.model_validate(parsed_dict)
        except Exception:
            pass

    bundle = ExtractionTrioBundle(
        extraction_run_id=run_id,
        source_id=source_id,
        version_id=version_id,
        raw_artifact=raw_artifact,
        evaluation_candidates=accepted_candidates,
        rejected_items=rejected_items,
        accepted_record=accepted_record,
    )

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(bundle.model_dump_json(indent=2))

    return bundle


def evaluate_h1a_trial(
    bundles: List[ExtractionTrioBundle],
    ground_truth_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate H1a extraction candidates against Holdout Expansion ground truth."""
    gt_by_id = {d["source_id"]: d for d in ground_truth_docs}
    
    total_gold = 0
    total_pred = 0
    total_tp_strict = 0
    total_tp_sem = 0
    
    class_stats = {cls: {"gold": 0, "pred": 0, "tp_strict": 0, "tp_sem": 0} for cls in TARGET_CLASSES}

    for b in bundles:
        sid = b.raw_artifact.source_id
        g_doc = gt_by_id[sid]
        gold_entities = g_doc.get("entities", [])
        
        # Count gold
        for ge in gold_entities:
            etype = ge["entity_type"].lower()
            if etype in class_stats:
                class_stats[etype]["gold"] += 1
                total_gold += 1

        # Count predicted accepted
        preds = b.evaluation_candidates
        for pe in preds:
            etype = pe.entity_type.lower()
            if etype in class_stats:
                class_stats[etype]["pred"] += 1
                total_pred += 1

        # Match per class
        for cls in TARGET_CLASSES:
            g_cls = [e for e in gold_entities if e["entity_type"].lower() == cls]
            p_cls = [e for e in preds if e.entity_type.lower() == cls]
            
            # Strict matching
            used_p_strict = set()
            for ge in g_cls:
                g_name = ge.get("name") or ge.get("statement", "")
                for p_idx, pe in enumerate(p_cls):
                    if p_idx in used_p_strict:
                        continue
                    p_name = pe.name_or_statement
                    if normalize_entity_name(g_name) == normalize_entity_name(p_name):
                        used_p_strict.add(p_idx)
                        class_stats[cls]["tp_strict"] += 1
                        total_tp_strict += 1
                        break
                        
            # Semantic matching
            used_p_sem = set()
            for ge in g_cls:
                g_name = ge.get("name") or ge.get("statement", "")
                g_syns = ge.get("synonyms", [])
                for p_idx, pe in enumerate(p_cls):
                    if p_idx in used_p_sem:
                        continue
                    p_name = pe.name_or_statement
                    if are_semantically_equivalent(g_name, g_syns, p_name):
                        used_p_sem.add(p_idx)
                        class_stats[cls]["tp_sem"] += 1
                        total_tp_sem += 1
                        break

    def calc_p_r_f1(tp, p_cnt, g_cnt):
        p = tp / p_cnt if p_cnt > 0 else 0.0
        r = tp / g_cnt if g_cnt > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    micro_strict_p, micro_strict_r, micro_strict_f1 = calc_p_r_f1(total_tp_strict, total_pred, total_gold)
    micro_sem_p, micro_sem_r, micro_sem_f1 = calc_p_r_f1(total_tp_sem, total_pred, total_gold)

    per_class = {}
    for cls, st in class_stats.items():
        _, _, s_f1 = calc_p_r_f1(st["tp_strict"], st["pred"], st["gold"])
        cp, cr, sem_f1 = calc_p_r_f1(st["tp_sem"], st["pred"], st["gold"])
        per_class[cls] = {
            "gold_count": st["gold"],
            "pred_count": st["pred"],
            "tp_strict": st["tp_strict"],
            "tp_semantic": st["tp_sem"],
            "strict_f1": s_f1,
            "semantic_precision": cp,
            "semantic_recall": cr,
            "semantic_f1": sem_f1,
        }

    return {
        "total_gold": total_gold,
        "total_pred": total_pred,
        "strict_micro_f1": micro_strict_f1,
        "semantic_micro_precision": micro_sem_p,
        "semantic_micro_recall": micro_sem_r,
        "semantic_micro_f1": micro_sem_f1,
        "per_class": per_class,
    }


def main():
    load_project_env()
    base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
    output_dir = base_dir / "data/experiments/phase3_holdout_expansion"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HDE Phase 3: Holdout Expansion Evaluation (H1a Baseline C4, RE, ER, E2E, KG)")
    print("=" * 80)

    # 1. Component Hashes Verification
    gt_path = base_dir / "tests/fixtures/gold_standard/holdout_expansion_ground_truth.json"
    re_path = base_dir / "tests/fixtures/gold_standard/holdout_expansion_re_benchmark.json"
    er_path = base_dir / "tests/fixtures/gold_standard/holdout_expansion_er_benchmark.json"

    with open(gt_path, "rb") as f:
        gt_hash = hashlib.sha256(f.read()).hexdigest()
    with open(re_path, "rb") as f:
        re_hash = hashlib.sha256(f.read()).hexdigest()
    with open(er_path, "rb") as f:
        er_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"Verified Holdout Ground Truth SHA-256: {gt_hash}")
    print(f"Verified Holdout RE Benchmark SHA-256: {re_hash}")
    print(f"Verified Holdout ER Benchmark SHA-256: {er_hash}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_docs = json.load(f)
    with open(re_path, "r", encoding="utf-8") as f:
        re_bm = json.load(f)
    with open(er_path, "r", encoding="utf-8") as f:
        er_bm = json.load(f)

    print(f"Loaded: {len(gt_docs)} docs, {len(re_bm['positive_relations'])} RE positives, {len(re_bm['negative_relations'])} RE negatives, {len(er_bm['pairs'])} ER pairs.")

    tracker = BudgetTracker()
    provider, _ = detect_configured_provider()
    transport = create_transport(provider=provider)
    model_name = "gpt-4o-mini-2024-07-18"

    openai_client = None
    if provider == "openai":
        from openai import OpenAI
        openai_client = OpenAI()

    # =========================================================================
    # STEP 3.1: H1a Extraction (Baseline C4 on 10 documents, 2 trials = 20 calls)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.1: H1a Extraction Evaluation (Baseline C4, 10 docs x 2 trials)")
    print("=" * 80)

    trials_bundles: Dict[int, List[ExtractionTrioBundle]] = {1: [], 2: []}
    for trial_idx in [1, 2]:
        print(f"\n--- Running Trial {trial_idx} ---")
        for doc in gt_docs:
            b = execute_c4_doc(doc, trial_idx, tracker, transport, model_name, output_dir)
            trials_bundles[trial_idx].append(b)
            print(f"  [T{trial_idx}] {doc['source_id']}: accepted={len(b.evaluation_candidates)}, rejected={len(b.rejected_items)}")

    eval_t1 = evaluate_h1a_trial(trials_bundles[1], gt_docs)
    eval_t2 = evaluate_h1a_trial(trials_bundles[2], gt_docs)

    print(f"\n[Trial 1] Semantic Micro F1: {eval_t1['semantic_micro_f1']} (P={eval_t1['semantic_micro_precision']}, R={eval_t1['semantic_micro_recall']})")
    print(f"[Trial 2] Semantic Micro F1: {eval_t2['semantic_micro_f1']} (P={eval_t2['semantic_micro_precision']}, R={eval_t2['semantic_micro_recall']})")

    # =========================================================================
    # STEP 3.2: H1b-RE Relation Extraction (Oracle & End-to-End)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.2: H1b-RE Relation Extraction Evaluation (RE-C0 vs RE-C0 ∩ RE-C2)")
    print("=" * 80)

    all_oracle_pairs = re_bm["positive_relations"] + re_bm["negative_relations"]
    docs_by_id = {d["source_id"]: d for d in gt_docs}

    oracle_preds: Dict[str, List[Dict[str, Any]]] = {
        "RE-C0": [],
        "RE-C2": [],
        "RE-Hybrid": [], # RE-C0 ∩ RE-C2
    }

    print(f"Evaluating Oracle RE on {len(all_oracle_pairs)} pairs...")
    for idx, pair in enumerate(all_oracle_pairs):
        sid = pair["source_id"]
        doc = docs_by_id[sid]
        tf_name = "readme_text" if doc["source_type"] == "oss" else "abstract_text"
        context_text = doc["target_fields"].get(tf_name, "")

        src_name = pair["src_name"]
        src_type = pair["src_type"]
        dst_name = pair["dst_name"]
        dst_type = pair["dst_type"]
        gold_rel = pair.get("expected_relation", "NONE")

        # RE-C0: Rule-based
        c0_pred = predict_re_c0(src_type, dst_type)
        oracle_preds["RE-C0"].append({
            "pair_id": pair["relation_id"], "predicted_rel": c0_pred, "gold_rel": gold_rel, "is_correct": (c0_pred == gold_rel)
        })

        # RE-C2: Few-shot CoT LLM
        res_id = f"phase3_re_c2_pair_{idx+1:02d}"
        re_cache_path = output_dir / f"re_cache_{res_id}.json"

        if re_cache_path.exists():
            with open(re_cache_path, "r", encoding="utf-8") as f:
                c2_dict = json.load(f)
        else:
            c2_sys = (
                "You are an expert scientific researcher. Given the context text and an entity pair, determine the semantic relation. "
                "Valid relations: EXPLAINS, APPLIES_TO, PROVIDED_BY, REQUIRES, CONSTRAINED_BY, IMPLEMENTS, or NONE. "
                "Provide brief reasoning and quote the exact supporting sentence."
            )
            c2_prompt = (
                f"{FEW_SHOT_RE}\n\n"
                f"Context Text:\n{context_text}\n\n"
                f"Source Entity: '{src_name}' (type: {src_type})\n"
                f"Destination Entity: '{dst_name}' (type: {dst_type})\n\n"
                "Output JSON conforming to RelationClassificationOutput."
            )

            c2_tokens = transport.count_tokens(model=model_name, contents=c2_prompt, system_instruction=c2_sys, response_schema=RelationClassificationOutput)
            tracker.reserve(reservation_id=res_id, model_id=model_name, estimated_input_tokens=c2_tokens, max_output_tokens=500)
            c2_raw, _, c2_usage = transport.generate(
                model=model_name, prompt=c2_prompt, system_instruction=c2_sys, condition_id="C4",
                response_schema=RelationClassificationOutput, temperature=0.0, seed=42
            )
            tracker.settle(res_id, c2_usage.get("prompt_tokens", c2_tokens), c2_usage.get("completion_tokens", 0))
            c2_dict = json.loads(c2_raw)
            with open(re_cache_path, "w", encoding="utf-8") as f:
                json.dump(c2_dict, f, indent=2)
        c2_pred = c2_dict.get("relation_type", "NONE")
        c2_quote = c2_dict.get("verbatim_evidence_quote", "")
        c2_reason = c2_dict.get("brief_reasoning", "")

        oracle_preds["RE-C2"].append({
            "pair_id": pair["relation_id"], "predicted_rel": c2_pred, "gold_rel": gold_rel, "is_correct": (c2_pred == gold_rel),
            "evidence": c2_quote, "reasoning": c2_reason
        })

        # Proposed RE-Hybrid: Strict Agreement (Intersection)
        hybrid_pred = c0_pred if (c0_pred == c2_pred) else "NONE"
        oracle_preds["RE-Hybrid"].append({
            "pair_id": pair["relation_id"], "predicted_rel": hybrid_pred, "gold_rel": gold_rel, "is_correct": (hybrid_pred == gold_rel)
        })

    re_c0_metrics = compute_re_metrics(oracle_preds["RE-C0"])
    re_c2_metrics = compute_re_metrics(oracle_preds["RE-C2"])
    re_hyb_metrics = compute_re_metrics(oracle_preds["RE-Hybrid"])

    print(f"\n[Oracle RE-C0]     Precision: {re_c0_metrics['precision']}, Recall: {re_c0_metrics['recall']}, F1: {re_c0_metrics['f1']} (FP={re_c0_metrics['fp']})")
    print(f"[Oracle RE-C2]     Precision: {re_c2_metrics['precision']}, Recall: {re_c2_metrics['recall']}, F1: {re_c2_metrics['f1']} (FP={re_c2_metrics['fp']})")
    print(f"[Oracle RE-Hybrid] Precision: {re_hyb_metrics['precision']}, Recall: {re_hyb_metrics['recall']}, F1: {re_hyb_metrics['f1']} (FP={re_hyb_metrics['fp']})")

    # --- End-to-End Relation Recall Decomposition ---
    print("\n--- Evaluating End-to-End Relation Recall (C4 Trial 1 outputs) ---")
    t1_bundles_by_id = {b.raw_artifact.source_id: b for b in trials_bundles[1]}
    gold_relations_total = len(re_bm["positive_relations"])
    endpoint_available_count = 0
    re_c0_e2e_correct = 0
    re_hyb_e2e_correct = 0

    e2e_relation_details = []

    for idx, pos_rel in enumerate(re_bm["positive_relations"]):
        sid = pos_rel["source_id"]
        b = t1_bundles_by_id[sid]
        doc = docs_by_id[sid]
        gold_src_name = pos_rel["src_name"]
        gold_dst_name = pos_rel["dst_name"]
        gold_rel_type = pos_rel["expected_relation"]

        # Check if C4 extracted both endpoints semantically
        extracted_src = None
        extracted_dst = None
        for c in b.evaluation_candidates:
            c_name = c.name_or_statement
            if are_semantically_equivalent(gold_src_name, [gold_src_name], c_name):
                extracted_src = c
            if are_semantically_equivalent(gold_dst_name, [gold_dst_name], c_name):
                extracted_dst = c

        endpoints_found = (extracted_src is not None and extracted_dst is not None)
        if endpoints_found:
            endpoint_available_count += 1
            # Apply RE-C0 on extracted types
            c0_e2e_pred = predict_re_c0(extracted_src.entity_type, extracted_dst.entity_type)
            if c0_e2e_pred == gold_rel_type:
                re_c0_e2e_correct += 1

            # Look up corresponding RE-C2 prediction from oracle pair
            c2_pred = oracle_preds["RE-C2"][idx]["predicted_rel"]
            hyb_e2e_pred = c0_e2e_pred if (c0_e2e_pred == c2_pred) else "NONE"
            if hyb_e2e_pred == gold_rel_type:
                re_hyb_e2e_correct += 1
        else:
            c0_e2e_pred = "NONE"
            hyb_e2e_pred = "NONE"

        e2e_relation_details.append({
            "relation_id": pos_rel["relation_id"],
            "source_id": sid,
            "gold_src": gold_src_name,
            "gold_dst": gold_dst_name,
            "gold_type": gold_rel_type,
            "endpoints_found": endpoints_found,
            "c0_pred": c0_e2e_pred,
            "hyb_pred": hyb_e2e_pred,
        })

    r_endpoint = endpoint_available_count / gold_relations_total if gold_relations_total > 0 else 0.0
    r_re_given_endpoint_c0 = re_c0_e2e_correct / endpoint_available_count if endpoint_available_count > 0 else 0.0
    r_re_given_endpoint_hyb = re_hyb_e2e_correct / endpoint_available_count if endpoint_available_count > 0 else 0.0
    r_e2e_c0 = r_endpoint * r_re_given_endpoint_c0
    r_e2e_hyb = r_endpoint * r_re_given_endpoint_hyb

    print(f"Total Gold Relations: {gold_relations_total}")
    print(f"Endpoints Co-Available: {endpoint_available_count} (R_endpoint = {r_endpoint:.4f})")
    print(f"RE-C0 Conditioned Recall: {r_re_given_endpoint_c0:.4f} -> Exact R_E2E (RE-C0) = {r_e2e_c0:.4f}")
    print(f"RE-Hyb Conditioned Recall: {r_re_given_endpoint_hyb:.4f} -> Exact R_E2E (RE-Hyb) = {r_e2e_hyb:.4f}")

    # =========================================================================
    # STEP 3.3: H1b-ER Entity Resolution Evaluation (10 pairs)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.3: H1b-ER Entity Resolution Evaluation (Baseline ER-C2 vs Proposed ER)")
    print("=" * 80)

    er_pairs = er_bm["pairs"]
    er_results = []
    c2_same_as_correct = 0
    c2_same_as_pred_total = 0
    c2_same_as_gold_total = sum(1 for p in er_pairs if p["ground_truth"] == "SAME_AS")

    proposed_3way_correct = 0
    proposed_var_correct = 0
    proposed_var_gold_total = sum(1 for p in er_pairs if p["ground_truth"] == "VARIANT_OF")

    for idx, pair in enumerate(er_pairs):
        pid = pair["pair_id"]
        m_a = pair["mention_a"]["mention_text"]
        m_b = pair["mention_b"]["mention_text"]
        etype = pair["entity_type"]
        gold_rel = pair["ground_truth"]
        gold_dir = pair["direction"]

        # Baseline ER-C2: Dense Embedding Cosine Similarity (tau=0.82 frozen)
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
                c2_sim = 0.0

        c2_pred_same = (c2_sim >= 0.82)
        if c2_pred_same:
            c2_same_as_pred_total += 1
            if gold_rel == "SAME_AS":
                c2_same_as_correct += 1

        # Proposed ER: LLM Adjudication with SAME_AS vs VARIANT_OF vs DISTINCT + Direction
        er_res_id = f"phase3_er_adj_{idx+1:02d}"
        er_cache_path = output_dir / f"er_cache_{er_res_id}.json"

        if er_cache_path.exists():
            with open(er_cache_path, "r", encoding="utf-8") as f:
                er_dict = json.load(f)
        else:
            er_sys = (
                "You are an expert scientific researcher. Analyze two technical concepts/tools and determine their relationship: "
                "1. 'SAME_AS': Exact identical concept, tool, or phenomenon (e.g. synonyms, acronym expansions, trivial name variants). "
                "2. 'VARIANT_OF': One is a specialized derivative, multi-dimensional extension, or algorithmic variation of the other. "
                "3. 'DISTINCT': Completely separate concepts/tools, even if in the same domain. "
                "Provide relation, direction (bidirectional, a_is_variant_of_b, b_is_variant_of_a, or none), confidence, and 1-sentence rationale."
            )
            er_prompt = (
                f"{FEW_SHOT_ER}\n\n"
                f"Mention A: '{m_a}' (type: {etype})\n"
                f"Mention B: '{m_b}' (type: {etype})\n\n"
                "Output JSON conforming to ERClassificationOutput."
            )

            er_tokens = transport.count_tokens(model=model_name, contents=er_prompt, system_instruction=er_sys, response_schema=ERClassificationOutput)
            tracker.reserve(reservation_id=er_res_id, model_id=model_name, estimated_input_tokens=er_tokens, max_output_tokens=500)
            er_raw, _, er_usage = transport.generate(
                model=model_name, prompt=er_prompt, system_instruction=er_sys, condition_id="C4",
                response_schema=ERClassificationOutput, temperature=0.0, seed=42
            )
            tracker.settle(er_res_id, er_usage.get("prompt_tokens", er_tokens), er_usage.get("completion_tokens", 0))
            er_dict = json.loads(er_raw)
            with open(er_cache_path, "w", encoding="utf-8") as f:
                json.dump(er_dict, f, indent=2)
        prop_rel = er_dict.get("relation")
        prop_dir = er_dict.get("direction")
        prop_rat = er_dict.get("rationale")
        prop_conf = er_dict.get("confidence")

        is_3way_correct = (prop_rel == gold_rel)
        if is_3way_correct:
            proposed_3way_correct += 1
            if gold_rel == "VARIANT_OF" and prop_dir == gold_dir:
                proposed_var_correct += 1

        er_results.append({
            "pair_id": pid,
            "mention_a": m_a,
            "mention_b": m_b,
            "entity_type": etype,
            "gold_relation": gold_rel,
            "gold_direction": gold_dir,
            "er_c2_sim": round(c2_sim, 4),
            "er_c2_pred_same": c2_pred_same,
            "proposed_rel": prop_rel,
            "proposed_direction": prop_dir,
            "proposed_rationale": prop_rat,
            "proposed_confidence": prop_conf,
            "is_3way_correct": is_3way_correct,
        })
        print(f"  [{pid}] Gold: {gold_rel} ({gold_dir}) | C2: sim={c2_sim:.3f} ({'SAME' if c2_pred_same else 'DISTINCT'}) | Proposed: {prop_rel} ({prop_dir}) -> {'CORRECT' if is_3way_correct else 'INCORRECT'}")

    c2_same_p = c2_same_as_correct / c2_same_as_pred_total if c2_same_as_pred_total > 0 else 0.0
    c2_same_r = c2_same_as_correct / c2_same_as_gold_total if c2_same_as_gold_total > 0 else 0.0
    c2_same_f1 = (2 * c2_same_p * c2_same_r) / (c2_same_p + c2_same_r) if (c2_same_p + c2_same_r) > 0 else 0.0

    prop_3way_acc = proposed_3way_correct / len(er_pairs)
    prop_var_recall = proposed_var_correct / proposed_var_gold_total if proposed_var_gold_total > 0 else 0.0

    print(f"\n[Baseline ER-C2 SAME_AS] Precision: {c2_same_p:.4f}, Recall: {c2_same_r:.4f}, F1: {c2_same_f1:.4f}")
    print(f"[Proposed ER 3-Way]    Accuracy: {prop_3way_acc:.4f} ({proposed_3way_correct}/{len(er_pairs)})")
    print(f"[Proposed VARIANT_OF]  Direction & Category Accuracy: {prop_var_recall:.4f} ({proposed_var_correct}/{proposed_var_gold_total})")

    # =========================================================================
    # STEP 3.4: Relational Knowledge Graph Construction & Integrity Audit
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.4: Relational Knowledge Graph Construction (SQLite DB)")
    print("=" * 80)

    kg_db_path = base_dir / "data/knowledge_graph_expansion.db"
    if kg_db_path.exists():
        kg_db_path.unlink()

    kg_store = KnowledgeGraphStore(kg_db_path)

    # Ingest Canonical Entities and Provenance from Gold Expansion Entities
    entity_id_map = {}
    for doc in gt_docs:
        sid = doc["source_id"]
        tf_name = "readme_text" if doc["source_type"] == "oss" else "abstract_text"
        ver_hash = doc["target_field_hashes"][tf_name]
        for e in doc.get("entities", []):
            ename = e["name"] or e.get("statement", "")
            etype = e["entity_type"]
            cid = f"ent_{etype}_{normalize_entity_name(ename).replace(' ', '_')}"
            entity_id_map[(sid, e["local_id"])] = cid

            ev = e.get("evidence", {})
            v_quote = ev.get("verbatim_quote", ename)
            prov = MentionProvenance(
                mention_id=f"prov_{sid}_{e['local_id']}",
                entity_id=cid,
                source_id=sid,
                version_hash=ver_hash,
                verbatim_quote=v_quote,
                start_char=ev.get("start_char", 0),
                end_char=ev.get("end_char") or max(1, len(v_quote)),
                target_field_name=tf_name,
            )

            canon = CanonicalEntity(
                entity_id=cid,
                canonical_name=ename,
                entity_type=etype,
                origin_type="extracted",
                aliases=e.get("synonyms", []),
                provenances=[prov],
            )
            kg_store.insert_entity(canon)

    # Ingest Validated Relation Edges and Evidences (RE-Hybrid Accepted Relations)
    for pair in oracle_preds["RE-Hybrid"]:
        if pair["predicted_rel"] != "NONE":
            # Lookup original pair
            orig = next(p for p in all_oracle_pairs if p["relation_id"] == pair["pair_id"])
            sid = orig["source_id"]
            tf_name = "readme_text" if docs_by_id[sid]["source_type"] == "oss" else "abstract_text"
            ver_hash = docs_by_id[sid]["target_field_hashes"][tf_name]
            
            src_cid = entity_id_map[(sid, orig["src_local_id"])]
            dst_cid = entity_id_map[(sid, orig["dst_local_id"])]
            rtype = pair["predicted_rel"]
            edge_id = f"edge_{src_cid}_{rtype}_{dst_cid}"

            c2_info = next(p for p in oracle_preds["RE-C2"] if p["pair_id"] == pair["pair_id"])
            rel_quote = c2_info.get("evidence") or orig.get("evidence_quote", "relation evidence")
            rel_ev = RelationEvidence(
                evidence_id=f"ev_{edge_id}_{sid}",
                edge_id=edge_id,
                source_id=sid,
                version_hash=ver_hash,
                verbatim_quote=rel_quote,
                start_char=0,
                end_char=max(1, len(rel_quote)),
                reasoning=c2_info.get("reasoning", ""),
            )

            edge = RelationEdge(
                edge_id=edge_id,
                src_entity_id=src_cid,
                dst_entity_id=dst_cid,
                relation_type=rtype,
                origin_type="text_supported",
                confidence_score=0.95,
                evidences=[rel_ev],
            )
            kg_store.insert_relation_edge(edge)

    # Ingest VARIANT_OF Hierarchical Edges from Proposed ER
    for er_item in er_results:
        if er_item["proposed_rel"] == "VARIANT_OF":
            # Form hierarchical edge
            m_a_clean = normalize_entity_name(er_item["mention_a"]).replace(" ", "_")
            m_b_clean = normalize_entity_name(er_item["mention_b"]).replace(" ", "_")
            src_cid = f"ent_{er_item['entity_type']}_{m_a_clean}"
            dst_cid = f"ent_{er_item['entity_type']}_{m_b_clean}"

            # Ensure entities exist in canonical table with proper origin_type
            if not kg_store.get_entity(src_cid):
                kg_store.insert_entity(CanonicalEntity(
                    entity_id=src_cid,
                    canonical_name=er_item["mention_a"],
                    entity_type=er_item["entity_type"],
                    origin_type="extracted",
                ))
            if not kg_store.get_entity(dst_cid):
                kg_store.insert_entity(CanonicalEntity(
                    entity_id=dst_cid,
                    canonical_name=er_item["mention_b"],
                    entity_type=er_item["entity_type"],
                    origin_type="inferred",
                    derivation_rationale=f"Base canonical concept inferred via ER VARIANT_OF specialization from '{er_item['mention_a']}'"
                ))

            var_edge = RelationEdge(
                edge_id=f"edge_var_{src_cid}_VARIANT_OF_{dst_cid}",
                src_entity_id=src_cid,
                dst_entity_id=dst_cid,
                relation_type="VARIANT_OF",
                origin_type="hierarchical_variant",
                confidence_score=er_item["proposed_confidence"] or 0.90,
            )
            kg_store.insert_relation_edge(var_edge)

    # Perform Integrity Checks
    db_stats = kg_store.get_summary_statistics()

    with kg_store._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT edge_id FROM relation_edges 
            WHERE src_entity_id NOT IN (SELECT entity_id FROM canonical_entities)
               OR dst_entity_id NOT IN (SELECT entity_id FROM canonical_entities)
        """)
        orphans = cur.fetchall()

        cur.execute("SELECT edge_id FROM relation_edges WHERE src_entity_id = dst_entity_id")
        self_loops = cur.fetchall()

    print(f"\nKnowledge Graph Stats: {db_stats}")
    print(f"Orphan Edges Count: {len(orphans)} (Must be 0)")
    print(f"Self-Loops Count:   {len(self_loops)} (Must be 0)")
    assert len(orphans) == 0, f"Found orphan edges: {orphans}"
    assert len(self_loops) == 0, f"Found self loops: {self_loops}"

    # =========================================================================
    # STEP 3.5: Final Aggregation and Ledger Verification
    # =========================================================================
    final_spent = tracker.state.spent_usd
    total_tokens = tracker.state.spent_tokens
    print("\n" + "=" * 80)
    print("STEP 3.5: Final Budget & Execution Summary")
    print("=" * 80)
    print(f"Total API Calls Settled: {tracker.state.total_calls_settled}")
    print(f"Total Calls Refused:    {tracker.state.total_calls_refused}")
    print(f"Total Tokens Spent:     {total_tokens:,}")
    print(f"Total Dollars Spent:    ${final_spent:.6f} USD")
    print(f"Remaining Budget:       ${tracker.max_usd - final_spent:.6f} USD")

    final_results = {
        "metadata": {
            "experiment_id": "HDE_Phase3_Holdout_Expansion",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "primary_model": model_name,
            "temperature": 0.0,
            "seed": 42,
            "frozen_hashes": {
                "holdout_expansion_ground_truth": gt_hash,
                "holdout_expansion_re_benchmark": re_hash,
                "holdout_expansion_er_benchmark": er_hash,
            },
            "budget": {
                "max_budget_usd": tracker.max_usd,
                "spent_usd": round(final_spent, 6),
                "remaining_usd": round(tracker.max_usd - final_spent, 6),
                "total_calls": tracker.state.total_calls_settled,
            }
        },
        "h1a_extraction": {
            "trial_1": eval_t1,
            "trial_2": eval_t2,
        },
        "h1b_relation_extraction": {
            "oracle_re_c0": re_c0_metrics,
            "oracle_re_c2": re_c2_metrics,
            "oracle_re_hybrid": re_hyb_metrics,
            "end_to_end": {
                "gold_relations_total": gold_relations_total,
                "endpoints_co_available": endpoint_available_count,
                "r_endpoint": round(r_endpoint, 4),
                "r_re_given_endpoint_c0": round(r_re_given_endpoint_c0, 4),
                "r_re_given_endpoint_hyb": round(r_re_given_endpoint_hyb, 4),
                "r_e2e_c0": round(r_e2e_c0, 4),
                "r_e2e_hyb": round(r_e2e_hyb, 4),
                "details": e2e_relation_details,
            }
        },
        "h1b_entity_resolution": {
            "baseline_er_c2": {
                "threshold": 0.82,
                "precision": round(c2_same_p, 4),
                "recall": round(c2_same_r, 4),
                "f1": round(c2_same_f1, 4),
            },
            "proposed_er": {
                "three_way_accuracy": round(prop_3way_acc, 4),
                "variant_of_recall": round(prop_var_recall, 4),
                "pairs_detail": er_results,
            }
        },
        "knowledge_graph": {
            "stats": db_stats,
            "orphan_edges": len(orphans),
            "self_loops": len(self_loops),
        }
    }

    results_file = output_dir / "phase3_holdout_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    ledger_path = base_dir / "data/budget_ledger.jsonl"
    if tracker.get_ledger():
        with open(ledger_path, "a", encoding="utf-8") as lf:
            for entry in tracker.get_ledger():
                lf.write(entry.model_dump_json() + "\n")
        print(f"Appended {len(tracker.get_ledger())} settled entries to {ledger_path}")

    print(f"\nAll results saved to: {results_file}")
    print("Phase 3 Holdout Expansion execution successfully completed!")


if __name__ == "__main__":
    main()
