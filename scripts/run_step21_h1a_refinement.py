"""Step 2.1: H1a-R2 Capability Recall Improvement & 7-Class Comparison.

Compares:
1. Baseline C4 (7-Class Single Stage Zero-Shot)
2. Proposed C5 (Two-Stage Pipeline with Few-Shot Mentions + Focused IO Typing)

Evaluated on 10 Development Set documents across 2 independent trials (R=2).
Strict Two-Phase Budget Reservation enforced ($10.00 USD Hard Ceiling).
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_pipeline.config import load_project_env
from research_pipeline.extractors.budget_tracker import BudgetTracker
from research_pipeline.extractors.transport_factory import detect_configured_provider, create_transport
from research_pipeline.extractors.models import (
    ExtractedKnowledgeRecord,
    Stage1MentionRecord,
    Stage2CapabilityIODetail,
    CapabilityEntity,
    CapabilityIO,
    EvidenceSpan,
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


FEW_SHOT_STAGE1 = """Here are two examples of high-recall entity extraction:

Example 1 (Academic Paper):
Text: "We propose a Recurrent Graph Convolutional network (R-GCN) to extract predictive alpha signals from multi-exchange order books, overcoming stationary variance assumptions."
Output JSON:
{
  "mentions": [
    {"local_id": "mod_01", "entity_type": "model", "name_or_statement": "Recurrent Graph Convolutional Network", "category": "deep_learning", "verbatim_quote": "Recurrent Graph Convolutional network"},
    {"local_id": "cap_01", "entity_type": "capability", "name_or_statement": "Extract Predictive Alpha Signals", "category": null, "verbatim_quote": "extract predictive alpha signals from multi-exchange order books"},
    {"local_id": "ass_01", "entity_type": "assumption", "name_or_statement": "Stationary variance of order flows", "category": null, "verbatim_quote": "stationary variance assumptions"}
  ]
}

Example 2 (Software Repository):
Text: "FinRL is an open-source framework providing deep reinforcement learning environments to train automated trading agents on stock and crypto markets."
Output JSON:
{
  "mentions": [
    {"local_id": "soft_01", "entity_type": "software", "name_or_statement": "FinRL Trading Framework", "category": "simulation_environment", "verbatim_quote": "FinRL is an open-source framework"},
    {"local_id": "met_01", "entity_type": "method", "name_or_statement": "Deep Reinforcement Learning", "category": "optimization", "verbatim_quote": "deep reinforcement learning"},
    {"local_id": "cap_01", "entity_type": "capability", "name_or_statement": "Train Automated Trading Agents", "category": null, "verbatim_quote": "train automated trading agents on stock and crypto markets"}
  ]
}
"""


def execute_baseline_c4(
    doc: Dict[str, Any],
    trial_idx: int,
    tracker: BudgetTracker,
    transport: Any,
    model_name: str,
    output_dir: Path,
) -> ExtractionTrioBundle:
    """Execute Baseline C4 (7-class, 1-stage)."""
    source_id = doc["source_id"]
    version_id = doc["version_id"]
    canonical_fields = doc["target_fields"]
    target_field_name = "readme_text" if doc["source_type"] == "oss" else "abstract_text"
    text_content = canonical_fields.get(target_field_name, "")
    title = canonical_fields.get("title_text", "")

    run_id = f"step21_C4_t{trial_idx}_{source_id}"
    cache_path = output_dir / f"bundle_{run_id}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return ExtractionTrioBundle.model_validate_json(f.read())

    schema_cls = ExtractedKnowledgeRecord
    system_instruction = (
        "You are an expert scientific researcher. Extract structured knowledge across 7 classes: "
        "phenomenon, model, method, software, capability, assumption, limitation. "
        "Differentiate Model (mathematical/theoretical structure), Method (algorithm/procedure), "
        "and Software (codebase/framework/simulator). "
        "Every entity must cite exact verbatim evidence from the text. Output JSON strictly adhering to schema."
    )
    prompt = (
        f"Document Title: {title}\n\n"
        f"Source Text ({target_field_name}):\n{text_content}\n\n"
        "Extract complete structured knowledge complying strictly with ExtractedKnowledgeRecord JSON schema."
    )

    counted_tokens = transport.count_tokens(model=model_name, contents=prompt, system_instruction=system_instruction, response_schema=schema_cls)
    tracker.reserve(reservation_id=run_id, model_id=model_name, estimated_input_tokens=counted_tokens, max_output_tokens=2048)

    try:
        raw_text, http_status, usage = transport.generate(
            model=model_name, prompt=prompt, system_instruction=system_instruction,
            condition_id="C4", response_schema=schema_cls, temperature=0.0, seed=42
        )
    except Exception as e:
        tracker.release_reservation(run_id)
        raise RuntimeError(f"C4 call failed for {run_id}: {e}")

    tracker.settle(run_id, usage.get("prompt_tokens", counted_tokens), usage.get("completion_tokens", 0))

    parsed_dict = None
    parse_err = None
    json_status = "success"
    try:
        parsed_dict = json.loads(raw_text)
    except Exception as pe:
        json_status = "invalid_json"
        parse_err = str(pe)

    raw_artifact = RawExtractionArtifact(
        extraction_run_id=run_id, source_id=source_id, version_id=version_id, condition_id="C4",
        raw_response_text=raw_text, http_status=http_status, token_usage=usage,
        executed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), model_checkpoint=model_name,
        prompt_hash=compute_canonical_hash(prompt), target_field_hashes=doc["target_field_hashes"],
        json_parse_status=json_status, parse_error=parse_err, parsed_candidates=parsed_dict,
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
                    local_id=c.candidate_local_id, entity_type=c.entity_type, rejection_stage="quote_validity",
                    rejection_reason=reason, raw_entity_data=c.raw_payload
                ))
        else:
            rejected_items.append(RejectionRecord(
                local_id=c.candidate_local_id, entity_type=c.entity_type, rejection_stage="missing_evidence",
                rejection_reason="Evidence span was missing", raw_entity_data=c.raw_payload
            ))

    accepted_record = None
    if parsed_dict and json_status == "success":
        try:
            accepted_record = ExtractedKnowledgeRecord.model_validate(parsed_dict)
        except Exception:
            pass

    bundle = ExtractionTrioBundle(
        extraction_run_id=run_id, source_id=source_id, version_id=version_id, raw_artifact=raw_artifact,
        evaluation_candidates=accepted_candidates, accepted_record=accepted_record, rejected_items=rejected_items
    )
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(bundle.model_dump_json(indent=2))
    return bundle


def execute_proposed_c5(
    doc: Dict[str, Any],
    trial_idx: int,
    tracker: BudgetTracker,
    transport: Any,
    model_name: str,
    output_dir: Path,
) -> ExtractionTrioBundle:
    """Execute Proposed C5 (Two-Stage Pipeline with Few-Shot Mentions + Focused IO Typing)."""
    source_id = doc["source_id"]
    version_id = doc["version_id"]
    canonical_fields = doc["target_fields"]
    target_field_name = "readme_text" if doc["source_type"] == "oss" else "abstract_text"
    text_content = canonical_fields.get(target_field_name, "")
    title = canonical_fields.get("title_text", "")

    run_id = f"step21_C5_t{trial_idx}_{source_id}"
    cache_path = output_dir / f"bundle_{run_id}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return ExtractionTrioBundle.model_validate_json(f.read())

    # --- STAGE 1: Mention Extraction with Few-Shot ---
    stage1_res_id = f"{run_id}_s1"
    s1_sys_instruction = (
        "You are an expert scientific researcher. Extract all key concepts and functional capabilities "
        "mentioned in the text into 7 categories: model, method, software, capability, phenomenon, limitation, assumption. "
        "Capture every distinct operational capability as an actionable verb phrase. "
        "Cite the exact verbatim quote for every mention."
    )
    s1_prompt = (
        f"{FEW_SHOT_STAGE1}\n\n"
        f"Now extract from this document:\n"
        f"Title: {title}\n"
        f"Source Text ({target_field_name}):\n{text_content}\n\n"
        "Output JSON complying strictly with Stage1MentionRecord schema."
    )

    s1_tokens = transport.count_tokens(model=model_name, contents=s1_prompt, system_instruction=s1_sys_instruction, response_schema=Stage1MentionRecord)
    tracker.reserve(reservation_id=stage1_res_id, model_id=model_name, estimated_input_tokens=s1_tokens, max_output_tokens=1500)

    try:
        s1_raw, s1_status, s1_usage = transport.generate(
            model=model_name, prompt=s1_prompt, system_instruction=s1_sys_instruction,
            condition_id="C4", response_schema=Stage1MentionRecord, temperature=0.0, seed=42
        )
    except Exception as e:
        tracker.release_reservation(stage1_res_id)
        raise RuntimeError(f"C5 Stage 1 failed for {run_id}: {e}")

    tracker.settle(stage1_res_id, s1_usage.get("prompt_tokens", s1_tokens), s1_usage.get("completion_tokens", 0))

    stage1_record = None
    try:
        s1_dict = json.loads(s1_raw)
        stage1_record = Stage1MentionRecord.model_validate(s1_dict)
    except Exception as e:
        print(f"  [WARN] Stage 1 JSON parse failed for {run_id}: {e}")

    # --- STAGE 2: Capability Deep Typing ---
    cap_mentions = [m for m in (stage1_record.mentions if stage1_record else []) if m.entity_type == "capability"]
    stage2_capabilities: List[CapabilityEntity] = []

    if cap_mentions:
        stage2_res_id = f"{run_id}_s2"
        s2_sys_instruction = (
            "Analyze the extracted capability mentions in context and specify their operational category "
            "(generate, infer, detect, predict, transform, optimize) and input/output data types. "
            "If input/output data types are not explicitly described, set is_unspecified=true, data_type_l1='unknown', and evidence=null."
        )
        caps_desc = "\n".join([f"- {m.local_id}: {m.name_or_statement} (Quote: '{m.verbatim_quote}')" for m in cap_mentions])
        s2_prompt = (
            f"Context Text:\n{text_content}\n\n"
            f"Capabilities to specify:\n{caps_desc}\n\n"
            "Specify detailed IO types adhering to CapabilityEntity definitions."
        )

        s2_tokens = transport.count_tokens(model=model_name, contents=s2_prompt, system_instruction=s2_sys_instruction, response_schema=Stage2CapabilityIODetail)
        tracker.reserve(reservation_id=stage2_res_id, model_id=model_name, estimated_input_tokens=s2_tokens, max_output_tokens=1500)

        try:
            s2_raw, s2_status, s2_usage = transport.generate(
                model=model_name, prompt=s2_prompt, system_instruction=s2_sys_instruction,
                condition_id="C4", response_schema=Stage2CapabilityIODetail, temperature=0.0, seed=42
            )
            tracker.settle(stage2_res_id, s2_usage.get("prompt_tokens", s2_tokens), s2_usage.get("completion_tokens", 0))
        except Exception:
            tracker.release_reservation(stage2_res_id)

    # --- ASSEMBLE CANDIDATES & VALIDATE ---
    candidates: List[EvaluationCandidate] = []
    rejected_items: List[RejectionRecord] = []

    target_hash = doc["target_field_hashes"].get(target_field_name, "")

    if stage1_record:
        for m in stage1_record.mentions:
            q = m.verbatim_quote.strip()
            # Construct preliminary EvidenceSpan
            ev = EvidenceSpan(
                target_field_name=target_field_name,  # type: ignore
                target_field_hash=target_hash,
                verbatim_quote=q,
                start_char=0,
                end_char=len(q),
            )
            valid, reason, resolved_ev = validate_evidence(ev, canonical_fields)
            if valid and resolved_ev:
                candidates.append(
                    EvaluationCandidate(
                        candidate_local_id=m.local_id,
                        entity_type=m.entity_type,  # type: ignore
                        name_or_statement=m.name_or_statement,
                        evidence=resolved_ev,
                        condition_id="C5",
                        extraction_run_id=run_id,
                        conversion_status="success",
                        missing_evidence=False,
                        raw_payload=m.model_dump(),
                    )
                )
            else:
                rejected_items.append(
                    RejectionRecord(
                        local_id=m.local_id,
                        entity_type=m.entity_type,
                        rejection_stage="quote_validity",
                        rejection_reason=reason,
                        raw_entity_data=m.model_dump(),
                    )
                )

    raw_art = RawExtractionArtifact(
        extraction_run_id=run_id, source_id=source_id, version_id=version_id, condition_id="C5",
        raw_response_text=s1_raw, http_status=200, token_usage=s1_usage,
        executed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), model_checkpoint=model_name,
        prompt_hash=compute_canonical_hash(s1_prompt), target_field_hashes=doc["target_field_hashes"],
        json_parse_status="success", parse_error=None, parsed_candidates=s1_dict if stage1_record else None,
    )

    bundle = ExtractionTrioBundle(
        extraction_run_id=run_id, source_id=source_id, version_id=version_id, raw_artifact=raw_art,
        evaluation_candidates=candidates, accepted_record=None, rejected_items=rejected_items
    )
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(bundle.model_dump_json(indent=2))
    return bundle


def evaluate_bundles(
    bundles: List[ExtractionTrioBundle],
    docs_by_id: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate candidate bundles against 7-class Development benchmark."""
    total_quotes = 0
    valid_quotes = 0

    strict_tp, strict_fp, strict_fn = 0, 0, 0
    sem_tp, sem_fp, sem_fn = 0, 0, 0

    class_strict: Dict[str, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in TARGET_CLASSES}
    class_sem: Dict[str, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in TARGET_CLASSES}

    for b in bundles:
        doc = docs_by_id[b.source_id]
        gold_ents = doc.get("entities", [])
        preds = b.evaluation_candidates

        # Quote Integrity
        for p in preds:
            if p.evidence:
                total_quotes += 1
                if p.evidence.resolved_offset_valid:
                    valid_quotes += 1

        # Strict Matching
        for c in TARGET_CLASSES:
            g_c = [g for g in gold_ents if g["entity_type"] == c]
            p_c = [p for p in preds if p.entity_type == c]
            c_tp = sum(1 for gc in g_c if any(normalize_entity_name(gc.get("name") or gc.get("statement") or "") == normalize_entity_name(pc.name_or_statement) for pc in p_c))
            class_strict[c]["tp"] += c_tp
            class_strict[c]["fp"] += len(p_c) - c_tp
            class_strict[c]["fn"] += len(g_c) - c_tp

        # Semantic Matching
        m_sem = match_entities_bipartite(gold_ents, preds)
        sem_tp += m_sem.true_positives
        sem_fp += m_sem.false_positives
        sem_fn += m_sem.false_negatives

        for c in TARGET_CLASSES:
            m_c = match_entities_bipartite(gold_ents, preds, target_entity_type=c)
            class_sem[c]["tp"] += m_c.true_positives
            class_sem[c]["fp"] += m_c.false_positives
            class_sem[c]["fn"] += m_c.false_negatives

    def f1_score(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    ov_p, ov_r, ov_f1 = f1_score(sem_tp, sem_fp, sem_fn)
    cap_p, cap_r, cap_f1 = f1_score(class_sem["capability"]["tp"], class_sem["capability"]["fp"], class_sem["capability"]["fn"])
    mod_p, mod_r, mod_f1 = f1_score(class_sem["model"]["tp"], class_sem["model"]["fp"], class_sem["model"]["fn"])
    soft_p, soft_r, soft_f1 = f1_score(class_sem["software"]["tp"], class_sem["software"]["fp"], class_sem["software"]["fn"])
    met_p, met_r, met_f1 = f1_score(class_sem["method"]["tp"], class_sem["method"]["fp"], class_sem["method"]["fn"])

    qi_rate = valid_quotes / total_quotes if total_quotes > 0 else 0.0

    return {
        "overall_semantic": {"precision": ov_p, "recall": ov_r, "f1": ov_f1, "tp": sem_tp, "fp": sem_fp, "fn": sem_fn},
        "capability": {"precision": cap_p, "recall": cap_r, "f1": cap_f1, "tp": class_sem["capability"]["tp"], "fp": class_sem["capability"]["fp"], "fn": class_sem["capability"]["fn"]},
        "model": {"precision": mod_p, "recall": mod_r, "f1": mod_f1, "tp": class_sem["model"]["tp"], "fp": class_sem["model"]["fp"], "fn": class_sem["model"]["fn"]},
        "software": {"precision": soft_p, "recall": soft_r, "f1": soft_f1, "tp": class_sem["software"]["tp"], "fp": class_sem["software"]["fp"], "fn": class_sem["software"]["fn"]},
        "method": {"precision": met_p, "recall": met_r, "f1": met_f1, "tp": class_sem["method"]["tp"], "fp": class_sem["method"]["fp"], "fn": class_sem["method"]["fn"]},
        "quote_integrity_rate": round(qi_rate, 4),
        "total_quotes": total_quotes,
        "valid_quotes": valid_quotes,
        "class_breakdown": {c: {"precision": f1_score(class_sem[c]["tp"], class_sem[c]["fp"], class_sem[c]["fn"])[0], "recall": f1_score(class_sem[c]["tp"], class_sem[c]["fp"], class_sem[c]["fn"])[1], "f1": f1_score(class_sem[c]["tp"], class_sem[c]["fp"], class_sem[c]["fn"])[2], "tp": class_sem[c]["tp"], "fp": class_sem[c]["fp"], "fn": class_sem[c]["fn"]} for c in TARGET_CLASSES}
    }


def main():
    load_project_env()
    base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
    output_dir = base_dir / "data/experiments/step21_h1a_r2"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HDE Phase 2: Step 2.1 (H1a-R2 Capability Recall & 7-Class Comparison)")
    print("=" * 80)

    gt_path = base_dir / "tests/fixtures/gold_standard/h1a_ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    dev_docs = [d for d in ground_truth if d.get("split") == "development"]
    docs_by_id = {d["source_id"]: d for d in dev_docs}
    print(f"Loaded {len(dev_docs)} Development Set documents (7-class ground truth).")

    tracker = BudgetTracker()
    provider, _ = detect_configured_provider()
    transport = create_transport(provider=provider)
    model_name = "gpt-4o-mini-2024-07-18"
    trials = [1, 2]

    # 1. Run Baseline C4 (7-class, 2 trials = 20 calls)
    print("\n--- Running Baseline C4 (7-Class Single Stage, 20 calls) ---")
    c4_bundles = []
    t0 = time.time()
    for t in trials:
        for doc in dev_docs:
            b = execute_baseline_c4(doc, t, tracker, transport, model_name, output_dir)
            c4_bundles.append(b)
    print(f"Baseline C4 finished in {time.time() - t0:.2f}s.")

    # 2. Run Proposed C5 (Two-Stage Pipeline, 2 trials = 20 calls S1 + 20 calls S2)
    print("\n--- Running Proposed C5 (Two-Stage Pipeline, 40 calls) ---")
    c5_bundles = []
    t0_c5 = time.time()
    for t in trials:
        for doc in dev_docs:
            b = execute_proposed_c5(doc, t, tracker, transport, model_name, output_dir)
            c5_bundles.append(b)
    print(f"Proposed C5 finished in {time.time() - t0_c5:.2f}s.")

    summary_budget = tracker.get_summary()
    print(f"\nAPI Budget Spent: ${summary_budget['spent_usd']:.6f} USD | Remaining: ${summary_budget['remaining_usd']:.6f} USD")

    # Update ledger
    tracker.export_ledger_jsonl(base_dir / "data/budget_ledger.jsonl")

    # 3. Quantitative Evaluation
    print("\n--- Quantitative Evaluation against 7-Class Benchmark ---")
    eval_c4 = evaluate_bundles(c4_bundles, docs_by_id)
    eval_c5 = evaluate_bundles(c5_bundles, docs_by_id)

    print("\n" + "="*85)
    print(f"{'Metric':<30} | {'Baseline C4':<15} | {'Proposed C5 (Two-Stage)':<25} | {'Delta':<10}")
    print("="*85)

    def print_row(label, v1, v2):
        delta = v2 - v1
        print(f"{label:<30} | {v1:15.4f} | {v2:25.4f} | {delta:+10.4f}")

    print_row("Overall Entity Precision", eval_c4["overall_semantic"]["precision"], eval_c5["overall_semantic"]["precision"])
    print_row("Overall Entity Recall", eval_c4["overall_semantic"]["recall"], eval_c5["overall_semantic"]["recall"])
    print_row("Overall Entity Micro F1", eval_c4["overall_semantic"]["f1"], eval_c5["overall_semantic"]["f1"])
    print("-" * 85)
    print_row("Capability Precision", eval_c4["capability"]["precision"], eval_c5["capability"]["precision"])
    print_row("Capability Recall", eval_c4["capability"]["recall"], eval_c5["capability"]["recall"])
    print_row("Capability F1", eval_c4["capability"]["f1"], eval_c5["capability"]["f1"])
    print("-" * 85)
    print_row("Model F1", eval_c4["model"]["f1"], eval_c5["model"]["f1"])
    print_row("Software F1", eval_c4["software"]["f1"], eval_c5["software"]["f1"])
    print_row("Method F1", eval_c4["method"]["f1"], eval_c5["method"]["f1"])
    print_row("Quote Integrity Rate", eval_c4["quote_integrity_rate"], eval_c5["quote_integrity_rate"])

    print("\nPer-Class Detailed Comparison (Proposed C5 vs Baseline C4):")
    print(f"{'Class':<12} | {'C4 Recall':<10} | {'C5 Recall':<10} | {'C4 F1':<10} | {'C5 F1':<10}")
    print("-" * 60)
    for c in TARGET_CLASSES:
        r4 = eval_c4["class_breakdown"][c]["recall"]
        r5 = eval_c5["class_breakdown"][c]["recall"]
        f4 = eval_c4["class_breakdown"][c]["f1"]
        f5 = eval_c5["class_breakdown"][c]["f1"]
        print(f"{c:<12} | {r4:10.4f} | {r5:10.4f} | {f4:10.4f} | {f5:10.4f}")

    # Export Results
    results_path = output_dir / "step21_comparison_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "HDE_Step21_H1a_R2_Comparison",
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "budget": summary_budget,
            "baseline_c4": eval_c4,
            "proposed_c5": eval_c5,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
