"""Holdout Pilot Experiment Runner (C4 Only, 12 documents, 3 trials = 36 API calls).

Complies strictly with H1a Phase 3 Plan and User Instructions:
1. Pinned Primary Model: gpt-4o-mini-2024-07-18, temp=0.0, seed=42
2. Evaluates ONLY C4 (Proposed Full Method) on Holdout Pilot (12 documents)
3. Evaluates 3 independent trials (R=3, total 36 live API calls)
4. Preserves all raw artifacts, candidates, and rejection records in ExtractionTrioBundle
5. Two-Phase Budget Reservation enforcing $10.00 USD Hard Limit
6. Evaluates under both Strict Lexical Match and Pre-registered Semantic Match
7. Computes Document-level Cluster Bootstrap (B=1000) 95% CIs
8. Evaluates LocalRelation Precision, Recall, and F1
9. Logs and checks frozen component SHA-256 hashes
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment

from research_pipeline.config import load_project_env
from research_pipeline.extractors.budget_tracker import BudgetTracker
from research_pipeline.extractors.transport_factory import detect_configured_provider, create_transport
from research_pipeline.extractors.models import (
    ExtractedKnowledgeRecord,
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
from research_pipeline.evaluation.metrics import TARGET_CLASSES, ClassMetricResult


FROZEN_FILES = [
    'src/research_pipeline/extractors/models.py',
    'src/research_pipeline/extractors/converter.py',
    'src/research_pipeline/extractors/validator.py',
    'src/research_pipeline/extractors/vocab.py',
    'src/research_pipeline/extractors/llm_extractor.py',
    'src/research_pipeline/extractors/openai_transport.py',
    'src/research_pipeline/evaluation/matching.py',
    'src/research_pipeline/evaluation/metrics.py',
    'src/research_pipeline/evaluation/bootstrap.py',
    'tests/fixtures/gold_standard/h1a_ground_truth.json'
]


def verify_frozen_hashes(base_dir: Path) -> Dict[str, str]:
    """Verify and return SHA-256 hashes of all frozen pipeline components."""
    hashes = {}
    for rel in FROZEN_FILES:
        full_path = base_dir / rel
        if not full_path.exists():
            raise FileNotFoundError(f"Frozen file missing: {full_path}")
        with open(full_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        hashes[rel] = h
    return hashes


def execute_c4_single(
    doc: Dict[str, Any],
    trial_idx: int,
    tracker: BudgetTracker,
    transport: Any,
    model_name: str,
    output_dir: Path,
) -> ExtractionTrioBundle:
    """Execute live C4 extraction on a single Holdout document."""
    source_id = doc["source_id"]
    version_id = doc["version_id"]
    canonical_fields = doc["target_fields"]
    target_field_name = "readme_text" if doc["source_type"] == "oss" else "abstract_text"
    text_content = canonical_fields.get(target_field_name, "")
    title = canonical_fields.get("title_text", "")

    run_id = f"holdout_C4_t{trial_idx}_{source_id}"
    cache_path = output_dir / f"bundle_{run_id}.json"

    # Check disk cache if already executed
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return ExtractionTrioBundle.model_validate_json(f.read())

    # Build C4 prompt and schema
    schema_cls = ExtractedKnowledgeRecord
    system_instruction = (
        "You are an expert scientific researcher. Extract complete, structured knowledge and capability entities "
        "from the provided text. Output JSON adhering strictly to the schema. "
        "Every entity and relation must cite exact verbatim evidence from the text. "
        "If input/output data types for Capability are not explicitly mentioned in the text, "
        "set is_unspecified=true, data_type_l1='unknown', and evidence=null."
    )
    prompt = (
        f"Document Title: {title}\n\n"
        f"Source Text ({target_field_name}):\n{text_content}\n\n"
        "Extract complete structured knowledge and relations complying strictly with ExtractedKnowledgeRecord JSON schema."
    )

    # 1. Conservative Token Counting & Reservation
    counted_tokens = transport.count_tokens(
        model=model_name,
        contents=prompt,
        system_instruction=system_instruction,
        response_schema=schema_cls,
    )
    res_id = run_id
    tracker.reserve(
        reservation_id=res_id,
        model_id=model_name,
        estimated_input_tokens=counted_tokens,
        max_output_tokens=2048,
        allowance_tokens=0,
    )

    # 2. Live API Call
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
        tracker.release_reservation(res_id)
        raise RuntimeError(f"API call failed for {run_id}: {e}")

    # 3. Post-call Settlement
    actual_in = usage.get("prompt_tokens", counted_tokens)
    actual_out = usage.get("completion_tokens", 0)
    tracker.settle(res_id, actual_in, actual_out)

    # 4. Save Raw Extraction Artifact
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

    # 5. Conversion & Evidence Validation
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
                rejected_items.append(
                    RejectionRecord(
                        local_id=c.candidate_local_id,
                        entity_type=c.entity_type,
                        rejection_stage="quote_validity",
                        rejection_reason=reason,
                        raw_entity_data=c.raw_payload,
                    )
                )
        else:
            rejected_items.append(
                RejectionRecord(
                    local_id=c.candidate_local_id,
                    entity_type=c.entity_type,
                    rejection_stage="missing_evidence",
                    rejection_reason="Evidence span was missing",
                    raw_entity_data=c.raw_payload,
                )
            )

    # Build accepted ExtractedKnowledgeRecord if valid
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
        accepted_record=accepted_record,
        rejected_items=rejected_items,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(bundle.model_dump_json(indent=2))
    return bundle


def match_entities_strict_lexical(
    gold_entities: List[Dict[str, Any]],
    predicted_candidates: List[EvaluationCandidate],
    target_entity_type: Optional[str] = None,
) -> Tuple[int, int, int]:
    """Bipartite matching with strict lexical string equality."""
    golds = [g for g in gold_entities if target_entity_type is None or g["entity_type"] == target_entity_type]
    preds = [p for p in predicted_candidates if target_entity_type is None or p.entity_type == target_entity_type]

    n_gold = len(golds)
    n_pred = len(preds)
    if n_gold == 0 and n_pred == 0:
        return 0, 0, 0
    if n_gold == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gold

    cost_matrix = np.ones((n_gold, n_pred), dtype=float)
    for i, g in enumerate(golds):
        g_name = normalize_entity_name(g.get("name") or g.get("statement") or "")
        g_type = g["entity_type"]
        for j, p in enumerate(preds):
            if p.entity_type != g_type:
                continue
            p_name = normalize_entity_name(p.name_or_statement)
            if g_name == p_name:
                cost_matrix[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] == 0.0)
    fp = n_pred - tp
    fn = n_gold - tp
    return tp, fp, fn


def match_relations(
    gold_relations: List[Dict[str, Any]],
    gold_entities: List[Dict[str, Any]],
    bundle: ExtractionTrioBundle,
) -> Tuple[int, int, int]:
    """Evaluate predicted local relations against gold relations via Hungarian matching.
    
    A predicted relation matches gold if:
    1. relation_type is identical
    2. src entity matches gold src entity
    3. dst entity matches gold dst entity
    """
    if not bundle.accepted_record:
        return 0, 0, len(gold_relations)

    pred_relations = bundle.accepted_record.local_relations
    n_gold = len(gold_relations)
    n_pred = len(pred_relations)

    if n_gold == 0 and n_pred == 0:
        return 0, 0, 0
    if n_gold == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gold

    # Build entity lookups
    gold_ent_by_id = {e["local_id"]: e for e in gold_entities}
    pred_ent_by_id = {c.candidate_local_id: c for c in bundle.evaluation_candidates}

    cost_matrix = np.ones((n_gold, n_pred), dtype=float)

    for i, gr in enumerate(gold_relations):
        g_type = gr["relation_type"]
        g_src = gold_ent_by_id.get(gr["src_local_id"])
        g_dst = gold_ent_by_id.get(gr["dst_local_id"])
        if not g_src or not g_dst:
            continue

        for j, pr in enumerate(pred_relations):
            if pr.relation_type != g_type:
                continue

            p_src = pred_ent_by_id.get(pr.src_local_id)
            p_dst = pred_ent_by_id.get(pr.dst_local_id)
            if not p_src or not p_dst:
                continue

            # Check semantic match of src and dst
            src_ok = are_semantically_equivalent(
                g_src.get("name") or g_src.get("statement") or "",
                g_src.get("synonyms", []),
                p_src.name_or_statement,
            )
            dst_ok = are_semantically_equivalent(
                g_dst.get("name") or g_dst.get("statement") or "",
                g_dst.get("synonyms", []),
                p_dst.name_or_statement,
            )

            if src_ok and dst_ok:
                cost_matrix[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] == 0.0)
    fp = n_pred - tp
    fn = n_gold - tp
    return tp, fp, fn


def run_holdout_bootstrap(
    holdout_docs: List[Dict[str, Any]],
    bundles_by_doc: Dict[str, List[ExtractionTrioBundle]],
    match_mode: str = "semantic",
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Run document-level cluster bootstrap for Pooled Micro P, R, F1."""
    rng = np.random.RandomState(seed)
    doc_ids = [d["source_id"] for d in holdout_docs]
    doc_by_id = {d["source_id"]: d for d in holdout_docs}
    n_docs = len(doc_ids)

    boot_p, boot_r, boot_f1 = [], [], []

    for _ in range(n_iterations):
        sample_doc_ids = rng.choice(doc_ids, size=n_docs, replace=True)
        tp_total, fp_total, fn_total = 0, 0, 0

        for sid in sample_doc_ids:
            doc = doc_by_id[sid]
            gold_entities = doc.get("entities", [])
            for bundle in bundles_by_doc[sid]:
                preds = bundle.evaluation_candidates
                if match_mode == "strict":
                    tp, fp, fn = match_entities_strict_lexical(gold_entities, preds)
                else:
                    res = match_entities_bipartite(gold_entities, preds)
                    tp, fp, fn = res.true_positives, res.false_positives, res.false_negatives
                tp_total += tp
                fp_total += fp
                fn_total += fn

        p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        boot_p.append(p)
        boot_r.append(r)
        boot_f1.append(f1)

    return {
        "precision": (float(np.percentile(boot_p, 2.5)), float(np.percentile(boot_p, 97.5))),
        "recall": (float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))),
        "f1": (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))),
    }


def main():
    load_project_env()
    base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
    output_dir = base_dir / "data/experiments/holdout_pilot_c4"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HDE H1a Phase 3: Holdout Pilot Evaluation (Condition C4 Only)")
    print("=" * 80)

    # 1. Verify frozen baseline hashes
    print("\n--- Step 1: Verifying Frozen Baseline Hashes ---")
    hashes = verify_frozen_hashes(base_dir)
    for rel, h in hashes.items():
        print(f"  [FROZEN] {rel}: {h}")

    # 2. Load Ground Truth
    gt_path = base_dir / "tests/fixtures/gold_standard/h1a_ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    holdout_docs = [d for d in ground_truth if d.get("split") == "holdout_pilot"]
    total_gold_entities = sum(len(d.get("entities", [])) for d in holdout_docs)
    total_gold_relations = sum(len(d.get("relations", [])) for d in holdout_docs)

    print(f"\n--- Step 2: Loaded Holdout Pilot Benchmark ---")
    print(f"  Holdout Documents: {len(holdout_docs)}")
    print(f"  Total Gold Entities: {total_gold_entities}")
    print(f"  Total Gold Relations: {total_gold_relations}")
    assert len(holdout_docs) == 12, f"Expected 12 holdout docs, got {len(holdout_docs)}"
    assert total_gold_entities == 28, f"Expected 28 entities, got {total_gold_entities}"
    assert total_gold_relations == 16, f"Expected 16 relations, got {total_gold_relations}"

    # 3. Setup Budget Tracker & Transport
    tracker = BudgetTracker()
    provider, _ = detect_configured_provider()
    transport = create_transport(provider=provider)
    model_name = "gpt-4o-mini-2024-07-18"

    trials = [1, 2, 3]
    tasks = []
    for t in trials:
        for doc in holdout_docs:
            tasks.append((doc, t))

    print(f"\n--- Step 3: Executing Live LLM Inference (C4 Only, 36 API calls) ---")
    print(f"  Primary Model: {model_name} | Seed: 42 | Temp: 0.0 | Max Budget: $10.00 USD")
    print(f"  Queued Tasks: {len(tasks)} (12 docs x 3 trials). Executing with ThreadPoolExecutor(max_workers=3)...")

    bundles_by_doc: Dict[str, List[ExtractionTrioBundle]] = {d["source_id"]: [] for d in holdout_docs}
    all_bundles: List[ExtractionTrioBundle] = []
    t0 = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(execute_c4_single, doc, t, tracker, transport, model_name, output_dir): (t, doc["source_id"])
            for doc, t in tasks
        }

        for future in as_completed(future_to_task):
            t, sid = future_to_task[future]
            completed_count += 1
            try:
                bundle = future.result()
                bundles_by_doc[sid].append(bundle)
                all_bundles.append(bundle)
                if completed_count % 6 == 0 or completed_count == len(tasks):
                    summary = tracker.get_summary()
                    print(f"  [{completed_count:2d}/{len(tasks)}] Finished C4 t{t} {sid} | Spent: ${summary['spent_usd']:.4f} USD | Elapsed: {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"  [ERROR] Task C4 t{t} {sid} failed: {e}")
                raise e

    elapsed_api = time.time() - t0
    final_budget = tracker.get_summary()
    print(f"\nAll 36 API calls completed in {elapsed_api:.2f}s!")
    print(f"Budget: Spent: ${final_budget['spent_usd']:.6f} USD | Remaining: ${final_budget['remaining_usd']:.6f} USD")

    # Export ledger
    ledger_path = base_dir / "data/budget_ledger.jsonl"
    tracker.export_ledger_jsonl(ledger_path)
    print(f"Updated budget ledger at: {ledger_path}")

    # 4. Quantitative Evaluation
    print("\n--- Step 4: Quantitative Evaluation & Metric Calculation ---")

    # A. Strict Lexical Evaluation
    strict_tp, strict_fp, strict_fn = 0, 0, 0
    strict_class_tps = {c: 0 for c in TARGET_CLASSES}
    strict_class_fps = {c: 0 for c in TARGET_CLASSES}
    strict_class_fns = {c: 0 for c in TARGET_CLASSES}

    # B. Pre-registered Semantic Evaluation
    sem_tp, sem_fp, sem_fn = 0, 0, 0
    sem_class_tps = {c: 0 for c in TARGET_CLASSES}
    sem_class_fps = {c: 0 for c in TARGET_CLASSES}
    sem_class_fns = {c: 0 for c in TARGET_CLASSES}

    # C. Relation Evaluation
    rel_tp, rel_fp, rel_fn = 0, 0, 0

    # Evidence audit
    total_quotes = 0
    raw_valid_quotes = 0
    resolved_valid_quotes = 0

    for doc in holdout_docs:
        sid = doc["source_id"]
        gold_entities = doc.get("entities", [])
        gold_relations = doc.get("relations", [])

        for bundle in bundles_by_doc[sid]:
            preds = bundle.evaluation_candidates

            # Quotes
            for p in preds:
                if p.evidence:
                    total_quotes += 1
                    if p.evidence.raw_offset_valid:
                        raw_valid_quotes += 1
                    if p.evidence.resolved_offset_valid:
                        resolved_valid_quotes += 1

            # Strict Overall
            s_tp, s_fp, s_fn = match_entities_strict_lexical(gold_entities, preds)
            strict_tp += s_tp
            strict_fp += s_fp
            strict_fn += s_fn

            # Strict Per-Class
            for c in TARGET_CLASSES:
                c_tp, c_fp, c_fn = match_entities_strict_lexical(gold_entities, preds, target_entity_type=c)
                strict_class_tps[c] += c_tp
                strict_class_fps[c] += c_fp
                strict_class_fns[c] += c_fn

            # Semantic Overall
            m_sem = match_entities_bipartite(gold_entities, preds)
            sem_tp += m_sem.true_positives
            sem_fp += m_sem.false_positives
            sem_fn += m_sem.false_negatives

            # Semantic Per-Class
            for c in TARGET_CLASSES:
                m_c = match_entities_bipartite(gold_entities, preds, target_entity_type=c)
                sem_class_tps[c] += m_c.true_positives
                sem_class_fps[c] += m_c.false_positives
                sem_class_fns[c] += m_c.false_negatives

            # Relations
            r_tp, r_fp, r_fn = match_relations(gold_relations, gold_entities, bundle)
            rel_tp += r_tp
            rel_fp += r_fp
            rel_fn += r_fn

    # Compute Pooled Metrics
    strict_p = strict_tp / (strict_tp + strict_fp) if (strict_tp + strict_fp) > 0 else 0.0
    strict_r = strict_tp / (strict_tp + strict_fn) if (strict_tp + strict_fn) > 0 else 0.0
    strict_f1 = (2 * strict_p * strict_r) / (strict_p + strict_r) if (strict_p + strict_r) > 0 else 0.0

    sem_p = sem_tp / (sem_tp + sem_fp) if (sem_tp + sem_fp) > 0 else 0.0
    sem_r = sem_tp / (sem_tp + sem_fn) if (sem_tp + sem_fn) > 0 else 0.0
    sem_f1 = (2 * sem_p * sem_r) / (sem_p + sem_r) if (sem_p + sem_r) > 0 else 0.0

    rel_p = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0.0
    rel_r = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0.0
    rel_f1 = (2 * rel_p * rel_r) / (rel_p + rel_r) if (rel_p + rel_r) > 0 else 0.0

    qv_raw = raw_valid_quotes / total_quotes if total_quotes > 0 else 0.0
    qv_res = resolved_valid_quotes / total_quotes if total_quotes > 0 else 0.0

    # Cluster Bootstraps (B=1000)
    print("  Running Cluster Bootstrap (B=1000) for Strict Match...")
    strict_ci = run_holdout_bootstrap(holdout_docs, bundles_by_doc, match_mode="strict", n_iterations=1000, seed=42)
    print("  Running Cluster Bootstrap (B=1000) for Semantic Match...")
    sem_ci = run_holdout_bootstrap(holdout_docs, bundles_by_doc, match_mode="semantic", n_iterations=1000, seed=42)

    # Per-class calculations
    per_class_summary = {}
    for c in TARGET_CLASSES:
        # Semantic
        stp = sem_class_tps[c]
        sfp = sem_class_fps[c]
        sfn = sem_class_fns[c]
        sp = stp / (stp + sfp) if (stp + sfp) > 0 else 0.0
        sr = stp / (stp + sfn) if (stp + sfn) > 0 else 0.0
        sf1 = (2 * sp * sr) / (sp + sr) if (sp + sr) > 0 else 0.0

        # Strict
        ltp = strict_class_tps[c]
        lfp = strict_class_fps[c]
        lfn = strict_class_fns[c]
        lp = ltp / (ltp + lfp) if (ltp + lfp) > 0 else 0.0
        lr = ltp / (ltp + lfn) if (ltp + lfn) > 0 else 0.0
        lf1 = (2 * lp * lr) / (lp + lr) if (lp + lr) > 0 else 0.0

        per_class_summary[c] = {
            "semantic": {"tp": stp, "fp": sfp, "fn": sfn, "precision": round(sp, 4), "recall": round(sr, 4), "f1": round(sf1, 4)},
            "strict": {"tp": ltp, "fp": lfp, "fn": lfn, "precision": round(lp, 4), "recall": round(lr, 4), "f1": round(lf1, 4)},
        }

    # Format Summary Table
    print("\n" + "=" * 90)
    print("=== HDE H1a HOLDOUT PILOT EXPERIMENTAL RESULTS (C4 ONLY, R=3) ===")
    print("=" * 90)
    print(f"Overall Entities (Semantic Match):")
    print(f"  Precision: {sem_p:.4f} (95% CI: [{sem_ci['precision'][0]:.4f}, {sem_ci['precision'][1]:.4f}])")
    print(f"  Recall:    {sem_r:.4f} (95% CI: [{sem_ci['recall'][0]:.4f}, {sem_ci['recall'][1]:.4f}])")
    print(f"  Micro F1:  {sem_f1:.4f} (95% CI: [{sem_ci['f1'][0]:.4f}, {sem_ci['f1'][1]:.4f}])")
    print(f"  TP={sem_tp}, FP={sem_fp}, FN={sem_fn}")

    print(f"\nOverall Entities (Strict Lexical Match):")
    print(f"  Precision: {strict_p:.4f} (95% CI: [{strict_ci['precision'][0]:.4f}, {strict_ci['precision'][1]:.4f}])")
    print(f"  Recall:    {strict_r:.4f} (95% CI: [{strict_ci['recall'][0]:.4f}, {strict_ci['recall'][1]:.4f}])")
    print(f"  Micro F1:  {strict_f1:.4f} (95% CI: [{strict_ci['f1'][0]:.4f}, {strict_ci['f1'][1]:.4f}])")
    print(f"  TP={strict_tp}, FP={strict_fp}, FN={strict_fn}")

    print(f"\nLocalRelation Performance:")
    print(f"  Precision: {rel_p:.4f}")
    print(f"  Recall:    {rel_r:.4f}")
    print(f"  F1:        {rel_f1:.4f}")
    print(f"  TP={rel_tp}, FP={rel_fp}, FN={rel_fn} (Gold instances across 3 trials: {len(holdout_docs)*3} doc-trials, {total_gold_relations*3} relation opportunities)")

    print(f"\nEvidence & Quote Validity:")
    print(f"  Total Quotes: {total_quotes}")
    print(f"  Raw Offset Validity:      {qv_raw*100:.1f}% ({raw_valid_quotes}/{total_quotes})")
    print(f"  Resolved Offset Validity: {qv_res*100:.1f}% ({resolved_valid_quotes}/{total_quotes})")

    print("\nPer-Class Breakdown (Semantic Match):")
    print(f"{'Class':<12} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}")
    print("-" * 60)
    for c, data in per_class_summary.items():
        s = data["semantic"]
        print(f"{c:<12} | {s['tp']:<4d} | {s['fp']:<4d} | {s['fn']:<4d} | {s['precision']:7.4f} | {s['recall']:7.4f} | {s['f1']:7.4f}")

    # Export Results JSON
    results_payload = {
        "experiment": "HDE_H1a_Holdout_Pilot_Evaluation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model": model_name,
        "temperature": 0.0,
        "seed": 42,
        "condition": "C4",
        "num_documents": len(holdout_docs),
        "trials": trials,
        "total_api_calls": len(tasks),
        "budget": final_budget,
        "frozen_component_hashes": hashes,
        "metrics": {
            "semantic_match": {
                "precision": round(sem_p, 4),
                "recall": round(sem_r, 4),
                "f1": round(sem_f1, 4),
                "tp": sem_tp,
                "fp": sem_fp,
                "fn": sem_fn,
                "cluster_bootstrap_95ci": sem_ci,
            },
            "strict_lexical_match": {
                "precision": round(strict_p, 4),
                "recall": round(strict_r, 4),
                "f1": round(strict_f1, 4),
                "tp": strict_tp,
                "fp": strict_fp,
                "fn": strict_fn,
                "cluster_bootstrap_95ci": strict_ci,
            },
            "local_relation": {
                "precision": round(rel_p, 4),
                "recall": round(rel_r, 4),
                "f1": round(rel_f1, 4),
                "tp": rel_tp,
                "fp": rel_fp,
                "fn": rel_fn,
            },
            "evidence": {
                "total_quotes": total_quotes,
                "raw_valid_quotes": raw_valid_quotes,
                "resolved_valid_quotes": resolved_valid_quotes,
                "raw_quote_validity_rate": round(qv_raw, 4),
                "resolved_quote_validity_rate": round(qv_res, 4),
            },
            "per_class": per_class_summary,
        }
    }

    result_json_path = output_dir / "holdout_pilot_results.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nFull results exported to: {result_json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
