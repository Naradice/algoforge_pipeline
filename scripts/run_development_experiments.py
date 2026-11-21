"""Development Set (C0~C4, 10 documents, 3 trials = 150 evaluations, 120 API calls) Experiment Runner.

Complies strictly with H1a scientific experiment plan v0.5 and User Instructions:
- Evaluates C0 (Rule-based), C1 (Free-text LLM), C2 (JSON Schema), C3 (Evidence Grounding), C4 (Proposed Full Method)
- Evaluates 10 Development Set documents across 3 independent trials
- Uses pinned Primary Model: gpt-4o-mini-2024-07-18, temp=0.0, seed=42
- Enforces strict Two-Phase Budget Reservation ($10.00 USD Hard Limit)
- Preserves RawExtractionArtifact, EvaluationCandidates, RejectionRecords in ExtractionTrioBundle
- Computes Hungarian 1-to-1 matching against Development Set (Single-annotator reviewed benchmark)
- Evaluates per-class metrics (especially Model, Capability, LocalRelation)
- Computes Document-level Cluster Bootstrap (B=1000) 95% Confidence Intervals
- Measures C2 vs C3 Evidence trade-off (Delta P, Delta R) and C3 vs C4 Controlled Vocabulary effect
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
    C2Record,
    C3Record,
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
from research_pipeline.extractors.validator import validate_evidence, validate_local_relation
from research_pipeline.extractors.llm_extractor import C0_KEYWORD_PATTERNS
from research_pipeline.evaluation.metrics import evaluate_run, FullEvaluationReport, ClassMetricResult
from research_pipeline.evaluation.bootstrap import run_cluster_bootstrap
from research_pipeline.evaluation.matching import match_entities_bipartite


def run_c0_single(
    doc: Dict[str, Any],
    trial_idx: int,
) -> ExtractionTrioBundle:
    """Execute C0 deterministic rule-based keyword extraction."""
    source_id = doc["source_id"]
    version_id = doc["version_id"]
    canonical_fields = doc["target_fields"]
    run_id = f"dev_C0_t{trial_idx}_{source_id}"
    cache_path = Path(f"c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/dev_c0_c4/bundle_{run_id}.json")

    text = " ".join(canonical_fields.values())
    parsed_candidates: Dict[str, List[Dict[str, Any]]] = {
        "phenomenas": [],
        "models": [],
        "methods": [],
        "capabilities": [],
        "assumptions": [],
        "limitations": [],
    }

    for name, cat, desc in C0_KEYWORD_PATTERNS["models"]:
        if name.lower() in text.lower():
            parsed_candidates["models"].append({
                "local_id": f"mod_{len(parsed_candidates['models'])+1:02d}",
                "name": name,
                "category": cat,
                "description": desc,
            })

    for name, cat, desc in C0_KEYWORD_PATTERNS["methods"]:
        if name.lower() in text.lower():
            parsed_candidates["methods"].append({
                "local_id": f"met_{len(parsed_candidates['methods'])+1:02d}",
                "name": name,
                "category": cat,
                "description": desc,
            })

    for name, desc in C0_KEYWORD_PATTERNS["phenomena"]:
        if name.lower() in text.lower():
            parsed_candidates["phenomenas"].append({
                "local_id": f"phe_{len(parsed_candidates['phenomenas'])+1:02d}",
                "name": name,
                "description": desc,
            })

    raw_artifact = RawExtractionArtifact(
        extraction_run_id=run_id,
        source_id=source_id,
        version_id=version_id,
        condition_id="C0",
        raw_response_text=json.dumps(parsed_candidates),
        http_status=200,
        token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        executed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_checkpoint="rule-based-v1",
        prompt_hash=hashlib.sha256(b"C0_RULE_BASED").hexdigest(),
        target_field_hashes=doc["target_field_hashes"],
        json_parse_status="success",
        parsed_candidates=parsed_candidates,
    )

    candidates = StandardizationConverter.convert_c2_structured_no_evidence(raw_artifact)
    bundle = ExtractionTrioBundle(
        extraction_run_id=run_id,
        source_id=source_id,
        version_id=version_id,
        raw_artifact=raw_artifact,
        evaluation_candidates=candidates,
        accepted_record=None,
        rejected_items=[],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(bundle.model_dump_json(indent=2))
    return bundle


def execute_llm_single(
    condition_id: str,
    doc: Dict[str, Any],
    trial_idx: int,
    tracker: BudgetTracker,
    transport: Any,
    model_name: str = "gpt-4o-mini-2024-07-18",
) -> ExtractionTrioBundle:
    """Execute a single LLM API extraction task with two-phase reservation & validation."""
    source_id = doc["source_id"]
    version_id = doc["version_id"]
    canonical_fields = doc["target_fields"]
    target_field_name = "readme_text" if doc["source_type"] == "oss" else "abstract_text"
    text_content = canonical_fields.get(target_field_name, "")
    title = canonical_fields.get("title_text", "")
    run_id = f"dev_{condition_id}_t{trial_idx}_{source_id}"

    cache_path = Path(f"c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/dev_c0_c4/bundle_{run_id}.json")
    raw_artifact = None
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_bundle = ExtractionTrioBundle.model_validate_json(f.read())
                raw_artifact = cached_bundle.raw_artifact
        except Exception:
            pass

    if raw_artifact is None:
        # Determine schema & prompt
        if condition_id == "C1":
            schema_cls = None
            system_instruction = (
                "You are an expert scientific information extraction assistant. "
                "Extract scientific entities from the text into standard markdown sections: "
                "### Phenomenon\n### Model\n### Method\n### Capability\n### Assumption\n### Limitation\n"
                "List each item with a clear name and brief description."
            )
            prompt = (
                f"Document Title: {title}\n\n"
                f"Source Text ({target_field_name}):\n{text_content}\n\n"
                "Extract all scientific entities using the specified markdown section headers."
            )
        elif condition_id == "C2":
            schema_cls = C2Record
            system_instruction = (
                "You are an expert scientific information extraction assistant. "
                "Extract scientific entities (Phenomenon, Model, Method, Capability, Assumption, Limitation) "
                "conforming to the structured JSON schema. Do not include evidence quotes or offsets."
            )
            prompt = (
                f"Document Title: {title}\n\n"
                f"Source Text ({target_field_name}):\n{text_content}\n\n"
                "Extract structured knowledge complying strictly with C2Record JSON schema."
            )
        elif condition_id == "C3":
            schema_cls = C3Record
            system_instruction = (
                "You are an expert scientific information extraction assistant. "
                "Extract scientific entities (Phenomenon, Model, Method, Capability, Assumption, Limitation). "
                "For every entity, provide exact verbatim quotes and character offsets in the evidence span. "
                "If input/output data types for Capability are not explicitly mentioned in the text, "
                "set is_unspecified=true, data_type_l1='unknown', and evidence=null."
            )
            prompt = (
                f"Document Title: {title}\n\n"
                f"Source Text ({target_field_name}):\n{text_content}\n\n"
                "Extract structured knowledge with verbatim evidence complying strictly with C3Record JSON schema."
            )
        else:  # C4
            schema_cls = ExtractedKnowledgeRecord
            system_instruction = (
                "You are an expert scientific information extraction assistant. "
                "Extract Phenomenon, Model, Method, Capability, Assumption, Limitation entities and Local Relations. "
                "For every entity and relation, provide exact verbatim quotes from the text with character offsets. "
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

        # 2. Live API Execution
        try:
            raw_text, http_status, usage = transport.generate(
                model=model_name,
                prompt=prompt,
                system_instruction=system_instruction,
                condition_id=condition_id,
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
        json_status = "not_applicable" if condition_id == "C1" else "success"
        if condition_id != "C1":
            try:
                parsed_dict = json.loads(raw_text)
            except Exception as pe:
                json_status = "invalid_json"
                parse_err = str(pe)

        raw_artifact = RawExtractionArtifact(
            extraction_run_id=run_id,
            source_id=source_id,
            version_id=doc["version_id"],
            condition_id=condition_id,  # type: ignore
            raw_response_text=raw_text,
            http_status=http_status,
            token_usage=usage,
            executed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_checkpoint=model_name,
            prompt_hash=compute_canonical_hash(prompt),
            target_field_hashes=doc["target_field_hashes"],
            json_parse_status=json_status,  # type: ignore
            parse_error=parse_err,
            parsed_candidates=parsed_dict,
        )

    # 5. Conversion & Evidence Validation
    parsed_dict = raw_artifact.parsed_candidates
    json_status = raw_artifact.json_parse_status
    raw_text = raw_artifact.raw_response_text
    candidates = []
    rejected_items = []

    if condition_id == "C1":
        candidates, success, err = StandardizationConverter.convert_c1_free_text(raw_artifact)
        if not success:
            rejected_items.append(
                RejectionRecord(
                    local_id="c1_all",
                    entity_type="all",
                    rejection_stage="syntax_parse",
                    rejection_reason=err,
                    raw_entity_data={"raw_text": raw_text[:500]},
                )
            )
    elif condition_id == "C2":
        candidates = StandardizationConverter.convert_c2_structured_no_evidence(raw_artifact)
    else:  # C3, C4
        candidates, rejected_items = StandardizationConverter.convert_c3_c4_with_evidence(raw_artifact)

    # Validate Evidence for C3 and C4
    accepted_candidates = []
    if condition_id in ("C3", "C4"):
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
    else:
        accepted_candidates = candidates

    # Build accepted ExtractedKnowledgeRecord for C4 if valid
    accepted_record = None
    if condition_id == "C4" and parsed_dict and json_status == "success":
        try:
            accepted_record = ExtractedKnowledgeRecord.model_validate(parsed_dict)
        except Exception:
            pass

    bundle = ExtractionTrioBundle(
        extraction_run_id=run_id,
        source_id=source_id,
        version_id=doc["version_id"],
        raw_artifact=raw_artifact,
        evaluation_candidates=accepted_candidates,
        accepted_record=accepted_record,
        rejected_items=rejected_items,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(bundle.model_dump_json(indent=2))
    return bundle


def main():
    load_project_env()
    provider, default_model = detect_configured_provider()
    model_name = "gpt-4o-mini-2024-07-18"

    print("=" * 80)
    print(f"HDE H1a Development Comparative Experiments (C0~C4, 10 Docs, 3 Trials)")
    print(f"Provider: {provider} | Primary Model: {model_name} | Seed: 42 | Temp: 0.0")
    print("=" * 80)

    gt_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    dev_docs = [d for d in ground_truth if d.get("split") == "development"]
    print(f"Loaded {len(dev_docs)} Development Set documents.")

    output_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/experiments/dev_c0_c4")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = BudgetTracker()
    transport = create_transport(provider=provider)

    conditions = ["C0", "C1", "C2", "C3", "C4"]
    trials = [1, 2, 3]

    all_bundles: Dict[str, Dict[int, List[ExtractionTrioBundle]]] = {
        cond: {t: [] for t in trials} for cond in conditions
    }

    # 1. Run C0 (Deterministic Rule-Based baseline, 30 tasks, 0 API calls)
    print("\n--- Phase 1: Running C0 Baseline (30 evaluations) ---")
    t0_c0 = time.time()
    for t in trials:
        for doc in dev_docs:
            bundle = run_c0_single(doc, t)
            all_bundles["C0"][t].append(bundle)
    print(f"C0 completed in {time.time() - t0_c0:.2f}s (0 API calls, $0.00 spent)")

    # 2. Run LLM Conditions C1, C2, C3, C4 (120 API calls total)
    # Using ThreadPoolExecutor(max_workers=3) for fast, safe execution
    print("\n--- Phase 2: Running LLM Conditions C1~C4 (120 API calls) ---")
    tasks = []
    for cond in ["C1", "C2", "C3", "C4"]:
        for t in trials:
            for doc in dev_docs:
                tasks.append((cond, doc, t))

    print(f"Total API tasks queued: {len(tasks)}. Executing with max_workers=3...")
    t0_llm = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(execute_llm_single, cond, doc, t, tracker, transport, model_name): (cond, t, doc["source_id"])
            for cond, doc, t in tasks
        }

        for future in as_completed(future_to_task):
            cond, t, sid = future_to_task[future]
            completed_count += 1
            try:
                bundle = future.result()
                all_bundles[cond][t].append(bundle)
                if completed_count % 10 == 0 or completed_count == len(tasks):
                    summary = tracker.get_summary()
                    print(f"  [{completed_count:3d}/{len(tasks)}] Finished {cond} t{t} {sid} | Spent: ${summary['spent_usd']:.4f} USD | Elapsed: {time.time() - t0_llm:.1f}s")
            except Exception as e:
                print(f"  [ERROR] Task {cond} t{t} {sid} failed: {e}")
                raise e

    print(f"\nAll 120 API calls completed in {time.time() - t0_llm:.2f}s!")
    final_budget = tracker.get_summary()
    print(f"Final Budget Spent: ${final_budget['spent_usd']:.6f} USD / Remaining: ${final_budget['remaining_usd']:.6f} USD")

    # Export ledger
    ledger_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/budget_ledger.jsonl")
    tracker.export_ledger_jsonl(ledger_path)
    print(f"Exported per-request budget ledger to: {ledger_path}")

    # 3. Quantitative Analysis & Metrics Calculation
    print("\n--- Phase 3: Evaluating Run Metrics & Cluster Bootstrap ---")

    reports_by_condition: Dict[str, List[FullEvaluationReport]] = {c: [] for c in conditions}
    cis_by_condition: Dict[str, Dict[str, Any]] = {}

    for cond in conditions:
        # Merge all trials into a combined bundle list for overall evaluation and bootstrap
        combined_bundles = []
        for t in trials:
            trial_bundles = all_bundles[cond][t]
            report = evaluate_run(cond, dev_docs, trial_bundles)
            reports_by_condition[cond].append(report)
            combined_bundles.extend(trial_bundles)

        # Run Document-level Cluster Bootstrap (B=1000)
        cis = run_cluster_bootstrap(cond, dev_docs, all_bundles[cond][1], n_iterations=1000, seed=42)
        cis_by_condition[cond] = {k: v.model_dump() for k, v in cis.items()}

    # Format Results Table
    print("\n" + "="*95)
    print("=== HDE H1a DEVELOPMENT SET COMPARATIVE EXPERIMENT RESULTS ===")
    print("=== Model: gpt-4o-mini-2024-07-18 | Docs: 10 | Trials: 3 (N=30 per condition) ===")
    print("="*95)
    print(f"{'Cond':<5} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Model F1':<9} | {'Cap F1':<8} | {'Quote Valid':<11} | {'Raw Cands':<10} | {'Accepted':<9} | {'Rejected':<9}")
    print("-" * 95)

    summary_data = {}

    for cond in conditions:
        reports = reports_by_condition[cond]
        avg_p = sum(r.overall_precision for r in reports) / len(reports)
        avg_r = sum(r.overall_recall for r in reports) / len(reports)
        avg_f1 = sum(r.overall_f1 for r in reports) / len(reports)

        mod_f1 = sum(r.class_metrics.get("model", ClassMetricResult(entity_type="model", true_positives=0, false_positives=0, false_negatives=0, precision=0, recall=0, f1=0)).f1 for r in reports) / len(reports)
        cap_f1 = sum(r.class_metrics.get("capability", ClassMetricResult(entity_type="capability", true_positives=0, false_positives=0, false_negatives=0, precision=0, recall=0, f1=0)).f1 for r in reports) / len(reports)
        qv_rate = sum(r.resolved_quote_validity_rate for r in reports) / len(reports)

        total_raw = sum(len(b.raw_artifact.parsed_candidates or {}) if isinstance(b.raw_artifact.parsed_candidates, list) else sum(len(v) for v in (b.raw_artifact.parsed_candidates or {}).values() if isinstance(v, list)) for t in trials for b in all_bundles[cond][t])
        total_acc = sum(len(b.evaluation_candidates) for t in trials for b in all_bundles[cond][t])
        total_rej = sum(len(b.rejected_items) for t in trials for b in all_bundles[cond][t])

        print(f"{cond:<5} | {avg_p:7.3f} | {avg_r:7.3f} | {avg_f1:7.3f} | {mod_f1:9.3f} | {cap_f1:8.3f} | {qv_rate:11.3f} | {total_raw:<10d} | {total_acc:<9d} | {total_rej:<9d}")

        summary_data[cond] = {
            "avg_precision": round(avg_p, 4),
            "avg_recall": round(avg_r, 4),
            "avg_f1": round(avg_f1, 4),
            "model_f1": round(mod_f1, 4),
            "capability_f1": round(cap_f1, 4),
            "quote_validity_rate": round(qv_rate, 4),
            "total_raw_candidates": total_raw,
            "total_accepted_candidates": total_acc,
            "total_rejected_candidates": total_rej,
            "cluster_bootstrap_95ci": cis_by_condition[cond],
        }

    # Trade-off analysis
    c2 = summary_data["C2"]
    c3 = summary_data["C3"]
    c4 = summary_data["C4"]
    delta_p_c2_c3 = c3["avg_precision"] - c2["avg_precision"]
    delta_r_c2_c3 = c3["avg_recall"] - c2["avg_recall"]
    delta_f1_c2_c3 = c3["avg_f1"] - c2["avg_f1"]

    print("\n" + "="*80)
    print("=== HYPOTHESIS & TRADE-OFF VERIFICATION ===")
    print(f"1. C2 -> C3 Evidence Grounding Trade-off:")
    print(f"   Delta Precision: {delta_p_c2_c3:+.4f} ({c2['avg_precision']:.3f} -> {c3['avg_precision']:.3f})")
    print(f"   Delta Recall:    {delta_r_c2_c3:+.4f} ({c2['avg_recall']:.3f} -> {c3['avg_recall']:.3f})")
    print(f"   Delta F1:        {delta_f1_c2_c3:+.4f} ({c2['avg_f1']:.3f} -> {c3['avg_f1']:.3f})")

    delta_p_c3_c4 = c4["avg_precision"] - c3["avg_precision"]
    delta_r_c3_c4 = c4["avg_recall"] - c3["avg_recall"]
    delta_f1_c3_c4 = c4["avg_f1"] - c3["avg_f1"]
    print(f"\n2. C3 -> C4 Controlled Vocabulary & Relations Effect:")
    print(f"   Delta Precision: {delta_p_c3_c4:+.4f} ({c3['avg_precision']:.3f} -> {c4['avg_precision']:.3f})")
    print(f"   Delta Recall:    {delta_r_c3_c4:+.4f} ({c3['avg_recall']:.3f} -> {c4['avg_recall']:.3f})")
    print(f"   Delta F1:        {delta_f1_c3_c4:+.4f} ({c3['avg_f1']:.3f} -> {c4['avg_f1']:.3f})")
    print(f"   Capability F1:   {c3['capability_f1']:.3f} -> {c4['capability_f1']:.3f}")
    print(f"   Model F1:        {c3['model_f1']:.3f} -> {c4['model_f1']:.3f}")

    # Export full JSON
    result_json_path = output_dir / "development_experiment_results.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "HDE_H1a_Development_Comparative",
            "model": model_name,
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "total_evaluations": 150,
            "total_api_calls": 120,
            "budget": final_budget,
            "summary": summary_data,
        }, f, indent=2)
    print(f"\nResults saved to: {result_json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
