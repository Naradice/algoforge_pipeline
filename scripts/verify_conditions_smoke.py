"""Verification script for C1~C4 condition-specific smoke tests.

Executes 1 live API call for each of C1, C2, C3, C4 on Development Set documents.
Validates:
1. Token reservation vs actual usage (covering different schemas, free-text, and long README inputs)
2. Cost reservation vs settled cost
3. Standardization converters and parse error handling
4. Evidence verification and deterministic offset resolution
5. Full per-request logging into budget ledger
"""

import sys
import json
import time
from pathlib import Path

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
)
from research_pipeline.extractors.converter import StandardizationConverter
from research_pipeline.extractors.validator import validate_evidence, validate_local_relation
from research_pipeline.evaluation.matching import match_entities_bipartite


def run_condition_smoke():
    load_project_env()
    provider, _ = detect_configured_provider()
    print(f"=== Starting C1~C4 Condition-Specific Smoke Test ({provider}) ===")

    gt_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    doc_map = {d["source_id"]: d for d in ground_truth if d.get("split") == "development"}

    # Selection: Papers for C1~C3, Long README for C4
    test_cases = [
        ("C1", "arxiv_2301_01001", "abstract_text", None),
        ("C2", "arxiv_2301_01002", "abstract_text", C2Record),
        ("C3", "arxiv_2301_01003", "abstract_text", C3Record),
        ("C4", "github_10101", "readme_text", ExtractedKnowledgeRecord),
    ]

    active_model = "gpt-4o-mini-2024-07-18"
    tracker = BudgetTracker()
    transport = create_transport(provider=provider)

    results = []

    for cond_id, source_id, target_field, schema_cls in test_cases:
        doc = doc_map[source_id]
        canonical_fields = doc["target_fields"]
        text_content = canonical_fields[target_field]
        title = canonical_fields["title_text"]

        print(f"\n" + "="*70)
        print(f"Testing {cond_id} on {source_id} ({target_field}, text len: {len(text_content)} chars)")
        print(f"Title: {title}")

        # Construct System Instruction & User Prompt per condition
        if cond_id == "C1":
            system_instruction = (
                "You are an expert scientific information extraction assistant. "
                "Extract scientific entities from the text into standard markdown sections: "
                "### Phenomenon\n### Model\n### Method\n### Capability\n### Assumption\n### Limitation\n"
                "List each item with a clear name and brief description."
            )
            prompt = (
                f"Document Title: {title}\n\n"
                f"Source Text:\n{text_content}\n\n"
                "Extract all scientific entities using the specified markdown section headers."
            )
        elif cond_id == "C2":
            system_instruction = (
                "You are an expert scientific information extraction assistant. "
                "Extract scientific entities (Phenomenon, Model, Method, Capability, Assumption, Limitation) "
                "conforming to the structured JSON schema. Do not include evidence quotes or offsets."
            )
            prompt = (
                f"Document Title: {title}\n\n"
                f"Source Text:\n{text_content}\n\n"
                "Extract structured knowledge complying strictly with C2Record JSON schema."
            )
        elif cond_id == "C3":
            system_instruction = (
                "You are an expert scientific information extraction assistant. "
                "Extract scientific entities (Phenomenon, Model, Method, Capability, Assumption, Limitation). "
                "For every entity, provide exact verbatim quotes and character offsets in the evidence span. "
                "If input/output data types for Capability are not explicitly mentioned in the text, "
                "set is_unspecified=true, data_type_l1='unknown', and evidence=null."
            )
            prompt = (
                f"Document Title: {title}\n\n"
                f"Source Text:\n{text_content}\n\n"
                "Extract structured knowledge with verbatim evidence complying strictly with C3Record JSON schema."
            )
        else:  # C4
            system_instruction = (
                "You are an expert scientific information extraction assistant. "
                "Extract Phenomenon, Model, Method, Capability, Assumption, Limitation entities and Local Relations. "
                "For every entity and relation, provide exact verbatim quotes from the text with character offsets. "
                "If input/output data types for Capability are not explicitly mentioned in the text, "
                "set is_unspecified=true, data_type_l1='unknown', and evidence=null."
            )
            prompt = (
                f"Document Title: {title}\n\n"
                f"Source Text:\n{text_content}\n\n"
                "Extract complete structured knowledge and relations complying strictly with ExtractedKnowledgeRecord JSON schema."
            )

        # 1. Pre-call Token Count
        counted_tokens = transport.count_tokens(
            model=active_model,
            contents=prompt,
            system_instruction=system_instruction,
            response_schema=schema_cls,
        )

        res_id = f"smoke_{cond_id}_{source_id}"
        reserved_cost = tracker.reserve(
            reservation_id=res_id,
            model_id=active_model,
            estimated_input_tokens=counted_tokens,
            max_output_tokens=2048,
            allowance_tokens=0,
        )
        print(f"[Step 1: Reservation] Pre-call In-Tokens: {counted_tokens}, Reserved Cost: ${reserved_cost:.6f}")

        # 2. Live API Execution
        t0 = time.time()
        raw_text, http_status, usage = transport.generate(
            model=active_model,
            prompt=prompt,
            system_instruction=system_instruction,
            condition_id=cond_id,
            response_schema=schema_cls,
            temperature=0.0,
            seed=42,
        )
        elapsed = time.time() - t0

        actual_in = usage.get("prompt_tokens", 0)
        actual_out = usage.get("completion_tokens", 0)
        actual_cost = tracker.settle(
            reservation_id=res_id,
            actual_input_tokens=actual_in,
            actual_output_tokens=actual_out,
        )

        print(f"[Step 2: Execution] {elapsed:.2f}s | HTTP {http_status}")
        print(f"       Actual In: {actual_in}, Actual Out: {actual_out}, Cost: ${actual_cost:.6f}")

        # Verification of Pre-call Reservation
        token_covered = (counted_tokens >= actual_in)
        cost_covered = (reserved_cost >= actual_cost)
        print(f"       Token Reservation Check: {counted_tokens} >= {actual_in} -> {'PASS' if token_covered else 'FAIL'} (delta: +{counted_tokens - actual_in})")
        print(f"       Cost Reservation Check:  ${reserved_cost:.6f} >= ${actual_cost:.6f} -> {'PASS' if cost_covered else 'FAIL'} (delta: +${reserved_cost - actual_cost:.6f})")

        if not token_covered or not cost_covered:
            print(f"[ERROR] Reservation underestimation detected in {cond_id}!")
            return 1

        # 3. Raw Artifact Preservation
        raw_artifact = RawExtractionArtifact(
            extraction_run_id=f"smoke_{cond_id}_01",
            source_id=source_id,
            version_id=doc["version_id"],
            condition_id=cond_id,  # type: ignore
            raw_response_text=raw_text,
            http_status=http_status,
            token_usage=usage,
            executed_at="2026-11-15T00:00:00Z",
            model_checkpoint=active_model,
            prompt_hash=compute_canonical_hash(prompt),
            target_field_hashes=doc["target_field_hashes"],
            json_parse_status="success" if cond_id != "C1" else "not_applicable",
            parsed_candidates=json.loads(raw_text) if cond_id != "C1" else None,
        )

        # 4. Standardization & Evidence Validation
        candidates = []
        rejections = []
        if cond_id == "C1":
            candidates, success, err = StandardizationConverter.convert_c1_free_text(raw_artifact)
            print(f"[Step 3: Conversion] C1 Free-Text Conversion: success={success}, candidates={len(candidates)}")
        elif cond_id == "C2":
            candidates = StandardizationConverter.convert_c2_structured_no_evidence(raw_artifact)
            print(f"[Step 3: Conversion] C2 No-Evidence Conversion: candidates={len(candidates)}")
        else:  # C3, C4
            candidates, rejections = StandardizationConverter.convert_c3_c4_with_evidence(raw_artifact)
            print(f"[Step 3: Conversion] {cond_id} Conversion: candidates={len(candidates)}, syntax rejections={len(rejections)}")

        # Evidence validation for C3 / C4
        validated_candidates = []
        if cond_id in ("C3", "C4"):
            for c in candidates:
                if c.evidence:
                    valid, reason, resolved_ev = validate_evidence(c.evidence, canonical_fields)
                    if valid and resolved_ev:
                        c.evidence = resolved_ev
                        validated_candidates.append(c)
                        print(f"       [EVIDENCE ACCEPTED] {c.entity_type}: '{c.name_or_statement}' (Quote: '{resolved_ev.verbatim_quote[:40]}...', status: {resolved_ev.offset_resolution_status})")
                    else:
                        print(f"       [EVIDENCE REJECTED] {c.entity_type}: '{c.name_or_statement}', reason: {reason}")
                else:
                    print(f"       [EVIDENCE MISSING] {c.entity_type}: '{c.name_or_statement}'")
        else:
            validated_candidates = candidates

        # 5. Matching against Gold Standard (Single-annotator reviewed benchmark)
        gold_entities = doc.get("entities", [])
        match_result = match_entities_bipartite(
            gold_entities=gold_entities,
            predicted_candidates=validated_candidates,
        )
        tp = match_result.true_positives
        fp = match_result.false_positives
        fn = match_result.false_negatives
        p = match_result.precision
        r = match_result.recall
        f1 = match_result.f1
        print(f"[Step 4: Benchmark Match] TP: {tp}, FP: {fp}, FN: {fn} -> P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}")

        results.append({
            "condition": cond_id,
            "source_id": source_id,
            "target_field": target_field,
            "input_chars": len(text_content),
            "reserved_in_tokens": counted_tokens,
            "actual_in_tokens": actual_in,
            "in_token_delta": counted_tokens - actual_in,
            "reserved_cost": reserved_cost,
            "actual_cost": actual_cost,
            "candidates_count": len(candidates),
            "accepted_count": len(validated_candidates),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    # Summary
    print("\n" + "="*80)
    print("=== FINAL C1~C4 VERIFICATION SUMMARY ===")
    print("| Cond | Doc ID | In Chars | Resv In | Act In | Delta | Resv Cost | Act Cost | Valid Cands | P | R | F1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['condition']} | {r['source_id']} | {r['input_chars']} | {r['reserved_in_tokens']} | {r['actual_in_tokens']} | +{r['in_token_delta']} | ${r['reserved_cost']:.5f} | ${r['actual_cost']:.5f} | {r['accepted_count']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} |")

    summary = tracker.get_summary()
    print("\nBudget Status:")
    print(f"Total spent across all tests: ${summary['spent_usd']:.6f} USD")
    print(f"Remaining budget: ${summary['remaining_usd']:.6f} USD")
    print("Exporting ledger table...")
    tracker.export_ledger_jsonl(Path("data/budget_ledger.jsonl"))
    print("\n" + tracker.format_ledger_table())
    return 0


if __name__ == "__main__":
    sys.exit(run_condition_smoke())
