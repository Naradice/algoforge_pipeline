"""Multi-Provider Smoke Test for Gemini and OpenAI on 1 Development Document.

Supports both:
- Google GenAI (gemini-2.5-flash)
- OpenAI (gpt-4o-mini or gpt-4o)

Usage:
  python scripts/smoke_test_llm.py --provider auto
  python scripts/smoke_test_llm.py --provider gemini
  python scripts/smoke_test_llm.py --provider openai
"""

import sys
import os
import json
import argparse
from pathlib import Path
import time

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from research_pipeline.config import load_project_env
from research_pipeline.extractors.budget_tracker import BudgetTracker
from research_pipeline.extractors.transport_factory import (
    create_transport,
    detect_configured_provider,
)

# Load .env from project hierarchy
load_project_env()
from research_pipeline.extractors.converter import StandardizationConverter
from research_pipeline.extractors.validator import validate_evidence
from research_pipeline.extractors.raw_artifact import (
    RawExtractionArtifact,
    RejectionRecord,
)
from research_pipeline.extractors.models import (
    ExtractedKnowledgeRecord,
    compute_canonical_hash,
)
from research_pipeline.evaluation.matching import match_entities_bipartite


def run_smoke_test(provider_choice: str = "auto", model_choice: str = "auto"):
    # 1. Detect provider
    detected_provider, default_model = detect_configured_provider()
    
    if provider_choice == "auto":
        target_provider = detected_provider
    else:
        target_provider = provider_choice

    if not target_provider:
        print("[AUTH_CHECK] Status: FAILED")
        print("[AUTH_CHECK] Error: Neither GEMINI_API_KEY nor OPENAI_API_KEY is configured in environment.")
        print("[AUTH_CHECK] Instruction: Please configure either:")
        print("    $env:GEMINI_API_KEY = '<your-key>'  (for Google Gemini)")
        print("  or")
        print("    $env:OPENAI_API_KEY = '<your-key>'  (for OpenAI)")
        return 2

    # Verify key for selected provider
    if target_provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            print("[AUTH_CHECK] Status: FAILED (GEMINI_API_KEY is not set)")
            return 2
        active_model = "gemini-2.5-flash" if model_choice == "auto" else model_choice
    elif target_provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("[AUTH_CHECK] Status: FAILED (OPENAI_API_KEY is not set)")
            return 2
        # Pin exact checkpoint version for reproducibility
        active_model = "gpt-4o-mini-2024-07-18" if model_choice == "auto" else model_choice
    else:
        print(f"Unknown provider: {target_provider}")
        return 1

    print(f"[AUTH_CHECK] Status: CONFIGURED")
    print(f"[PROVIDER]: {target_provider.upper()} (Pinned Model: {active_model}) - Key detected, value hidden")

    # 2. Load Development Document 1 (Minority Game)
    gt_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gold_standard" / "h1a_ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    doc1 = [d for d in ground_truth if d["source_id"] == "arxiv_2301_01001"][0]
    canonical_fields = doc1["target_fields"]
    abstract_text = canonical_fields["abstract_text"]

    print(f"[DOCUMENT] Target: {doc1['source_id']} ({doc1['target_fields']['title_text']})")

    # 3. Setup Budget Tracker & Transport
    tracker = BudgetTracker()
    transport = create_transport(provider=target_provider)

    system_instruction = (
        "You are an expert scientific information extraction assistant. "
        "Extract Phenomenon, Model, Method, Capability, Assumption, and Limitation entities. "
        "For each entity and relation, provide exact verbatim quotes from the text with character offsets. "
        "If input/output data types for Capability are not explicitly mentioned in the text, "
        "set is_unspecified=true, data_type_l1='unknown', and evidence=null."
    )

    prompt = (
        f"Document Title: {doc1['target_fields']['title_text']}\n\n"
        f"Abstract:\n{abstract_text}\n\n"
        "Extract structured knowledge complying strictly with ExtractedKnowledgeRecord schema."
    )

    # 4. Token Counting API Call / Reservation (Includes System Prompt, Contents, and Schema)
    print("\n--- Step 1: Pre-call Token Count & Reservation ---")
    counted_tokens = transport.count_tokens(
        model=active_model,
        contents=prompt,
        system_instruction=system_instruction,
        response_schema=ExtractedKnowledgeRecord,
    )
    print(f"Pre-call Conservative Token Count: {counted_tokens} tokens (includes system, prompt, and JSON Schema)")

    res_id = f"smoke_test:{doc1['source_id']}:{target_provider}"
    reserved_cost = tracker.reserve(
        reservation_id=res_id,
        model_id=active_model,
        estimated_input_tokens=counted_tokens,
        max_output_tokens=2048,
        allowance_tokens=0,  # Safety allowance is already factored into conservative count_tokens
    )
    print(f"Pre-call Reserved cost: ${reserved_cost:.6f} USD")

    # 5. Live API Execution
    print(f"\n--- Step 2: Live API Execution ({target_provider}) ---")
    start_time = time.time()
    try:
        raw_text, http_status, usage = transport.generate(
            model=active_model,
            prompt=prompt,
            system_instruction=system_instruction,
            condition_id="C4",
            response_schema=ExtractedKnowledgeRecord,
            temperature=0.0,
            seed=42,
        )
        latency = time.time() - start_time
        print(f"API Response received in {latency:.2f}s (HTTP {http_status})")
    except Exception as e:
        tracker.release_reservation(res_id)
        print(f"API Call Failed: {e}")
        return 1

    # 6. Post-call Settlement
    actual_in = usage.get("prompt_tokens", 0)
    actual_out = usage.get("completion_tokens", 0)
    actual_cost = tracker.settle(res_id, actual_in, actual_out)
    print(f"Actual tokens from API: {actual_in} in, {actual_out} out (Total: {actual_in + actual_out})")
    print(f"Post-call Settled cost: ${actual_cost:.6f} USD")

    # Verification of Budget Reservation Integrity (User Directive 1)
    print("\n--- Verification: Pre-call Reservation Integrity ---")
    token_delta = counted_tokens - actual_in
    cost_delta = reserved_cost - actual_cost
    token_check_pass = token_delta >= 0
    cost_check_pass = cost_delta >= 0

    print(f"  [TOKEN CHECK] Reserved Input: {counted_tokens} >= Actual Input: {actual_in} -> {'PASS' if token_check_pass else 'FAIL'} (Buffer: +{token_delta} tokens)")
    print(f"  [COST CHECK] Reserved Cost: ${reserved_cost:.6f} >= Actual Cost: ${actual_cost:.6f} -> {'PASS' if cost_check_pass else 'FAIL'} (Buffer: +${cost_delta:.6f})")

    if not token_check_pass or not cost_check_pass:
        print("\n[FATAL ERROR] Pre-call reservation failed to cover actual usage! Aborting.")
        return 1

    # 7. Raw Artifact Preservation
    raw_artifact = RawExtractionArtifact(
        extraction_run_id=f"smoke_{target_provider}_01",
        source_id=doc1["source_id"],
        version_id=doc1["version_id"],
        condition_id="C4",
        raw_response_text=raw_text,
        http_status=http_status,
        token_usage=usage,
        executed_at="2026-11-15T00:00:00Z",
        model_checkpoint=active_model,
        prompt_hash=compute_canonical_hash(prompt),
        target_field_hashes=doc1["target_field_hashes"],
        json_parse_status="success",
        parsed_candidates=json.loads(raw_text),
    )

    # 8. Convert & Validate
    print("\n--- Step 3: Standardization & Evidence Validation ---")
    candidates, rejections = StandardizationConverter.convert_c3_c4_with_evidence(raw_artifact)
    print(f"Parsed candidates count: {len(candidates)}")
    print(f"Schema syntax rejections: {len(rejections)}")

    validated_candidates = []
    for c in candidates:
        if c.evidence:
            valid, reason, resolved_ev = validate_evidence(c.evidence, canonical_fields)
            if valid and resolved_ev:
                c.evidence = resolved_ev
                validated_candidates.append(c)
                print(f"  [ACCEPTED] {c.entity_type}: '{c.name_or_statement}' (Quote: '{resolved_ev.verbatim_quote}', status: {resolved_ev.offset_resolution_status})")
            else:
                rejections.append(
                    RejectionRecord(
                        local_id=c.candidate_local_id,
                        entity_type=c.entity_type,
                        rejection_stage="quote_validity",
                        rejection_reason=reason,
                        raw_entity_data=c.raw_payload,
                    )
                )
                print(f"  [REJECTED] {c.entity_type}: '{c.name_or_statement}' Reason: {reason}")

    # 9. Bipartite Matching against Gold
    print("\n--- Step 4: Bipartite Matching against Gold Standard ---")
    match_result = match_entities_bipartite(doc1["entities"], validated_candidates)
    print(f"True Positives: {match_result.true_positives}")
    print(f"False Positives: {match_result.false_positives}")
    print(f"False Negatives: {match_result.false_negatives}")
    print(f"Precision: {match_result.precision:.4f}, Recall: {match_result.recall:.4f}, F1: {match_result.f1:.4f}")

    print("\n--- Step 5: Final Budget Status & Per-Request Ledger ---")
    summary = tracker.get_summary()
    print(f"Max Budget: ${summary['max_budget_usd']:.2f} USD")
    print(f"Total Spent: ${summary['spent_usd']:.6f} USD")
    print(f"Remaining: ${summary['remaining_usd']:.6f} USD")
    print(f"Total Tokens: {summary['spent_tokens']:,}")
    print("\n[Per-Request Ledger Table]:")
    print(tracker.format_ledger_table())

    # Export ledger to file
    ledger_path = Path(__file__).resolve().parent.parent / "data" / "budget_ledger.jsonl"
    tracker.export_ledger_jsonl(ledger_path)
    print(f"\nLedger exported to: {ledger_path}")
    print(f"\n[SMOKE TEST RESULT]: SUCCESS ({target_provider.upper()})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="auto", choices=["auto", "gemini", "openai"])
    parser.add_argument("--model", default="auto")
    args = parser.parse_args()

    code = run_smoke_test(args.provider, args.model)
    sys.exit(code)
