"""End-to-end smoke test for Gemini API on 1 Development Document (Section 9 & Phase 2 Checklist).

Verifies the complete pipeline on real API:
- GEMINI_API_KEY presence (never prints keys)
- countTokens estimation vs actual token usage
- Pre-call Reservation and Post-call Settlement
- Structured JSON output with Pydantic schema
- RawExtractionArtifact storage
- Deterministic quote offset resolution
- 3-tier validation (Raw, Accepted, Rejected)
- Hungarian bipartite matching and single-doc evaluation
"""

import sys
import os
import json
from pathlib import Path
import time

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from research_pipeline.extractors.budget_tracker import BudgetTracker
from research_pipeline.extractors.gemini_transport import GeminiTransport
from research_pipeline.extractors.converter import StandardizationConverter
from research_pipeline.extractors.validator import validate_evidence, validate_local_relation
from research_pipeline.extractors.raw_artifact import (
    RawExtractionArtifact,
    ExtractionTrioBundle,
    RejectionRecord,
)
from research_pipeline.extractors.models import (
    ExtractedKnowledgeRecord,
    CapabilityEntity,
    CapabilityIO,
    compute_canonical_hash,
)
from research_pipeline.evaluation.matching import match_entities_bipartite


def run_smoke_test():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[AUTH_CHECK] Status: FAILED")
        print("[AUTH_CHECK] Error: GEMINI_API_KEY is not set in environment.")
        print("[AUTH_CHECK] Instruction: Please configure $env:GEMINI_API_KEY='<your-key>' in PowerShell.")
        return 2

    print("[AUTH_CHECK] Status: CONFIGURED (Key detected, value hidden)")

    # 1. Load Development Document 1 (Minority Game)
    gt_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gold_standard" / "h1a_ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    doc1 = [d for d in ground_truth if d["source_id"] == "arxiv_2301_01001"][0]
    canonical_fields = doc1["target_fields"]
    abstract_text = canonical_fields["abstract_text"]

    print(f"[DOCUMENT] Target: {doc1['source_id']} ({doc1['target_fields']['title_text']})")

    # 2. Setup Budget Tracker & Transport
    tracker = BudgetTracker(max_usd=30.00, max_tokens=5_000_000)
    transport = GeminiTransport(api_key=api_key)
    model_name = "gemini-2.5-flash"

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

    # 3. Token Counting API Call
    print("\n--- Step 1: Pre-call Token Count & Reservation ---")
    counted_tokens = transport.count_tokens(
        model=model_name,
        contents=prompt,
        system_instruction=system_instruction,
    )
    print(f"API countTokens result: {counted_tokens} tokens")

    res_id = f"smoke_test:{doc1['source_id']}:C4"
    reserved_cost = tracker.reserve(
        reservation_id=res_id,
        model_id=model_name,
        estimated_input_tokens=counted_tokens,
        max_output_tokens=2048,
    )
    print(f"Pre-call Reserved cost: ${reserved_cost:.6f}")

    # 4. Generate Content Call
    print("\n--- Step 2: Live API Execution ---")
    start_time = time.time()
    try:
        raw_text, http_status, usage = transport.generate(
            model=model_name,
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

    # 5. Post-call Settlement
    actual_in = usage.get("prompt_tokens", counted_tokens)
    actual_out = usage.get("completion_tokens", 0)
    actual_cost = tracker.settle(res_id, actual_in, actual_out)
    print(f"Actual tokens: {actual_in} in, {actual_out} out (Total: {actual_in + actual_out})")
    print(f"Token count delta: {actual_in - counted_tokens} tokens")
    print(f"Post-call Settled cost: ${actual_cost:.6f}")

    # 6. Raw Artifact Preservation
    raw_artifact = RawExtractionArtifact(
        extraction_run_id="smoke_trial_01",
        source_id=doc1["source_id"],
        version_id=doc1["version_id"],
        condition_id="C4",
        raw_response_text=raw_text,
        http_status=http_status,
        token_usage=usage,
        executed_at="2026-11-15T00:00:00Z",
        model_checkpoint=model_name,
        prompt_hash=compute_canonical_hash(prompt),
        target_field_hashes=doc1["target_field_hashes"],
        json_parse_status="success",
        parsed_candidates=json.loads(raw_text),
    )

    # 7. Convert & Validate
    print("\n--- Step 3: Standardization & Evidence Validation ---")
    candidates, rejections = StandardizationConverter.convert_c3_c4_with_evidence(raw_artifact)
    print(f"Parsed candidates count: {len(candidates)}")
    print(f"Schema syntax rejections: {len(rejections)}")

    accepted_entities = {}
    validated_candidates = []
    for c in candidates:
        if c.evidence:
            valid, reason, resolved_ev = validate_evidence(c.evidence, canonical_fields)
            if valid and resolved_ev:
                c.evidence = resolved_ev
                accepted_entities[c.candidate_local_id] = c.entity_type
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

    # 8. Evaluation Matching against Gold
    print("\n--- Step 4: Bipartite Matching against Gold Standard ---")
    match_result = match_entities_bipartite(doc1["entities"], validated_candidates)
    print(f"True Positives: {match_result.true_positives}")
    print(f"False Positives: {match_result.false_positives}")
    print(f"False Negatives: {match_result.false_negatives}")
    print(f"Precision: {match_result.precision:.4f}, Recall: {match_result.recall:.4f}, F1: {match_result.f1:.4f}")

    print("\n--- Step 5: Final Budget Status ---")
    summary = tracker.get_summary()
    print(f"Spent USD: ${summary['spent_usd']:.4f} / ${tracker.max_usd:.2f}")
    print(f"Spent Tokens: {summary['spent_tokens']} / {tracker.max_tokens}")
    print("\n[SMOKE TEST RESULT]: SUCCESS")
    return 0


if __name__ == "__main__":
    code = run_smoke_test()
    sys.exit(code)
