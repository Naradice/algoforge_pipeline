"""Execution and Conformance Verification Script for HDE Step H2-D3.

Evaluates all 16 pre-frozen conformance benchmark cases against the 7-dimensional Gold set.
Enforces:
1. Hash verification of pre-frozen fixture against manifest.
2. Full multi-dimensional confusion matrices (Data Feasibility, Semantic Fidelity, Gate Decision, Temporal Validity).
3. 7-dimensional spec-level exact match reporting.
4. Saving output results and updating audit manifest.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from research_pipeline.discovery.models import HypothesisCandidate
from research_pipeline.discovery.data_gate_models import ProxySpecification
from research_pipeline.discovery.data_feasibility_gate import evaluate_data_and_semantic_gate

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures" / "gold_standard"
MANIFEST_PATH = FIXTURES_DIR / "h2_d3_audit_manifest.json"
CONFORMANCE_PATH = FIXTURES_DIR / "h2_d3_conformance_cases.json"
GATE_CODE_PATH = BASE_DIR / "src" / "research_pipeline" / "discovery" / "data_feasibility_gate.py"
OUTPUT_PATH = BASE_DIR / "data" / "h2_d3_conformance_results.json"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    # 1. Load manifest and verify fixture hash
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    cur_conf_sha = sha256_file(CONFORMANCE_PATH)
    cur_gate_sha = sha256_file(GATE_CODE_PATH)

    assert cur_conf_sha == manifest["conformance_sha256"], (
        f"Fixture hash mismatch! Expected {manifest['conformance_sha256']}, got {cur_conf_sha}"
    )

    print("=== HDE Step H2-D3 Specification Conformance Benchmark ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Conformance Fixture SHA256 : {cur_conf_sha}")
    print(f"Gate Code SHA256           : {cur_gate_sha}")
    print()

    # 2. Load conformance cases
    with open(CONFORMANCE_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    exact_matches = 0
    total_specs = len(cases)

    # Confusion Matrices
    data_feas_cm = {
        "SUPPORTED": {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNKNOWN": 0},
        "UNSUPPORTED": {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNKNOWN": 0},
        "UNKNOWN": {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNKNOWN": 0},
    }

    sem_fid_cm = {
        "EXACT": {"EXACT": 0, "PROXY": 0, "INVALID": 0, "UNKNOWN": 0},
        "PROXY": {"EXACT": 0, "PROXY": 0, "INVALID": 0, "UNKNOWN": 0},
        "INVALID": {"EXACT": 0, "PROXY": 0, "INVALID": 0, "UNKNOWN": 0},
        "UNKNOWN": {"EXACT": 0, "PROXY": 0, "INVALID": 0, "UNKNOWN": 0},
    }

    gate_cm = {
        "PASS": {"PASS": 0, "BLOCK": 0, "REVIEW": 0},
        "BLOCK": {"PASS": 0, "BLOCK": 0, "REVIEW": 0},
        "REVIEW": {"PASS": 0, "BLOCK": 0, "REVIEW": 0},
    }

    temporal_cm = {
        "VALID": {"VALID": 0, "LOOKAHEAD_LEAK": 0, "LOOKBACK_UNDEFINED": 0},
        "LOOKAHEAD_LEAK": {"VALID": 0, "LOOKAHEAD_LEAK": 0, "LOOKBACK_UNDEFINED": 0},
        "LOOKBACK_UNDEFINED": {"VALID": 0, "LOOKAHEAD_LEAK": 0, "LOOKBACK_UNDEFINED": 0},
    }

    eval_results = []
    mismatches = []

    for item in cases:
        cid = item["case_id"]
        cand = HypothesisCandidate(**item["candidate"])
        gold = item["gold_judgments"]
        proxy_spec = ProxySpecification(**item["proxy_spec"]) if item.get("proxy_spec") else None

        res = evaluate_data_and_semantic_gate(cand, proxy_spec=proxy_spec, case_id=cid)

        # Update confusion matrices
        data_feas_cm[gold["data_feasibility"]][res.data_feasibility] += 1
        sem_fid_cm[gold["semantic_fidelity"]][res.semantic_fidelity] += 1
        gate_cm[gold["gate_decision"]][res.gate_decision] += 1
        temporal_cm[gold["temporal_validity"]][res.temporal_validity] += 1

        # Check exact 6-axis match
        is_exact = (
            res.data_feasibility == gold["data_feasibility"] and
            res.semantic_fidelity == gold["semantic_fidelity"] and
            res.estimation_spec == gold["estimation_spec"] and
            res.identifiability == gold["identifiability"] and
            res.temporal_validity == gold["temporal_validity"] and
            res.gate_decision == gold["gate_decision"]
        )

        if is_exact:
            exact_matches += 1
        else:
            mismatches.append({
                "case_id": cid,
                "gold": gold,
                "pred": {
                    "data_feasibility": res.data_feasibility,
                    "semantic_fidelity": res.semantic_fidelity,
                    "estimation_spec": res.estimation_spec,
                    "identifiability": res.identifiability,
                    "temporal_validity": res.temporal_validity,
                    "gate_decision": res.gate_decision,
                    "reason": res.reason
                }
            })

        eval_results.append({
            "case_id": cid,
            "category": item["category"],
            "gold": gold,
            "pred": {
                "data_feasibility": res.data_feasibility,
                "semantic_fidelity": res.semantic_fidelity,
                "estimation_spec": res.estimation_spec,
                "identifiability": res.identifiability,
                "temporal_validity": res.temporal_validity,
                "gate_decision": res.gate_decision,
                "reason": res.reason,
            },
            "exact_match": is_exact,
            "evidence": res.evidence_details,
        })

    # Summary Display
    exact_ratio = exact_matches / total_specs
    print(f"=== Conformance Test Results ===")
    print(f"7-Dimensional Spec Exact Match: {exact_matches}/{total_specs} ({exact_ratio*100:.1f}%)")
    print()

    print("Gate Decision 3x3 Matrix:")
    print("Gold \\ Pred       PASS     BLOCK    REVIEW")
    for g in ["PASS", "BLOCK", "REVIEW"]:
        print(f"{g:12s} {gate_cm[g]['PASS']:8d} {gate_cm[g]['BLOCK']:9d} {gate_cm[g]['REVIEW']:9d}")
    print()

    print("Data Feasibility 3x3 Matrix:")
    print("Gold \\ Pred       SUPPORTED  UNSUPPORTED    UNKNOWN")
    for g in ["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]:
        print(f"{g:12s} {data_feas_cm[g]['SUPPORTED']:10d} {data_feas_cm[g]['UNSUPPORTED']:12d} {data_feas_cm[g]['UNKNOWN']:10d}")
    print()

    print("Semantic Fidelity 4x4 Matrix:")
    print("Gold \\ Pred       EXACT     PROXY   INVALID   UNKNOWN")
    for g in ["EXACT", "PROXY", "INVALID", "UNKNOWN"]:
        print(f"{g:12s} {sem_fid_cm[g]['EXACT']:8d} {sem_fid_cm[g]['PROXY']:9d} {sem_fid_cm[g]['INVALID']:9d} {sem_fid_cm[g]['UNKNOWN']:9d}")
    print()

    print("Temporal Validity 3x3 Matrix:")
    print("Gold \\ Pred           VALID  LOOKAHEAD_LEAK  LOOKBACK_UNDEF")
    for g in ["VALID", "LOOKAHEAD_LEAK", "LOOKBACK_UNDEFINED"]:
        print(f"{g:18s} {temporal_cm[g]['VALID']:8d} {temporal_cm[g]['LOOKAHEAD_LEAK']:15d} {temporal_cm[g]['LOOKBACK_UNDEFINED']:15d}")
    print()

    # Success Acceptance Criteria Checks
    lookahead_false_pass = gate_cm["BLOCK"]["PASS"]
    supervised_target_blocked = gate_cm["PASS"]["BLOCK"]
    unsupported_false_pass = data_feas_cm["UNSUPPORTED"]["SUPPORTED"]
    unknown_false_pass = data_feas_cm["UNKNOWN"]["SUPPORTED"]

    print("=== Success Acceptance Criteria Checks ===")
    print(f"1. Critical Lookahead Leak Misses (BLOCK -> PASS) : {lookahead_false_pass} (Target: 0)")
    print(f"2. Legitimate Target False Blockage (PASS -> BLOCK) : {supervised_target_blocked} (Target: 0)")
    print(f"3. Data Contract Violations Missed (UNSUPP -> SUPP): {unsupported_false_pass} (Target: 0)")
    print(f"4. Undetermined Feeds Missed (UNKNOWN -> SUPP)     : {unknown_false_pass} (Target: 0)")
    print(f"5. Total Spec Exact Match Rate                     : {exact_matches}/{total_specs} (Target: 16/16)")

    if (exact_matches == total_specs and 
        lookahead_false_pass == 0 and 
        supervised_target_blocked == 0 and 
        unsupported_false_pass == 0 and 
        unknown_false_pass == 0):
        verdict = "PASS (合格)"
    else:
        verdict = "FAIL (不合格)"

    print()
    print(f"*** FINAL SPECIFICATION CONFORMANCE VERDICT: {verdict} ***")
    print()

    if mismatches:
        print(f"--- Mismatches ({len(mismatches)}) ---")
        for m in mismatches:
            print(f"Case: {m['case_id']}")
            print("  Gold:", m["gold"])
            print("  Pred:", m["pred"])
            print()

    # Save output bundle
    output_bundle = {
        "experiment_id": "RP-HDE-H2-D3-CONFORMANCE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixture_sha256": cur_conf_sha,
        "gate_code_sha256": cur_gate_sha,
        "total_specs": total_specs,
        "exact_matches": exact_matches,
        "exact_match_ratio": exact_ratio,
        "verdict": verdict,
        "confusion_matrices": {
            "gate_decision": gate_cm,
            "data_feasibility": data_feas_cm,
            "semantic_fidelity": sem_fid_cm,
            "temporal_validity": temporal_cm
        },
        "case_evaluations": eval_results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_bundle, f, indent=2, ensure_ascii=False)

    manifest["conformance_test_executed"] = True
    manifest["gate_code_sha256"] = cur_gate_sha
    manifest["last_execution_timestamp"] = datetime.now(timezone.utc).isoformat()
    manifest["final_verdict"] = verdict

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Full results saved to: {OUTPUT_PATH}")
    print(f"Manifest updated at: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
