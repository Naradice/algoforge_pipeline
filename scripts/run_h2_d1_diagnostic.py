"""HDE Step H2-D1 Diagnostic Experiment Runner.

Executes 4 concept archetypes x 3 conditions (D1-C0, D1-C1, D1-C2) = 12 total calls.
Conditions:
- D1-C0: Raw Literature + Natural Language Relational Facts (Baseline)
- D1-C1: Raw Literature + Structured Knowledge Graph Paths (Identical Facts)
- D1-C2: Raw Literature + Structured KG Paths + Mathematical IO, State/Control Variables & Assumptions

Blinded evaluation: Evaluator grades strictly against Independent Gold Specifications without seeing condition_id.
Primary Output: 5 Objective Error Flags (E_role, E_math, E_data, E_leap, E_inop) and Operability per call.
"""

import os
import json
import uuid
import datetime
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from research_pipeline.discovery.models import HypothesisCandidate, AlgoForgeBacktestSpec
from research_pipeline.discovery.diagnostic_models import DiagnosticEvaluationItem
from research_pipeline.discovery.diagnostic_generator import (
    CONCEPTS,
    ConceptSpecification,
    build_diagnostic_prompt,
    DIAGNOSTIC_SYSTEM_INSTRUCTION,
)
from research_pipeline.discovery.diagnostic_evaluator import evaluate_diagnostic_hypothesis
from research_pipeline.extractors.budget_tracker import BudgetExceededError

INPUT_TOKEN_PRICE_PER_M = 0.150   # $0.150 / 1M tokens
OUTPUT_TOKEN_PRICE_PER_M = 0.600  # $0.600 / 1M tokens
HARD_BUDGET_CEILING_USD = 10.00
HISTORICAL_CUMULATIVE_BASE_USD = 0.158341  # Reconciled base: $0.153190 (Phase 3) + $0.005151 (H2 pilot)


def load_fixture_docs(fixture_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(fixture_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {d["source_id"]: d for d in docs}


def record_budget_usage(ledger_path: Path, call_id: str, model_id: str, in_tokens: int, out_tokens: int) -> float:
    cost = (in_tokens / 1_000_000.0) * INPUT_TOKEN_PRICE_PER_M + (out_tokens / 1_000_000.0) * OUTPUT_TOKEN_PRICE_PER_M
    cum_spent = HISTORICAL_CUMULATIVE_BASE_USD
    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "h2_d1_" in record.get("call_id", ""):
                            cum_spent = record.get("cumulative_spent_usd", cum_spent)
                    except Exception:
                        pass
    new_cum = cum_spent + cost
    rem = HARD_BUDGET_CEILING_USD - new_cum
    if new_cum > HARD_BUDGET_CEILING_USD:
        raise BudgetExceededError(f"Budget ceiling exceeded: {new_cum:.6f} > {HARD_BUDGET_CEILING_USD}")

    entry = {
        "call_id": call_id,
        "model_id": model_id,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "cost_usd": round(cost, 6),
        "cumulative_spent_usd": round(new_cum, 6),
        "remaining_budget_usd": round(rem, 6),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return cost


def execute_llm_hypothesis_call(
    prompt: str,
    system_instruction: str,
    model_name: str = "gpt-4o-mini-2024-07-18",
) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        seed=42,
    )

    usage = response.usage
    raw_content = response.choices[0].message.content
    parsed_json = json.loads(raw_content)

    return {
        "content": parsed_json,
        "in_tokens": usage.prompt_tokens,
        "out_tokens": usage.completion_tokens,
    }


def run_h2_d1_diagnostic(
    fixture_path: Path,
    output_results_path: Path,
    ledger_path: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute the 4 concepts x 3 conditions = 12 call diagnostic experiment."""
    docs_map = load_fixture_docs(fixture_path)
    conditions = ["D1-C0", "D1-C1", "D1-C2"]

    results = []
    total_cost_usd = 0.0

    for concept_id, concept in CONCEPTS.items():
        doc = docs_map.get(concept.doc_id, {})

        for cond_id in conditions:
            call_id = f"h2_d1_{concept_id}_{cond_id.lower().replace('-', '_')}"
            print(f"Executing [{call_id}] (Concept: {concept.concept_name}, Condition: {cond_id})...")

            prompt = build_diagnostic_prompt(concept, cond_id, doc)
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

            if dry_run:
                # Deterministic mock candidate for test coverage
                cand = HypothesisCandidate(
                    hypothesis_id=f"hyp_{call_id}",
                    condition_id=cond_id,
                    task_id=f"d1_{concept_id}",
                    title=f"Diagnostic Test Hypothesis for {concept.concept_name}",
                    target_phenomenon="Market execution inefficiency",
                    proposed_mechanism="Formal mechanism derived from literature",
                    analogical_source=concept.concept_name,
                    claim="Explicit modeling of control variables prevents slippage",
                    grounding_rationale="Literature grounding aligned with independent Gold specifications",
                    testable_falsification_condition="Sharpe ratio fails to exceed 1.0 over 6-month out-of-sample testing",
                    backtest_spec=AlgoForgeBacktestSpec(
                        target_asset="USDJPY",
                        timeframe="M5",
                        feature_engineering="Transient impact kernel decay rate",
                        entry_condition="Initiate liquidation order for X_0 shares when signal triggers",
                        exit_condition="Cease execution when remaining inventory x(t) = 0",
                        risk_management="Max execution slippage threshold 15 bps",
                        benchmark_baseline="TWAP liquidation baseline",
                        falsification_metric="Sharpe ratio < 1.0 or slippage > 15 bps",
                    )
                )
                in_tokens, out_tokens = 600, 350
                cost = 0.0
            else:
                resp = execute_llm_hypothesis_call(prompt, DIAGNOSTIC_SYSTEM_INSTRUCTION)
                cand_data = resp["content"]
                cand_data["condition_id"] = cond_id
                cand_data["task_id"] = f"d1_{concept_id}"
                if "hypothesis_id" not in cand_data:
                    cand_data["hypothesis_id"] = f"hyp_{call_id}"
                cand = HypothesisCandidate.model_validate(cand_data)

                in_tokens = resp["in_tokens"]
                out_tokens = resp["out_tokens"]
                cost = record_budget_usage(ledger_path, call_id, "gpt-4o-mini-2024-07-18", in_tokens, out_tokens)
                total_cost_usd += cost

            # Blinded Evaluation against Independent Gold Spec
            eval_item = evaluate_diagnostic_hypothesis(cand, concept_id)

            record = {
                "call_id": call_id,
                "concept_id": concept_id,
                "concept_name": concept.concept_name,
                "condition_id": cond_id,
                "prompt_sha256": prompt_sha256,
                "candidate": cand.model_dump(),
                "evaluation": eval_item.model_dump(),
                "tokens": {"input": in_tokens, "output": out_tokens},
                "cost_usd": cost,
            }
            results.append(record)

    summary = {
        "experiment_id": "HDE_H2_D1_Diagnostic",
        "total_calls": len(results),
        "total_cost_usd": total_cost_usd,
        "reconciled_cumulative_spent_usd": round(HISTORICAL_CUMULATIVE_BASE_USD + total_cost_usd, 6),
        "reconciled_remaining_budget_usd": round(HARD_BUDGET_CEILING_USD - (HISTORICAL_CUMULATIVE_BASE_USD + total_cost_usd), 6),
        "results": results,
    }

    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    fix = root / "tests" / "fixtures" / "gold_standard" / "holdout_expansion_ground_truth.json"
    out = root / "data" / "h2_d1_diagnostic_results.json"
    led = root / "data" / "budget_ledger.jsonl"

    res = run_h2_d1_diagnostic(fix, out, led, dry_run=args.dry_run)
    print(f"Completed {res['total_calls']} diagnostic calls. Cost: ${res['total_cost_usd']:.6f} USD.")
