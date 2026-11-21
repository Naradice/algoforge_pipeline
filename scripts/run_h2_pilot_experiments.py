"""HDE Step H2: Revised Pilot Comparative Experiment Runner.

Executes 3 independent tasks x 4 experimental conditions (H2-C0, H2-C1, H2-C2, H2-C3) = 12 total calls.
Conditions:
- H2-C0: Raw Literature Baseline
- H2-C1: Literature + Identical Facts in Unstructured Natural Language
- H2-C2: Literature + Identical Facts Structured as Knowledge Graph Paths
- H2-C3: Literature + Structured KG Paths + Verbatim Textual Evidence

Blinded evaluation: Evaluator cannot see condition_id.
Disqualification gates: Hard disqualification for unfalsifiable claims, lack of quantitative metrics, or vacuous specs.
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

from research_pipeline.discovery.models import (
    HypothesisCandidate,
    HypothesisEvaluationResult,
    AlgoForgeBacktestSpec,
)
from research_pipeline.discovery.hypothesis_generator import (
    TASKS,
    TaskSpecification,
    build_task_prompt,
    H2_SYSTEM_INSTRUCTION,
)
from research_pipeline.discovery.evaluator import evaluate_hypothesis_blind
from research_pipeline.extractors.budget_tracker import BudgetExceededError

# Cost calculation constants for gpt-4o-mini-2024-07-18
INPUT_TOKEN_PRICE_PER_M = 0.150  # $0.150 / 1M tokens
OUTPUT_TOKEN_PRICE_PER_M = 0.600  # $0.600 / 1M tokens
HARD_BUDGET_CEILING_USD = 10.00


def load_fixture_docs(fixture_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(fixture_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {d["source_id"]: d for d in docs}


def record_budget_usage(ledger_path: Path, call_id: str, model_id: str, in_tokens: int, out_tokens: int) -> float:
    cost = (in_tokens / 1_000_000.0) * INPUT_TOKEN_PRICE_PER_M + (out_tokens / 1_000_000.0) * OUTPUT_TOKEN_PRICE_PER_M
    cum_spent = 0.0
    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        cum_spent = record.get("cumulative_spent_usd", cum_spent)
                    except Exception:
                        pass
    new_cum = cum_spent + cost
    rem = HARD_BUDGET_CEILING_USD - new_cum
    if new_cum > HARD_BUDGET_CEILING_USD:
        raise BudgetExceededError(f"Budget exceeded: {new_cum:.6f} > {HARD_BUDGET_CEILING_USD}")

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
    """Execute live OpenAI API call requesting structured JSON object."""
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


def run_h2_pilot_experiments(
    fixture_path: Path,
    output_results_path: Path,
    ledger_path: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute the full 3x4 = 12 call pilot experiment matrix."""
    docs_map = load_fixture_docs(fixture_path)
    conditions = ["H2-C0", "H2-C1", "H2-C2", "H2-C3"]
    
    experiment_results = []
    total_cost_usd = 0.0

    for task_id, task in TASKS.items():
        task_docs = [docs_map[sid] for sid in task.doc_ids if sid in docs_map]

        for cond_id in conditions:
            call_id = f"h2_{task_id}_{cond_id.lower().replace('-', '_')}"
            print(f"Executing [{call_id}] (Task: {task.task_name}, Condition: {cond_id})...")

            prompt = build_task_prompt(task, cond_id, task_docs)

            if dry_run:
                # Mock generation for dry-run validation
                cand = HypothesisCandidate(
                    hypothesis_id=f"hyp_{call_id}",
                    condition_id=cond_id,
                    task_id=task_id,
                    title=f"Mock Hypothesis for {task.task_name}",
                    target_phenomenon="Market inefficiency",
                    proposed_mechanism="Mathematical mechanism derived from literature",
                    analogical_source="Point process / optimization",
                    claim="Mechanism improves risk-adjusted returns",
                    grounding_rationale="Grounded in referenced papers",
                    testable_falsification_condition="Sharpe ratio fails to exceed 1.5 with p < 0.05",
                    backtest_spec=AlgoForgeBacktestSpec(
                        target_asset="USDJPY",
                        timeframe="M5",
                        feature_engineering="Feature extraction",
                        entry_condition="Enter on signal > threshold",
                        exit_condition="Exit at 1.5x ATR stop",
                        risk_management="2% risk limit",
                        benchmark_baseline="TWAP baseline",
                        falsification_metric="T-stat < 2.0",
                    )
                )
                in_tokens, out_tokens = 500, 300
                cost = 0.0
            else:
                resp = execute_llm_hypothesis_call(prompt, H2_SYSTEM_INSTRUCTION)
                cand_data = resp["content"]
                # Enforce metadata
                cand_data["condition_id"] = cond_id
                cand_data["task_id"] = task_id
                if "hypothesis_id" not in cand_data:
                    cand_data["hypothesis_id"] = f"hyp_{call_id}"
                cand = HypothesisCandidate.model_validate(cand_data)

                in_tokens = resp["in_tokens"]
                out_tokens = resp["out_tokens"]
                cost = record_budget_usage(ledger_path, call_id, "gpt-4o-mini-2024-07-18", in_tokens, out_tokens)
                total_cost_usd += cost

            # Blinded Evaluation
            eval_result = evaluate_hypothesis_blind(cand, reference_facts=task.facts_nl)

            record = {
                "call_id": call_id,
                "task_id": task_id,
                "task_name": task.task_name,
                "condition_id": cond_id,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "candidate": cand.model_dump(),
                "evaluation": eval_result.model_dump(),
                "tokens": {"input": in_tokens, "output": out_tokens},
                "cost_usd": cost,
            }
            experiment_results.append(record)

    summary = {
        "total_calls": len(experiment_results),
        "total_cost_usd": total_cost_usd,
        "results": experiment_results,
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

    pipeline_root = Path(__file__).resolve().parent.parent
    fixture = pipeline_root / "tests" / "fixtures" / "gold_standard" / "holdout_expansion_ground_truth.json"
    results_out = pipeline_root / "data" / "h2_pilot_results.json"
    ledger = pipeline_root / "data" / "budget_ledger.jsonl"

    res = run_h2_pilot_experiments(fixture, results_out, ledger, dry_run=args.dry_run)
    print(f"Completed {res['total_calls']} calls. Total cost: ${res['total_cost_usd']:.6f} USD.")
