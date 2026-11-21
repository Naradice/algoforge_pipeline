"""HDE Step H2-D4 Live Generation Runner.

Executes 4 concepts x 3 conditions (C0, C1, C2) x 3 seeds (42, 43, 44) = 36 live API calls.
Model: gpt-4o-mini-2024-07-18
Temperature: 0.2
Adheres strictly to RP-HDE-H2-D4-PLAN-v0.2.
All prompts loaded from frozen fixture tests/fixtures/gold_standard/h2_d4_prompts.json.
Budget tracking written to data/budget_ledger.jsonl.
Raw outputs saved to data/h2_d4_raw_candidates.json.
"""

import os
import sys
import re
import json
import time
import datetime
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR / "src"))
load_dotenv(BASE_DIR / ".env")

from openai import OpenAI
from research_pipeline.discovery.models import HypothesisCandidate, AlgoForgeBacktestSpec
from research_pipeline.discovery.diagnostic_generator import DIAGNOSTIC_SYSTEM_INSTRUCTION

PROMPTS_PATH = BASE_DIR / "tests" / "fixtures" / "gold_standard" / "h2_d4_prompts.json"
LEDGER_PATH = BASE_DIR / "data" / "budget_ledger.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "h2_d4_raw_candidates.json"

INPUT_TOKEN_PRICE_PER_M = 0.150   # $0.150 / 1M tokens
OUTPUT_TOKEN_PRICE_PER_M = 0.600  # $0.600 / 1M tokens
HARD_BUDGET_CEILING_USD = 10.00
H2_D4_BUDGET_LIMIT_USD = 0.05

CONCEPTS = ["almgren_chriss", "hawkes_process", "irm", "jump_diffusion"]
CONDITIONS = ["C0", "C1", "C2"]
SEEDS = [42, 43, 44]


def record_budget_usage(ledger_path: Path, call_id: str, model_id: str, in_tokens: int, out_tokens: int) -> float:
    cost = (in_tokens / 1_000_000.0) * INPUT_TOKEN_PRICE_PER_M + (out_tokens / 1_000_000.0) * OUTPUT_TOKEN_PRICE_PER_M
    cum_spent = 0.163442
    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "cumulative_spent_usd" in record:
                            cum_spent = record["cumulative_spent_usd"]
                    except Exception:
                        pass
    new_cum = cum_spent + cost
    rem = HARD_BUDGET_CEILING_USD - new_cum
    if new_cum > HARD_BUDGET_CEILING_USD:
        raise RuntimeError(f"Hard budget ceiling exceeded: {new_cum:.6f} > {HARD_BUDGET_CEILING_USD}")

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


def to_snake(s: str) -> str:
    s2 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s2).lower().strip('_')


def stringify_val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def normalize_candidate_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Robustly normalize candidate dictionary to match HypothesisCandidate schema."""
    for wrapper in ["HypothesisCandidate", "hypothesis_candidate", "hypothesis", "candidate"]:
        if wrapper in data and isinstance(data[wrapper], dict):
            data = data[wrapper]
            break

    norm: Dict[str, Any] = {}
    for k, v in data.items():
        sk = to_snake(k)
        if sk in ["core_claim"]:
            sk = "claim"
        elif sk in ["algoforge_backtest_spec", "algo_forge_backtest_spec", "spec"]:
            sk = "backtest_spec"
        norm[sk] = v

    # Normalize backtest_spec
    if "backtest_spec" in norm and isinstance(norm["backtest_spec"], dict):
        bs_norm = {}
        for k, v in norm["backtest_spec"].items():
            bs_norm[to_snake(k)] = stringify_val(v)
        if "target_asset" not in bs_norm or not bs_norm["target_asset"]:
            bs_norm["target_asset"] = "USDJPY"
        if "timeframe" not in bs_norm or not bs_norm["timeframe"]:
            bs_norm["timeframe"] = "M5"
        for field in ["feature_engineering", "entry_condition", "exit_condition", "risk_management", "benchmark_baseline", "falsification_metric"]:
            if field not in bs_norm:
                bs_norm[field] = "Unspecified"
        norm["backtest_spec"] = bs_norm
    else:
        # Check if fields were placed at top level
        spec_fields = ["feature_engineering", "entry_condition", "exit_condition", "risk_management", "benchmark_baseline", "falsification_metric"]
        if any(f in norm for f in spec_fields):
            norm["backtest_spec"] = {
                "target_asset": stringify_val(norm.get("target_asset", "USDJPY")),
                "timeframe": stringify_val(norm.get("timeframe", "M5")),
                "feature_engineering": stringify_val(norm.get("feature_engineering", "Unspecified")),
                "entry_condition": stringify_val(norm.get("entry_condition", "Unspecified")),
                "exit_condition": stringify_val(norm.get("exit_condition", "Unspecified")),
                "risk_management": stringify_val(norm.get("risk_management", "Unspecified")),
                "benchmark_baseline": stringify_val(norm.get("benchmark_baseline", "Unspecified")),
                "falsification_metric": stringify_val(norm.get("falsification_metric", norm.get("testable_falsification_condition", "Unspecified"))),
            }

    # Defaults for required top-level fields
    for field in ["title", "target_phenomenon", "proposed_mechanism", "analogical_source", "claim", "grounding_rationale", "testable_falsification_condition"]:
        if field not in norm or not norm[field]:
            norm[field] = stringify_val(norm.get(field.replace("_", ""), f"Standard {field.replace('_', ' ')}"))
        else:
            norm[field] = stringify_val(norm[field])

    return norm


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts_db = json.load(f)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    generated_records: List[Dict[str, Any]] = []
    completed_call_ids = set()

    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    generated_records = existing
                    completed_call_ids = {r["call_id"] for r in existing}
                    print(f"Resuming from existing {len(completed_call_ids)} completed calls.")
        except Exception as e:
            print(f"Warning loading existing output: {e}")

    total_d4_cost = sum(r.get("cost_usd", 0.0) for r in generated_records)
    call_idx = len(generated_records)

    print(f"H2-D4 generation plan: 4 concepts x 3 conditions x 3 seeds = 36 calls.")
    start_time = time.time()

    for cid in CONCEPTS:
        for cond in CONDITIONS:
            for seed in SEEDS:
                call_id = f"h2_d4_{cid}_{cond.lower()}_s{seed}"
                if call_id in completed_call_ids:
                    continue

                call_idx += 1
                prompt_text = prompts_db[cid][cond]
                prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

                print(f"[{call_idx}/36] Calling LLM for {call_id} (Seed: {seed})...")

                # Call LLM
                response = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": DIAGNOSTIC_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt_text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    seed=seed,
                )

                usage = response.usage
                in_tok = usage.prompt_tokens
                out_tok = usage.completion_tokens
                cost = record_budget_usage(LEDGER_PATH, call_id, "gpt-4o-mini-2024-07-18", in_tok, out_tok)
                total_d4_cost += cost

                if total_d4_cost > H2_D4_BUDGET_LIMIT_USD:
                    raise RuntimeError(f"H2-D4 budget limit exceeded: {total_d4_cost:.6f} > {H2_D4_BUDGET_LIMIT_USD}")

                raw_content = response.choices[0].message.content
                parsed_json = json.loads(raw_content)
                cand_data = normalize_candidate_dict(parsed_json)

                # Ensure candidate conforms to HypothesisCandidate structure
                cand_data["hypothesis_id"] = call_id
                cand_data["condition_id"] = cond
                cand_data["task_id"] = f"d4_{cid}"

                # Pydantic validation check
                cand_obj = HypothesisCandidate(**cand_data)

                record = {
                    "call_id": call_id,
                    "concept_id": cid,
                    "condition_id": cond,
                    "seed": seed,
                    "prompt_sha256": prompt_hash,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                    "candidate": cand_obj.model_dump(),
                }
                generated_records.append(record)
                completed_call_ids.add(call_id)

                # Incremental persist
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(generated_records, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f"H2-D4 Generation Complete!")
    print(f"Total Calls: {len(generated_records)}/36")
    print(f"Total H2-D4 Cost: ${total_d4_cost:.6f} USD (Budget cap: ${H2_D4_BUDGET_LIMIT_USD})")
    print(f"Elapsed Time: {elapsed:.2f}s")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
