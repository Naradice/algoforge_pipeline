"""
Script: run_h2_d6_p1_generation.py
Executes the approved H2-D6 P1 2x2 Factorial Disentanglement Experiment:
- 2 concepts (hawkes_process, jump_diffusion) x 3 seeds (42, 43, 44) = 6 blocks
- Per block: 7 API calls:
  1. A-init: Single-stage initial generation (max_tokens: 1,000)
  2. B-init Stage 1: Mathematical Model Identification (max_tokens: 500)
  3. B-init Stage 2: Strategy Binding (max_tokens: 500)
  4. A0: Generic revision of A-init (max_tokens: 600)
  5. A1: Checklist revision of A-init (max_tokens: 600)
  6. B0: Generic revision of B-init (max_tokens: 600)
  7. B1: Checklist revision of B-init (max_tokens: 600)
Total API calls = 6 blocks x 7 calls = 42 API calls.
Evaluated series = 6 blocks x 4 conditions (A0, A1, B0, B1) = 24 series.

Model: gpt-4o-mini-2024-07-18, temperature = 0.2
Adheres strictly to H2-D6-P1-PLAN-v0.7.
"""

import os
import sys
import re
import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from openai import OpenAI
from research_pipeline.discovery.models import HypothesisCandidate, AlgoForgeBacktestSpec
from research_pipeline.discovery.diagnostic_generator import DIAGNOSTIC_SYSTEM_INSTRUCTION
from scripts.build_h2_d6_prompts import (
    build_s0_prompt,
    build_s1_stage1_prompt,
    build_s1_stage2_prompt,
    build_s2a_patch_prompt,
    build_s2b_patch_prompt,
)
from scripts.run_h2_d6_generation import (
    to_snake,
    stringify_val,
    safe_json_loads,
    normalize_candidate_dict,
    apply_patch,
    inspect_missing_parameters_for_checklist,
)

GOLD_PARAMS_PATH = BASE_DIR / "tests" / "fixtures" / "gold_standard" / "h2_d6_gold_parameters.json"
LEDGER_PATH = BASE_DIR / "data" / "budget_ledger.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "h2_d6_p1_raw_candidates.json"

INPUT_TOKEN_PRICE_PER_M = 0.150   # $0.150 / 1M tokens
OUTPUT_TOKEN_PRICE_PER_M = 0.600  # $0.600 / 1M tokens
HARD_BUDGET_CEILING_USD = 10.00
H2_D6_P1_BUDGET_LIMIT_USD = 0.030000

CONCEPTS = ["hawkes_process", "jump_diffusion"]
SEEDS = [42, 43, 44]


def normalize_candidate_dict_safe(data: Dict[str, Any]) -> Dict[str, Any]:
    norm = normalize_candidate_dict(data)
    if "backtest_spec" not in norm or norm["backtest_spec"] is None or not isinstance(norm["backtest_spec"], dict):
        norm["backtest_spec"] = {
            "target_asset": "USDJPY",
            "timeframe": "M5",
            "feature_engineering": "Unspecified",
            "entry_condition": "Unspecified",
            "exit_condition": "Unspecified",
            "risk_management": "Unspecified",
            "benchmark_baseline": "Unspecified",
            "falsification_metric": stringify_val(norm.get("testable_falsification_condition", "Unspecified")),
        }
    return norm


def apply_patch_safe(
    base_candidate: Dict[str, Any],
    patch_obj: Dict[str, Any],
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(base_candidate))
    if updated.get("backtest_spec") is None or not isinstance(updated.get("backtest_spec"), dict):
        updated["backtest_spec"] = {
            "target_asset": "USDJPY",
            "timeframe": "M5",
            "feature_engineering": "Unspecified",
            "entry_condition": "Unspecified",
            "exit_condition": "Unspecified",
            "risk_management": "Unspecified",
            "benchmark_baseline": "Unspecified",
            "falsification_metric": "Unspecified",
        }
    spec = updated["backtest_spec"]

    updates = patch_obj.get("updates") or {}
    if not isinstance(updates, dict):
        return updated

    for k, v in updates.items():
        sk = to_snake(k)
        if v and isinstance(v, (str, dict, list)):
            val_str = stringify_val(v)
            if sk in ["feature_engineering", "entry_condition", "exit_condition", "risk_management", "falsification_metric", "benchmark_baseline"]:
                spec[sk] = val_str
            elif sk in ["title", "target_phenomenon", "proposed_mechanism", "claim", "grounding_rationale"]:
                updated[sk] = val_str

    updated["backtest_spec"] = spec
    return updated


def record_budget_usage(ledger_path: Path, call_id: str, model_id: str, in_tokens: int, out_tokens: int) -> float:
    cost = (in_tokens / 1_000_000.0) * INPUT_TOKEN_PRICE_PER_M + (out_tokens / 1_000_000.0) * OUTPUT_TOKEN_PRICE_PER_M
    cum_spent = 0.199263
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


def execute_llm_call(
    client: OpenAI,
    prompt: str,
    max_tokens: int,
    seed: int,
    call_id: str,
) -> Tuple[Dict[str, Any], int, int, str, float]:
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": DIAGNOSTIC_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        seed=seed,
        max_tokens=max_tokens,
    )

    usage = response.usage
    in_tok = usage.prompt_tokens
    out_tok = usage.completion_tokens
    finish_reason = response.choices[0].finish_reason or "unknown"
    cost = record_budget_usage(LEDGER_PATH, call_id, "gpt-4o-mini-2024-07-18", in_tok, out_tok)

    raw_content = response.choices[0].message.content
    parsed_json = safe_json_loads(raw_content)

    return parsed_json, in_tok, out_tok, finish_reason, cost


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    with open(GOLD_PARAMS_PATH, "r", encoding="utf-8") as f:
        gold_params_db = json.load(f)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    generated_blocks = []
    completed_block_ids = set()

    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    generated_blocks = existing
                    completed_block_ids = {b["block_id"] for b in existing}
                    print(f"Resuming from existing {len(completed_block_ids)} completed blocks.")
        except Exception as e:
            print(f"Warning loading existing output: {e}")

    total_p1_cost = sum(b.get("total_block_cost_usd", 0.0) for b in generated_blocks)
    block_idx = len(generated_blocks)

    print("================================================================================")
    print("H2-D6 P1 Experiment Runner: 2x2 Factorial Disentanglement")
    print("Plan: 2 concepts x 3 seeds = 6 blocks x 7 calls = 42 API calls total.")
    print("Evaluated series = 6 blocks x 4 conditions (A0, A1, B0, B1) = 24 series.")
    print("================================================================================")
    start_time = time.time()

    for cid in CONCEPTS:
        gold_concept_spec = gold_params_db[cid]

        for seed in SEEDS:
            block_id = f"h2_d6_p1_{cid}_s{seed}"
            if block_id in completed_block_ids:
                print(f"Skipping already completed block: {block_id}")
                continue

            block_idx += 1
            print(f"\n==================================================")
            print(f"[{block_idx}/6] Executing Block: {block_id} (Concept: {cid}, Seed: {seed})")
            print(f"==================================================")

            block_cost = 0.0
            diagnostics_log = {}

            # -------------------------------------------------------------
            # Call 1: A-init (Single-Stage Initial Generation)
            # max_tokens = 1,000
            # -------------------------------------------------------------
            call_id_a_init = f"{block_id}_a_init"
            print(f"  [1/7] Calling A-init (Single-stage simultaneous): {call_id_a_init} (max: 1,000)...")
            prompt_a_init = build_s0_prompt(cid)
            json_a_init, in_a, out_a, fin_a, cost_a = execute_llm_call(client, prompt_a_init, 1000, seed, call_id_a_init)
            block_cost += cost_a
            diagnostics_log["A_init"] = {
                "call_id": call_id_a_init,
                "in_tokens": in_a,
                "out_tokens": out_a,
                "finish_reason": fin_a,
                "cost_usd": cost_a,
            }

            norm_a_init = normalize_candidate_dict_safe(json_a_init)
            norm_a_init["hypothesis_id"] = call_id_a_init
            norm_a_init["condition_id"] = "A_init"
            norm_a_init["task_id"] = f"p1_{cid}"
            cand_a_init = HypothesisCandidate(**norm_a_init)
            a_init_spec_text = json.dumps(cand_a_init.model_dump(), indent=2, ensure_ascii=False)

            # -------------------------------------------------------------
            # Call 2: B-init Stage 1 (Two-Stage Math Identification)
            # max_tokens = 500
            # -------------------------------------------------------------
            call_id_b_st1 = f"{block_id}_b_stage1"
            print(f"  [2/7] Calling B-init Stage 1 (Math Model ID): {call_id_b_st1} (max: 500)...")
            prompt_b_st1 = build_s1_stage1_prompt(cid)
            json_b_st1, in_b1, out_b1, fin_b1, cost_b1 = execute_llm_call(client, prompt_b_st1, 500, seed, call_id_b_st1)
            block_cost += cost_b1
            stage1_text = json.dumps(json_b_st1, indent=2, ensure_ascii=False)
            diagnostics_log["B_stage1"] = {
                "call_id": call_id_b_st1,
                "in_tokens": in_b1,
                "out_tokens": out_b1,
                "finish_reason": fin_b1,
                "cost_usd": cost_b1,
            }

            # -------------------------------------------------------------
            # Call 3: B-init Stage 2 (Two-Stage Strategy Binding)
            # max_tokens = 500
            # -------------------------------------------------------------
            call_id_b_init = f"{block_id}_b_init"
            print(f"  [3/7] Calling B-init Stage 2 (Strategy Binding): {call_id_b_init} (max: 500)...")
            prompt_b_st2 = build_s1_stage2_prompt(cid, stage1_text)
            json_b_init, in_b2, out_b2, fin_b2, cost_b2 = execute_llm_call(client, prompt_b_st2, 500, seed, call_id_b_init)
            block_cost += cost_b2
            diagnostics_log["B_init"] = {
                "call_id": call_id_b_init,
                "in_tokens": in_b2,
                "out_tokens": out_b2,
                "finish_reason": fin_b2,
                "cost_usd": cost_b2,
            }

            norm_b_init = normalize_candidate_dict_safe(json_b_init)
            norm_b_init["hypothesis_id"] = call_id_b_init
            norm_b_init["condition_id"] = "B_init"
            norm_b_init["task_id"] = f"p1_{cid}"
            cand_b_init = HypothesisCandidate(**norm_b_init)
            b_init_spec_text = json.dumps(cand_b_init.model_dump(), indent=2, ensure_ascii=False)

            # -------------------------------------------------------------
            # Call 4: Condition A0 (Generic Revision of Single-Stage A-init)
            # max_tokens = 600
            # -------------------------------------------------------------
            call_id_a0 = f"{block_id}_a0"
            print(f"  [4/7] Calling A0 (Generic revision of Single-stage): {call_id_a0} (max: 600)...")
            prompt_a0 = build_s2a_patch_prompt(a_init_spec_text)
            json_a0_patch, in_a0, out_a0, fin_a0, cost_a0 = execute_llm_call(client, prompt_a0, 600, seed, call_id_a0)
            block_cost += cost_a0

            final_dict_a0 = apply_patch_safe(cand_a_init.model_dump(), json_a0_patch)
            final_dict_a0["hypothesis_id"] = call_id_a0
            final_dict_a0["condition_id"] = "A0"
            final_dict_a0["task_id"] = f"p1_{cid}"
            cand_a0 = HypothesisCandidate(**final_dict_a0)

            diagnostics_log["A0"] = {
                "call_id": call_id_a0,
                "in_tokens": in_a0,
                "out_tokens": out_a0,
                "finish_reason": fin_a0,
                "cost_usd": cost_a0,
            }

            # -------------------------------------------------------------
            # Call 5: Condition A1 (Checklist Revision of Single-Stage A-init)
            # max_tokens = 600
            # -------------------------------------------------------------
            call_id_a1 = f"{block_id}_a1"
            print(f"  [5/7] Inspecting & Calling A1 (Checklist revision of Single-stage): {call_id_a1} (max: 600)...")
            missing_a, audit_log_a = inspect_missing_parameters_for_checklist(cid, cand_a_init.model_dump(), gold_concept_spec)
            prompt_a1 = build_s2b_patch_prompt(a_init_spec_text, missing_a)
            json_a1_patch, in_a1, out_a1, fin_a1, cost_a1 = execute_llm_call(client, prompt_a1, 600, seed, call_id_a1)
            block_cost += cost_a1

            final_dict_a1 = apply_patch_safe(cand_a_init.model_dump(), json_a1_patch)
            final_dict_a1["hypothesis_id"] = call_id_a1
            final_dict_a1["condition_id"] = "A1"
            final_dict_a1["task_id"] = f"p1_{cid}"
            cand_a1 = HypothesisCandidate(**final_dict_a1)

            diagnostics_log["A1"] = {
                "call_id": call_id_a1,
                "in_tokens": in_a1,
                "out_tokens": out_a1,
                "finish_reason": fin_a1,
                "cost_usd": cost_a1,
                "missing_checklist": missing_a,
                "audit_log": audit_log_a,
            }

            # -------------------------------------------------------------
            # Call 6: Condition B0 (Generic Revision of Two-Stage B-init)
            # max_tokens = 600
            # -------------------------------------------------------------
            call_id_b0 = f"{block_id}_b0"
            print(f"  [6/7] Calling B0 (Generic revision of Two-stage): {call_id_b0} (max: 600)...")
            prompt_b0 = build_s2a_patch_prompt(b_init_spec_text)
            json_b0_patch, in_b0, out_b0, fin_b0, cost_b0 = execute_llm_call(client, prompt_b0, 600, seed, call_id_b0)
            block_cost += cost_b0

            final_dict_b0 = apply_patch_safe(cand_b_init.model_dump(), json_b0_patch)
            final_dict_b0["hypothesis_id"] = call_id_b0
            final_dict_b0["condition_id"] = "B0"
            final_dict_b0["task_id"] = f"p1_{cid}"
            cand_b0 = HypothesisCandidate(**final_dict_b0)

            diagnostics_log["B0"] = {
                "call_id": call_id_b0,
                "in_tokens": in_b0,
                "out_tokens": out_b0,
                "finish_reason": fin_b0,
                "cost_usd": cost_b0,
            }

            # -------------------------------------------------------------
            # Call 7: Condition B1 (Checklist Revision of Two-Stage B-init)
            # max_tokens = 600
            # -------------------------------------------------------------
            call_id_b1 = f"{block_id}_b1"
            print(f"  [7/7] Inspecting & Calling B1 (Checklist revision of Two-stage): {call_id_b1} (max: 600)...")
            missing_b, audit_log_b = inspect_missing_parameters_for_checklist(cid, cand_b_init.model_dump(), gold_concept_spec)
            prompt_b1 = build_s2b_patch_prompt(b_init_spec_text, missing_b)
            json_b1_patch, in_b1_c, out_b1_c, fin_b1_c, cost_b1_c = execute_llm_call(client, prompt_b1, 600, seed, call_id_b1)
            block_cost += cost_b1_c

            final_dict_b1 = apply_patch_safe(cand_b_init.model_dump(), json_b1_patch)
            final_dict_b1["hypothesis_id"] = call_id_b1
            final_dict_b1["condition_id"] = "B1"
            final_dict_b1["task_id"] = f"p1_{cid}"
            cand_b1 = HypothesisCandidate(**final_dict_b1)
            final_dict_b1["hypothesis_id"] = call_id_b1
            final_dict_b1["condition_id"] = "B1"
            final_dict_b1["task_id"] = f"p1_{cid}"
            cand_b1 = HypothesisCandidate(**final_dict_b1)

            diagnostics_log["B1"] = {
                "call_id": call_id_b1,
                "in_tokens": in_b1_c,
                "out_tokens": out_b1_c,
                "finish_reason": fin_b1_c,
                "cost_usd": cost_b1_c,
                "missing_checklist": missing_b,
                "audit_log": audit_log_b,
            }

            # Compile Block Record
            block_record = {
                "block_id": block_id,
                "concept_id": cid,
                "seed": seed,
                "total_block_cost_usd": round(block_cost, 6),
                "candidates": {
                    "A_init": cand_a_init.model_dump(),
                    "B_stage1": json_b_st1,
                    "B_init": cand_b_init.model_dump(),
                    "A0": cand_a0.model_dump(),
                    "A1": cand_a1.model_dump(),
                    "B0": cand_b0.model_dump(),
                    "B1": cand_b1.model_dump(),
                },
                "patches": {
                    "A0_patch": json_a0_patch,
                    "A1_patch": json_a1_patch,
                    "B0_patch": json_b0_patch,
                    "B1_patch": json_b1_patch,
                },
                "diagnostics": diagnostics_log,
            }

            generated_blocks.append(block_record)
            completed_block_ids.add(block_id)
            total_p1_cost += block_cost

            # Checkpoint output after each block
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(generated_blocks, f, indent=2, ensure_ascii=False)

            print(f"Block {block_id} complete! Cost: ${block_cost:.6f} USD. Cumulative P1: ${total_p1_cost:.6f} USD.")

            if total_p1_cost > H2_D6_P1_BUDGET_LIMIT_USD:
                print(f"Warning: Cumulative P1 spend ${total_p1_cost:.6f} exceeded soft limit ${H2_D6_P1_BUDGET_LIMIT_USD}. Stopping.")
                break

    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f"H2-D6 P1 Generation Finished!")
    print(f"Completed blocks: {len(generated_blocks)}/6 (Total calls: {len(generated_blocks) * 7}/42)")
    print(f"Total P1 Cost: ${total_p1_cost:.6f} USD (Budget cap: ${H2_D6_P1_BUDGET_LIMIT_USD:.6f} USD)")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
