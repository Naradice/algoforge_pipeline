"""
Script: run_h2_d7_generation.py
Executes the approved H2-D7 Diagnostic Experiment:
- 2x2 Factorial Design across Capacity (Standard [1000/600] vs Expanded [2000/1200])
  and Skeletal Guidance (None [C00, C10] vs Verified Skeleton [C01, C11]).
- 2 concepts (hawkes_process, jump_diffusion) x 3 seeds (42, 43, 44) = 6 blocks.
- Per block: 4 conditions (C00, C10, C01, C11), each with initial generation + checklist patch = 8 calls.
- Total API calls = 6 blocks x 8 calls = 48 API calls.
- Hard budget limit: $0.050000 USD, enforced via pre-call payload token reservation.

Model: gpt-4o-mini-2024-07-18, temperature = 0.2
Adheres strictly to H2-D7-PLAN-v0.5 (APPROVED).
"""

import os
import sys
import re
import json
import time
import random
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from openai import OpenAI
from research_pipeline.discovery.models import HypothesisCandidate
from research_pipeline.discovery.diagnostic_generator import DIAGNOSTIC_SYSTEM_INSTRUCTION
from scripts.build_h2_d7_prompts import (
    build_c00_c10_initial_prompt,
    build_c01_c11_initial_prompt,
    build_checklist_patch_prompt,
)
from scripts.run_h2_d6_generation import (
    to_snake,
    stringify_val,
    safe_json_loads,
    inspect_missing_parameters_for_checklist,
)
from scripts.run_h2_d6_p1_generation import (
    normalize_candidate_dict_safe,
    apply_patch_safe,
)

GOLD_PARAMS_PATH = BASE_DIR / "tests" / "fixtures" / "gold_standard" / "h2_d6_gold_parameters.json"
LEDGER_PATH = BASE_DIR / "data" / "budget_ledger.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "h2_d7_raw_candidates.json"

INPUT_TOKEN_PRICE_PER_M = 0.150   # $0.150 / 1M tokens
OUTPUT_TOKEN_PRICE_PER_M = 0.600  # $0.600 / 1M tokens
H2_D7_HARD_BUDGET_CAP_USD = 0.050000

CONCEPTS = ["hawkes_process", "jump_diffusion"]
SEEDS = [42, 43, 44]

# Condition Configs: (initial_max_tokens, patch_max_tokens, has_skeleton)
CONDITION_SPECS = {
    "C00": {"initial_tokens": 1000, "patch_tokens": 600, "has_skeleton": False},
    "C10": {"initial_tokens": 2000, "patch_tokens": 1200, "has_skeleton": False},
    "C01": {"initial_tokens": 1000, "patch_tokens": 600, "has_skeleton": True},
    "C11": {"initial_tokens": 2000, "patch_tokens": 1200, "has_skeleton": True},
}


def get_current_cumulative_spend(ledger_path: Path) -> float:
    cum = 0.216597
    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "cumulative_spent_usd" in record:
                            cum = record["cumulative_spent_usd"]
                    except Exception:
                        pass
    return cum


def check_and_record_budget(
    ledger_path: Path,
    call_id: str,
    model_id: str,
    in_tokens: int,
    out_tokens: int,
    base_cum_spend: float,
) -> float:
    cost = (in_tokens / 1_000_000.0) * INPUT_TOKEN_PRICE_PER_M + (out_tokens / 1_000_000.0) * OUTPUT_TOKEN_PRICE_PER_M
    current_ledger_cum = get_current_cumulative_spend(ledger_path)
    new_cum = current_ledger_cum + cost

    entry = {
        "call_id": call_id,
        "model_id": model_id,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "cost_usd": round(cost, 6),
        "cumulative_spent_usd": round(new_cum, 6),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return cost


def execute_llm_call_with_reservation(
    client: OpenAI,
    prompt: str,
    max_tokens: int,
    seed: int,
    call_id: str,
    base_experiment_spent: float,
) -> Tuple[Dict[str, Any], int, int, str, float]:
    # 1. Measure approximate input token count for safety reservation
    # Rough estimate + safety margin: ~4 chars per token, or API dry run
    estimated_in_tokens = int(len(prompt + DIAGNOSTIC_SYSTEM_INSTRUCTION) / 3.0) + 150
    worst_case_call_cost = (estimated_in_tokens / 1_000_000.0) * INPUT_TOKEN_PRICE_PER_M + (max_tokens / 1_000_000.0) * OUTPUT_TOKEN_PRICE_PER_M

    if base_experiment_spent + worst_case_call_cost > H2_D7_HARD_BUDGET_CAP_USD:
        raise RuntimeError(
            f"Pre-call budget reservation failed for {call_id}: "
            f"current {base_experiment_spent:.6f} + worst_case {worst_case_call_cost:.6f} "
            f"> cap {H2_D7_HARD_BUDGET_CAP_USD:.6f}"
        )

    # 2. Execute API Call
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

    # 3. Exact Accounting
    actual_cost = check_and_record_budget(
        LEDGER_PATH, call_id, "gpt-4o-mini-2024-07-18", in_tok, out_tok, base_experiment_spent
    )

    raw_content = response.choices[0].message.content
    parsed_json = safe_json_loads(raw_content)

    return parsed_json, in_tok, out_tok, finish_reason, actual_cost


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
                    print(f"Resuming H2-D7 from existing {len(completed_block_ids)} completed blocks.")
        except Exception as e:
            print(f"Warning loading existing output: {e}")

    total_d7_cost = sum(b.get("total_block_cost_usd", 0.0) for b in generated_blocks)
    block_idx = len(generated_blocks)

    print("================================================================================")
    print("H2-D7 Experiment Runner: Disentangling Mathematical Specification Failure")
    print("Plan: 2x2 Factorial Design across Capacity x Skeletal Guidance")
    print("6 blocks x 4 conditions (C00, C10, C01, C11) x 2 calls = 48 API calls.")
    print(f"Hard Budget Cap: ${H2_D7_HARD_BUDGET_CAP_USD:.6f} USD")
    print("================================================================================")
    start_time = time.time()

    for cid in CONCEPTS:
        gold_concept_spec = gold_params_db[cid]

        for seed in SEEDS:
            block_id = f"h2_d7_{cid}_s{seed}"
            if block_id in completed_block_ids:
                print(f"Skipping already completed block: {block_id}")
                continue

            block_idx += 1
            print(f"\n==================================================")
            print(f"[{block_idx}/6] Executing Block: {block_id} (Concept: {cid}, Seed: {seed})")
            print(f"==================================================")

            block_cost = 0.0
            block_candidates = {}
            block_patches = {}
            diagnostics_log = {}

            # Shuffle condition execution order within block to eliminate API order / caching bias
            conditions_order = ["C00", "C10", "C01", "C11"]
            random.seed(seed + 100)
            random.shuffle(conditions_order)
            print(f"  Execution order for block: {conditions_order}")

            for cond in conditions_order:
                spec_cfg = CONDITION_SPECS[cond]
                init_max_tokens = spec_cfg["initial_tokens"]
                patch_max_tokens = spec_cfg["patch_tokens"]
                has_skeleton = spec_cfg["has_skeleton"]

                # -------------------------------------------------------------
                # Call A: Initial Generation
                # -------------------------------------------------------------
                call_id_init = f"{block_id}_{cond}_init"
                print(f"  [Call] {cond} Initial: {call_id_init} (max: {init_max_tokens}, skel: {has_skeleton})...")

                if has_skeleton:
                    prompt_init = build_c01_c11_initial_prompt(cid)
                else:
                    prompt_init = build_c00_c10_initial_prompt(cid)

                json_init, in_init, out_init, fin_init, cost_init = execute_llm_call_with_reservation(
                    client, prompt_init, init_max_tokens, seed, call_id_init, total_d7_cost + block_cost
                )
                block_cost += cost_init

                norm_init = normalize_candidate_dict_safe(json_init)
                norm_init["hypothesis_id"] = call_id_init
                norm_init["condition_id"] = f"{cond}_init"
                norm_init["task_id"] = f"d7_{cid}"
                cand_init = HypothesisCandidate(**norm_init)
                cand_init_dict = cand_init.model_dump()
                init_spec_text = json.dumps(cand_init_dict, indent=2, ensure_ascii=False)

                # -------------------------------------------------------------
                # Call B: Checklist-Guided Revision Patch
                # -------------------------------------------------------------
                call_id_patch = f"{block_id}_{cond}_patch"
                print(f"  [Call] {cond} Patch: {call_id_patch} (max: {patch_max_tokens})...")

                missing_items, audit_log = inspect_missing_parameters_for_checklist(cid, cand_init_dict, gold_concept_spec)
                prompt_patch = build_checklist_patch_prompt(init_spec_text, missing_items)

                json_patch, in_patch, out_patch, fin_patch, cost_patch = execute_llm_call_with_reservation(
                    client, prompt_patch, patch_max_tokens, seed, call_id_patch, total_d7_cost + block_cost
                )
                block_cost += cost_patch

                final_cand_dict = apply_patch_safe(cand_init_dict, json_patch)
                final_cand_dict["hypothesis_id"] = f"{block_id}_{cond}"
                final_cand_dict["condition_id"] = cond
                final_cand_dict["task_id"] = f"d7_{cid}"
                cand_final = HypothesisCandidate(**final_cand_dict)

                block_candidates[f"{cond}_init"] = cand_init_dict
                block_candidates[cond] = cand_final.model_dump()
                block_patches[f"{cond}_patch"] = json_patch

                diagnostics_log[cond] = {
                    "initial_call": {
                        "call_id": call_id_init,
                        "in_tokens": in_init,
                        "out_tokens": out_init,
                        "finish_reason": fin_init,
                        "cost_usd": cost_init,
                    },
                    "patch_call": {
                        "call_id": call_id_patch,
                        "in_tokens": in_patch,
                        "out_tokens": out_patch,
                        "finish_reason": fin_patch,
                        "cost_usd": cost_patch,
                        "missing_checklist": missing_items,
                        "audit_log": audit_log,
                    },
                    "total_condition_cost_usd": cost_init + cost_patch,
                }

            # Compile Block Record
            block_record = {
                "block_id": block_id,
                "concept_id": cid,
                "seed": seed,
                "execution_order": conditions_order,
                "total_block_cost_usd": round(block_cost, 6),
                "candidates": block_candidates,
                "patches": block_patches,
                "diagnostics": diagnostics_log,
            }

            generated_blocks.append(block_record)
            completed_block_ids.add(block_id)
            total_d7_cost += block_cost

            # Checkpoint output after each block
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(generated_blocks, f, indent=2, ensure_ascii=False)

            print(f"Block {block_id} complete! Cost: ${block_cost:.6f} USD. Cumulative H2-D7: ${total_d7_cost:.6f} USD.")

    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f"H2-D7 Generation Finished!")
    print(f"Completed blocks: {len(generated_blocks)}/6 (Total calls: {len(generated_blocks) * 8}/48)")
    print(f"Total H2-D7 Cost: ${total_d7_cost:.6f} USD (Budget cap: ${H2_D7_HARD_BUDGET_CAP_USD:.6f} USD)")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
