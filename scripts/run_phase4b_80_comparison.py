"""Phase 4-B: 80-Candidate 4-Way Comparison (20 each: OpenAI API, Claude Code, Antigravity, Evolution).

Reports full distribution (Min, Q25, Median, Q75, Max), duplicates, cache hits, cost, and Diag40d reference.
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import BacktestConfig, GateConfig, compute_indicators, run_causal_fx_backtest
from research_pipeline.backtest.candidate_generators import AgentJSONGenerator, BaseAPIGenerator, BudgetGuard, EvoGenerator, GenerationContext
from research_pipeline.backtest.declarative_strategy import build_causal_htf_features, evaluate_declarative_signals
from research_pipeline.backtest.strategy_registry import StrategyRegistry

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from run_phase3_diversity_and_cv import evaluate_p3
from run_phase4b_openai import load_env_key


def openai_transport_20(model: str, n: int) -> dict:
    api_key = os.environ["OPENAI_API_KEY"]
    prompt = (
        f"Output ONLY a raw JSON array (no markdown fences, no explanation) of {n} diverse USDJPY M5 trading strategy objects matching this exact schema: "
        "each object has keys 'long' ({'op': 'AND'|'OR', 'clauses': [1-2 clauses]}), 'short' ({'op': 'AND'|'OR', 'clauses': [1-2 clauses]}), "
        "'htf_filter' ({'h1_align': bool, 'h4_align': bool}), 'hours_utc' (list of ints 0-23, or [] for all hours), "
        "'risk' ({'k_sl': float>0, 'k_tp': float>0, 'max_hold_bars': int>0}). "
        "Allowed clause types: {'type':'ema_cross','fast':int,'slow':int,'dir':1|-1} (fast<slow), "
        "{'type':'rsi_lt','period':int,'level':int} (5-95), {'type':'rsi_gt','period':int,'level':int} (5-95), "
        "{'type':'bb_break_upper','period':int,'std_mult':float}, {'type':'bb_break_lower','period':int,'std_mult':float}, "
        "{'type':'donchian_break_high','period':int}, {'type':'donchian_break_low','period':int}."
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3500,
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    usage = data.get("usage", {})
    return {
        "text": text,
        "usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")},
    }


def main():
    t0 = time.time()
    load_env_key(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")), "OPENAI_API_KEY")

    # Ensure claude_code_batch_20.json is UTF-8
    cc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "agent_batches", "claude_code_batch_20.json"))
    if os.path.exists(cc_path):
        raw_b = open(cc_path, "rb").read()
        if raw_b.startswith(b"\xff\xfe") or raw_b.startswith(b"\xfe\xff"):
            txt = raw_b.decode("utf-16").strip()
            if txt.startswith("```"):
                txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            open(cc_path, "w", encoding="utf-8").write(txt)

    ag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "agent_batches", "antigravity_batch_20.json"))

    ledger_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "budget_ledger.jsonl"))
    guard = BudgetGuard(max_budget_usd=0.05, ledger_path=ledger_path)
    model_id = "gpt-4o-mini-2024-07-18"

    # Generate 20 from each of the 4 sources
    openai_gen = BaseAPIGenerator("openai", model_id, "OPENAI_API_KEY", guard, EvoGenerator(42), transport_fn=openai_transport_20)
    cc_gen = AgentJSONGenerator(cc_path, declared_model="claude_code")
    ag_gen = AgentJSONGenerator(ag_path, declared_model="gemini_antigravity")
    evo_gen = EvoGenerator(seed=12345)

    batches = [
        ("OpenAI_API", openai_gen.generate(GenerationContext(), 20), guard.spent_usd),
        ("Claude_Code", cc_gen.generate(GenerationContext(), 20), 0.0),
        ("Antigravity", ag_gen.generate(GenerationContext(), 20), 0.0),
        ("Evolution", evo_gen.generate(GenerationContext(), 20), 0.0),
    ]

    snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    with open(snap_path, "rb") as f:
        dataset_sha = hashlib.sha256(f.read()).hexdigest()

    df = pd.read_parquet(snap_path)
    df = compute_indicators(df, BacktestConfig())
    htf = build_causal_htf_features(df)
    df["h1_trend"] = htf["h1_trend"]
    df["h4_trend"] = htf["h4_trend"]
    df["date"] = df["time"].dt.date
    dates = sorted(df["date"].unique())

    explore_start_idx = int(df[df["date"] == dates[60]].index[0])
    p2_split_date = dates[106]
    diag_start_idx = int(df[df["date"] == dates[152]].index[0])
    df_explore = df.iloc[:diag_start_idx].copy().reset_index(drop=True)

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "strategy_registry.db"))
    registry = StrategyRegistry(db_path)

    comparison = {}
    print("\n=== Phase 4-B: 80-Candidate 4-Way Distribution Comparison ===")
    for name, cands, cost_usd in batches:
        res_list = []
        seen_local = set()
        intra_dups = 0
        for gc in cands:
            r = evaluate_p3(gc.strategy, df_explore, explore_start_idx, p2_split_date, dataset_sha, registry, None, None, f"p4b_{name}", "phase4b_80")
            if r["strategy_hash"] in seen_local:
                intra_dups += 1
            seen_local.add(r["strategy_hash"])
            res_list.append(r)

        valid = [r for r in res_list if r["status"] != "REJECTED_SCHEMA"]
        n_unique = len(seen_local)
        n_cache = sum(1 for r in res_list if r["cache_hit"])
        eligible = [r for r in valid if r["metrics"].get("trade_count", 0) >= 30]
        sharpes = [r["metrics"]["annualized_sharpe"] for r in eligible]
        min_cvs = [r["metrics"]["min_cv_sharpe"] for r in eligible]
        profitable = sum(1 for s in sharpes if s > 0)

        dist_sh = {
            "min": float(np.min(sharpes)) if sharpes else None,
            "q25": float(np.percentile(sharpes, 25)) if sharpes else None,
            "median": float(np.median(sharpes)) if sharpes else None,
            "q75": float(np.percentile(sharpes, 75)) if sharpes else None,
            "max": float(np.max(sharpes)) if sharpes else None,
        }

        eligible.sort(key=lambda x: x["metrics"].get("min_cv_sharpe", -999.0), reverse=True)
        diag_top1 = {}
        if eligible:
            st = eligible[0]["strategy"]
            cfg = BacktestConfig(k_sl=st["risk"]["k_sl"], k_tp=st["risk"]["k_tp"], max_hold_bars=st["risk"]["max_hold_bars"])
            ls, ss = evaluate_declarative_signals(df, st)
            _, _, diag_top1 = run_causal_fx_backtest(df, cfg, GateConfig(), eval_start_idx=diag_start_idx, long_signal=ls, short_signal=ss)

        print(f"  {name:12s} | Sub={len(cands)} | Valid={len(valid)} | Uniq={n_unique} (Dup={intra_dups}) | CacheHit={n_cache} | Elig={len(eligible)} | Prof={profitable} | Cost=${cost_usd:.5f}")
        if sharpes:
            print(f"    Explore Sharpe Dist: Min={dist_sh['min']:+.3f}, Q25={dist_sh['q25']:+.3f}, Med={dist_sh['median']:+.3f}, Q75={dist_sh['q75']:+.3f}, Max={dist_sh['max']:+.3f}")
            print(f"    Top1 Diag40d (Ref): Sharpe={diag_top1.get('annualized_sharpe',0):+.3f}, Return={diag_top1.get('total_return_pct',0):+.2f}%, Trades={diag_top1.get('trade_count',0)}")

        comparison[name] = {
            "submitted": len(cands), "valid_json": len(valid), "unique": n_unique,
            "intra_dups": intra_dups, "cache_hits": n_cache, "eligible": len(eligible),
            "profitable": profitable, "cost_usd": cost_usd,
            "explore_sharpe_dist": dist_sh, "top1_diag40d": diag_top1,
        }

    registry.close()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase4b_80_comparison_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time() - t0, "comparison": comparison}, f, indent=2)
    print(f"\n[*] Phase 4-B 80-Candidate Comparison Complete in {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    main()
