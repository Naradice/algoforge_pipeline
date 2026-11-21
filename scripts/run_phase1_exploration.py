"""Phase 1 Exploration Experiment: Random Search (120) vs Evolutionary Search (120).

Uses persistent StrategyRegistry for deduplication & caching.
Data split (2024): 60d Warmup + 92d Exploration + 40d Diagnostic (held out from selection).
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import BacktestConfig, GateConfig, compute_indicators, run_causal_fx_backtest
from research_pipeline.backtest.declarative_strategy import (
    build_causal_htf_features,
    evaluate_declarative_signals,
    validate_declarative_strategy,
)
from research_pipeline.backtest.strategy_registry import (
    StrategyRegistry,
    compute_evaluation_hash,
    compute_strategy_hash,
)

ENGINE_VERSION = "causal_fx_backtest_v1.0"


def random_clause(rng: np.random.Generator, direction: int) -> Dict[str, Any]:
    ctype = str(rng.choice(["ema_cross", "rsi_extreme", "bb_break", "donchian_break"]))
    if ctype == "ema_cross":
        fast = int(rng.choice([8, 12, 16]))
        slow = int(rng.choice([32, 48, 64]))
        return {"type": "ema_cross", "fast": fast, "slow": slow, "dir": direction}
    if ctype == "rsi_extreme":
        period = int(rng.choice([10, 14, 21]))
        if direction == 1:
            return {"type": "rsi_lt", "period": period, "level": int(rng.choice([25, 30, 35]))}
        return {"type": "rsi_gt", "period": period, "level": int(rng.choice([65, 70, 75]))}
    if ctype == "bb_break":
        period = int(rng.choice([20, 30]))
        mult = float(rng.choice([1.5, 2.0, 2.5]))
        return {"type": "bb_break_upper" if direction == 1 else "bb_break_lower", "period": period, "std_mult": mult}
    period = int(rng.choice([12, 20, 36]))
    return {"type": "donchian_break_high" if direction == 1 else "donchian_break_low", "period": period}


def sample_random_strategy(rng: np.random.Generator) -> Dict[str, Any]:
    n_long = int(rng.choice([1, 2]))
    n_short = int(rng.choice([1, 2]))
    hour_preset = int(rng.choice([0, 1, 2]))
    hours_utc = [] if hour_preset == 0 else ([7, 8, 9, 10, 11, 12, 13, 14, 15, 16] if hour_preset == 1 else [0, 1, 2, 3, 4, 5, 6])
    return {
        "long": {"op": str(rng.choice(["AND", "OR"])), "clauses": [random_clause(rng, 1) for _ in range(n_long)]},
        "short": {"op": str(rng.choice(["AND", "OR"])), "clauses": [random_clause(rng, -1) for _ in range(n_short)]},
        "htf_filter": {"h1_align": bool(rng.choice([True, False])), "h4_align": bool(rng.choice([True, False]))},
        "hours_utc": hours_utc,
        "risk": {
            "k_sl": float(rng.choice([1.5, 2.0, 2.5])),
            "k_tp": float(rng.choice([2.5, 3.0, 4.0])),
            "max_hold_bars": int(rng.choice([24, 48, 72])),
        },
    }


def mutate_strategy(rng: np.random.Generator, parent: Dict[str, Any]) -> Dict[str, Any]:
    child = json.loads(json.dumps(parent))
    mut_type = str(rng.choice(["risk", "htf", "hours", "clause"]))
    if mut_type == "risk":
        child["risk"]["k_sl"] = float(rng.choice([1.5, 2.0, 2.5]))
        child["risk"]["k_tp"] = float(rng.choice([2.5, 3.0, 4.0]))
    elif mut_type == "htf":
        child["htf_filter"]["h1_align"] = bool(rng.choice([True, False]))
        child["htf_filter"]["h4_align"] = bool(rng.choice([True, False]))
    elif mut_type == "hours":
        hour_preset = int(rng.choice([0, 1, 2]))
        child["hours_utc"] = [] if hour_preset == 0 else ([7, 8, 9, 10, 11, 12, 13, 14, 15, 16] if hour_preset == 1 else [0, 1, 2, 3, 4, 5, 6])
    else:
        if rng.random() < 0.5:
            child["long"]["clauses"][0] = random_clause(rng, 1)
        else:
            child["short"]["clauses"][0] = random_clause(rng, -1)
    return child


def evaluate_candidate(
    cand: Dict[str, Any],
    df_explore: pd.DataFrame,
    eval_start_idx: int,
    dataset_sha: str,
    eval_period: str,
    registry: StrategyRegistry,
    gen_method: str,
    parent_hash: Optional[str],
    exp_id: str,
) -> Dict[str, Any]:
    s_hash = compute_strategy_hash(cand)
    cost_cfg = {"spread": 0.010, "slippage": 0.002}
    e_hash = compute_evaluation_hash(s_hash, dataset_sha, eval_period, ENGINE_VERSION, cost_cfg)

    cached = registry.lookup(e_hash)
    if cached is not None:
        return {
            "strategy_hash": s_hash,
            "evaluation_hash": e_hash,
            "cache_hit": True,
            "status": cached["status"],
            "reason": cached["rejection_reason"],
            "metrics": cached["metrics"],
            "saved_sec": cached["elapsed_sec"],
            "strategy": cand,
        }

    t0 = time.time()
    ok, msg = validate_declarative_strategy(cand)
    if not ok:
        elapsed = time.time() - t0
        registry.record(e_hash, s_hash, cand, "REJECTED_SCHEMA", msg, {}, gen_method, parent_hash, exp_id, elapsed)
        return {"strategy_hash": s_hash, "evaluation_hash": e_hash, "cache_hit": False, "status": "REJECTED_SCHEMA", "reason": msg, "metrics": {}, "saved_sec": 0.0, "strategy": cand}

    cfg = BacktestConfig(
        k_sl=cand["risk"]["k_sl"],
        k_tp=cand["risk"]["k_tp"],
        max_hold_bars=cand["risk"]["max_hold_bars"],
    )
    long_sig, short_sig = evaluate_declarative_signals(df_explore, cand)
    _, _, summary = run_causal_fx_backtest(
        df_explore, cfg, GateConfig(), eval_start_idx=eval_start_idx,
        long_signal=long_sig, short_signal=short_sig,
    )
    elapsed = time.time() - t0

    if summary["trade_count"] < 30:
        status, reason = "REJECTED_PERF", "REJECTED_LOW_TRADES"
    elif summary["annualized_sharpe"] <= 0.0:
        status, reason = "REJECTED_PERF", "REJECTED_NEGATIVE_SHARPE"
    else:
        status, reason = "PROMOTED", "PASS_MIN_CRITERIA"

    registry.record(e_hash, s_hash, cand, status, reason, summary, gen_method, parent_hash, exp_id, elapsed)
    return {
        "strategy_hash": s_hash, "evaluation_hash": e_hash, "cache_hit": False,
        "status": status, "reason": reason, "metrics": summary, "saved_sec": 0.0, "strategy": cand,
    }


def main():
    t_total = time.time()
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

    # Split: 60d warmup, 92d explore (days 60..151), 40d diagnostic (days 152..191)
    explore_start_idx = int(df[df["date"] == dates[60]].index[0])
    diag_start_idx = int(df[df["date"] == dates[152]].index[0])
    df_explore = df.iloc[:diag_start_idx].copy().reset_index(drop=True)

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "strategy_registry.db"))
    registry = StrategyRegistry(db_path)

    # 1. Random Search (120 evaluations)
    rng_rand = np.random.default_rng(42)
    rand_results = []
    for _ in range(120):
        cand = sample_random_strategy(rng_rand)
        res = evaluate_candidate(cand, df_explore, explore_start_idx, dataset_sha, "2024_explore_92d", registry, "random", None, "phase1_random")
        rand_results.append(res)

    # 2. Evolutionary Search (40 initial + 80 mutated = 120 evaluations)
    rng_evo = np.random.default_rng(42)
    evo_results = []
    for _ in range(40):
        cand = sample_random_strategy(rng_evo)
        res = evaluate_candidate(cand, df_explore, explore_start_idx, dataset_sha, "2024_explore_92d", registry, "evo_init", None, "phase1_evo")
        evo_results.append(res)

    valid_pool = [r for r in evo_results if r["metrics"].get("trade_count", 0) >= 30]
    valid_pool.sort(key=lambda x: x["metrics"]["annualized_sharpe"], reverse=True)
    top_parents = valid_pool[:10] if len(valid_pool) >= 10 else evo_results[:10]

    for _ in range(80):
        parent = top_parents[int(rng_evo.integers(0, len(top_parents)))]
        child = mutate_strategy(rng_evo, parent["strategy"])
        res = evaluate_candidate(child, df_explore, explore_start_idx, dataset_sha, "2024_explore_92d", registry, "evo_mutate", parent["strategy_hash"], "phase1_evo")
        evo_results.append(res)
        if res["metrics"].get("trade_count", 0) >= 30:
            valid_pool.append(res)
            valid_pool.sort(key=lambda x: x["metrics"]["annualized_sharpe"], reverse=True)
            top_parents = valid_pool[:10]

    def summarize_run(name: str, res_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        hits = sum(1 for r in res_list if r["cache_hit"])
        saved_sec = sum(r["saved_sec"] for r in res_list)
        unique_hashes = len(set(r["strategy_hash"] for r in res_list))
        dup_rate = 1.0 - (unique_hashes / len(res_list))
        valid = [r for r in res_list if "annualized_sharpe" in r["metrics"]]
        valid.sort(key=lambda x: x["metrics"]["annualized_sharpe"], reverse=True)
        profitable = sum(1 for r in valid if r["metrics"]["annualized_sharpe"] > 0)
        best = valid[0]["metrics"] if valid else {}
        print(f"  {name:14s} | Evals={len(res_list)} | Unique={unique_hashes} (Dup={dup_rate*100:.1f}%) | CacheHits={hits} (Saved={saved_sec:.2f}s) | Profitable={profitable} | BestSharpe={best.get('annualized_sharpe', 0):+.4f}")
        return {
            "evals": len(res_list), "unique": unique_hashes, "dup_rate": dup_rate,
            "cache_hits": hits, "saved_sec": saved_sec, "profitable": profitable,
            "best_metrics": best, "top_strategy": valid[0]["strategy"] if valid else None,
        }

    print("\n[*] Phase 1 Exploration Comparison (2024 92-day Explore):")
    s_rand = summarize_run("Random_120", rand_results)
    s_evo = summarize_run("Evo_120", evo_results)

    # Evaluate top 1 from each on Diagnostic 40d (held out from selection)
    diag_summary = {}
    for label, s_obj in [("Random_Top1", s_rand), ("Evo_Top1", s_evo)]:
        if s_obj["top_strategy"] is not None:
            st = s_obj["top_strategy"]
            cfg = BacktestConfig(k_sl=st["risk"]["k_sl"], k_tp=st["risk"]["k_tp"], max_hold_bars=st["risk"]["max_hold_bars"])
            ls, ss = evaluate_declarative_signals(df, st)
            _, _, d_sum = run_causal_fx_backtest(df, cfg, GateConfig(), eval_start_idx=diag_start_idx, long_signal=ls, short_signal=ss)
            diag_summary[label] = d_sum
            print(f"  [Diagnostic 40d] {label:12s} | Sharpe={d_sum['annualized_sharpe']:+.4f} | Return={d_sum['total_return_pct']:+.2f}% | MaxDD={d_sum['max_drawdown_pct']:.2f}% | Trades={d_sum['trade_count']}")

    registry.close()
    total_sec = time.time() - t_total
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_sec": total_sec,
        "api_spend_usd": 0.0,
        "random_search": s_rand,
        "evolutionary_search": s_evo,
        "diagnostic_40d": diag_summary,
    }
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase1_exploration_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] Phase 1 Complete in {total_sec:.2f}s. Results saved to {out_path}")


if __name__ == "__main__":
    main()
