"""Phase 2: Scale Observation (500 evals x 2 methods x 3 seeds = 3,000 submissions).

Observes:
- Submitted candidates vs Unique candidates vs Actual new backtests executed
- Intra-run duplicates vs Cross-run StrategyRegistry cache hits
- Diversity, lineage concentration, generation-by-generation stagnation
- Exploration (92d) vs Diagnostic (40d) generalization gap
"""

import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

# Reuse generators from Phase 1
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from run_phase1_exploration import mutate_strategy, sample_random_strategy

ENGINE_VERSION = "causal_fx_backtest_v1.0"


def evaluate_with_tracking(
    cand: Dict[str, Any],
    df_explore: pd.DataFrame,
    explore_start_idx: int,
    dataset_sha: str,
    registry: StrategyRegistry,
    seen_in_run: set,
    gen_method: str,
    parent_hash: Optional[str],
    root_hash: Optional[str],
    exp_id: str,
) -> Dict[str, Any]:
    s_hash = compute_strategy_hash(cand)
    is_intra_dup = s_hash in seen_in_run
    seen_in_run.add(s_hash)

    cost_cfg = {"spread": 0.010, "slippage": 0.002}
    e_hash = compute_evaluation_hash(s_hash, dataset_sha, "2024_explore_92d", ENGINE_VERSION, cost_cfg)

    cached = registry.lookup(e_hash)
    if cached is not None:
        return {
            "strategy_hash": s_hash,
            "root_hash": root_hash or s_hash,
            "intra_dup": is_intra_dup,
            "cache_hit": True,
            "executed_backtest": False,
            "status": cached["status"],
            "metrics": cached["metrics"],
            "saved_sec": cached["elapsed_sec"],
            "strategy": cand,
        }

    t0 = time.time()
    ok, msg = validate_declarative_strategy(cand)
    if not ok:
        elapsed = time.time() - t0
        registry.record(e_hash, s_hash, cand, "REJECTED_SCHEMA", msg, {}, gen_method, parent_hash, exp_id, elapsed)
        return {
            "strategy_hash": s_hash, "root_hash": root_hash or s_hash,
            "intra_dup": is_intra_dup, "cache_hit": False, "executed_backtest": False,
            "status": "REJECTED_SCHEMA", "metrics": {}, "saved_sec": 0.0, "strategy": cand,
        }

    cfg = BacktestConfig(
        k_sl=cand["risk"]["k_sl"],
        k_tp=cand["risk"]["k_tp"],
        max_hold_bars=cand["risk"]["max_hold_bars"],
    )
    ls, ss = evaluate_declarative_signals(df_explore, cand)
    _, _, summary = run_causal_fx_backtest(
        df_explore, cfg, GateConfig(), eval_start_idx=explore_start_idx,
        long_signal=ls, short_signal=ss,
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
        "strategy_hash": s_hash, "root_hash": root_hash or s_hash,
        "intra_dup": is_intra_dup, "cache_hit": False, "executed_backtest": True,
        "status": status, "metrics": summary, "saved_sec": 0.0, "strategy": cand,
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

    explore_start_idx = int(df[df["date"] == dates[60]].index[0])
    diag_start_idx = int(df[df["date"] == dates[152]].index[0])
    df_explore = df.iloc[:diag_start_idx].copy().reset_index(drop=True)

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "strategy_registry.db"))
    registry = StrategyRegistry(db_path)

    seeds = [42, 101, 202]
    all_seed_reports = {}

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        # 1. Random Search (500)
        rng_r = np.random.default_rng(seed)
        seen_r = set()
        res_r = []
        for _ in range(500):
            cand = sample_random_strategy(rng_r)
            res_r.append(evaluate_with_tracking(cand, df_explore, explore_start_idx, dataset_sha, registry, seen_r, "random", None, None, f"p2_rand_s{seed}"))

        # 2. Evolutionary Search (100 init + 400 mutate = 500)
        rng_e = np.random.default_rng(seed)
        seen_e = set()
        res_e = []
        for _ in range(100):
            cand = sample_random_strategy(rng_e)
            res_e.append(evaluate_with_tracking(cand, df_explore, explore_start_idx, dataset_sha, registry, seen_e, "evo_init", None, None, f"p2_evo_s{seed}"))

        pool = [r for r in res_e if r["metrics"].get("trade_count", 0) >= 30]
        pool.sort(key=lambda x: x["metrics"]["annualized_sharpe"], reverse=True)
        top_p = pool[:15] if len(pool) >= 15 else res_e[:15]

        for _ in range(400):
            parent = top_p[int(rng_e.integers(0, len(top_p)))]
            child = mutate_strategy(rng_e, parent["strategy"])
            r = evaluate_with_tracking(child, df_explore, explore_start_idx, dataset_sha, registry, seen_e, "evo_mutate", parent["strategy_hash"], parent["root_hash"], f"p2_evo_s{seed}")
            res_e.append(r)
            if r["metrics"].get("trade_count", 0) >= 30:
                pool.append(r)
                pool.sort(key=lambda x: x["metrics"]["annualized_sharpe"], reverse=True)
                top_p = pool[:15]

        def analyze(label: str, rlist: List[Dict[str, Any]]) -> Dict[str, Any]:
            n_sub = len(rlist)
            n_uniq = len(set(r["strategy_hash"] for r in rlist))
            n_intra_dup = sum(1 for r in rlist if r["intra_dup"])
            n_cache = sum(1 for r in rlist if r["cache_hit"])
            n_exec = sum(1 for r in rlist if r["executed_backtest"])
            saved_s = sum(r["saved_sec"] for r in rlist)

            valid = [r for r in rlist if "annualized_sharpe" in r["metrics"] and r["metrics"].get("trade_count", 0) >= 30]
            valid.sort(key=lambda x: x["metrics"]["annualized_sharpe"], reverse=True)
            best_sh = valid[0]["metrics"]["annualized_sharpe"] if valid else 0.0
            med_sh = float(np.median([r["metrics"]["annualized_sharpe"] for r in valid])) if valid else 0.0
            n_prof = sum(1 for r in valid if r["metrics"]["annualized_sharpe"] > 0)

            # Lineage concentration (Top-1 root_hash share among promoted/valid)
            roots = Counter(r["root_hash"] for r in rlist)
            top1_root_share = roots.most_common(1)[0][1] / n_sub if roots else 0.0

            # Generation trajectory (chunks of 100)
            traj = []
            for g in range(5):
                chunk = rlist[g * 100 : (g + 1) * 100]
                c_valid = [r["metrics"]["annualized_sharpe"] for r in chunk if "annualized_sharpe" in r["metrics"] and r["metrics"].get("trade_count", 0) >= 30]
                traj.append({
                    "gen": g + 1,
                    "new_exec": sum(1 for r in chunk if r["executed_backtest"]),
                    "best_sharpe": max(c_valid) if c_valid else None,
                    "med_sharpe": float(np.median(c_valid)) if c_valid else None,
                })

            # Diagnostic 40d on Top 1
            diag_res = {}
            if valid:
                st = valid[0]["strategy"]
                cfg = BacktestConfig(k_sl=st["risk"]["k_sl"], k_tp=st["risk"]["k_tp"], max_hold_bars=st["risk"]["max_hold_bars"])
                ls, ss = evaluate_declarative_signals(df, st)
                _, _, d_sum = run_causal_fx_backtest(df, cfg, GateConfig(), eval_start_idx=diag_start_idx, long_signal=ls, short_signal=ss)
                diag_res = d_sum

            print(f"  {label:10s} | Sub={n_sub} | Uniq={n_uniq} | IntraDup={n_intra_dup} | CacheHit={n_cache} | Exec={n_exec} | Prof={n_prof} | Best={best_sh:+.3f} | Med={med_sh:+.3f} | Top1Root={top1_root_share*100:.1f}% | DiagSharpe={diag_res.get('annualized_sharpe', 0):+.3f} (Trades={diag_res.get('trade_count', 0)})")
            return {
                "submitted": n_sub, "unique": n_uniq, "intra_dup": n_intra_dup,
                "cache_hit": n_cache, "executed_backtest": n_exec, "saved_sec": saved_s,
                "profitable": n_prof, "best_sharpe": best_sh, "median_sharpe": med_sh,
                "top1_root_share": top1_root_share, "trajectory": traj, "diag_top1": diag_res,
            }

        all_seed_reports[f"seed_{seed}"] = {
            "random": analyze("Random", res_r),
            "evolution": analyze("Evolution", res_e),
        }

    registry.close()
    total_sec = time.time() - t_total
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_sec": total_sec,
        "api_spend_usd": 0.0,
        "seeds": all_seed_reports,
    }
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase2_scale_observation_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] Phase 2 Complete in {total_sec:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
