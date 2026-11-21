"""Phase 3: Diversity Control (B), Multi-Period CV (C), and Combined (D) vs Phase 2 Baseline (A).

Runs B, C, D for 500 target adoptions x 3 seeds (42, 101, 202), reusing A from Phase 2.
"""

import hashlib
import json
import os
import sys
import time
from collections import Counter
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

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from run_phase1_exploration import mutate_strategy, sample_random_strategy

ENGINE_VERSION = "causal_fx_backtest_v1.0"


def _calc_sharpe(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return -999.0
    std = float(np.std(rets, ddof=1))
    if std < 1e-12:
        return -999.0
    return float(np.mean(rets) * 252.0) / (std * np.sqrt(252.0))


def evaluate_p3(
    cand: Dict[str, Any],
    df_explore: pd.DataFrame,
    explore_start_idx: int,
    p2_split_date: Any,
    dataset_sha: str,
    registry: StrategyRegistry,
    root_hash: Optional[str],
    parent_hash: Optional[str],
    gen_method: str,
    exp_id: str,
) -> Dict[str, Any]:
    s_hash = compute_strategy_hash(cand)
    cost_cfg = {"spread": 0.010, "slippage": 0.002}
    # Use a v1.1 eval key so 2-period CV metrics are cached cleanly
    e_hash = compute_evaluation_hash(s_hash, dataset_sha, "2024_explore_92d_cv2", ENGINE_VERSION, cost_cfg)

    cached = registry.lookup(e_hash)
    if cached is not None:
        return {
            "strategy_hash": s_hash,
            "root_hash": root_hash or s_hash,
            "cache_hit": True,
            "executed_backtest": False,
            "status": cached["status"],
            "metrics": cached["metrics"],
            "strategy": cand,
        }

    t0 = time.time()
    ok, msg = validate_declarative_strategy(cand)
    if not ok:
        elapsed = time.time() - t0
        registry.record(e_hash, s_hash, cand, "REJECTED_SCHEMA", msg, {}, gen_method, parent_hash, exp_id, elapsed)
        return {"strategy_hash": s_hash, "root_hash": root_hash or s_hash, "cache_hit": False, "executed_backtest": False, "status": "REJECTED_SCHEMA", "metrics": {}, "strategy": cand}

    cfg = BacktestConfig(k_sl=cand["risk"]["k_sl"], k_tp=cand["risk"]["k_tp"], max_hold_bars=cand["risk"]["max_hold_bars"])
    ls, ss = evaluate_declarative_signals(df_explore, cand)
    _, daily_df, summary = run_causal_fx_backtest(
        df_explore, cfg, GateConfig(), eval_start_idx=explore_start_idx, long_signal=ls, short_signal=ss
    )
    elapsed = time.time() - t0

    # Compute 2-period CV Sharpes (Part 1: first 46d, Part 2: second 46d)
    p1_rets = daily_df[daily_df["date"] < p2_split_date]["net_return"].values
    p2_rets = daily_df[daily_df["date"] >= p2_split_date]["net_return"].values
    sh_p1 = _calc_sharpe(p1_rets)
    sh_p2 = _calc_sharpe(p2_rets)
    summary["sharpe_p1"] = sh_p1
    summary["sharpe_p2"] = sh_p2
    summary["min_cv_sharpe"] = min(sh_p1, sh_p2)

    if summary["trade_count"] < 30:
        status, reason = "REJECTED_PERF", "REJECTED_LOW_TRADES"
    elif summary["annualized_sharpe"] <= 0.0:
        status, reason = "REJECTED_PERF", "REJECTED_NEGATIVE_SHARPE"
    else:
        status, reason = "PROMOTED", "PASS_MIN_CRITERIA"

    registry.record(e_hash, s_hash, cand, status, reason, summary, gen_method, parent_hash, exp_id, elapsed)
    return {
        "strategy_hash": s_hash, "root_hash": root_hash or s_hash,
        "cache_hit": False, "executed_backtest": True,
        "status": status, "metrics": summary, "strategy": cand,
    }


def run_condition(
    cond_name: str,
    use_diversity: bool,
    use_cv: bool,
    seed: int,
    df_full: pd.DataFrame,
    df_explore: pd.DataFrame,
    explore_start_idx: int,
    diag_start_idx: int,
    p2_split_date: Any,
    dataset_sha: str,
    registry: StrategyRegistry,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    seen_in_run = set()
    root_counts: Counter = Counter()
    results: List[Dict[str, Any]] = []
    gen_attempts = 0
    retry_failures = 0

    def score_fn(r: Dict[str, Any]) -> float:
        m = r["metrics"]
        if not m or m.get("trade_count", 0) < 30:
            return -9999.0
        return m.get("min_cv_sharpe", -999.0) if use_cv else m.get("annualized_sharpe", -999.0)

    # 100 initial candidates
    for _ in range(100):
        cand = None
        for _try in range(10 if use_diversity else 1):
            gen_attempts += 1
            c_try = sample_random_strategy(rng)
            sh_try = compute_strategy_hash(c_try)
            if not use_diversity or (sh_try not in seen_in_run and not registry.is_strategy_rejected(sh_try)):
                cand = c_try
                break
        if cand is None:
            retry_failures += 1
            continue
        seen_in_run.add(compute_strategy_hash(cand))
        res = evaluate_p3(cand, df_explore, explore_start_idx, p2_split_date, dataset_sha, registry, None, None, f"p3_{cond_name}_init", f"p3_{cond_name}_s{seed}")
        root_counts[res["root_hash"]] += 1
        results.append(res)

    pool = [r for r in results if r["metrics"].get("trade_count", 0) >= 30]
    pool.sort(key=score_fn, reverse=True)

    # 400 mutations
    for _ in range(400):
        eligible_parents = [r for r in pool if (not use_diversity or root_counts[r["root_hash"]] < 50)]
        if not eligible_parents:
            eligible_parents = results
        top_p = eligible_parents[:15]

        cand = None
        chosen_parent = None
        for _try in range(10 if use_diversity else 1):
            gen_attempts += 1
            parent = top_p[int(rng.integers(0, len(top_p)))]
            c_try = mutate_strategy(rng, parent["strategy"])
            sh_try = compute_strategy_hash(c_try)
            if not use_diversity or (sh_try not in seen_in_run and not registry.is_strategy_rejected(sh_try)):
                cand = c_try
                chosen_parent = parent
                break
        if cand is None or chosen_parent is None:
            retry_failures += 1
            continue

        seen_in_run.add(compute_strategy_hash(cand))
        res = evaluate_p3(cand, df_explore, explore_start_idx, p2_split_date, dataset_sha, registry, chosen_parent["root_hash"], chosen_parent["strategy_hash"], f"p3_{cond_name}_mut", f"p3_{cond_name}_s{seed}")
        root_counts[res["root_hash"]] += 1
        results.append(res)
        if res["metrics"].get("trade_count", 0) >= 30:
            pool.append(res)
            pool.sort(key=score_fn, reverse=True)

    valid = [r for r in results if r["metrics"].get("trade_count", 0) >= 30]
    valid.sort(key=score_fn, reverse=True)
    best_sh = valid[0]["metrics"]["annualized_sharpe"] if valid else 0.0
    med_sh = float(np.median([r["metrics"]["annualized_sharpe"] for r in valid])) if valid else 0.0
    top1_root = root_counts.most_common(1)[0][1] / max(len(results), 1) if root_counts else 0.0

    diag_res = {}
    if valid:
        st = valid[0]["strategy"]
        cfg = BacktestConfig(k_sl=st["risk"]["k_sl"], k_tp=st["risk"]["k_tp"], max_hold_bars=st["risk"]["max_hold_bars"])
        ls, ss = evaluate_declarative_signals(df_full, st)
        _, _, diag_res = run_causal_fx_backtest(df_full, cfg, GateConfig(), eval_start_idx=diag_start_idx, long_signal=ls, short_signal=ss)

    return {
        "attempts": gen_attempts,
        "adopted": len(results),
        "retry_failures": retry_failures,
        "unique": len(set(r["strategy_hash"] for r in results)),
        "executed_backtests": sum(1 for r in results if r["executed_backtest"]),
        "cache_hits": sum(1 for r in results if r["cache_hit"]),
        "top1_root_share": top1_root,
        "best_explore_sharpe": best_sh,
        "median_explore_sharpe": med_sh,
        "diag_sharpe": diag_res.get("annualized_sharpe", 0.0),
        "diag_return_pct": diag_res.get("total_return_pct", 0.0),
        "diag_trades": diag_res.get("trade_count", 0),
    }


def main():
    t0 = time.time()
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
    p2_split_date = dates[106]  # 46d + 46d split
    diag_start_idx = int(df[df["date"] == dates[152]].index[0])
    df_explore = df.iloc[:diag_start_idx].copy().reset_index(drop=True)

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "strategy_registry.db"))
    registry = StrategyRegistry(db_path)

    # Load Phase 2 A (Baseline)
    p2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase2_scale_observation_results.json"))
    with open(p2_path, "r", encoding="utf-8") as f:
        p2_data = json.load(f)

    seeds = [42, 101, 202]
    summary_all = {}
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        a_base = p2_data["seeds"][f"seed_{seed}"]["evolution"]
        print(f"  A (Baseline)  | Adopt=500 | Uniq={a_base['unique']} | Exec={a_base['executed_backtest']} | Top1Root={a_base['top1_root_share']*100:.1f}% | BestExp={a_base['best_sharpe']:+.3f} | Diag={a_base['diag_top1'].get('annualized_sharpe',0):+.3f} (Tr={a_base['diag_top1'].get('trade_count',0)})")

        res_b = run_condition("B_Div", True, False, seed, df, df_explore, explore_start_idx, diag_start_idx, p2_split_date, dataset_sha, registry)
        print(f"  B (Diversity) | Adopt={res_b['adopted']} (Fail={res_b['retry_failures']}) | Uniq={res_b['unique']} | Exec={res_b['executed_backtests']} | Top1Root={res_b['top1_root_share']*100:.1f}% | BestExp={res_b['best_explore_sharpe']:+.3f} | Diag={res_b['diag_sharpe']:+.3f} (Tr={res_b['diag_trades']})")

        res_c = run_condition("C_CV", False, True, seed, df, df_explore, explore_start_idx, diag_start_idx, p2_split_date, dataset_sha, registry)
        print(f"  C (2-Per CV)  | Adopt={res_c['adopted']} | Uniq={res_c['unique']} | Exec={res_c['executed_backtests']} | Top1Root={res_c['top1_root_share']*100:.1f}% | BestExp={res_c['best_explore_sharpe']:+.3f} | Diag={res_c['diag_sharpe']:+.3f} (Tr={res_c['diag_trades']})")

        res_d = run_condition("D_Comb", True, True, seed, df, df_explore, explore_start_idx, diag_start_idx, p2_split_date, dataset_sha, registry)
        print(f"  D (Combined)  | Adopt={res_d['adopted']} (Fail={res_d['retry_failures']}) | Uniq={res_d['unique']} | Exec={res_d['executed_backtests']} | Top1Root={res_d['top1_root_share']*100:.1f}% | BestExp={res_d['best_explore_sharpe']:+.3f} | Diag={res_d['diag_sharpe']:+.3f} (Tr={res_d['diag_trades']})")

        summary_all[f"seed_{seed}"] = {"A": a_base, "B": res_b, "C": res_c, "D": res_d}

    registry.close()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase3_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time() - t0, "api_spend_usd": 0.0, "seeds": summary_all}, f, indent=2)
    print(f"\n[*] Phase 3 Complete in {time.time()-t0:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
