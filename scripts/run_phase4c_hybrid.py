"""Phase 4-C: Hybrid Exploration (LLM Initial Population + Phase 3 Condition D Evolution).

Compares 4 conditions (100 candidate budget each = 20 initial + 80 Evolution):
- A: Random (20) -> Evolution (80)
- B: Claude Code (20) -> Evolution (80)
- C: Gemini Antigravity (20) -> Evolution (80)
- D: Mixed Claude (10) + Antigravity (10) -> Evolution (80)

Separates initial 20 MinCV distribution from post-evolution 80 MinCV distribution and tracks delta_top1_min_cv.
"""

import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import BacktestConfig, GateConfig, compute_indicators, run_causal_fx_backtest
from research_pipeline.backtest.candidate_generators import AgentJSONGenerator, GenerationContext
from research_pipeline.backtest.declarative_strategy import build_causal_htf_features, evaluate_declarative_signals
from research_pipeline.backtest.strategy_registry import StrategyRegistry, compute_strategy_hash

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from run_phase1_exploration import mutate_strategy, sample_random_strategy
from run_phase3_diversity_and_cv import evaluate_p3


def run_hybrid_condition(
    cond_name: str,
    init_strategies: List[Dict[str, Any]],
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
    init_results: List[Dict[str, Any]] = []
    evo_results: List[Dict[str, Any]] = []
    gen_attempts = 0
    retry_failures = 0

    def score_fn(r: Dict[str, Any]) -> float:
        m = r["metrics"]
        if not m or m.get("trade_count", 0) < 30:
            return -9999.0
        return m.get("min_cv_sharpe", -999.0)

    # 1. Evaluate 20 initial candidates (included in 10% = 10-per-root limit)
    for st in init_strategies:
        gen_attempts += 1
        sh = compute_strategy_hash(st)
        seen_in_run.add(sh)
        res = evaluate_p3(st, df_explore, explore_start_idx, p2_split_date, dataset_sha, registry, None, None, f"p4c_{cond_name}_init", f"p4c_{cond_name}")
        root_counts[res["root_hash"]] += 1
        init_results.append(res)

    all_results = list(init_results)
    pool = [r for r in all_results if r["metrics"].get("trade_count", 0) >= 30]
    pool.sort(key=score_fn, reverse=True)

    # 2. Subsequent 80 Evolution candidates (Phase 3 Condition D: diversity + 10% root cap + MinCV)
    for _ in range(80):
        eligible_parents = [r for r in pool if root_counts[r["root_hash"]] < 10]
        if not eligible_parents:
            eligible_parents = all_results
        top_p = eligible_parents[:10]

        cand = None
        chosen_parent = None
        for _try in range(10):
            gen_attempts += 1
            parent = top_p[int(rng.integers(0, len(top_p)))]
            c_try = mutate_strategy(rng, parent["strategy"])
            sh_try = compute_strategy_hash(c_try)
            if sh_try not in seen_in_run and not registry.is_strategy_rejected(sh_try):
                cand = c_try
                chosen_parent = parent
                break
        if cand is None or chosen_parent is None:
            retry_failures += 1
            continue

        seen_in_run.add(compute_strategy_hash(cand))
        res = evaluate_p3(cand, df_explore, explore_start_idx, p2_split_date, dataset_sha, registry, chosen_parent["root_hash"], chosen_parent["strategy_hash"], f"p4c_{cond_name}_evo", f"p4c_{cond_name}")
        root_counts[res["root_hash"]] += 1
        evo_results.append(res)
        all_results.append(res)
        if res["metrics"].get("trade_count", 0) >= 30:
            pool.append(res)
            pool.sort(key=score_fn, reverse=True)

    def dist_stats(rlist: List[Dict[str, Any]]) -> Dict[str, Any]:
        elig = [r for r in rlist if r["metrics"].get("trade_count", 0) >= 30]
        mcvs = [r["metrics"]["min_cv_sharpe"] for r in elig]
        pos_sh = sum(1 for r in elig if r["metrics"]["annualized_sharpe"] > 0)
        return {
            "count": len(rlist),
            "eligible": len(elig),
            "positive_sharpe": pos_sh,
            "median_min_cv": float(np.median(mcvs)) if mcvs else -999.0,
            "max_min_cv": float(np.max(mcvs)) if mcvs else -999.0,
        }

    init_stats = dist_stats(init_results)
    evo_stats = dist_stats(evo_results)
    final_stats = dist_stats(all_results)
    delta_top1 = final_stats["max_min_cv"] - init_stats["max_min_cv"]

    top1_root = root_counts.most_common(1)[0][1] / max(len(all_results), 1) if root_counts else 0.0
    valid_all = [r for r in all_results if r["metrics"].get("trade_count", 0) >= 30]
    valid_all.sort(key=score_fn, reverse=True)
    diag_res = {}
    if valid_all:
        st = valid_all[0]["strategy"]
        cfg = BacktestConfig(k_sl=st["risk"]["k_sl"], k_tp=st["risk"]["k_tp"], max_hold_bars=st["risk"]["max_hold_bars"])
        ls, ss = evaluate_declarative_signals(df_full, st)
        _, _, diag_res = run_causal_fx_backtest(df_full, cfg, GateConfig(), eval_start_idx=diag_start_idx, long_signal=ls, short_signal=ss)

    return {
        "attempts": gen_attempts,
        "adopted": len(all_results),
        "retry_failures": retry_failures,
        "unique": len(set(r["strategy_hash"] for r in all_results)),
        "executed_backtests": sum(1 for r in all_results if r["executed_backtest"]),
        "cache_hits": sum(1 for r in all_results if r["cache_hit"]),
        "top1_root_share": top1_root,
        "init_20": init_stats,
        "evo_80": evo_stats,
        "final_100": final_stats,
        "delta_top1_min_cv": delta_top1,
        "diag_top1_ref": diag_res,
    }


def main():
    t0 = time.time()
    cc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "agent_batches", "claude_code_p4c_20.json"))
    if os.path.exists(cc_path):
        raw_b = open(cc_path, "rb").read()
        if raw_b.startswith(b"\xff\xfe") or raw_b.startswith(b"\xfe\xff"):
            txt = raw_b.decode("utf-16").strip()
            if txt.startswith("```"):
                txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            open(cc_path, "w", encoding="utf-8").write(txt)

    ag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "agent_batches", "antigravity_p4c_20.json"))

    cc_cands = [gc.strategy for gc in AgentJSONGenerator(cc_path, "claude_code").generate(GenerationContext(), 20)]
    ag_cands = [gc.strategy for gc in AgentJSONGenerator(ag_path, "gemini_antigravity").generate(GenerationContext(), 20)]
    rng_init = np.random.default_rng(777)
    rand_cands = [sample_random_strategy(rng_init) for _ in range(20)]
    mixed_cands = cc_cands[:10] + ag_cands[:10]

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

    print("\n=== Phase 4-C: Hybrid Exploration Comparison (100-Candidate Budget Each) ===")
    reports = {}
    for cond, init_list in [("A_Random", rand_cands), ("B_Claude", cc_cands), ("C_Antigravity", ag_cands), ("D_Mixed", mixed_cands)]:
        r = run_hybrid_condition(cond, init_list, 42, df, df_explore, explore_start_idx, diag_start_idx, p2_split_date, dataset_sha, registry)
        reports[cond] = r
        print(f"  {cond:13s} | Adopt={r['adopted']} (Uniq={r['unique']}, Exec={r['executed_backtests']}, Cache={r['cache_hits']}) | Top1Root={r['top1_root_share']*100:.1f}%")
        print(f"    Init20: MedMinCV={r['init_20']['median_min_cv']:+.3f}, MaxMinCV={r['init_20']['max_min_cv']:+.3f}, Prof={r['init_20']['positive_sharpe']}/{r['init_20']['eligible']}")
        print(f"    Evo80 : MedMinCV={r['evo_80']['median_min_cv']:+.3f}, MaxMinCV={r['evo_80']['max_min_cv']:+.3f}, Prof={r['evo_80']['positive_sharpe']}/{r['evo_80']['eligible']}")
        print(f"    Final : MaxMinCV={r['final_100']['max_min_cv']:+.3f} (Delta={r['delta_top1_min_cv']:+.3f}) | Diag40d(Ref)={r['diag_top1_ref'].get('annualized_sharpe',0):+.3f} (Tr={r['diag_top1_ref'].get('trade_count',0)})")

    registry.close()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase4c_hybrid_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time() - t0, "api_spend_usd": 0.0, "conditions": reports}, f, indent=2)
    print(f"\n[*] Phase 4-C Complete in {time.time()-t0:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
