"""Phase 1B-2: AlgoForge v1 Small-Scale Evolutionary Exploration (Max 25 Trials, Seed=42)."""

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import BacktestConfig, GateConfig, compute_indicators, run_causal_fx_backtest
from research_pipeline.backtest.strategy_registry import (
    StrategyRegistry,
    compute_algoforge_evaluation_hash,
    compute_algoforge_strategy_hash,
    compute_strategy_hash,
)
from research_pipeline.integration.algoforge_adapter import (
    compute_algoforge_indicators,
    evaluate_algoforge_conditions_vectorized,
    evaluate_algoforge_signals,
    validate_algoforge_strategy,
)

PHASE_1B1_COMMIT = "cfe8126"
ENGINE_VERSION = "causal_fx_backtest_v1.0"
COST_CFG = {"spread": 0.010, "slippage": 0.002}
UNIT_VERSIONS = {"indicators": "algoforge_v1"}


class AlgoForgeEvoGenerator:
    """Type-safe AlgoForge v1 strategy generator & mutator (seed=42)."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_valid(self) -> Dict[str, Any]:
        fam = str(self.rng.choice(["ema_cross", "rsi_extreme", "bb_break", "donchian_break", "macd_cross"]))
        k_sl = float(self.rng.choice([1.5, 2.0, 2.5]))
        k_tp = float(self.rng.choice([2.5, 3.0, 4.0]))
        max_hold = int(self.rng.choice([24, 48, 72]))

        if fam == "ema_cross":
            f, s = int(self.rng.choice([8, 12, 16])), int(self.rng.choice([32, 48, 64]))
            inds = [{"id": "ema_f", "type": "ema", "params": {"period": f}}, {"id": "ema_s", "type": "ema", "params": {"period": s}}]
            le = {"logic": "and", "conditions": [{"left": "ema_f", "op": ">", "right": "ema_s"}]}
            se = {"logic": "and", "conditions": [{"left": "ema_f", "op": "<", "right": "ema_s"}]}
        elif fam == "rsi_extreme":
            p = int(self.rng.choice([10, 14, 21]))
            os_lvl, ob_lvl = int(self.rng.choice([25, 30, 35])), int(self.rng.choice([65, 70, 75]))
            inds = [{"id": "rsi1", "type": "rsi", "params": {"period": p}}]
            le = {"logic": "and", "conditions": [{"left": "rsi1", "op": "<", "right": os_lvl}]}
            se = {"logic": "and", "conditions": [{"left": "rsi1", "op": ">", "right": ob_lvl}]}
        elif fam == "bb_break":
            p, m = int(self.rng.choice([20, 30])), float(self.rng.choice([1.5, 2.0, 2.5]))
            inds = [{"id": "bb1", "type": "bb", "params": {"period": p, "std": m}}]
            le = {"logic": "and", "conditions": [{"left": "close", "op": ">", "right": "bb1_upper"}]}
            se = {"logic": "and", "conditions": [{"left": "close", "op": "<", "right": "bb1_lower"}]}
        elif fam == "donchian_break":
            p = int(self.rng.choice([12, 20, 36]))
            inds = [{"id": "dc1", "type": "donchian", "params": {"period": p}}]
            le = {"logic": "and", "conditions": [{"left": "close", "op": ">=", "right": "dc1_upper"}]}
            se = {"logic": "and", "conditions": [{"left": "close", "op": "<=", "right": "dc1_lower"}]}
        else:  # macd_cross
            inds = [{"id": "macd1", "type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}]
            le = {"logic": "and", "conditions": [{"left": "macd1_line", "op": ">", "right": "macd1_signal"}]}
            se = {"logic": "and", "conditions": [{"left": "macd1_line", "op": "<", "right": "macd1_signal"}]}

        return {
            "symbol": "USDJPY",
            "indicators": inds,
            "long": {"entry": le, "exit": None},
            "short": {"entry": se, "exit": {}},
            "risk": {"k_sl": k_sl, "k_tp": k_tp, "max_hold_bars": max_hold},
        }

    def mutate(self, parent: Dict[str, Any]) -> Dict[str, Any]:
        child = json.loads(json.dumps(parent))
        if self.rng.random() < 0.6:
            child["risk"]["k_sl"] = float(self.rng.choice([1.5, 2.0, 2.5]))
            child["risk"]["k_tp"] = float(self.rng.choice([2.5, 3.0, 4.0]))
            child["risk"]["max_hold_bars"] = int(self.rng.choice([24, 48, 72]))
        else:
            for ind in child["indicators"]:
                if "period" in ind.get("params", {}):
                    ind["params"]["period"] = max(5, int(ind["params"]["period"]) + int(self.rng.choice([-4, 4])))
        return child


def run_pre_contracts(df_sub: pd.DataFrame, snap_sha: str) -> None:
    """Verify hash stability, parameter sensitivity, nested logic & causality before exploration."""
    nested = {
        "symbol": "USDJPY",
        "indicators": [
            {"id": "rsi14", "type": "rsi", "params": {"period": 14}},
            {"id": "bb20", "type": "bb", "params": {"period": 20, "std": 2.0}},
        ],
        "long": {"entry": {"logic": "or", "conditions": [{"left": "rsi14", "op": "<", "right": 35}, {"left": "close", "op": "<=", "right": "bb20_lower"}]}},
        "short": {"entry": {"logic": "and", "conditions": [{"left": "rsi14", "op": ">", "right": 65}]}},
        "risk": {"k_sl": 2.0, "k_tp": 3.0, "max_hold_bars": 48},
    }
    h1 = compute_algoforge_strategy_hash(nested)
    h2 = compute_algoforge_strategy_hash(nested)
    assert h1 == h2, "Hash instability!"
    nested_mod = json.loads(json.dumps(nested))
    nested_mod["risk"]["k_tp"] = 4.0
    assert compute_algoforge_strategy_hash(nested_mod) != h1, "Risk change did not alter hash!"
    eh1 = compute_algoforge_evaluation_hash(h1, snap_sha, "p1", ENGINE_VERSION, COST_CFG, UNIT_VERSIONS)
    eh2 = compute_algoforge_evaluation_hash(h1, snap_sha, "p2", ENGINE_VERSION, COST_CFG, UNIT_VERSIONS)
    assert eh1 != eh2, "Period change did not alter evaluation_hash!"
    assert h1 != compute_strategy_hash(nested), "algoforge_v1 collided with pipeline_v1!"

    af_ind = compute_algoforge_indicators(df_sub, nested["indicators"])
    ls_seq, ss_seq = evaluate_algoforge_signals(af_ind, nested["long"]["entry"], nested["short"]["entry"])
    ls_vec, ss_vec = evaluate_algoforge_conditions_vectorized(af_ind, nested["long"]["entry"], nested["short"]["entry"])
    assert np.array_equal(ls_seq, ls_vec) and np.array_equal(ss_seq, ss_vec), "Nested logic vec != seq!"

    # Causality check at T=300
    df_pert = df_sub.copy()
    df_pert.loc[301:, ["open", "high", "low", "close"]] += 5.0
    af_p = compute_algoforge_indicators(df_pert, nested["indicators"])
    ls_p, ss_p = evaluate_algoforge_conditions_vectorized(af_p, nested["long"]["entry"], nested["short"]["entry"])
    assert np.array_equal(ls_vec[:301], ls_p[:301]) and np.array_equal(ss_vec[:301], ss_p[:301]), "Lookahead leak!"
    print("[PASS] Pre-exploration hash, nested-logic, and causality contracts verified.")


def main():
    t_start = time.time()
    snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    with open(snap_path, "rb") as f:
        snap_sha = hashlib.sha256(f.read()).hexdigest()
    assert snap_sha == "bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0"

    df = pd.read_parquet(snap_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    pl_df = compute_indicators(df, BacktestConfig())
    eval_start_idx = 17185  # 60d warmup -> 132d eval

    run_pre_contracts(df.iloc[:600].copy().reset_index(drop=True), snap_sha)

    gen = AlgoForgeEvoGenerator(seed=42)
    # 3 intentional unsupported candidates + 7 valid initial = 10 initial, then 15 mutations = 25 max trials
    base_v = gen.sample_valid()
    unsupported_3 = [
        {**json.loads(json.dumps(base_v)), "long": {"entry": base_v["long"]["entry"], "exit": {"logic": "and", "conditions": [{"left": "close", "op": ">", "right": 150}]}}},
        {**json.loads(json.dumps(base_v)), "risk": {**base_v["risk"], "trailing_stop": True}},
        {**json.loads(json.dumps(base_v)), "long": {"entry": {"type": "streak", "left": "close", "op": ">", "right": 150, "min_streak": 3}}},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        reg = StrategyRegistry(os.path.join(tmp, "phase1b2_temp.db"))
        seen_in_run = set()
        root_counts: Counter = Counter()
        adopted_pool: List[Dict[str, Any]] = []
        trial_logs: List[Dict[str, Any]] = []
        cap_requests: List[Dict[str, Any]] = []
        fallback_count = 0
        executed_backtests = 0
        cache_hits = 0
        duplicates = 0
        stop_reason = "COMPLETED_25_TRIALS"

        for trial_idx in range(1, 26):
            if trial_idx <= 3:
                cand = unsupported_3[trial_idx - 1]
                parent_root = None
                gen_type = "intentional_unsupported"
            elif trial_idx <= 10:
                cand = gen.sample_valid()
                parent_root = None
                gen_type = "evo_init"
            else:
                eligible_parents = [
                    p for p in adopted_pool
                    if p["metrics"].get("trade_count", 0) >= 30
                    and math.isfinite(p["metrics"].get("annualized_sharpe", float("nan")))
                    and root_counts[p["root_hash"]] < 3
                ]
                eligible_parents.sort(
                    key=lambda x: (
                        x["metrics"]["annualized_sharpe"],
                        x["metrics"]["max_drawdown_pct"],
                        x["metrics"]["trade_count"],
                    ),
                    reverse=True,
                )
                if eligible_parents:
                    top_p = eligible_parents[:5]
                    chosen = top_p[int(gen.rng.integers(0, len(top_p)))]
                    cand = gen.mutate(chosen["strategy"])
                    parent_root = chosen["root_hash"]
                    gen_type = "evo_mutate"
                else:
                    cand = gen.sample_valid()
                    parent_root = None
                    gen_type = "random_fallback"
                    fallback_count += 1

            ok, msg, cap = validate_algoforge_strategy(cand)
            if not ok:
                if cap is not None:
                    cap_requests.append(cap)
                trial_logs.append({"trial": trial_idx, "gen_type": gen_type, "status": "REJECTED_VALIDATION", "reason": msg})
                continue

            s_hash = compute_algoforge_strategy_hash(cand)
            e_hash = compute_algoforge_evaluation_hash(s_hash, snap_sha, "2024_132d", ENGINE_VERSION, COST_CFG, UNIT_VERSIONS)

            if s_hash in seen_in_run:
                duplicates += 1
                cached = reg.lookup(e_hash)
                if cached is not None:
                    cache_hits += 1
                trial_logs.append({"trial": trial_idx, "gen_type": gen_type, "strategy_hash": s_hash, "evaluation_hash": e_hash, "status": "DUPLICATE_SKIPPED", "cache_hit": cached is not None})
                continue

            seen_in_run.add(s_hash)
            root_hash = parent_root or s_hash
            root_counts[root_hash] += 1
            if root_counts[root_hash] > 3:
                stop_reason = f"ABORTED_ROOT_CAP_VIOLATION:{root_hash}"
                break

            cached = reg.lookup(e_hash)
            if cached is not None:
                cache_hits += 1
                trial_logs.append({"trial": trial_idx, "gen_type": gen_type, "strategy_hash": s_hash, "evaluation_hash": e_hash, "root_hash": root_hash, "status": "CACHE_HIT", "metrics": cached["metrics"]})
                continue

            af_ind = compute_algoforge_indicators(df, cand["indicators"])
            ls, ss = evaluate_algoforge_conditions_vectorized(af_ind, cand["long"]["entry"], cand["short"]["entry"])
            cfg = BacktestConfig(k_sl=cand["risk"]["k_sl"], k_tp=cand["risk"]["k_tp"], max_hold_bars=cand["risk"]["max_hold_bars"])
            _, _, summary = run_causal_fx_backtest(pl_df, cfg, GateConfig(), eval_start_idx=eval_start_idx, long_signal=ls, short_signal=ss)
            executed_backtests += 1

            status = "PROMOTED" if (summary["trade_count"] >= 30 and summary["annualized_sharpe"] > 0) else "REJECTED_PERF"
            reg.record(e_hash, s_hash, cand, status, status, summary, gen_type, parent_root, "phase1b2", 0.0)
            entry = {"trial": trial_idx, "gen_type": gen_type, "strategy_hash": s_hash, "evaluation_hash": e_hash, "root_hash": root_hash, "status": status, "metrics": summary, "strategy": cand}
            adopted_pool.append(entry)
            trial_logs.append({"trial": trial_idx, "gen_type": gen_type, "strategy_hash": s_hash, "evaluation_hash": e_hash, "root_hash": root_hash, "status": status, "metrics": summary})

        reg.close()

    elapsed = time.time() - t_start
    print(f"\n=== Phase 1B-2 Complete ({stop_reason}) in {elapsed:.2f}s ===")
    print(f"  Trials=25 | Adopted={len(adopted_pool)} | Duplicates={duplicates} | CacheHits={cache_hits} | Backtests={executed_backtests} | RejectedVal={25 - len(adopted_pool) - duplicates} | CapReqs={len(cap_requests)} | Fallbacks={fallback_count}")
    print(f"  Max root_count={max(root_counts.values()) if root_counts else 0} (cap=3) | API Spend=$0.00")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase1b2_exploration_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase1b1_commit": PHASE_1B1_COMMIT,
            "snapshot_2024_sha256": snap_sha,
            "seed": 42,
            "stop_reason": stop_reason,
            "elapsed_sec": elapsed,
            "api_spend_usd": 0.0,
            "counts": {
                "trials": 25,
                "adopted_unique": len(adopted_pool),
                "duplicates": duplicates,
                "cache_hits": cache_hits,
                "executed_backtests": executed_backtests,
                "rejected_validation": 25 - len(adopted_pool) - duplicates,
                "capability_requests": len(cap_requests),
                "random_fallbacks": fallback_count,
            },
            "root_counts": dict(root_counts),
            "capability_requests": cap_requests,
            "trial_logs": trial_logs,
        }, f, indent=2)
    print(f"[*] Saved to {out_path}")


if __name__ == "__main__":
    main()
