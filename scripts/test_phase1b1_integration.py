"""Phase 1B-1 Integration Verification: Fixed AlgoForge Strategies (No Exploration)."""

import hashlib
import os
import sys
import tempfile
import time

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

AF_EMA_CROSS_FIXED = {
    "symbol": "USDJPY",
    "indicators": [
        {"id": "ema12", "type": "ema", "params": {"period": 12}},
        {"id": "ema48", "type": "ema", "params": {"period": 48}},
    ],
    "long": {"entry": {"logic": "and", "conditions": [{"left": "ema12", "op": ">", "right": "ema48"}]}, "exit": None},
    "short": {"entry": {"logic": "and", "conditions": [{"left": "ema12", "op": "<", "right": "ema48"}]}, "exit": {}},
    "risk": {"k_sl": 2.0, "k_tp": 4.0, "max_hold_bars": 48},
}

AF_RSI_BB_FIXED = {
    "symbol": "USDJPY",
    "indicators": [
        {"id": "rsi14", "type": "rsi", "params": {"period": 14}},
        {"id": "bb20", "type": "bb", "params": {"period": 20, "std": 2.0}},
    ],
    "long": {"entry": {"logic": "and", "conditions": [{"left": "rsi14", "op": "<", "right": 30}, {"left": "close", "op": "<", "right": "bb20_lower"}]}},
    "short": {"entry": {"logic": "and", "conditions": [{"left": "rsi14", "op": ">", "right": 70}, {"left": "close", "op": ">", "right": "bb20_upper"}]}},
    "risk": {"k_sl": 2.0, "k_tp": 3.0, "max_hold_bars": 48},
}


def main():
    snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    with open(snap_path, "rb") as f:
        snap_sha = hashlib.sha256(f.read()).hexdigest()
    df = pd.read_parquet(snap_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    pl_df = compute_indicators(df, BacktestConfig())

    # 1. Validation & Explicit Rejection Tests
    print("=== 1. Validation & Capability Request Tests ===")
    for name, st in [("af_ema_cross_fixed", AF_EMA_CROSS_FIXED), ("af_rsi_bb_fixed", AF_RSI_BB_FIXED)]:
        ok, msg, cap = validate_algoforge_strategy(st)
        assert ok and msg == "OK" and cap is None, f"Validation failed for {name}: {msg}"
        print(f"  [PASS] {name}: valid (exit=None/{{}} accepted)")

    bad_exit = {**AF_EMA_CROSS_FIXED, "long": {"entry": AF_EMA_CROSS_FIXED["long"]["entry"], "exit": {"left": "ema12", "op": "<", "right": "ema48"}}}
    ok_e, msg_e, cap_e = validate_algoforge_strategy(bad_exit)
    assert not ok_e and "UNSUPPORTED_EXIT_CONDITION" in msg_e and cap_e is not None
    print(f"  [PASS] Non-empty exit rejected: {msg_e} (Capability Request generated)")

    bad_risk = {**AF_EMA_CROSS_FIXED, "risk": {**AF_EMA_CROSS_FIXED["risk"], "trailing_stop": True}}
    ok_r, msg_r, cap_r = validate_algoforge_strategy(bad_risk)
    assert not ok_r and "UNSUPPORTED_RISK_PARAM" in msg_r and cap_r is not None
    print(f"  [PASS] Unsupported risk rejected: {msg_r} (Capability Request generated)")

    # 2. Vectorized vs Sequential Equivalence (1,500-bar slice)
    print("\n=== 2. Vectorized vs Sequential Equivalence (1,500 bars) ===")
    sub = df.iloc[:1500].copy().reset_index(drop=True)
    for name, st in [("af_ema_cross_fixed", AF_EMA_CROSS_FIXED), ("af_rsi_bb_fixed", AF_RSI_BB_FIXED)]:
        af_ind = compute_algoforge_indicators(sub, st["indicators"])
        ls_seq, ss_seq = evaluate_algoforge_signals(af_ind, st["long"]["entry"], st["short"]["entry"])
        ls_vec, ss_vec = evaluate_algoforge_conditions_vectorized(af_ind, st["long"]["entry"], st["short"]["entry"])
        assert np.array_equal(ls_seq, ls_vec), f"Long mismatch in {name}"
        assert np.array_equal(ss_seq, ss_vec), f"Short mismatch in {name}"
        print(f"  [PASS] {name}: Vectorized == Sequential 100.00% (1,500 bars)")

    # 3. Causality / Future-Perturbation Invariance
    print("\n=== 3. Causality Test (T=500) ===")
    sub_pert = sub.copy()
    sub_pert.loc[501:, ["open", "high", "low", "close"]] += 5.0
    for name, st in [("af_ema_cross_fixed", AF_EMA_CROSS_FIXED), ("af_rsi_bb_fixed", AF_RSI_BB_FIXED)]:
        af_o = compute_algoforge_indicators(sub, st["indicators"])
        af_p = compute_algoforge_indicators(sub_pert, st["indicators"])
        ls_o, ss_o = evaluate_algoforge_conditions_vectorized(af_o, st["long"]["entry"], st["short"]["entry"])
        ls_p, ss_p = evaluate_algoforge_conditions_vectorized(af_p, st["long"]["entry"], st["short"]["entry"])
        assert np.array_equal(ls_o[:501], ls_p[:501]) and np.array_equal(ss_o[:501], ss_p[:501])
        print(f"  [PASS] {name}: 0 lookahead diffs <= T=500")

    # 4. Hash Isolation & Registry Cache Reuse + Full 55,201-bar Timing Breakdown
    print("\n=== 4. Full 55,201-bar Evaluation, Timing & Registry Cache Reuse ===")
    with tempfile.TemporaryDirectory() as tmp:
        reg = StrategyRegistry(os.path.join(tmp, "test_p1b1.db"))
        h_af1 = compute_algoforge_strategy_hash(AF_EMA_CROSS_FIXED)
        h_pl1 = compute_strategy_hash({"long": {}, "short": {}, "risk": AF_EMA_CROSS_FIXED["risk"]})
        assert h_af1 != h_pl1, "Hash collision between algoforge_v1 and pipeline_v1!"

        for name, st in [("af_ema_cross_fixed", AF_EMA_CROSS_FIXED), ("af_rsi_bb_fixed", AF_RSI_BB_FIXED)]:
            s_hash = compute_algoforge_strategy_hash(st)
            e_hash = compute_algoforge_evaluation_hash(s_hash, snap_sha, "2024_full_132d", "causal_fx_backtest_v1.0", {"spread": 0.010, "slippage": 0.002}, {"indicators": "algoforge_v1"})

            t0 = time.time()
            af_full = compute_algoforge_indicators(df, st["indicators"])
            t_ind = time.time() - t0

            t1 = time.time()
            ls_full, ss_full = evaluate_algoforge_conditions_vectorized(af_full, st["long"]["entry"], st["short"]["entry"])
            t_cond = time.time() - t1

            t2 = time.time()
            cfg = BacktestConfig(k_sl=st["risk"]["k_sl"], k_tp=st["risk"]["k_tp"], max_hold_bars=st["risk"]["max_hold_bars"])
            _, _, summary = run_causal_fx_backtest(pl_df, cfg, GateConfig(), eval_start_idx=17185, long_signal=ls_full, short_signal=ss_full)
            t_bt = time.time() - t2

            t3 = time.time()
            reg.record(e_hash, s_hash, st, "PROMOTED" if summary["annualized_sharpe"] > 0 else "REJECTED_PERF", "PHASE1B1_TEST", summary, "fixed", None, "p1b1", t_ind + t_cond + t_bt)
            hit = reg.lookup(e_hash)
            t_reg = time.time() - t3
            assert hit is not None and hit["metrics"]["trade_count"] == summary["trade_count"]
            print(f"  {name:18s} | Trades={summary['trade_count']:4d} | Sharpe={summary['annualized_sharpe']:+.4f} | Eq={summary['final_equity']:.2f}")
            print(f"    Timing (55,201 bars): ind={t_ind:.4f}s, cond_vec={t_cond:.4f}s, backtest={t_bt:.4f}s, reg={t_reg:.4f}s | Total={t_ind+t_cond+t_bt+t_reg:.4f}s | CacheHit=OK")
        reg.close()

    print("\nALL PHASE 1B-1 TESTS PASSED.")


if __name__ == "__main__":
    main()
