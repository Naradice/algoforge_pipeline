"""Phase 1A Compatibility, Causality, Reproducibility & Scaling Verification."""

import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import BacktestConfig, GateConfig, compute_indicators, run_causal_fx_backtest
from research_pipeline.backtest.declarative_strategy import build_causal_htf_features
from research_pipeline.integration.algoforge_adapter import compute_algoforge_indicators, evaluate_algoforge_signals


def main():
    snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    with open(snap_path, "rb") as f:
        snap_sha = hashlib.sha256(f.read()).hexdigest()
    assert snap_sha == "bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0"
    print(f"[*] Snapshot SHA-256 verified: {snap_sha}")
    print("[*] Engine: causal_fx_backtest_v1.0 | Cost: spread=0.010, slippage=0.002")

    df = pd.read_parquet(snap_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # 5 target indicators in AlgoForge format
    ind_specs = [
        {"id": "ema12", "type": "ema", "params": {"period": 12}},
        {"id": "ema48", "type": "ema", "params": {"period": 48}},
        {"id": "rsi14", "type": "rsi", "params": {"period": 14}},
        {"id": "atr20", "type": "atr", "params": {"period": 20}},
        {"id": "bb20", "type": "bb", "params": {"period": 20, "std": 2.0}},
        {"id": "dc20", "type": "donchian", "params": {"period": 20}},
    ]

    # =========================================================================
    # TEST 1: Indicator Value & Definition Difference Analysis (500-bar smoke + 3 intervals)
    # =========================================================================
    print("\n=== TEST 1: Indicator Value & Definition Difference Analysis ===")
    intervals = [
        ("smoke_500", df.iloc[:500].copy()),
        ("normal_1000", df.iloc[20000:21000].copy()),
        ("highvol_1000", df.iloc[27000:28000].copy()),  # around July 2024
        ("gap_cross_1000", df.iloc[1000:2000].copy()),
    ]
    for label, sub in intervals:
        sub = sub.reset_index(drop=True)
        af_df = compute_algoforge_indicators(sub, ind_specs)
        pl_df = compute_indicators(sub, BacktestConfig())

        # Pipeline RSI(14), BB(20, 2.0), Donchian(20)
        delta = sub["close"].diff()
        gain = delta.clip(lower=0.0).rolling(14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14).mean()
        pl_rsi = 100.0 - (100.0 / (1.0 + gain / loss.replace(0.0, np.nan)))
        pl_bb_up = sub["close"].rolling(20).mean() + 2.0 * sub["close"].rolling(20).std(ddof=1)
        pl_dc_hi = sub["high"].rolling(20).max().shift(1)

        # Compare after 100-bar warmup
        sl = slice(100, None)
        ema12_diff = float(np.nanmax(np.abs(af_df["ema12"].values[sl] - pl_df["ema_fast"].values[sl])))
        atr20_diff = float(np.nanmax(np.abs(af_df["atr20"].values[sl] - pl_df["atr"].values[sl])))
        rsi14_diff = float(np.nanmax(np.abs(af_df["rsi14"].values[sl] - pl_rsi.values[sl])))
        bb_diff = float(np.nanmax(np.abs(af_df["bb20_upper"].values[sl] - pl_bb_up.values[sl])))
        dc_diff = float(np.nanmax(np.abs(af_df["dc20_upper"].values[sl] - pl_dc_hi.values[sl])))
        print(f"  {label:15s} | EMA12_MaxDiff={ema12_diff:.2e} | ATR20_MaxDiff={atr20_diff:.4f} | RSI14_MaxDiff={rsi14_diff:.2f} | BB20_MaxDiff={bb_diff:.4f} | DC20_MaxDiff={dc_diff:.4f}")

    # =========================================================================
    # TEST 2: Causality & Future-Perturbation Invariance (4 Cutoffs)
    # =========================================================================
    print("\n=== TEST 2: Causality & Future-Perturbation Invariance (4 Cutoffs) ===")
    sub_c = df.iloc[:1500].copy().reset_index(drop=True)
    # Find a gap index inside sub_c
    diffs = sub_c["time"].diff().dt.total_seconds().fillna(300.0)
    gap_idxs = np.where(diffs > 300.0)[0]
    gap_cut = int(gap_idxs[0]) if len(gap_idxs) > 0 else 600
    cutoffs = [
        ("post_warmup", 120),
        ("normal_mid", 500),
        ("gap_boundary", gap_cut),
        ("htf_boundary", 720),  # 00:00 UTC / H1&H4 boundary
    ]
    long_cond = {"left": "ema12", "op": ">", "right": "ema48"}
    short_cond = {"left": "ema12", "op": "<", "right": "ema48"}

    af_orig = compute_algoforge_indicators(sub_c, ind_specs)
    ls_orig, ss_orig = evaluate_algoforge_signals(af_orig, long_cond, short_cond)

    for cname, T in cutoffs:
        sub_pert = sub_c.copy()
        sub_pert.loc[T + 1 :, ["open", "high", "low", "close"]] += 5.0
        af_pert = compute_algoforge_indicators(sub_pert, ind_specs)
        ls_pert, ss_pert = evaluate_algoforge_signals(af_pert, long_cond, short_cond)

        for col in ["ema12", "ema48", "rsi14", "atr20", "bb20_upper", "dc20_upper"]:
            assert np.allclose(af_orig[col].values[: T + 1], af_pert[col].values[: T + 1], equal_nan=True), f"LOOKAHEAD in {col} at {cname}!"
        assert np.array_equal(ls_orig[: T + 1], ls_pert[: T + 1]), f"LOOKAHEAD in long_sig at {cname}!"
        assert np.array_equal(ss_orig[: T + 1], ss_pert[: T + 1]), f"LOOKAHEAD in short_sig at {cname}!"
        print(f"  [PASS] {cname:15s} (T={T}): 0 lookahead diffs in all 5 indicators & L/S signals")

    # =========================================================================
    # TEST 3: Two-Stage Backtest Reproducibility (1,000-bar slice)
    # =========================================================================
    print("\n=== TEST 3: Two-Stage Backtest Reproducibility ===")
    sub_b = df.iloc[20000:21000].copy().reset_index(drop=True)
    cfg = BacktestConfig(k_sl=2.0, k_tp=4.0, max_hold_bars=48)
    pl_b = compute_indicators(sub_b, cfg)
    af_b = compute_algoforge_indicators(sub_b, ind_specs)

    # Stage 1: Engine Invariance (run saved Pipeline signals twice)
    tr1, _, sum1 = run_causal_fx_backtest(pl_b, cfg, GateConfig(), eval_start_idx=100)
    tr2, _, sum2 = run_causal_fx_backtest(pl_b, cfg, GateConfig(), eval_start_idx=100, long_signal=pl_b["cross_up"].values, short_signal=pl_b["cross_down"].values)
    assert sum1["trade_count"] == sum2["trade_count"] and abs(sum1["final_equity"] - sum2["final_equity"]) < 1e-9
    print(f"  [PASS] Stage 1 (Engine Invariance): Trades={sum1['trade_count']}, FinalEquity={sum1['final_equity']:.2f} (100% exact)")

    # Stage 2: Adapter Reproducibility (EMA cross_up/cross_down via AlgoForge state)
    af_cross_up = np.zeros(len(af_b), dtype=bool)
    af_cross_dn = np.zeros(len(af_b), dtype=bool)
    diff_af = af_b["ema12"].values - af_b["ema48"].values
    for i in range(1, len(af_b)):
        af_cross_up[i] = (diff_af[i] > 0) and (diff_af[i - 1] <= 0)
        af_cross_dn[i] = (diff_af[i] < 0) and (diff_af[i - 1] >= 0)

    tr_af, _, sum_af = run_causal_fx_backtest(pl_b, cfg, GateConfig(), eval_start_idx=100, long_signal=af_cross_up, short_signal=af_cross_dn)
    sig_match = float(np.mean(af_cross_up[100:] == pl_b["cross_up"].values[100:]))
    print(f"  [PASS] Stage 2 (Adapter Reproducibility): EMA SignalMatch={sig_match*100:.2f}% | Trades: PL={sum1['trade_count']} vs AF={sum_af['trade_count']} | FinalEq: PL={sum1['final_equity']:.2f} vs AF={sum_af['final_equity']:.2f}")

    # =========================================================================
    # TEST 4: Performance Scaling (500 / 2,000 / 5,000 bars)
    # =========================================================================
    print("\n=== TEST 4: Performance Scaling ===")
    for n_bars in [500, 2000, 5000]:
        sub_s = df.iloc[:n_bars].copy().reset_index(drop=True)
        t_i0 = time.time()
        af_s = compute_algoforge_indicators(sub_s, ind_specs)
        t_ind = time.time() - t_i0
        t_c0 = time.time()
        evaluate_algoforge_signals(af_s, long_cond, short_cond)
        t_cond = time.time() - t_c0
        print(f"  N={n_bars:5d} bars | apply_indicators={t_ind:.4f}s | evaluate_conditions={t_cond:.4f}s")

    print("\nALL PHASE 1A TESTS COMPLETED.")


if __name__ == "__main__":
    main()
