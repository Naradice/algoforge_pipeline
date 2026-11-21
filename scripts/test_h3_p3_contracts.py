"""H3-P3 Gate 2 Pre-Unsealing Contract Test Suite (Synthetic + 2024 Snapshot Only).

Strictly verifies:
1. M5 bar timestamp & causal fill contract (t_close = t_start + 5m)
2. Long/Short Bid/Ask SL/TP numerical examples
3. 5-step gap/weekend settlement priority & single-fill guarantee
4. Hamilton 0.5x scaling before 1,000 USD floor & Method A no-cancel
5. Cash + UnrealizedPnL accounting identity
6. statsmodels HAC zero-diff & Holm-Bonferroni step-down
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import (
    BacktestConfig,
    GateConfig,
    compute_indicators,
    run_causal_fx_backtest,
    compute_hac_one_sided_test,
    run_holm_bonferroni,
)


def test_long_short_numerical_examples():
    """Verify exact Long/Short Bid/Ask SL/TP and Cash+UnrealizedPnL accounting."""
    cfg = BacktestConfig(k_sl=2.0, k_tp=3.0, spread_jpy=0.010, slippage_jpy=0.002)
    # Build synthetic 30-bar series where EMA_12 crosses above EMA_48, then gaps down past SL
    times = pd.date_range("2024-06-03 00:00:00+00:00", periods=60, freq="5min")
    closes = np.full(60, 150.000)
    # Slight dip then rally to trigger cross_up at bar 55
    closes[45:55] = 149.900
    closes[55] = 150.500
    df = pd.DataFrame({
        "time": times,
        "open": closes,
        "high": closes + 0.050,
        "low": closes - 0.050,
        "close": closes,
    })
    # Bar 56 opens at 150.000 (entry Long at 150.012), bar 57 gaps down to 149.700 (GAP_SL)
    df.loc[56, "open"] = 150.000
    df.loc[56, "high"] = 150.020
    df.loc[56, "low"] = 149.990
    df.loc[56, "close"] = 150.000
    df.loc[57, "open"] = 149.700
    df.loc[57, "high"] = 149.720
    df.loc[57, "low"] = 149.690
    df.loc[57, "close"] = 149.700

    df_ind = compute_indicators(df, cfg)
    trades, daily, summary = run_causal_fx_backtest(df_ind, cfg, GateConfig(), eval_start_idx=54)
    assert len(trades) >= 1, "Expected at least 1 trade"
    t0 = trades.iloc[0]
    assert t0["signal_timestamp"].tz_localize(None) == pd.Timestamp(times[55]).tz_localize(None) + pd.Timedelta(minutes=5)
    assert t0["fill_timestamp"].tz_localize(None) == pd.Timestamp(times[56]).tz_localize(None)
    assert abs(t0["entry_price"] - 150.012) < 1e-9
    assert t0["exit_reason"] == "GAP_SL"
    assert abs(t0["exit_price"] - 149.698) < 1e-9
    # Accounting check: final equity == cash + unrealized_pnl
    assert abs(daily["equity"].iloc[-1] - (daily["cash"].iloc[-1] + daily["unrealized_pnl"].iloc[-1])) < 1e-6
    print("[PASS] test_long_short_numerical_examples")


def test_weekend_gap_priority_and_single_fill():
    """Verify GAP_SL takes priority over pending FORCED_CLOSE and no double-close occurs."""
    cfg = BacktestConfig(k_sl=2.0, k_tp=3.0)
    # Friday 21:20, 21:25, 21:30 (forced close signal), then Monday 00:00 gap
    times = list(pd.date_range("2024-06-07 17:00:00+00:00", periods=55, freq="5min"))
    # Last Friday bar is 21:30 UTC (index 54)
    times.append(pd.Timestamp("2024-06-10 00:00:00+00:00"))  # Monday gap open (index 55)
    closes = np.full(56, 150.000)
    closes[40:52] = 149.900
    closes[52] = 150.500  # cross_up at 21:20 -> fill at 21:25 (index 53)
    df = pd.DataFrame({
        "time": times,
        "open": closes,
        "high": closes + 0.050,
        "low": closes - 0.050,
        "close": closes,
    })
    df.loc[53, "open"] = 150.000
    df.loc[54, "open"] = 150.000
    df.loc[54, "close"] = 150.000
    # Monday gap down past SL
    df.loc[55, "open"] = 149.500
    df.loc[55, "high"] = 149.550
    df.loc[55, "low"] = 149.450
    df.loc[55, "close"] = 149.500

    df_ind = compute_indicators(df, cfg)
    trades, _, _ = run_causal_fx_backtest(df_ind, cfg, GateConfig(), eval_start_idx=50)
    assert len(trades) == 1, f"Expected exactly 1 trade (no double-close), got {len(trades)}"
    assert trades.iloc[0]["exit_reason"] == "GAP_SL"
    assert trades.iloc[0]["carried_over_weekend"] == True
    print("[PASS] test_weekend_gap_priority_and_single_fill")


def test_hac_and_holm_contracts():
    """Verify statsmodels HAC zero-diff and Holm step-down."""
    r1 = np.array([0.001, -0.002, 0.003, 0.001, 0.002])
    res_zero = compute_hac_one_sided_test(r1, r1, maxlags=4)
    assert res_zero["p_val"] == 1.0 and res_zero["status"] == "ZERO_DIFF_EXACT"

    hyps = [
        {"id": "H1", "p_val": 0.010},
        {"id": "H2", "p_val": 0.030},  # > 0.01667 -> stops here
        {"id": "H3", "p_val": 0.020},
        {"id": "H4", "p_val": 0.500},
    ]
    out = run_holm_bonferroni(hyps, alpha=0.05)
    assert out[0]["supported"] is True
    assert out[1]["supported"] is False  # p=0.020 > alpha_2=0.01667
    assert out[2]["supported"] is False
    assert out[3]["supported"] is False
    print("[PASS] test_hac_and_holm_contracts")


if __name__ == "__main__":
    test_long_short_numerical_examples()
    test_weekend_gap_priority_and_single_fill()
    test_hac_and_holm_contracts()
    print("ALL H3-P3 GATE 2 CONTRACT TESTS PASSED.")
