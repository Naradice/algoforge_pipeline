"""Phase 2B-2b contract tests for unit:hamilton_pred_prob."""

import os
import sys
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from research_pipeline.integration.unit_registry import (
    UNIT_REGISTRY,
    compute_algoforge_unit_evaluation_hash,
)
from research_pipeline.integration.algoforge_adapter import apply_units_with_time_check
from research_pipeline.backtest.strategy_registry import compute_algoforge_evaluation_hash
from research_pipeline.discovery.econometric_modules import HamiltonLogRVRegimeModel


def _make_multiday_m5(n_days: int = 65, bars_per_day: int = 12, seed: int = 123) -> pd.DataFrame:
    """Create synthetic M5 bars (e.g., 12 bars/day at 00:00..00:55 and 23:55) spanning n_days."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 150.0
    start_day = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    for d in range(n_days):
        day_start = start_day + pd.Timedelta(days=d)
        vol = 0.0008 if d % 10 < 7 else 0.0025
        # Include 00:00..00:50 (11 bars) and 23:55 (1 bar) to test UTC boundary explicitly
        offsets = [5 * k for k in range(bars_per_day - 1)] + [23 * 60 + 55]
        for m in offsets:
            t = day_start + pd.Timedelta(minutes=m)
            price = price * (1.0 + rng.normal(0.0, vol))
            rows.append({"time": t, "open": price, "high": price * 1.0002, "low": price * 0.9998, "close": price, "volume": 100.0})
    return pd.DataFrame(rows)


def test_hamilton_first_valid_day_and_invalid_window() -> None:
    df = _make_multiday_m5(n_days=65, bars_per_day=12)
    df_out, _ = apply_units_with_time_check(
        df, [{"id": "ham", "type": "unit:hamilton_pred_prob", "params": {"window_days": 60, "min_bars_per_day": 12}}]
    )
    # Day 0 has only 11 valid returns (ret[0] is NaN) < min_bars_per_day=12, so days 0..60 are invalid; days 61..64 valid
    assert not df_out["ham_valid"].iloc[: 61 * 12].any(), "Warmup days 0..60 must be invalid"
    assert df_out["ham"].iloc[: 61 * 12].isna().all()
    assert df_out["ham_valid"].iloc[61 * 12 :].all(), "Days 61..64 must be valid"
    assert np.isfinite(df_out["ham"].iloc[61 * 12 :]).all()

    # Drop 2 bars from day 10 -> day 10 has < 12 bars -> window covering day 10..69 must be invalid
    df_drop = df.drop(df.index[10 * 12 + 1 : 10 * 12 + 3]).reset_index(drop=True)
    df_drop_out, _ = apply_units_with_time_check(
        df_drop, [{"id": "ham", "type": "unit:hamilton_pred_prob", "params": {"window_days": 60, "min_bars_per_day": 12}}]
    )
    assert not df_drop_out["ham_valid"].any(), "Invalid day in 60-day window must fail-closed (not skip)"
    print("[PASS] test_hamilton_first_valid_day_and_invalid_window")


def test_hamilton_utc_boundary_2355_to_0000() -> None:
    df = _make_multiday_m5(n_days=64, bars_per_day=12)
    df_out, _ = apply_units_with_time_check(
        df, [{"id": "ham", "type": "unit:hamilton_pred_prob", "params": {"window_days": 60, "min_bars_per_day": 12}}]
    )
    # At day 61: 23:55 UTC (day 60) must be invalid, 00:00 UTC (day 61) must be valid
    idx_d60_2355 = 61 * 12 - 1
    idx_d61_0000 = 61 * 12
    assert df_out["time"].iloc[idx_d60_2355].hour == 23 and df_out["time"].iloc[idx_d60_2355].minute == 55
    assert not df_out["ham_valid"].iloc[idx_d60_2355], "23:55 UTC on day 60 must not receive day 61 prediction"
    assert df_out["time"].iloc[idx_d61_0000].hour == 0 and df_out["time"].iloc[idx_d61_0000].minute == 0
    assert df_out["ham_valid"].iloc[idx_d61_0000], "00:00 UTC on day 61 must be valid"
    # Day 62 00:00 UTC must differ from Day 61 23:55 UTC (new daily update)
    idx_d61_2355 = 62 * 12 - 1
    idx_d62_0000 = 62 * 12
    assert df_out["ham"].iloc[idx_d61_0000] == df_out["ham"].iloc[idx_d61_2355]
    assert df_out["ham"].iloc[idx_d62_0000] != df_out["ham"].iloc[idx_d61_2355]
    print("[PASS] test_hamilton_utc_boundary_2355_to_0000")


def test_hamilton_asymmetric_transition_hand_calc() -> None:
    model = HamiltonLogRVRegimeModel()
    model.P = np.array([[0.90, 0.10], [0.25, 0.75]], dtype=np.float64)
    xi_prev = np.array([0.80, 0.20], dtype=np.float64)
    xi_next = model.P.T @ xi_prev
    np.testing.assert_allclose(xi_next, np.array([0.77, 0.23]), rtol=1e-12, atol=1e-12)
    print("[PASS] test_hamilton_asymmetric_transition_hand_calc")


def test_hamilton_em_fallbacks_and_state_order() -> None:
    model = HamiltonLogRVRegimeModel()
    y_const = np.full(60, -8.5, dtype=np.float64)
    # Force EM failure by monkeypatching fit_em
    orig_fit = model.fit_em
    model.fit_em = lambda y, init_params=None: False
    try:
        res0 = model.step_daily_update_with_fallback(y_const, day_idx=0, prev_params=None)
        assert res0["fallback_mode"] == "quantile_initial"
        assert res0["mu"][0] <= res0["mu"][1]

        prev = {"mu": np.array([-9.5, -7.2]), "sigma": np.array([0.3, 0.5]), "P": np.array([[0.9, 0.1], [0.2, 0.8]])}
        res1 = model.step_daily_update_with_fallback(y_const, day_idx=1, prev_params=prev)
        assert res1["fallback_mode"] == "lag1_carryover"
        np.testing.assert_allclose(res1["mu"], prev["mu"])
        assert res1["mu"][0] < res1["mu"][1]
    finally:
        model.fit_em = orig_fit
    print("[PASS] test_hamilton_em_fallbacks_and_state_order")


def test_hamilton_future_perturbation_invariance() -> None:
    df_base = _make_multiday_m5(n_days=65, bars_per_day=12, seed=99)
    df_pert = df_base.copy()
    cutoff_idx = 62 * 12  # Day 62 00:00 UTC
    df_pert.loc[cutoff_idx + 1 :, "close"] = df_pert.loc[cutoff_idx + 1 :, "close"] * 1.50

    out_base, _ = apply_units_with_time_check(
        df_base, [{"id": "ham", "type": "unit:hamilton_pred_prob", "params": {"window_days": 60, "min_bars_per_day": 12}}]
    )
    out_pert, _ = apply_units_with_time_check(
        df_pert, [{"id": "ham", "type": "unit:hamilton_pred_prob", "params": {"window_days": 60, "min_bars_per_day": 12}}]
    )
    # Day 61 (and Day 62 00:00 UTC) must be bit-identical
    np.testing.assert_allclose(
        out_base["ham"].iloc[61 * 12 : cutoff_idx + 1].to_numpy(),
        out_pert["ham"].iloc[61 * 12 : cutoff_idx + 1].to_numpy(),
        rtol=0.0,
        atol=0.0,
    )
    print("[PASS] test_hamilton_future_perturbation_invariance")


def test_hamilton_cache_separation() -> None:
    ds_sha = "bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0"
    eval_period = "2024-01-01/2024-12-31"
    engine_ver = "v1"
    cost_cfg = {"spread": 0.002}

    s1 = {"indicators": [{"id": "h1", "type": "unit:hamilton_pred_prob", "params": {"window_days": 60}}], "entry": {}, "exit": {}}
    s2 = {"indicators": [{"id": "h1", "type": "unit:hamilton_pred_prob", "params": {"window_days": 45}}], "entry": {}, "exit": {}}
    m = [{"unit_id": "unit:hamilton_pred_prob", "version": UNIT_REGISTRY["unit:hamilton_pred_prob"].version, "impl_id": UNIT_REGISTRY["unit:hamilton_pred_prob"].impl_id, "dependencies": []}]

    h1 = compute_algoforge_unit_evaluation_hash(s1, m, ds_sha, eval_period, engine_ver, cost_cfg)
    h2 = compute_algoforge_unit_evaluation_hash(s2, m, ds_sha, eval_period, engine_ver, cost_cfg)
    hv1 = compute_algoforge_evaluation_hash(s1, ds_sha, eval_period, engine_ver, cost_cfg)
    assert h1 != h2
    assert h1 != hv1
    print("[PASS] test_hamilton_cache_separation")


if __name__ == "__main__":
    test_hamilton_first_valid_day_and_invalid_window()
    test_hamilton_utc_boundary_2355_to_0000()
    test_hamilton_asymmetric_transition_hand_calc()
    test_hamilton_em_fallbacks_and_state_order()
    test_hamilton_future_perturbation_invariance()
    test_hamilton_cache_separation()
    print("ALL PHASE 2B-2B HAMILTON TESTS PASSED (6/6)")
