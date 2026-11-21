"""Phase 2B-2a contract tests for unit:hawkes_hazard, time contracts, and Phase 2B-1 residual tests."""

import os
import sys
import tempfile
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from research_pipeline.integration.unit_registry import (
    UNIT_REGISTRY,
    UnitSpec,
    compute_files_sha256,
    compute_algoforge_unit_evaluation_hash,
)
from research_pipeline.integration.algoforge_adapter import (
    apply_units_with_time_check,
    eval_condition_with_mask,
)
from research_pipeline.backtest.strategy_registry import compute_algoforge_evaluation_hash


def _make_synthetic_ohlcv(n: int = 700, seed: int = 42, gap_idx: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.001, size=n)
    for idx in [100, 102, 250, 400, 520, 560]:
        if idx < n:
            rets[idx] = 0.01
    close = 150.0 * np.cumprod(1.0 + rets)
    ts = pd.date_range("2024-01-01 00:00:00", periods=n, freq="5min", tz="UTC")
    if gap_idx is not None and 0 < gap_idx < n:
        ts_list = list(ts)
        for i in range(gap_idx, n):
            ts_list[i] = ts_list[i] + pd.Timedelta(hours=2)
        ts = pd.DatetimeIndex(ts_list)
    return pd.DataFrame(
        {
            "time": ts,
            "timestamp": ts,
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": np.ones(n) * 100.0,
        }
    )


def test_hawkes_first_valid_bar_and_mask() -> None:
    df = _make_synthetic_ohlcv(n=600)
    spec = UNIT_REGISTRY["unit:hawkes_hazard"]
    assert spec.warmup_bars == 501
    assert spec.time_contract == "bar_close"
    out = spec.fn(df, "hawkes", {})
    assert len(out) == 600
    assert not out["hawkes_valid"].iloc[:500].any(), "Bars 0..499 must be invalid"
    assert out["hawkes"].iloc[:500].isna().all(), "Bars 0..499 must be NaN"
    assert out["hawkes_valid"].iloc[500:].all(), "Bars 500..599 must be valid"
    assert np.isfinite(out["hawkes"].iloc[500:]).all()
    print("[PASS] test_hawkes_first_valid_bar_and_mask")


def test_hawkes_future_perturbation_invariance() -> None:
    df_base = _make_synthetic_ohlcv(n=650, seed=7, gap_idx=575)
    spec = UNIT_REGISTRY["unit:hawkes_hazard"]
    out_base = spec.fn(df_base, "hawkes", {})

    for cutoff in [500, 550, 575]:
        df_pert = df_base.copy()
        df_pert.loc[cutoff + 1 :, "close"] = df_pert.loc[cutoff + 1 :, "close"] * 1.25
        out_pert = spec.fn(df_pert, "hawkes", {})
        np.testing.assert_allclose(
            out_base["hawkes"].iloc[500 : cutoff + 1].to_numpy(),
            out_pert["hawkes"].iloc[500 : cutoff + 1].to_numpy(),
            rtol=0.0,
            atol=0.0,
            err_msg=f"Future perturbation leaked at cutoff={cutoff}",
        )
    print("[PASS] test_hawkes_future_perturbation_invariance (cutoffs 500, 550, 575)")


def test_hawkes_gap_reset() -> None:
    df_gap = _make_synthetic_ohlcv(n=600, seed=11, gap_idx=560)
    df_gap.loc[560, "close"] = df_gap.loc[559, "close"] * 1.10
    spec = UNIT_REGISTRY["unit:hawkes_hazard"]
    out = spec.fn(df_gap, "hawkes", {})
    assert np.isfinite(out["hawkes"].iloc[560])
    assert out["hawkes_valid"].iloc[560]
    print("[PASS] test_hawkes_gap_reset")


def test_hawkes_cache_and_param_separation() -> None:
    ds_sha = "bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0"
    eval_period = "2024-01-01/2024-12-31"
    engine_ver = "v1"
    cost_cfg = {"spread": 0.002}

    strategy_base = {
        "indicators": [{"id": "h1", "type": "unit:hawkes_hazard", "params": {"mle_window": 500, "refit_interval": 50}}],
        "entry": {"and": [{"left": "h1", "op": ">", "right": 0.02}]},
        "exit": {"and": [{"left": "h1", "op": "<", "right": 0.01}]},
    }
    strategy_alt = {
        "indicators": [{"id": "h1", "type": "unit:hawkes_hazard", "params": {"mle_window": 400, "refit_interval": 25}}],
        "entry": {"and": [{"left": "h1", "op": ">", "right": 0.02}]},
        "exit": {"and": [{"left": "h1", "op": "<", "right": 0.01}]},
    }
    manifest_hawkes = [
        {
            "unit_id": "unit:hawkes_hazard",
            "version": UNIT_REGISTRY["unit:hawkes_hazard"].version,
            "impl_id": UNIT_REGISTRY["unit:hawkes_hazard"].impl_id,
            "dependencies": [],
        }
    ]
    h_unit = compute_algoforge_unit_evaluation_hash(
        strategy_base, manifest_hawkes, ds_sha, eval_period, engine_ver, cost_cfg
    )
    h_unit_alt = compute_algoforge_unit_evaluation_hash(
        strategy_alt, manifest_hawkes, ds_sha, eval_period, engine_ver, cost_cfg
    )
    h_v1 = compute_algoforge_evaluation_hash(
        strategy_base, ds_sha, eval_period, engine_ver, cost_cfg
    )
    assert h_unit != h_v1, "algoforge_unit_v1 hash must differ from algoforge_v1"
    assert h_unit != h_unit_alt, "Hawkes estimation param change must change evaluation hash"
    print("[PASS] test_hawkes_cache_and_param_separation")


def test_residual_impl_id_auto_change() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dep = os.path.join(tmp, "dep.py")
        with open(dep, "w", encoding="utf-8") as f:
            f.write("def x(): return 1\n")
        h1 = compute_files_sha256([dep])
        with open(dep, "w", encoding="utf-8") as f:
            f.write("def x(): return 2\n")
        h2 = compute_files_sha256([dep])
        assert h1 != h2, "impl_id must change when dependency file content changes"
    print("[PASS] test_residual_impl_id_auto_change")


def test_time_contract_negative_cases() -> None:
    df = _make_synthetic_ohlcv(n=30)

    # 1. available_at > decision_time -> RuntimeError
    def _future_fn(frame: pd.DataFrame, col_id: str, params: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col_id: np.ones(len(frame)),
                f"{col_id}_valid": np.ones(len(frame), dtype=bool),
                "available_at": frame["time"] + pd.Timedelta(minutes=10),
            },
            index=frame.index,
        )

    # 2. NaT in available_at -> ValueError
    def _nat_fn(frame: pd.DataFrame, col_id: str, params: dict) -> pd.DataFrame:
        av = (frame["time"] + pd.Timedelta(minutes=5)).copy()
        av.iloc[0] = pd.NaT
        return pd.DataFrame(
            {col_id: np.ones(len(frame)), f"{col_id}_valid": np.ones(len(frame), dtype=bool), "available_at": av},
            index=frame.index,
        )

    # 3. tz-naive available_at -> ValueError
    def _naive_fn(frame: pd.DataFrame, col_id: str, params: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col_id: np.ones(len(frame)),
                f"{col_id}_valid": np.ones(len(frame), dtype=bool),
                "available_at": frame["time"].dt.tz_localize(None),
            },
            index=frame.index,
        )

    # 4. explicit_available_at missing available_at -> ValueError
    def _missing_av_fn(frame: pd.DataFrame, col_id: str, params: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {col_id: np.ones(len(frame)), f"{col_id}_valid": np.ones(len(frame), dtype=bool)},
            index=frame.index,
        )

    UNIT_REGISTRY["unit:test_future"] = UnitSpec("unit:test_future", "1.0", "t", [], 0, {"value": "continuous"}, _future_fn, "explicit_available_at")
    UNIT_REGISTRY["unit:test_nat"] = UnitSpec("unit:test_nat", "1.0", "t", [], 0, {"value": "continuous"}, _nat_fn, "explicit_available_at")
    UNIT_REGISTRY["unit:test_naive"] = UnitSpec("unit:test_naive", "1.0", "t", [], 0, {"value": "continuous"}, _naive_fn, "explicit_available_at")
    UNIT_REGISTRY["unit:test_missing_av"] = UnitSpec("unit:test_missing_av", "1.0", "t", [], 0, {"value": "continuous"}, _missing_av_fn, "explicit_available_at")

    try:
        for uid, expected_exc in [
            ("unit:test_future", RuntimeError),
            ("unit:test_nat", ValueError),
            ("unit:test_naive", ValueError),
            ("unit:test_missing_av", ValueError),
        ]:
            raised = False
            try:
                apply_units_with_time_check(df, [{"id": "x", "type": uid, "params": {}}])
            except expected_exc:
                raised = True
            assert raised, f"Expected {expected_exc.__name__} for {uid}"

        # 5. tz-naive input df["time"] -> ValueError
        df_naive = df.copy()
        df_naive["time"] = df_naive["time"].dt.tz_localize(None)
        raised_naive_in = False
        try:
            apply_units_with_time_check(df_naive, [{"id": "e", "type": "unit:ema", "params": {"period": 5}}])
        except ValueError:
            raised_naive_in = True
        assert raised_naive_in, "Expected ValueError for tz-naive input time"

        # 6. conflicting unit keys -> ValueError
        raised_conflict = False
        try:
            apply_units_with_time_check(df, [{"id": "e", "type": "unit:ema", "unit_id": "unit:rsi", "params": {}}])
        except ValueError:
            raised_conflict = True
        assert raised_conflict, "Expected ValueError for conflicting unit keys"
    finally:
        for k in ["unit:test_future", "unit:test_nat", "unit:test_naive", "unit:test_missing_av"]:
            UNIT_REGISTRY.pop(k, None)

    print("[PASS] test_time_contract_negative_cases (6/6 fail-closed)")


def test_residual_ema_backtest_match() -> None:
    snap_path = os.path.join(_REPO_ROOT, "data", "usdjpy_m5_2024_oanda_snapshot.parquet")
    df = pd.read_parquet(snap_path).iloc[:500].copy()
    if "time" not in df.columns and "timestamp" in df.columns:
        df["time"] = df["timestamp"]
    indicators = [{"id": "ema_10", "type": "unit:ema", "params": {"period": 10}}]
    df_unit, _ = apply_units_with_time_check(df, indicators)
    cond = {"left": "close", "op": ">", "right": "ema_10"}
    sig_unit, valid_mask = eval_condition_with_mask(df_unit, cond)

    ema_direct = df["close"].ewm(span=10, adjust=False).mean()
    sig_direct = (df["close"] > ema_direct).to_numpy()
    assert np.array_equal(sig_unit[valid_mask], sig_direct[valid_mask])
    print("[PASS] test_residual_ema_backtest_match")


if __name__ == "__main__":
    test_hawkes_first_valid_bar_and_mask()
    test_hawkes_future_perturbation_invariance()
    test_hawkes_gap_reset()
    test_hawkes_cache_and_param_separation()
    test_residual_impl_id_auto_change()
    test_time_contract_negative_cases()
    test_residual_ema_backtest_match()
    print("ALL PHASE 2B-2A TESTS PASSED (7/7)")
