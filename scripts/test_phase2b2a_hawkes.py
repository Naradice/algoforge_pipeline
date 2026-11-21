"""Phase 2B-2a contract tests for unit:hawkes_hazard and Phase 2B-1 residual tests."""

import os
import shutil
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
    # Inject a few spikes to trigger Hawkes events
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
    out = spec.fn(df, "hawkes", {})
    assert len(out) == 600
    assert not out["is_valid"].iloc[:500].any(), "Bars 0..499 must be invalid"
    assert out["value"].iloc[:500].isna().all(), "Bars 0..499 must be NaN"
    assert out["is_valid"].iloc[500:].all(), "Bars 500..599 must be valid"
    assert np.isfinite(out["value"].iloc[500:]).all()
    print("[PASS] test_hawkes_first_valid_bar_and_mask")


def test_hawkes_future_perturbation_invariance() -> None:
    df_base = _make_synthetic_ohlcv(n=650, seed=7, gap_idx=575)
    spec = UNIT_REGISTRY["unit:hawkes_hazard"]
    out_base = spec.fn(df_base, "hawkes", {})

    # Perturb future bars after initial MLE (500), periodic update (550), and gap (575)
    for cutoff in [500, 550, 575]:
        df_pert = df_base.copy()
        df_pert.loc[cutoff + 1 :, "close"] = df_pert.loc[cutoff + 1 :, "close"] * 1.25
        out_pert = spec.fn(df_pert, "hawkes", {})
        np.testing.assert_allclose(
            out_base["value"].iloc[500 : cutoff + 1].to_numpy(),
            out_pert["value"].iloc[500 : cutoff + 1].to_numpy(),
            rtol=0.0,
            atol=0.0,
            err_msg=f"Future perturbation leaked at cutoff={cutoff}",
        )
    print("[PASS] test_hawkes_future_perturbation_invariance (cutoffs 500, 550, 575)")


def test_hawkes_gap_reset() -> None:
    df_gap = _make_synthetic_ohlcv(n=600, seed=11, gap_idx=560)
    # Force a huge return on the gap bar to verify 0.0-fill + mu reset
    df_gap.loc[560, "close"] = df_gap.loc[559, "close"] * 1.10
    spec = UNIT_REGISTRY["unit:hawkes_hazard"]
    out = spec.fn(df_gap, "hawkes", {})
    assert np.isfinite(out["value"].iloc[560])
    assert out["is_valid"].iloc[560]
    print("[PASS] test_hawkes_gap_reset")


def test_hawkes_cache_separation() -> None:
    strategy_base = {
        "indicators": [{"id": "h1", "unit": "unit:hawkes_hazard", "params": {}}],
        "entry": {"and": [{"left": "h1", "op": ">", "right": 0.02}]},
        "exit": {"and": [{"left": "h1", "op": "<", "right": 0.01}]},
    }
    ds_sha = "bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0"
    eval_period = "2024-01-01/2024-12-31"
    engine_ver = "v1"
    cost_cfg = {"spread": 0.002}
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
    h_v1 = compute_algoforge_evaluation_hash(
        strategy_base, ds_sha, eval_period, engine_ver, cost_cfg
    )
    assert h_unit != h_v1, "algoforge_unit_v1 hash must differ from algoforge_v1"

    strategy_ema = {
        "indicators": [{"id": "h1", "unit": "unit:ema", "params": {"period": 10}}],
        "entry": {"and": [{"left": "h1", "op": ">", "right": 0.02}]},
        "exit": {"and": [{"left": "h1", "op": "<", "right": 0.01}]},
    }
    manifest_ema = [
        {
            "unit_id": "unit:ema",
            "version": UNIT_REGISTRY["unit:ema"].version,
            "impl_id": UNIT_REGISTRY["unit:ema"].impl_id,
            "dependencies": [],
        }
    ]
    h_ema = compute_algoforge_unit_evaluation_hash(
        strategy_ema, manifest_ema, ds_sha, eval_period, engine_ver, cost_cfg
    )
    assert h_unit != h_ema
    print("[PASS] test_hawkes_cache_separation")


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


def test_residual_time_contract_negative() -> None:
    from research_pipeline.integration.unit_registry import UnitSpec

    df = _make_synthetic_ohlcv(n=30)
    df["decision_time"] = pd.to_datetime(df["timestamp"])

    def _bad_fn(frame: pd.DataFrame, col_id: str, params: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "value": np.ones(len(frame)),
                "is_valid": np.ones(len(frame), dtype=bool),
                "available_at": pd.to_datetime(frame["timestamp"]) + pd.Timedelta(minutes=5),
            },
            index=frame.index,
        )

    UNIT_REGISTRY["unit:bad_time"] = UnitSpec(
        unit_id="unit:bad_time",
        version="1.0.0",
        impl_id="test",
        dependencies=[],
        warmup_bars=0,
        output_schema={"value": "continuous"},
        fn=_bad_fn,
    )
    indicators = [{"id": "bad", "unit": "unit:bad_time", "params": {}}]
    raised = False
    try:
        apply_units_with_time_check(df, indicators)
    except ValueError as exc:
        raised = "available_at" in str(exc) or "decision_time" in str(exc) or "causal" in str(exc).lower()
    finally:
        UNIT_REGISTRY.pop("unit:bad_time", None)
    assert raised, "Expected ValueError on available_at > decision_time"
    print("[PASS] test_residual_time_contract_negative")


def test_residual_ema_backtest_match() -> None:
    snap_path = os.path.join(_REPO_ROOT, "data", "usdjpy_m5_2024_oanda_snapshot.parquet")
    df = pd.read_parquet(snap_path).iloc[:500].copy()
    indicators = [{"id": "ema_10", "unit": "unit:ema", "params": {"period": 10}}]
    df_unit, masks = apply_units_with_time_check(df, indicators)
    cond = {"left": "close", "op": ">", "right": "ema_10"}
    sig_unit, valid_mask = eval_condition_with_mask(df_unit, masks, cond)

    # Compare against direct pandas/algoforge EMA on valid bars
    ema_direct = df["close"].ewm(span=10, adjust=False).mean()
    sig_direct = df["close"] > ema_direct
    valid_idx = valid_mask.to_numpy()
    assert np.array_equal(sig_unit.to_numpy()[valid_idx], sig_direct.to_numpy()[valid_idx])
    print("[PASS] test_residual_ema_backtest_match")


if __name__ == "__main__":
    test_hawkes_first_valid_bar_and_mask()
    test_hawkes_future_perturbation_invariance()
    test_hawkes_gap_reset()
    test_hawkes_cache_separation()
    test_residual_impl_id_auto_change()
    test_residual_time_contract_negative()
    test_residual_ema_backtest_match()
    print("ALL PHASE 2B-2A TESTS PASSED (7/7)")
