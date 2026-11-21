"""Phase 2B-1 Contract, Regression, Mask, Causality & Cache Isolation Tests."""

import hashlib
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.integration.algoforge_adapter import (
    apply_units_with_time_check,
    compute_algoforge_indicators,
    eval_condition_with_mask,
    evaluate_algoforge_conditions_vectorized,
)
from research_pipeline.integration.unit_registry import (
    UNIT_REGISTRY,
    canonicalize_unit_manifest,
    compute_algoforge_unit_evaluation_hash,
    compute_files_sha256,
)


def main():
    snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    with open(snap_path, "rb") as f:
        snap_sha = hashlib.sha256(f.read()).hexdigest()
    df = pd.read_parquet(snap_path).iloc[:1000].copy().reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # 1. File SHA-256 impl_id & Fail-Closed
    print("=== 1. impl_id File SHA-256 & Fail-Closed ===")
    assert len(UNIT_REGISTRY["unit:ema"].impl_id) == 16
    try:
        compute_files_sha256(["/nonexistent/path/xyz.py"])
        raise AssertionError("Should have failed closed!")
    except FileNotFoundError:
        print("  [PASS] Fail-closed on missing dependency file")

    # 2. Unit Manifest Canonicalization (Duplicate & Cycle Rejection)
    print("\n=== 2. Unit Manifest Canonicalization ===")
    try:
        canonicalize_unit_manifest([{"unit_id": "u1"}, {"unit_id": "u1"}])
        raise AssertionError("Should reject duplicate unit_id")
    except ValueError:
        print("  [PASS] Duplicate unit_id rejected")
    try:
        canonicalize_unit_manifest([{"unit_id": "u1", "dependencies": ["u2"]}, {"unit_id": "u2", "dependencies": ["u1"]}])
        raise AssertionError("Should reject cyclic dependency")
    except ValueError:
        print("  [PASS] Cyclic unit dependency rejected")

    # 3. Phase 1B Regression Match (unit:ema & unit:rsi vs AlgoForge apply_indicators)
    print("\n=== 3. Phase 1B Regression Match (Values, NaN, Signals) ===")
    unit_specs = [
        {"id": "ema12", "type": "unit:ema", "params": {"period": 12}},
        {"id": "rsi14", "type": "unit:rsi", "params": {"period": 14}},
    ]
    af_specs = [
        {"id": "ema12", "type": "ema", "params": {"period": 12}},
        {"id": "rsi14", "type": "rsi", "params": {"period": 14}},
    ]
    u_df, manifest = apply_units_with_time_check(df, unit_specs)
    af_df = compute_algoforge_indicators(df, af_specs)
    assert np.allclose(u_df["ema12"].values, af_df["ema12"].values, equal_nan=True)
    assert np.allclose(u_df["rsi14"].values, af_df["rsi14"].values, equal_nan=True)

    cond = {"logic": "and", "conditions": [{"left": "ema12", "op": ">", "right": 150.0}, {"left": "rsi14", "op": "<", "right": 35.0}]}
    v_mask, m_mask = eval_condition_with_mask(u_df, cond)
    v_p1b, _ = evaluate_algoforge_conditions_vectorized(af_df, cond, cond)
    # Match on valid interval
    assert np.array_equal(v_mask[20:], v_p1b[20:])
    print("  [PASS] unit:ema & unit:rsi values, NaN positions & valid-interval signals match 100%")

    # 4. Validity Mask Propagation (NaN, Inf, NOT, Non-Finite Constant, Nested Reject)
    print("\n=== 4. Validity Mask & NOT/Inf/NaN Safety ===")
    u_test = u_df.copy()
    u_test.loc[50, "rsi14"] = np.nan
    u_test.loc[51, "rsi14"] = np.inf
    u_test.loc[50:51, "rsi14_valid"] = False
    not_cond = {"not": {"left": "rsi14", "op": "<", "right": 35.0}}
    v_not, m_not = eval_condition_with_mask(u_test, not_cond)
    assert not v_not[50] and not m_not[50] and not v_not[51] and not m_not[51], "NOT fired on NaN/Inf!"
    assert not v_not[u_test["rsi14"].isna().values].any(), "NOT fired on NaN!"
    try:
        eval_condition_with_mask(u_test, {"left": "rsi14", "op": "<", "right": float("nan")})
        raise AssertionError("Should reject non-finite constant")
    except ValueError:
        pass
    try:
        eval_condition_with_mask(u_test, {"logic": "and", "conditions": [{"logic": "or", "conditions": [{"left": "rsi14", "op": "<", "right": 30}]}]})
        raise AssertionError("Should reject nested logic in production")
    except ValueError:
        pass
    print("  [PASS] Zero false triggers on NaN/Inf/warmup/NOT; non-finite constant & nested logic rejected")

    # 5. Causality (Multiple Cutoffs)
    print("\n=== 5. Causality Test (Cutoffs T=120, T=500) ===")
    for T in [120, 500]:
        df_p = df.copy()
        df_p.loc[T + 1 :, ["open", "high", "low", "close"]] += 5.0
        u_p, _ = apply_units_with_time_check(df_p, unit_specs)
        v_p, _ = eval_condition_with_mask(u_p, cond)
        assert np.array_equal(v_mask[: T + 1], v_p[: T + 1])
    print("  [PASS] 0 lookahead diffs across multiple cutoffs")

    # 6. Cache Separation (impl_id / version change)
    print("\n=== 6. Cache Separation (algoforge_unit_v1) ===")
    eh1 = compute_algoforge_unit_evaluation_hash("sh1", manifest, snap_sha, "2024_132d", "v1", {"spread": 0.01})
    m_mod = [dict(manifest[0], impl_id="deadbeef12345678"), manifest[1]]
    eh2 = compute_algoforge_unit_evaluation_hash("sh1", m_mod, snap_sha, "2024_132d", "v1", {"spread": 0.01})
    assert eh1 != eh2
    print("  [PASS] impl_id change separates evaluation_hash cleanly")

    print("\nALL PHASE 2B-1 TESTS PASSED.")


if __name__ == "__main__":
    main()
