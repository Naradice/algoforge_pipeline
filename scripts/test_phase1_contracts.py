"""Phase 1 Contract Tests: H1/H4 Causality & Strategy Registry Deduplication."""

import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest.declarative_strategy import build_causal_htf_features
from research_pipeline.backtest.strategy_registry import (
    StrategyRegistry,
    compute_evaluation_hash,
    compute_strategy_hash,
)


def test_htf_causality():
    """Modifying prices inside an in-progress H1/H4 bar after T must not affect <= T."""
    times = pd.date_range("2024-06-03 00:00:00+00:00", periods=100, freq="5min")
    rng = np.random.default_rng(7)
    closes = 150.0 + np.cumsum(rng.normal(0, 0.02, size=100))
    df1 = pd.DataFrame({"time": times, "open": closes, "high": closes + 0.02, "low": closes - 0.02, "close": closes})
    df2 = df1.copy()
    # Bar 65 is 05:25 UTC (inside 05:00-06:00 H1 bar and 04:00-08:00 H4 bar).
    # Perturb bar 66..99
    df2.loc[66:, ["open", "high", "low", "close"]] += 10.0

    htf1 = build_causal_htf_features(df1)
    htf2 = build_causal_htf_features(df2)

    assert np.array_equal(htf1["h1_trend"].values[:66], htf2["h1_trend"].values[:66]), "H1 lookahead leak!"
    assert np.array_equal(htf1["h4_trend"].values[:66], htf2["h4_trend"].values[:66]), "H4 lookahead leak!"
    print("[PASS] test_htf_causality (zero lookahead from in-progress H1/H4 bars)")


def test_strategy_registry_dedup():
    """Verify commutative equivalence, evaluation_hash sensitivity, and persistent caching."""
    s1 = {
        "long": {"op": "AND", "clauses": [{"type": "ema_cross", "fast": 12, "slow": 48, "dir": 1}, {"type": "rsi_lt", "period": 14, "level": 35}]},
        "short": {"op": "AND", "clauses": [{"type": "ema_cross", "fast": 12, "slow": 48, "dir": -1}]},
        "htf_filter": {"h1_align": True, "h4_align": False},
        "hours_utc": [8, 9, 10],
        "risk": {"k_sl": 2.0, "k_tp": 3.0, "max_hold_bars": 48},
    }
    # Commutative swap of long clauses + unsorted hours_utc
    s2 = {
        "long": {"op": "AND", "clauses": [{"type": "rsi_lt", "period": 14, "level": 35}, {"type": "ema_cross", "fast": 12, "slow": 48, "dir": 1}]},
        "short": {"op": "AND", "clauses": [{"type": "ema_cross", "fast": 12, "slow": 48, "dir": -1}]},
        "htf_filter": {"h1_align": True, "h4_align": False},
        "hours_utc": [10, 8, 9],
        "risk": {"k_sl": 2.0, "k_tp": 3.0, "max_hold_bars": 48},
    }
    h1 = compute_strategy_hash(s1)
    h2 = compute_strategy_hash(s2)
    assert h1 == h2, "Commutative equivalent strategies produced different hashes!"

    eh1 = compute_evaluation_hash(h1, "data_sha", "2024_92d", "v1", {"spread": 0.01})
    eh2 = compute_evaluation_hash(h1, "data_sha", "2024_40d", "v1", {"spread": 0.01})
    assert eh1 != eh2, "Different eval periods must produce different evaluation_hash!"

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "reg.db")
        reg = StrategyRegistry(db_path)
        assert reg.lookup(eh1) is None
        reg.record(eh1, h1, s1, "REJECTED_PERF", "NEGATIVE_SHARPE", {"sharpe": -1.2}, "random", None, "exp1", 0.45)
        reg.close()

        # Re-open (simulating process restart)
        reg2 = StrategyRegistry(db_path)
        hit = reg2.lookup(eh1)
        assert hit is not None and hit["metrics"]["sharpe"] == -1.2
        assert reg2.is_strategy_rejected(h1) is True
        reg2.close()
    print("[PASS] test_strategy_registry_dedup (equivalence + persistence verified)")


if __name__ == "__main__":
    test_htf_causality()
    test_strategy_registry_dedup()
    print("ALL PHASE 1 CONTRACT TESTS PASSED (2/2).")
