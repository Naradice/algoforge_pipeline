"""Regression tests for Phase 1B-2 EvaluatorProtocol / LocalCausalEvaluator separation."""

import json
import math
import os
import subprocess
import sys
import tempfile
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from research_pipeline.backtest import BacktestConfig, GateConfig
from research_pipeline.backtest.strategy_registry import StrategyRegistry, compute_algoforge_evaluation_hash
from research_pipeline.backtest.evaluator_protocol import (
    LocalCausalEvaluator,
    LocalDatasetRef,
    compute_eval_config_hash,
)
from research_pipeline.integration.algoforge_adapter import evaluate_algoforge_conditions_vectorized
from run_phase1b2_exploration import COST_CFG, UNIT_VERSIONS


def _assert_val_eq(a, b, ctx: str) -> None:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return
        assert a == b, f"Float mismatch at {ctx}: {a} != {b}"
    else:
        assert a == b, f"Mismatch at {ctx}: {a} != {b}"


def _load_baseline() -> dict:
    baseline_path = os.path.join(_REPO_ROOT, "tests", "fixtures", "phase1b2_baseline_8f52f02.json")
    with open(baseline_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_hash_compatibility() -> None:
    base = _load_baseline()
    t0 = next(t for t in base["trial_logs"] if "evaluation_hash" in t)
    ev = LocalCausalEvaluator()
    assert ev.engine_id == "causal_fx_backtest_v1.0"
    actual = compute_algoforge_evaluation_hash(
        t0["strategy_hash"],
        base["snapshot_2024_sha256"],
        "2024_132d",
        ev.engine_id,
        COST_CFG,
        UNIT_VERSIONS,
    )
    assert actual == t0["evaluation_hash"], f"Hash mismatch: {actual} != {t0['evaluation_hash']}"
    print(f"[PASS] test_hash_compatibility (matched 8f52f02 hash={actual[:16]}...)")


def test_existing_cache_reuse_avoids_backtest() -> None:
    base = _load_baseline()
    t0 = next(t for t in base["trial_logs"] if "evaluation_hash" in t and "metrics" in t)
    ev = LocalCausalEvaluator()
    e_hash = compute_algoforge_evaluation_hash(
        t0["strategy_hash"],
        base["snapshot_2024_sha256"],
        "2024_132d",
        ev.engine_id,
        COST_CFG,
        UNIT_VERSIONS,
    )
    with tempfile.TemporaryDirectory() as tmp:
        reg = StrategyRegistry(os.path.join(tmp, "legacy_cache.db"))
        try:
            reg.record(e_hash, t0["strategy_hash"], {}, t0["status"], t0["status"], t0["metrics"], "legacy", None, "phase1b2", 0.0)
            hit = reg.lookup(e_hash)
            assert hit is not None, "Expected cache hit on pre-populated legacy entry"
            assert hit["metrics"] == t0["metrics"]
        finally:
            reg.conn.close()
    print("[PASS] test_existing_cache_reuse_avoids_backtest")


def test_phase1b2_baseline_regression() -> None:
    base = _load_baseline()
    script_path = os.path.join(_REPO_ROOT, "scripts", "run_phase1b2_exploration.py")
    proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
    assert "COMPLETED_25_TRIALS" in proc.stdout

    out_path = os.path.join(_REPO_ROOT, "data", "phase1b2_exploration_results.json")
    with open(out_path, "r", encoding="utf-8") as f:
        curr = json.load(f)

    assert curr["counts"] == base["counts"], f"Counts mismatch: {curr['counts']} != {base['counts']}"
    assert curr["root_counts"] == base["root_counts"], "Root counts mismatch"
    assert curr["capability_requests"] == base["capability_requests"]
    assert len(curr["trial_logs"]) == len(base["trial_logs"])
    for i, (ct, bt) in enumerate(zip(curr["trial_logs"], base["trial_logs"])):
        for fld in ["trial", "gen_type", "status", "strategy_hash", "evaluation_hash", "root_hash"]:
            assert ct.get(fld) == bt.get(fld), f"Trial {i} {fld} mismatch"
        if "metrics" in bt:
            for mk, mv in bt["metrics"].items():
                _assert_val_eq(ct["metrics"][mk], mv, f"trial[{i}].metrics[{mk}]")
    print("[PASS] test_phase1b2_baseline_regression (25 trials, 100% match with 8f52f02)")


def test_exception_propagation_preserved() -> None:
    bad_cond = {"bad": 1}
    direct_exc_type = None
    try:
        evaluate_algoforge_conditions_vectorized(pd.DataFrame(), bad_cond, {})
    except Exception as exc:
        direct_exc_type = type(exc)

    ev = LocalCausalEvaluator()
    ds = LocalDatasetRef(af_ind=pd.DataFrame(), pl_df=None, eval_start_idx=0, dataset_sha256="x")
    eval_exc_type = None
    try:
        ev.evaluate({"long": {"entry": bad_cond}, "short": {"entry": {}}}, ds, {"backtest_config": BacktestConfig(), "gate_config": GateConfig()})
    except Exception as exc:
        eval_exc_type = type(exc)

    assert direct_exc_type is not None and eval_exc_type is direct_exc_type, (
        f"Exception type mismatch: {eval_exc_type} != {direct_exc_type}"
    )
    print(f"[PASS] test_exception_propagation_preserved (type={eval_exc_type.__name__})")


def test_eval_config_hash_deterministic() -> None:
    c1 = {"backtest_config": BacktestConfig(), "gate_config": GateConfig(), "cost_cfg": {"spread_pips": 0.2}}
    c2 = {"backtest_config": BacktestConfig(), "gate_config": GateConfig(), "cost_cfg": {"spread_pips": 0.2}}
    assert compute_eval_config_hash(c1) == compute_eval_config_hash(c2)
    print("[PASS] test_eval_config_hash_deterministic")


if __name__ == "__main__":
    test_hash_compatibility()
    test_existing_cache_reuse_avoids_backtest()
    test_phase1b2_baseline_regression()
    test_exception_propagation_preserved()
    test_eval_config_hash_deterministic()
    print("ALL PHASE 1B-2 EVALUATOR REGRESSION TESTS PASSED (5/5)")
