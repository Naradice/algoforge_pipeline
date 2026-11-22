"""Regression tests for Phase 1B-2 EvaluatorProtocol / LocalCausalEvaluator separation."""

import json
import math
import os
import subprocess
import sys
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from research_pipeline.backtest import BacktestConfig, GateConfig
from research_pipeline.backtest.evaluator_protocol import LocalCausalEvaluator, LocalDatasetRef
from research_pipeline.backtest.strategy_registry import compute_algoforge_evaluation_hash


def _assert_val_eq(a, b, ctx: str) -> None:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return
        assert a == b, f"Float mismatch at {ctx}: {a} != {b}"
    else:
        assert a == b, f"Mismatch at {ctx}: {a} != {b}"


def test_hash_compatibility() -> None:
    ev = LocalCausalEvaluator()
    assert ev.engine_id == "causal_fx_backtest_v1.0"
    h1 = compute_algoforge_evaluation_hash("abc123", "snap456", "2024_132d", ev.engine_id, {"spread": 0.002}, {"ema": "1.0"})
    h2 = compute_algoforge_evaluation_hash("abc123", "snap456", "2024_132d", "causal_fx_backtest_v1.0", {"spread": 0.002}, {"ema": "1.0"})
    assert h1 == h2
    print(f"[PASS] test_hash_compatibility (hash={h1})")


def test_phase1b2_baseline_regression() -> None:
    baseline_path = os.path.join(_REPO_ROOT, "tests", "fixtures", "phase1b2_baseline_8f52f02.json")
    with open(baseline_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    script_path = os.path.join(_REPO_ROOT, "scripts", "run_phase1b2_exploration.py")
    proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
    assert "COMPLETED_25_TRIALS" in proc.stdout

    out_path = os.path.join(_REPO_ROOT, "data", "phase1b2_exploration_results.json")
    with open(out_path, "r", encoding="utf-8") as f:
        curr = json.load(f)

    for k in ["total_trials", "adopted_count", "duplicates_skipped", "cache_hits", "executed_backtests", "rejected_validation", "capability_requests_count"]:
        assert curr[k] == base[k], f"Count mismatch {k}: {curr[k]} != {base[k]}"

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
    ev = LocalCausalEvaluator()
    ds = LocalDatasetRef(af_ind=pd.DataFrame(), pl_df=None, eval_start_idx=0, dataset_sha256="x")
    raised = False
    try:
        ev.evaluate({"long": {"entry": {"bad": 1}}, "short": {"entry": {}}}, ds, {"backtest_config": BacktestConfig(), "gate_config": GateConfig()})
    except Exception:
        raised = True
    assert raised, "Expected exception to propagate without being swallowed"
    print("[PASS] test_exception_propagation_preserved")


if __name__ == "__main__":
    test_hash_compatibility()
    test_phase1b2_baseline_regression()
    test_exception_propagation_preserved()
    print("ALL PHASE 1B-2 EVALUATOR REGRESSION TESTS PASSED (3/3)")
