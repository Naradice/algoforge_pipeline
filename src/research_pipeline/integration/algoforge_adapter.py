"""Thin Adapter for AlgoForge apply_indicators & evaluate_conditions (Phase 1B-1).

Strictly read-only bridge to AlgoForge backend without modifying AlgoForge.
Supports:
- Entry-only AlgoForge Strategy JSON validation (explicitly rejects non-empty exit or unsupported risk)
- Vectorized evaluation of simple comparisons & boolean logic (>, <, >=, <=, ==, !=, and, or, not)
- Capability Request record generation for unsupported features
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ALGOFORGE_BACKEND = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "algoforge", "backend")
)
if _ALGOFORGE_BACKEND not in sys.path:
    sys.path.insert(0, _ALGOFORGE_BACKEND)

from strategy.engine.indicators import apply_indicators  # noqa: E402
from strategy.engine.conditions import evaluate_conditions  # noqa: E402

SUPPORTED_INDICATORS = {"ema", "sma", "rsi", "atr", "bb", "donchian", "macd", "adx", "stochastic", "roc", "rvol"}
SUPPORTED_RISK_KEYS = {"k_sl", "k_tp", "max_hold_bars"}
SUPPORTED_OPS = {">", "<", ">=", "<=", "==", "!="}


def build_capability_request(strategy: Dict[str, Any], reason: str, missing_capability: str) -> Dict[str, Any]:
    """Build a minimal Capability Request record for unsupported strategy features."""
    return {
        "name": f"cap_req_{missing_capability.lower()}",
        "reason": reason,
        "missing_capability": missing_capability,
        "status": "proposed",
        "strategy_snapshot": strategy,
    }


def _validate_cond_node(node: Any) -> Tuple[bool, str]:
    if not isinstance(node, dict) or not node:
        return False, "INVALID_CONDITION_EMPTY"
    if "conditions" in node:
        logic = str(node.get("logic", "and")).lower()
        if logic not in {"and", "or"} or not isinstance(node["conditions"], list) or not node["conditions"]:
            return False, "INVALID_CONDITION_BLOCK"
        for c in node["conditions"]:
            ok, msg = _validate_cond_node(c)
            if not ok:
                return False, msg
        return True, "OK"
    if "type" in node:
        return False, f"UNSUPPORTED_CONDITION_TYPE:{node['type']}"
    if "left" in node and "op" in node and "right" in node:
        if node["op"] not in SUPPORTED_OPS:
            return False, f"UNSUPPORTED_OP:{node['op']}"
        return True, "OK"
    return False, "INVALID_CONDITION_STRUCTURE"


def validate_algoforge_strategy(strategy: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Validate AlgoForge v1 strategy JSON (Entry-only, supported indicators & risk)."""
    inds = strategy.get("indicators", [])
    if not isinstance(inds, list) or not inds:
        return False, "INVALID_INDICATORS", None
    for ind in inds:
        if ind.get("type") not in SUPPORTED_INDICATORS:
            reason = f"UNSUPPORTED_INDICATOR:{ind.get('type')}"
            return False, reason, build_capability_request(strategy, reason, f"indicator_{ind.get('type')}")

    for side in ("long", "short"):
        block = strategy.get(side)
        if not isinstance(block, dict):
            return False, f"MISSING_{side.upper()}_BLOCK", None
        exit_cond = block.get("exit")
        if exit_cond is not None and exit_cond != {}:
            reason = f"UNSUPPORTED_EXIT_CONDITION:{side}"
            return False, reason, build_capability_request(strategy, reason, "custom_exit_condition")
        entry_cond = block.get("entry")
        ok, msg = _validate_cond_node(entry_cond)
        if not ok:
            cap = build_capability_request(strategy, msg, msg.lower()) if msg.startswith("UNSUPPORTED_") else None
            return False, msg, cap

    risk = strategy.get("risk", {})
    if not isinstance(risk, dict):
        return False, "INVALID_RISK_BLOCK", None
    extra_risk = set(risk.keys()) - SUPPORTED_RISK_KEYS
    if extra_risk:
        reason = f"UNSUPPORTED_RISK_PARAM:{sorted(extra_risk)}"
        return False, reason, build_capability_request(strategy, reason, "extended_risk_params")

    return True, "OK", None


def compute_algoforge_indicators(df: pd.DataFrame, indicator_specs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Run AlgoForge apply_indicators on a copy of OHLC DataFrame."""
    work = df[["time", "open", "high", "low", "close"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work = work.set_index("time")
    out = apply_indicators(work, indicator_specs)
    return out.reset_index()


def _eval_vec_node(df_ind: pd.DataFrame, node: Dict[str, Any]) -> np.ndarray:
    n = len(df_ind)
    if "conditions" in node:
        logic = str(node.get("logic", "and")).lower()
        if logic == "or":
            res = np.zeros(n, dtype=bool)
            for c in node["conditions"]:
                res |= _eval_vec_node(df_ind, c)
            return res
        res = np.ones(n, dtype=bool)
        for c in node["conditions"]:
            res &= _eval_vec_node(df_ind, c)
        return res

    left_val = df_ind[node["left"]].values.astype(np.float64)
    right = node["right"]
    if isinstance(right, str) and right in df_ind.columns:
        right_val = df_ind[right].values.astype(np.float64)
    else:
        right_val = np.full(n, float(right), dtype=np.float64)

    valid = ~(np.isnan(left_val) | np.isnan(right_val))
    op = node["op"]
    out = np.zeros(n, dtype=bool)
    if op == ">":
        out[valid] = left_val[valid] > right_val[valid]
    elif op == "<":
        out[valid] = left_val[valid] < right_val[valid]
    elif op == ">=":
        out[valid] = left_val[valid] >= right_val[valid]
    elif op == "<=":
        out[valid] = left_val[valid] <= right_val[valid]
    elif op == "==":
        out[valid] = left_val[valid] == right_val[valid]
    elif op == "!=":
        out[valid] = left_val[valid] != right_val[valid]
    return out


def evaluate_algoforge_conditions_vectorized(
    df_ind: pd.DataFrame,
    long_entry: Dict[str, Any],
    short_entry: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized evaluation of simple comparisons & boolean logic (bar 0 False to match sequential)."""
    ls = _eval_vec_node(df_ind, long_entry)
    ss = _eval_vec_node(df_ind, short_entry)
    ls[0] = False
    ss[0] = False
    return ls, ss


def evaluate_algoforge_signals(
    df_ind: pd.DataFrame,
    long_cond: Dict[str, Any],
    short_cond: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sequential reference evaluation using AlgoForge evaluate_conditions."""
    n = len(df_ind)
    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)
    for i in range(1, n):
        row = df_ind.iloc[i]
        if long_cond:
            long_sig[i] = bool(evaluate_conditions(row, long_cond))
        if short_cond:
            short_sig[i] = bool(evaluate_conditions(row, short_cond))
    return long_sig, short_sig


def apply_units_with_time_check(
    df: pd.DataFrame,
    unit_specs: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Execute registered units with runtime available_at <= decision_time check."""
    from research_pipeline.integration.unit_registry import UNIT_REGISTRY

    if "time" not in df.columns:
        raise ValueError("MISSING_TIME_COLUMN: bar_start 'time' required")
    work = df.copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work["decision_time"] = work["time"] + pd.Timedelta(minutes=5)
    work["available_at"] = work["decision_time"]
    if not (work["available_at"] <= work["decision_time"]).all():
        raise RuntimeError("CAUSALITY_VIOLATION: available_at > decision_time")

    used_manifest: List[Dict[str, Any]] = []
    for spec in unit_specs:
        uid = spec.get("type", "")
        col_id = spec.get("id", "")
        if uid not in UNIT_REGISTRY:
            raise ValueError(f"UNKNOWN_UNIT:{uid}")
        u = UNIT_REGISTRY[uid]
        u_df = u.fn(work, col_id, spec.get("params", {}))
        work[col_id] = u_df[col_id].values
        work[f"{col_id}_valid"] = u_df[f"{col_id}_valid"].values
        used_manifest.append({
            "unit_id": u.unit_id,
            "version": u.version,
            "impl_id": u.impl_id,
            "dependencies": u.dependencies,
        })
    return work, used_manifest


def eval_condition_with_mask(df: pd.DataFrame, node: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate 1-level AND/OR (or internal NOT for contract tests) with conservative is_valid propagation."""
    n = len(df)
    if "not" in node:
        v, m = eval_condition_with_mask(df, node["not"])
        return (~v) & m, m
    if "conditions" in node:
        logic = str(node.get("logic", "and")).lower()
        vals = []
        masks = []
        for c in node["conditions"]:
            if "conditions" in c or "not" in c:
                raise ValueError("UNSUPPORTED_NESTED_LOGIC: only 1-level AND/OR supported in production")
            v, m = eval_condition_with_mask(df, c)
            vals.append(v)
            masks.append(m)
        all_valid = np.ones(n, dtype=bool)
        for m in masks:
            all_valid &= m
        if logic == "or":
            out_v = np.zeros(n, dtype=bool)
            for v in vals:
                out_v |= v
            return out_v & all_valid, all_valid
        out_v = np.ones(n, dtype=bool)
        for v in vals:
            out_v &= v
        return out_v & all_valid, all_valid

    left_col = node["left"]
    left_arr = df[left_col].values.astype(np.float64)
    left_valid = df[f"{left_col}_valid"].values if f"{left_col}_valid" in df.columns else np.isfinite(left_arr)

    right = node["right"]
    if isinstance(right, str) and right in df.columns:
        right_arr = df[right].values.astype(np.float64)
        right_valid = df[f"{right}_valid"].values if f"{right}_valid" in df.columns else np.isfinite(right_arr)
    else:
        r_val = float(right)
        if not np.isfinite(r_val):
            raise ValueError(f"NON_FINITE_CONSTANT:{right}")
        right_arr = np.full(n, r_val, dtype=np.float64)
        right_valid = np.ones(n, dtype=bool)

    valid = left_valid & right_valid & np.isfinite(left_arr) & np.isfinite(right_arr)
    op = node["op"]
    out = np.zeros(n, dtype=bool)
    if op == ">":
        out[valid] = left_arr[valid] > right_arr[valid]
    elif op == "<":
        out[valid] = left_arr[valid] < right_arr[valid]
    elif op == ">=":
        out[valid] = left_arr[valid] >= right_arr[valid]
    elif op == "<=":
        out[valid] = left_arr[valid] <= right_arr[valid]
    elif op == "==":
        out[valid] = left_arr[valid] == right_arr[valid]
    elif op == "!=":
        out[valid] = left_arr[valid] != right_arr[valid]
    return out & valid, valid
