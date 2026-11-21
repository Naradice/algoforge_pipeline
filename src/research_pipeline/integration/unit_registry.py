"""Minimal Unit Registry for AlgoForge Phase 2B-1 (unit:ema, unit:rsi).

Implements:
- File-content SHA-256 based impl_id (fail-closed if unreadable)
- Independent canonicalize_unit_manifest with duplicate ID & cycle detection
- Isolated algoforge_unit_v1 evaluation hash
- Value + np.isfinite validity mask output
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

_ALGOFORGE_INDICATORS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "algoforge", "backend", "strategy", "engine", "indicators.py")
)
_ECONOMETRIC_MODULES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "discovery", "econometric_modules.py")
)


def compute_files_sha256(paths: List[str]) -> str:
    """Compute deterministic SHA-256 across sorted file paths (fail-closed)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"FAIL_CLOSED: Required dependency file missing: {p}")
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode("utf-8"))
            h.update(f.read())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    version: str
    impl_id: str
    dependencies: List[str]
    warmup_bars: int
    output_schema: Dict[str, str]
    fn: Callable[[pd.DataFrame, str, Dict[str, Any]], pd.DataFrame]
    time_contract: str = "bar_close"


def canonicalize_unit_manifest(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize unit manifest, rejecting duplicate IDs and cyclic dependencies."""
    seen_ids = set()
    adj: Dict[str, List[str]] = {}
    norm_list: List[Dict[str, Any]] = []

    for u in units:
        uid = u.get("unit_id")
        if not uid or uid in seen_ids:
            raise ValueError(f"INVALID_UNIT_MANIFEST: duplicate or missing unit_id {uid}")
        seen_ids.add(uid)
        deps = sorted(list(u.get("dependencies", [])))
        adj[uid] = deps
        norm_list.append({
            "unit_id": uid,
            "version": str(u.get("version", "")),
            "impl_id": str(u.get("impl_id", "")),
            "dependencies": deps,
        })

    # Cycle detection via DFS
    state: Dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited

    def dfs(node: str) -> None:
        st = state.get(node, 0)
        if st == 1:
            raise ValueError(f"CYCLIC_UNIT_DEPENDENCY detected at {node}")
        if st == 2:
            return
        state[node] = 1
        for nxt in adj.get(node, []):
            dfs(nxt)
        state[node] = 2

    for uid in seen_ids:
        dfs(uid)

    norm_list.sort(key=lambda x: x["unit_id"])
    return norm_list


def compute_algoforge_unit_evaluation_hash(
    strategy_hash: str,
    unit_manifest: List[Dict[str, Any]],
    dataset_sha256: str,
    eval_period: str,
    engine_version: str,
    cost_config: Dict[str, Any],
) -> str:
    """Isolated evaluation hash for algoforge_unit_v1."""
    canon_manifest = canonicalize_unit_manifest(unit_manifest)
    payload = json.dumps(
        {
            "schema_version": "algoforge_unit_v1",
            "strategy_hash": strategy_hash,
            "unit_manifest": canon_manifest,
            "dataset_sha256": dataset_sha256,
            "eval_period": eval_period,
            "engine_version": engine_version,
            "cost_config": cost_config,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_algoforge_unit(df: pd.DataFrame, col_id: str, ind_type: str, params: Dict[str, Any]) -> pd.DataFrame:
    from research_pipeline.integration.algoforge_adapter import compute_algoforge_indicators

    if col_id in df.columns or f"{col_id}_valid" in df.columns:
        raise ValueError(f"COLUMN_COLLISION: {col_id} or {col_id}_valid already exists")

    af_df = compute_algoforge_indicators(df, [{"id": col_id, "type": ind_type, "params": params}])
    vals = af_df[col_id].values.astype(np.float64)
    out = pd.DataFrame({
        col_id: vals,
        f"{col_id}_valid": np.isfinite(vals),
    })
    return out


def _unit_ema(df: pd.DataFrame, col_id: str, params: Dict[str, Any]) -> pd.DataFrame:
    return _run_algoforge_unit(df, col_id, "ema", params)


def _unit_rsi(df: pd.DataFrame, col_id: str, params: Dict[str, Any]) -> pd.DataFrame:
    return _run_algoforge_unit(df, col_id, "rsi", params)


def _unit_hawkes_hazard(df: pd.DataFrame, col_id: str, params: Dict[str, Any]) -> pd.DataFrame:
    from research_pipeline.discovery.econometric_modules import HawkesDiscreteHazardModel

    lookback = int(params.get("lookback_window", 20))
    threshold_sigma = float(params.get("threshold_sigma", 2.0))
    mle_window = int(params.get("mle_window", 500))
    refit_interval = int(params.get("refit_interval", 50))
    gap_seconds = float(params.get("gap_threshold_seconds", 300.0))

    n = len(df)
    values = np.full(n, np.nan, dtype=np.float64)
    is_valid = np.zeros(n, dtype=bool)

    if n <= mle_window:
        return pd.DataFrame({col_id: values, f"{col_id}_valid": is_valid}, index=df.index)

    close = df["close"].to_numpy(dtype=np.float64)
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] != 0.0, close[:-1], 1.0)

    events, _ = HawkesDiscreteHazardModel.extract_events_from_returns(
        returns, lookback_window=lookback, threshold_mult=threshold_sigma
    )
    events = events.copy()

    gap_mask = np.zeros(n, dtype=bool)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        diffs = ts.diff().dt.total_seconds().fillna(0.0).to_numpy()
        gap_mask = diffs > gap_seconds
    elif isinstance(df.index, pd.DatetimeIndex):
        diffs = df.index.to_series().diff().dt.total_seconds().fillna(0.0).to_numpy()
        gap_mask = diffs > gap_seconds

    events[gap_mask] = 0

    model = HawkesDiscreteHazardModel()
    curr_h = model.mu
    for t in range(mle_window, n):
        if t == mle_window or (t - mle_window) % refit_interval == 0:
            window_events = events[t - mle_window : t + 1]
            model.fit(window_events, start_idx=50)
            if t == mle_window:
                curr_h = model.mu
        if gap_mask[t]:
            curr_h = model.mu
        alpha = model.gamma * (1.0 - model.rho)
        curr_h = model.mu + model.rho * (curr_h - model.mu) + alpha * float(events[t])
        curr_h = max(curr_h, 1e-8)
        values[t] = curr_h
        is_valid[t] = True

    return pd.DataFrame({col_id: values, f"{col_id}_valid": is_valid}, index=df.index)


def _unit_hamilton_pred_prob(df: pd.DataFrame, col_id: str, params: Dict[str, Any]) -> pd.DataFrame:
    from research_pipeline.discovery.econometric_modules import HamiltonLogRVRegimeModel

    window_days = int(params.get("window_days", 60))
    min_bars_per_day = int(params.get("min_bars_per_day", 1))
    regime_idx = int(params.get("regime_idx", 1))

    n = len(df)
    values = np.full(n, np.nan, dtype=np.float64)
    is_valid = np.zeros(n, dtype=bool)

    ts = df["time"] if "time" in df.columns else df["timestamp"]
    ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    avail_out = ts_utc + pd.Timedelta(minutes=5)

    close = df["close"].to_numpy(dtype=np.float64)
    ret = np.full(n, np.nan, dtype=np.float64)
    ret[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] != 0.0, close[:-1], 1.0)

    bar_dates = ts_utc.dt.date.to_numpy()
    unique_days = sorted(set(bar_dates))
    if len(unique_days) <= window_days:
        return pd.DataFrame(
            {col_id: values, f"{col_id}_valid": is_valid, "available_at": avail_out},
            index=df.index,
        )

    day_log_rv = np.full(len(unique_days), np.nan, dtype=np.float64)
    day_valid = np.zeros(len(unique_days), dtype=bool)
    for idx, d in enumerate(unique_days):
        mask = (bar_dates == d) & np.isfinite(ret)
        cnt = int(np.sum(mask))
        if cnt >= min_bars_per_day:
            rv = float(np.sum(ret[mask] ** 2))
            if rv > 0.0 and np.isfinite(rv):
                day_log_rv[idx] = float(np.log(rv))
                day_valid[idx] = True

    model = HamiltonLogRVRegimeModel()
    prev_params = None
    em_step = 0
    day_pred: Dict[Any, float] = {}
    day_avail: Dict[Any, pd.Timestamp] = {}

    for d_idx in range(window_days, len(unique_days)):
        target_day = unique_days[d_idx]
        win_valid = day_valid[d_idx - window_days : d_idx]
        if not bool(np.all(win_valid)):
            continue
        y_window = day_log_rv[d_idx - window_days : d_idx]
        res = model.step_daily_update_with_fallback(y_window, day_idx=em_step, prev_params=prev_params)
        em_step += 1
        if float(res["mu"][0]) > float(res["mu"][1]):
            raise RuntimeError(f"HAMILTON_STATE_ORDER_VIOLATION: mu[0]={res['mu'][0]} > mu[1]={res['mu'][1]}")
        prev_params = {"mu": res["mu"], "sigma": res["sigma"], "P": res["P"]}
        day_pred[target_day] = float(res["pred_prob_next_day"][regime_idx])
        day_avail[target_day] = pd.Timestamp(target_day, tz="UTC")

    for i in range(n):
        d = bar_dates[i]
        if d in day_pred:
            av = day_avail[d]
            vu = av + pd.Timedelta(days=1)
            dec = ts_utc.iloc[i] + pd.Timedelta(minutes=5)
            if av <= dec <= vu:
                values[i] = day_pred[d]
                is_valid[i] = True
                avail_out.iloc[i] = av

    return pd.DataFrame(
        {col_id: values, f"{col_id}_valid": is_valid, "available_at": avail_out},
        index=df.index,
    )

    close = df["close"].to_numpy(dtype=np.float64)
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] != 0.0, close[:-1], 1.0)

    events, _ = HawkesDiscreteHazardModel.extract_events_from_returns(
        returns, lookback_window=lookback, threshold_mult=threshold_sigma
    )
    events = events.copy()

    # Detect timestamp gaps (> gap_seconds)
    gap_mask = np.zeros(n, dtype=bool)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        diffs = ts.diff().dt.total_seconds().fillna(0.0).to_numpy()
        gap_mask = diffs > gap_seconds
    elif isinstance(df.index, pd.DatetimeIndex):
        diffs = df.index.to_series().diff().dt.total_seconds().fillna(0.0).to_numpy()
        gap_mask = diffs > gap_seconds

    events[gap_mask] = 0

    model = HawkesDiscreteHazardModel()
    curr_h = model.mu
    for t in range(mle_window, n):
        if t == mle_window or (t - mle_window) % refit_interval == 0:
            window_events = events[t - mle_window : t + 1]
            model.fit(window_events, start_idx=50)
            if t == mle_window:
                curr_h = model.mu
        if gap_mask[t]:
            curr_h = model.mu
        alpha = model.gamma * (1.0 - model.rho)
        curr_h = model.mu + model.rho * (curr_h - model.mu) + alpha * float(events[t])
        curr_h = max(curr_h, 1e-8)
        values[t] = curr_h
        is_valid[t] = True

    return pd.DataFrame({col_id: values, f"{col_id}_valid": is_valid}, index=df.index)


def _unit_hamilton_pred_prob(df: pd.DataFrame, col_id: str, params: Dict[str, Any]) -> pd.DataFrame:
    from research_pipeline.discovery.econometric_modules import HamiltonLogRVRegimeModel

    window_days = int(params.get("window_days", 60))
    min_bars_per_day = int(params.get("min_bars_per_day", 1))
    regime_idx = int(params.get("regime_idx", 1))

    n = len(df)
    values = np.full(n, np.nan, dtype=np.float64)
    is_valid = np.zeros(n, dtype=bool)

    ts = df["time"] if "time" in df.columns else df["timestamp"]
    ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    avail_out = ts_utc + pd.Timedelta(minutes=5)

    close = df["close"].to_numpy(dtype=np.float64)
    ret = np.full(n, np.nan, dtype=np.float64)
    ret[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] != 0.0, close[:-1], 1.0)

    bar_dates = ts_utc.dt.date.to_numpy()
    unique_days = sorted(set(bar_dates))
    if len(unique_days) <= window_days:
        return pd.DataFrame(
            {col_id: values, f"{col_id}_valid": is_valid, "available_at": avail_out},
            index=df.index,
        )

    day_log_rv = np.full(len(unique_days), np.nan, dtype=np.float64)
    day_valid = np.zeros(len(unique_days), dtype=bool)
    for idx, d in enumerate(unique_days):
        mask = (bar_dates == d) & np.isfinite(ret)
        cnt = int(np.sum(mask))
        if cnt >= min_bars_per_day:
            rv = float(np.sum(ret[mask] ** 2))
            if rv > 0.0 and np.isfinite(rv):
                day_log_rv[idx] = float(np.log(rv))
                day_valid[idx] = True

    model = HamiltonLogRVRegimeModel()
    prev_params = None
    em_step = 0
    day_pred: Dict[Any, float] = {}
    day_avail: Dict[Any, pd.Timestamp] = {}

    for d_idx in range(window_days, len(unique_days)):
        target_day = unique_days[d_idx]
        win_valid = day_valid[d_idx - window_days : d_idx]
        if not bool(np.all(win_valid)):
            continue
        y_window = day_log_rv[d_idx - window_days : d_idx]
        res = model.step_daily_update_with_fallback(y_window, day_idx=em_step, prev_params=prev_params)
        em_step += 1
        if float(res["mu"][0]) > float(res["mu"][1]):
            raise RuntimeError(f"HAMILTON_STATE_ORDER_VIOLATION: mu[0]={res['mu'][0]} > mu[1]={res['mu'][1]}")
        prev_params = {"mu": res["mu"], "sigma": res["sigma"], "P": res["P"]}
        day_pred[target_day] = float(res["pred_prob_next_day"][regime_idx])
        day_avail[target_day] = pd.Timestamp(target_day, tz="UTC")

    for i in range(n):
        d = bar_dates[i]
        if d in day_pred:
            av = day_avail[d]
            vu = av + pd.Timedelta(days=1)
            dec = ts_utc.iloc[i] + pd.Timedelta(minutes=5)
            if av <= dec <= vu:
                values[i] = day_pred[d]
                is_valid[i] = True
                avail_out.iloc[i] = av

    return pd.DataFrame(
        {col_id: values, f"{col_id}_valid": is_valid, "available_at": avail_out},
        index=df.index,
    )


def build_default_unit_registry() -> Dict[str, UnitSpec]:
    impl_id = compute_files_sha256([os.path.abspath(__file__), _ALGOFORGE_INDICATORS_PATH])
    hawkes_impl_id = compute_files_sha256([os.path.abspath(__file__), _ECONOMETRIC_MODULES_PATH])
    return {
        "unit:ema": UnitSpec(
            unit_id="unit:ema",
            version="1.0.0",
            impl_id=impl_id,
            dependencies=[],
            warmup_bars=1,
            output_schema={"value": "continuous"},
            fn=_unit_ema,
        ),
        "unit:rsi": UnitSpec(
            unit_id="unit:rsi",
            version="1.0.0",
            impl_id=impl_id,
            dependencies=[],
            warmup_bars=14,
            output_schema={"value": "continuous"},
            fn=_unit_rsi,
        ),
        "unit:hawkes_hazard": UnitSpec(
            unit_id="unit:hawkes_hazard",
            version="1.0.0",
            impl_id=hawkes_impl_id,
            dependencies=[],
            warmup_bars=501,
            output_schema={"value": "continuous"},
            fn=_unit_hawkes_hazard,
        ),
        "unit:hamilton_pred_prob": UnitSpec(
            unit_id="unit:hamilton_pred_prob",
            version="1.0.0",
            impl_id=hawkes_impl_id,
            dependencies=[],
            warmup_bars=60,
            output_schema={"value": "probability"},
            fn=_unit_hamilton_pred_prob,
            time_contract="explicit_available_at",
        ),
    }


UNIT_REGISTRY: Dict[str, UnitSpec] = build_default_unit_registry()
