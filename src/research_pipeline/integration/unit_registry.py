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


def build_default_unit_registry() -> Dict[str, UnitSpec]:
    impl_id = compute_files_sha256([os.path.abspath(__file__), _ALGOFORGE_INDICATORS_PATH])
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
    }


UNIT_REGISTRY: Dict[str, UnitSpec] = build_default_unit_registry()
