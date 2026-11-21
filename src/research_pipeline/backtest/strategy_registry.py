"""Persistent Canonical Strategy Registry for Deduplication & Evaluation Caching."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def canonicalize_condition_node(node: Any) -> Any:
    """Recursively canonicalize AND/OR commutative condition trees."""
    if isinstance(node, dict):
        if "op" in node and node["op"] in {"AND", "OR"} and isinstance(node.get("clauses"), list):
            canon_clauses = [canonicalize_condition_node(c) for c in node["clauses"]]
            canon_clauses.sort(key=lambda x: json.dumps(x, sort_keys=True))
            return {"op": node["op"], "clauses": canon_clauses}
        return {k: canonicalize_condition_node(v) for k, v in sorted(node.items())}
    if isinstance(node, list):
        return [canonicalize_condition_node(x) for x in node]
    if isinstance(node, float):
        return round(node, 10)
    return node


def canonicalize_strategy(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize strategy JSON (excluding metadata like candidate_id/parent_hash)."""
    core = {
        "long": canonicalize_condition_node(strategy.get("long", {})),
        "short": canonicalize_condition_node(strategy.get("short", {})),
        "htf_filter": canonicalize_condition_node(strategy.get("htf_filter", {})),
        "hours_utc": sorted(list(set(strategy.get("hours_utc", [])))),
        "risk": canonicalize_condition_node(strategy.get("risk", {})),
    }
    return core


def compute_strategy_hash(strategy: Dict[str, Any]) -> str:
    """Compute stable SHA-256 hash of canonicalized strategy."""
    canon = canonicalize_strategy(strategy)
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_evaluation_hash(
    strategy_hash: str,
    dataset_sha256: str,
    eval_period: str,
    engine_version: str,
    cost_config: Dict[str, Any],
) -> str:
    """Compute unique evaluation hash covering strategy + data + engine + cost."""
    payload = json.dumps(
        {
            "strategy_hash": strategy_hash,
            "dataset_sha256": dataset_sha256,
            "eval_period": eval_period,
            "engine_version": engine_version,
            "cost_config": canonicalize_condition_node(cost_config),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StrategyRegistry:
    """SQLite-backed persistent registry for evaluated strategies."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_hash TEXT PRIMARY KEY,
                strategy_hash TEXT NOT NULL,
                strategy_json TEXT NOT NULL,
                status TEXT NOT NULL,
                rejection_reason TEXT,
                metrics_json TEXT,
                generation_method TEXT,
                parent_hash TEXT,
                experiment_id TEXT,
                elapsed_sec REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_strategy_hash ON evaluations(strategy_hash)"
        )
        self.conn.commit()

    def lookup(self, evaluation_hash: str) -> Optional[Dict[str, Any]]:
        """Return cached complete evaluation if present (never returns ERROR/INCOMPLETE)."""
        cur = self.conn.execute(
            """
            SELECT strategy_hash, status, rejection_reason, metrics_json, elapsed_sec
            FROM evaluations
            WHERE evaluation_hash = ? AND status IN ('PROMOTED', 'REJECTED_PERF', 'REJECTED_SCHEMA')
            """,
            (evaluation_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "strategy_hash": row[0],
            "status": row[1],
            "rejection_reason": row[2],
            "metrics": json.loads(row[3]) if row[3] else {},
            "elapsed_sec": float(row[4]) if row[4] is not None else 0.0,
        }

    def is_strategy_rejected(self, strategy_hash: str) -> bool:
        """Check if strategy_hash was previously rejected."""
        cur = self.conn.execute(
            "SELECT 1 FROM evaluations WHERE strategy_hash = ? AND status IN ('REJECTED_PERF', 'REJECTED_SCHEMA') LIMIT 1",
            (strategy_hash,),
        )
        return cur.fetchone() is not None

    def record(
        self,
        evaluation_hash: str,
        strategy_hash: str,
        strategy: Dict[str, Any],
        status: str,
        rejection_reason: str,
        metrics: Dict[str, Any],
        generation_method: str,
        parent_hash: Optional[str],
        experiment_id: str,
        elapsed_sec: float,
    ) -> None:
        """Persist evaluation record."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO evaluations (
                evaluation_hash, strategy_hash, strategy_json, status,
                rejection_reason, metrics_json, generation_method,
                parent_hash, experiment_id, elapsed_sec, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_hash,
                strategy_hash,
                json.dumps(canonicalize_strategy(strategy), sort_keys=True),
                status,
                rejection_reason,
                json.dumps(metrics),
                generation_method,
                parent_hash,
                experiment_id,
                elapsed_sec,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
