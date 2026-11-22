"""Minimal evaluator boundary protocol and LocalCausalEvaluator wrapping causal_fx_backtest."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Optional, Protocol

import pandas as pd

from research_pipeline.backtest import BacktestConfig, GateConfig, run_causal_fx_backtest
from research_pipeline.integration.algoforge_adapter import evaluate_algoforge_conditions_vectorized


@dataclass(frozen=True)
class LocalDatasetRef:
    af_ind: pd.DataFrame
    pl_df: Any
    eval_start_idx: int
    dataset_sha256: str


@dataclass(frozen=True)
class EvalResult:
    status: str
    summary: Dict[str, Any]
    engine_id: str
    dataset_sha256: str
    eval_config_hash: str
    run_ref: Optional[str] = None
    error: Optional[str] = None


class EvaluatorProtocol(Protocol):
    engine_id: str

    def evaluate(
        self,
        strategy_json: Dict[str, Any],
        dataset_ref: Any,
        eval_config: Dict[str, Any],
    ) -> EvalResult: ...


def compute_eval_config_hash(eval_config: Dict[str, Any]) -> str:
    """Deterministic SHA-256 metadata hash for evaluation configuration."""
    payload = json.dumps(eval_config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class LocalCausalEvaluator:
    """Wraps existing Phase 1B-2 vectorized condition evaluation and run_causal_fx_backtest."""

    engine_id: str = "causal_fx_backtest_v1.0"

    def evaluate(
        self,
        strategy_json: Dict[str, Any],
        dataset_ref: LocalDatasetRef,
        eval_config: Dict[str, Any],
    ) -> EvalResult:
        cfg: BacktestConfig = eval_config["backtest_config"]
        gate_cfg: GateConfig = eval_config["gate_config"]
        ls, ss = evaluate_algoforge_conditions_vectorized(
            dataset_ref.af_ind,
            strategy_json["long"]["entry"],
            strategy_json["short"]["entry"],
        )
        _, _, summary = run_causal_fx_backtest(
            dataset_ref.pl_df,
            cfg,
            gate_cfg,
            eval_start_idx=dataset_ref.eval_start_idx,
            long_signal=ls,
            short_signal=ss,
        )
        return EvalResult(
            status="ok",
            summary=summary,
            engine_id=self.engine_id,
            dataset_sha256=dataset_ref.dataset_sha256,
            eval_config_hash=compute_eval_config_hash(eval_config),
            run_ref=None,
            error=None,
        )
