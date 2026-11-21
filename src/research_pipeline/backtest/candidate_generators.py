"""Multi-Provider Declarative Strategy Candidate Generators & Pre-Call BudgetGuard."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np


DEFAULT_MODEL_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


@dataclass
class GenerationContext:
    """Model-agnostic context passed to any candidate generator."""
    top_strategies: List[Dict[str, Any]] = field(default_factory=list)
    rejection_summary: Dict[str, int] = field(default_factory=dict)
    seed: int = 42


@dataclass
class GeneratedCandidate:
    """Declarative strategy candidate with full provenance metadata."""
    strategy: Dict[str, Any]
    requested_generator: str
    actual_generator: str
    provider: str
    model: str
    fallback_reason: Optional[str] = None
    error_type: Optional[str] = None
    generation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    source_file: Optional[str] = None
    declared_model: Optional[str] = None


class BudgetGuard:
    """Pre-call budget enforcement and secret-free ledger logging."""

    def __init__(
        self,
        max_budget_usd: float,
        ledger_path: str,
        pricing_table: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.max_budget_usd = max_budget_usd
        self.ledger_path = ledger_path
        self.pricing_table = pricing_table or DEFAULT_MODEL_PRICING_USD_PER_1M
        self.spent_usd = 0.0
        self.unknown_cost_calls = 0

    def estimate_max_cost(self, model: str, est_input_tokens: int = 1500, max_output_tokens: int = 1000) -> float:
        prices = self.pricing_table.get(model)
        if prices is None:
            # Conservative fallback estimate for unknown models
            return (est_input_tokens * 5.0 + max_output_tokens * 15.0) / 1_000_000.0
        return (est_input_tokens * prices["input"] + max_output_tokens * prices["output"]) / 1_000_000.0

    def can_afford(self, model: str, est_input_tokens: int = 1500, max_output_tokens: int = 1000) -> bool:
        est = self.estimate_max_cost(model, est_input_tokens, max_output_tokens)
        return (self.spent_usd + est) <= self.max_budget_usd + 1e-12

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        generation_id: str,
    ) -> float:
        cost_unknown = (input_tokens is None) or (output_tokens is None)
        if cost_unknown:
            self.unknown_cost_calls += 1
            cost = 0.0
        else:
            prices = self.pricing_table.get(model, {"input": 5.0, "output": 15.0})
            cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000.0
            self.spent_usd += cost

        entry = {
            "generation_id": generation_id,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 8),
            "cost_unknown": cost_unknown,
            "cumulative_spent_usd": round(self.spent_usd, 8),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.ledger_path)), exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return cost


class CandidateGenerator(Protocol):
    def generate(self, context: GenerationContext, n: int) -> List[GeneratedCandidate]:
        ...


class EvoGenerator:
    """Local evolutionary/random declarative generator ($0.00)."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _random_clause(self, direction: int) -> Dict[str, Any]:
        ctype = str(self.rng.choice(["ema_cross", "rsi_extreme", "bb_break", "donchian_break"]))
        if ctype == "ema_cross":
            return {"type": "ema_cross", "fast": int(self.rng.choice([8, 12, 16])), "slow": int(self.rng.choice([32, 48, 64])), "dir": direction}
        if ctype == "rsi_extreme":
            p = int(self.rng.choice([10, 14, 21]))
            return {"type": "rsi_lt" if direction == 1 else "rsi_gt", "period": p, "level": int(self.rng.choice([25, 30, 35] if direction == 1 else [65, 70, 75]))}
        if ctype == "bb_break":
            return {"type": "bb_break_upper" if direction == 1 else "bb_break_lower", "period": int(self.rng.choice([20, 30])), "std_mult": float(self.rng.choice([1.5, 2.0, 2.5]))}
        return {"type": "donchian_break_high" if direction == 1 else "donchian_break_low", "period": int(self.rng.choice([12, 20, 36]))}

    def generate(self, context: GenerationContext, n: int, requested_generator: str = "evo", fallback_reason: Optional[str] = None, error_type: Optional[str] = None) -> List[GeneratedCandidate]:
        out: List[GeneratedCandidate] = []
        for _ in range(n):
            strat = {
                "long": {"op": str(self.rng.choice(["AND", "OR"])), "clauses": [self._random_clause(1)]},
                "short": {"op": str(self.rng.choice(["AND", "OR"])), "clauses": [self._random_clause(-1)]},
                "htf_filter": {"h1_align": bool(self.rng.choice([True, False])), "h4_align": bool(self.rng.choice([True, False]))},
                "hours_utc": [],
                "risk": {"k_sl": float(self.rng.choice([1.5, 2.0, 2.5])), "k_tp": float(self.rng.choice([2.5, 3.0, 4.0])), "max_hold_bars": int(self.rng.choice([24, 48, 72]))},
            }
            out.append(
                GeneratedCandidate(
                    strategy=strat,
                    requested_generator=requested_generator,
                    actual_generator="evo",
                    provider="local",
                    model="evo_v1",
                    fallback_reason=fallback_reason,
                    error_type=error_type,
                )
            )
        return out


class BaseAPIGenerator:
    """Base class for API LLM generators with pre-call BudgetGuard and Evo fallback."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key_env: str,
        budget_guard: BudgetGuard,
        fallback_evo: EvoGenerator,
        transport_fn: Optional[Callable[[str, int], Dict[str, Any]]] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.budget_guard = budget_guard
        self.fallback_evo = fallback_evo
        self.transport_fn = transport_fn
        self.error_counts: Dict[str, int] = {}

    def _record_error(self, err_type: str) -> None:
        self.error_counts[err_type] = self.error_counts.get(err_type, 0) + 1

    def generate(self, context: GenerationContext, n: int) -> List[GeneratedCandidate]:
        req_name = f"{self.provider}_api"
        # 1. Check API key
        if not os.environ.get(self.api_key_env) and self.transport_fn is None:
            self._record_error("MISSING_API_KEY")
            return self.fallback_evo.generate(context, n, requested_generator=req_name, fallback_reason="MISSING_API_KEY", error_type="MISSING_API_KEY")

        # 2. Pre-call budget check
        if not self.budget_guard.can_afford(self.model):
            self._record_error("BUDGET_EXCEEDED")
            return self.fallback_evo.generate(context, n, requested_generator=req_name, fallback_reason="BUDGET_EXCEEDED", error_type="BUDGET_EXCEEDED")

        gen_id = str(uuid.uuid4())[:12]
        try:
            if self.transport_fn is None:
                raise RuntimeError("No live HTTP transport configured in offline mode")
            resp = self.transport_fn(self.model, n)
            usage = resp.get("usage") or {}
            self.budget_guard.record_usage(
                self.provider,
                self.model,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                gen_id,
            )
            raw_text = resp.get("text", "")
            parsed = json.loads(raw_text)
            if not isinstance(parsed, list):
                raise ValueError("Expected JSON list of strategies")
            out: List[GeneratedCandidate] = []
            for item in parsed[:n]:
                out.append(
                    GeneratedCandidate(
                        strategy=item,
                        requested_generator=req_name,
                        actual_generator=req_name,
                        provider=self.provider,
                        model=self.model,
                        generation_id=gen_id,
                    )
                )
            if len(out) < n:
                out.extend(self.fallback_evo.generate(context, n - len(out), requested_generator=req_name, fallback_reason="PARTIAL_API_OUTPUT"))
            return out
        except json.JSONDecodeError as e:
            self._record_error("INVALID_JSON")
            return self.fallback_evo.generate(context, n, requested_generator=req_name, fallback_reason="INVALID_JSON", error_type=str(type(e).__name__))
        except Exception as e:
            self._record_error("API_ERROR")
            return self.fallback_evo.generate(context, n, requested_generator=req_name, fallback_reason="API_ERROR", error_type=str(type(e).__name__))


class AgentJSONGenerator:
    """Ingests declarative JSON batches generated by Agent AI (Claude Code / Antigravity)."""

    def __init__(self, json_path: str, declared_model: str = "agent_unknown"):
        self.json_path = json_path
        self.declared_model = declared_model

    def generate(self, context: GenerationContext, n: int) -> List[GeneratedCandidate]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Agent JSON file must contain a JSON list of strategies")
        out: List[GeneratedCandidate] = []
        for item in data[:n]:
            out.append(
                GeneratedCandidate(
                    strategy=item,
                    requested_generator="agent_json",
                    actual_generator="agent_json",
                    provider="agent_subscription",
                    model=f"declared:{self.declared_model}",
                    source_file=os.path.basename(self.json_path),
                    declared_model=self.declared_model,
                )
            )
        return out
