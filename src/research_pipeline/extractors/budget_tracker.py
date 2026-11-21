"""Two-phase Budget Tracker for LLM inference (H1a Step 10 & User Instruction 4).

Features:
- Pre-call Reservation (estimated input tokens + max output tokens + allowance)
- Post-call Settlement (actual usage from response headers)
- Configurable budget ceiling via HDE_MAX_BUDGET_USD (default: $10.00 USD)
- Per-request cost ledger tracking every individual API call
- Markdown table formatting and JSONL audit export
- Dual caps: USD budget and token budget (Hard Stop on either reached)
"""

import os
import threading
import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field


class BudgetExceededError(Exception):
    """Raised when an API call would exceed the monetary or token budget."""
    pass


class ModelPricing(BaseModel):
    """Pricing per 1M tokens in USD."""
    input_price_per_1m: float
    output_price_per_1m: float


# Default conservative pricing table (USD per 1M tokens)
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # Google GenAI models
    "gemini-2.5-flash": ModelPricing(input_price_per_1m=0.10, output_price_per_1m=0.40),
    "gemini-2.5-pro": ModelPricing(input_price_per_1m=1.50, output_price_per_1m=6.00),
    # OpenAI models
    "gpt-4o-mini": ModelPricing(input_price_per_1m=0.15, output_price_per_1m=0.60),
    "gpt-4o": ModelPricing(input_price_per_1m=2.50, output_price_per_1m=10.00),
    "gpt-4-turbo": ModelPricing(input_price_per_1m=10.00, output_price_per_1m=30.00),
    "default": ModelPricing(input_price_per_1m=2.50, output_price_per_1m=10.00),
}


class RequestCostEntry(BaseModel):
    """Detailed audit record for a single settled API call."""
    call_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    cumulative_spent_usd: float
    remaining_budget_usd: float
    timestamp: str


class BudgetState(BaseModel):
    """Snapshot of current budget consumption, reservations, and request ledger."""
    spent_usd: float = 0.0
    spent_tokens: int = 0
    reserved_usd: float = 0.0
    reserved_tokens: int = 0
    unsettled_reservations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ledger: List[RequestCostEntry] = Field(default_factory=list)
    total_calls_settled: int = 0
    total_calls_refused: int = 0


class BudgetTracker:
    """Thread-safe two-phase budget manager with per-request audit ledger."""

    DEFAULT_BUDGET_USD: float = 10.00
    DEFAULT_BUDGET_TOKENS: int = 5_000_000

    def __init__(
        self,
        max_usd: Optional[float] = None,
        max_tokens: int = DEFAULT_BUDGET_TOKENS,
        pricing_table: Optional[Dict[str, ModelPricing]] = None,
    ):
        env_budget = os.environ.get("HDE_MAX_BUDGET_USD")
        if max_usd is not None:
            self.max_usd = max_usd
        elif env_budget:
            try:
                self.max_usd = float(env_budget)
            except ValueError:
                self.max_usd = self.DEFAULT_BUDGET_USD
        else:
            self.max_usd = self.DEFAULT_BUDGET_USD

        self.max_tokens = max_tokens
        self.pricing_table = pricing_table or DEFAULT_PRICING
        self._lock = threading.Lock()
        self.state = BudgetState()

    def get_pricing(self, model_id: str) -> ModelPricing:
        for key in self.pricing_table:
            if key in model_id:
                return self.pricing_table[key]
        return self.pricing_table["default"]

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.get_pricing(model_id)
        cost = (input_tokens / 1_000_000) * pricing.input_price_per_1m + (
            output_tokens / 1_000_000
        ) * pricing.output_price_per_1m
        return round(cost, 6)

    def reserve(
        self,
        reservation_id: str,
        model_id: str,
        estimated_input_tokens: int,
        max_output_tokens: int = 4096,
        allowance_tokens: int = 200,
    ) -> float:
        """Reserve worst-case budget before making an LLM API call (Pre-call Reservation)."""
        worst_case_input = estimated_input_tokens + allowance_tokens
        worst_case_tokens = worst_case_input + max_output_tokens
        worst_case_cost = self.estimate_cost(model_id, worst_case_input, max_output_tokens)

        with self._lock:
            projected_usd = self.state.spent_usd + self.state.reserved_usd + worst_case_cost
            projected_tokens = self.state.spent_tokens + self.state.reserved_tokens + worst_case_tokens

            if projected_usd > self.max_usd:
                self.state.total_calls_refused += 1
                raise BudgetExceededError(
                    f"Monetary budget exceeded: projected ${projected_usd:.4f} > limit ${self.max_usd:.2f} "
                    f"(spent=${self.state.spent_usd:.4f}, reserved=${self.state.reserved_usd:.4f}, new=${worst_case_cost:.4f})"
                )

            if projected_tokens > self.max_tokens:
                self.state.total_calls_refused += 1
                raise BudgetExceededError(
                    f"Token budget exceeded: projected {projected_tokens} tokens > limit {self.max_tokens} "
                    f"(spent={self.state.spent_tokens}, reserved={self.state.reserved_tokens}, new={worst_case_tokens})"
                )

            self.state.reserved_usd += worst_case_cost
            self.state.reserved_tokens += worst_case_tokens
            self.state.unsettled_reservations[reservation_id] = {
                "model_id": model_id,
                "reserved_usd": worst_case_cost,
                "reserved_tokens": worst_case_tokens,
            }
            return worst_case_cost

    def settle(
        self,
        reservation_id: str,
        actual_input_tokens: int,
        actual_output_tokens: int,
    ) -> float:
        """Settle actual token consumption, record into request ledger, and update totals."""
        with self._lock:
            if reservation_id not in self.state.unsettled_reservations:
                model_id = "default"
                reserved_usd = 0.0
                reserved_tokens = 0
            else:
                entry = self.state.unsettled_reservations.pop(reservation_id)
                model_id = entry["model_id"]
                reserved_usd = entry["reserved_usd"]
                reserved_tokens = entry["reserved_tokens"]

            actual_total_tokens = actual_input_tokens + actual_output_tokens
            actual_cost = self.estimate_cost(model_id, actual_input_tokens, actual_output_tokens)

            # Atomic release and update
            self.state.reserved_usd = max(0.0, self.state.reserved_usd - reserved_usd)
            self.state.reserved_tokens = max(0, self.state.reserved_tokens - reserved_tokens)
            self.state.spent_usd += actual_cost
            self.state.spent_tokens += actual_total_tokens
            self.state.total_calls_settled += 1

            remaining_usd = max(0.0, self.max_usd - self.state.spent_usd - self.state.reserved_usd)

            # Record into per-request audit ledger
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self.state.ledger.append(
                RequestCostEntry(
                    call_id=reservation_id,
                    model_id=model_id,
                    input_tokens=actual_input_tokens,
                    output_tokens=actual_output_tokens,
                    total_tokens=actual_total_tokens,
                    cost_usd=round(actual_cost, 6),
                    cumulative_spent_usd=round(self.state.spent_usd, 6),
                    remaining_budget_usd=round(remaining_usd, 6),
                    timestamp=now_iso,
                )
            )

            return actual_cost

    def release_reservation(self, reservation_id: str) -> None:
        """Release reservation without incurring cost (e.g. if call aborted before network dispatch)."""
        with self._lock:
            if reservation_id in self.state.unsettled_reservations:
                entry = self.state.unsettled_reservations.pop(reservation_id)
                self.state.reserved_usd = max(0.0, self.state.reserved_usd - entry["reserved_usd"])
                self.state.reserved_tokens = max(0, self.state.reserved_tokens - entry["reserved_tokens"])

    def get_ledger(self) -> List[RequestCostEntry]:
        """Return a copy of the request cost audit ledger."""
        with self._lock:
            return list(self.state.ledger)

    def format_ledger_table(self) -> str:
        """Format the per-request ledger into an aligned Markdown table."""
        with self._lock:
            if not self.state.ledger:
                return "No API calls recorded in budget ledger."

            lines = [
                "| # | Call ID | Model | In Tokens | Out Tokens | Cost (USD) | Cumulative (USD) | Remaining (USD) |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for idx, entry in enumerate(self.state.ledger, 1):
                lines.append(
                    f"| {idx} | `{entry.call_id}` | `{entry.model_id}` | {entry.input_tokens:,} | "
                    f"{entry.output_tokens:,} | ${entry.cost_usd:.6f} | ${entry.cumulative_spent_usd:.6f} | ${entry.remaining_budget_usd:.4f} |"
                )
            return "\n".join(lines)

    def export_ledger_jsonl(self, path: Path) -> None:
        """Export the request ledger to a JSONL file."""
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for entry in self.state.ledger:
                    f.write(entry.model_dump_json() + "\n")

    def get_summary(self) -> Dict[str, Any]:
        """Return budget summary for experiment auditing."""
        with self._lock:
            return {
                "max_budget_usd": self.max_usd,
                "spent_usd": round(self.state.spent_usd, 6),
                "spent_tokens": self.state.spent_tokens,
                "reserved_usd": round(self.state.reserved_usd, 6),
                "reserved_tokens": self.state.reserved_tokens,
                "remaining_usd": round(max(0.0, self.max_usd - self.state.spent_usd - self.state.reserved_usd), 6),
                "remaining_tokens": max(0, self.max_tokens - self.state.spent_tokens - self.state.reserved_tokens),
                "total_calls_settled": self.state.total_calls_settled,
                "total_calls_refused": self.state.total_calls_refused,
                "total_ledger_entries": len(self.state.ledger),
            }
