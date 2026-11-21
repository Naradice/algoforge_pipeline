"""Offline Mock Test Suite for Multi-Provider Candidate Generators & BudgetGuard."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest.candidate_generators import (
    AgentJSONGenerator,
    BaseAPIGenerator,
    BudgetGuard,
    EvoGenerator,
    GenerationContext,
)
from research_pipeline.backtest.declarative_strategy import validate_declarative_strategy
from research_pipeline.backtest.strategy_registry import compute_strategy_hash

VALID_STRAT = {
    "long": {"op": "AND", "clauses": [{"type": "ema_cross", "fast": 12, "slow": 48, "dir": 1}]},
    "short": {"op": "AND", "clauses": [{"type": "ema_cross", "fast": 12, "slow": 48, "dir": -1}]},
    "htf_filter": {"h1_align": True, "h4_align": False},
    "hours_utc": [],
    "risk": {"k_sl": 2.0, "k_tp": 3.0, "max_hold_bars": 48},
}


def test_1_pre_call_budget_insufficient():
    """API must NOT be called when pre-call budget check fails; falls back to Evo."""
    with tempfile.TemporaryDirectory() as tmp:
        guard = BudgetGuard(max_budget_usd=0.0001, ledger_path=os.path.join(tmp, "ledger.jsonl"))
        called = {"count": 0}

        def mock_transport(model, n):
            called["count"] += 1
            return {"text": "[]", "usage": {"input_tokens": 100, "output_tokens": 100}}

        gen = BaseAPIGenerator("anthropic", "claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY", guard, EvoGenerator(42), transport_fn=mock_transport)
        res = gen.generate(GenerationContext(), 2)
        assert called["count"] == 0, "API should NOT have been called!"
        assert len(res) == 2
        assert res[0].requested_generator == "anthropic_api"
        assert res[0].actual_generator == "evo"
        assert res[0].fallback_reason == "BUDGET_EXCEEDED"
    print("[PASS] Test 1: Pre-call budget insufficient -> Evo fallback without calling API")


def test_2_missing_usage_recorded_as_unknown():
    """Missing API usage records cost_unknown=True without infinite retry."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.jsonl")
        guard = BudgetGuard(max_budget_usd=1.0, ledger_path=ledger)

        def mock_transport(model, n):
            return {"text": json.dumps([VALID_STRAT]), "usage": {}}

        gen = BaseAPIGenerator("openai", "gpt-4o-mini-2024-07-18", "OPENAI_API_KEY", guard, EvoGenerator(42), transport_fn=mock_transport)
        res = gen.generate(GenerationContext(), 1)
        assert len(res) == 1 and res[0].actual_generator == "openai_api"
        assert guard.unknown_cost_calls == 1
        with open(ledger, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["cost_unknown"] is True
    print("[PASS] Test 2: Missing API usage recorded as cost_unknown=True")


def test_3_invalid_json_fallback():
    """Invalid JSON from API triggers fallback and records error."""
    with tempfile.TemporaryDirectory() as tmp:
        guard = BudgetGuard(max_budget_usd=1.0, ledger_path=os.path.join(tmp, "ledger.jsonl"))

        def mock_transport(model, n):
            return {"text": "{not valid json", "usage": {"input_tokens": 100, "output_tokens": 50}}

        gen = BaseAPIGenerator("anthropic", "claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY", guard, EvoGenerator(42), transport_fn=mock_transport)
        res = gen.generate(GenerationContext(), 2)
        assert len(res) == 2
        assert res[0].actual_generator == "evo"
        assert res[0].fallback_reason == "INVALID_JSON"
        assert gen.error_counts["INVALID_JSON"] == 1
    print("[PASS] Test 3: Invalid JSON triggers Evo fallback & records error")


def test_4_missing_api_key_fallback():
    """Missing API key falls back to Evo cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        guard = BudgetGuard(max_budget_usd=1.0, ledger_path=os.path.join(tmp, "ledger.jsonl"))
        os.environ.pop("NONEXISTENT_KEY_XYZ", None)
        gen = BaseAPIGenerator("anthropic", "claude-3-5-sonnet-20241022", "NONEXISTENT_KEY_XYZ", guard, EvoGenerator(42), transport_fn=None)
        res = gen.generate(GenerationContext(), 1)
        assert res[0].actual_generator == "evo"
        assert res[0].fallback_reason == "MISSING_API_KEY"
    print("[PASS] Test 4: Missing API key -> Evo fallback")


def test_5_same_strategy_different_models_share_hash():
    """Same strategy from Claude and OpenAI shares strategy_hash while keeping separate metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        guard = BudgetGuard(max_budget_usd=1.0, ledger_path=os.path.join(tmp, "ledger.jsonl"))
        mock_t = lambda m, n: {"text": json.dumps([VALID_STRAT]), "usage": {"input_tokens": 50, "output_tokens": 50}}
        c_gen = BaseAPIGenerator("anthropic", "claude-3-5-sonnet-20241022", "K", guard, EvoGenerator(42), transport_fn=mock_t)
        o_gen = BaseAPIGenerator("openai", "gpt-4o-mini-2024-07-18", "K", guard, EvoGenerator(42), transport_fn=mock_t)

        r_c = c_gen.generate(GenerationContext(), 1)[0]
        r_o = o_gen.generate(GenerationContext(), 1)[0]
        assert compute_strategy_hash(r_c.strategy) == compute_strategy_hash(r_o.strategy)
        assert r_c.provider == "anthropic" and r_o.provider == "openai"
    print("[PASS] Test 5: Same strategy from different models shares strategy_hash with separate provenance")


def test_6_agent_json_invalid_rejected():
    """Agent JSON with unknown/invalid fields is rejected by schema validator."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "agent_batch.json")
        bad_strat = {"long": {"op": "AND", "clauses": [{"type": "magic_unknown"}]}, "short": VALID_STRAT["short"], "risk": VALID_STRAT["risk"]}
        with open(p, "w", encoding="utf-8") as f:
            json.dump([bad_strat], f)
        agen = AgentJSONGenerator(p, declared_model="claude_code_sonnet")
        res = agen.generate(GenerationContext(), 1)
        assert res[0].declared_model == "claude_code_sonnet"
        ok, msg = validate_declarative_strategy(res[0].strategy)
        assert not ok and "unknown clause" in msg
    print("[PASS] Test 6: Invalid Agent JSON rejected by schema validator")


def test_7_fallback_provenance_distinguished():
    """Fallback provenance distinguishes requested_generator from actual_generator."""
    # Already verified in Tests 1, 3, 4; explicit check
    with tempfile.TemporaryDirectory() as tmp:
        guard = BudgetGuard(max_budget_usd=0.0, ledger_path=os.path.join(tmp, "ledger.jsonl"))
        gen = BaseAPIGenerator("openai", "gpt-4o", "K", guard, EvoGenerator(42), transport_fn=lambda m, n: {})
        res = gen.generate(GenerationContext(), 1)[0]
        assert res.requested_generator == "openai_api"
        assert res.actual_generator == "evo"
        assert res.provider == "local"
    print("[PASS] Test 7: Fallback provenance (requested vs actual) distinguished")


if __name__ == "__main__":
    test_1_pre_call_budget_insufficient()
    test_2_missing_usage_recorded_as_unknown()
    test_3_invalid_json_fallback()
    test_4_missing_api_key_fallback()
    test_5_same_strategy_different_models_share_hash()
    test_6_agent_json_invalid_rejected()
    test_7_fallback_provenance_distinguished()
    print("ALL 7 CANDIDATE GENERATOR TESTS PASSED.")
