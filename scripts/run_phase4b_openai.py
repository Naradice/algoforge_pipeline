"""Phase 4-B: OpenAI API Small-Scale Comparison (gpt-4o-mini-2024-07-18, 10 candidates)."""

import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import BacktestConfig, GateConfig, compute_indicators, run_causal_fx_backtest
from research_pipeline.backtest.candidate_generators import BaseAPIGenerator, BudgetGuard, EvoGenerator, GenerationContext
from research_pipeline.backtest.declarative_strategy import build_causal_htf_features, evaluate_declarative_signals
from research_pipeline.backtest.strategy_registry import StrategyRegistry

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from run_phase3_diversity_and_cv import evaluate_p3


def load_env_key(env_path: str, key_name: str) -> None:
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key_name}=") and not os.environ.get(key_name):
                    os.environ[key_name] = line.split("=", 1)[1].strip().strip('"').strip("'")


def openai_transport(model: str, n: int) -> dict:
    api_key = os.environ["OPENAI_API_KEY"]
    prompt = (
        f"Output ONLY a raw JSON array (no markdown fences, no explanation) of {n} diverse USDJPY M5 trading strategy objects matching this exact schema: "
        "each object has keys 'long' ({'op': 'AND'|'OR', 'clauses': [1-2 clauses]}), 'short' ({'op': 'AND'|'OR', 'clauses': [1-2 clauses]}), "
        "'htf_filter' ({'h1_align': bool, 'h4_align': bool}), 'hours_utc' (list of ints 0-23, or [] for all hours), "
        "'risk' ({'k_sl': float>0, 'k_tp': float>0, 'max_hold_bars': int>0}). "
        "Allowed clause types: {'type':'ema_cross','fast':int,'slow':int,'dir':1|-1} (fast<slow), "
        "{'type':'rsi_lt','period':int,'level':int} (5-95), {'type':'rsi_gt','period':int,'level':int} (5-95), "
        "{'type':'bb_break_upper','period':int,'std_mult':float}, {'type':'bb_break_lower','period':int,'std_mult':float}, "
        "{'type':'donchian_break_high','period':int}, {'type':'donchian_break_low','period':int}."
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    usage = data.get("usage", {})
    return {
        "text": text,
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        },
    }


def main():
    t0 = time.time()
    load_env_key(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")), "OPENAI_API_KEY")

    ledger_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "budget_ledger.jsonl"))
    guard = BudgetGuard(max_budget_usd=0.05, ledger_path=ledger_path)
    model_id = "gpt-4o-mini-2024-07-18"
    gen = BaseAPIGenerator("openai", model_id, "OPENAI_API_KEY", guard, EvoGenerator(42), transport_fn=openai_transport)

    cands = gen.generate(GenerationContext(), 10)
    print(f"[*] Generated {len(cands)} candidates via {cands[0].actual_generator} (model={cands[0].model}, spent=${guard.spent_usd:.6f})")

    snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    with open(snap_path, "rb") as f:
        dataset_sha = hashlib.sha256(f.read()).hexdigest()

    df = pd.read_parquet(snap_path)
    df = compute_indicators(df, BacktestConfig())
    htf = build_causal_htf_features(df)
    df["h1_trend"] = htf["h1_trend"]
    df["h4_trend"] = htf["h4_trend"]
    df["date"] = df["time"].dt.date
    dates = sorted(df["date"].unique())

    explore_start_idx = int(df[df["date"] == dates[60]].index[0])
    p2_split_date = dates[106]
    diag_start_idx = int(df[df["date"] == dates[152]].index[0])
    df_explore = df.iloc[:diag_start_idx].copy().reset_index(drop=True)

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "strategy_registry.db"))
    registry = StrategyRegistry(db_path)

    res_list = []
    for gc in cands:
        r = evaluate_p3(gc.strategy, df_explore, explore_start_idx, p2_split_date, dataset_sha, registry, None, None, f"api_{gc.actual_generator}", "phase4b_openai")
        res_list.append(r)

    valid = [r for r in res_list if r["status"] != "REJECTED_SCHEMA"]
    n_unique = len(set(r["strategy_hash"] for r in res_list))
    n_cache = sum(1 for r in res_list if r["cache_hit"])
    eligible = [r for r in valid if r["metrics"].get("trade_count", 0) >= 30]
    eligible.sort(key=lambda x: x["metrics"].get("min_cv_sharpe", -999.0), reverse=True)

    diag_top1 = {}
    if eligible:
        st = eligible[0]["strategy"]
        cfg = BacktestConfig(k_sl=st["risk"]["k_sl"], k_tp=st["risk"]["k_tp"], max_hold_bars=st["risk"]["max_hold_bars"])
        ls, ss = evaluate_declarative_signals(df, st)
        _, _, diag_top1 = run_causal_fx_backtest(df, cfg, GateConfig(), eval_start_idx=diag_start_idx, long_signal=ls, short_signal=ss)

    print(f"\n=== Phase 4-B: OpenAI API ({model_id}) ===")
    print(f"  Submitted={len(cands)} | ValidJSON={len(valid)} | Unique={n_unique} | CacheHits={n_cache} | Eligible(Tr>=30)={len(eligible)}")
    if eligible:
        m = eligible[0]["metrics"]
        print(f"  Top1 Explore: Sharpe={m['annualized_sharpe']:+.3f}, MinCV={m['min_cv_sharpe']:+.3f}, Trades={m['trade_count']}")
        print(f"  Top1 Diag40d: Sharpe={diag_top1.get('annualized_sharpe', 0):+.3f}, Return={diag_top1.get('total_return_pct', 0):+.2f}%, Trades={diag_top1.get('trade_count', 0)}")

    registry.close()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase4b_openai_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "actual_generator": cands[0].actual_generator,
            "fallback_reason": cands[0].fallback_reason,
            "api_spend_usd": guard.spent_usd,
            "submitted": len(cands),
            "valid_json": len(valid),
            "unique": n_unique,
            "cache_hits": n_cache,
            "eligible": len(eligible),
            "top1_explore": eligible[0]["metrics"] if eligible else {},
            "top1_diag40d": diag_top1,
            "elapsed_sec": time.time() - t0,
        }, f, indent=2)
    print(f"[*] Phase 4-B Complete in {time.time()-t0:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
