"""Phase 4-A: E2E Connection Test for Agent AI Declarative Strategy JSON."""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import BacktestConfig, GateConfig, compute_indicators, run_causal_fx_backtest
from research_pipeline.backtest.candidate_generators import AgentJSONGenerator, GenerationContext
from research_pipeline.backtest.declarative_strategy import build_causal_htf_features, evaluate_declarative_signals
from research_pipeline.backtest.strategy_registry import StrategyRegistry

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from run_phase3_diversity_and_cv import evaluate_p3


def main():
    t0 = time.time()
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

    batch_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "agent_batches"))
    batches = [
        ("gemini_antigravity", os.path.join(batch_dir, "antigravity_batch_10.json")),
        ("claude_code", os.path.join(batch_dir, "claude_code_batch_10.json")),
    ]

    report = {}
    for model_name, b_path in batches:
        if not os.path.exists(b_path):
            print(f"[SKIP] {model_name}: {b_path} not found")
            continue
        agen = AgentJSONGenerator(b_path, declared_model=model_name)
        cands = agen.generate(GenerationContext(), 10)
        res_list = []
        for gc in cands:
            r = evaluate_p3(
                gc.strategy, df_explore, explore_start_idx, p2_split_date, dataset_sha,
                registry, None, None, f"agent_{model_name}", "phase4a_e2e"
            )
            r["declared_model"] = gc.declared_model
            r["source_file"] = gc.source_file
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

        print(f"\n=== Agent E2E: {model_name} ===")
        print(f"  Submitted={len(cands)} | ValidJSON={len(valid)} | Unique={n_unique} | CacheHits={n_cache} | Eligible(Tr>=30)={len(eligible)}")
        if eligible:
            m = eligible[0]["metrics"]
            print(f"  Top1 Explore: Sharpe={m['annualized_sharpe']:+.3f}, MinCV={m['min_cv_sharpe']:+.3f}, Trades={m['trade_count']}")
            print(f"  Top1 Diag40d: Sharpe={diag_top1.get('annualized_sharpe', 0):+.3f}, Return={diag_top1.get('total_return_pct', 0):+.2f}%, Trades={diag_top1.get('trade_count', 0)}")

        report[model_name] = {
            "submitted": len(cands),
            "valid_json": len(valid),
            "unique": n_unique,
            "cache_hits": n_cache,
            "eligible": len(eligible),
            "top1_explore": eligible[0]["metrics"] if eligible else {},
            "top1_diag40d": diag_top1,
        }

    registry.close()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase4a_e2e_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time() - t0, "api_spend_usd": 0.0, "models": report}, f, indent=2)
    print(f"\n[*] Phase 4-A E2E Complete in {time.time()-t0:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
