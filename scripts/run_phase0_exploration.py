"""Phase 0-C: Minimal Automated Declarative Strategy Exploration Pipeline.

Generates 30 declarative JSON strategy candidates (including 3 intentional invalid schemas
to verify graceful rejection), validates schemas, runs common causal backtest on 2024 data
(60-day warmup + 132-day evaluation), filters/ranks candidates, and logs full timing & cost.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.backtest import (
    BacktestConfig,
    GateConfig,
    compute_indicators,
    run_causal_fx_backtest,
)


def generate_declarative_candidates(seed: int = 42, n_valid: int = 27) -> List[Dict[str, Any]]:
    """Generate declarative JSON strategy candidates with fixed seed + 3 invalid ones."""
    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Any]] = []
    families = ["ema_cross", "rsi_reversion", "bb_breakout", "donchian_breakout"]

    for i in range(n_valid):
        fam = families[i % len(families)]
        k_sl = float(rng.choice([1.5, 2.0, 2.5]))
        k_tp = float(rng.choice([2.5, 3.0, 4.0]))
        max_hold = int(rng.choice([24, 48, 72]))

        if fam == "ema_cross":
            fast = int(rng.choice([8, 12, 16]))
            slow = int(rng.choice([32, 48, 64]))
            params = {"fast": fast, "slow": slow}
        elif fam == "rsi_reversion":
            period = int(rng.choice([10, 14, 21]))
            ob = int(rng.choice([70, 75]))
            os_lvl = int(rng.choice([25, 30]))
            params = {"period": period, "overbought": ob, "oversold": os_lvl}
        elif fam == "bb_breakout":
            period = int(rng.choice([20, 30]))
            std_mult = float(rng.choice([1.5, 2.0, 2.5]))
            params = {"period": period, "std_mult": std_mult}
        else:  # donchian_breakout
            period = int(rng.choice([12, 20, 36]))
            params = {"period": period}

        candidates.append({
            "candidate_id": f"cand_{i+1:02d}_{fam}",
            "family": fam,
            "params": params,
            "risk": {"k_sl": k_sl, "k_tp": k_tp, "max_hold_bars": max_hold},
        })

    # Add 3 intentional invalid candidates to test graceful rejection
    candidates.append({"candidate_id": "cand_28_invalid_ema", "family": "ema_cross", "params": {"fast": 50, "slow": 20}, "risk": {"k_sl": 2.0, "k_tp": 3.0, "max_hold_bars": 48}})
    candidates.append({"candidate_id": "cand_29_invalid_sl", "family": "donchian_breakout", "params": {"period": 20}, "risk": {"k_sl": -1.0, "k_tp": 3.0, "max_hold_bars": 48}})
    candidates.append({"candidate_id": "cand_30_unknown_fam", "family": "unknown_magic", "params": {}, "risk": {"k_sl": 2.0, "k_tp": 3.0, "max_hold_bars": 48}})
    return candidates


def validate_candidate(cand: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate declarative JSON schema and parameter constraints."""
    fam = cand.get("family")
    p = cand.get("params", {})
    r = cand.get("risk", {})
    if fam not in {"ema_cross", "rsi_reversion", "bb_breakout", "donchian_breakout"}:
        return False, f"REJECTED_INVALID_SCHEMA: unsupported family {fam}"
    if r.get("k_sl", 0) <= 0 or r.get("k_tp", 0) <= 0 or r.get("max_hold_bars", 0) <= 0:
        return False, "REJECTED_INVALID_SCHEMA: non-positive risk params"
    if fam == "ema_cross" and p.get("fast", 0) >= p.get("slow", 0):
        return False, "REJECTED_INVALID_SCHEMA: ema fast >= slow"
    return True, "OK"


def build_signals(df: pd.DataFrame, cand: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Build causal boolean long/short entry signals from declarative JSON."""
    fam = cand["family"]
    p = cand["params"]
    close = df["close"]
    n = len(df)

    if fam == "ema_cross":
        ef = close.ewm(span=p["fast"], adjust=False).mean()
        es = close.ewm(span=p["slow"], adjust=False).mean()
        diff = ef - es
        pdiff = diff.shift(1)
        return ((diff > 0) & (pdiff <= 0)).values, ((diff < 0) & (pdiff >= 0)).values

    if fam == "rsi_reversion":
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(p["period"]).mean()
        loss = (-delta.clip(upper=0.0)).rolling(p["period"]).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        prsi = rsi.shift(1)
        long_sig = ((rsi > p["oversold"]) & (prsi <= p["oversold"])).fillna(False).values
        short_sig = ((rsi < p["overbought"]) & (prsi >= p["overbought"])).fillna(False).values
        return long_sig, short_sig

    if fam == "bb_breakout":
        ma = close.rolling(p["period"]).mean()
        sd = close.rolling(p["period"]).std(ddof=1)
        upper = ma + p["std_mult"] * sd
        lower = ma - p["std_mult"] * sd
        long_sig = ((close > upper) & (close.shift(1) <= upper.shift(1))).fillna(False).values
        short_sig = ((close < lower) & (close.shift(1) >= lower.shift(1))).fillna(False).values
        return long_sig, short_sig

    if fam == "donchian_breakout":
        hi = df["high"].rolling(p["period"]).max().shift(1)
        lo = df["low"].rolling(p["period"]).min().shift(1)
        long_sig = (close > hi).fillna(False).values
        short_sig = (close < lo).fillna(False).values
        return long_sig, short_sig

    return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)


def main():
    t_start = time.time()
    snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    df = pd.read_parquet(snap_path)
    base_cfg = BacktestConfig()
    df = compute_indicators(df, base_cfg)
    df["date"] = df["time"].dt.date
    dates = sorted(df["date"].unique())
    eval_start_idx = int(df[df["date"] == dates[60]].index[0])

    candidates = generate_declarative_candidates(seed=42, n_valid=27)
    records = []
    n_success = 0
    n_rejected_schema = 0
    n_rejected_perf = 0

    for cand in candidates:
        t_c = time.time()
        ok, msg = validate_candidate(cand)
        if not ok:
            n_rejected_schema += 1
            records.append({
                "candidate_id": cand["candidate_id"],
                "status": "REJECTED",
                "reason": msg,
                "elapsed_sec": time.time() - t_c,
            })
            continue

        cfg = BacktestConfig(
            k_sl=cand["risk"]["k_sl"],
            k_tp=cand["risk"]["k_tp"],
            max_hold_bars=cand["risk"]["max_hold_bars"],
        )
        long_sig, short_sig = build_signals(df, cand)
        _, _, summary = run_causal_fx_backtest(
            df, cfg, GateConfig(), eval_start_idx=eval_start_idx,
            long_signal=long_sig, short_signal=short_sig,
        )
        elapsed = time.time() - t_c
        n_success += 1

        if summary["trade_count"] < 30:
            status = "REJECTED"
            reason = "REJECTED_LOW_TRADES"
            n_rejected_perf += 1
        elif summary["annualized_sharpe"] <= 0.0:
            status = "REJECTED"
            reason = "REJECTED_NEGATIVE_SHARPE"
            n_rejected_perf += 1
        else:
            status = "PROMOTED"
            reason = "PASS_MIN_CRITERIA"

        records.append({
            "candidate_id": cand["candidate_id"],
            "family": cand["family"],
            "params": cand["params"],
            "risk": cand["risk"],
            "status": status,
            "reason": reason,
            "summary": summary,
            "elapsed_sec": elapsed,
        })

    total_elapsed = time.time() - t_start
    valid_evals = [r for r in records if "summary" in r]
    valid_evals.sort(key=lambda x: x["summary"]["annualized_sharpe"], reverse=True)

    print(f"[*] Phase 0-C Complete: {len(candidates)} candidates in {total_elapsed:.2f}s (avg {total_elapsed/max(n_success,1):.2f}s/eval)")
    print(f"    - Evaluated: {n_success} | Schema Rejected: {n_rejected_schema} | Perf Rejected: {n_rejected_perf} | Promoted: {sum(1 for r in records if r['status']=='PROMOTED')}")
    print("\n[*] Top 5 Candidates (2024 132-day Exploration):")
    for r in valid_evals[:5]:
        s = r["summary"]
        print(f"  {r['candidate_id']:26s} | Sharpe={s['annualized_sharpe']:+.4f} | Return={s['total_return_pct']:+.2f}% | MaxDD={s['max_drawdown_pct']:.2f}% | Trades={s['trade_count']} | {r['reason']}")

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(candidates),
        "evaluated": n_success,
        "rejected_schema": n_rejected_schema,
        "rejected_perf": n_rejected_perf,
        "promoted": sum(1 for r in records if r["status"] == "PROMOTED"),
        "total_elapsed_sec": total_elapsed,
        "avg_sec_per_candidate": total_elapsed / max(n_success, 1),
        "api_spend_usd": 0.0,
        "top_5": valid_evals[:5],
        "all_records": records,
    }
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase0_exploration_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[*] Saved to {out_path}")


if __name__ == "__main__":
    main()
