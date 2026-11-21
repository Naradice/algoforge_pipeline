"""H3-P3 Phase B: One-Shot 2025 Blind Holdout Confirmatory Evaluation.

Strictly uses frozen config (SHA-256: a61434766dd6891eaca8f09c09df2b2a4b259aeed018cf369d90b0794945ec70).
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.discovery.econometric_modules import (
    HawkesDiscreteHazardModel,
    HamiltonLogRVRegimeModel,
    HamiltonInitialParams,
)
from research_pipeline.backtest import (
    BacktestConfig,
    GateConfig,
    compute_indicators,
    run_causal_fx_backtest,
    compute_hac_one_sided_test,
    run_holm_bonferroni,
)


def main():
    t0 = time.time()
    algoforge_dir = r"c:\Users\ksato\workspace\finance\algoforge"
    pre_git = subprocess.run(["git", "-C", algoforge_dir, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()

    # 1. Verify frozen config SHA-256
    frozen_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "h3_p3_frozen_config.json"))
    with open(frozen_path, "rb") as f:
        frozen_sha = hashlib.sha256(f.read()).hexdigest()
    expected_frozen_sha = "a61434766dd6891eaca8f09c09df2b2a4b259aeed018cf369d90b0794945ec70"
    assert frozen_sha == expected_frozen_sha, f"Frozen config SHA-256 mismatch: {frozen_sha}"
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen = json.load(f)
    print(f"[*] Frozen config SHA-256 verified: {frozen_sha}")

    # 2. Load 2024 snapshot
    snap24_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    with open(snap24_path, "rb") as f:
        sha24 = hashlib.sha256(f.read()).hexdigest()
    assert sha24 == frozen["snapshot_2024_sha256"]
    df24 = pd.read_parquet(snap24_path)
    df24["time"] = pd.to_datetime(df24["time"], utc=True)

    # 3. Unseal 2025 data from OANDA MT5 CSV
    raw_csv = r"L:\data\mt5\OANDA-Japan MT5 Live\mt5_USDJPY_min5.csv"
    print(f"[*] Unsealing 2025 data from: {raw_csv}")
    df_raw = pd.read_csv(raw_csv)
    # Standardize column names
    df_raw.columns = [c.lower().strip() for c in df_raw.columns]
    if "datetime" in df_raw.columns and "time" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"datetime": "time"})
    df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)
    df25 = df_raw[(df_raw["time"] >= "2025-01-01 00:00:00+00:00") & (df_raw["time"] < "2026-01-01 00:00:00+00:00")].copy()
    df25 = df25.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    snap25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2025_oanda_snapshot.parquet"))
    df25.to_parquet(snap25_path, index=False)
    with open(snap25_path, "rb") as f:
        sha25 = hashlib.sha256(f.read()).hexdigest()
    print(f"[*] 2025 Snapshot saved: {len(df25)} bars, SHA-256: {sha25}")

    # 4. Pre-Registered 2025 Data Quality Audit
    bad_bars = df25["open"].isna() | df25["high"].isna() | df25["low"].isna() | df25["close"].isna() | (df25["high"] < df25["low"]) | (df25["close"] <= 0)
    bad_rate = float(bad_bars.mean())
    dates25 = sorted(df25["time"].dt.date.unique())
    num_days25 = len(dates25)
    print(f"[*] 2025 Quality Audit: {num_days25} trading days, bad_bar_rate={bad_rate:.6f}")
    if bad_rate > 0.005 or num_days25 < 120:
        raise RuntimeError(f"ABORTED_DATA_QUALITY: bad_rate={bad_rate}, num_days={num_days25}")

    # 5. Concatenate 2024 + 2025 for causal state continuity (eval starts at 2025 Day 0)
    df_all = pd.concat([df24[["time", "open", "high", "low", "close"]], df25[["time", "open", "high", "low", "close"]]], ignore_index=True)
    bc = frozen["backtest_config"]
    cfg = BacktestConfig(
        initial_capital=bc["initial_capital"],
        spread_jpy=bc["spread_jpy"],
        slippage_jpy=bc["slippage_jpy"],
        risk_fraction=bc["risk_fraction"],
        max_leverage=bc["max_leverage"],
        lot_step_usd=bc["lot_step_usd"],
        k_sl=bc["k_sl"],
        k_tp=bc["k_tp"],
        max_hold_bars=bc["max_hold_bars"],
        gap_cooldown_bars=bc["gap_cooldown_bars"],
    )
    df_all = compute_indicators(df_all, cfg)
    total_bars = len(df_all)
    eval_start_idx = len(df24)
    print(f"[*] Combined bars: {total_bars} (2024={eval_start_idx}, 2025={len(df25)}, eval_start_idx={eval_start_idx})")

    df_all["ret"] = np.log(df_all["close"] / df_all["close"].shift(1)).fillna(0.0)
    df_all.loc[df_all["is_gap_reopen"], "ret"] = 0.0
    df_all["date"] = df_all["time"].dt.date
    all_dates = sorted(df_all["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    df_all["day_idx"] = df_all["date"].map(date_to_idx)
    day0_2025_idx = date_to_idx[dates25[0]]

    # 6. Causal Hawkes (full-history state continuity + 500-bar rolling fit every 50 bars)
    returns_arr = df_all["ret"].values
    is_gap_arr = df_all["is_gap_reopen"].values
    events, _ = HawkesDiscreteHazardModel.extract_events_from_returns(returns_arr, lookback_window=20, threshold_mult=2.0)
    h_periodic = np.zeros(total_bars, dtype=np.float64)
    model_p = HawkesDiscreteHazardModel(mu=0.005, rho=0.80, gamma=0.30)
    curr_h = model_p.mu
    for t in range(total_bars):
        if is_gap_arr[t]:
            curr_h = model_p.mu
        h_periodic[t] = curr_h
        if t >= 500 and (t - 500) % 50 == 0:
            model_p.fit(events, start_idx=t - 500, end_idx=t + 1)
        alpha = model_p.gamma * (1.0 - model_p.rho)
        curr_h = max(model_p.mu + model_p.rho * (curr_h - model_p.mu) + alpha * events[t], 1e-8)

    # 7. Causal Jump STD 72-bar backward window
    jump_flags = np.zeros(total_bars, dtype=np.int32)
    for t in range(72, total_bars):
        win = returns_arr[t - 72 : t]
        if np.sum(win != 0.0) >= 50:
            std_est = float(np.std(win, ddof=1))
            if std_est > 1e-12 and abs(returns_arr[t]) > 3.0 * std_est:
                jump_flags[t] = 1

    # 8. Causal Hamilton 30-day rolling Quantile (xi_predicted_before_day)
    daily_rv = df_all.groupby("day_idx").agg(
        date=("date", "first"),
        rv=("ret", lambda x: float(np.sum(x**2))),
    ).reset_index()
    daily_rv["log_rv"] = np.log(daily_rv["rv"].clip(lower=1e-12))

    day_pred_prob = {}
    for d in range(day0_2025_idx, len(all_dates)):
        y_win = daily_rv.iloc[d - 30 : d]["log_rv"].values
        q25 = float(np.percentile(y_win, 25))
        q75 = float(np.percentile(y_win, 75))
        sy = float(max(np.std(y_win, ddof=1), 1e-3))
        sig_q = sy / np.sqrt(2.0)
        q_init = HamiltonInitialParams(
            mu=np.array([q25, q75]), sigma=np.array([sig_q, sig_q]), P=np.array([[0.90, 0.10], [0.20, 0.80]])
        )
        m_q = HamiltonLogRVRegimeModel(
            mu_0=q25, mu_1=q75, sigma_0=sig_q, sigma_1=sig_q, p00=0.90, p11=0.80, window_days=30, max_iter=100, tol=1e-6
        )
        m_q.fit_em(y_win, init_params=q_init)
        _, f_q, _ = m_q.forward_filter(y_win)
        p_pred = m_q.P.T @ f_q[-1]
        day_pred_prob[d] = float(p_pred[1])

    ham_prob_arr = df_all["day_idx"].map(lambda d: day_pred_prob.get(d, 0.0)).values

    # 9. Run 5 configurations on 2025
    fg = frozen["s_full_gates"]
    configs = {
        "S0": GateConfig(use_hawkes=False, use_jump=False, use_hamilton=False),
        "S_full": GateConfig(
            use_hawkes=True, hawkes_threshold=fg["hawkes_threshold"],
            use_jump=True, jump_cooldown_bars=fg["jump_cooldown_bars"],
            use_hamilton=True, hamilton_threshold=fg["hamilton_threshold"], hamilton_scale=fg["hamilton_scale"],
        ),
        "S_full_minus_Hawkes": GateConfig(
            use_hawkes=False,
            use_jump=True, jump_cooldown_bars=fg["jump_cooldown_bars"],
            use_hamilton=True, hamilton_threshold=fg["hamilton_threshold"], hamilton_scale=fg["hamilton_scale"],
        ),
        "S_full_minus_Jump": GateConfig(
            use_hawkes=True, hawkes_threshold=fg["hawkes_threshold"],
            use_jump=False,
            use_hamilton=True, hamilton_threshold=fg["hamilton_threshold"], hamilton_scale=fg["hamilton_scale"],
        ),
        "S_full_minus_Hamilton": GateConfig(
            use_hawkes=True, hawkes_threshold=fg["hawkes_threshold"],
            use_jump=True, jump_cooldown_bars=fg["jump_cooldown_bars"],
            use_hamilton=False,
        ),
    }

    results_2025 = {}
    daily_returns = {}
    print("\n[*] Running 2025 One-Shot Confirmatory Evaluations:")
    for name, gcfg in configs.items():
        _, daily_df, summary = run_causal_fx_backtest(
            df_all, cfg, gcfg,
            hawkes_hazard=h_periodic,
            jump_flags=jump_flags,
            hamilton_pred_prob=ham_prob_arr,
            eval_start_idx=eval_start_idx,
        )
        results_2025[name] = summary
        daily_returns[name] = daily_df["net_return"].values
        print(f"  {name:22s} | Sharpe={summary['annualized_sharpe']:+.4f} | MaxDD={summary['max_drawdown_pct']:.2f}% | Return={summary['total_return_pct']:+.2f}% | Trades={summary['trade_count']}")

    # 10. 4 Confirmatory Hypotheses (HAC + Holm-Bonferroni)
    r_sfull = daily_returns["S_full"]
    hypotheses = []
    for hid, ref_name, desc in [
        ("H_P3-1", "S0", "S_full vs S0"),
        ("H_P3-2", "S_full_minus_Hawkes", "S_full vs S_full - Hawkes"),
        ("H_P3-3", "S_full_minus_Jump", "S_full vs S_full - Jump"),
        ("H_P3-4", "S_full_minus_Hamilton", "S_full vs S_full - Hamilton"),
    ]:
        hac = compute_hac_one_sided_test(r_sfull, daily_returns[ref_name], maxlags=4)
        hypotheses.append({
            "id": hid, "description": desc,
            "mean_diff": hac["mean_diff"], "se_hac": hac["se_hac"],
            "t_stat": hac["t_stat"], "p_val": hac["p_val"], "status": hac["status"],
        })

    holm_out = run_holm_bonferroni(hypotheses, alpha=0.05)
    print("\n[*] Holm-Bonferroni Confirmatory Results (M=4, FWER=0.05):")
    for h in holm_out:
        print(f"  Rank {h['rank']}: {h['id']} ({h['description']}) | mean_diff={h['mean_diff']:+.6f} | t={h['t_stat']:+.4f} | p={h['p_val']:.6f} vs alpha={h['holm_alpha']:.5f} -> {h['decision']}")

    post_git = subprocess.run(["git", "-C", algoforge_dir, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    assert pre_git == post_git, "AlgoForge was modified!"

    out_json = {
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_config_sha256": frozen_sha,
        "snapshot_2025_sha256": sha25,
        "trading_days_2025": num_days25,
        "bars_2025": len(df25),
        "bad_bar_rate_2025": bad_rate,
        "algoforge_clean": True,
        "api_spend_usd": 0.0,
        "strategies_2025": results_2025,
        "holm_bonferroni_2025": holm_out,
    }
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "h3_p3_phase_b_2025_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[*] Results saved to {out_path} in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
