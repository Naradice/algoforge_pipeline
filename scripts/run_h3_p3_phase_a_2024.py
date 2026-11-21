"""H3-P3 Gate 2: 2024 Phase A 10-Trial Selection & Specification Freeze.

Strictly uses ONLY 2024 snapshot (data/usdjpy_m5_2024_oanda_snapshot.parquet).
2025 holdout remains 100% sealed and untouched.
"""

import hashlib
import json
import os
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
    LocalJumpDetector,
)
from research_pipeline.backtest import (
    BacktestConfig,
    GateConfig,
    compute_indicators,
    run_causal_fx_backtest,
    compute_hac_one_sided_test,
)


def main():
    t0 = time.time()
    snapshot_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet")
    )
    with open(snapshot_path, "rb") as f:
        snap_sha = hashlib.sha256(f.read()).hexdigest()
    assert snap_sha == "bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0"
    print(f"[*] 2024 Snapshot SHA-256 verified: {snap_sha}")

    df = pd.read_parquet(snapshot_path)
    base_cfg = BacktestConfig()
    df = compute_indicators(df, base_cfg)
    total_bars = len(df)
    assert total_bars == 55201

    df["ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df.loc[df["is_gap_reopen"], "ret"] = 0.0
    df["date"] = df["time"].dt.date
    dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    df["day_idx"] = df["date"].map(date_to_idx)

    eval_start_idx = int(df[df["day_idx"] == 60].index[0])
    print(f"[*] 2024 Calendar: {len(dates)} days, eval_start_idx={eval_start_idx}")

    # 1. Hawkes 50-bar periodic hazard
    returns_arr = df["ret"].values
    is_gap_arr = df["is_gap_reopen"].values
    events, _ = HawkesDiscreteHazardModel.extract_events_from_returns(
        returns_arr, lookback_window=20, threshold_mult=2.0
    )
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

    q85 = float(np.percentile(h_periodic[:eval_start_idx], 85))
    q90 = float(np.percentile(h_periodic[:eval_start_idx], 90))
    print(f"[*] Hawkes warmup quantiles: q85={q85:.6f}, q90={q90:.6f}")

    # 2. Jump STD 72-bar backward window
    jump_flags = np.zeros(total_bars, dtype=np.int32)
    for t in range(72, total_bars):
        win = returns_arr[t - 72 : t]
        # Only valid if >= 50 non-zero / non-gap bars
        if np.sum(win != 0.0) >= 50:
            std_est = float(np.std(win, ddof=1))
            if std_est > 1e-12 and abs(returns_arr[t]) > 3.0 * std_est:
                jump_flags[t] = 1
    print(f"[*] Jump STD flags detected: {int(jump_flags.sum())} total")

    # 3. Hamilton 30-day rolling Quantile (xi_predicted_before_day)
    daily_rv = df.groupby("day_idx").agg(
        date=("date", "first"),
        rv=("ret", lambda x: float(np.sum(x**2))),
    ).reset_index()
    daily_rv["log_rv"] = np.log(daily_rv["rv"].clip(lower=1e-12))

    day_pred_prob = {}
    for d in range(60, len(dates)):
        y_win = daily_rv.iloc[d - 30 : d]["log_rv"].values
        q25 = float(np.percentile(y_win, 25))
        q75 = float(np.percentile(y_win, 75))
        sy = float(max(np.std(y_win, ddof=1), 1e-3))
        sig_q = sy / np.sqrt(2.0)
        q_init = HamiltonInitialParams(
            mu=np.array([q25, q75]),
            sigma=np.array([sig_q, sig_q]),
            P=np.array([[0.90, 0.10], [0.20, 0.80]]),
        )
        m_q = HamiltonLogRVRegimeModel(
            mu_0=q25, mu_1=q75, sigma_0=sig_q, sigma_1=sig_q,
            p00=0.90, p11=0.80, window_days=30, max_iter=100, tol=1e-6,
        )
        m_q.fit_em(y_win, init_params=q_init)
        _, f_q, _ = m_q.forward_filter(y_win)
        p_pred = m_q.P.T @ f_q[-1]
        day_pred_prob[d] = float(p_pred[1])

    ham_prob_arr = df["day_idx"].map(lambda d: day_pred_prob.get(d, 0.0)).values

    # Step 1: S0 SL/TP selection (2 trials)
    print("\n[*] Step 1: S0 SL/TP Selection (2 trials)")
    s0_candidates = [(2.0, 3.0), (2.0, 4.0)]
    s0_results = []
    for idx, (k_sl, k_tp) in enumerate(s0_candidates):
        cfg = BacktestConfig(k_sl=k_sl, k_tp=k_tp)
        _, daily_s0, sum_s0 = run_causal_fx_backtest(
            df, cfg, GateConfig(), eval_start_idx=eval_start_idx
        )
        s0_results.append({
            "idx": idx, "k_sl": k_sl, "k_tp": k_tp,
            "summary": sum_s0, "daily": daily_s0,
        })
        print(f"  Trial {idx+1}: SL={k_sl}, TP={k_tp} | Sharpe={sum_s0['annualized_sharpe']:.4f}, MaxDD={sum_s0['max_drawdown_pct']:.2f}%, Trades={sum_s0['trade_count']}")

    best_s0 = max(
        s0_results,
        key=lambda x: (
            x["summary"]["annualized_sharpe"],
            x["summary"]["max_drawdown_pct"],  # less negative is better
            x["summary"]["trade_count"],
            -x["idx"],
        ),
    )
    chosen_sl, chosen_tp = best_s0["k_sl"], best_s0["k_tp"]
    print(f"[*] Selected S0: SL={chosen_sl}, TP={chosen_tp}")

    # Step 2: S_full gate selection (8 trials)
    print("\n[*] Step 2: S_full Gate Selection (8 trials)")
    cfg_fixed = BacktestConfig(k_sl=chosen_sl, k_tp=chosen_tp)
    sfull_candidates = []
    idx = 0
    for h_name, h_th in [("q85", q85), ("q90", q90)]:
        for j_m in [3, 6]:
            for ham_th in [0.60, 0.80]:
                gates = GateConfig(
                    use_hawkes=True, hawkes_threshold=h_th,
                    use_jump=True, jump_cooldown_bars=j_m,
                    use_hamilton=True, hamilton_threshold=ham_th, hamilton_scale=0.5,
                )
                _, daily_sf, sum_sf = run_causal_fx_backtest(
                    df, cfg_fixed, gates,
                    hawkes_hazard=h_periodic,
                    jump_flags=jump_flags,
                    hamilton_pred_prob=ham_prob_arr,
                    eval_start_idx=eval_start_idx,
                )
                sfull_candidates.append({
                    "idx": idx, "hawkes_label": h_name, "hawkes_th": h_th,
                    "jump_m": j_m, "ham_th": ham_th,
                    "summary": sum_sf, "daily": daily_sf,
                })
                print(f"  Trial {idx+1}: H={h_name}({h_th:.5f}), J_m={j_m}, Ham={ham_th} | Sharpe={sum_sf['annualized_sharpe']:.4f}, MaxDD={sum_sf['max_drawdown_pct']:.2f}%, Trades={sum_sf['trade_count']}")
                idx += 1

    best_sfull = max(
        sfull_candidates,
        key=lambda x: (
            x["summary"]["annualized_sharpe"],
            x["summary"]["max_drawdown_pct"],
            x["summary"]["trade_count"],
            -x["idx"],
        ),
    )
    print(f"[*] Selected S_full: H={best_sfull['hawkes_label']}, J_m={best_sfull['jump_m']}, Ham={best_sfull['ham_th']}")

    # Compute 2024 HAC reference MDE (S_full vs S0)
    r_sf = best_sfull["daily"]["net_return"].values
    r_s0 = best_s0["daily"]["net_return"].values
    hac_2024 = compute_hac_one_sided_test(r_sf, r_s0, maxlags=4)
    # 80% power MDE at alpha=0.05 (z_0.95 + z_0.80 = 1.645 + 0.842 = 2.487) * SE_hac
    mde_2024_daily = 2.487 * hac_2024["se_hac"] if hac_2024["se_hac"] > 0 else 0.0

    frozen = {
        "frozen_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_2024_sha256": snap_sha,
        "backtest_config": {
            "initial_capital": cfg_fixed.initial_capital,
            "spread_jpy": cfg_fixed.spread_jpy,
            "slippage_jpy": cfg_fixed.slippage_jpy,
            "risk_fraction": cfg_fixed.risk_fraction,
            "max_leverage": cfg_fixed.max_leverage,
            "lot_step_usd": cfg_fixed.lot_step_usd,
            "k_sl": chosen_sl,
            "k_tp": chosen_tp,
            "max_hold_bars": cfg_fixed.max_hold_bars,
            "gap_cooldown_bars": cfg_fixed.gap_cooldown_bars,
        },
        "s_full_gates": {
            "hawkes_label": best_sfull["hawkes_label"],
            "hawkes_threshold": best_sfull["hawkes_th"],
            "jump_cooldown_bars": best_sfull["jump_m"],
            "hamilton_threshold": best_sfull["ham_th"],
            "hamilton_scale": 0.5,
        },
        "phase_a_2024_summary": {
            "s0": best_s0["summary"],
            "s_full": best_sfull["summary"],
            "hac_se_2024": hac_2024["se_hac"],
            "mde_2024_daily_return": mde_2024_daily,
        },
    }

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "h3_p3_frozen_config.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    with open(out_path, "rb") as f:
        frozen_sha = hashlib.sha256(f.read()).hexdigest()

    print(f"\n[*] Frozen config saved to {out_path}")
    print(f"[*] Frozen config SHA-256: {frozen_sha}")
    print(f"[*] Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
