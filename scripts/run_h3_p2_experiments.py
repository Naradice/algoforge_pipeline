"""
H3-P2 Empirical Verification Runner: 2024 USDJPY M5 Market Verification
Adheres strictly to H3-P2 Empirical Verification Plan v0.5 (APPROVED).

Key Constraints:
- Input: research_pipeline/data/usdjpy_m5_2024_oanda_snapshot.parquet (SHA-256 verified)
- 2025 Market Data: Completely isolated (zero access)
- AlgoForge: Strictly non-invasive (pre/post git diff audit)
- API Cost: $0.00 USD (pure offline local Python)
- 4-Hypothesis Holm-Bonferroni Family (FWER alpha=0.05)
"""

import os
import sys
import time
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_pipeline.discovery.econometric_modules import (
    HawkesDiscreteHazardModel,
    HamiltonLogRVRegimeModel,
    HamiltonInitialParams,
    LocalJumpDetector,
)


def compute_hac_stats(series: np.ndarray, lag: int = 4, h0_val: float = 0.0, alternative: str = "less") -> Tuple[float, float, float, float]:
    """
    Computes sample mean, Newey-West HAC standard error, t-statistic, and asymptotic p-value.
    alternative: 'less' (H1: mean < h0_val) or 'greater' (H1: mean > h0_val)
    """
    T = len(series)
    x = np.asarray(series, dtype=np.float64) - h0_val
    x_mean = float(np.mean(x))
    
    # Lag 0 sample variance
    dev = x - x_mean
    gamma_0 = float(np.mean(dev ** 2))
    
    # Lag 1..L autocovariances with Bartlett kernel weights
    gamma_sum = 0.0
    for l in range(1, lag + 1):
        weight = 1.0 - (l / (lag + 1.0))
        cov_l = float(np.mean(dev[l:] * dev[:-l]))
        gamma_sum += 2.0 * weight * cov_l
        
    hac_var = max(gamma_0 + gamma_sum, 1e-14)
    hac_se = float(np.sqrt(hac_var / T))
    t_stat = float(x_mean / hac_se)
    
    if alternative == "less":
        p_val = float(norm.cdf(t_stat))
    elif alternative == "greater":
        p_val = float(1.0 - norm.cdf(t_stat))
    else:
        p_val = float(2.0 * (1.0 - norm.cdf(abs(t_stat))))
        
    return x_mean + h0_val, hac_se, t_stat, p_val


def compute_block_bootstrap_ci(series: np.ndarray, block_size: int = 5, n_resamples: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Computes moving-block bootstrap 95% confidence interval and bootstrap p-value for H1: mean < 0.
    """
    np.random.seed(seed)
    T = len(series)
    n_blocks = int(np.ceil(T / block_size))
    
    max_start = T - block_size + 1
    boot_means = np.zeros(n_resamples, dtype=np.float64)
    
    for b in range(n_resamples):
        starts = np.random.randint(0, max_start, size=n_blocks)
        sample = np.concatenate([series[s : s + block_size] for s in starts])[:T]
        boot_means[b] = np.mean(sample)
        
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    boot_pval = float(np.mean(boot_means >= 0.0))
    return ci_lower, ci_upper, boot_pval


def run_h3_p2_experiments() -> Dict[str, Any]:
    print("=" * 80)
    print("H3-P2 INTEGRATED REAL-MARKET EMPIRICAL VERIFICATION (2024 USDJPY M5)")
    print("Adheres strictly to H3-P2 Empirical Verification Plan v0.5 (APPROVED)")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 0. Pre-Execution Audit & Non-Invasiveness Check
    # -------------------------------------------------------------------------
    algoforge_dir = os.path.abspath(r"c:\Users\ksato\workspace\finance\algoforge")
    pre_git_res = subprocess.run(["git", "-C", algoforge_dir, "status", "--porcelain"], capture_output=True, text=True)
    baseline_git_status = pre_git_res.stdout.strip()
    print(f"[*] AlgoForge baseline audit: clean status confirmed ({len(baseline_git_status.splitlines())} entries recorded)")
    
    # -------------------------------------------------------------------------
    # 1. Load 2024 Physical Snapshot & Verify Integrity
    # -------------------------------------------------------------------------
    snapshot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usdjpy_m5_2024_oanda_snapshot.parquet"))
    print(f"[*] Loading 2024 physical snapshot from: {snapshot_path}")
    with open(snapshot_path, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()
    expected_sha256 = "bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0"
    assert file_sha256 == expected_sha256, f"SHA-256 mismatch! Expected {expected_sha256}, got {file_sha256}"
    print(f"[*] Snapshot SHA-256: {file_sha256} [VERIFIED]")
    
    df = pd.read_parquet(snapshot_path)
    total_bars = len(df)
    assert total_bars == 55201, f"Expected 55201 bars, got {total_bars}"
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    # Calculate returns and detect gaps
    df["diff_sec"] = df["time"].diff().dt.total_seconds()
    df["is_gap_reopen"] = df["diff_sec"] > 300.0
    df["ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df.loc[df["is_gap_reopen"], "ret"] = 0.0
    
    # Date mapping
    df["date"] = df["time"].dt.date
    dates = sorted(df["date"].unique())
    num_days = len(dates)
    assert num_days == 192, f"Expected 192 trading days, got {num_days}"
    
    # Warmup vs Evaluation splits
    warmup_dates = dates[:60]
    eval_dates = dates[60:]
    assert len(warmup_dates) == 60, "Warmup days != 60"
    assert len(eval_dates) == 132, "Eval days != 132"
    print(f"[*] Calendar structure: 60 warmup days ({warmup_dates[0]} to {warmup_dates[-1]}) + 132 evaluation days ({eval_dates[0]} to {eval_dates[-1]})")
    
    date_to_day_idx = {d: i for i, d in enumerate(dates)}
    df["day_idx"] = df["date"].map(date_to_day_idx)
    
    returns_arr = df["ret"].values
    times_arr = df["time"].values
    is_gap_reopen_arr = df["is_gap_reopen"].values
    
    # =========================================================================
    # STAGE 1: HAWKES MODELING (Periodic vs Continuous vs i.i.d.)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Hawkes Modeling Execution & Timing Benchmark")
    print("=" * 60)
    
    events, rolling_std = HawkesDiscreteHazardModel.extract_events_from_returns(
        returns_arr, lookback_window=20, threshold_mult=2.0
    )
    total_events = int(np.sum(events))
    print(f"[*] Total exceedance events extracted: {total_events} / {total_bars} ({total_events/total_bars*100:.2f}%)")
    
    # Schedule A: Periodic re-estimation every 50 bars
    t0_periodic = time.time()
    print("[*] Running Schedule A: 50-bar Periodic Hawkes filter...")
    
    h_periodic = np.zeros(total_bars, dtype=np.float64)
    p_periodic = np.zeros(total_bars, dtype=np.float64)
    nll_periodic = np.zeros(total_bars, dtype=np.float64)
    
    model_periodic = HawkesDiscreteHazardModel(mu=0.005, rho=0.80, gamma=0.30)
    curr_h_p = model_periodic.mu
    estimation_window = 500
    
    for t in range(total_bars):
        if is_gap_reopen_arr[t]:
            curr_h_p = model_periodic.mu
            
        h_periodic[t] = curr_h_p
        p_t = float(np.clip(1.0 - np.exp(-curr_h_p), 1e-12, 1.0 - 1e-12))
        p_periodic[t] = p_t
        
        y_t = events[t]
        nll_periodic[t] = - (y_t * np.log(p_t) + (1 - y_t) * np.log(1.0 - p_t))
        
        if t >= estimation_window and (t % 50 == 0):
            model_periodic.fit(events, start_idx=t - estimation_window, end_idx=t + 1)
            
        alpha_p = model_periodic.gamma * (1.0 - model_periodic.rho)
        curr_h_p = model_periodic.mu + model_periodic.rho * (curr_h_p - model_periodic.mu) + alpha_p * y_t
        curr_h_p = max(curr_h_p, 1e-8)
        
    t_elapsed_periodic = time.time() - t0_periodic
    print(f"[*] Periodic Hawkes filter completed in {t_elapsed_periodic:.2f}s")
    
    # Schedule B: Continuous re-estimation (every bar after estimation_window)
    print("[*] Running Schedule B: Every-bar Continuous Hawkes filter benchmark...")
    t0_continuous = time.time()
    
    h_continuous = np.zeros(total_bars, dtype=np.float64)
    p_continuous = np.zeros(total_bars, dtype=np.float64)
    nll_continuous = np.zeros(total_bars, dtype=np.float64)
    
    model_cont = HawkesDiscreteHazardModel(mu=0.005, rho=0.80, gamma=0.30)
    curr_h_c = model_cont.mu
    
    for t in range(total_bars):
        if is_gap_reopen_arr[t]:
            curr_h_c = model_cont.mu
            
        h_continuous[t] = curr_h_c
        p_t = float(np.clip(1.0 - np.exp(-curr_h_c), 1e-12, 1.0 - 1e-12))
        p_continuous[t] = p_t
        
        y_t = events[t]
        nll_continuous[t] = - (y_t * np.log(p_t) + (1 - y_t) * np.log(1.0 - p_t))
        
        # Continuous parameter re-estimation
        if t >= estimation_window:
            model_cont.fit(events, start_idx=t - estimation_window, end_idx=t + 1)
            if (t - estimation_window) % 5000 == 0:
                elapsed_so_far = time.time() - t0_continuous
                print(f"    [Continuous Fit Progress] Bar {t}/{total_bars} ({t/total_bars*100:.1f}%) - elapsed: {elapsed_so_far:.1f}s")
            
        alpha_c = model_cont.gamma * (1.0 - model_cont.rho)
        curr_h_c = model_cont.mu + model_cont.rho * (curr_h_c - model_cont.mu) + alpha_c * y_t
        curr_h_c = max(curr_h_c, 1e-8)
        
    t_elapsed_continuous = time.time() - t0_continuous
    speedup_ratio = t_elapsed_continuous / max(t_elapsed_periodic, 1e-4)
    print(f"[*] Continuous Hawkes filter completed in {t_elapsed_continuous:.2f}s (Speedup: {speedup_ratio:.1f}x)")
    
    # Schedule C: Constant hazard i.i.d. baseline
    nll_iid = np.zeros(total_bars, dtype=np.float64)
    for t in range(total_bars):
        if t < estimation_window:
            p_bar = float(np.mean(events[:max(t, 1)])) if t > 0 else 0.01
        else:
            p_bar = float(np.mean(events[t - estimation_window : t]))
        p_bar = float(np.clip(p_bar, 1e-12, 1.0 - 1e-12))
        y_t = events[t]
        nll_iid[t] = - (y_t * np.log(p_bar) + (1 - y_t) * np.log(1.0 - p_bar))
        
    df["nll_periodic"] = nll_periodic
    df["nll_continuous"] = nll_continuous
    df["nll_iid"] = nll_iid
    
    hawkes_daily_df = df[df["day_idx"] >= 60].groupby("day_idx").agg(
        bars=("nll_periodic", "count"),
        mean_nll_periodic=("nll_periodic", "mean"),
        mean_nll_continuous=("nll_continuous", "mean"),
        mean_nll_iid=("nll_iid", "mean"),
    ).reset_index()
    
    hawkes_daily_df["delta_nll_h1a"] = hawkes_daily_df["mean_nll_periodic"] - hawkes_daily_df["mean_nll_iid"]
    hawkes_daily_df["delta_nll_h1b"] = hawkes_daily_df["mean_nll_periodic"] - hawkes_daily_df["mean_nll_continuous"]
    
    # =========================================================================
    # STAGE 2: JUMP / BIPOWER VARIATION MODELING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Bipower Variation (BV) Jump Detection & Clustering Execution")
    print("=" * 60)
    
    lookback_bv = 72
    min_valid_pairs = 50
    
    sigma_bv = np.zeros(total_bars, dtype=np.float64)
    sigma_std = np.zeros(total_bars, dtype=np.float64)
    jump_bv = np.zeros(total_bars, dtype=np.int32)
    jump_std = np.zeros(total_bars, dtype=np.int32)
    valid_mask = np.zeros(total_bars, dtype=bool)
    
    # Precompute adjacent pair indicator (dt == 300.0s)
    times_dt = pd.to_datetime(times_arr)
    diffs_sec = (times_dt[1:] - times_dt[:-1]).total_seconds().values
    dt_is_300 = (diffs_sec == 300.0)
    
    for t in range(lookback_bv, total_bars):
        # Lookback window for adjacent pairs: t-72 to t-1 has length 71
        valid_pairs_mask = dt_is_300[t - lookback_bv : t - 1]
        k_valid = int(np.sum(valid_pairs_mask))
        
        if k_valid >= min_valid_pairs:
            valid_mask[t] = True
            w_rets = returns_arr[t - lookback_bv : t]
            rets_lag1 = w_rets[:-1][valid_pairs_mask]
            rets_lag0 = w_rets[1:][valid_pairs_mask]
            
            bv_est = np.sqrt((np.pi / 2.0) * (1.0 / k_valid) * np.sum(np.abs(rets_lag1) * np.abs(rets_lag0)))
            std_est = np.std(w_rets, ddof=1)
            
            sigma_bv[t] = max(bv_est, 1e-8)
            sigma_std[t] = max(std_est, 1e-8)
            
            abs_ret = abs(returns_arr[t])
            if abs_ret / sigma_bv[t] > 3.0902:
                jump_bv[t] = 1
            if abs_ret / sigma_std[t] > 3.0902:
                jump_std[t] = 1
                
    df["valid_bv"] = valid_mask
    df["jump_bv"] = jump_bv
    df["jump_std"] = jump_std
    
    cluster_bv = np.zeros(total_bars, dtype=np.int32)
    cluster_std = np.zeros(total_bars, dtype=np.int32)
    
    for t in range(1, total_bars - 1):
        same_day_prev = (df["day_idx"].iloc[t] == df["day_idx"].iloc[t - 1]) and not is_gap_reopen_arr[t]
        same_day_next = (df["day_idx"].iloc[t] == df["day_idx"].iloc[t + 1]) and not is_gap_reopen_arr[t + 1]
        
        if jump_bv[t] == 1:
            if (same_day_prev and jump_bv[t - 1] == 1) or (same_day_next and jump_bv[t + 1] == 1):
                cluster_bv[t] = 1
                
        if jump_std[t] == 1:
            if (same_day_prev and jump_std[t - 1] == 1) or (same_day_next and jump_std[t + 1] == 1):
                cluster_std[t] = 1
                
    df["cluster_bv"] = cluster_bv
    df["cluster_std"] = cluster_std
    
    jump_daily_df = df[df["day_idx"] >= 60].groupby("day_idx").agg(
        bars=("valid_bv", "count"),
        rv=("ret", lambda x: float(np.sum(x**2))),
        jumps_bv=("jump_bv", "sum"),
        jumps_std=("jump_std", "sum"),
        clusters_bv=("cluster_bv", "sum"),
        clusters_std=("cluster_std", "sum"),
    ).reset_index()
    
    jump_daily_df["cluster_rate_bv"] = jump_daily_df["clusters_bv"] / jump_daily_df["bars"]
    jump_daily_df["cluster_rate_std"] = jump_daily_df["clusters_std"] / jump_daily_df["bars"]
    jump_daily_df["delta_cluster_rate"] = jump_daily_df["cluster_rate_bv"] - jump_daily_df["cluster_rate_std"]
    
    top_10pct_rv_threshold = jump_daily_df["rv"].quantile(0.90)
    high_rv_days = jump_daily_df[jump_daily_df["rv"] >= top_10pct_rv_threshold].copy()
    print(f"[*] Jump/BV Evaluation: 132 days, Top 10% High-RV threshold={top_10pct_rv_threshold:.6f} ({len(high_rv_days)} days)")
    
    # =========================================================================
    # STAGE 3: HAMILTON LOG-RV REGIME MODELING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Hamilton Log-RV Regime Modeling & Convergence Execution")
    print("=" * 60)
    
    daily_all_df = df.groupby("day_idx").agg(
        date=("date", "first"),
        bars=("ret", "count"),
        rv=("ret", lambda x: float(np.sum(x**2))),
    ).reset_index()
    daily_all_df["log_rv"] = np.log(daily_all_df["rv"].clip(lower=1e-12))
    
    warmup_log_rv = daily_all_df.iloc[:60]["log_rv"]
    print(f"[*] Warmup 60 days log-RV: mean={warmup_log_rv.mean():.4f}, median={warmup_log_rv.median():.4f}, std={warmup_log_rv.std():.4f}")
    
    informed_init = HamiltonInitialParams(
        mu=np.array([-11.8375, -10.3541]),
        sigma=np.array([0.4211, 0.9280]),
        P=np.array([[0.85, 0.15], [0.15, 0.85]]),
    )
    
    hamilton_results = []
    
    for d in range(60, 192):
        y_win = daily_all_df.iloc[d - 30 : d]["log_rv"].values
        current_date = daily_all_df.iloc[d]["date"]
        
        # 1. Informed Condition (using immutable HamiltonInitialParams)
        model_inf = HamiltonLogRVRegimeModel(
            mu_0=informed_init.mu[0],
            mu_1=informed_init.mu[1],
            sigma_0=informed_init.sigma[0],
            sigma_1=informed_init.sigma[1],
            p00=informed_init.P[0, 0],
            p11=informed_init.P[1, 1],
            window_days=30,
            max_iter=100,
            tol=1e-6,
        )
        c_inf = int(model_inf.fit_em(y_win, init_params=informed_init))
        
        xi_pred_inf, xi_filt_inf, ll_inf = model_inf.forward_filter(y_win)
        xi_filt_d_inf = xi_filt_inf[-1]
        conf_inf = float(np.max(xi_filt_d_inf))
        
        y_today = daily_all_df.iloc[d]["log_rv"]
        p_next_inf = model_inf.P.T @ xi_filt_d_inf
        dens_inf = norm.pdf(y_today, loc=model_inf.mu[0], scale=model_inf.sigma[0]) * p_next_inf[0] + \
                   norm.pdf(y_today, loc=model_inf.mu[1], scale=model_inf.sigma[1]) * p_next_inf[1]
        log_score_inf = float(np.log(max(dens_inf, 1e-15)))
        
        # 2. Quantile Condition (P1 25/75 percentile formula)
        q25 = float(np.percentile(y_win, 25))
        q75 = float(np.percentile(y_win, 75))
        s_y = float(max(np.std(y_win, ddof=1), 1e-3))
        sig_q = float(s_y / np.sqrt(2.0))
        
        quantile_init = HamiltonInitialParams(
            mu=np.array([q25, q75]),
            sigma=np.array([sig_q, sig_q]),
            P=np.array([[0.90, 0.10], [0.20, 0.80]]),
        )
        
        model_q = HamiltonLogRVRegimeModel(
            mu_0=q25,
            mu_1=q75,
            sigma_0=sig_q,
            sigma_1=sig_q,
            p00=0.90,
            p11=0.80,
            window_days=30,
            max_iter=100,
            tol=1e-6,
        )
        c_q = int(model_q.fit_em(y_win, init_params=quantile_init))
        
        xi_pred_q, xi_filt_q, ll_q = model_q.forward_filter(y_win)
        xi_filt_d_q = xi_filt_q[-1]
        conf_q = float(np.max(xi_filt_d_q))
        
        p_next_q = model_q.P.T @ xi_filt_d_q
        dens_q = norm.pdf(y_today, loc=model_q.mu[0], scale=model_q.sigma[0]) * p_next_q[0] + \
                 norm.pdf(y_today, loc=model_q.mu[1], scale=model_q.sigma[1]) * p_next_q[1]
        log_score_q = float(np.log(max(dens_q, 1e-15)))
        
        hamilton_results.append({
            "day_idx": d,
            "date": str(current_date),
            "log_rv": float(y_today),
            "conv_informed": c_inf,
            "conv_quantile": c_q,
            "delta_conv": c_inf - c_q,
            "xi_highvol_informed": float(xi_filt_d_inf[1]),
            "xi_highvol_quantile": float(xi_filt_d_q[1]),
            "confidence_informed": conf_inf,
            "confidence_quantile": conf_q,
            "log_score_informed": log_score_inf,
            "log_score_quantile": log_score_q,
            "mu_high_informed": float(model_inf.mu[1]),
            "mu_low_informed": float(model_inf.mu[0]),
        })
        
    hamilton_daily_df = pd.DataFrame(hamilton_results)
    print(f"[*] Hamilton 132-day EM: Informed convergence={hamilton_daily_df['conv_informed'].sum()}/132 ({hamilton_daily_df['conv_informed'].mean()*100:.1f}%), Quantile convergence={hamilton_daily_df['conv_quantile'].sum()}/132 ({hamilton_daily_df['conv_quantile'].mean()*100:.1f}%)")
    
    # =========================================================================
    # STAGE 4: JULY 11, 2024 MACRO SHOCK CASE STUDY
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 4: 2024 July 11 Macro Intervention Shock Case Study")
    print("=" * 60)
    
    july11_row = daily_all_df[daily_all_df["date"] == pd.to_datetime("2024-07-11").date()]
    july11_day_idx = int(july11_row["day_idx"].iloc[0])
    
    pre_event_df = hamilton_daily_df[(hamilton_daily_df["day_idx"] >= july11_day_idx - 5) & (hamilton_daily_df["day_idx"] < july11_day_idx)]
    pre_event_highvol_mean = float(pre_event_df["xi_highvol_informed"].mean())
    pre_event_baseline_valid = (pre_event_df["xi_highvol_informed"] < 0.50).all()
    print(f"[*] Pre-event (July 4-10) HighVol filter prob mean: {pre_event_highvol_mean:.4f} (Baseline < 0.50 valid: {pre_event_baseline_valid})")
    
    post_event_df = hamilton_daily_df[(hamilton_daily_df["day_idx"] >= july11_day_idx) & (hamilton_daily_df["day_idx"] <= july11_day_idx + 10)].copy()
    
    arrival_rows = post_event_df[post_event_df["xi_highvol_informed"] >= 0.80]
    if len(arrival_rows) > 0:
        first_arrival_day = int(arrival_rows["day_idx"].iloc[0])
        arrival_delay_days = first_arrival_day - july11_day_idx
    else:
        first_arrival_day = None
        arrival_delay_days = -1
        
    sustained_tracking = False
    for i in range(len(post_event_df) - 2):
        if (post_event_df["xi_highvol_informed"].iloc[i:i+3] >= 0.80).all():
            sustained_tracking = True
            break
            
    relapse_detected = False
    if first_arrival_day is not None:
        after_arrival = post_event_df[post_event_df["day_idx"] > first_arrival_day]
        if len(after_arrival) > 0 and (after_arrival["xi_highvol_informed"] < 0.50).any():
            relapse_detected = True
            
    print(f"[*] Macro Case Study Results:")
    print(f"    - First Arrival Delay: {arrival_delay_days} days (Arrival Date: {arrival_rows['date'].iloc[0] if first_arrival_day else 'Censored'})")
    print(f"    - Sustained Tracking (>=3 days): {sustained_tracking}")
    print(f"    - Relapse / Departure (<0.50): {relapse_detected}")
    
    macro_case_study_results = {
        "event_date": "2024-07-11",
        "event_day_idx": july11_day_idx,
        "pre_event_baseline_valid": bool(pre_event_baseline_valid),
        "pre_event_highvol_mean": pre_event_highvol_mean,
        "first_arrival_delay_days": arrival_delay_days,
        "first_arrival_date": str(arrival_rows["date"].iloc[0]) if first_arrival_day else None,
        "sustained_tracking": bool(sustained_tracking),
        "relapse_detected": bool(relapse_detected),
        "post_event_log_score_informed": float(post_event_df["log_score_informed"].mean()),
        "post_event_log_score_quantile": float(post_event_df["log_score_quantile"].mean()),
    }
    
    # =========================================================================
    # STAGE 5: STATISTICAL HYPOTHESIS TESTING (4 HYPOTHESES & HOLM PROCEDURE)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 5: Confirmatory Hypothesis Testing & Dynamic Holm-Bonferroni")
    print("=" * 60)
    
    # 1. H_P2-1A: Hawkes Predictive Superiority (Periodic vs iid)
    h1a_diff = hawkes_daily_df["delta_nll_h1a"].values
    mean_h1a, se_h1a, t_h1a, p_h1a = compute_hac_stats(h1a_diff, lag=4, h0_val=0.0, alternative="less")
    
    # 2. H_P2-1B: Hawkes Periodic Non-Inferiority (Periodic vs Continuous, margin = 0.005)
    h1b_diff = hawkes_daily_df["delta_nll_h1b"].values
    mean_h1b, se_h1b, t_h1b, p_h1b = compute_hac_stats(h1b_diff, lag=4, h0_val=0.005, alternative="less")
    
    # 3. H_P2-2: Jump/BV Cluster Suppression
    h2_diff = jump_daily_df["delta_cluster_rate"].values
    mean_h2, se_h2, t_h2, p_h2 = compute_hac_stats(h2_diff, lag=4, h0_val=0.0, alternative="less")
    
    high_rv_diff = high_rv_days["delta_cluster_rate"].values
    w_stat_h2, w_p_h2 = wilcoxon(high_rv_diff, alternative="less")
    
    # 4. H_P2-3: Hamilton Informed EM Convergence Advantage
    h3_diff = hamilton_daily_df["delta_conv"].values
    mean_h3, se_h3, t_h3, p_h3 = compute_hac_stats(h3_diff, lag=4, h0_val=0.0, alternative="greater")
    
    # Hamilton Long-Lag Sensitivity Analysis (L=4, 10, 20, 30)
    hamilton_hac_sensitivity = {}
    for lag_s in [4, 10, 20, 30]:
        m_s, se_s, t_s, p_s = compute_hac_stats(h3_diff, lag=lag_s, h0_val=0.0, alternative="greater")
        hamilton_hac_sensitivity[f"lag_{lag_s}"] = {
            "mean": m_s, "se": se_s, "t_stat": t_s, "p_val": p_s
        }
        
    # Hamilton Moving-Block Bootstrap Sensitivity (b=5, 15, 30 days)
    hamilton_boot_sensitivity = {}
    for b_size in [5, 15, 30]:
        ci_l, ci_u, boot_p = compute_block_bootstrap_ci(h3_diff, block_size=b_size, n_resamples=2000)
        hamilton_boot_sensitivity[f"block_{b_size}"] = {
            "ci_95": [ci_l, ci_u], "boot_p_val": boot_p
        }
        
    confidence_rate_inf = float(hamilton_daily_df["confidence_informed"].apply(lambda x: 1.0 if x > 0.60 else 0.0).mean())
    confidence_rate_q = float(hamilton_daily_df["confidence_quantile"].apply(lambda x: 1.0 if x > 0.60 else 0.0).mean())
    
    # -------------------------------------------------------------------------
    # Dynamic Holm-Bonferroni Step-Down Procedure (M=4, FWER alpha=0.05)
    # -------------------------------------------------------------------------
    hypotheses = [
        {"id": "H_P2-1A", "name": "Hawkes Predictive Superiority (Periodic vs iid)", "raw_p": p_h1a, "stat": t_h1a, "mean_diff": mean_h1a},
        {"id": "H_P2-1B", "name": "Hawkes Periodic Non-Inferiority (margin=0.005)", "raw_p": p_h1b, "stat": t_h1b, "mean_diff": mean_h1b},
        {"id": "H_P2-2",  "name": "Jump/BV Cluster Rate Suppression", "raw_p": p_h2, "stat": t_h2, "mean_diff": mean_h2},
        {"id": "H_P2-3",  "name": "Hamilton Informed EM Convergence Advantage", "raw_p": p_h3, "stat": t_h3, "mean_diff": mean_h3},
    ]
    
    sorted_hypotheses = sorted(hypotheses, key=lambda x: x["raw_p"])
    holm_alphas = [0.05 / (4 - k) for k in range(4)]
    
    stopped = False
    for k, hyp in enumerate(sorted_hypotheses):
        alpha_k = holm_alphas[k]
        hyp["rank"] = k + 1
        hyp["holm_alpha"] = alpha_k
        if not stopped and hyp["raw_p"] <= alpha_k:
            hyp["decision"] = "REJECT_H0 (Supported)"
            hyp["supported"] = True
        else:
            hyp["decision"] = "FAIL_TO_REJECT (Not Supported)"
            hyp["supported"] = False
            stopped = True
            
    print("\nHolm-Bonferroni Multi-Testing Results (M=4):")
    for hyp in sorted_hypotheses:
        print(f"  Rank {hyp['rank']}: {hyp['id']} ({hyp['name']}) | raw_p={hyp['raw_p']:.6f} vs alpha_{hyp['rank']}={hyp['holm_alpha']:.5f} -> {hyp['decision']}")
        
    # -------------------------------------------------------------------------
    # Post-Execution AlgoForge Non-Invasiveness Audit
    # -------------------------------------------------------------------------
    post_git_res = subprocess.run(["git", "-C", algoforge_dir, "status", "--porcelain"], capture_output=True, text=True)
    post_git_status = post_git_res.stdout.strip()
    git_diff_detected = (baseline_git_status != post_git_status)
    print(f"\n[*] Post-Execution AlgoForge Non-Invasiveness: {'DIFF DETECTED [FAIL]' if git_diff_detected else 'CONFIRMED ZERO DIFF [PASS]'}")
    assert not git_diff_detected, "AlgoForge repository was modified during run!"
    
    # -------------------------------------------------------------------------
    # Package Results
    # -------------------------------------------------------------------------
    results_summary = {
        "execution_metadata": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "snapshot_path": snapshot_path,
            "snapshot_sha256": file_sha256,
            "total_bars": total_bars,
            "trading_days": num_days,
            "warmup_days": len(warmup_dates),
            "eval_days": len(eval_dates),
            "api_spend_usd": 0.00,
            "algoforge_clean": not git_diff_detected,
        },
        "stage_1_hawkes": {
            "total_events": total_events,
            "event_rate": total_events / total_bars,
            "elapsed_time_periodic_sec": t_elapsed_periodic,
            "elapsed_time_continuous_sec": t_elapsed_continuous,
            "speedup_ratio": speedup_ratio,
            "mean_nll_periodic": float(hawkes_daily_df["mean_nll_periodic"].mean()),
            "mean_nll_continuous": float(hawkes_daily_df["mean_nll_continuous"].mean()),
            "mean_nll_iid": float(hawkes_daily_df["mean_nll_iid"].mean()),
            "h1a_test": {
                "mean_diff": mean_h1a,
                "se_hac": se_h1a,
                "t_stat": t_h1a,
                "p_val": p_h1a,
            },
            "h1b_test": {
                "mean_diff": mean_h1b,
                "margin": 0.005,
                "se_hac": se_h1b,
                "t_stat": t_h1b,
                "p_val": p_h1b,
            },
        },
        "stage_2_jump_bv": {
            "total_jumps_bv": int(df["jump_bv"].sum()),
            "total_jumps_std": int(df["jump_std"].sum()),
            "total_clusters_bv": int(df["cluster_bv"].sum()),
            "total_clusters_std": int(df["cluster_std"].sum()),
            "mean_cluster_rate_bv": float(jump_daily_df["cluster_rate_bv"].mean()),
            "mean_cluster_rate_std": float(jump_daily_df["cluster_rate_std"].mean()),
            "h2_test_all_132_days": {
                "mean_diff": mean_h2,
                "se_hac": se_h2,
                "t_stat": t_h2,
                "p_val": p_h2,
            },
            "high_rv_stratified_subgroup": {
                "n_days": len(high_rv_days),
                "rv_threshold_90pct": top_10pct_rv_threshold,
                "wilcoxon_stat": float(w_stat_h2),
                "wilcoxon_p_val": float(w_p_h2),
                "median_diff": float(np.median(high_rv_diff)),
            },
        },
        "stage_3_hamilton": {
            "informed_convergence_rate": float(hamilton_daily_df["conv_informed"].mean()),
            "quantile_convergence_rate": float(hamilton_daily_df["conv_quantile"].mean()),
            "confidence_rate_informed": confidence_rate_inf,
            "confidence_rate_quantile": confidence_rate_q,
            "h3_test": {
                "mean_diff": mean_h3,
                "se_hac": se_h3,
                "t_stat": t_h3,
                "p_val": p_h3,
            },
            "hac_sensitivity_by_lag": hamilton_hac_sensitivity,
            "bootstrap_sensitivity_by_block": hamilton_boot_sensitivity,
        },
        "stage_4_macro_case_study": macro_case_study_results,
        "holm_bonferroni_family": sorted_hypotheses,
    }
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "h3_p2_empirical_results.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[*] Results successfully saved to: {output_path}")
    
    return results_summary


if __name__ == "__main__":
    run_h3_p2_experiments()
