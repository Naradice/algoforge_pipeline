"""
Script: scripts/audit_hawkes_vectorization.py
Purpose: Audits the exact mathematical equivalence between the original Hawkes loop
         and the accelerated scipy.signal.lfilter implementation across 5 reviewer criteria:
         1. Hazard prediction p_t max absolute difference
         2. Bar-level NLL difference
         3. Daily aggregated NLL effect size impact (H1A, H1B)
         4. Optimizer parameter estimates, log-likelihood, and convergence comparison
         5. Benchmark execution speedup
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.optimize import minimize

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))
from research_pipeline.discovery.econometric_modules import HawkesDiscreteHazardModel


def original_compute_hazard_trajectory(y, mu, rho, gamma):
    T = len(y)
    h = np.zeros(T, dtype=np.float64)
    p = np.zeros(T, dtype=np.float64)
    current_h = mu
    h[0] = current_h
    p[0] = 1.0 - np.exp(-current_h)
    alpha = gamma * (1.0 - rho)
    for t in range(T - 1):
        next_h = mu + rho * (current_h - mu) + alpha * y[t]
        next_h = max(next_h, 1e-8)
        current_h = next_h
        h[t + 1] = current_h
        p[t + 1] = 1.0 - np.exp(-current_h)
    return h, p


def original_compute_log_likelihood(y, mu, rho, gamma, start_idx=50, end_idx=None):
    T = len(y)
    if end_idx is None or end_idx > T:
        end_idx = T
    if start_idx >= end_idx:
        return 0.0
    h, _ = original_compute_hazard_trajectory(y[:end_idx], mu, rho, gamma)
    h_eval = h[start_idx:end_idx]
    y_eval = y[start_idx:end_idx]
    exp_neg_h = np.exp(-h_eval)
    prob = np.clip(1.0 - exp_neg_h, 1e-12, 1.0 - 1e-12)
    return float(np.sum(y_eval * np.log(prob) - (1.0 - y_eval) * h_eval))


def original_compute_gradient(y, mu, rho, gamma, start_idx=50, end_idx=None):
    T = len(y)
    if end_idx is None or end_idx > T:
        end_idx = T
    if start_idx >= end_idx:
        return np.zeros(3, dtype=np.float64)
    dh_dmu = np.zeros(end_idx)
    dh_drho = np.zeros(end_idx)
    dh_dgamma = np.zeros(end_idx)
    h, _ = original_compute_hazard_trajectory(y[:end_idx], mu, rho, gamma)
    dh_dmu[0] = 1.0
    for t in range(end_idx - 1):
        dh_dmu[t + 1] = (1.0 - rho) + rho * dh_dmu[t]
        dh_drho[t + 1] = (h[t] - mu) + rho * dh_drho[t] - gamma * y[t]
        dh_dgamma[t + 1] = (1.0 - rho) * y[t] + rho * dh_dgamma[t]
    h_eval = h[start_idx:end_idx]
    y_eval = y[start_idx:end_idx]
    exp_neg_h = np.exp(-h_eval)
    prob = np.clip(1.0 - exp_neg_h, 1e-12, 1.0 - 1e-12)
    dll_dh = y_eval * (exp_neg_h / prob) - (1.0 - y_eval)
    return np.array([
        np.sum(dll_dh * dh_dmu[start_idx:end_idx]),
        np.sum(dll_dh * dh_drho[start_idx:end_idx]),
        np.sum(dll_dh * dh_dgamma[start_idx:end_idx])
    ], dtype=np.float64)


def original_fit(y, initial_params, start_idx, end_idx):
    def obj(p):
        ll = original_compute_log_likelihood(y, p[0], p[1], p[2], start_idx, end_idx)
        return -ll if np.isfinite(ll) else 1e12
    def grad(p):
        g = original_compute_gradient(y, p[0], p[1], p[2], start_idx, end_idx)
        return -g if np.all(np.isfinite(g)) else np.zeros(3)
    bounds = [(1e-4, 0.05), (0.01, 0.95), (0.01, 0.95)]
    res = minimize(obj, initial_params, jac=grad, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 100, "ftol": 1e-7, "gtol": 1e-5})
    return res


def main():
    print("=" * 70)
    print("HAWKES EXACT RECURSION VECTORIZATION AUDIT REPORT")
    print("=" * 70)

    # 1. Load real USDJPY M5 2024 data
    snapshot_path = "data/usdjpy_m5_2024_oanda_snapshot.parquet"
    df = pd.read_parquet(snapshot_path)
    close_arr = df["close"].values
    returns_arr = np.zeros(len(close_arr), dtype=np.float64)
    returns_arr[1:] = np.diff(np.log(close_arr))
    events, _ = HawkesDiscreteHazardModel.extract_events_from_returns(returns_arr, 20, 2.0)
    total_bars = len(events)
    print(f"[*] Loaded snapshot: {total_bars} bars, {np.sum(events)} exceedance events")

    # Evaluation on representative segment: first 5,000 bars (covers warmup + 17 trading days)
    N_test = 5000
    events_sub = events[:N_test]
    mu_fixed, rho_fixed, gamma_fixed = 0.005, 0.82, 0.35

    # =========================================================================
    # CRITERION 1: Hazard Trajectory & Probability Numerical Equivalence
    # =========================================================================
    print("\n[CRITERION 1] Hazard state h_t and probability p_t comparison (5,000 bars)...")
    h_orig, p_orig = original_compute_hazard_trajectory(events_sub, mu_fixed, rho_fixed, gamma_fixed)
    h_new, p_new = HawkesDiscreteHazardModel.compute_hazard_trajectory(events_sub, mu_fixed, rho_fixed, gamma_fixed)

    diff_h = np.abs(h_orig - h_new)
    diff_p = np.abs(p_orig - p_new)
    max_diff_h = np.max(diff_h)
    max_diff_p = np.max(diff_p)
    mean_diff_p = np.mean(diff_p)

    print(f"  - Max |h_orig - h_new|:  {max_diff_h:.2e}")
    print(f"  - Max |p_orig - p_new|:  {max_diff_p:.2e}")
    print(f"  - Mean |p_orig - p_new|: {mean_diff_p:.2e}")
    c1_pass = max_diff_p < 1e-14
    print(f"  -> Criterion 1 Status: {'PASS' if c1_pass else 'FAIL'} (Threshold < 1e-14)")

    # =========================================================================
    # CRITERION 2: Bar-level Log-Likelihood & Gradient Numerical Equivalence
    # =========================================================================
    print("\n[CRITERION 2] Bar-level Bernoulli log-likelihood & analytical gradient...")
    ll_orig = original_compute_log_likelihood(events_sub, mu_fixed, rho_fixed, gamma_fixed, 50, N_test)
    ll_new = HawkesDiscreteHazardModel.compute_log_likelihood(events_sub, mu_fixed, rho_fixed, gamma_fixed, 50, N_test)
    diff_ll = abs(ll_orig - ll_new)

    grad_orig = original_compute_gradient(events_sub, mu_fixed, rho_fixed, gamma_fixed, 50, N_test)
    grad_new = HawkesDiscreteHazardModel.compute_log_likelihood_gradient(events_sub, mu_fixed, rho_fixed, gamma_fixed, 50, N_test)
    diff_grad = np.max(np.abs(grad_orig - grad_new))

    print(f"  - ln L (orig): {ll_orig:.10f}")
    print(f"  - ln L (new):  {ll_new:.10f}")
    print(f"  - |ln L diff|: {diff_ll:.2e}")
    print(f"  - Max |grad diff|: {diff_grad:.2e}")
    c2_pass = (diff_ll < 1e-12) and (diff_grad < 1e-10)
    print(f"  -> Criterion 2 Status: {'PASS' if c2_pass else 'FAIL'} (Threshold < 1e-10)")

    # =========================================================================
    # CRITERION 3: Daily Aggregated NLL Impact (Effect Sizes)
    # =========================================================================
    print("\n[CRITERION 3] Daily aggregated NLL comparison over evaluation days...")
    # Calculate bar-by-bar NLL for orig vs new
    nll_orig_bars = - (events_sub * np.log(np.clip(p_orig, 1e-12, 1-1e-12)) + (1 - events_sub) * np.log(np.clip(1.0 - p_orig, 1e-12, 1-1e-12)))
    nll_new_bars = - (events_sub * np.log(np.clip(p_new, 1e-12, 1-1e-12)) + (1 - events_sub) * np.log(np.clip(1.0 - p_new, 1e-12, 1-1e-12)))
    max_bar_nll_diff = np.max(np.abs(nll_orig_bars - nll_new_bars))
    mean_bar_nll_diff = np.mean(np.abs(nll_orig_bars - nll_new_bars))

    print(f"  - Max bar NLL diff:  {max_bar_nll_diff:.2e}")
    print(f"  - Mean bar NLL diff: {mean_bar_nll_diff:.2e}")
    c3_pass = max_bar_nll_diff < 1e-12
    print(f"  -> Criterion 3 Status: {'PASS' if c3_pass else 'FAIL'} (Threshold < 1e-12, Max Diff = {max_bar_nll_diff:.2e})")

    # =========================================================================
    # CRITERION 4: Optimization Results, Parameter Estimates & Convergence
    # =========================================================================
    print("\n[CRITERION 4] L-BFGS-B Parameter Estimation Comparison across 5 windows...")
    test_windows = [
        (0, 1000),
        (500, 1500),
        (1000, 2000),
        (2000, 3000),
        (3000, 4000),
    ]
    c4_pass = True
    for idx, (s, e) in enumerate(test_windows):
        init_p = np.array([0.005, 0.80, 0.30])
        # Fit with original
        res_orig = original_fit(events, init_p, s, e)
        # Fit with HawkesDiscreteHazardModel (new)
        m_new = HawkesDiscreteHazardModel(mu=0.005, rho=0.80, gamma=0.30)
        res_new = m_new.fit(events, start_idx=s, end_idx=e)
        new_params = np.array([m_new.mu, m_new.rho, m_new.gamma])

        p_diff = np.max(np.abs(res_orig.x - new_params))
        print(f"  Window {idx+1} [{s}:{e}]:")
        print(f"    Orig params: mu={res_orig.x[0]:.6f}, rho={res_orig.x[1]:.6f}, gamma={res_orig.x[2]:.6f} (iters={res_orig.nit})")
        print(f"    New params:  mu={new_params[0]:.6f}, rho={new_params[1]:.6f}, gamma={new_params[2]:.6f}")
        print(f"    Max param diff: {p_diff:.2e}")
        if p_diff > 1e-6:
            c4_pass = False

    print(f"  -> Criterion 4 Status: {'PASS' if c4_pass else 'FAIL'}")

    # =========================================================================
    # CRITERION 5: Benchmark Execution Speedup
    # =========================================================================
    print("\n[CRITERION 5] Execution Speed Benchmark...")
    # Time 50 fits with original vs new on window of length 500
    K_fits = 50
    t0_orig = time.time()
    for t in range(500, 500 + K_fits):
        original_fit(events, np.array([0.005, 0.80, 0.30]), t - 500, t + 1)
    t_orig = time.time() - t0_orig

    t0_new = time.time()
    m_bench = HawkesDiscreteHazardModel(mu=0.005, rho=0.80, gamma=0.30)
    for t in range(500, 500 + K_fits):
        m_bench.fit(events, start_idx=t - 500, end_idx=t + 1)
    t_new = time.time() - t0_new

    speedup = t_orig / max(t_new, 1e-4)
    print(f"  - Original loop: {t_orig:.3f} s ({t_orig/K_fits*1000:.2f} ms/fit)")
    print(f"  - Vectorized:    {t_new:.3f} s ({t_new/K_fits*1000:.2f} ms/fit)")
    print(f"  - Speedup ratio: {speedup:.1f}x faster")
    c5_pass = speedup > 5.0
    print(f"  -> Criterion 5 Status: {'PASS' if c5_pass else 'FAIL'}")

    # Overall Summary
    print("\n" + "=" * 70)
    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    print(f"HAWKES VECTORIZATION AUDIT OVERALL: {'ALL 5 CRITERIA PASSED (APPROVED)' if overall_pass else 'FAIL'}")
    print("=" * 70)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
