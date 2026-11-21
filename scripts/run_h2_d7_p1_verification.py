"""
Execution and Verification Script for H2-D7-P1
Location: scripts/run_h2_d7_p1_verification.py

Runs the complete 5-Group Verification Architecture approved in H2-D7-P1 v0.3:
- Groups A, B, C: Unit test suite execution via pytest
- Group D: Statistical Characterization & Identifiability (D-1 to D-7)
- Group E: Execution Latency & Performance Benchmarks (E-1, E-2)
Outputs complete quantitative results to data/h2_d7_p1_unit_verification_results.json.
"""

import sys
import os
import json
import time
import numpy as np
import pytest
from scipy.stats import norm
from scipy.optimize import approx_fprime

# Ensure research_pipeline/src is in sys.path
pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(pipeline_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from research_pipeline.discovery.econometric_modules import (
    HawkesDiscreteHazardModel,
    LocalJumpDetector,
    HamiltonLogRVRegimeModel,
)


import subprocess

def run_unit_tests() -> dict:
    print("=" * 70, flush=True)
    print("STEP 1: Running Groups A, B, C Unit Tests via pytest...", flush=True)
    print("=" * 70, flush=True)

    test_file = os.path.join(pipeline_root, "tests", "unit", "test_h2_d7_p1_modules.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = src_path

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:asyncio",
            "-p",
            "no:anyio",
            "-v",
            test_file,
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    print(res.stdout, flush=True)
    if res.stderr:
        print("STDERR:\n" + res.stderr, flush=True)

    passed = (res.returncode == 0)
    print(f"Pytest return code: {res.returncode} (All Pass: {passed})", flush=True)
    return {
        "status": "PASS" if passed else "FAIL",
        "return_code": int(res.returncode),
        "test_file": test_file,
        "stdout": res.stdout,
    }


def run_group_d_statistical_characterization() -> dict:
    print("\n" + "=" * 70)
    print("STEP 2: Running Group D Statistical Characterization & Identifiability...")
    print("=" * 70)

    results_d = {}

    # -------------------------------------------------------------------------
    # D-1: Hawkes Analytical vs Numerical Gradient
    # -------------------------------------------------------------------------
    print("--- Running D-1: Hawkes Gradient Consistency ---")
    np.random.seed(42)
    y_synth, _, _ = HawkesDiscreteHazardModel.generate_synthetic_events(500, mu=0.01, rho=0.8, gamma=0.3, seed=42)
    test_params = [
        (0.01, 0.80, 0.30),
        (0.005, 0.70, 0.20),
        (0.02, 0.90, 0.40),
    ]
    grad_errors = []
    for m, r, g in test_params:
        analytical_grad = HawkesDiscreteHazardModel.compute_log_likelihood_gradient(
            y_synth, m, r, g, start_idx=20, end_idx=500
        )
        
        def obj(p):
            return HawkesDiscreteHazardModel.compute_log_likelihood(
                y_synth, p[0], p[1], p[2], start_idx=20, end_idx=500
            )

        eps = 1e-6
        numerical_grad = approx_fprime(np.array([m, r, g]), obj, eps)
        rel_diffs = np.abs(analytical_grad - numerical_grad) / (np.abs(numerical_grad) + 1e-8)
        grad_errors.append({
            "params": [m, r, g],
            "max_rel_error": float(np.max(rel_diffs)),
            "analytical_grad": analytical_grad.tolist(),
            "numerical_grad": numerical_grad.tolist(),
        })

    max_grad_err = max(e["max_rel_error"] for e in grad_errors)
    median_grad_err = float(np.median([e["max_rel_error"] for e in grad_errors]))
    print(f"D-1 Hawkes Max Relative Gradient Error: {max_grad_err:.6e}, Median: {median_grad_err:.6e}", flush=True)
    results_d["D1_hawkes_gradient"] = {
        "max_relative_error": max_grad_err,
        "median_relative_error": median_grad_err,
        "details": grad_errors,
        "status": "PASS" if max_grad_err < 1e-3 else "FAIL",
    }

    # -------------------------------------------------------------------------
    # D-2: Hawkes Hazard Model Truth Recovery across N in {500, 5000, 50000}
    # -------------------------------------------------------------------------
    print("--- Running D-2: Hawkes Hazard Model Truth Recovery Scaling ---")
    true_mu, true_rho, true_gamma = 0.01, 0.80, 0.30
    true_theta = np.array([true_mu, true_rho, true_gamma])
    sample_sizes = [500, 5000, 50000]
    seeds = [42, 43, 44, 45, 46]

    d2_results = {}
    for N in sample_sizes:
        errors = []
        event_counts = []
        cond_numbers = []
        for s in seeds:
            y, _, _ = HawkesDiscreteHazardModel.generate_synthetic_events(
                N, mu=true_mu, rho=true_rho, gamma=true_gamma, seed=s
            )
            event_counts.append(int(np.sum(y)))
            model = HawkesDiscreteHazardModel(mu=0.015, rho=0.75, gamma=0.35)
            # fit on events directly
            model.fit(y, start_idx=10, end_idx=N)
            est_theta = np.array([model.mu, model.rho, model.gamma])
            err = np.linalg.norm(est_theta - true_theta)
            errors.append(float(err))

            # Numerical Hessian condition number at optimum
            def obj(p):
                return -HawkesDiscreteHazardModel.compute_log_likelihood(
                    y, p[0], p[1], p[2], start_idx=10, end_idx=N
                )
            # finite diff hessian
            eps = 1e-4
            p0 = est_theta
            grad0 = approx_fprime(p0, obj, eps)
            hess = np.zeros((3, 3))
            for i in range(3):
                p_plus = p0.copy()
                p_plus[i] += eps
                grad_plus = approx_fprime(p_plus, obj, eps)
                hess[i] = (grad_plus - grad0) / eps
            hess = 0.5 * (hess + hess.T)
            try:
                eigvals = np.linalg.eigvalsh(hess)
                eigvals = eigvals[eigvals > 1e-12]
                cond = float(np.max(eigvals) / np.min(eigvals)) if len(eigvals) > 1 else np.nan
            except Exception:
                cond = np.nan
            cond_numbers.append(cond)

        d2_results[str(N)] = {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors, ddof=1)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "mean_events": float(np.mean(event_counts)),
            "mean_hessian_condition": float(np.nanmean(cond_numbers)),
        }
        print(f"  N={N:5d}: Mean Error={np.mean(errors):.4f}, Mean Events={np.mean(event_counts):.1f}, Cond={np.nanmean(cond_numbers):.1f}")

    results_d["D2_hawkes_parameter_recovery"] = d2_results

    # -------------------------------------------------------------------------
    # D-3: BV Relative Bias on 100 Gaussian Paths
    # -------------------------------------------------------------------------
    print("--- Running D-3: BV Estimator Relative Bias on 100 Gaussian Paths ---")
    num_paths = 100
    path_len = 1000
    true_sigma = 0.01
    true_var = true_sigma ** 2
    detector = LocalJumpDetector(window_k=72)
    path_biases = []

    for s in range(num_paths):
        np.random.seed(1000 + s)
        r = np.random.normal(0, true_sigma, path_len)
        bv_sq = detector.compute_bipower_variation_squared(r)
        # evaluate for t >= 72
        valid_bv = bv_sq[72:]
        mean_est_var = np.mean(valid_bv)
        rel_bias = (mean_est_var / true_var) - 1.0
        path_biases.append(rel_bias)

    mean_bias = float(np.mean(path_biases))
    std_err_bias = float(np.std(path_biases, ddof=1) / np.sqrt(num_paths))
    print(f"D-3 BV Mean Relative Bias: {mean_bias * 100:.3f}% +/- {std_err_bias * 100:.3f}%")
    results_d["D3_bv_relative_bias"] = {
        "mean_relative_bias": mean_bias,
        "standard_error": std_err_bias,
        "num_paths": num_paths,
    }

    # -------------------------------------------------------------------------
    # D-4: Local Jump False Positive Rate (Size)
    # -------------------------------------------------------------------------
    print("--- Running D-4: Local Jump False Positive Rate on Gaussian Paths ---")
    fpr_paths = 10
    fpr_len = 10000
    total_evals = 0
    total_fps = 0

    for s in range(fpr_paths):
        np.random.seed(2000 + s)
        r = np.random.normal(0, 0.01, fpr_len)
        res = detector.detect_jumps(r)
        eval_slice = res["l_statistic"][72:]
        total_evals += len(eval_slice)
        total_fps += int(np.sum(eval_slice > 3.5))

    empirical_fpr = total_fps / total_evals
    theoretical_asymptotic_fpr = 2.0 * (1.0 - norm.cdf(3.5))  # ~0.00046526 (0.0465%)
    print(f"D-4 Jump Empirical FPR: {empirical_fpr * 100:.4f}% ({total_fps}/{total_evals}), Asymptotic Ref: {theoretical_asymptotic_fpr * 100:.4f}%")
    results_d["D4_jump_false_positive_rate"] = {
        "empirical_fpr": empirical_fpr,
        "theoretical_asymptotic_fpr": theoretical_asymptotic_fpr,
        "total_evaluations": total_evals,
        "false_positive_count": total_fps,
    }

    # -------------------------------------------------------------------------
    # D-5: Local Jump Detection Power (Additive Jumps)
    # -------------------------------------------------------------------------
    print("--- Running D-5: Local Jump Detection Power (Additive Jumps) ---")
    magnitudes = [3.0, 5.0, 8.0]
    num_jump_trials = 100
    power_results = {}

    for mag in magnitudes:
        detected_pos = 0
        detected_neg = 0
        for trial in range(num_jump_trials):
            np.random.seed(3000 + int(mag * 100) + trial)
            r = np.random.normal(0, 0.01, 150)
            sigma_bar = 0.01
            # Positive jump at bar 100
            r_pos = r.copy()
            r_pos[100] += mag * sigma_bar
            res_pos = detector.detect_jumps(r_pos)
            if res_pos["jump_signals"][100] == 1:
                detected_pos += 1

            # Negative jump at bar 100
            r_neg = r.copy()
            r_neg[100] -= mag * sigma_bar
            res_neg = detector.detect_jumps(r_neg)
            if res_neg["jump_signals"][100] == -1:
                detected_neg += 1

        power_pos = detected_pos / num_jump_trials
        power_neg = detected_neg / num_jump_trials
        power_avg = 0.5 * (power_pos + power_neg)
        power_results[f"{mag}sigma"] = {
            "power_positive": power_pos,
            "power_negative": power_neg,
            "power_average": power_avg,
        }
        print(f"  Jump Magnitude {mag:+.1f}sigma: Power = {power_avg * 100:.1f}% (Pos: {power_pos * 100:.1f}%, Neg: {power_neg * 100:.1f}%)")

    results_d["D5_jump_detection_power"] = power_results

    # -------------------------------------------------------------------------
    # D-6: Post-Jump BV Distortion (Contamination over subsequent k bars)
    # -------------------------------------------------------------------------
    print("--- Running D-6: Post-Jump BV Distortion ---")
    k_lags = [1, 2, 5, 10]
    num_d6_trials = 100
    d6_biases = {k: [] for k in k_lags}

    for trial in range(num_d6_trials):
        np.random.seed(4000 + trial)
        r = np.random.normal(0, 0.01, 120)
        # Inject +8 sigma jump at bar 80
        r[80] += 8.0 * 0.01
        bv_sq = detector.compute_bipower_variation_squared(r)
        true_var = 0.01 ** 2
        for k in k_lags:
            idx = 80 + k
            if idx < len(bv_sq):
                rel_bias = (bv_sq[idx] / true_var) - 1.0
                d6_biases[k].append(rel_bias)

    d6_summary = {}
    for k in k_lags:
        mean_k_bias = float(np.mean(d6_biases[k]))
        d6_summary[f"k_{k}"] = {
            "mean_relative_bias": mean_k_bias,
            "std_relative_bias": float(np.std(d6_biases[k], ddof=1)),
        }
        print(f"  Lag k={k:2d}: Mean BV Relative Bias = {mean_k_bias * 100:+.2f}%")

    results_d["D6_post_jump_bv_distortion"] = d6_summary

    # -------------------------------------------------------------------------
    # D-7: Hamilton Regime Identifiability & State Recall
    # -------------------------------------------------------------------------
    print("--- Running D-7: Hamilton Regime Identifiability across T_days in {60, 250, 1000} ---")
    t_days_list = [60, 250, 1000]
    seeds_hamilton = [42, 43, 44, 45, 46]
    true_mu = (-9.0, -7.0)
    true_sigma = (0.5, 1.0)
    true_p00, true_p11 = 0.90, 0.80

    d7_results = {}
    for T_d in t_days_list:
        mu_errors = []
        p_errors = []
        accuracies_filt = []
        accuracies_smooth = []
        recall_0_list = []
        recall_1_list = []
        starvation_count = 0

        for s in seeds_hamilton:
            y, true_s = HamiltonLogRVRegimeModel.generate_synthetic_regime_series(
                T_d, mu=true_mu, sigma=true_sigma, p00=true_p00, p11=true_p11, seed=s
            )
            # check regime starvation in sample
            prop_state_1 = np.mean(true_s == 1)
            if prop_state_1 < 0.10 or prop_state_1 > 0.90:
                starvation_count += 1

            model = HamiltonLogRVRegimeModel()
            model.fit_em(y)

            # Parameter error
            err_mu = float(np.linalg.norm(model.mu - np.array(true_mu)))
            err_p = float(np.linalg.norm(np.array([model.P[0, 0], model.P[1, 1]]) - np.array([true_p00, true_p11])))
            mu_errors.append(err_mu)
            p_errors.append(err_p)

            # Filtering & Smoothing state classification
            xi_pred, xi_filt, _ = model.forward_filter(y)
            xi_smooth, _ = model.backward_smoother(xi_pred, xi_filt)

            pred_s_filt = np.argmax(xi_filt, axis=1)
            pred_s_smooth = np.argmax(xi_smooth, axis=1)

            acc_f = np.mean(pred_s_filt == true_s)
            acc_s = np.mean(pred_s_smooth == true_s)
            accuracies_filt.append(float(acc_f))
            accuracies_smooth.append(float(acc_s))

            # State-specific recall
            mask_0 = (true_s == 0)
            mask_1 = (true_s == 1)
            rec_0 = np.mean(pred_s_smooth[mask_0] == 0) if np.sum(mask_0) > 0 else 0.0
            rec_1 = np.mean(pred_s_smooth[mask_1] == 1) if np.sum(mask_1) > 0 else 0.0
            recall_0_list.append(float(rec_0))
            recall_1_list.append(float(rec_1))

        d7_results[str(T_d)] = {
            "mean_mu_error": float(np.mean(mu_errors)),
            "mean_p_error": float(np.mean(p_errors)),
            "mean_filtered_accuracy": float(np.mean(accuracies_filt)),
            "mean_smoothed_accuracy": float(np.mean(accuracies_smooth)),
            "mean_recall_0": float(np.mean(recall_0_list)),
            "mean_recall_1": float(np.mean(recall_1_list)),
            "starvation_frequency": float(starvation_count / len(seeds_hamilton)),
        }
        print(f"  T={T_d:4d} days: Mu Err={np.mean(mu_errors):.3f}, P Err={np.mean(p_errors):.3f}, Filt Acc={np.mean(accuracies_filt)*100:.1f}%, Rec0={np.mean(recall_0_list)*100:.1f}%, Rec1={np.mean(recall_1_list)*100:.1f}%, Starvation={starvation_count}/{len(seeds_hamilton)}")

    results_d["D7_hamilton_identifiability"] = d7_results
    return results_d


def run_group_e_performance_benchmarks() -> dict:
    print("\n" + "=" * 70)
    print("STEP 3: Running Group E Latency & Performance Benchmarks...")
    print("=" * 70)

    results_e = {}

    # -------------------------------------------------------------------------
    # E-1: Hamilton 60-Day EM Latency (100 runs)
    # -------------------------------------------------------------------------
    print("--- Running E-1: Hamilton 60-day EM Latency Benchmark (100 runs) ---")
    np.random.seed(999)
    y_60 = np.random.normal(-8.0, 0.7, 60)
    em_times_ms = []

    for _ in range(100):
        model = HamiltonLogRVRegimeModel()
        t0 = time.perf_counter()
        model.fit_em(y_60)
        t1 = time.perf_counter()
        em_times_ms.append((t1 - t0) * 1000.0)

    e1_mean = float(np.mean(em_times_ms))
    e1_p95 = float(np.percentile(em_times_ms, 95))
    e1_p99 = float(np.percentile(em_times_ms, 99))
    e1_max = float(np.max(em_times_ms))

    print(f"E-1 Hamilton 60-day EM: Mean={e1_mean:.2f} ms, p95={e1_p95:.2f} ms, p99={e1_p99:.2f} ms, Max={e1_max:.2f} ms (Target: < 100 ms)")
    results_e["E1_hamilton_latency_ms"] = {
        "mean": e1_mean,
        "p95": e1_p95,
        "p99": e1_p99,
        "max": e1_max,
        "target_ms": 100.0,
        "target_met": bool(e1_mean < 100.0),
    }

    # -------------------------------------------------------------------------
    # E-2: Hawkes Single-Step Recursive State Update (10,000 steps)
    # -------------------------------------------------------------------------
    print("--- Running E-2: Hawkes Single-Step State Update Latency (10,000 steps) ---")
    model = HawkesDiscreteHazardModel()
    curr_h = model.mu
    alpha = model.gamma * (1.0 - model.rho)
    update_times_us = []

    # Run 10,000 recursive steps
    for step in range(10000):
        y_val = 1 if (step % 50 == 0) else 0
        r_val = 0.001 if y_val == 1 else 0.0
        t0 = time.perf_counter()
        curr_h = model.mu + model.rho * (curr_h - model.mu) + alpha * y_val
        p_val = 1.0 - np.exp(-curr_h)
        sig = 1 if (p_val > 0.5 and r_val > 0) else 0
        t1 = time.perf_counter()
        update_times_us.append((t1 - t0) * 1e6)

    e2_mean_us = float(np.mean(update_times_us))
    e2_p99_us = float(np.percentile(update_times_us, 99))
    e2_mean_ms = e2_mean_us / 1000.0
    e2_p99_ms = e2_p99_us / 1000.0

    print(f"E-2 Hawkes Single-Step Update: Mean={e2_mean_us:.2f} us ({e2_mean_ms:.4f} ms), p99={e2_p99_us:.2f} us ({e2_p99_ms:.4f} ms) (Target: < 1.0 ms)")
    results_e["E2_hawkes_single_step_latency"] = {
        "mean_us": e2_mean_us,
        "p99_us": e2_p99_us,
        "mean_ms": e2_mean_ms,
        "p99_ms": e2_p99_ms,
        "target_ms": 1.0,
        "target_met": bool(e2_mean_ms < 1.0),
    }

    return results_e


def main():
    print("Starting H2-D7-P1 Unit Verification Suite...")
    start_total = time.time()

    # Step 1: Groups A, B, C Unit Tests
    unit_test_res = run_unit_tests()

    # Step 2: Group D Statistical Characterization
    group_d_res = run_group_d_statistical_characterization()

    # Step 3: Group E Latency Benchmarks
    group_e_res = run_group_e_performance_benchmarks()

    elapsed_total = time.time() - start_total

    final_payload = {
        "metadata": {
            "task": "H2-D7-P1 Unit Verification Protocol",
            "version": "v0.3-APPROVED",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_elapsed_seconds": elapsed_total,
            "cost_usd": 0.0,
        },
        "group_abc_unit_tests": unit_test_res,
        "group_d_statistical_characterization": group_d_res,
        "group_e_performance_benchmarks": group_e_res,
    }

    # Save to data directory
    output_path = os.path.join(pipeline_root, "data", "h2_d7_p1_unit_verification_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print(f"\nAll verification tasks completed successfully in {elapsed_total:.2f}s.")
    print(f"Quantitative results saved to: {output_path}")


if __name__ == "__main__":
    main()
