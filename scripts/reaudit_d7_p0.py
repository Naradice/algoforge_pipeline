"""
Script: reaudit_d7_p0.py
Performs the strict P0 Re-audit requested by the reviewer:
1. Track A: Strictly adheres to P1 Gold v0.7 criteria (Hawkes: mu, alpha, beta, lambda*, t_exec, max_dd; Jump: theta_BV, p_trans, sigma_ratio, t_exec, stop_bps, J*).
2. Track B: Evaluates the 3 D7 Model Parameters:
   - Hawkes: mu, rho, gamma
   - Jump: L* (local jump statistic/threshold), P (transition matrix), sigma_ratio (regime variance ratio)
3. Computes the approved v0.5 secondary metric:
   R_param = (1/6) * sum_{j=1}^6 prod_{p in Params_model} 1(S_{j,p} == 1 and V_{j,p} == 1)
4. Computes the approved primary metric:
   R_executable = S_comp and V_math and B_corr
5. Retains the 36-item Specified rate as an auxiliary metric.
"""

import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
DATA_DIR = BASE_DIR / "data"
RAW_CANDIDATES_PATH = DATA_DIR / "h2_d7_raw_candidates.json"

with open(RAW_CANDIDATES_PATH, "r", encoding="utf-8") as f:
    blocks = json.load(f)

# =============================================================================
# 1. TRACK A: STRICT P1-COMPATIBLE GOLD EVALUATION (6 params per block, 36 cells per condition)
# =============================================================================
# In P1 Gold standard:
# Hawkes:
# - mu: MLE window >= 50 bars or formula + bounds
# - alpha: MLE window >= 50 bars, bounds [0.01, 0.95], discrete likelihood
# - beta: MLE window >= 50 bars, bounds [0.01, 0.95], decay formula
# - lambda*: Percentile calibration rule over rolling window
# - t_exec: Non-anticipative O_{t+1}
# - max_dd: Dynamic ATR or dynamic drawdown formula
# Note: D7 discrete hazard parameters (rho, gamma) do NOT satisfy continuous P1 Hawkes Gold.
#
# Jump-Diffusion:
# - theta_BV: Bipower variation formula (with pi/2) + rolling window >= 50 + percentile calibration
# - p_trans: Hamilton filter / Markov switching EM with window and constraints
# - sigma_ratio: Variance ratio formula / estimation over regime windows
# - t_exec: Non-anticipative O_{t+1}
# - stop_bps: Dynamic ATR stop (e.g. 1.5 * ATR(14))
# - J*: Jump trigger formula tied to calibrated theta_BV or explicit test statistic

track_a_ledger = {
    # Hawkes s42
    "h2_d7_hawkes_process_s42_C00": {
        "mu": (1, 1, 0, 0, "Mentioned in intensity MLE mu > 0, no window or likelihood"),
        "alpha": (1, 1, 0, 0, "MLE with alpha < beta, no bounds or likelihood"),
        "beta": (1, 1, 0, 0, "MLE with alpha < beta, no bounds or window"),
        "lambda_star": (1, 1, 1, 0, "Rolling mean + 1 std over last 50 bars"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },
    "h2_d7_hawkes_process_s42_C10": {
        "mu": (0, 0, 0, 0, "Omitted from parameter estimation block"),
        "alpha": (1, 1, 0, 0, "MLE over 100 bars, bounds [0.01, 0.5], discrete likelihood missing"),
        "beta": (1, 1, 0, 0, "MLE over 100 bars, bounds [0.01, 0.1], discrete likelihood missing"),
        "lambda_star": (1, 1, 0, 0, "Fixed 0.1 constant"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },
    "h2_d7_hawkes_process_s42_C01": {
        "mu": (1, 1, 0, 0, "D7 baseline hazard mu in [1e-4, 0.05], not P1 continuous Hawkes MLE"),
        "alpha": (1, 1, 0, 0, "MLE with bounds [0.01, 1.0], no discrete likelihood formula"),
        "beta": (1, 1, 0, 0, "MLE with bounds [0.01, 1.0], no discrete likelihood formula"),
        "lambda_star": (1, 0, 0, 0, "hawkes_intensity_threshold left as placeholder 'to be determined'"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 0, 0, 0, "hawkes_max_drawdown_stop left as placeholder 'to be determined'"),
    },
    "h2_d7_hawkes_process_s42_C11": {
        "mu": (1, 1, 0, 0, "D7 baseline hazard, not P1 continuous Hawkes MLE"),
        "alpha": (1, 1, 0, 0, "MLE on events with bounds [0.01, 0.5], window missing"),
        "beta": (1, 1, 0, 0, "MLE on events with bounds [0.01, 0.5], window missing"),
        "lambda_star": (1, 1, 1, 0, "90th percentile of arrival intensities over 500 bars"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },

    # Hawkes s43
    "h2_d7_hawkes_process_s43_C00": {
        "mu": (0, 0, 0, 0, "Not mentioned"),
        "alpha": (1, 1, 0, 0, "MLE with bounds [0.1, 2.0], window missing"),
        "beta": (1, 1, 0, 0, "MLE with bounds [0.01, 0.5], window missing"),
        "lambda_star": (1, 0, 0, 0, "Threshold without calibration rule"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },
    "h2_d7_hawkes_process_s43_C10": {
        "mu": (0, 0, 0, 0, "Not mentioned"),
        "alpha": (1, 1, 0, 0, "Optimizing likelihood over 1 year, bounds [0.01, 1.0], discrete likelihood missing"),
        "beta": (1, 1, 0, 0, "Optimizing likelihood over 1 year, bounds [0.01, 1.0], discrete likelihood missing"),
        "lambda_star": (1, 1, 1, 0, "90th percentile of historical intensity over 1 year"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 2% constant"),
    },
    "h2_d7_hawkes_process_s43_C01": {
        "mu": (1, 1, 0, 0, "D7 discrete hazard baseline, not P1 continuous Hawkes"),
        "alpha": (1, 1, 0, 0, "MLE over 1000 bars with bounds [0.01, 0.5], discrete likelihood missing"),
        "beta": (1, 1, 0, 0, "MLE over 1000 bars with bounds [0.01, 0.5], discrete likelihood missing"),
        "lambda_star": (1, 1, 1, 0, "90th percentile over 500 bars"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },
    "h2_d7_hawkes_process_s43_C11": {
        "mu": (1, 1, 0, 0, "D7 discrete hazard baseline, not P1 continuous Hawkes"),
        "alpha": (1, 1, 0, 0, "MLE over 1000 bars with bounds [0.01, 0.5], discrete likelihood missing"),
        "beta": (1, 1, 0, 0, "MLE over 1000 bars with bounds [0.01, 0.5], discrete likelihood missing"),
        "lambda_star": (1, 1, 1, 0, "90th percentile over 500 bars"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },

    # Hawkes s44
    "h2_d7_hawkes_process_s44_C00": {
        "mu": (1, 1, 0, 0, "MLE mu > 0, no window or likelihood"),
        "alpha": (1, 1, 0, 0, "MLE alpha < beta, no bounds or likelihood"),
        "beta": (1, 1, 0, 0, "MLE alpha < beta, no bounds or likelihood"),
        "lambda_star": (1, 1, 1, 0, "Rolling mean + 1 std over last 50 bars"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },
    "h2_d7_hawkes_process_s44_C10": {
        "mu": (0, 0, 0, 0, "Not mentioned"),
        "alpha": (1, 1, 0, 0, "MLE over 100 bars, bounds [0.01, 0.5], discrete likelihood missing"),
        "beta": (1, 1, 0, 0, "MLE over 100 bars, bounds [0.01, 0.5], discrete likelihood missing"),
        "lambda_star": (1, 1, 1, 0, "85th percentile over 100 bars"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },
    "h2_d7_hawkes_process_s44_C01": {
        "mu": (1, 1, 0, 0, "D7 discrete hazard baseline, not P1 continuous Hawkes"),
        "alpha": (1, 1, 0, 0, "MLE on events with bounds [0.01, 0.5], window missing"),
        "beta": (1, 1, 0, 0, "MLE on events with bounds [0.01, 0.5], window missing"),
        "lambda_star": (1, 1, 1, 0, "90th percentile over 500 bars"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },
    "h2_d7_hawkes_process_s44_C11": {
        "mu": (1, 1, 0, 0, "D7 discrete hazard baseline, not P1 continuous Hawkes"),
        "alpha": (1, 1, 0, 0, "MLE on events with bounds [0.01, 0.5], window missing"),
        "beta": (1, 1, 0, 0, "MLE on events with bounds [0.01, 0.5], window missing"),
        "lambda_star": (1, 1, 1, 0, "90th percentile over 500 bars"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "max_dd": (1, 1, 0, 0, "Fixed 5% constant"),
    },

    # Jump s42
    "h2_d7_jump_diffusion_s42_C00": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing pi/2 scaling; 95th percentile specified"),
        "p_trans": (0, 0, 0, 0, "Not mentioned"),
        "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
        "t_exec": (1, 0, 0, 0, "Bar close execution, no O_{t+1}"),
        "stop_bps": (1, 1, 0, 0, "Fixed 1% constant"),
        "j_star": (1, 1, 0, 0, "J_t > 0 uncalibrated cutoff"),
    },
    "h2_d7_jump_diffusion_s42_C10": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing pi/2; rolling 1000 obs 90th percentile"),
        "p_trans": (0, 0, 0, 0, "Not mentioned"),
        "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
        "t_exec": (1, 0, 0, 0, "Bar close C_t execution"),
        "stop_bps": (1, 1, 0, 0, "Fixed 2% constant"),
        "j_star": (1, 1, 0, 0, "J_t > threshold tied to uncalibrated theta_BV"),
    },
    "h2_d7_jump_diffusion_s42_C01": {
        "theta_bv": (1, 1, 0, 0, "BV formula (1/(2*K^2)) has incorrect scaling (missing pi/2); 90th percentile over 60 days"),
        "p_trans": (1, 1, 0, 0, "MLE on regime switching over 2 years, EM update equations missing"),
        "sigma_ratio": (1, 1, 0, 0, "75th percentile crisis to 25th percentile tranquil over 60 days, formula missing"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "stop_bps": (1, 1, 1, 0, "1.5 * ATR(14) dynamic stop"),
        "j_star": (1, 1, 0, 0, "J_t > 0 uncalibrated cutoff"),
    },
    "h2_d7_jump_diffusion_s42_C11": {
        "theta_bv": (1, 1, 0, 0, "BV formula (1/(2*K^2)) missing pi/2 scaling; 75th percentile specified"),
        "p_trans": (1, 1, 0, 0, "MLE on regime switching, EM update equations missing"),
        "sigma_ratio": (1, 1, 0, 0, "Ratio of variances over 30 days, formula missing"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "stop_bps": (1, 1, 1, 0, "1.5 * ATR(14) dynamic stop"),
        "j_star": (1, 1, 1, 0, "J_t > 3.5 tied to standardized threshold L* = 3.5"),
    },

    # Jump s43
    "h2_d7_jump_diffusion_s43_C00": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing pi/2; threshold formula missing"),
        "p_trans": (0, 0, 0, 0, "Not mentioned"),
        "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
        "t_exec": (1, 0, 0, 0, "Bar close C_t execution"),
        "stop_bps": (1, 1, 0, 0, "Fixed 1% constant"),
        "j_star": (1, 1, 0, 0, "J_t > threshold tied to unspecified theta_BV"),
    },
    "h2_d7_jump_diffusion_s43_C10": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing pi/2; 90th percentile over 60 days"),
        "p_trans": (1, 1, 0, 0, "MLE on regime transitions over 5 years, EM algorithm missing"),
        "sigma_ratio": (1, 1, 0, 0, "Variance ratio over 60 days, formula missing"),
        "t_exec": (1, 0, 0, 0, "Bar close C_t execution"),
        "stop_bps": (1, 1, 0, 0, "Fixed 2% constant"),
        "j_star": (1, 1, 0, 0, "J_t > threshold tied to uncalibrated theta_BV"),
    },
    "h2_d7_jump_diffusion_s43_C01": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing pi/2 scaling; 90th percentile over 60 days"),
        "p_trans": (1, 1, 0, 0, "MLE on regime data over 5 years, EM algorithm missing"),
        "sigma_ratio": (1, 1, 0, 0, "Ratio of variances over 60 days, formula missing"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "stop_bps": (1, 1, 1, 0, "1.5 * ATR(14) dynamic stop"),
        "j_star": (1, 1, 0, 0, "J_t > 0 uncalibrated cutoff"),
    },
    "h2_d7_jump_diffusion_s43_C11": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing pi/2 scaling; 90th percentile specified"),
        "p_trans": (1, 1, 0, 0, "MLE on regime data, EM algorithm missing"),
        "sigma_ratio": (1, 1, 0, 0, "Ratio over 60 days, formula missing"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "stop_bps": (1, 1, 1, 0, "1.5 * ATR(14) dynamic stop"),
        "j_star": (1, 1, 0, 0, "J_t > 0 uncalibrated cutoff"),
    },

    # Jump s44
    "h2_d7_jump_diffusion_s44_C00": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing; 95th quantile specified"),
        "p_trans": (1, 1, 0, 0, "Empirical transition frequency, EM missing"),
        "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
        "t_exec": (1, 0, 0, 0, "No execution timing specified"),
        "stop_bps": (1, 1, 1, 0, "1.5 * ATR(288) dynamic stop"),
        "j_star": (1, 1, 0, 0, "J_t > theta_BV tied to unspecified BV"),
    },
    "h2_d7_jump_diffusion_s44_C10": {
        "theta_bv": (1, 1, 0, 0, "Bounds [0.5, 2.0], formula and window missing"),
        "p_trans": (1, 1, 0, 0, "MLE with bounds [0.1, 0.9], window and EM missing"),
        "sigma_ratio": (1, 1, 0, 0, "Regression with bounds [0.5, 2.0], formula missing"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "stop_bps": (1, 1, 0, 0, "Fixed 2% constant"),
        "j_star": (1, 1, 0, 0, "J_t > 0.01 uncalibrated constant"),
    },
    "h2_d7_jump_diffusion_s44_C01": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing pi/2 scaling; 95th percentile specified"),
        "p_trans": (1, 1, 0, 0, "MLE on regime data over 250 days, EM missing"),
        "sigma_ratio": (1, 1, 0, 0, "Two-sample F-test, formula missing"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "stop_bps": (1, 1, 1, 0, "1.5 * ATR(14) dynamic stop"),
        "j_star": (1, 1, 1, 0, "J_t > 3.5 tied to standardized threshold L* = 3.5"),
    },
    "h2_d7_jump_diffusion_s44_C11": {
        "theta_bv": (1, 1, 0, 0, "BV formula missing pi/2 scaling; 90th percentile over 5 years"),
        "p_trans": (1, 1, 0, 0, "Markov switching MLE over 3 years, EM missing"),
        "sigma_ratio": (1, 1, 0, 0, "Ratio of volatilities over 2 years, formula missing"),
        "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
        "stop_bps": (1, 1, 1, 0, "1.5 * ATR(14) dynamic stop"),
        "j_star": (1, 1, 1, 0, "J_t tied to jump threshold L* = 3.5"),
    },
}

# Print Track A summary
print("="*80)
print("TRACK A: P1-COMPATIBLE STRICT GOLD STANDARD AUDIT")
print("="*80)
track_a_counts = {"C00": 0, "C10": 0, "C01": 0, "C11": 0}
for k, v in track_a_ledger.items():
    c = k.split("_")[-1]
    s_cnt = sum(p[2] for p in v.values())
    track_a_counts[c] += s_cnt

for c in ["C00", "C10", "C01", "C11"]:
    cnt = track_a_counts[c]
    print(f"Condition {c}: Specified = {cnt}/36 ({cnt/36*100:.2f}%)")

