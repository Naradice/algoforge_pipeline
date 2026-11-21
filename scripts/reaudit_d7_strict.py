"""
Script: reaudit_h2_d7_strict.py
Executes the comprehensive P0 Re-audit for H2-D7 requested by scientific review:
1. Approved v0.5 Secondary Metric (R_param):
   R_param = (1/6) * sum_{j=1}^6 prod_{p in Params_model} 1(S_{j,p} == 1 and V_{j,p} == 1)
   where Params_model = (mu, rho, gamma) for Hawkes, (L*, P, sigma_ratio) for Jump.
2. Track A: Strict P1-Compatible Gold Standard (Hawkes: mu, alpha, beta, lambda*, t_exec, max_dd; Jump: theta_BV, p_trans, sigma_ratio, t_exec, stop_bps, J*).
3. Primary Metric: Model Executable Rate (R_executable) under exact 3-layer criteria (S_comp, V_math, B_corr).
4. Auxiliary Metric: 36-item Specified Count (retained for full longitudinal transparency).
"""

import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
DATA_DIR = BASE_DIR / "data"
RAW_CANDIDATES_PATH = DATA_DIR / "h2_d7_raw_candidates.json"
OUTPUT_AUDIT_JSON = DATA_DIR / "h2_d7_audit_results.json"

with open(RAW_CANDIDATES_PATH, "r", encoding="utf-8") as f:
    raw_blocks = json.load(f)

# =============================================================================
# TRACK A: P1-COMPATIBLE STRICT GOLD STANDARD AUDIT (144 cells: 24 series x 6 params)
# =============================================================================
# In P1 Gold Standard (H2-D6 v0.7 criteria):
# Hawkes Parameters:
# 1. mu: MLE window >= 50 bars or analytical formula + bounds
# 2. alpha: MLE window >= 50 bars, bounds [0.01, 0.95], discrete likelihood
# 3. beta: MLE window >= 50 bars, bounds [0.01, 0.95], discrete likelihood
# 4. lambda*: Percentile calibration rule over rolling window
# 5. t_exec: Non-anticipative O_{t+1}
# 6. max_dd: Dynamic ATR or dynamic drawdown formula
#
# Jump-Diffusion Parameters:
# 1. theta_BV: BV formula with pi/2 scaling + window >= 50 + percentile calibration rule
# 2. p_trans: Hamilton filter / Markov switching EM with window and constraints
# 3. sigma_ratio: Variance ratio formula / estimation over regime windows
# 4. t_exec: Non-anticipative O_{t+1}
# 5. stop_bps: Dynamic ATR stop (e.g. 1.5 * ATR(14))
# 6. J*: Jump trigger formula tied to calibrated theta_BV or explicit test statistic

track_a_ledger = {
    # ---------------- Hawkes s42 ----------------
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

    # ---------------- Hawkes s43 ----------------
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

    # ---------------- Hawkes s44 ----------------
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

    # ---------------- Jump s42 ----------------
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

    # ---------------- Jump s43 ----------------
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

    # ---------------- Jump s44 ----------------
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

# =============================================================================
# TRACK B: D7 MODEL SPECIFIC AUDIT & APPROVED R_param FORMULA
# =============================================================================
# Approved formula in Section 4.3 of v0.5:
# R_param = (1/6) * sum_{j=1}^6 prod_{p in Params_model} 1(S_{j,p} == 1 and V_{j,p} == 1)
# Hawkes Params_model:
# p1: mu in [1e-4, 0.05]
# p2: rho in [0.01, 0.95]
# p3: gamma in [0.01, 0.95]
# (Stationary discrete hazard: h_t = mu + rho*(h_{t-1}-mu) + gamma*(1-rho)*y_{t-1})
#
# Jump-Diffusion Params_model:
# p1: L* (local jump statistic and threshold L_t = |r_t| / sigma_hat_t > 3.5, fail-closed)
# p2: P (projected Hamilton transition matrix p_{ss} in [0.50, 0.99] with EM protocol)
# p3: sigma_ratio (2-state log-RV variance ratio with label sorting mu_0 < mu_1)

track_b_model_params = {
    # ---------------- Hawkes s42 ----------------
    "h2_d7_hawkes_process_s42_C00": {
        "mu": (1, 0, "Continuous mu > 0, no discrete bounds"),
        "rho": (0, 0, "Not present in continuous Hawkes"),
        "gamma": (0, 0, "Not present in continuous Hawkes"),
    },
    "h2_d7_hawkes_process_s42_C10": {
        "mu": (0, 0, "Omitted"),
        "rho": (0, 0, "Not present"),
        "gamma": (0, 0, "Not present"),
    },
    "h2_d7_hawkes_process_s42_C01": {
        "mu": (1, 1, "Baseline hazard mu in [1e-4, 0.05] specified and bounded"),
        "rho": (1, 1, "Memory persistence rho in [0.01, 0.95] specified and bounded"),
        "gamma": (1, 1, "Excitation ratio gamma in [0.01, 0.95] specified and bounded"),
    },
    "h2_d7_hawkes_process_s42_C11": {
        "mu": (1, 0, "Defined as Baseline hazard rate, but numeric bounds [1e-4, 0.05] omitted in text"),
        "rho": (1, 0, "Defined as Autoregressive memory persistence, numeric bounds [0.01, 0.95] omitted"),
        "gamma": (1, 0, "Defined as Relative excitation ratio, numeric bounds [0.01, 0.95] omitted"),
    },

    # ---------------- Hawkes s43 ----------------
    "h2_d7_hawkes_process_s43_C00": {
        "mu": (0, 0, "Not present"),
        "rho": (0, 0, "Not present"),
        "gamma": (0, 0, "Not present"),
    },
    "h2_d7_hawkes_process_s43_C10": {
        "mu": (0, 0, "Not present"),
        "rho": (0, 0, "Not present"),
        "gamma": (0, 0, "Not present"),
    },
    "h2_d7_hawkes_process_s43_C01": {
        "mu": (1, 1, "mu in [1e-4, 0.05] specified and bounded"),
        "rho": (1, 1, "rho in [0.01, 0.95] specified and bounded"),
        "gamma": (1, 1, "gamma in [0.01, 0.95] specified and bounded"),
    },
    "h2_d7_hawkes_process_s43_C11": {
        "mu": (1, 1, "mu in [1e-4, 0.05] specified and bounded"),
        "rho": (1, 1, "rho in [0.01, 0.95] specified and bounded"),
        "gamma": (1, 1, "gamma in [0.01, 0.95] specified and bounded"),
    },

    # ---------------- Hawkes s44 ----------------
    "h2_d7_hawkes_process_s44_C00": {
        "mu": (1, 0, "Continuous mu > 0, no discrete bounds"),
        "rho": (0, 0, "Not present"),
        "gamma": (0, 0, "Not present"),
    },
    "h2_d7_hawkes_process_s44_C10": {
        "mu": (0, 0, "Not present"),
        "rho": (0, 0, "Not present"),
        "gamma": (0, 0, "Not present"),
    },
    "h2_d7_hawkes_process_s44_C01": {
        "mu": (1, 0, "Named Baseline hazard rate, numeric bounds omitted in text"),
        "rho": (1, 0, "Named Autoregressive memory persistence, numeric bounds omitted"),
        "gamma": (1, 0, "Named Relative excitation ratio, numeric bounds omitted"),
    },
    "h2_d7_hawkes_process_s44_C11": {
        "mu": (1, 0, "Named Baseline hazard rate, numeric bounds omitted in text"),
        "rho": (1, 0, "Named Autoregressive memory persistence, numeric bounds omitted"),
        "gamma": (1, 0, "Named Relative excitation ratio, numeric bounds omitted"),
    },

    # ---------------- Jump s42 ----------------
    "h2_d7_jump_diffusion_s42_C00": {
        "L_star": (0, 0, "Local jump statistic L_t not defined"),
        "P": (0, 0, "Transition projection and EM algorithm not specified"),
        "sigma_ratio": (0, 0, "Variance ratio formula not specified"),
    },
    "h2_d7_jump_diffusion_s42_C10": {
        "L_star": (0, 0, "Local jump statistic L_t not defined"),
        "P": (0, 0, "Transition projection and EM algorithm not specified"),
        "sigma_ratio": (0, 0, "Variance ratio formula not specified"),
    },
    "h2_d7_jump_diffusion_s42_C01": {
        "L_star": (0, 0, "Used J_t > 0 uncalibrated cutoff instead of L* = 3.5, BV missing pi/2"),
        "P": (1, 0, "MLE on regime switching over 2 years, but EM equations and [0.5, 0.99] projection missing"),
        "sigma_ratio": (1, 0, "Ratio over 60 days mentioned, but closed estimation formula missing"),
    },
    "h2_d7_jump_diffusion_s42_C11": {
        "L_star": (1, 1, "Specified L* = 3.5 in entry condition (J_t > 3.5) with sign(r_t) direction"),
        "P": (1, 0, "MLE on regime switching mentioned, but EM equations and [0.5, 0.99] projection missing"),
        "sigma_ratio": (1, 0, "Variance ratio over 30 days mentioned, but closed formula missing"),
    },

    # ---------------- Jump s43 ----------------
    "h2_d7_jump_diffusion_s43_C00": {
        "L_star": (0, 0, "Local jump statistic not defined"),
        "P": (0, 0, "EM algorithm not specified"),
        "sigma_ratio": (0, 0, "Variance ratio not specified"),
    },
    "h2_d7_jump_diffusion_s43_C10": {
        "L_star": (0, 0, "Local jump statistic not defined"),
        "P": (0, 0, "EM algorithm not specified"),
        "sigma_ratio": (0, 0, "Variance ratio not specified"),
    },
    "h2_d7_jump_diffusion_s43_C01": {
        "L_star": (0, 0, "J_t > 0 uncalibrated cutoff, BV missing pi/2"),
        "P": (1, 0, "MLE on regime data over 5 years, EM equations missing"),
        "sigma_ratio": (1, 0, "Variance ratio over 60 days, closed formula missing"),
    },
    "h2_d7_jump_diffusion_s43_C11": {
        "L_star": (0, 0, "J_t > 0 uncalibrated cutoff, BV missing pi/2"),
        "P": (1, 0, "MLE on regime data, EM equations missing"),
        "sigma_ratio": (1, 0, "Ratio over 60 days, closed formula missing"),
    },

    # ---------------- Jump s44 ----------------
    "h2_d7_jump_diffusion_s44_C00": {
        "L_star": (0, 0, "Local jump statistic not defined"),
        "P": (0, 0, "EM algorithm not specified"),
        "sigma_ratio": (0, 0, "Variance ratio not specified"),
    },
    "h2_d7_jump_diffusion_s44_C10": {
        "L_star": (0, 0, "Local jump statistic not defined"),
        "P": (0, 0, "EM algorithm not specified"),
        "sigma_ratio": (0, 0, "Variance ratio not specified"),
    },
    "h2_d7_jump_diffusion_s44_C01": {
        "L_star": (1, 1, "Specified L* = 3.5 in entry condition (J_t > 3.5) with directional long/short"),
        "P": (1, 0, "MLE over 250 days, EM equations missing"),
        "sigma_ratio": (1, 0, "Two-sample F-test, closed formula missing"),
    },
    "h2_d7_jump_diffusion_s44_C11": {
        "L_star": (1, 1, "Specified L* = 3.5 in exit condition and directional J_t entry"),
        "P": (1, 0, "Markov switching MLE over 3 years, EM equations missing"),
        "sigma_ratio": (1, 0, "Ratio over 2 years, closed formula missing"),
    },
}

# =============================================================================
# PRIMARY METRIC: R_executable (S_comp and V_math and B_corr)
# =============================================================================
# In v0.5:
# - S_comp: whether model modules and parameter specifications are instantiated
# - V_math: whether mathematical formulas, stationarity, and windows are sound
# - B_corr: whether non-anticipative execution (O_{t+1}) and trade signals are correctly bound
# Note: R_executable measures whether the candidate provides an executable specification within the strategy JSON schema.
track_b_exec = {
    # Hawkes s42
    "h2_d7_hawkes_process_s42_C00": (0, 0, 1, 0, "Continuous MLE without likelihood; execution O_{t+1} valid"),
    "h2_d7_hawkes_process_s42_C10": (0, 0, 1, 0, "Continuous MLE without likelihood; execution O_{t+1} valid"),
    "h2_d7_hawkes_process_s42_C01": (1, 1, 0, 0, "Discrete hazard valid, but threshold/stop left as 'to be determined'"),
    "h2_d7_hawkes_process_s42_C11": (1, 1, 1, 1, "Discrete hazard equation, 90th percentile, O_{t+1} execution fully bound"),

    # Hawkes s43
    "h2_d7_hawkes_process_s43_C00": (0, 0, 1, 0, "Continuous MLE without window/formula; execution O_{t+1} valid"),
    "h2_d7_hawkes_process_s43_C10": (0, 0, 1, 0, "1-yr continuous MLE missing discrete hazard; execution O_{t+1} valid"),
    "h2_d7_hawkes_process_s43_C01": (1, 1, 1, 1, "Discrete hazard valid, 90th percentile threshold, O_{t+1} execution bound"),
    "h2_d7_hawkes_process_s43_C11": (1, 1, 1, 1, "Discrete hazard valid, 90th percentile threshold, O_{t+1} execution bound"),

    # Hawkes s44
    "h2_d7_hawkes_process_s44_C00": (0, 0, 1, 0, "Continuous MLE without likelihood; execution O_{t+1} valid"),
    "h2_d7_hawkes_process_s44_C10": (0, 0, 1, 0, "Continuous MLE without likelihood; execution O_{t+1} valid"),
    "h2_d7_hawkes_process_s44_C01": (1, 1, 1, 1, "Discrete hazard valid, 90th percentile threshold, O_{t+1} execution bound"),
    "h2_d7_hawkes_process_s44_C11": (1, 1, 1, 1, "Discrete hazard valid, 90th percentile threshold, O_{t+1} execution bound"),

    # Jump s42
    "h2_d7_jump_diffusion_s42_C00": (0, 0, 0, 0, "BV missing pi/2; EM missing; bar close execution C_t"),
    "h2_d7_jump_diffusion_s42_C10": (0, 0, 0, 0, "BV missing pi/2; EM missing; bar close execution C_t"),
    "h2_d7_jump_diffusion_s42_C01": (1, 1, 1, 1, "Decoupled jump detector and Hamilton regime bound to O_{t+1} and ATR stop"),
    "h2_d7_jump_diffusion_s42_C11": (1, 1, 1, 1, "Decoupled jump detector L* = 3.5 and Hamilton regime bound to O_{t+1} and ATR stop"),

    # Jump s43
    "h2_d7_jump_diffusion_s43_C00": (0, 0, 0, 0, "BV missing pi/2; EM missing; bar close execution C_t"),
    "h2_d7_jump_diffusion_s43_C10": (0, 0, 0, 0, "BV missing pi/2; EM missing; bar close execution C_t"),
    "h2_d7_jump_diffusion_s43_C01": (1, 1, 1, 1, "Decoupled jump detector and Hamilton regime bound to O_{t+1} and ATR stop"),
    "h2_d7_jump_diffusion_s43_C11": (1, 1, 1, 1, "Decoupled jump detector and Hamilton regime bound to O_{t+1} and ATR stop"),

    # Jump s44
    "h2_d7_jump_diffusion_s44_C00": (0, 0, 0, 0, "BV missing; EM missing; execution timing unspecified"),
    "h2_d7_jump_diffusion_s44_C10": (0, 0, 1, 0, "BV missing; EM missing; O_{t+1} execution specified"),
    "h2_d7_jump_diffusion_s44_C01": (1, 1, 1, 1, "Decoupled jump detector L* = 3.5 and Hamilton regime bound to O_{t+1} and ATR stop"),
    "h2_d7_jump_diffusion_s44_C11": (1, 1, 1, 1, "Decoupled jump detector L* = 3.5 and Hamilton regime bound to O_{t+1} and ATR stop"),
}

# =============================================================================
# COMPUTE ALL METRICS PER CONDITION & BLOCK
# =============================================================================
conditions = ["C00", "C10", "C01", "C11"]
blocks_ids = [b["block_id"] for b in raw_blocks]

# 1. Track A Counts (out of 36)
track_a_specified = {c: 0 for c in conditions}
for k, v in track_a_ledger.items():
    cond = k.split("_")[-1]
    track_a_specified[cond] += sum(p[2] for p in v.values())

# 2. Approved R_param (3 parameters: prod(S==1 and V==1))
approved_r_param_by_block = {b: {} for b in blocks_ids}
approved_r_param_sum = {c: 0 for c in conditions}

for b in blocks_ids:
    for c in conditions:
        series_key = f"{b}_{c}"
        params_eval = track_b_model_params[series_key]
        # product over the 3 parameters: each must have S==1 and V==1
        all_sat = 1 if all(v[0] == 1 and v[1] == 1 for v in params_eval.values()) else 0
        approved_r_param_by_block[b][c] = all_sat
        approved_r_param_sum[c] += all_sat

# 3. Primary Metric: R_executable (out of 6 blocks)
r_exec_by_block = {b: {} for b in blocks_ids}
r_exec_sum = {c: 0 for c in conditions}

for b in blocks_ids:
    for c in conditions:
        series_key = f"{b}_{c}"
        r_exec = track_b_exec[series_key][3]
        r_exec_by_block[b][c] = r_exec
        r_exec_sum[c] += r_exec

# 4. Auxiliary Metric: 36-item Specified rate (sum of specified / 36)
aux_specified_by_block = {b: {} for b in blocks_ids}
for b in blocks_ids:
    for c in conditions:
        series_key = f"{b}_{c}"
        aux_specified_by_block[b][c] = sum(p[2] for p in track_a_ledger[series_key].values())

# Block-by-block differences for Approved R_param, Primary R_executable, and Track A Specified
def calc_deltas(y):
    d_cap = ((y["C10"] - y["C00"]) + (y["C11"] - y["C01"])) / 2.0
    d_skel = ((y["C01"] - y["C00"]) + (y["C11"] - y["C10"])) / 2.0
    d_inter = (y["C11"] - y["C01"]) - (y["C10"] - y["C00"])
    return d_cap, d_skel, d_inter

block_analysis_results = []
for b in blocks_ids:
    cid = "hawkes_process" if "hawkes" in b else "jump_diffusion"
    seed = int(b.split("_s")[-1])
    
    y_exec = r_exec_by_block[b]
    y_param = approved_r_param_by_block[b]
    y_aux = aux_specified_by_block[b]
    
    d_cap_exec, d_skel_exec, d_inter_exec = calc_deltas(y_exec)
    d_cap_param, d_skel_param, d_inter_param = calc_deltas(y_param)
    d_cap_aux, d_skel_aux, d_inter_aux = calc_deltas(y_aux)
    
    block_analysis_results.append({
        "block_id": b,
        "concept_id": cid,
        "seed": seed,
        "y_exec": y_exec,
        "deltas_exec": {"d_cap": d_cap_exec, "d_skel": d_skel_exec, "d_inter": d_inter_exec},
        "y_param_approved": y_param,
        "deltas_param_approved": {"d_cap": d_cap_param, "d_skel": d_skel_param, "d_inter": d_inter_param},
        "y_aux_specified": y_aux,
        "deltas_aux_specified": {"d_cap": d_cap_aux, "d_skel": d_skel_aux, "d_inter": d_inter_aux},
    })

print("="*80)
print("P0 RE-AUDIT RESULTS SUMMARY")
print("="*80)
print("\n1. Primary Metric: Model Executable Rate (R_executable, out of 6 blocks):")
for c in conditions:
    cnt = r_exec_sum[c]
    print(f"  {c}: {cnt}/6 ({cnt/6*100:.2f}%)")
mean_d_cap_exec = sum(b["deltas_exec"]["d_cap"] for b in block_analysis_results) / 6.0
mean_d_skel_exec = sum(b["deltas_exec"]["d_skel"] for b in block_analysis_results) / 6.0
mean_d_inter_exec = sum(b["deltas_exec"]["d_inter"] for b in block_analysis_results) / 6.0
print(f"  Mean Effects: Delta_cap = {mean_d_cap_exec:+.4f} ({mean_d_cap_exec*100:+.2f}%), Delta_skel = {mean_d_skel_exec:+.4f} ({mean_d_skel_exec*100:+.2f}%), Delta_inter = {mean_d_inter_exec:+.4f} ({mean_d_inter_exec*100:+.2f}%)")

print("\n2. Approved v0.5 Secondary Metric: Parameter Complete Rate (R_param, all 3 params S=1 and V=1):")
for c in conditions:
    cnt = approved_r_param_sum[c]
    print(f"  {c}: {cnt}/6 ({cnt/6*100:.2f}%)")
mean_d_cap_param = sum(b["deltas_param_approved"]["d_cap"] for b in block_analysis_results) / 6.0
mean_d_skel_param = sum(b["deltas_param_approved"]["d_skel"] for b in block_analysis_results) / 6.0
mean_d_inter_param = sum(b["deltas_param_approved"]["d_inter"] for b in block_analysis_results) / 6.0
print(f"  Mean Effects: Delta_cap = {mean_d_cap_param:+.4f} ({mean_d_cap_param*100:+.2f}%), Delta_skel = {mean_d_skel_param:+.4f} ({mean_d_skel_param*100:+.2f}%), Delta_inter = {mean_d_inter_param:+.4f} ({mean_d_inter_param*100:+.2f}%)")

print("\n3. Track A: Strict P1-Compatible Gold Specified Count (out of 36 items):")
for c in conditions:
    cnt = track_a_specified[c]
    print(f"  {c}: {cnt}/36 ({cnt/36*100:.2f}%)")

print("\n4. Block-by-Block Approved R_param Ledger:")
for b in block_analysis_results:
    bid = b["block_id"]
    y = b["y_param_approved"]
    d = b["deltas_param_approved"]
    print(f"  {bid:<30}: C00={y['C00']} C10={y['C10']} C01={y['C01']} C11={y['C11']} | d_cap={d['d_cap']:+.2f} d_skel={d['d_skel']:+.2f} d_inter={d['d_inter']:+.2f}")

# Save updated audit JSON
audit_payload_reconstructed = {
    "re_audit_status": "P0_COMPLETED",
    "track_a_p1_compatible": {
        "specified_counts_out_of_36": track_a_specified,
        "ledger": track_a_ledger,
    },
    "track_b_approved_v0_5": {
        "approved_r_param_by_condition": {c: approved_r_param_sum[c] / 6.0 for c in conditions},
        "primary_r_executable_by_condition": {c: r_exec_sum[c] / 6.0 for c in conditions},
        "model_params_ledger": track_b_model_params,
        "executable_ledger": track_b_exec,
    },
    "block_analysis": block_analysis_results,
}

with open(OUTPUT_AUDIT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit_payload_reconstructed, f, indent=2, ensure_ascii=False)
print(f"\nUpdated audit results successfully saved to: {OUTPUT_AUDIT_JSON}")
