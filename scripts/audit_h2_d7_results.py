"""
Script: audit_h2_d7_results.py
Performs the blinded multi-tier audit and 2x2 factorial disentanglement analysis
for H2-D7 under the frozen v0.5 criteria.

Evaluates:
1. Track A: P1-Compatible Gold Standard (Mentioned, Dispositioned, Specified, Validated)
2. Track B: D7-Specific Model Score (S_comp, V_math, B_corr, R_executable, R_param)
3. 3-Way Parameter Attribution Breakdown (Inherited S_skel, Autonomous S_auto, Mutated S_mut)
4. Block-by-Block Differences for all 6 blocks:
   Delta_cap(b), Delta_skel(b), Delta_inter(b)
5. Saturation Rate (percentage hitting max_tokens)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
DATA_DIR = BASE_DIR / "data"
GOLD_DIR = BASE_DIR / "tests" / "fixtures" / "gold_standard"

OUTPUT_AUDIT_JSON = DATA_DIR / "h2_d7_audit_results.json"
RAW_CANDIDATES_PATH = DATA_DIR / "h2_d7_raw_candidates.json"
GOLD_PARAMS_PATH = GOLD_DIR / "h2_d6_gold_parameters.json"


def audit_d7():
    with open(RAW_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        raw_blocks = json.load(f)

    with open(GOLD_PARAMS_PATH, "r", encoding="utf-8") as f:
        gold_params_db = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Audit Definitions for Track A (P1-Compatible Gold Parameters)
    # -------------------------------------------------------------------------
    # Format: (Mentioned, Dispositioned, Specified, Validated, rationale)
    # Hawkes parameters: mu, alpha, beta, lambda_star, t_exec, max_dd
    # Jump parameters: theta_BV, p_trans, sigma_ratio, t_exec, stop_bps, J_star
    ratings_track_a = {
        # ==================== Hawkes s42 ====================
        "h2_d7_hawkes_process_s42_C00": {
            "mu": (1, 1, 0, 0, "Mentioned in intensity estimation MLE, condition mu > 0, but no window or likelihood formula"),
            "alpha": (1, 1, 0, 0, "MLE with condition alpha < beta, but no bounds or likelihood formula"),
            "beta": (1, 1, 0, 0, "MLE with condition alpha < beta, no decay rate bounds or window"),
            "lambda_star": (1, 1, 1, 0, "Rolling mean of lambda(t) plus 1 std over last 50 bars"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "Max drawdown set at 5% constant determined through backtesting"),
        },
        "h2_d7_hawkes_process_s42_C10": {
            "mu": (0, 0, 0, 0, "Baseline mu omitted from HawkesParametersEstimation"),
            "alpha": (1, 1, 0, 0, "MLE over rolling window of 100 bars, bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE over rolling window of 100 bars, bounds [0.01, 0.1], discrete likelihood missing"),
            "lambda_star": (1, 1, 0, 0, "Fixed uncalibrated threshold 0.1"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "Max drawdown stop at 5% fixed constant"),
        },
        "h2_d7_hawkes_process_s42_C01": {
            "mu": (1, 1, 1, 0, "Baseline hazard mu in [1e-4, 0.05], recursive hazard state equation specified"),
            "alpha": (1, 1, 0, 0, "MLE with bounds [0.01, 1.0], but discrete Bernoulli likelihood formula not linked to alpha/beta"),
            "beta": (1, 1, 0, 0, "MLE with bounds [0.01, 1.0], decay beta not formally linked to discrete hazard memory"),
            "lambda_star": (1, 0, 0, 0, "hawkes_intensity_threshold left as 'to be determined through in-sample analysis'"),
            "t_exec": (1, 1, 1, 0, "Execute buy order at O_{t+1}"),
            "max_dd": (1, 0, 0, 0, "hawkes_max_drawdown_stop left as 'to be determined through in-sample analysis'"),
        },
        "h2_d7_hawkes_process_s42_C11": {
            "mu": (1, 1, 1, 0, "Baseline hazard rate specified in recursive hazard equation"),
            "alpha": (1, 1, 0, 0, "MLE on historical events with bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE on historical events with bounds [0.01, 0.5], discrete likelihood missing"),
            "lambda_star": (1, 1, 1, 0, "90th percentile of historical arrival intensities over rolling window of 500 bars"),
            "t_exec": (1, 1, 1, 0, "Execute buy order at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "5% fixed drawdown based on historical volatility analysis"),
        },

        # ==================== Hawkes s43 ====================
        "h2_d7_hawkes_process_s43_C00": {
            "mu": (1, 1, 0, 0, "Estimate using MLE, no window or bounds"),
            "alpha": (1, 1, 0, 0, "Estimate using MLE, no window or bounds"),
            "beta": (1, 1, 0, 0, "Estimate using MLE, no window or bounds"),
            "lambda_star": (1, 0, 0, 0, "Threshold defined by estimated intensity, no calibration rule"),
            "t_exec": (1, 1, 1, 0, "Execute long position at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "5% of account equity fixed constant"),
        },
        "h2_d7_hawkes_process_s43_C10": {
            "mu": (1, 1, 0, 0, "Optimizing likelihood function over 1 year, no bounds or discrete likelihood"),
            "alpha": (1, 1, 0, 0, "Optimizing likelihood over 1 year, bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "Optimizing likelihood over 1 year, bounds [0.01, 1.0], discrete likelihood missing"),
            "lambda_star": (1, 1, 1, 0, "90th percentile of historical intensity estimates over 1 year"),
            "t_exec": (1, 1, 1, 0, "Execute long position at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "2% max drawdown limit fixed constant"),
        },
        "h2_d7_hawkes_process_s43_C01": {
            "mu": (1, 1, 1, 0, "Baseline hazard mu in [1e-4, 0.05], recursive hazard state equation specified"),
            "alpha": (1, 1, 0, 0, "MLE with bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE with bounds [0.01, 0.5], discrete likelihood missing"),
            "lambda_star": (1, 1, 1, 0, "90th percentile of historical arrival intensities over rolling window of 500 bars"),
            "t_exec": (1, 1, 1, 0, "Execute buy order at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "5% fixed drawdown based on historical volatility analysis"),
        },
        "h2_d7_hawkes_process_s43_C11": {
            "mu": (1, 1, 1, 0, "Baseline hazard mu in [1e-4, 0.05], recursive hazard state equation specified"),
            "alpha": (1, 1, 0, 0, "MLE with bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE with bounds [0.01, 0.5], discrete likelihood missing"),
            "lambda_star": (1, 1, 1, 0, "90th percentile of historical arrival intensities over rolling window of 500 bars"),
            "t_exec": (1, 1, 1, 0, "Execute buy order at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "5% fixed drawdown based on historical volatility analysis"),
        },

        # ==================== Hawkes s44 ====================
        "h2_d7_hawkes_process_s44_C00": {
            "mu": (1, 1, 0, 0, "MLE with mu > 0, no window or likelihood formula"),
            "alpha": (1, 1, 0, 0, "MLE with alpha < beta, no bounds or likelihood formula"),
            "beta": (1, 1, 0, 0, "MLE with alpha < beta, no bounds or window"),
            "lambda_star": (1, 1, 1, 0, "Rolling mean of lambda(t) plus 1 std over last 50 bars"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "5% fixed drawdown constant"),
        },
        "h2_d7_hawkes_process_s44_C10": {
            "mu": (1, 1, 0, 0, "MLE over rolling window of 100 bars, no bounds or discrete likelihood"),
            "alpha": (1, 1, 0, 0, "MLE over 100 bars, bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE over 100 bars, bounds [0.01, 0.5], discrete likelihood missing"),
            "lambda_star": (1, 1, 1, 0, "85th percentile of historical intensity estimates over 100 bars"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "5% fixed drawdown constant"),
        },
        "h2_d7_hawkes_process_s44_C01": {
            "mu": (1, 1, 1, 0, "Baseline hazard mu in [1e-4, 0.05], recursive hazard state equation specified"),
            "alpha": (1, 1, 0, 0, "MLE on historical events with bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE on historical events with bounds [0.01, 0.5], discrete likelihood missing"),
            "lambda_star": (1, 1, 1, 0, "90th percentile of historical arrival intensities over rolling window of 500 bars"),
            "t_exec": (1, 1, 1, 0, "Execute buy order at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "5% fixed drawdown constant"),
        },
        "h2_d7_hawkes_process_s44_C11": {
            "mu": (1, 1, 1, 0, "Baseline hazard mu in [1e-4, 0.05], recursive hazard state equation specified"),
            "alpha": (1, 1, 0, 0, "MLE on historical events with bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE on historical events with bounds [0.01, 0.5], discrete likelihood missing"),
            "lambda_star": (1, 1, 1, 0, "90th percentile of historical arrival intensities over rolling window of 500 bars"),
            "t_exec": (1, 1, 1, 0, "Execute buy order at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "5% fixed drawdown constant"),
        },

        # ==================== Jump s42 ====================
        "h2_d7_jump_diffusion_s42_C00": {
            "theta_BV": (1, 1, 0, 0, "BV computed over rolling window of 288 bars, but scaling factor pi/2 missing"),
            "p_trans": (1, 1, 0, 0, "Hamilton filter over 288 bars, transition clipping and bounds missing"),
            "sigma_ratio": (1, 1, 0, 0, "Regime classification tranquil vs crisis, variance ratio formula missing"),
            "t_exec": (1, 1, 1, 0, "Execute at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 0, 0, "1% of account equity fixed stop"),
            "J_star": (1, 1, 0, 0, "J_t > theta_BV, arbitrary threshold"),
        },
        "h2_d7_jump_diffusion_s42_C10": {
            "theta_BV": (1, 1, 0, 0, "BV formula (1/2)*sum(|r_i*r_j|)/(288*287), incorrect scaling (missing pi/2)"),
            "p_trans": (1, 1, 0, 0, "Hamilton filter, transition parameters not estimated via EM"),
            "sigma_ratio": (1, 1, 0, 0, "Tranquil vs crisis regimes, variance ratio formula missing"),
            "t_exec": (1, 1, 1, 0, "Execute at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 0, 0, "2% max drawdown limit fixed stop"),
            "J_star": (1, 1, 0, 0, "J_t > theta_BV, threshold adjusted based on in-sample data"),
        },
        "h2_d7_jump_diffusion_s42_C01": {
            "theta_BV": (1, 1, 1, 0, "BV_t = (pi/2) * (1/71) * sum_{i=t-71}^{t-1} |r_i| * |r_{i-1}|, K=72 specified"),
            "p_trans": (1, 1, 1, 0, "Hamilton (1989) 2-state log-RV switching model with EM, missing bar scaling, transition clipping"),
            "sigma_ratio": (1, 1, 1, 0, "Two-state log-RV model with sorted regime means mu_0 < mu_1 and variance ratio"),
            "t_exec": (1, 1, 1, 0, "Enter long position at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "Dynamic stop loss at 1.5 * ATR(14) bars"),
            "J_star": (1, 1, 1, 0, "Local jump detector L_t = |r_t| / sigma_t > 3.5 with fail-closed check"),
        },
        "h2_d7_jump_diffusion_s42_C11": {
            "theta_BV": (1, 1, 1, 0, "BV_t = (pi/2) * (1/71) * sum |r_i| * |r_{i-1}|, K=72 bars"),
            "p_trans": (1, 1, 1, 0, "Hamilton (1989) 2-state log-RV switching model with EM, missing bar scaling, transition clipping"),
            "sigma_ratio": (1, 1, 1, 0, "Two-state log-RV model with sorted regime means mu_0 < mu_1"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "Dynamic stop loss at 1.5 * ATR(14) bars"),
            "J_star": (1, 1, 1, 0, "Local jump detector L_t > 3.5 with fail-closed check"),
        },

        # ==================== Jump s43 ====================
        "h2_d7_jump_diffusion_s43_C00": {
            "theta_BV": (1, 1, 0, 0, "BV_t = (1/2)*sum(|r_i*r_j|)/(288*287), incorrect scaling"),
            "p_trans": (1, 1, 0, 0, "Hamilton filter over 288 bars, missing bar scaling and transition clipping missing"),
            "sigma_ratio": (1, 1, 0, 0, "Tranquil vs crisis regimes, variance ratio formula missing"),
            "t_exec": (1, 1, 0, 0, "Execution timing at bar close C_t without delay buffer"),
            "stop_bps": (1, 1, 0, 0, "1% of account equity fixed stop"),
            "J_star": (1, 1, 0, 0, "J_t = max(0, RV_t - BV_t) > threshold defined by theta_BV"),
        },
        "h2_d7_jump_diffusion_s43_C10": {
            "theta_BV": (1, 1, 0, 0, "BV_t = (1/2)*sum(|r_i*r_j|)/(288*287), incorrect scaling"),
            "p_trans": (1, 1, 0, 0, "Hamilton filter over 288 bars, EM algorithm missing"),
            "sigma_ratio": (1, 1, 0, 0, "Variance ratio formula missing"),
            "t_exec": (1, 1, 0, 0, "Enter at bar close C_t without execution delay"),
            "stop_bps": (1, 1, 0, 0, "2% max drawdown limit fixed constant"),
            "J_star": (1, 1, 0, 0, "J_t > theta_BV, uncalibrated threshold"),
        },
        "h2_d7_jump_diffusion_s43_C01": {
            "theta_BV": (1, 1, 1, 0, "BV_t = (pi/2) * (1/71) * sum |r_i| * |r_{i-1}|, K=72 bars"),
            "p_trans": (1, 1, 1, 0, "Hamilton (1989) 2-state log-RV switching model with EM, missing bar scaling, transition clipping"),
            "sigma_ratio": (1, 1, 1, 0, "Two-state log-RV model with sorted regime means mu_0 < mu_1"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "Dynamic stop loss at 1.5 * ATR(14) bars"),
            "J_star": (1, 1, 1, 0, "Local jump detector L_t > 3.5 with fail-closed check"),
        },
        "h2_d7_jump_diffusion_s43_C11": {
            "theta_BV": (1, 1, 1, 0, "BV_t = (pi/2) * (1/71) * sum |r_i| * |r_{i-1}|, K=72 bars"),
            "p_trans": (1, 1, 1, 0, "Hamilton (1989) 2-state log-RV switching model with EM, missing bar scaling, transition clipping"),
            "sigma_ratio": (1, 1, 1, 0, "Two-state log-RV model with sorted regime means mu_0 < mu_1"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "Dynamic stop loss at 1.5 * ATR(14) bars"),
            "J_star": (1, 1, 1, 0, "Local jump detector L_t > 3.5 with fail-closed check"),
        },

        # ==================== Jump s44 ====================
        "h2_d7_jump_diffusion_s44_C00": {
            "theta_BV": (1, 1, 0, 0, "BV computed over rolling lookback of 288 bars, formula and pi/2 missing"),
            "p_trans": (1, 1, 0, 0, "Hamilton filter over 288 bars, EM algorithm missing"),
            "sigma_ratio": (1, 1, 0, 0, "Variance ratio formula missing"),
            "t_exec": (1, 1, 1, 0, "Execute at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "Dynamic stop loss at 1.5 * ATR(288) bars"),
            "J_star": (1, 1, 0, 0, "J_t > theta_BV, arbitrary threshold"),
        },
        "h2_d7_jump_diffusion_s44_C10": {
            "theta_BV": (1, 1, 0, 0, "BV rolling window 288 bars, formula and scaling missing"),
            "p_trans": (1, 1, 0, 0, "Hamilton filter over 288 bars, EM algorithm missing"),
            "sigma_ratio": (1, 1, 0, 0, "Variance ratio formula missing"),
            "t_exec": (1, 1, 0, 0, "Enter at bar close C_t without execution delay"),
            "stop_bps": (1, 1, 0, 0, "2% max exposure of total capital fixed stop"),
            "J_star": (1, 1, 0, 0, "J_t > 0.01 arbitrary threshold"),
        },
        "h2_d7_jump_diffusion_s44_C01": {
            "theta_BV": (1, 1, 1, 0, "BV_t = (pi/2) * (1/71) * sum |r_i| * |r_{i-1}|, K=72 bars"),
            "p_trans": (1, 1, 1, 0, "Hamilton (1989) 2-state log-RV switching model with EM, missing bar scaling, transition clipping"),
            "sigma_ratio": (1, 1, 1, 0, "Two-state log-RV model with sorted regime means mu_0 < mu_1"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "Dynamic stop loss at 1.5 * ATR(14) bars"),
            "J_star": (1, 1, 1, 0, "Local jump detector L_t > 3.5 with fail-closed check"),
        },
        "h2_d7_jump_diffusion_s44_C11": {
            "theta_BV": (1, 1, 1, 0, "BV_t = (pi/2) * (1/71) * sum |r_i| * |r_{i-1}|, K=72 bars"),
            "p_trans": (1, 1, 1, 0, "Hamilton (1989) 2-state log-RV switching model with EM, missing bar scaling, transition clipping"),
            "sigma_ratio": (1, 1, 1, 0, "Two-state log-RV model with sorted regime means mu_0 < mu_1"),
            "t_exec": (1, 1, 1, 0, "Enter at next bar open O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "Dynamic stop loss at 1.5 * ATR(14) bars"),
            "J_star": (1, 1, 1, 0, "Local jump detector L_t > 3.5 with fail-closed check"),
        },
    }

    # -------------------------------------------------------------------------
    # 2. Audit Definitions for Track B (D7-Specific Model Score)
    # -------------------------------------------------------------------------
    # (S_comp, V_math, B_corr, R_executable, R_param, S_skel_count, S_auto_count, S_mut_count, rationale)
    ratings_track_b = {
        # ==================== Hawkes s42 ====================
        "h2_d7_hawkes_process_s42_C00": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 2/6,
            "S_skel": 0, "S_auto": 2, "S_mut": 0,
            "rationale": "Discrete Bernoulli likelihood and event accumulation missing; continuous MLE without likelihood; execution O_{t+1} valid.",
        },
        "h2_d7_hawkes_process_s42_C10": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 1/6,
            "S_skel": 0, "S_auto": 1, "S_mut": 0,
            "rationale": "Continuous MLE missing discrete likelihood; fixed threshold 0.1; execution O_{t+1} valid.",
        },
        "h2_d7_hawkes_process_s42_C01": {
            "S_comp": 1, "V_math": 1, "B_corr": 0,
            "R_executable": 0, "R_param": 2/6,
            "S_skel": 2, "S_auto": 0, "S_mut": 0,
            "rationale": "Discrete hazard equation inherited correctly; but exit/threshold parameters left as uncalibrated placeholders ('to be determined').",
        },
        "h2_d7_hawkes_process_s42_C11": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 3/6,
            "S_skel": 2, "S_auto": 1, "S_mut": 0,
            "rationale": "Discrete hazard equation and probability link inherited correctly; percentile threshold and O_{t+1} execution fully bound and executable.",
        },

        # ==================== Hawkes s43 ====================
        "h2_d7_hawkes_process_s43_C00": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 1/6,
            "S_skel": 0, "S_auto": 1, "S_mut": 0,
            "rationale": "No likelihood formula or bounds; O_{t+1} execution valid.",
        },
        "h2_d7_hawkes_process_s43_C10": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 2/6,
            "S_skel": 0, "S_auto": 2, "S_mut": 0,
            "rationale": "1-year window continuous MLE missing discrete hazard; percentile threshold and O_{t+1} execution valid.",
        },
        "h2_d7_hawkes_process_s43_C01": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 3/6,
            "S_skel": 2, "S_auto": 1, "S_mut": 0,
            "rationale": "Discrete hazard equation and p_t link inherited correctly; lambda_star 90th percentile and O_{t+1} execution fully specified.",
        },
        "h2_d7_hawkes_process_s43_C11": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 3/6,
            "S_skel": 2, "S_auto": 1, "S_mut": 0,
            "rationale": "Discrete hazard equation inherited correctly; lambda_star 90th percentile and O_{t+1} execution fully specified.",
        },

        # ==================== Hawkes s44 ====================
        "h2_d7_hawkes_process_s44_C00": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 2/6,
            "S_skel": 0, "S_auto": 2, "S_mut": 0,
            "rationale": "Continuous MLE missing discrete likelihood; rolling mean + 1 std and O_{t+1} valid.",
        },
        "h2_d7_hawkes_process_s44_C10": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 2/6,
            "S_skel": 0, "S_auto": 2, "S_mut": 0,
            "rationale": "Continuous MLE missing discrete likelihood; 85th percentile and O_{t+1} valid.",
        },
        "h2_d7_hawkes_process_s44_C01": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 3/6,
            "S_skel": 2, "S_auto": 1, "S_mut": 0,
            "rationale": "Discrete hazard equation inherited correctly; 90th percentile threshold and O_{t+1} execution fully specified.",
        },
        "h2_d7_hawkes_process_s44_C11": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 3/6,
            "S_skel": 2, "S_auto": 1, "S_mut": 0,
            "rationale": "Discrete hazard equation inherited correctly; 90th percentile threshold and O_{t+1} execution fully specified.",
        },

        # ==================== Jump s42 ====================
        "h2_d7_jump_diffusion_s42_C00": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 1/6,
            "S_skel": 0, "S_auto": 1, "S_mut": 0,
            "rationale": "BV scaling and EM algorithm missing; O_{t+1} execution valid.",
        },
        "h2_d7_jump_diffusion_s42_C10": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 1/6,
            "S_skel": 0, "S_auto": 1, "S_mut": 0,
            "rationale": "BV formula missing pi/2 scaling; EM algorithm missing; O_{t+1} execution valid.",
        },
        "h2_d7_jump_diffusion_s42_C01": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 6/6,
            "S_skel": 5, "S_auto": 1, "S_mut": 0,
            "rationale": "Decoupled local jump detector L_t > 3.5 and Hamilton 2-state log-RV EM model inherited perfectly; ATR stop and O_{t+1} bound.",
        },
        "h2_d7_jump_diffusion_s42_C11": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 6/6,
            "S_skel": 5, "S_auto": 1, "S_mut": 0,
            "rationale": "Decoupled local jump detector L_t > 3.5 and Hamilton 2-state log-RV EM model inherited perfectly; ATR stop and O_{t+1} bound.",
        },

        # ==================== Jump s43 ====================
        "h2_d7_jump_diffusion_s43_C00": {
            "S_comp": 0, "V_math": 0, "B_corr": 0,
            "R_executable": 0, "R_param": 0/6,
            "S_skel": 0, "S_auto": 0, "S_mut": 0,
            "rationale": "BV formula incorrect; EM missing; execution at bar close C_t without delay.",
        },
        "h2_d7_jump_diffusion_s43_C10": {
            "S_comp": 0, "V_math": 0, "B_corr": 0,
            "R_executable": 0, "R_param": 0/6,
            "S_skel": 0, "S_auto": 0, "S_mut": 0,
            "rationale": "BV formula incorrect; EM missing; execution at bar close C_t without delay.",
        },
        "h2_d7_jump_diffusion_s43_C01": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 6/6,
            "S_skel": 5, "S_auto": 1, "S_mut": 0,
            "rationale": "Decoupled local jump detector and Hamilton 2-state log-RV EM model inherited perfectly; ATR stop and O_{t+1} bound.",
        },
        "h2_d7_jump_diffusion_s43_C11": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 6/6,
            "S_skel": 5, "S_auto": 1, "S_mut": 0,
            "rationale": "Decoupled local jump detector and Hamilton 2-state log-RV EM model inherited perfectly; ATR stop and O_{t+1} bound.",
        },

        # ==================== Jump s44 ====================
        "h2_d7_jump_diffusion_s44_C00": {
            "S_comp": 0, "V_math": 0, "B_corr": 1,
            "R_executable": 0, "R_param": 2/6,
            "S_skel": 0, "S_auto": 2, "S_mut": 0,
            "rationale": "BV formula missing; EM missing; ATR stop and O_{t+1} execution valid.",
        },
        "h2_d7_jump_diffusion_s44_C10": {
            "S_comp": 0, "V_math": 0, "B_corr": 0,
            "R_executable": 0, "R_param": 0/6,
            "S_skel": 0, "S_auto": 0, "S_mut": 0,
            "rationale": "BV formula missing; EM missing; execution at bar close C_t without delay buffer.",
        },
        "h2_d7_jump_diffusion_s44_C01": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 6/6,
            "S_skel": 5, "S_auto": 1, "S_mut": 0,
            "rationale": "Decoupled local jump detector and Hamilton 2-state log-RV EM model inherited perfectly; ATR stop and O_{t+1} bound.",
        },
        "h2_d7_jump_diffusion_s44_C11": {
            "S_comp": 1, "V_math": 1, "B_corr": 1,
            "R_executable": 1, "R_param": 6/6,
            "S_skel": 5, "S_auto": 1, "S_mut": 0,
            "rationale": "Decoupled local jump detector and Hamilton 2-state log-RV EM model inherited perfectly; ATR stop and O_{t+1} bound.",
        },
    }

    # -------------------------------------------------------------------------
    # 3. Compile Audit Ledger & Summary
    # -------------------------------------------------------------------------
    # Check saturation rate from raw_blocks
    total_calls = 0
    length_finish_count = 0
    token_usage_by_cond = {"C00": [], "C10": [], "C01": [], "C11": []}

    for b in raw_blocks:
        for c in ["C00", "C10", "C01", "C11"]:
            diag = b["diagnostics"][c]
            for call_type in ["initial_call", "patch_call"]:
                total_calls += 1
                fin = diag[call_type]["finish_reason"]
                if fin == "length":
                    length_finish_count += 1
            token_usage_by_cond[c].append(diag["initial_call"]["out_tokens"] + diag["patch_call"]["out_tokens"])

    saturation_rate = length_finish_count / total_calls if total_calls > 0 else 0.0

    # Aggregate Track A & Track B per condition
    summary_track_a = {"C00": {"M": 0, "D": 0, "S": 0, "V": 0},
                       "C10": {"M": 0, "D": 0, "S": 0, "V": 0},
                       "C01": {"M": 0, "D": 0, "S": 0, "V": 0},
                       "C11": {"M": 0, "D": 0, "S": 0, "V": 0}}

    summary_track_b = {"C00": {"S_comp": 0, "V_math": 0, "B_corr": 0, "R_exec": 0, "R_param_sum": 0.0, "S_skel": 0, "S_auto": 0, "S_mut": 0},
                       "C10": {"S_comp": 0, "V_math": 0, "B_corr": 0, "R_exec": 0, "R_param_sum": 0.0, "S_skel": 0, "S_auto": 0, "S_mut": 0},
                       "C01": {"S_comp": 0, "V_math": 0, "B_corr": 0, "R_exec": 0, "R_param_sum": 0.0, "S_skel": 0, "S_auto": 0, "S_mut": 0},
                       "C11": {"S_comp": 0, "V_math": 0, "B_corr": 0, "R_exec": 0, "R_param_sum": 0.0, "S_skel": 0, "S_auto": 0, "S_mut": 0}}

    for key, val_a in ratings_track_a.items():
        cond = key.split("_")[-1]
        for p, scores in val_a.items():
            summary_track_a[cond]["M"] += scores[0]
            summary_track_a[cond]["D"] += scores[1]
            summary_track_a[cond]["S"] += scores[2]
            summary_track_a[cond]["V"] += scores[3]

    for key, val_b in ratings_track_b.items():
        cond = key.split("_")[-1]
        summary_track_b[cond]["S_comp"] += val_b["S_comp"]
        summary_track_b[cond]["V_math"] += val_b["V_math"]
        summary_track_b[cond]["B_corr"] += val_b["B_corr"]
        summary_track_b[cond]["R_exec"] += val_b["R_executable"]
        summary_track_b[cond]["R_param_sum"] += val_b["R_param"]
        summary_track_b[cond]["S_skel"] += val_b["S_skel"]
        summary_track_b[cond]["S_auto"] += val_b["S_auto"]
        summary_track_b[cond]["S_mut"] += val_b["S_mut"]

    # Block-by-block differences
    blocks_analysis = []
    for b in raw_blocks:
        bid = b["block_id"]
        # extract scores for each condition in this block
        y_exec = {}
        y_param = {}
        y_spec = {}
        for cond in ["C00", "C10", "C01", "C11"]:
            series_key = f"{bid}_{cond}"
            rb = ratings_track_b[series_key]
            y_exec[cond] = rb["R_executable"]
            y_param[cond] = rb["R_param"]
            # count specified
            y_spec[cond] = sum(ratings_track_a[series_key][p][2] for p in ratings_track_a[series_key])

        # Delta formulas:
        # Delta_cap = ([Y(C10) - Y(C00)] + [Y(C11) - Y(C01)]) / 2
        # Delta_skel = ([Y(C01) - Y(C00)] + [Y(C11) - Y(C10)]) / 2
        # Delta_inter = [Y(C11) - Y(C01)] - [Y(C10) - Y(C00)]

        def calc_deltas(y):
            d_cap = ((y["C10"] - y["C00"]) + (y["C11"] - y["C01"])) / 2.0
            d_skel = ((y["C01"] - y["C00"]) + (y["C11"] - y["C10"])) / 2.0
            d_inter = (y["C11"] - y["C01"]) - (y["C10"] - y["C00"])
            return d_cap, d_skel, d_inter

        d_cap_exec, d_skel_exec, d_inter_exec = calc_deltas(y_exec)
        d_cap_param, d_skel_param, d_inter_param = calc_deltas(y_param)
        d_cap_spec, d_skel_spec, d_inter_spec = calc_deltas(y_spec)

        blocks_analysis.append({
            "block_id": bid,
            "concept_id": b["concept_id"],
            "seed": b["seed"],
            "y_exec": y_exec,
            "y_param": y_param,
            "y_spec": y_spec,
            "deltas_exec": {"d_cap": d_cap_exec, "d_skel": d_skel_exec, "d_inter": d_inter_exec},
            "deltas_param": {"d_cap": d_cap_param, "d_skel": d_skel_param, "d_inter": d_inter_param},
            "deltas_spec": {"d_cap": d_cap_spec, "d_skel": d_skel_spec, "d_inter": d_inter_spec},
        })

    audit_payload = {
        "metadata": {
            "total_blocks": len(raw_blocks),
            "total_series": len(raw_blocks) * 4,
            "total_api_calls": total_calls,
            "saturation_rate": saturation_rate,
            "token_usage_averages": {c: sum(vals)/len(vals) for c, vals in token_usage_by_cond.items()},
        },
        "summary_track_a": summary_track_a,
        "summary_track_b": summary_track_b,
        "blocks_analysis": blocks_analysis,
        "ratings_track_a": ratings_track_a,
        "ratings_track_b": ratings_track_b,
    }

    with open(OUTPUT_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2, ensure_ascii=False)

    print("================================================================================")
    print("H2-D7 Blinded Multi-Tier Audit Completed Successfully!")
    print(f"Saturation Rate: {saturation_rate * 100:.2f}% ({length_finish_count}/{total_calls} calls hit max_tokens)")
    print("\nTrack A (P1-Compatible Gold Specified count out of 36):")
    for c in ["C00", "C10", "C01", "C11"]:
        s_cnt = summary_track_a[c]["S"]
        print(f"  {c}: Specified = {s_cnt}/36 ({s_cnt/36*100:.2f}%)")

    print("\nTrack B (D7-Specific Model Score out of 6 blocks):")
    for c in ["C00", "C10", "C01", "C11"]:
        r_ex = summary_track_b[c]["R_exec"]
        r_pm = summary_track_b[c]["R_param_sum"] / 6.0
        skel = summary_track_b[c]["S_skel"]
        auto = summary_track_b[c]["S_auto"]
        mut = summary_track_b[c]["S_mut"]
        print(f"  {c}: R_executable = {r_ex}/6 ({r_ex/6*100:.2f}%), R_param = {r_pm*100:.2f}% | S_skel={skel}, S_auto={auto}, S_mut={mut}")

    print("\nAverage Factorial Effects across 6 Blocks (Primary Metric: R_executable):")
    mean_d_cap_exec = sum(b["deltas_exec"]["d_cap"] for b in blocks_analysis) / 6.0
    mean_d_skel_exec = sum(b["deltas_exec"]["d_skel"] for b in blocks_analysis) / 6.0
    mean_d_inter_exec = sum(b["deltas_exec"]["d_inter"] for b in blocks_analysis) / 6.0
    print(f"  Delta_capacity: {mean_d_cap_exec:+.4f} ({mean_d_cap_exec*100:+.2f}%)")
    print(f"  Delta_skeleton: {mean_d_skel_exec:+.4f} ({mean_d_skel_exec*100:+.2f}%)")
    print(f"  Delta_interaction: {mean_d_inter_exec:+.4f} ({mean_d_inter_exec*100:+.2f}%)")

    print("\nAverage Factorial Effects across 6 Blocks (Secondary Metric: R_param):")
    mean_d_cap_param = sum(b["deltas_param"]["d_cap"] for b in blocks_analysis) / 6.0
    mean_d_skel_param = sum(b["deltas_param"]["d_skel"] for b in blocks_analysis) / 6.0
    mean_d_inter_param = sum(b["deltas_param"]["d_inter"] for b in blocks_analysis) / 6.0
    print(f"  Delta_capacity: {mean_d_cap_param:+.4f} ({mean_d_cap_param*100:+.2f}%)")
    print(f"  Delta_skeleton: {mean_d_skel_param:+.4f} ({mean_d_skel_param*100:+.2f}%)")
    print(f"  Delta_interaction: {mean_d_inter_param:+.4f} ({mean_d_inter_param*100:+.2f}%)")
    print("================================================================================")


if __name__ == "__main__":
    audit_d7()
