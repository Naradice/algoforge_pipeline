"""
Script: audit_h2_d6_p1_results.py
Performs the blinded multi-tier audit and 2x2 factorial disentanglement analysis
for H2-D6 P1 under the frozen v0.7 criteria.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
DATA_DIR = BASE_DIR / "data"
GOLD_DIR = BASE_DIR / "tests" / "fixtures" / "gold_standard"

OUTPUT_AUDIT_JSON = DATA_DIR / "h2_d6_p1_audit_results.json"
RAW_CANDIDATES_PATH = DATA_DIR / "h2_d6_p1_raw_candidates.json"
GOLD_PARAMS_PATH = GOLD_DIR / "h2_d6_gold_parameters.json"


def audit_p1():
    with open(RAW_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        raw_blocks = json.load(f)

    with open(GOLD_PARAMS_PATH, "r", encoding="utf-8") as f:
        gold_params_db = json.load(f)

    # 144 cells ledger (6 blocks x 4 conditions x 6 parameters)
    ledger_entries = []

    # Detailed parameter evaluation mapping
    # Hawkes parameters:
    # p1: mu, p2: alpha, p3: beta, p4: lambda*, p5: t_exec, p6: max_dd
    # Jump parameters:
    # p1: theta_BV, p2: p_trans, p3: sigma_ratio, p4: t_exec, p5: stop_bps, p6: J*

    ratings = {
        # ---------------- Hawkes s42 ----------------
        "h2_d6_p1_hawkes_process_s42_A0": {
            "mu": (1, 0, 0, 0, "Mentioned in exit as baseline mu, but no window or estimation formula"),
            "alpha": (0, 0, 0, 0, "Not mentioned in candidate text"),
            "beta": (0, 0, 0, 0, "Not mentioned in candidate text"),
            "lambda_star": (1, 1, 1, 0, "75th percentile of historical lambda over past 100 bars"),
            "t_exec": (1, 1, 1, 0, "Execute long position at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "Fixed stop-loss at 30 pips below entry, no dynamic ATR formula"),
        },
        "h2_d6_p1_hawkes_process_s42_A1": {
            "mu": (1, 0, 0, 0, "Mentioned in exit as baseline mu, no window"),
            "alpha": (1, 1, 0, 0, "MLE on historical event data with bounds [0.1, 1.0], but window and likelihood formula missing"),
            "beta": (1, 1, 0, 0, "Method of moments with bounds [0.01, 0.1], window missing"),
            "lambda_star": (1, 0, 0, 0, "Threshold derived from estimated intensity, percentile window dropped in patch"),
            "t_exec": (1, 1, 1, 0, "Execution at next available bar after condition met (O_{t+1})"),
            "max_dd": (1, 1, 0, 0, "Max drawdown stop at 2% / stop-loss 30 pips, fixed constant"),
        },
        "h2_d6_p1_hawkes_process_s42_B0": {
            "mu": (0, 0, 0, 0, "Not mentioned"),
            "alpha": (0, 0, 0, 0, "Not mentioned"),
            "beta": (0, 0, 0, 0, "Not mentioned"),
            "lambda_star": (1, 1, 0, 0, "Predefined threshold e.g. 0.005 uncalibrated constant"),
            "t_exec": (1, 0, 0, 0, "No execution timing ordering specified"),
            "max_dd": (1, 1, 1, 0, "1.5 times ATR of last 10 M5 bars, recalculated every 10 bars"),
        },
        "h2_d6_p1_hawkes_process_s42_B1": {
            "mu": (0, 0, 0, 0, "Not mentioned"),
            "alpha": (1, 1, 0, 0, "MLE grid search [0.01, 0.1] 1000 obs, but event definition / likelihood missing in B"),
            "beta": (1, 1, 0, 0, "MLE grid search [0.1, 1.0] 1000 obs, event definition / likelihood missing"),
            "lambda_star": (1, 1, 0, 0, "Predefined threshold e.g. 0.005 uncalibrated constant"),
            "t_exec": (1, 1, 1, 0, "Execute trades at close of M5 bar where entry condition met"),
            "max_dd": (1, 1, 1, 0, "1.5 times ATR of last 10 M5 bars"),
        },

        # ---------------- Hawkes s43 ----------------
        "h2_d6_p1_hawkes_process_s43_A0": {
            "mu": (1, 1, 0, 0, "MLE bounds [0.0001, 0.01], 20 bar window insufficient for 10 events, discrete likelihood missing"),
            "alpha": (1, 1, 0, 0, "MLE bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE bounds [0.01, 1.0], discrete likelihood missing"),
            "lambda_star": (1, 1, 0, 0, "Threshold e.g. 0.005 uncalibrated constant"),
            "t_exec": (1, 1, 1, 0, "Enter long position at O_{t+1}"),
            "max_dd": (1, 1, 0, 0, "Stop-loss of 1% fixed constant"),
        },
        "h2_d6_p1_hawkes_process_s43_A1": {
            "mu": (1, 1, 0, 0, "MLE bounds [0.0001, 0.01], discrete likelihood missing"),
            "alpha": (1, 1, 0, 0, "MLE bounds [0.01, 0.5], discrete likelihood missing"),
            "beta": (1, 1, 0, 0, "MLE bounds [0.01, 1.0], discrete likelihood missing"),
            "lambda_star": (1, 1, 0, 0, "Threshold e.g. 0.005 uncalibrated constant"),
            "t_exec": (1, 1, 1, 0, "Enter long position at O_{t+1}, open of next bar"),
            "max_dd": (1, 1, 0, 0, "MaxDrawdown 0.05 / 2% fixed fraction"),
        },
        "h2_d6_p1_hawkes_process_s43_B0": {
            "mu": (0, 0, 0, 0, "Not mentioned"),
            "alpha": (0, 0, 0, 0, "Not mentioned"),
            "beta": (0, 0, 0, 0, "Not mentioned"),
            "lambda_star": (1, 1, 0, 0, "Threshold e.g. 0.001 uncalibrated constant"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "max_dd": (1, 1, 0, 0, "Stop loss 20 pips fixed constant"),
        },
        "h2_d6_p1_hawkes_process_s43_B1": {
            "mu": (0, 0, 0, 0, "Not mentioned"),
            "alpha": (1, 1, 0, 0, "MLE bounds [0.01, 1.0] 1000 bars, but event definition missing in B"),
            "beta": (1, 1, 0, 0, "MLE bounds [0.01, 1.0] 1000 bars, event definition missing in B"),
            "lambda_star": (1, 1, 0, 0, "Threshold e.g. 0.001 uncalibrated constant"),
            "t_exec": (1, 1, 1, 0, "Executed at close of M5 bar"),
            "max_dd": (1, 1, 0, 0, "Stop loss 20 pips fixed constant"),
        },

        # ---------------- Hawkes s44 ----------------
        "h2_d6_p1_hawkes_process_s44_A0": {
            "mu": (0, 0, 0, 0, "Not mentioned"),
            "alpha": (0, 0, 0, 0, "Not mentioned"),
            "beta": (0, 0, 0, 0, "Not mentioned"),
            "lambda_star": (1, 1, 1, 0, "90th percentile of lambda(t) over past 100 bars"),
            "t_exec": (1, 1, 1, 0, "Enter long position at next bar open O_{t+1}"),
            "max_dd": (1, 1, 1, 0, "1.5 times ATR of last 100 bars, updated dynamically every 10 bars"),
        },
        "h2_d6_p1_hawkes_process_s44_A1": {
            "mu": (0, 0, 0, 0, "Not mentioned"),
            "alpha": (1, 1, 0, 0, "MLE on historical event data with bounds [0.1, 1.0], window missing"),
            "beta": (1, 1, 0, 0, "MLE on historical event data with bounds [0.01, 0.5], window missing"),
            "lambda_star": (1, 1, 1, 0, "90th percentile of lambda(t) over past 100 bars"),
            "t_exec": (1, 1, 1, 0, "Next bar open O_{t+1}"),
            "max_dd": (1, 1, 1, 0, "1.5 times ATR of last 100 bars"),
        },
        "h2_d6_p1_hawkes_process_s44_B0": {
            "mu": (0, 0, 0, 0, "Not mentioned"),
            "alpha": (0, 0, 0, 0, "Not mentioned"),
            "beta": (0, 0, 0, 0, "Not mentioned"),
            "lambda_star": (1, 1, 0, 0, "Threshold of 0.05 uncalibrated constant"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "max_dd": (1, 1, 0, 0, "StopLoss 0.01 fixed constant"),
        },
        "h2_d6_p1_hawkes_process_s44_B1": {
            "mu": (0, 0, 0, 0, "Not mentioned"),
            "alpha": (1, 1, 0, 0, "MLE bounds [0.01, 0.5], window missing"),
            "beta": (1, 1, 0, 0, "MLE bounds [0.01, 1.0], window missing"),
            "lambda_star": (1, 1, 0, 0, "Threshold of 0.05 uncalibrated constant"),
            "t_exec": (1, 1, 1, 0, "Execution timing at next M5 bar open"),
            "max_dd": (1, 1, 0, 0, "StopLoss 0.01 / MaxDrawdown 5% fixed constant"),
        },

        # ---------------- Jump-Diffusion s42 ----------------
        "h2_d6_p1_jump_diffusion_s42_A0": {
            "theta_bv": (1, 0, 0, 0, "BV_t defined over 288 bars, but truncation threshold theta_BV absent"),
            "p_trans": (0, 0, 0, 0, "Not mentioned"),
            "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
            "t_exec": (1, 0, 0, 0, "No execution timing rule specified"),
            "stop_bps": (1, 1, 0, 0, "Stop-loss of 1.5 times ATR, but ATR lookback window missing"),
            "j_star": (1, 1, 0, 0, "J_t > threshold calibrated based on historical data, formula missing"),
        },
        "h2_d6_p1_jump_diffusion_s42_A1": {
            "theta_bv": (1, 1, 1, 0, "theta_BV = 1.5 * mean(BV_t) over rolling window of 288 M5 bars"),
            "p_trans": (1, 1, 0, 0, "MLE over historical regime labels, window missing"),
            "sigma_ratio": (1, 1, 0, 0, "Ratio estimated over historical data, window missing"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "stop_bps": (1, 1, 1, 0, "stop_bps = 0.5 * ATR of last 10 M5 bars"),
            "j_star": (1, 1, 1, 0, "J_t > theta_BV tied to specified theta_BV threshold"),
        },
        "h2_d6_p1_jump_diffusion_s42_B0": {
            "theta_bv": (0, 0, 0, 0, "Not mentioned"),
            "p_trans": (0, 0, 0, 0, "Not mentioned"),
            "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "stop_bps": (1, 1, 1, 0, "1.5 times ATR of last 10 M5 bars"),
            "j_star": (1, 1, 0, 0, "J_t > 0.01 uncalibrated constant"),
        },
        "h2_d6_p1_jump_diffusion_s42_B1": {
            "theta_bv": (1, 1, 0, 0, "95th percentile of jump sizes on historical jump data, window missing"),
            "p_trans": (1, 1, 0, 0, "MLE on regime transitions over 1000 obs, but regime filter missing"),
            "sigma_ratio": (1, 1, 0, 0, "Ratio over 500 obs, regime separation method missing"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "stop_bps": (1, 1, 1, 0, "1.5 times ATR of last 10 M5 bars"),
            "j_star": (1, 1, 0, 0, "Threshold e.g. 0.01 uncalibrated constant"),
        },

        # ---------------- Jump-Diffusion s43 ----------------
        "h2_d6_p1_jump_diffusion_s43_A0": {
            "theta_bv": (0, 0, 0, 0, "Not mentioned"),
            "p_trans": (0, 0, 0, 0, "Not mentioned"),
            "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "stop_bps": (1, 1, 0, 0, "20 pips / 1% fixed constant"),
            "j_star": (1, 1, 0, 0, "J_t > 0.05 uncalibrated constant"),
        },
        "h2_d6_p1_jump_diffusion_s43_A1": {
            "theta_bv": (1, 1, 0, 0, "95th percentile of BV during tranquil periods, window missing"),
            "p_trans": (1, 1, 0, 0, "Hidden Markov model on historical data, window missing"),
            "sigma_ratio": (1, 1, 0, 0, "Ratio over 100 obs, tranquil/crisis separation missing"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "stop_bps": (1, 1, 1, 0, "1.5 times ATR over last 14 periods"),
            "j_star": (1, 1, 0, 0, "J_t > theta_BV tied to unspecified theta_BV"),
        },
        "h2_d6_p1_jump_diffusion_s43_B0": {
            "theta_bv": (0, 0, 0, 0, "Not mentioned"),
            "p_trans": (0, 0, 0, 0, "Not mentioned"),
            "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
            "t_exec": (1, 1, 1, 0, "Execute at close of current bar C_t for next bar O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "1.5 times ATR of last 10 bars"),
            "j_star": (1, 1, 0, 0, "J_t > 0.01 uncalibrated constant"),
        },
        "h2_d6_p1_jump_diffusion_s43_B1": {
            "theta_bv": (1, 1, 0, 0, "Bounds [1.5, 3.0] empirical range without window or formula"),
            "p_trans": (1, 1, 0, 0, "MLE on historical regime data without window"),
            "sigma_ratio": (1, 1, 0, 0, "Theoretical range [0.5, 2.0] without window"),
            "t_exec": (1, 1, 1, 0, "Execute at close of current bar C_t for next bar O_{t+1}"),
            "stop_bps": (1, 1, 1, 0, "1.5 times ATR of last 10 bars (and trailing 20 bps adjusted by ATR)"),
            "j_star": (1, 1, 0, 0, "J_t > 0.01 uncalibrated constant"),
        },

        # ---------------- Jump-Diffusion s44 ----------------
        "h2_d6_p1_jump_diffusion_s44_A0": {
            "theta_bv": (0, 0, 0, 0, "Not mentioned"),
            "p_trans": (0, 0, 0, 0, "Not mentioned"),
            "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
            "t_exec": (1, 1, 0, 0, "Announcement bar close mentioned, but entry execution timing rule unspecified"),
            "stop_bps": (1, 1, 0, 0, "Stop-loss 1.5% fixed below entry"),
            "j_star": (1, 1, 0, 0, "J_t > 0.01 uncalibrated constant"),
        },
        "h2_d6_p1_jump_diffusion_s44_A1": {
            "theta_bv": (1, 1, 1, 0, "theta_BV = 1.5 * median(BV_t) over last 288 bars"),
            "p_trans": (1, 1, 0, 0, "MLE on regime-labeled data without window"),
            "sigma_ratio": (1, 1, 0, 0, "Rolling window 288 bars, but regime classification rule missing"),
            "t_exec": (1, 1, 1, 0, "t_exec = execute at open of bar following announcement"),
            "stop_bps": (1, 1, 1, 0, "stop_bps = 1.5 * volatility estimate from last 288 bars"),
            "j_star": (1, 1, 0, 0, "J_t > 0.01 uncalibrated constant"),
        },
        "h2_d6_p1_jump_diffusion_s44_B0": {
            "theta_bv": (0, 0, 0, 0, "Not mentioned"),
            "p_trans": (0, 0, 0, 0, "Not mentioned"),
            "sigma_ratio": (0, 0, 0, 0, "Not mentioned"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "stop_bps": (1, 1, 1, 0, "1.5 times ATR of last 10 M5 bars"),
            "j_star": (1, 1, 0, 0, "J_t > 0 uncalibrated"),
        },
        "h2_d6_p1_jump_diffusion_s44_B1": {
            "theta_bv": (1, 1, 1, 0, "1.5 times median jump size over last 100 observations"),
            "p_trans": (1, 1, 0, 0, "Markov chain over last 500 observations, but regime filter missing"),
            "sigma_ratio": (1, 1, 0, 0, "Ratio over last 200 observations, regime separation missing"),
            "t_exec": (1, 0, 0, 0, "No execution timing specified"),
            "stop_bps": (1, 1, 1, 0, "1.5 times ATR of last 10 M5 bars"),
            "j_star": (1, 1, 0, 0, "J_t > 0 uncalibrated"),
        },
    }

    # Verify that all 24 series are present in ratings
    assert len(ratings) == 24, f"Expected 24 series, found {len(ratings)}"

    # Compile condition-level and concept-level metrics
    conditions = ["A0", "A1", "B0", "B1"]
    concepts = ["hawkes_process", "jump_diffusion"]

    series_summary = []
    condition_aggregates = {c: {"M": 0, "D": 0, "S": 0, "V": 0, "total_params": 0, "tokens": {"in": 0, "out": 0, "total": 0}, "cost_usd": 0.0} for c in conditions}
    concept_aggregates = {cid: {c: {"M": 0, "D": 0, "S": 0, "V": 0, "total_params": 0} for c in conditions} for cid in concepts}
    block_aggregates = {}

    # Category aggregates: model_estimation, decision_threshold, execution_timing, risk_stop
    category_aggregates = {
        "model_estimation": {c: {"M": 0, "D": 0, "S": 0, "V": 0, "total": 0} for c in conditions},
        "decision_threshold": {c: {"M": 0, "D": 0, "S": 0, "V": 0, "total": 0} for c in conditions},
        "execution_timing": {c: {"M": 0, "D": 0, "S": 0, "V": 0, "total": 0} for c in conditions},
        "risk_stop": {c: {"M": 0, "D": 0, "S": 0, "V": 0, "total": 0} for c in conditions},
    }

    for block in raw_blocks:
        bid = block["block_id"]
        cid = block["concept_id"]
        seed = block["seed"]
        block_aggregates[bid] = {}

        for cond in conditions:
            series_id = f"{bid}_{cond}"
            cand_ratings = ratings[series_id]
            diag = block["diagnostics"].get(cond, {})
            in_tok = diag.get("in_tokens", 0)
            out_tok = diag.get("out_tokens", 0)
            cost = diag.get("cost_usd", 0.0)

            # Cumulative tokens for conditions
            condition_aggregates[cond]["tokens"]["in"] += in_tok
            condition_aggregates[cond]["tokens"]["out"] += out_tok
            condition_aggregates[cond]["tokens"]["total"] += (in_tok + out_tok)
            condition_aggregates[cond]["cost_usd"] += cost

            m_sum = sum(v[0] for v in cand_ratings.values())
            d_sum = sum(v[1] for v in cand_ratings.values())
            s_sum = sum(v[2] for v in cand_ratings.values())
            v_sum = sum(v[3] for v in cand_ratings.values())
            n_params = len(cand_ratings)

            condition_aggregates[cond]["M"] += m_sum
            condition_aggregates[cond]["D"] += d_sum
            condition_aggregates[cond]["S"] += s_sum
            condition_aggregates[cond]["V"] += v_sum
            condition_aggregates[cond]["total_params"] += n_params

            concept_aggregates[cid][cond]["M"] += m_sum
            concept_aggregates[cid][cond]["D"] += d_sum
            concept_aggregates[cid][cond]["S"] += s_sum
            concept_aggregates[cid][cond]["V"] += v_sum
            concept_aggregates[cid][cond]["total_params"] += n_params

            block_aggregates[bid][cond] = {"M": m_sum, "D": d_sum, "S": s_sum, "V": v_sum}

            # Category mapping
            for p_name, (m, d, s, v, rationale) in cand_ratings.items():
                if p_name in ["mu", "alpha", "beta", "theta_bv", "p_trans", "sigma_ratio"]:
                    cat = "model_estimation"
                elif p_name in ["lambda_star", "j_star"]:
                    cat = "decision_threshold"
                elif p_name in ["t_exec"]:
                    cat = "execution_timing"
                elif p_name in ["max_dd", "stop_bps"]:
                    cat = "risk_stop"
                else:
                    raise ValueError(f"Unknown param: {p_name}")

                category_aggregates[cat][cond]["M"] += m
                category_aggregates[cat][cond]["D"] += d
                category_aggregates[cat][cond]["S"] += s
                category_aggregates[cat][cond]["V"] += v
                category_aggregates[cat][cond]["total"] += 1

            series_summary.append({
                "series_id": series_id,
                "block_id": bid,
                "concept_id": cid,
                "seed": seed,
                "condition": cond,
                "M": m_sum,
                "D": d_sum,
                "S": s_sum,
                "V": v_sum,
                "grounded": (s_sum == 6 and v_sum == 6),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "cost_usd": round(cost, 6),
                "parameter_ratings": {k: {"M": v[0], "D": v[1], "S": v[2], "V": v[3], "rationale": v[4]} for k, v in cand_ratings.items()}
            })

    # Calculations of main effects and interaction effects on Specified rate Y_spec
    # Primary metric: Y_spec = S / total_params
    y_spec = {c: condition_aggregates[c]["S"] / condition_aggregates[c]["total_params"] for c in conditions}
    y_mention = {c: condition_aggregates[c]["M"] / condition_aggregates[c]["total_params"] for c in conditions}
    y_disp = {c: condition_aggregates[c]["D"] / condition_aggregates[c]["total_params"] for c in conditions}

    delta_checklist_single = y_spec["A1"] - y_spec["A0"]
    delta_checklist_two = y_spec["B1"] - y_spec["B0"]
    delta_interaction = delta_checklist_two - delta_checklist_single
    delta_twostage_generic = y_spec["B0"] - y_spec["A0"]
    delta_twostage_checklist = y_spec["B1"] - y_spec["A1"]
    delta_twostage_overall = (y_spec["B0"] + y_spec["B1"]) / 2 - (y_spec["A0"] + y_spec["A1"]) / 2
    delta_checklist_overall = (y_spec["A1"] + y_spec["B1"]) / 2 - (y_spec["A0"] + y_spec["B0"]) / 2

    results = {
        "metadata": {
            "experiment": "H2-D6 P1: 2x2 Factorial Disentanglement",
            "protocol_version": "v0.7",
            "total_blocks": len(raw_blocks),
            "total_series": len(series_summary),
            "total_api_calls": 42,
        },
        "condition_aggregates": condition_aggregates,
        "concept_aggregates": concept_aggregates,
        "block_aggregates": block_aggregates,
        "category_aggregates": category_aggregates,
        "factorial_effects": {
            "y_spec": y_spec,
            "y_mention": y_mention,
            "y_disp": y_disp,
            "delta_checklist_single": round(delta_checklist_single, 6),
            "delta_checklist_two": round(delta_checklist_two, 6),
            "delta_interaction": round(delta_interaction, 6),
            "delta_twostage_generic": round(delta_twostage_generic, 6),
            "delta_twostage_checklist": round(delta_twostage_checklist, 6),
            "delta_twostage_overall": round(delta_twostage_overall, 6),
            "delta_checklist_overall": round(delta_checklist_overall, 6),
        },
        "series_summary": series_summary,
    }

    with open(OUTPUT_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("================================================================================")
    print("H2-D6 P1 Blinded Multi-Tier Audit Completed Successfully!")
    print("================================================================================")
    print(f"Total series evaluated: {len(series_summary)}")
    print("\n--- Condition Specified Rates (Y_spec) ---")
    for c in conditions:
        s_cnt = condition_aggregates[c]["S"]
        tot = condition_aggregates[c]["total_params"]
        pct = y_spec[c] * 100
        cost = condition_aggregates[c]["cost_usd"]
        tok = condition_aggregates[c]["tokens"]["total"]
        print(f"  {c}: {s_cnt}/{tot} ({pct:.2f}%) | Cost: ${cost:.6f} | Total Tokens: {tok}")

    print("\n--- Factorial Effects ---")
    print(f"  Checklist Main Effect in Single-Stage (A1 - A0): {delta_checklist_single*100:+.2f}%")
    print(f"  Checklist Main Effect in Two-Stage    (B1 - B0): {delta_checklist_two*100:+.2f}%")
    print(f"  Overall Checklist Effect:                        {delta_checklist_overall*100:+.2f}%")
    print(f"  Two-Stage Main Effect in Generic      (B0 - A0): {delta_twostage_generic*100:+.2f}%")
    print(f"  Two-Stage Main Effect in Checklist    (B1 - A1): {delta_twostage_checklist*100:+.2f}%")
    print(f"  Overall Two-Stage Effect:                        {delta_twostage_overall*100:+.2f}%")
    print(f"  Interaction Effect (Delta_two - Delta_single):   {delta_interaction*100:+.2f}%")

    print("\n--- Category Breakdown (Specified Counts) ---")
    for cat, cdata in category_aggregates.items():
        row_str = " | ".join([f"{c}: {cdata[c]['S']}/{cdata[c]['total']}" for c in conditions])
        print(f"  {cat:20s}: {row_str}")

    print(f"\nAudit results saved to: {OUTPUT_AUDIT_JSON}")
    print("================================================================================")


if __name__ == "__main__":
    audit_p1()
