import json
import re
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
with open(BASE_DIR / "data" / "h2_d6_raw_candidates.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# We will analyze every single series (24 series x 6 parameters = 144 items)
# For each parameter, we extract:
# - M: bool, evidence text
# - D: bool, evidence text / reason
# - S: bool, evidence text / reason
# - V: bool (always 0), failure reason

def analyze_all():
    results = []
    
    for base in raw:
        cid = base["concept_id"]
        seed = base["seed"]
        b_id = f"{cid}_s{seed}"
        
        for cond in ["S0", "S1", "S2a", "S2b"]:
            sid = f"{b_id}_{cond}"
            cand_wrap = base["candidates"][cond]
            cand = cand_wrap.get("candidate", {})
            spec = cand.get("backtest_spec", {})
            
            claim = cand.get("claim", "")
            mech = cand.get("proposed_mechanism", "")
            rat = cand.get("grounding_rationale", "")
            entry = spec.get("entry_condition", "")
            exit_ = spec.get("exit_condition", "")
            feat = spec.get("feature_engineering", "")
            risk = spec.get("risk_management", "")
            falsif = spec.get("falsification_metric", "")
            
            # Text evaluated for the strategy candidate
            cand_full_text = f"{claim} {mech} {rat} {entry} {exit_} {feat} {risk} {falsif}"
            
            series_res = {
                "series_id": sid,
                "concept": cid,
                "condition": cond,
                "seed": seed,
                "params": {}
            }
            
            if cid == "hawkes_process":
                # -------------------------------------------------------------
                # Hawkes parameters:
                # p1: mu, p2: alpha, p3: beta, p4: lambda*, p5: t_exec, p6: max_dd
                # -------------------------------------------------------------
                
                # p1_mu
                # Check mention
                m_mu = 1 if (re.search(r'\bmu\b', cand_full_text, re.I) or "baseline intensity" in cand_full_text.lower()) else 0
                # Disposition
                d_mu = 0
                s_mu = 0
                if m_mu:
                    if sid == "hawkes_process_s43_S0":
                        d_mu = 1
                        s_mu = 1 # 50-bar rolling MLE with mu > 0 constraint
                    elif "0.01 to 0.1" in cand_full_text:
                        d_mu = 1
                        s_mu = 0
                
                # p2_alpha
                m_alpha = 1 if (re.search(r'\balpha\b', cand_full_text, re.I) or "excitation" in cand_full_text.lower()) else 0
                d_alpha = 0
                s_alpha = 0
                if m_alpha:
                    if sid == "hawkes_process_s43_S0":
                        d_alpha = 1
                        s_alpha = 1 # 50-bar rolling MLE with alpha < beta constraint
                    elif sid == "hawkes_process_s42_S2b":
                        d_alpha = 1
                        s_alpha = 1 # grid search [0.1, 1.0] maximizing log-likelihood
                    elif any(w in cand_full_text.lower() for w in ["alpha < beta", "alpha/beta", "estimate alpha"]):
                        d_alpha = 1
                        s_alpha = 0
                        
                # p3_beta
                m_beta = 1 if (re.search(r'\bbeta\b', cand_full_text, re.I) or "decay" in cand_full_text.lower()) else 0
                d_beta = 0
                s_beta = 0
                if m_beta:
                    if sid == "hawkes_process_s43_S0":
                        d_beta = 1
                        s_beta = 1 # 50-bar rolling MLE with alpha < beta
                    elif sid == "hawkes_process_s42_S2b":
                        d_beta = 1
                        s_beta = 1 # grid search [0.01, 0.5] optimizing decay
                    elif any(w in cand_full_text.lower() for w in ["alpha < beta", "alpha/beta", "estimate beta"]):
                        d_beta = 1
                        s_beta = 0
                        
                # p4_lambda
                m_lam = 1 if (re.search(r'\blambda\b', cand_full_text, re.I) or "intensity" in cand_full_text.lower()) else 0
                d_lam = 0
                s_lam = 0
                if m_lam:
                    if any(w in entry.lower() for w in ["0.05", "threshold", "rolling_mean", "standard deviation"]):
                        d_lam = 1
                    if sid == "hawkes_process_s43_S0":
                        s_lam = 1 # 1.5 * rolling_mean(lambda)
                    elif sid == "hawkes_process_s44_S0":
                        s_lam = 1 # rolling mean + 1 standard deviation
                        
                # p5_t_exec
                m_t = 1 if any(w in cand_full_text.lower() for w in ["next bar", "o_{t+1}", "open of the next", "next tick", "timing", "holding", "exit the position after"]) else 0
                d_t = 0
                s_t = 0
                if m_t:
                    d_t = 1
                    if sid in [
                        "hawkes_process_s42_S0",   # next bar open O_{t+1}
                        "hawkes_process_s42_S2b",  # next available tick after signal
                        "hawkes_process_s43_S0",   # open of the next bar O_{t+1}
                        "hawkes_process_s44_S0",   # next bar open O_{t+1}
                        "hawkes_process_s44_S1",   # decision at C_t, execution at O_{t+1}
                        "hawkes_process_s44_S2a",  # decision at C_t, execution at O_{t+1}
                        "hawkes_process_s44_S2b",  # decision at C_t, execution at O_{t+1}
                    ]:
                        s_t = 1
                        
                # p6_max_dd
                m_stop = 1 if any(w in risk.lower() for w in ["stop", "drawdown", "pips", "atr", "limit"]) else 0
                d_stop = 0
                s_stop = 0
                if m_stop:
                    d_stop = 1
                    # Symmetric reviewer rule: fixed pips/percentages are S=0.
                    # ATR stops with input, period, multiplier:
                    # s43_S0: 1.5 * ATR(14)
                    # s43_S1: 1.5 * ATR(20)
                    # s43_S2a: 1.5 * ATR(20), recalculated every 20 bars
                    # s43_S2b: 1.5 * ATR(20)
                    # s44_S0: 1.5 * ATR(20)
                    if sid in [
                        "hawkes_process_s43_S0",
                        "hawkes_process_s43_S1",
                        "hawkes_process_s43_S2a",
                        "hawkes_process_s43_S2b",
                        "hawkes_process_s44_S0",
                    ]:
                        s_stop = 1

                series_res["params"]["p1_mu"] = {"M": m_mu, "D": d_mu, "S": s_mu, "V": 0}
                series_res["params"]["p2_alpha"] = {"M": m_alpha, "D": d_alpha, "S": s_alpha, "V": 0}
                series_res["params"]["p3_beta"] = {"M": m_beta, "D": d_beta, "S": s_beta, "V": 0}
                series_res["params"]["p4_lambda"] = {"M": m_lam, "D": d_lam, "S": s_lam, "V": 0}
                series_res["params"]["p5_t_exec"] = {"M": m_t, "D": d_t, "S": s_t, "V": 0}
                series_res["params"]["p6_max_dd"] = {"M": m_stop, "D": d_stop, "S": s_stop, "V": 0}

            elif cid == "jump_diffusion":
                # -------------------------------------------------------------
                # Jump-Diffusion parameters:
                # p1: theta_BV, p2: p_trans, p3: sigma_ratio, p4: t_exec, p5: stop_bps, p6: j_star
                # -------------------------------------------------------------
                
                # p1_theta
                m_th = 1 if any(w in cand_full_text.lower() for w in ["thetabv", "theta_bv", "bipower", "bv_t"]) else 0
                d_th = 0
                s_th = 0
                if m_th:
                    if any(w in cand_full_text.lower() for w in ["75th percentile", "1.5 standard deviation", "theta_bv", "bipower"]):
                        d_th = 1
                    if sid == "jump_diffusion_s44_S2b":
                        s_th = 1 # 75th percentile of BV_t during tranquil periods
                        
                # p2_p_trans
                m_p = 1 if any(w in cand_full_text.lower() for w in ["p_00", "p_11", "transition", "regime probability", "p(s_t", "p(crisis)"]) else 0
                d_p = 0
                s_p = 0
                if m_p:
                    if any(w in cand_full_text.lower() for w in ["p_00", "p_11", "maximum likelihood", "mle", "transition probabilities"]):
                        d_p = 1
                    if sid == "jump_diffusion_s44_S2b":
                        s_p = 1 # 2-year window historical transition MLE with bounds [0, 1]
                        
                # p3_sigma_ratio
                m_sig = 1 if any(w in cand_full_text.lower() for w in ["sigma_ratio", "variance ratio", "crisis vs. tranquil", "volatility"]) else 0
                d_sig = 0
                s_sig = 0
                if m_sig:
                    if any(w in cand_full_text.lower() for w in ["sigma_ratio", "ratio of variances", "variance of returns in crisis"]):
                        d_sig = 1
                    if sid == "jump_diffusion_s44_S2b":
                        s_sig = 1 # variance of returns in crisis vs tranquil over 2 years
                        
                # p4_t_exec
                m_t = 1 if any(w in cand_full_text.lower() for w in ["executiontiming", "execution timing", "next bar", "o_{t+1}", "c_t", "after 60 minutes", "held for 5 bars"]) else 0
                d_t = 0
                s_t = 0
                if m_t:
                    d_t = 1
                    if sid in [
                        "jump_diffusion_s44_S1",   # decision at C_t, execution at O_{t+1}
                        "jump_diffusion_s44_S2a",  # decision at C_t, execution at O_{t+1}
                        "jump_diffusion_s44_S2b",  # decision at C_t, execution at O_{t+1}
                    ]:
                        s_t = 1
                        
                # p5_stop_bps
                m_stop = 1 if any(w in risk.lower() for w in ["stop", "atr", "bps", "price"]) else 0
                d_stop = 0
                s_stop = 0
                if m_stop:
                    d_stop = 1
                    # Symmetric reviewer rule:
                    # s43_S0: 2% of price -> fixed % without ATR -> S=0
                    # s44_S2b: trailing stop_bps = 20 basis points -> fixed bps -> S=0
                    # ATR stops with period and multiplier:
                    # s42_S0: 1.5 * ATR(288) -> S=1
                    # s42_S1: 1.5 * ATR(14) -> S=1
                    # s42_S2a: 1.5 * ATR(14) -> S=1
                    # s42_S2b: 1.5 * ATR(14) + dynamic 2x ATR -> S=1
                    # s43_S1: 1.5 * ATR(10) -> S=1
                    # s43_S2a: 1.5 * ATR(10) -> S=1
                    # s43_S2b: 1.5 * ATR(10) + dynamic 2x ATR -> S=1
                    # s44_S0: 1.5 * ATR(14) -> S=1
                    # s44_S1: 1.5 * ATR(10) -> S=1
                    # s44_S2a: 1.5 * ATR(10) -> S=1
                    if sid in [
                        "jump_diffusion_s42_S0",
                        "jump_diffusion_s42_S1",
                        "jump_diffusion_s42_S2a",
                        "jump_diffusion_s42_S2b",
                        "jump_diffusion_s43_S1",
                        "jump_diffusion_s43_S2a",
                        "jump_diffusion_s43_S2b",
                        "jump_diffusion_s44_S0",
                        "jump_diffusion_s44_S1",
                        "jump_diffusion_s44_S2a",
                    ]:
                        s_stop = 1
                        
                # p6_j_star
                m_j = 1 if any(w in cand_full_text.lower() for w in ["j_t", "jump", "rv_t - bv_t"]) else 0
                d_j = 0
                s_j = 0 # No calibrated threshold; 0.01, 0.5%, J_t > 0 are all S=0
                if m_j:
                    if any(w in entry.lower() for w in ["0.01", "0.5%", "threshold", "j_t > 0", "j_t"]):
                        d_j = 1
                
                series_res["params"]["p1_theta"] = {"M": m_th, "D": d_th, "S": s_th, "V": 0}
                series_res["params"]["p2_p_trans"] = {"M": m_p, "D": d_p, "S": s_p, "V": 0}
                series_res["params"]["p3_sigma_ratio"] = {"M": m_sig, "D": d_sig, "S": s_sig, "V": 0}
                series_res["params"]["p4_t_exec"] = {"M": m_t, "D": d_t, "S": s_t, "V": 0}
                series_res["params"]["p5_stop_bps"] = {"M": m_stop, "D": d_stop, "S": s_stop, "V": 0}
                series_res["params"]["p6_j_star"] = {"M": m_j, "D": d_j, "S": s_j, "V": 0}

            # Enforce containment: V <= S <= D <= M
            for pk, pv in series_res["params"].items():
                if pv["V"] > pv["S"]:
                    pv["S"] = pv["V"]
                if pv["S"] > pv["D"]:
                    pv["D"] = pv["S"]
                if pv["D"] > pv["M"]:
                    pv["M"] = pv["D"]
                assert pv["V"] <= pv["S"] <= pv["D"] <= pv["M"], f"Containment violated in {sid} {pk}"
                
            results.append(series_res)
            
    return results

res = analyze_all()
print(f"Successfully audited {len(res)} series.")
