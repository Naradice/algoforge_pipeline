import json
import re
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
with open(BASE_DIR / "data" / "h2_d6_raw_candidates.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

def audit_series():
    ledger = {}
    
    for base in raw:
        cid = base["concept_id"]
        seed = base["seed"]
        b_id = f"{cid}_s{seed}"
        
        for cond in ["S0", "S1", "S2a", "S2b"]:
            sid = f"{b_id}_{cond}"
            cand_wrap = base["candidates"][cond]
            cand = cand_wrap.get("candidate", {})
            spec = cand.get("backtest_spec", {})
            stage1 = cand_wrap.get("stage1_model_spec", {})
            
            # The prompt in S2a/S2b passes s1 candidate spec, and patch updates backtest_spec
            # For evaluating candidate:
            full_text = (json.dumps(cand, ensure_ascii=False) + " " + json.dumps(stage1, ensure_ascii=False)).lower()
            spec_text = json.dumps(spec, ensure_ascii=False).lower()
            entry_text = (spec.get("entry_condition") or "").lower()
            exit_text = (spec.get("exit_condition") or "").lower()
            risk_text = (spec.get("risk_management") or "").lower()
            feat_text = (spec.get("feature_engineering") or "").lower()
            
            params = {}
            
            if cid == "hawkes_process":
                # p1: mu (baseline intensity)
                # Mention: \bmu\b or baseline intensity
                p1_m = 1 if (re.search(r'\bmu\b', full_text) or "baseline intensity" in full_text) else 0
                # Disposition: value, bounds, or estimation stated
                p1_d = 1 if (p1_m and any(w in full_text for w in ["mu > 0", "mu:", "0.01 to 0.1", "50", "mle", "baseline"])) else 0
                # Specified: concrete window + bounds/mle
                # s43_S0 had 50-bar rolling MLE with mu > 0
                p1_s = 1 if (sid == "hawkes_process_s43_S0") else 0
                params["p1_mu"] = {"M": p1_m, "D": p1_d, "S": p1_s, "V": 0}
                
                # p2: alpha (excitation)
                p2_m = 1 if (re.search(r'\balpha\b', full_text) or "excitation" in full_text) else 0
                p2_d = 1 if (p2_m and any(w in full_text for w in ["alpha < beta", "grid search", "mle", "alpha:"])) else 0
                # Specified: s43_S0 (50 bar window, stationary bound), s42_S2b (grid search [0.1, 1.0] log-likelihood)
                p2_s = 1 if (sid in ["hawkes_process_s43_S0", "hawkes_process_s42_S2b"]) else 0
                params["p2_alpha"] = {"M": p2_m, "D": p2_d, "S": p2_s, "V": 0}
                
                # p3: beta (decay)
                p3_m = 1 if (re.search(r'\bbeta\b', full_text) or "decay" in full_text) else 0
                p3_d = 1 if (p3_m and any(w in full_text for w in ["alpha < beta", "grid search", "mle", "beta:"])) else 0
                p3_s = 1 if (sid in ["hawkes_process_s43_S0", "hawkes_process_s42_S2b"]) else 0
                params["p3_beta"] = {"M": p3_m, "D": p3_d, "S": p3_s, "V": 0}
                
                # p4: lambda* (intensity threshold)
                p4_m = 1 if ("lambda" in full_text or "intensity" in full_text) else 0
                p4_d = 1 if (p4_m and any(w in entry_text for w in ["0.05", "threshold", "rolling_mean", "standard deviation"])) else 0
                # Specified: Must have objective calibration (rolling_mean in s43_S0, rolling mean + 1 std in s44_S0). Constants like 0.05 are S=0.
                p4_s = 1 if (sid in ["hawkes_process_s43_S0", "hawkes_process_s44_S0"]) else 0
                params["p4_lambda"] = {"M": p4_m, "D": p4_d, "S": p4_s, "V": 0}
                
                # p5: t_exec (causal execution timing)
                p5_m = 1 if (any(w in full_text for w in ["execution timing", "next bar", "o_{t+1}", "open of the next", "next tick", "after the signal", "holding", "exit the position after"])) else 0
                p5_d = 1 if (p5_m and any(w in (entry_text + " " + exit_text) for w in ["o_{t+1}", "next bar", "next tick", "after 3", "after 5", "after 10", "after 20"])) else 0
                # Specified: causal entry ordering required (e.g. O_{t+1}, next tick). Holding horizon / exit time only is S=0.
                p5_s = 1 if (sid in [
                    "hawkes_process_s42_S0", "hawkes_process_s42_S2b",
                    "hawkes_process_s43_S0",
                    "hawkes_process_s44_S0", "hawkes_process_s44_S1", "hawkes_process_s44_S2a", "hawkes_process_s44_S2b"
                ]) else 0
                params["p5_t_exec"] = {"M": p5_m, "D": p5_d, "S": p5_s, "V": 0}
                
                # p6: max_dd (risk stop loss)
                p6_m = 1 if (any(w in risk_text for w in ["stop", "drawdown", "pips", "atr"])) else 0
                p6_d = 1 if (p6_m and any(w in risk_text for w in ["20 pips", "50 pips", "atr", "2%", "5%"])) else 0
                # Specified: Reviewer rule: Fixed stops (20 pips, 50 pips, 2%, 5%) without ATR/volatility calculation rule are S=0!
                # ATR-linked with period and multiplier:
                # s43_S0: 1.5 * ATR(14) -> S=1
                # s43_S1: 1.5 * ATR(20) -> S=1
                # s43_S2a: 1.5 * ATR(20) -> S=1
                # s43_S2b: 1.5 * ATR(20) -> S=1
                # s44_S0: 1.5 * ATR(20) -> S=1
                # All other hawkes series have fixed pips (20 pips, 50 pips) or fixed 2%/5% limit -> S=0!
                p6_s = 1 if (sid in [
                    "hawkes_process_s43_S0", "hawkes_process_s43_S1", "hawkes_process_s43_S2a", "hawkes_process_s43_S2b",
                    "hawkes_process_s44_S0"
                ]) else 0
                params["p6_max_dd"] = {"M": p6_m, "D": p6_d, "S": p6_s, "V": 0}

            elif cid == "jump_diffusion":
                # p1: theta_BV (bipower threshold / estimation)
                p1_m = 1 if ("thetabv" in full_text or "theta_bv" in full_text or "bipower" in full_text) else 0
                p1_d = 1 if (p1_m and any(w in full_text for w in ["75th percentile", "1.5 standard deviation", "bv_t", "bipower"])) else 0
                # Specified: concrete calibration (s42_S2b: news announcement dependency -> S=0, s43_S2b: 1.5 std jump size -> S=0 ungrounded, s44_S2b: 75th percentile of BV_t tranquil -> S=1)
                p1_s = 1 if (sid == "jump_diffusion_s44_S2b") else 0
                params["p1_theta"] = {"M": p1_m, "D": p1_d, "S": p1_s, "V": 0}
                
                # p2: p_trans (p_00, p_11 transition probabilities)
                p2_m = 1 if ("p_00" in full_text or "p_11" in full_text or "transition" in full_text or "regime" in full_text) else 0
                p2_d = 1 if (p2_m and any(w in full_text for w in ["p_00", "p_11", "maximum likelihood", "mle", "transition probabilities"])) else 0
                # Specified: s44_S2b: 2-year window historical transition MLE -> S=1
                p2_s = 1 if (sid == "jump_diffusion_s44_S2b") else 0
                params["p2_p_trans"] = {"M": p2_m, "D": p2_d, "S": p2_s, "V": 0}
                
                # p3: sigma_ratio (regime variance ratio)
                p3_m = 1 if ("sigma_ratio" in full_text or "variance ratio" in full_text or "regime" in full_text) else 0
                p3_d = 1 if (p3_m and any(w in full_text for w in ["sigma_ratio", "ratio of variances", "crisis vs. tranquil"])) else 0
                # Specified: s44_S2b: variance ratio in crisis vs tranquil over 2-year window -> S=1
                p3_s = 1 if (sid == "jump_diffusion_s44_S2b") else 0
                params["p3_sigma_ratio"] = {"M": p3_m, "D": p3_d, "S": p3_s, "V": 0}
                
                # p4: t_exec (causal execution timing)
                p4_m = 1 if (any(w in full_text for w in ["executiontiming", "execution timing", "next bar", "o_{t+1}", "c_t", "after 60 minutes", "held for 5 bars"])) else 0
                p4_d = 1 if (p4_m and any(w in (entry_text + " " + exit_text) for w in ["o_{t+1}", "c_t", "60 minutes", "5 bars"])) else 0
                # Specified: s44_S1, s44_S2a, s44_S2b specify decision at C_t, execution at O_{t+1}.
                p4_s = 1 if (sid in ["jump_diffusion_s44_S1", "jump_diffusion_s44_S2a", "jump_diffusion_s44_S2b"]) else 0
                params["p4_t_exec"] = {"M": p4_m, "D": p4_d, "S": p4_s, "V": 0}
                
                # p5: stop_bps (risk stop loss)
                p5_m = 1 if (any(w in risk_text for w in ["stop", "atr", "bps", "price"])) else 0
                p5_d = 1 if (p5_m and any(w in risk_text for w in ["atr", "2%", "20 basis points", "stop_bps"])) else 0
                # Specified: Reviewer rule: Fixed stops (2% in s43_S0, 20 bps in s44_S2b) without ATR rule are S=0!
                # ATR-linked stop loss with period and multiplier:
                # s42_S0: 1.5x ATR(288) -> S=1
                # s42_S1: 1.5x ATR(14) -> S=1
                # s42_S2a: 1.5x ATR(14) -> S=1
                # s42_S2b: 1.5x ATR(14) + dynamic 2x ATR -> S=1
                # s43_S0: 2% of entry price -> Fixed % without ATR -> S=0 (CORRECTED per reviewer!)
                # s43_S1: 1.5x ATR(10) -> S=1
                # s43_S2a: 1.5x ATR(10) -> S=1
                # s43_S2b: 1.5x ATR(10) + dynamic 2x ATR -> S=1
                # s44_S0: 1.5x ATR(14) -> S=1
                # s44_S1: 1.5x ATR(10) -> S=1
                # s44_S2a: 1.5x ATR(10) -> S=1
                # s44_S2b: Has 1.5x ATR(10) base stop! (Plus a 20 bps trailing). Since base stop is 1.5 ATR(10), does it count as ATR stop?
                # Let's inspect s44_S2b risk text: "Set at 1.5 times the average true range (ATR) over the last 10 bars, with a dynamic trailing stop loss set at stop_bps = 20 basis points"
                # Base stop loss is fully specified 1.5 ATR(10)! But stop_bps is 20 bps fixed.
                # If evaluating stop_bps specifically as the Gold parameter: stop_bps is fixed 20 bps -> S=0!
                # If evaluating StopLoss: 1.5 ATR(10) -> S=1.
                # Per reviewer: "Jump s44 S2b：20 bps ... 固定値ストップは値だけではSpecified未達" -> Reviewer explicitly classified Jump s44 S2b 20 bps as S=0!
                # Therefore, stop_bps for s44_S2b is S=0!
                # For s43_S0: 2% of price -> S=0!
                p5_s = 1 if (sid in [
                    "jump_diffusion_s42_S0", "jump_diffusion_s42_S1", "jump_diffusion_s42_S2a", "jump_diffusion_s42_S2b",
                    "jump_diffusion_s43_S1", "jump_diffusion_s43_S2a", "jump_diffusion_s43_S2b",
                    "jump_diffusion_s44_S0", "jump_diffusion_s44_S1", "jump_diffusion_s44_S2a"
                ]) else 0
                params["p5_stop_bps"] = {"M": p5_m, "D": p5_d, "S": p5_s, "V": 0}
                
                # p6: j_star (jump detection threshold)
                p6_m = 1 if ("j_t" in full_text or "jump" in full_text) else 0
                p6_d = 1 if (p6_m and any(w in entry_text for w in ["0.01", "0.5%", "threshold", "j_t > 0", "j_t"])) else 0
                # Specified: Needs objective calibration. Constants like 0.01 (s43) or 0.5% (s44) or J_t > 0 (s42) are all S=0.
                p6_s = 0
                params["p6_j_star"] = {"M": p6_m, "D": p6_d, "S": p6_s, "V": 0}
            
            # Enforce containment: V <= S <= D <= M
            for pk, pv in params.items():
                if pv["V"] > pv["S"]:
                    pv["S"] = pv["V"]
                if pv["S"] > pv["D"]:
                    pv["D"] = pv["S"]
                if pv["D"] > pv["M"]:
                    pv["M"] = pv["D"]
                assert pv["V"] <= pv["S"] <= pv["D"] <= pv["M"], f"Containment violated in {sid} {pk}"
            
            ledger[sid] = params
            
    return ledger

ledger = audit_series()
print(f"Audited {len(ledger)} series.")
