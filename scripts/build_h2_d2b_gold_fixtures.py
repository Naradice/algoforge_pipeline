"""Script to construct and pre-freeze H2-D2b Dev36 Gold and Held-out 8 Quarantine Fixtures.

Strictly adheres to RP-HDE-H2-D2b-PLAN-v0.2:
- All 36 dev cases (24 synthetic + 12 live generated) evaluated across all 5 axes.
- Tri-state data feasibility: SUPPORTED, UNSUPPORTED, UNKNOWN.
- 3-element evidence: verbatim_span, missing_requirement, contract_reference.
- 8 held-out quarantine cases created with linguistic paraphrasing and compound errors (>= 2 simultaneous errors).
- SHA256 hashes logged and saved to an audit manifest.
"""

import json
import hashlib
from pathlib import Path

# Paths
BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"
GOLD_DIR = FIXTURES_DIR / "gold_standard"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

DEV36_PATH = GOLD_DIR / "h2_d2b_dev36_gold.json"
HELDOUT_PATH = GOLD_DIR / "h2_d2b_held_out_quarantine.json"
MANIFEST_PATH = GOLD_DIR / "h2_d2b_audit_manifest.json"

# Load synthetic cases
with open(FIXTURES_DIR / "h2_d2_benchmark_cases.json", encoding="utf-8") as f:
    synth_cases = json.load(f)

# Load live generated cases
with open(BASE_DIR / "data" / "h2_d1_diagnostic_results.json", encoding="utf-8") as f:
    live_data = json.load(f)
live_cases = live_data["results"]


def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()


dev36_list = []

# ==============================================================================
# 1. Synthetic 24 Cases Pre-Fixation
# ==============================================================================
for c in synth_cases:
    cid = c["case_id"]
    concept = c["concept_id"]
    ptype = c["perturbation_type"]
    cand = c["candidate"]
    spec = cand.get("backtest_spec") or {}
    
    flags = {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0}
    feasibility = "SUPPORTED"
    ev = {}
    notes = ""

    # Map injected errors
    if ptype == "e_role":
        flags["E_role"] = 1
        ev["E_role"] = {
            "verbatim_span": spec.get("entry_condition", ""),
            "missing_requirement": "Execution/representation model used as speculative entry signal or order routing engine",
            "contract_reference": "Model archetype does not generate directional trade signals"
        }
    elif ptype == "e_math":
        flags["E_math"] = 1
        ev["E_math"] = {
            "verbatim_span": spec.get("entry_condition", "") or cand.get("claim", ""),
            "missing_requirement": "Inventory trajectory x(t) conflated with price S(t), or intensity treated as return",
            "contract_reference": "Mathematical definition of state variables violated"
        }
    elif ptype == "e_data":
        flags["E_data"] = 1
        feasibility = "UNSUPPORTED"
        ev["E_data"] = {
            "verbatim_span": spec.get("feature_engineering", ""),
            "missing_requirement": "Requires microsecond events, trade logs, or L3 order book without contract data",
            "contract_reference": "Contract provides only USDJPY M5 OHLCV"
        }
    elif ptype == "e_leap":
        flags["E_leap"] = 1
        ev["E_leap"] = {
            "verbatim_span": cand.get("claim", ""),
            "missing_requirement": "Claim states empirical profitability as established literature fact",
            "contract_reference": "Epistemological requirement: hypothesis must be falsifiable conjecture"
        }
    elif ptype == "e_inop":
        flags["E_inop"] = 1
        ev["E_inop"] = {
            "verbatim_span": spec.get("entry_condition", ""),
            "missing_requirement": "Entry condition relies on qualitative uncomputable condition ('market conditions favorable', 'appropriate timing')",
            "contract_reference": "Trading rules must be deterministically computable from inputs"
        }

    # Hawkes Secondary Multi-label Audit (Addressing the Hawkes data issue in Gold)
    if concept == "hawkes_process":
        feat = spec.get("feature_engineering", "")
        # case_07 is clean, case_08 to case_12
        if cid == "case_07":
            # "Intraday proxy of Hawkes conditional intensity lambda(t) computed from tick event stream arrivals."
            # Tick event stream is outside USDJPY M5 OHLCV contract!
            # Under strict M5 contract without external tick feed, demanding tick event stream arrivals is UNSUPPORTED.
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires tick event stream arrivals under M5 OHLCV only contract",
                "contract_reference": "Contract is USDJPY M5 OHLCV only; tick stream is unavailable"
            }
            notes = "Re-audited: tick event stream violates strict M5 OHLCV contract."
        elif cid in ["case_08", "case_09", "case_11", "case_12"]:
            # Feature states only 'Hawkes branching ratio' or 'Hawkes intensity' on M5 timeframe without event source or proxy specification
            flags["E_data"] = 1
            feasibility = "UNKNOWN"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Hawkes point process requires discrete event arrivals, but no data source or reconstruction method is specified for M5 bars",
                "contract_reference": "Contract is USDJPY M5 OHLCV only; point process input is undefined"
            }
            notes = "Re-audited: secondary data inconsistency (UNKNOWN event source on M5)."

    dev36_list.append({
        "spec_id": cid,
        "concept_id": concept,
        "source": "synthetic_h2_d2",
        "perturbation_type": ptype,
        "candidate": cand,
        "gold_flags": flags,
        "gold_feasibility_status": feasibility,
        "evidence_details": ev,
        "adjudication_notes": notes or f"Synthetic benchmark case {cid} ({ptype})."
    })

# ==============================================================================
# 2. Live Generated 12 Cases Pre-Fixation (H2-D1)
# ==============================================================================
for item in live_cases:
    cid = f"{item['concept_id']}_{item['condition_id']}"
    concept = item["concept_id"]
    cond = item["condition_id"]
    cand = item["candidate"]
    spec = cand.get("backtest_spec") or {}

    flags = {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0}
    feasibility = "SUPPORTED"
    ev = {}
    notes = ""

    entry = spec.get("entry_condition", "")
    exit_ = spec.get("exit_condition", "")
    feat = spec.get("feature_engineering", "")
    claim = cand.get("claim", "")

    # 1. Almgren-Chriss
    if concept == "almgren_chriss":
        if cond == "D1-C0":
            # Speculative entry on volatility ATR
            flags["E_role"] = 1
            ev["E_role"] = {
                "verbatim_span": entry,
                "missing_requirement": "Almgren-Chriss liquidation model used as speculative volatility entry signal",
                "contract_reference": "Model archetype: execution only"
            }
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires order book depth and historical price impact metrics",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            flags["E_inop"] = 1
            ev["E_inop"] = {
                "verbatim_span": entry,
                "missing_requirement": "Condition 'and the market is trending' is qualitative and undefined",
                "contract_reference": "Entry condition must be deterministically computable"
            }
            notes = "D1-C0: E_role, E_data, E_inop"

        elif cond == "D1-C1":
            # Initiate trade when price impact indicates favorable execution window
            flags["E_role"] = 1
            ev["E_role"] = {
                "verbatim_span": entry,
                "missing_requirement": "Trading initiated based on predicted price impact rather than parent portfolio order",
                "contract_reference": "Model archetype: execution only"
            }
            flags["E_inop"] = 1
            ev["E_inop"] = {
                "verbatim_span": entry,
                "missing_requirement": "Decision directly depends on 'favorable execution window based on current market conditions' without computational definition",
                "contract_reference": "Execution triggers must be computable formulas"
            }
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires order book depth on M5 bars",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            notes = "D1-C1: E_role, E_inop, E_data"

        elif cond == "D1-C2":
            # Initiate liquidation when inventory x(t) > threshold and favorable execution environment
            flags["E_inop"] = 1
            ev["E_inop"] = {
                "verbatim_span": entry,
                "missing_requirement": "Decision relies on 'favorable execution environment'; lacks initial inventory x_0, liquidation trajectory schedule, and terminal horizon T control law",
                "contract_reference": "Finite-horizon complete liquidation requires complete schedule and boundary conditions"
            }
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires historical transaction data and order book events for eta and rho",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            notes = "D1-C2: E_inop, E_data"

    # 2. Hawkes Process
    elif concept == "hawkes_process":
        if cond == "D1-C0":
            flags["E_math"] = 1
            ev["E_math"] = {
                "verbatim_span": entry,
                "missing_requirement": "Point process arrival intensity lambda conflated with directional upward price movement",
                "contract_reference": "Mathematical property: intensity is arrival frequency, not directional drift"
            }
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires order sizes, timestamps, and order flow data on M5 bars",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            flags["E_inop"] = 1
            ev["E_inop"] = {
                "verbatim_span": entry,
                "missing_requirement": "'high likelihood of upward price movement' is qualitative and lacks numerical threshold",
                "contract_reference": "Entry must be computable"
            }
            notes = "D1-C0: E_math, E_data, E_inop"

        elif cond == "D1-C1":
            flags["E_math"] = 1
            ev["E_math"] = {
                "verbatim_span": entry,
                "missing_requirement": "Couples order flow intensity with directional price reversal prediction",
                "contract_reference": "Hawkes models event clustering, not directional reversal"
            }
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires cancellation rates and order flow intensity on M5 bars",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            flags["E_inop"] = 1
            ev["E_inop"] = {
                "verbatim_span": entry,
                "missing_requirement": "'significant increase in order flow intensity' lacks numerical threshold or estimation formula",
                "contract_reference": "Entry must be computable"
            }
            notes = "D1-C1: E_math, E_data, E_inop"

        elif cond == "D1-C2":
            # Hawkes C2: states 'Construct a timestamped event sequence of order arrivals and cancellations'
            # Under strict USDJPY M5 OHLCV contract with no external source, this requirement is UNKNOWN / UNSUPPORTED
            # Since it explicitly demands constructing it without an external feed, it is UNKNOWN (missing source) / UNSUPPORTED under contract.
            flags["E_data"] = 1
            feasibility = "UNKNOWN"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Demands timestamped event sequence of order arrivals/cancellations, but data source and reconstruction method under M5 contract are undetermined",
                "contract_reference": "Contract is USDJPY M5 OHLCV only; event arrivals cannot be verified"
            }
            flags["E_inop"] = 1
            ev["E_inop"] = {
                "verbatim_span": entry,
                "missing_requirement": "'lambda(t) exceeds a predefined threshold' without specifying value or parameter estimation procedure",
                "contract_reference": "Entry threshold must be bound"
            }
            notes = "D1-C2: E_data (UNKNOWN), E_inop"

    # 3. IRM
    elif concept == "irm":
        if cond == "D1-C0":
            flags["E_data"] = 1
            feasibility = "UNKNOWN"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires macroeconomic indicators and historical order flow data without specified source",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            notes = "D1-C0: E_data (UNKNOWN)"

        elif cond == "D1-C1":
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires order book imbalance on M5 OHLC bars without a proxy formula",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            notes = "D1-C1: E_data (UNSUPPORTED)"

        elif cond == "D1-C2":
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires order book imbalance across sessions on M5 OHLC bars without proxy",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            notes = "D1-C2: E_data (UNSUPPORTED)"

    # 4. Jump Diffusion
    elif concept == "jump_diffusion":
        if cond == "D1-C0":
            flags["E_data"] = 1
            feasibility = "UNKNOWN"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires central bank announcement schedules without external feed contract",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            flags["E_inop"] = 1
            ev["E_inop"] = {
                "verbatim_span": entry,
                "missing_requirement": "'predicts a high jump intensity' lacks quantitative threshold",
                "contract_reference": "Trigger threshold must be bound"
            }
            notes = "D1-C0: E_data (UNKNOWN), E_inop"

        elif cond == "D1-C1":
            flags["E_data"] = 1
            feasibility = "UNSUPPORTED"
            ev["E_data"] = {
                "verbatim_span": feat,
                "missing_requirement": "Requires sentiment analysis from news articles and interest rate differentials",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
            notes = "D1-C1: E_data (UNSUPPORTED)"

        elif cond == "D1-C2":
            # Log-returns, regime probabilities and jump intensity from log returns -> SUPPORTED
            feasibility = "SUPPORTED"
            flags["E_inop"] = 1
            ev["E_inop"] = {
                "verbatim_span": entry,
                "missing_requirement": "'lambda_bull > threshold' and 'jump intensity falls below a certain threshold' without bound values or estimation formulas",
                "contract_reference": "Thresholds must be bound or have explicit estimation specification"
            }
            notes = "D1-C2: E_inop (unbound thresholds). Feasibility is SUPPORTED from log-returns."

    dev36_list.append({
        "spec_id": cid,
        "concept_id": concept,
        "source": "live_generated_h2_d1",
        "condition_id": cond,
        "candidate": cand,
        "gold_flags": flags,
        "gold_feasibility_status": feasibility,
        "evidence_details": ev,
        "adjudication_notes": notes
    })

# Save Dev36 Gold
with open(DEV36_PATH, "w", encoding="utf-8") as f:
    json.dump(dev36_list, f, indent=2, ensure_ascii=False)

dev36_sha = compute_sha256(DEV36_PATH.read_bytes())
print(f"Dev36 Gold created: {len(dev36_list)} cases. SHA256: {dev36_sha}")


# ==============================================================================
# 3. Held-Out 8 Quarantine Cases Creation
# ==============================================================================
# 4 concepts x 2 cases (1 Sound, 1 Compound Error >= 2 axes)
# Uses linguistic paraphrasing (divergent keywords from dev set)
heldout_list = [
    # 1. Almgren-Chriss Sound (Finite-horizon liquidation with complete trajectory)
    {
        "case_id": "heldout_01_ac_sound",
        "concept_id": "almgren_chriss",
        "candidate": {
            "hypothesis_id": "heldout_ac_sound",
            "title": "Optimal Liquidation Variance Reduction via Discrete Trajectory",
            "target_phenomenon": "Execution slippage under temporary price impact",
            "analogical_source": "Almgren-Chriss Optimal Liquidation Model",
            "claim": "Executing target block order of N shares via discrete Almgren-Chriss trajectory minimizes liquidation variance compared to continuous linear TWAP.",
            "proposed_mechanism": "Computes hyperbolic trajectory x_k = X_0 * sinh(kappa*(T - t_k)) / sinh(kappa*T) to balance temporary price concession against price volatility risk.",
            "grounding_rationale": "Theoretical benchmark established by Almgren & Chriss (2000) for finite horizon inventory exhaustion.",
            "testable_falsification_condition": "Liquidation slippage variance fails to achieve 15% reduction vs linear TWAP with p < 0.05.",
            "backtest_spec": {
                "target_asset": "USDJPY",
                "timeframe": "M5",
                "entry_condition": "Initiate planned liquidation schedule of initial quantity X_0 = 1,000,000 USDJPY units at market open t_0 = 0 under fixed horizon T = 12 M5 intervals.",
                "exit_condition": "Liquidate exactly v_k = x_{k-1} - x_k units each interval k; enforce absolute termination with final market sweep x_K = 0 at interval K = 12.",
                "feature_engineering": "Historical 20-day M5 Garman-Klass volatility sigma and linear temporary impact parameter estimated from in-sample regression.",
                "risk_management": "Strict participation ceiling capped at 10% of 5-minute bar volume.",
                "benchmark_baseline": "Standard linear TWAP schedule over 12 intervals.",
                "falsification_metric": "Liquidation cost exceeds TWAP baseline cost."
            }
        },
        "gold_flags": {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0},
        "gold_feasibility_status": "SUPPORTED",
        "evidence_details": {},
        "adjudication_notes": "Sound finite-horizon execution spec with complete schedule and boundary conditions."
    },
    # 2. Almgren-Chriss Compound Error (E_role + E_math + E_inop)
    {
        "case_id": "heldout_02_ac_compound",
        "concept_id": "almgren_chriss",
        "candidate": {
            "hypothesis_id": "heldout_ac_compound",
            "title": "Almgren-Chriss Speculative Trend Exploitation",
            "target_phenomenon": "Price momentum following trajectory deviations",
            "analogical_source": "Almgren-Chriss Model",
            "claim": "Predicting exchange rate breakout rallies by tracking deviations from the Almgren-Chriss liquidation trajectory generates superior speculative returns.",
            "proposed_mechanism": "Exploiting discrepancies between mid-market spot quotes and optimal liquidation curves to trigger aggressive directional speculative entries.",
            "grounding_rationale": "Speculative momentum can be inferred when spot rate deviates from trajectory.",
            "testable_falsification_condition": "Annualized Sharpe ratio fails to exceed 1.0.",
            "backtest_spec": {
                "target_asset": "USDJPY",
                "timeframe": "M5",
                "entry_condition": "Buy USDJPY aggressively whenever the spot exchange rate drifts below the optimal liquidation trajectory curve and broader liquidity conditions appear favorable.",
                "exit_condition": "Close speculative position whenever spot rate snaps back toward the trajectory curve or whenever liquidity appears to dry up.",
                "feature_engineering": "Almgren-Chriss liquidation curve estimated from 20-period rolling close prices.",
                "risk_management": "Fixed stop-loss of 20 pips.",
                "benchmark_baseline": "20-period SMA crossover baseline.",
                "falsification_metric": "Sharpe ratio < 1.0."
            }
        },
        "gold_flags": {"E_role": 1, "E_math": 1, "E_data": 0, "E_leap": 0, "E_inop": 1},
        "gold_feasibility_status": "SUPPORTED",
        "evidence_details": {
            "E_role": {
                "verbatim_span": "Buy USDJPY aggressively whenever the spot exchange rate drifts below the optimal liquidation trajectory curve",
                "missing_requirement": "Execution liquidation trajectory misused as directional speculative buy signal",
                "contract_reference": "Model archetype: execution only"
            },
            "E_math": {
                "verbatim_span": "spot exchange rate drifts below the optimal liquidation trajectory curve",
                "missing_requirement": "Conflates remaining inventory trajectory x(t) with spot price quote S(t)",
                "contract_reference": "State variable definition violated"
            },
            "E_inop": {
                "verbatim_span": "and broader liquidity conditions appear favorable / whenever liquidity appears to dry up",
                "missing_requirement": "Decision depends on vague qualitative condition without mathematical definition",
                "contract_reference": "Rules must be deterministically computable"
            }
        },
        "adjudication_notes": "Compound error: E_role (speculative entry), E_math (x(t) vs S(t)), E_inop (favorable liquidity)."
    },
    # 3. Hawkes Process Sound (Event arrival proxy defined via price changes on M5)
    {
        "case_id": "heldout_03_hawkes_sound",
        "concept_id": "hawkes_process",
        "candidate": {
            "hypothesis_id": "heldout_hawkes_sound",
            "title": "Hawkes Volatility Clustering Risk Control",
            "target_phenomenon": "Endogenous tail-risk cascades during extreme volatility",
            "analogical_source": "Self-Exciting Hawkes Point Process",
            "claim": "Self-exciting volatility bursts detected via high-frequency bar activity clustering indicate tail-risk expansion in USDJPY.",
            "proposed_mechanism": "Discretizing M5 price jumps exceeding 2.0 empirical standard deviations into a synthetic point event series to estimate self-excitation parameters.",
            "grounding_rationale": "Hawkes processes model temporal clustering of extreme volatility events.",
            "testable_falsification_condition": "Maximum drawdown reduction fails to achieve 20% vs passive carry.",
            "backtest_spec": {
                "target_asset": "USDJPY",
                "timeframe": "M5",
                "entry_condition": "Reduce carry exposure to 0 when estimated self-excitation branching ratio alpha/beta exceeds 0.85 computed on trailing 288-bar M5 window.",
                "exit_condition": "Restore standard position size when branching ratio decays below 0.40 for 6 consecutive M5 intervals.",
                "feature_engineering": "Binary exceedance events defined as abs(return_t) > 2.0 * rolling_std_288; exponential kernel estimation of alpha and beta via maximum likelihood.",
                "risk_management": "Binary de-risking: 0% exposure during high excitation, 100% normal exposure otherwise.",
                "benchmark_baseline": "Static buy-and-hold USDJPY carry position.",
                "falsification_metric": "Drawdown reduction < 20%."
            }
        },
        "gold_flags": {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0},
        "gold_feasibility_status": "SUPPORTED",
        "evidence_details": {},
        "adjudication_notes": "Sound risk-management spec using an explicit computable M5 bar exceedance proxy for point events."
    },
    # 4. Hawkes Process Compound Error (E_data + E_inop)
    {
        "case_id": "heldout_04_hawkes_compound",
        "concept_id": "hawkes_process",
        "candidate": {
            "hypothesis_id": "heldout_hawkes_compound",
            "title": "High-Frequency Hawkes Message Traffic Scalping",
            "target_phenomenon": "Sub-millisecond liquidity holes",
            "analogical_source": "Multivariate Hawkes Process",
            "claim": "Direct Hawkes point process calibration yields profitable scalping by anticipating order book microbursts.",
            "proposed_mechanism": "Tracking instantaneous sub-millisecond message traffic bursts across currency dealers.",
            "grounding_rationale": "Order cancellations cascade into liquidity holes.",
            "testable_falsification_condition": "Annualized net profit factor fails to exceed 1.2 after transaction fees.",
            "backtest_spec": {
                "target_asset": "USDJPY",
                "timeframe": "M5",
                "entry_condition": "Go long when dealer limit order replacement intensity exceeds baseline frequency under opportune market momentum.",
                "exit_condition": "Flatten trade when intensity calms down to normal levels.",
                "feature_engineering": "Sub-millisecond dealer quote message stream and limit order cancel event logs.",
                "risk_management": "Tight trailing stop of 3 pips.",
                "benchmark_baseline": "Random entry baseline with equivalent holding time.",
                "falsification_metric": "Profit factor < 1.2."
            }
        },
        "gold_flags": {"E_role": 0, "E_math": 0, "E_data": 1, "E_leap": 0, "E_inop": 1},
        "gold_feasibility_status": "UNSUPPORTED",
        "evidence_details": {
            "E_data": {
                "verbatim_span": "Sub-millisecond dealer quote message stream and limit order cancel event logs",
                "missing_requirement": "Requires sub-millisecond dealer quote message stream and cancellation logs",
                "contract_reference": "Contract provides only USDJPY M5 OHLCV bars"
            },
            "E_inop": {
                "verbatim_span": "under opportune market momentum / when intensity calms down to normal levels",
                "missing_requirement": "Decision relies on undefined qualitative terms without mathematical threshold",
                "contract_reference": "Rules must be computable"
            }
        },
        "adjudication_notes": "Compound error: E_data (UNSUPPORTED sub-ms logs), E_inop (opportune momentum)."
    },
    # 5. IRM Sound (Multi-session invariance with explicit M5 features)
    {
        "case_id": "heldout_05_irm_sound",
        "concept_id": "irm",
        "candidate": {
            "hypothesis_id": "heldout_irm_sound",
            "title": "Cross-Session Invariant Risk Minimization for Directional Return",
            "target_phenomenon": "Predictor performance decay under inter-session regime shifts",
            "analogical_source": "Invariant Risk Minimization (IRM)",
            "claim": "Learning representation vectors that satisfy invariant gradient optimality across Asian, European, and US trading sessions reduces out-of-sample prediction error.",
            "proposed_mechanism": "Minimizing empirical cross-entropy while penalizing squared gradient norm of linear classifier across distinct trading session environments.",
            "grounding_rationale": "Arjovsky et al. (2019) Invariant Risk Minimization framework for out-of-distribution generalization.",
            "testable_falsification_condition": "Out-of-sample directional accuracy fails to exceed standard ERM baseline by 3.0 percentage points with p < 0.05.",
            "backtest_spec": {
                "target_asset": "USDJPY",
                "timeframe": "M5",
                "entry_condition": "Go long when invariant predictor output f(x) > 0.65; go short when f(x) < 0.35 at session opening bars.",
                "exit_condition": "Close trade on opposite signal or when trailing stop of 25 pips is breached.",
                "feature_engineering": "M5 log returns, 14-period RSI, Garman-Klass volatility, and signed volume momentum partitioned into Asian (UTC 0-8), European (UTC 8-16), and US (UTC 16-24) environments.",
                "risk_management": "Fixed risk allocation of 1.0% equity per trade.",
                "benchmark_baseline": "Standard Empirical Risk Minimization (ERM) linear model.",
                "falsification_metric": "Out-of-sample Sharpe ratio < ERM baseline Sharpe."
            }
        },
        "gold_flags": {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0},
        "gold_feasibility_status": "SUPPORTED",
        "evidence_details": {},
        "adjudication_notes": "Sound IRM representation spec using native M5 OHLCV features and explicit session environments."
    },
    # 6. IRM Compound Error (E_math + E_leap)
    {
        "case_id": "heldout_06_irm_compound",
        "concept_id": "irm",
        "candidate": {
            "hypothesis_id": "heldout_irm_compound",
            "title": "Guaranteed Zero-Risk IRM Arbitrage Discovery",
            "target_phenomenon": "Structural regime breaks",
            "analogical_source": "Invariant Risk Minimization",
            "claim": "It is an established mathematical fact that IRM guarantees absolute invariance and completely eliminates downside risk in foreign exchange trading, proven to yield 35% annual return.",
            "proposed_mechanism": "Invariant feature representations eradicate all market regime instability by proving exact causal invariance across all future market states.",
            "grounding_rationale": "Literature proves that invariant classifiers never degrade under any structural break.",
            "testable_falsification_condition": "Annualized strategy return fails to exceed 35%.",
            "backtest_spec": {
                "target_asset": "USDJPY",
                "timeframe": "M5",
                "entry_condition": "Enter long whenever invariant score exceeds 0.50.",
                "exit_condition": "Exit when score drops below 0.50.",
                "feature_engineering": "Normalized returns and rolling standard deviations.",
                "risk_management": "No stop-loss required due to zero-risk guarantee.",
                "benchmark_baseline": "Zero benchmark.",
                "falsification_metric": "Annual return < 35%."
            }
        },
        "gold_flags": {"E_role": 0, "E_math": 1, "E_data": 0, "E_leap": 1, "E_inop": 0},
        "gold_feasibility_status": "SUPPORTED",
        "evidence_details": {
            "E_math": {
                "verbatim_span": "IRM guarantees absolute invariance and completely eliminates downside risk in foreign exchange trading",
                "missing_requirement": "Claims IRM guarantees elimination of all market risk and universal invariance under arbitrary shifts",
                "contract_reference": "Mathematical property of IRM is non-linear heuristic penalty, not guaranteed zero-risk oracle"
            },
            "E_leap": {
                "verbatim_span": "proven to yield 35% annual return / established mathematical fact that strategy is profitable",
                "missing_requirement": "Asserts empirical excess profitability as an established proven literature fact",
                "contract_reference": "Epistemological requirement"
            }
        },
        "adjudication_notes": "Compound error: E_math (guaranteed zero risk), E_leap (proven 35% annual return)."
    },
    # 7. Jump Diffusion Sound (Hamilton filter regime detection from M5 log-returns)
    {
        "case_id": "heldout_07_jump_sound",
        "concept_id": "jump_diffusion",
        "candidate": {
            "hypothesis_id": "heldout_jump_sound",
            "title": "Bipower Variation Jump Filtering for Dynamic Carry Hedging",
            "target_phenomenon": "Abrupt currency crash risk during Poisson jump episodes",
            "analogical_source": "Jump Diffusion and Threshold Bipower Variation",
            "claim": "Disentangling Brownian diffusive volatility from compound Poisson jump variance via threshold bipower variation dynamically shields USDJPY carry portfolios.",
            "proposed_mechanism": "Calculating continuous bipower variation and jump component on M5 returns to infer regime transition probabilities via a 2-state Markov chain.",
            "grounding_rationale": "Mancini (2009) threshold estimation separates jump components from diffusive Gaussian volatility.",
            "testable_falsification_condition": "Maximum drawdown reduction fails to achieve 25% vs unhedged carry baseline.",
            "backtest_spec": {
                "target_asset": "USDJPY",
                "timeframe": "M5",
                "entry_condition": "Maintain baseline long carry trade when filtered tranquil regime probability P(S_t = 0 | F_t) > 0.75.",
                "exit_condition": "Liquidate all carry positions immediately if jump indicator J_t = (RV_t - BV_t) exceeds 1.5 * BV_t or if tranquil regime probability drops below 0.50.",
                "feature_engineering": "Realized variance RV_t and bipower variation BV_t computed from trailing 288 M5 return bars.",
                "risk_management": "Dynamic deleveraging from 1.0x to 0.0x based on jump probability.",
                "benchmark_baseline": "Unhedged USDJPY carry portfolio.",
                "falsification_metric": "Drawdown reduction < 25%."
            }
        },
        "gold_flags": {"E_role": 0, "E_math": 0, "E_data": 0, "E_leap": 0, "E_inop": 0},
        "gold_feasibility_status": "SUPPORTED",
        "evidence_details": {},
        "adjudication_notes": "Sound jump-diffusion risk-control spec using computable bipower variation on M5 bars."
    },
    # 8. Jump Diffusion Compound Error (E_role + E_data)
    {
        "case_id": "heldout_08_jump_compound",
        "concept_id": "jump_diffusion",
        "candidate": {
            "hypothesis_id": "heldout_jump_compound",
            "title": "Multi-Venue Poisson Jump Order Queue Placement",
            "target_phenomenon": "Adverse selection in fragmented limit order books",
            "analogical_source": "Jump Diffusion Process",
            "claim": "Exploiting jump diffusion parameters to orchestrate automated limit order execution routing optimizes venue execution queues.",
            "proposed_mechanism": "Routing child limit orders across multiple FX venues based on real-time Poisson jump intensity estimates.",
            "grounding_rationale": "Venues with lower jump intensity offer tighter spread capture.",
            "testable_falsification_condition": "Execution effective spread fails to achieve 10% improvement vs single-venue execution.",
            "backtest_spec": {
                "target_asset": "USDJPY",
                "timeframe": "M5",
                "entry_condition": "Route child buy order to venue A when venue Poisson jump rate is below 0.10 and venue order book queue depth exceeds 5M units.",
                "exit_condition": "Cancel order when queue position drops below 10th percentile.",
                "feature_engineering": "Multi-venue real-time order book queue depth feeds and venue-specific trade execution streams.",
                "risk_management": "Max venue inventory concentration limit of $1M USD.",
                "benchmark_baseline": "Single primary venue limit order placement.",
                "falsification_metric": "Effective spread savings < 10%."
            }
        },
        "gold_flags": {"E_role": 1, "E_math": 0, "E_data": 1, "E_leap": 0, "E_inop": 0},
        "gold_feasibility_status": "UNSUPPORTED",
        "evidence_details": {
            "E_role": {
                "verbatim_span": "orchestrate automated limit order execution routing optimizes venue execution queues / Route child buy order to venue A",
                "missing_requirement": "Jump diffusion asset pricing model repurposed as multi-venue order execution routing scheduler",
                "contract_reference": "Model archetype mismatch: asset pricing / risk model is not an order routing scheduler"
            },
            "E_data": {
                "verbatim_span": "Multi-venue real-time order book queue depth feeds and venue-specific trade execution streams",
                "missing_requirement": "Requires multi-venue real-time L2/L3 queue depth and trade feeds under M5 OHLCV contract",
                "contract_reference": "Contract is USDJPY M5 OHLCV only"
            }
        },
        "adjudication_notes": "Compound error: E_role (execution routing scheduler), E_data (UNSUPPORTED multi-venue L2/L3 depth)."
    }
]

with open(HELDOUT_PATH, "w", encoding="utf-8") as f:
    json.dump(heldout_list, f, indent=2, ensure_ascii=False)

heldout_sha = compute_sha256(HELDOUT_PATH.read_bytes())
print(f"Held-out 8 Quarantine created: {len(heldout_list)} cases. SHA256: {heldout_sha}")

# Write Manifest
manifest = {
    "experiment_id": "RP-HDE-H2-D2b",
    "protocol_version": "v0.2",
    "created_at": "2026-11-18T03:53:00Z",
    "data_contract": "USDJPY_M5_OHLCV_STRICT",
    "dev36_path": str(DEV36_PATH),
    "dev36_cases_count": len(dev36_list),
    "dev36_sha256": dev36_sha,
    "heldout_path": str(HELDOUT_PATH),
    "heldout_cases_count": len(heldout_list),
    "heldout_sha256": heldout_sha,
    "heldout_quarantine_enforced": True,
    "single_shot_evaluation_logged": False
}

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest written to {MANIFEST_PATH}")
