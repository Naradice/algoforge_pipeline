"""Independent Gold Specifications for H2-D1 Diagnostic Experiment.

Provides non-circular, mathematically rigorous truth criteria for evaluating
the 5 objective error flags across the 4 tested concept archetypes.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ConceptGoldSpecification(BaseModel):
    """Independent Gold criteria for a concept archetype."""
    concept_id: str
    concept_name: str
    legitimate_roles: List[str]
    illegitimate_roles: List[str]
    correct_state_variables: List[str]
    correct_control_variables: List[str]
    mathematical_confusions: List[str]
    data_requirements: List[str]
    untenable_data_assumptions: List[str]
    operable_minimum_standards: List[str]


GOLD_SPECIFICATIONS: Dict[str, ConceptGoldSpecification] = {
    "almgren_chriss": ConceptGoldSpecification(
        concept_id="almgren_chriss",
        concept_name="Almgren-Chriss Transient Impact Model",
        legitimate_roles=[
            "Optimal execution of a pre-determined inventory position",
            "Minimization of execution cost and market risk under price impact",
            "Schedule-based liquidation or accumulation over fixed horizon T"
        ],
        illegitimate_roles=[
            "Speculative directional trade entry signal (e.g. buying because trajectory points up)",
            "Alpha forecasting model predicting whether market price will rise or fall",
            "General technical analysis indicator for timing market turning points"
        ],
        correct_state_variables=[
            "Inventory remaining x(t) in [0, X_0] over time t",
            "Transient price impact state I(t) = integral_0^t eta * exp(-rho*(t-s)) * v(s) ds"
        ],
        correct_control_variables=[
            "Trading speed / liquidation rate v(t) = -dx(t)/dt",
            "Discrete block order sizes n_k = x_{k-1} - x_k"
        ],
        mathematical_confusions=[
            "Confusing inventory trajectory x(t) with price trajectory S(t) (e.g. 'price deviates from optimal trajectory')",
            "Treating liquidation schedule completion as a profit take-profit exit rather than order fulfillment",
            "Assuming trading speed v(t) represents expected return or price momentum"
        ],
        data_requirements=[
            "Execution fill logs or tick-level transaction data to identify transient impact parameter eta and decay rho",
            "Initial order quantity X_0 and desired execution window T"
        ],
        untenable_data_assumptions=[
            "Assuming transient price impact parameters can be directly calibrated from raw M5 OHLC bars alone without trade logs or order book data",
            "Assuming zero bid-ask spread or instantaneous fills in limit order book"
        ],
        operable_minimum_standards=[
            "Entry/Initiation defines an execution order for a specified inventory quantity X_0",
            "Exit/Termination defines completion when inventory x(T) = 0 or max horizon T is reached"
        ]
    ),
    "hawkes_process": ConceptGoldSpecification(
        concept_id="hawkes_process",
        concept_name="Multivariate Hawkes Point Process",
        legitimate_roles=[
            "Modeling self-excitation and cross-excitation in discrete financial event streams",
            "Detecting endogenous cascade risks and liquidity clustering in order arrival sequences",
            "Estimating instantaneous conditional intensity lambda(t) as a volatility/risk filter"
        ],
        illegitimate_roles=[
            "Direct portfolio asset allocation optimizer",
            "Automated order routing or market-making fill execution engine",
            "Deterministic price level target predictor"
        ],
        correct_state_variables=[
            "Conditional intensity vector lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta*(t - t_i))",
            "Branching ratio matrix Gamma = alpha / beta (spectral radius < 1 for stationarity)"
        ],
        correct_control_variables=[
            "None (Hawkes is an event-stream statistical model, not a stochastic control model)"
        ],
        mathematical_confusions=[
            "Confusing conditional intensity lambda(t) with asset price level S(t) or expected return mu_R",
            "Confusing Hawkes branching ratio Gamma with trade size or portfolio leverage",
            "Assuming Poisson baseline intensity mu guarantees positive price drift"
        ],
        data_requirements=[
            "Discrete event series with timestamps of sufficient resolution to distinguish event order (arrivals, trades, cancellations)"
        ],
        untenable_data_assumptions=[
            "Assuming Hawkes intensity can be directly estimated from fixed-interval M5 OHLC bars without an explicit intraday event reconstruction mechanism or microstructure proxy"
        ],
        operable_minimum_standards=[
            "Specifies exact mathematical feature (e.g. normalized intensity lambda(t) or excitation ratio)",
            "Specifies exact numeric percentile or threshold for signal generation",
            "Acknowledges required event resolution or provides an intraday proxy"
        ]
    ),
    "irm": ConceptGoldSpecification(
        concept_id="irm",
        concept_name="Invariant Risk Minimization (IRM)",
        legitimate_roles=[
            "Regularized estimation of directional prediction features invariant across training environments",
            "Mitigating out-of-distribution performance degradation under known regime partitions",
            "Feature representation learning penalizing environment-specific gradient variances"
        ],
        illegitimate_roles=[
            "Order execution trajectory scheduler",
            "Transaction cost or slippage minimization engine",
            "Guaranteed causal discovery tool proving universal invariance under arbitrary unknown shifts"
        ],
        correct_state_variables=[
            "Feature representation Phi(X)",
            "Classifier / predictor weights w acting on Phi(X)",
            "Environment index e in E"
        ],
        correct_control_variables=[
            "None (IRM is a supervised representation learning / regression objective)"
        ],
        mathematical_confusions=[
            "Claiming IRM mathematically guarantees causal invariance or risk-free alpha under unseen market regimes",
            "Assuming IRM eliminates market risk rather than environment-specific spurious correlation",
            "Treating the invariance penalty term as a risk management stop-loss rule"
        ],
        data_requirements=[
            "Environment-partitioned training data across distinct market regimes (e.g. high vs low volatility sessions, trading sessions)"
        ],
        untenable_data_assumptions=[
            "Training IRM on a single unpartitioned stationary time series without defining environment indices E"
        ],
        operable_minimum_standards=[
            "Defines explicit feature inputs and target prediction label (e.g. forward return sign)",
            "Specifies concrete probabilistic or score threshold for trade signal (e.g. probability > 0.65)",
            "Defines explicit operational stop-loss and take-profit"
        ]
    ),
    "jump_diffusion": ConceptGoldSpecification(
        concept_id="jump_diffusion",
        concept_name="Regime-Switching Jump Diffusion Model",
        legitimate_roles=[
            "Characterizing continuous diffusion combined with discrete compound Poisson jumps in asset returns",
            "Filtering latent market regimes (e.g. calm vs crisis) using Hamilton filter / EM algorithm",
            "Tail-risk estimation and volatility surface recalibration across regime shifts"
        ],
        illegitimate_roles=[
            "Deterministic trade execution trajectory planner",
            "High-frequency limit order book queuing model",
            "Arbitrage-free algorithmic order routing mechanism"
        ],
        correct_state_variables=[
            "Discrete latent regime state s_t in {1, ..., K}",
            "Continuous state: asset log-return r_t or mid-price S_t",
            "Jump intensity lambda_{s_t} and jump size distribution parameters (mu_J, sigma_J)"
        ],
        correct_control_variables=[
            "None (Jump diffusion is a continuous-time stochastic process / filtering model)"
        ],
        mathematical_confusions=[
            "Confusing latent regime probability P(s_t = k | F_t) with deterministic price trend or realized return",
            "Confusing jump intensity lambda_J with asset drift mu or volatility sigma",
            "Assuming jump occurrences are deterministic rather than stochastic Poisson arrivals"
        ],
        data_requirements=[
            "Asset return series r_t with sufficient sample size per regime to ensure parameter identifiability"
        ],
        untenable_data_assumptions=[
            "Claiming deterministic identification of regime states on M5 without probabilistic filtering or inference assumptions"
        ],
        operable_minimum_standards=[
            "Defines probabilistic filter threshold (e.g. crisis regime posterior probability > 0.80)",
            "Specifies mathematical risk scaling or position sizing rule based on filtered regime",
            "Provides unambiguous programmatic entry and exit rules"
        ]
    )
}
