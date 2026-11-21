"""Controlled Vocabulary for Capability Extraction (H1a).

Two-tier structure:
- Layer 1: Universal Types (applicable across science/engineering domains)
- Layer 2: Domain-Specific Specializations (finance, physics, biology, etc.)
"""

from typing import Set, Dict, List, Optional
from enum import Enum


class UniversalDataTypeL1(str, Enum):
    """L1 Universal Types applicable across domains."""
    TIMESERIES_1D = "timeseries_1d"
    TIMESERIES_MULTIVARIATE = "timeseries_multivariate"
    TABULAR_RECORD = "tabular_record"
    GRAPH_NETWORK = "graph_network"
    DISCRETE_SEQUENCE = "discrete_sequence"
    PROBABILITY_DISTRIBUTION = "probability_distribution"
    PARAMETER_VECTOR = "parameter_vector"
    SCALAR_VALUE = "scalar_value"
    EVENT_STREAM = "event_stream"
    UNKNOWN = "unknown"


L1_VALID_TYPES: Set[str] = {t.value for t in UniversalDataTypeL1}

# L2 Domain Specialization Mappings (initial registry for Finance and Complex Systems)
L2_DOMAIN_REGISTRY: Dict[str, List[str]] = {
    "timeseries_1d": [
        "price_timeseries",
        "return_series",
        "log_return_series",
        "volatility_series",
        "heartbeat_series",
        "seismic_waveform",
    ],
    "timeseries_multivariate": [
        "ohlcv_bars",
        "macroeconomic_indicators",
        "orderbook_depth_levels",
        "multi_channel_sensor_stream",
    ],
    "tabular_record": [
        "financial_report_ratios",
        "analyst_forecast",
        "balance_sheet_record",
        "clinical_trial_record",
    ],
    "graph_network": [
        "supply_chain_network",
        "cross_currency_graph",
        "interbank_exposure_network",
        "social_interaction_graph",
    ],
    "discrete_sequence": [
        "order_event_stream",
        "news_token_stream",
        "dna_sequence",
        "transaction_action_trace",
    ],
    "probability_distribution": [
        "predictive_return_distribution",
        "posterior_parameters",
        "state_transition_distribution",
        "risk_neutral_density",
    ],
    "parameter_vector": [
        "agent_behavior_parameters",
        "neural_network_weights",
        "physical_constants",
        "simulation_hyperparameters",
    ],
    "scalar_value": [
        "sharpe_ratio",
        "volatility_estimate",
        "anomaly_score",
        "lyapunov_exponent",
        "critical_point_parameter",
    ],
    "event_stream": [
        "tick_trades",
        "earnings_announcements",
        "limit_order_cancellations",
        "server_fault_events",
    ],
    "unknown": [],
}


def is_valid_l1_type(data_type: str) -> bool:
    """Check if string is a valid L1 Universal Type."""
    return data_type in L1_VALID_TYPES


def is_known_l2_type(l1_type: str, l2_type: Optional[str]) -> bool:
    """Check if L2 specialization is registered under given L1 type."""
    if l2_type is None:
        return True
    if l1_type not in L2_DOMAIN_REGISTRY:
        return False
    return l2_type in L2_DOMAIN_REGISTRY[l1_type]


def register_l2_type(l1_type: str, l2_type: str) -> None:
    """Register a new L2 domain type during Development Set evaluation."""
    if l1_type not in L2_DOMAIN_REGISTRY:
        raise ValueError(f"Unknown L1 type: {l1_type}")
    if l2_type not in L2_DOMAIN_REGISTRY[l1_type]:
        L2_DOMAIN_REGISTRY[l1_type].append(l2_type)
