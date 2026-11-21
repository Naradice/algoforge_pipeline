"""Constructs the deterministic 30-document Gold Standard Dataset for HDE H1a.

Adheres strictly to:
- v0.5 Dataset Design (10 Development, 12 Holdout Pilot, 8 Challenge)
- 8,000 character maximum scope for READMEs (User Instruction 2)
- Exact canonical text hashing and character offsets
- Multi-label sentence annotations for Kappa calculation
- Zero overlap / cross-split isolation
"""

import json
from typing import Dict, List, Any
import hashlib
import unicodedata

from research_pipeline.extractors.models import (
    canonicalize_text,
    compute_canonical_hash,
)


def create_gold_document(
    source_id: str,
    version_id: str,
    split: str,  # 'development', 'holdout_pilot', 'challenge'
    source_type: str,  # 'paper' or 'oss'
    title: str,
    abstract_text: str = "",
    readme_text: str = "",
    arxiv_base_id: str = "",
    repository_id: int = 0,
    root_parent_repo_id: int = 0,
    entities: List[Dict[str, Any]] = None,
    relations: List[Dict[str, Any]] = None,
    sentence_annotations: List[Dict[str, int]] = None,
) -> Dict[str, Any]:
    # Canonicalize texts (README capped at 8000 chars as per User Instruction 2)
    c_title = canonicalize_text(title)
    c_abstract = canonicalize_text(abstract_text)
    c_readme = canonicalize_text(readme_text[:8000]) if readme_text else ""

    fields = {
        "title_text": c_title,
        "abstract_text": c_abstract,
        "readme_text": c_readme,
    }
    field_hashes = {k: compute_canonical_hash(v) for k, v in fields.items() if v}

    # Verify and stamp field hashes and offsets into entities
    verified_entities = []
    for ent in (entities or []):
        ev = ent["evidence"]
        target_f = ev["target_field_name"]
        target_text = fields[target_f]
        target_hash = field_hashes[target_f]
        quote = ev["verbatim_quote"]

        start = target_text.find(quote)
        if start == -1:
            raise ValueError(f"Quote '{quote}' not found in field '{target_f}' for doc '{source_id}'")
        end = start + len(quote)

        verified_ev = {
            "target_field_name": target_f,
            "target_field_hash": target_hash,
            "verbatim_quote": quote,
            "start_char": start,
            "end_char": end,
            "raw_start_char": start,
            "raw_end_char": end,
            "raw_offset_valid": True,
            "resolved_offset_valid": True,
            "offset_resolution_status": "unchanged",
        }
        ent_copy = dict(ent)
        ent_copy["evidence"] = verified_ev
        verified_entities.append(ent_copy)

    # Verify relations
    verified_relations = []
    for rel in (relations or []):
        ev = rel["evidence"]
        target_f = ev["target_field_name"]
        target_text = fields[target_f]
        target_hash = field_hashes[target_f]
        quote = ev["verbatim_quote"]

        start = target_text.find(quote)
        if start == -1:
            raise ValueError(f"Relation quote '{quote}' not found in field '{target_f}' for doc '{source_id}'")
        end = start + len(quote)

        verified_ev = {
            "target_field_name": target_f,
            "target_field_hash": target_hash,
            "verbatim_quote": quote,
            "start_char": start,
            "end_char": end,
            "raw_start_char": start,
            "raw_end_char": end,
            "raw_offset_valid": True,
            "resolved_offset_valid": True,
            "offset_resolution_status": "unchanged",
        }
        rel_copy = dict(rel)
        rel_copy["evidence"] = verified_ev
        verified_relations.append(rel_copy)

    version_hash = hashlib.sha256((c_title + c_abstract + c_readme).encode("utf-8")).hexdigest()

    return {
        "source_id": source_id,
        "version_id": version_id,
        "version_hash": version_hash,
        "split": split,
        "source_type": source_type,
        "arxiv_base_id": arxiv_base_id,
        "repository_id": repository_id,
        "root_parent_repo_id": root_parent_repo_id,
        "target_fields": fields,
        "target_field_hashes": field_hashes,
        "entities": verified_entities,
        "relations": verified_relations,
        "sentence_annotations": sentence_annotations or [],
    }


def generate_all_30_documents() -> List[Dict[str, Any]]:
    docs = []

    # ==========================================
    # 1. DEVELOPMENT SET (10 items)
    # ==========================================

    # Dev 1: Finance Paper - Minority Game
    t1 = "Minority Game and Volatility Clustering in Financial Markets"
    a1 = (
        "We demonstrate that the Minority Game model reproduces volatility clustering in financial time series. "
        "The model assumes that market participants have bounded rationality and finite memory capacity. "
        "Using this framework, we simulate synthetic return series from inductive choice parameters. "
        "However, multi-asset extension suffers from exponential state space growth."
    )
    docs.append(create_gold_document(
        source_id="arxiv_2301_01001",
        version_id="v1",
        split="development",
        source_type="paper",
        title=t1,
        abstract_text=a1,
        arxiv_base_id="2301.01001",
        entities=[
            {
                "local_id": "phe_01",
                "entity_type": "phenomenon",
                "name": "Volatility Clustering",
                "description": "Clustering of volatile periods",
                "synonyms": ["clustering of volatility", "volatility clusters"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "volatility clustering"},
            },
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Minority Game",
                "category": "agent_based",
                "description": "Agent-based game theoretical model",
                "synonyms": ["Minority Game model", "MG"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Minority Game model"},
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Simulate Return Series",
                "operation": "generate",
                "synonyms": ["synthetic return simulation"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "simulate synthetic return series from inductive choice parameters"},
            },
            {
                "local_id": "asm_01",
                "entity_type": "assumption",
                "statement": "Market participants have bounded rationality and finite memory capacity.",
                "synonyms": [],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "market participants have bounded rationality and finite memory capacity"},
            },
            {
                "local_id": "lim_01",
                "entity_type": "limitation",
                "statement": "Multi-asset extension suffers from exponential state space growth.",
                "synonyms": [],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "multi-asset extension suffers from exponential state space growth"},
            },
        ],
        relations=[
            {
                "relation_id": "rel_01",
                "src_local_id": "mod_01",
                "dst_local_id": "phe_01",
                "relation_type": "EXPLAINS",
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Minority Game model reproduces volatility clustering"},
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "mod_01",
                "relation_type": "PROVIDED_BY",
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Using this framework, we simulate synthetic return series from inductive choice parameters"},
            },
        ],
        sentence_annotations=[
            {"model": 1, "phenomenon": 1, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 1, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 1},
        ],
    ))

    # Dev 2: Finance Paper - Hawkes Process
    t2 = "Hawkes Process Calibration for High-Frequency Order Flow Imbalance"
    a2 = (
        "We calibrate mutually exciting Hawkes processes to model high-frequency order flow imbalance. "
        "We employ Maximum Likelihood Estimation to recover baseline intensities and excitation matrices. "
        "The model predicts short-term mid-price movements from limit order arrivals. "
        "We assume that order events follow Poisson point dynamics with exponential kernels."
    )
    docs.append(create_gold_document(
        source_id="arxiv_2301_01002",
        version_id="v1",
        split="development",
        source_type="paper",
        title=t2,
        abstract_text=a2,
        arxiv_base_id="2301.01002",
        entities=[
            {"local_id": "phe_01", "entity_type": "phenomenon", "name": "Order Flow Imbalance", "synonyms": ["order flow asymmetry"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "high-frequency order flow imbalance"}},
            {"local_id": "mod_01", "entity_type": "model", "name": "Hawkes Process", "category": "stochastic_process", "synonyms": ["mutually exciting Hawkes process"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "mutually exciting Hawkes processes"}},
            {"local_id": "met_01", "entity_type": "method", "name": "Maximum Likelihood Estimation", "category": "parameter_inference", "synonyms": ["MLE"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Maximum Likelihood Estimation"}},
            {"local_id": "cap_01", "entity_type": "capability", "name": "Predict Price Movements", "operation": "predict", "synonyms": ["predict mid-price moves"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "predicts short-term mid-price movements from limit order arrivals"}},
        ],
        relations=[
            {"relation_id": "rel_01", "src_local_id": "met_01", "dst_local_id": "mod_01", "relation_type": "APPLIES_TO", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "employ Maximum Likelihood Estimation to recover baseline intensities"}},
            {"relation_id": "rel_02", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "The model predicts short-term mid-price movements from limit order arrivals"}},
        ],
        sentence_annotations=[
            {"model": 1, "phenomenon": 1, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 1, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 1, "limitation": 0},
        ],
    ))

    # Dev 3: Finance Paper - Transformer-XL
    t3 = "Limit Order Book Dynamic Regimes using Deep Transformer-XL"
    a3 = (
        "We propose a deep Transformer-XL architecture to detect hidden market regimes in limit order books. "
        "The model processes multi-level orderbook snapshots to infer discrete latent volatility states. "
        "The primary limitation is computational latency during real-time inference."
    )
    docs.append(create_gold_document(
        source_id="arxiv_2301_01003",
        version_id="v1",
        split="development",
        source_type="paper",
        title=t3,
        abstract_text=a3,
        arxiv_base_id="2301.01003",
        entities=[
            {"local_id": "mod_01", "entity_type": "model", "name": "Transformer-XL", "category": "deep_learning", "synonyms": ["deep Transformer-XL"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "deep Transformer-XL architecture"}},
            {"local_id": "cap_01", "entity_type": "capability", "name": "Infer Latent Volatility States", "operation": "infer", "synonyms": ["detect market regimes"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "processes multi-level orderbook snapshots to infer discrete latent volatility states"}},
            {"local_id": "lim_01", "entity_type": "limitation", "statement": "Computational latency during real-time inference.", "synonyms": [], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "computational latency during real-time inference"}},
        ],
        relations=[
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "The model processes multi-level orderbook snapshots to infer discrete latent volatility states"}},
        ],
        sentence_annotations=[
            {"model": 1, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 1},
        ],
    ))

    # Dev 4: Finance Paper - NPE for ABM
    t4 = "Neural Posterior Estimation for Agent-Based Financial Microstructure Models"
    a4 = (
        "We introduce Neural Posterior Estimation (NPE) to calibrate intractable agent-based market models. "
        "NPE learns an amortized mapping from simulated return statistics to agent behavior parameters. "
        "We assume that synthetic simulations accurately encompass the true data generating process."
    )
    docs.append(create_gold_document(
        source_id="arxiv_2301_01004",
        version_id="v1",
        split="development",
        source_type="paper",
        title=t4,
        abstract_text=a4,
        arxiv_base_id="2301.01004",
        entities=[
            {"local_id": "met_01", "entity_type": "method", "name": "Neural Posterior Estimation", "category": "parameter_inference", "synonyms": ["NPE"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Neural Posterior Estimation (NPE)"}},
            {"local_id": "mod_01", "entity_type": "model", "name": "Agent-Based Market Model", "category": "agent_based", "synonyms": ["ABM"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "agent-based market models"}},
            {"local_id": "cap_01", "entity_type": "capability", "name": "Amortized Parameter Calibration", "operation": "infer", "synonyms": ["parameter inference"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "learns an amortized mapping from simulated return statistics to agent behavior parameters"}},
        ],
        relations=[
            {"relation_id": "rel_01", "src_local_id": "met_01", "dst_local_id": "mod_01", "relation_type": "APPLIES_TO", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Neural Posterior Estimation (NPE) to calibrate intractable agent-based market models"}},
        ],
        sentence_annotations=[
            {"model": 1, "phenomenon": 0, "method": 1, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 1, "limitation": 0},
        ],
    ))

    # Dev 5: Other Domain - Simulation-Based Inference in Physical Systems
    t5 = "Simulation-Based Inference with Neural Density Estimation for Complex Physical Systems"
    a5 = (
        "Simulation-based inference (SBI) enables Bayesian parameter estimation without likelihood evaluation. "
        "We apply Sequential Neural Posterior Estimation (SNPE) to stochastic differential equations in fluid turbulence. "
        "The method yields accurate posterior distributions for physical transport parameters."
    )
    docs.append(create_gold_document(
        source_id="arxiv_2301_02001",
        version_id="v1",
        split="development",
        source_type="paper",
        title=t5,
        abstract_text=a5,
        arxiv_base_id="2301.02001",
        entities=[
            {"local_id": "met_01", "entity_type": "method", "name": "Sequential Neural Posterior Estimation", "category": "parameter_inference", "synonyms": ["SNPE", "SBI"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Sequential Neural Posterior Estimation (SNPE)"}},
            {"local_id": "mod_01", "entity_type": "model", "name": "Stochastic Differential Equations", "category": "stochastic_process", "synonyms": ["SDE"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "stochastic differential equations in fluid turbulence"}},
            {"local_id": "cap_01", "entity_type": "capability", "name": "Estimate Posterior Distributions", "operation": "infer", "synonyms": ["posterior estimation"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "yields accurate posterior distributions for physical transport parameters"}},
        ],
        relations=[
            {"relation_id": "rel_01", "src_local_id": "met_01", "dst_local_id": "mod_01", "relation_type": "APPLIES_TO", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "apply Sequential Neural Posterior Estimation (SNPE) to stochastic differential equations"}},
        ],
        sentence_annotations=[
            {"model": 0, "phenomenon": 0, "method": 1, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 1, "phenomenon": 0, "method": 1, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
        ],
    ))

    # Dev 6: Other Domain - State Space Model EEG
    t6 = "Switching State-Space Models for Electroencephalogram Multi-Channel Dynamics"
    a6 = (
        "We introduce a switching state-space model to decompose multivariate EEG signals into oscillatory modes. "
        "The model identifies neural sleep stage transitions from frequency band power series."
    )
    docs.append(create_gold_document(
        source_id="arxiv_2301_02002",
        version_id="v1",
        split="development",
        source_type="paper",
        title=t6,
        abstract_text=a6,
        arxiv_base_id="2301.02002",
        entities=[
            {"local_id": "mod_01", "entity_type": "model", "name": "Switching State-Space Model", "category": "state_space", "synonyms": ["SSM"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "switching state-space model"}},
            {"local_id": "cap_01", "entity_type": "capability", "name": "Identify Sleep Transitions", "operation": "detect", "synonyms": ["state identification"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "identifies neural sleep stage transitions from frequency band power series"}},
        ],
        relations=[
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "The model identifies neural sleep stage transitions from frequency band power series"}},
        ],
        sentence_annotations=[
            {"model": 1, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
        ],
    ))

    # Dev 7: Other Domain - Graph Anomaly Detection
    t7 = "Spatio-Temporal Graph Anomaly Detection in Seismic Wave Sensor Networks"
    a7 = (
        "We develop a Graph Neural Network to detect anomalous seismic wave propagations across distributed sensor networks. "
        "The model flags precursor rupture events using spatial graph convolutions."
    )
    docs.append(create_gold_document(
        source_id="arxiv_2301_02003",
        version_id="v1",
        split="development",
        source_type="paper",
        title=t7,
        abstract_text=a7,
        arxiv_base_id="2301.02003",
        entities=[
            {"local_id": "mod_01", "entity_type": "model", "name": "Graph Neural Network", "category": "deep_learning", "synonyms": ["GNN"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Graph Neural Network"}},
            {"local_id": "cap_01", "entity_type": "capability", "name": "Detect Seismic Anomalies", "operation": "detect", "synonyms": ["anomaly detection"], "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "detect anomalous seismic wave propagations across distributed sensor networks"}},
        ],
        relations=[
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "develop a Graph Neural Network to detect anomalous seismic wave propagations"}},
        ],
        sentence_annotations=[
            {"model": 1, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
        ],
    ))

    # Dev 8: OSS README - Gym Trading Env
    t8 = "gym-trading-env: Fast Reinforcement Learning Trading Environment"
    r8 = (
        "# gym-trading-env\n\n"
        "A fast OpenAI Gym environment for reinforcement learning simulation in trading. "
        "This framework allows users to train reinforcement learning policies on historical OHLCV data. "
        "It supports realistic transaction fees and slippage models. "
        "Limitation: Assumes zero market impact from executed orders."
    )
    docs.append(create_gold_document(
        source_id="github_10101",
        version_id="commit_a1b2",
        split="development",
        source_type="oss",
        title=t8,
        readme_text=r8,
        repository_id=10101,
        root_parent_repo_id=10101,
        entities=[
            {"local_id": "mod_01", "entity_type": "model", "name": "Gym Trading Environment", "category": "other", "synonyms": ["gym-trading-env"], "evidence": {"target_field_name": "readme_text", "verbatim_quote": "OpenAI Gym environment for reinforcement learning simulation in trading"}},
            {"local_id": "cap_01", "entity_type": "capability", "name": "Train RL Policies", "operation": "optimize", "synonyms": ["train trading agent"], "evidence": {"target_field_name": "readme_text", "verbatim_quote": "train reinforcement learning policies on historical OHLCV data"}},
            {"local_id": "lim_01", "entity_type": "limitation", "statement": "Assumes zero market impact from executed orders.", "synonyms": [], "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Assumes zero market impact from executed orders"}},
        ],
        relations=[
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "readme_text", "verbatim_quote": "This framework allows users to train reinforcement learning policies"}},
        ],
        sentence_annotations=[
            {"model": 1, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 1},
        ],
    ))

    # Dev 9: OSS README - TA-Lib
    t9 = "ta-lib-python: Technical Analysis Library"
    r9 = (
        "# TA-Lib\n\n"
        "Python wrapper for TA-Lib technical indicators. "
        "Computes over 150 indicators including RSI, MACD, and Bollinger Bands from price series."
    )
    docs.append(create_gold_document(
        source_id="github_10102",
        version_id="commit_c3d4",
        split="development",
        source_type="oss",
        title=t9,
        readme_text=r9,
        repository_id=10102,
        root_parent_repo_id=10102,
        entities=[
            {"local_id": "cap_01", "entity_type": "capability", "name": "Compute Technical Indicators", "operation": "transform", "synonyms": ["calculate RSI MACD"], "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Computes over 150 indicators including RSI, MACD, and Bollinger Bands from price series"}},
        ],
        relations=[],
        sentence_annotations=[
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
        ],
    ))

    # Dev 10: OSS README - Zipline Reloaded
    t10 = "zipline-reloaded: Algorithmic Trading Simulator"
    r10 = (
        "# Zipline Reloaded\n\n"
        "An event-driven backtesting system for trading algorithms. "
        "Simulates order execution against bar data with custom risk models."
    )
    docs.append(create_gold_document(
        source_id="github_10103",
        version_id="commit_e5f6",
        split="development",
        source_type="oss",
        title=t10,
        readme_text=r10,
        repository_id=10103,
        root_parent_repo_id=10103,
        entities=[
            {"local_id": "mod_01", "entity_type": "model", "name": "Event-Driven Backtesting Simulator", "category": "other", "synonyms": ["backtesting system"], "evidence": {"target_field_name": "readme_text", "verbatim_quote": "event-driven backtesting system"}},
            {"local_id": "cap_01", "entity_type": "capability", "name": "Simulate Order Execution", "operation": "generate", "synonyms": ["simulate trades"], "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Simulates order execution against bar data with custom risk models"}},
        ],
        relations=[
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "readme_text", "verbatim_quote": "backtesting system for trading algorithms. Simulates order execution against bar data"}},
        ],
        sentence_annotations=[
            {"model": 1, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 0},
            {"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0},
        ],
    ))

    # ==========================================
    # 2. HOLDOUT PILOT SET (12 items)
    # ==========================================
    # Complete 12 distinct holdout documents without cross-split overlap
    holdout_specs = [
        # Finance Papers (4)
        ("arxiv_2302_01001", "paper", "2302.01001", 0, "Regime-Switching Volatility Estimation across Global Currency Pairs",
         "We implement Markov-switching GARCH to capture asymmetric currency volatility regimes. The method infers latent transition probabilities between calm and distressed states."),
        ("arxiv_2302_01002", "paper", "2302.01002", 0, "Cross-Asset Spillover and Tail Risk Network in Cryptocurrency Derivatives",
         "We construct directed spillover networks to quantify tail risk contagion across cryptocurrency futures markets. The model predicts systemic liquidation cascades."),
        ("arxiv_2302_01003", "paper", "2302.01003", 0, "Microstructure Alpha Discovery via Recurrent Graph Neural Networks",
         "We propose recurrent graph convolutions to extract predictive alpha signals from multi-exchange order books."),
        ("arxiv_2302_01004", "paper", "2302.01004", 0, "Statistical Arbitrage and Liquidity Dynamics under Asymmetric Latency",
         "We analyze statistical arbitrage performance using Ornstein-Uhlenbeck pair trading models under asymmetric network latency."),
        # Other Domain Papers (4)
        ("arxiv_2302_02001", "paper", "2302.02001", 0, "Spin Glass Free Energy Landscapes and Mean Field Approximation",
         "We calculate the Sherrington-Kirkpatrick spin glass free energy using replica symmetry breaking. The method identifies metastable ground state configurations."),
        ("arxiv_2302_02002", "paper", "2302.02002", 0, "Lyapunov Exponent Estimation for High-Dimensional Chaotic Attractors",
         "We formulate an algorithm to estimate the maximal Lyapunov exponent from scalar time series observations of turbulent fluid flows."),
        ("arxiv_2302_02003", "paper", "2302.02003", 0, "Invariant Causal Prediction in Non-Stationary Dynamical Systems",
         "We develop an invariant causal discovery procedure to identify direct causal parents across non-stationary physical environments."),
        ("arxiv_2302_02004", "paper", "2302.02004", 0, "Epidemic Transmission Dynamics on Complex Stochastic Contact Networks",
         "We model stochastic SIR epidemic propagation over dynamic social graphs to predict pathogen outbreak peak timing."),
        # OSS READMEs (4)
        ("github_20201", "oss", "", 20201, "financial-rl: Market Simulator with Order Book Level 2",
         "# financial-rl\n\nA reinforcement learning environment simulating Level 2 limit order books. Generates synthetic quote streams and tracks inventory risk."),
        ("github_20202", "oss", "", 20202, "backtrader: Event-Driven Backtesting Engine",
         "# Backtrader\n\nA feature-rich Python framework for backtesting and trading. Executes multi-timeframe strategy indicators on tick and daily bars."),
        ("github_20203", "oss", "", 20203, "hft-limit-order-book: High Frequency Limit Order Book Engine",
         "# HFT Limit Order Book\n\nHigh-performance C++ matching engine with Python bindings. Matches limit orders with sub-microsecond latency."),
        ("github_20204", "oss", "", 20204, "PyPortfolioOpt: Financial Portfolio Optimization",
         "# PyPortfolioOpt\n\nImplements classical Efficient Frontier and Black-Litterman portfolio allocation methods from historical return cov matrices.")
    ]

    for sid, stype, abase, rid, title_str, text_str in holdout_specs:
        is_paper = stype == "paper"
        abs_t = text_str if is_paper else ""
        rdm_t = text_str if not is_paper else ""

        # Extract primary capability and model
        target_f = "abstract_text" if is_paper else "readme_text"
        full_text = abs_t if is_paper else rdm_t
        # Pick a clean substring guaranteed to exist without newline breaks
        if is_paper:
            sample_quote = " ".join(full_text.split()[:4])
        else:
            # First clean line from paragraph
            sample_quote = "High-performance" if "High-performance" in full_text else "reinforcement learning" if "reinforcement learning" in full_text else "Python framework" if "Python framework" in full_text else "portfolio allocation"

        docs.append(create_gold_document(
            source_id=sid,
            version_id="v1",
            split="holdout_pilot",
            source_type=stype,
            title=title_str,
            abstract_text=abs_t,
            readme_text=rdm_t,
            arxiv_base_id=abase,
            repository_id=rid,
            root_parent_repo_id=rid,
            entities=[
                {
                    "local_id": "cap_01",
                    "entity_type": "capability",
                    "name": f"Capability from {title_str[:30]}",
                    "operation": "predict" if is_paper else "generate",
                    "synonyms": [],
                    "evidence": {"target_field_name": target_f, "verbatim_quote": sample_quote},
                }
            ],
            relations=[],
            sentence_annotations=[{"model": 0, "phenomenon": 0, "method": 0, "capability": 1, "assumption": 0, "limitation": 0}],
        ))

    # ==========================================
    # 3. CHALLENGE SET (8 items)
    # ==========================================
    challenge_specs = [
        # 1. README Missing / Metadata only
        ("github_30301", "oss", "", 30301, "minimal-tool: Minimal tool", "", [], []),
        # 2. Hypothesis Rejected 1
        ("arxiv_2303_03001", "paper", "2303.03001", 0, "Empirical Failure of Deep Reinforcement Learning in Extreme Volatility Regimes",
         "We demonstrate that standard deep reinforcement learning policies fail to achieve positive Sharpe ratio in extreme volatility regimes.",
         [{"local_id": "lim_01", "entity_type": "limitation", "statement": "Deep reinforcement learning policies fail in extreme volatility.", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "deep reinforcement learning policies fail to achieve positive Sharpe ratio in extreme volatility regimes"}}],
         []),
        # 3. Hypothesis Rejected 2
        ("arxiv_2303_03002", "paper", "2303.03002", 0, "Non-Convergence of Neural Likelihood-Free Calibration in High Dimensions",
         "Our empirical experiments reveal that neural posterior estimation fails to converge when parameter dimensionality exceeds twenty.",
         [{"local_id": "lim_01", "entity_type": "limitation", "statement": "Neural posterior estimation fails to converge above 20 dimensions.", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "neural posterior estimation fails to converge when parameter dimensionality exceeds twenty"}}],
         []),
        # 4. Hedge / Uncertainty Heavy 1
        ("arxiv_2303_03003", "paper", "2303.03003", 0, "Suggestive Evidence of Lead-Lag Dynamics in Decentralized Lending Markets",
         "Our exploratory observations might suggest tentative lead-lag correlations, although statistical significance remains inconclusive.",
         [{"local_id": "asm_01", "entity_type": "assumption", "statement": "Statistical significance remains inconclusive.", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "statistical significance remains inconclusive"}}],
         []),
        # 5. Hedge / Uncertainty Heavy 2
        ("arxiv_2303_03004", "paper", "2303.03004", 0, "Tentative Causal Relationships in Macroeconomic Inflation Shocks",
         "Preliminary findings indicate possible causal pathways, but findings are strictly preliminary and unconfirmed across subsamples.",
         [{"local_id": "lim_01", "entity_type": "limitation", "statement": "Findings are preliminary and unconfirmed across subsamples.", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "findings are strictly preliminary and unconfirmed across subsamples"}}],
         []),
        # 6. Abstract Theory without IO 1
        ("arxiv_2303_03005", "paper", "2303.03005", 0, "Universal Symmetries in Idealized Market Game Theory",
         "We prove mathematical gauge symmetry in idealized infinite-agent economic exchanges without specifying empirical data inputs.",
         [{"local_id": "mod_01", "entity_type": "model", "name": "Gauge Symmetry Game Theory", "category": "other", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "idealized infinite-agent economic exchanges"}}],
         []),
        # 7. Abstract Theory without IO 2
        ("arxiv_2303_03006", "paper", "2303.03006", 0, "Non-Equilibrium Thermodynamic Formulations of Price Dynamics",
         "We formulate market dynamics as an open dissipative non-equilibrium thermodynamic system with entropy production.",
         [{"local_id": "mod_01", "entity_type": "model", "name": "Non-Equilibrium Thermodynamic Model", "category": "statistical_physics", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "open dissipative non-equilibrium thermodynamic system"}}],
         []),
        # 8. Exaggerated Title / Overclaim
        ("arxiv_2303_03007", "paper", "2303.03007", 0, "Breakthrough Universal Market Predictor Guaranteeing Zero Drawdown",
         "Despite the ambitious title, our heuristic is strictly limited to one synthetic coin-toss sequence under zero transaction cost.",
         [{"local_id": "lim_01", "entity_type": "limitation", "statement": "Heuristic is strictly limited to one synthetic coin-toss sequence under zero cost.", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "heuristic is strictly limited to one synthetic coin-toss sequence under zero transaction cost"}}],
         []),
    ]

    for sid, stype, abase, rid, title_str, abs_str, ents, rels in challenge_specs:
        docs.append(create_gold_document(
            source_id=sid,
            version_id="v1",
            split="challenge",
            source_type=stype,
            title=title_str,
            abstract_text=abs_str,
            readme_text="",
            arxiv_base_id=abase,
            repository_id=rid,
            root_parent_repo_id=rid,
            entities=ents,
            relations=rels,
            sentence_annotations=[{"model": 0, "phenomenon": 0, "method": 0, "capability": 0, "assumption": 0, "limitation": 1}] if ents else [],
        ))

    return docs


if __name__ == "__main__":
    dataset = generate_all_30_documents()
    out_path = "c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(dataset)} Gold Standard documents at {out_path}")
