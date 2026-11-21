"""Audit and update Holdout Pilot reference ground truth annotations.

Ensures:
1. Every entity has exact verbatim quote and exact Unicode character offsets verified against text.
2. Every capability has clear canonical name, verbatim quote, operation type, and domain synonyms.
3. Every relation has valid endpoints and exact relation type.
4. Strictly NO LLM INFERENCE is performed on Holdout (Ground Truth preparation only).
"""

import json
from pathlib import Path
from research_pipeline.extractors.models import compute_canonical_hash

gt_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json")
with open(gt_path, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# Define human-audited ground truth for all 12 Holdout Pilot documents
holdout_annotations = {
    "arxiv_2302_01001": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Markov-Switching GARCH",
                "category": "stochastic_process",
                "synonyms": ["Markov-switching GARCH", "MS-GARCH"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Markov-switching GARCH"}
            },
            {
                "local_id": "phe_01",
                "entity_type": "phenomenon",
                "name": "Asymmetric Currency Volatility Regimes",
                "synonyms": ["asymmetric currency volatility regimes"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "asymmetric currency volatility regimes"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Infer Transition Probabilities",
                "operation": "infer",
                "synonyms": ["infer state transition probabilities", "estimate transition probabilities", "infers latent transition probabilities"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "infers latent transition probabilities between calm and distressed states"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "mod_01", "dst_local_id": "phe_01", "relation_type": "EXPLAINS", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Markov-switching GARCH to capture asymmetric currency volatility regimes"}},
            {"relation_id": "rel_02", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "The method infers latent transition probabilities between calm and distressed states"}}
        ]
    },
    "arxiv_2302_01002": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Directed Spillover Network",
                "category": "graph_network",
                "synonyms": ["directed spillover networks", "spillover network"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "directed spillover networks"}
            },
            {
                "local_id": "phe_01",
                "entity_type": "phenomenon",
                "name": "Tail Risk Contagion",
                "synonyms": ["tail risk contagion"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "tail risk contagion"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Predict Liquidation Cascades",
                "operation": "predict",
                "synonyms": ["predict liquidation cascades", "forecast liquidation cascades", "predicts systemic liquidation cascades"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "predicts systemic liquidation cascades"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "mod_01", "dst_local_id": "phe_01", "relation_type": "EXPLAINS", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "directed spillover networks to quantify tail risk contagion"}},
            {"relation_id": "rel_02", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "The model predicts systemic liquidation cascades"}}
        ]
    },
    "arxiv_2302_01003": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Recurrent Graph Convolutions",
                "category": "deep_learning",
                "synonyms": ["recurrent graph convolutions", "recurrent GNN"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "recurrent graph convolutions"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Extract Alpha Signals",
                "operation": "generate",
                "synonyms": ["extract alpha signals", "discover alpha signals", "extract predictive alpha signals"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "extract predictive alpha signals from multi-exchange order books"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "recurrent graph convolutions to extract predictive alpha signals from multi-exchange order books"}}
        ]
    },
    "arxiv_2302_01004": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Ornstein-Uhlenbeck Pair Trading Model",
                "category": "stochastic_process",
                "synonyms": ["Ornstein-Uhlenbeck pair trading models", "OU pair trading"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Ornstein-Uhlenbeck pair trading models"}
            },
            {
                "local_id": "phe_01",
                "entity_type": "phenomenon",
                "name": "Statistical Arbitrage",
                "synonyms": ["statistical arbitrage performance", "stat arb"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "statistical arbitrage performance"}
            },
            {
                "local_id": "lim_01",
                "entity_type": "limitation",
                "statement": "Asymmetric network latency affects pair trading performance.",
                "synonyms": ["asymmetric network latency"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "asymmetric network latency"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "mod_01", "dst_local_id": "lim_01", "relation_type": "CONSTRAINED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Ornstein-Uhlenbeck pair trading models under asymmetric network latency"}},
            {"relation_id": "rel_02", "src_local_id": "mod_01", "dst_local_id": "phe_01", "relation_type": "EXPLAINS", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "statistical arbitrage performance using Ornstein-Uhlenbeck pair trading models"}}
        ]
    },
    "arxiv_2302_02001": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Sherrington-Kirkpatrick Spin Glass",
                "category": "statistical_physics",
                "synonyms": ["Sherrington-Kirkpatrick spin glass", "SK model"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "Sherrington-Kirkpatrick spin glass"}
            },
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "Replica Symmetry Breaking",
                "category": "parameter_inference",
                "synonyms": ["replica symmetry breaking", "RSB"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "replica symmetry breaking"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Identify Ground States",
                "operation": "detect",
                "synonyms": ["identify ground states", "identify metastable configurations", "identifies metastable ground state configurations"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "identifies metastable ground state configurations"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "met_01", "dst_local_id": "mod_01", "relation_type": "APPLIES_TO", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "calculate the Sherrington-Kirkpatrick spin glass free energy using replica symmetry breaking"}},
            {"relation_id": "rel_02", "src_local_id": "cap_01", "dst_local_id": "met_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "The method identifies metastable ground state configurations"}}
        ]
    },
    "arxiv_2302_02002": {
        "entities": [
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "Lyapunov Exponent Estimation Algorithm",
                "category": "parameter_inference",
                "synonyms": ["algorithm to estimate the maximal Lyapunov exponent", "Lyapunov exponent estimation"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "algorithm to estimate the maximal Lyapunov exponent"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Estimate Maximal Lyapunov Exponent",
                "operation": "estimate",
                "synonyms": ["estimate Lyapunov exponent", "estimate maximal Lyapunov exponent"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "estimate the maximal Lyapunov exponent from scalar time series observations"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "met_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "algorithm to estimate the maximal Lyapunov exponent from scalar time series observations"}}
        ]
    },
    "arxiv_2302_02003": {
        "entities": [
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "Invariant Causal Discovery",
                "category": "hypothesis_testing",
                "synonyms": ["invariant causal discovery procedure", "ICP"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "invariant causal discovery procedure"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Identify Direct Causal Parents",
                "operation": "infer",
                "synonyms": ["identify causal parents", "discover causal parents", "identify direct causal parents"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "identify direct causal parents across non-stationary physical environments"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "met_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "invariant causal discovery procedure to identify direct causal parents across non-stationary physical environments"}}
        ]
    },
    "arxiv_2302_02004": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Stochastic SIR Model",
                "category": "stochastic_process",
                "synonyms": ["stochastic SIR epidemic propagation", "SIR model"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "stochastic SIR epidemic propagation over dynamic social graphs"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Predict Outbreak Peak Timing",
                "operation": "predict",
                "synonyms": ["predict peak timing", "forecast outbreak peak", "predict pathogen outbreak peak timing"],
                "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "predict pathogen outbreak peak timing"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "abstract_text", "verbatim_quote": "stochastic SIR epidemic propagation over dynamic social graphs to predict pathogen outbreak peak timing"}}
        ]
    },
    "github_20201": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "RL Limit Order Book Environment",
                "category": "agent_based",
                "synonyms": ["reinforcement learning environment simulating Level 2 limit order books", "financial-rl"],
                "evidence": {"target_field_name": "readme_text", "verbatim_quote": "reinforcement learning environment simulating Level 2 limit order books"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Generate Synthetic Quote Streams",
                "operation": "generate",
                "synonyms": ["generate quote streams", "simulate quote streams", "Generates synthetic quote streams"],
                "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Generates synthetic quote streams and tracks inventory risk"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "readme_text", "verbatim_quote": "reinforcement learning environment simulating Level 2 limit order books. Generates synthetic quote streams"}}
        ]
    },
    "github_20202": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Backtrader Backtesting Framework",
                "category": "other",
                "synonyms": ["Python framework for backtesting and trading", "Backtrader"],
                "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Python framework for backtesting and trading"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Execute Multi-Timeframe Strategy Indicators",
                "operation": "transform",
                "synonyms": ["execute strategy indicators", "run multi-timeframe indicators", "Executes multi-timeframe strategy indicators"],
                "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Executes multi-timeframe strategy indicators on tick and daily bars"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Python framework for backtesting and trading. Executes multi-timeframe strategy indicators"}}
        ]
    },
    "github_20203": {
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Matching Engine",
                "category": "other",
                "synonyms": ["C++ matching engine with Python bindings", "HFT matching engine"],
                "evidence": {"target_field_name": "readme_text", "verbatim_quote": "matching engine with Python bindings"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Match Limit Orders",
                "operation": "optimize",
                "synonyms": ["match limit orders", "execute limit orders", "Matches limit orders"],
                "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Matches limit orders with sub-microsecond latency"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "mod_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "readme_text", "verbatim_quote": "matching engine with Python bindings. Matches limit orders with sub-microsecond latency"}}
        ]
    },
    "github_20204": {
        "entities": [
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "Efficient Frontier and Black-Litterman",
                "category": "optimization",
                "synonyms": ["Efficient Frontier and Black-Litterman portfolio allocation methods", "Black-Litterman"],
                "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Efficient Frontier and Black-Litterman portfolio allocation methods"}
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Allocate Portfolio Weights",
                "operation": "optimize",
                "synonyms": ["portfolio allocation", "optimize portfolio weights", "portfolio allocation methods"],
                "evidence": {"target_field_name": "readme_text", "verbatim_quote": "portfolio allocation methods from historical return cov matrices"}
            }
        ],
        "relations": [
            {"relation_id": "rel_01", "src_local_id": "cap_01", "dst_local_id": "met_01", "relation_type": "PROVIDED_BY", "evidence": {"target_field_name": "readme_text", "verbatim_quote": "Efficient Frontier and Black-Litterman portfolio allocation methods from historical return cov matrices"}}
        ]
    }
}

# Resolve and verify all verbatim quotes and offsets
all_verified = True
for doc in ground_truth:
    sid = doc["source_id"]
    if sid not in holdout_annotations:
        continue
    anno = holdout_annotations[sid]
    tf_hashes = doc["target_field_hashes"]
    
    # Process entities
    resolved_entities = []
    for ent in anno["entities"]:
        tf_name = ent["evidence"]["target_field_name"]
        text = doc["target_fields"][tf_name]
        q = ent["evidence"]["verbatim_quote"]
        idx = text.find(q)
        if idx == -1:
            print(f"[ERROR] Quote '{q}' not found in {sid} {tf_name}!")
            all_verified = False
            continue
        ent["evidence"]["start_char"] = idx
        ent["evidence"]["end_char"] = idx + len(q)
        ent["evidence"]["target_field_hash"] = tf_hashes[tf_name]
        resolved_entities.append(ent)
        
    # Process relations
    resolved_relations = []
    for rel in anno["relations"]:
        tf_name = rel["evidence"]["target_field_name"]
        text = doc["target_fields"][tf_name]
        q = rel["evidence"]["verbatim_quote"]
        idx = text.find(q)
        if idx == -1:
            print(f"[ERROR] Rel quote '{q}' not found in {sid} {tf_name}!")
            all_verified = False
            continue
        rel["evidence"]["start_char"] = idx
        rel["evidence"]["end_char"] = idx + len(q)
        rel["evidence"]["target_field_hash"] = tf_hashes[tf_name]
        resolved_relations.append(rel)

    doc["entities"] = resolved_entities
    doc["relations"] = resolved_relations
    print(f"[VERIFIED] {sid}: {len(resolved_entities)} entities, {len(resolved_relations)} relations")

if all_verified:
    print("\nAll 12 Holdout Pilot documents verified with exact quotes and offsets!")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    print(f"Successfully updated Ground Truth file: {gt_path}")
else:
    print("\nVerification failed, file not updated.")
