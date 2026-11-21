"""Create Holdout Expansion Ground Truth (10 new documents: 7 papers, 3 OSS).

Adheres strictly to:
- 7-Class Ontology (Model, Method, Software, Capability, Phenomenon, Limitation, Assumption)
- Exact Unicode character offsets and SHA-256 target field hashes
- Strictly ZERO LLM inference calls performed (Sealed Ground Truth)
"""

import json
from pathlib import Path
from research_pipeline.extractors.models import compute_canonical_hash

expansion_docs = [
    {
        "source_id": "arxiv_2303_01001",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "paper",
        "title": "Stochastic Volatility Calibration via Deep Equilibrium Models",
        "abstract_text": "We formulate stochastic volatility calibration as a root-finding problem in deep equilibrium models. The proposed method enables gradient-based calibration of rough volatility surfaces with implicit layer depth.",
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Deep Equilibrium Model",
                "category": "deep_learning",
                "synonyms": ["deep equilibrium models", "DEQ"],
                "quote": "deep equilibrium models"
            },
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "Implicit Layer Root-Finding Calibration",
                "category": "parameter_inference",
                "synonyms": ["stochastic volatility calibration", "gradient-based calibration"],
                "quote": "gradient-based calibration of rough volatility surfaces"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Calibrate Volatility Surfaces",
                "operation": "optimize",
                "synonyms": ["calibrate rough volatility surfaces", "calibrate volatility surfaces"],
                "quote": "calibration of rough volatility surfaces with implicit layer depth"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "met_01",
                "dst_local_id": "mod_01",
                "relation_type": "APPLIES_TO",
                "quote": "stochastic volatility calibration as a root-finding problem in deep equilibrium models"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "met_01",
                "relation_type": "PROVIDED_BY",
                "quote": "method enables gradient-based calibration of rough volatility surfaces"
            }
        ]
    },
    {
        "source_id": "arxiv_2303_01002",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "paper",
        "title": "Optimal Execution under Transient Price Impact and Linear Quadratic Controls",
        "abstract_text": "We investigate optimal execution under transient price impact using an Almgren-Chriss framework with linear quadratic controls. The model computes optimal liquidation trajectories that balance market risk against execution slippage.",
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Almgren-Chriss Transient Impact Model",
                "category": "stochastic_process",
                "synonyms": ["Almgren-Chriss framework", "transient impact model"],
                "quote": "Almgren-Chriss framework with linear quadratic controls"
            },
            {
                "local_id": "phe_01",
                "entity_type": "phenomenon",
                "name": "Transient Price Impact",
                "synonyms": ["transient price impact"],
                "quote": "transient price impact"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Compute Optimal Liquidation Trajectories",
                "operation": "optimize",
                "synonyms": ["optimal execution", "computes optimal liquidation trajectories"],
                "quote": "computes optimal liquidation trajectories that balance market risk against execution slippage"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "mod_01",
                "dst_local_id": "phe_01",
                "relation_type": "EXPLAINS",
                "quote": "optimal execution under transient price impact using an Almgren-Chriss framework"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "mod_01",
                "relation_type": "PROVIDED_BY",
                "quote": "The model computes optimal liquidation trajectories"
            }
        ]
    },
    {
        "source_id": "arxiv_2303_01003",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "paper",
        "title": "Graph Attention Networks for Cross-Border Financial Contagion",
        "abstract_text": "We introduce graph attention networks to capture cross-border financial contagion across global banking networks. The architecture forecasts systemic default cascades by attending to dense interbank exposures.",
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Graph Attention Network",
                "category": "deep_learning",
                "synonyms": ["graph attention networks", "GAT"],
                "quote": "graph attention networks"
            },
            {
                "local_id": "phe_01",
                "entity_type": "phenomenon",
                "name": "Cross-Border Financial Contagion",
                "synonyms": ["cross-border financial contagion", "financial contagion"],
                "quote": "cross-border financial contagion"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Forecast Systemic Default Cascades",
                "operation": "predict",
                "synonyms": ["forecast default cascades", "forecasts systemic default cascades"],
                "quote": "forecasts systemic default cascades by attending to dense interbank exposures"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "mod_01",
                "dst_local_id": "phe_01",
                "relation_type": "EXPLAINS",
                "quote": "graph attention networks to capture cross-border financial contagion"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "mod_01",
                "relation_type": "PROVIDED_BY",
                "quote": "The architecture forecasts systemic default cascades"
            }
        ]
    },
    {
        "source_id": "arxiv_2303_01004",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "paper",
        "title": "Regime-Switching Jump Diffusion Models for Foreign Exchange Parities",
        "abstract_text": "We develop a regime-switching jump diffusion model to characterize currency rate jump dynamics during central bank interest rate announcements. The formulation infers latent jump intensities across crisis periods.",
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Regime-Switching Jump Diffusion Model",
                "category": "stochastic_process",
                "synonyms": ["regime-switching jump diffusion model", "jump diffusion"],
                "quote": "regime-switching jump diffusion model"
            },
            {
                "local_id": "phe_01",
                "entity_type": "phenomenon",
                "name": "Currency Rate Jump Dynamics",
                "synonyms": ["currency rate jump dynamics", "jump dynamics"],
                "quote": "currency rate jump dynamics"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Infer Latent Jump Intensities",
                "operation": "infer",
                "synonyms": ["infer jump intensities", "infers latent jump intensities"],
                "quote": "infers latent jump intensities across crisis periods"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "mod_01",
                "dst_local_id": "phe_01",
                "relation_type": "EXPLAINS",
                "quote": "regime-switching jump diffusion model to characterize currency rate jump dynamics"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "mod_01",
                "relation_type": "PROVIDED_BY",
                "quote": "The formulation infers latent jump intensities across crisis periods"
            }
        ]
    },
    {
        "source_id": "arxiv_2303_01005",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "paper",
        "title": "Physics-Informed Neural Networks for High-Dimensional Hamilton-Jacobi-Bellman Equations",
        "abstract_text": "We apply physics-informed neural networks to solve high-dimensional Hamilton-Jacobi-Bellman partial differential equations arising in multi-agent market equilibrium. The algorithm calculates equilibrium policy functions without spatial discretization.",
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Hamilton-Jacobi-Bellman Equation",
                "category": "other",
                "synonyms": ["Hamilton-Jacobi-Bellman partial differential equations", "HJB equation"],
                "quote": "Hamilton-Jacobi-Bellman partial differential equations"
            },
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "Physics-Informed Neural Networks",
                "category": "deep_learning",
                "synonyms": ["physics-informed neural networks", "PINN"],
                "quote": "physics-informed neural networks"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Calculate Equilibrium Policy Functions",
                "operation": "infer",
                "synonyms": ["solve HJB equations", "calculates equilibrium policy functions"],
                "quote": "calculates equilibrium policy functions without spatial discretization"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "met_01",
                "dst_local_id": "mod_01",
                "relation_type": "APPLIES_TO",
                "quote": "physics-informed neural networks to solve high-dimensional Hamilton-Jacobi-Bellman partial differential equations"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "met_01",
                "relation_type": "PROVIDED_BY",
                "quote": "The algorithm calculates equilibrium policy functions without spatial discretization"
            }
        ]
    },
    {
        "source_id": "arxiv_2303_01006",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "paper",
        "title": "Invariant Risk Minimization for Non-Stationary Order Flow Prediction",
        "abstract_text": "We evaluate invariant risk minimization for predicting non-stationary order flow across fragmented trading venues. Out-of-distribution regime shifts degrade out-of-sample predictability when latent market environments change rapidly.",
        "entities": [
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "Invariant Risk Minimization",
                "category": "optimization",
                "synonyms": ["invariant risk minimization", "IRM"],
                "quote": "invariant risk minimization"
            },
            {
                "local_id": "lim_01",
                "entity_type": "limitation",
                "statement": "Out-of-distribution regime shifts degrade out-of-sample predictability.",
                "synonyms": ["regime shifts degrade predictability"],
                "quote": "Out-of-distribution regime shifts degrade out-of-sample predictability"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Predict Non-Stationary Order Flow",
                "operation": "predict",
                "synonyms": ["predicting non-stationary order flow", "order flow prediction"],
                "quote": "predicting non-stationary order flow across fragmented trading venues"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "cap_01",
                "dst_local_id": "met_01",
                "relation_type": "PROVIDED_BY",
                "quote": "invariant risk minimization for predicting non-stationary order flow"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "met_01",
                "dst_local_id": "lim_01",
                "relation_type": "CONSTRAINED_BY",
                "quote": "invariant risk minimization for predicting non-stationary order flow across fragmented trading venues. Out-of-distribution regime shifts degrade out-of-sample predictability"
            }
        ]
    },
    {
        "source_id": "arxiv_2303_01007",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "paper",
        "title": "Multivariate Hawkes Point Processes for High-Frequency Flash Crashes",
        "abstract_text": "We apply multivariate Hawkes point processes to model high-frequency flash crash propagation in equity markets. The system detects endogenous cascade risks generated by mutual order cancellation feedback.",
        "entities": [
            {
                "local_id": "mod_01",
                "entity_type": "model",
                "name": "Multivariate Hawkes Point Process",
                "category": "stochastic_process",
                "synonyms": ["multivariate Hawkes point processes", "multivariate Hawkes"],
                "quote": "multivariate Hawkes point processes"
            },
            {
                "local_id": "phe_01",
                "entity_type": "phenomenon",
                "name": "High-Frequency Flash Crash Propagation",
                "synonyms": ["flash crash propagation", "flash crash"],
                "quote": "high-frequency flash crash propagation"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Detect Endogenous Cascade Risks",
                "operation": "detect",
                "synonyms": ["detects cascade risks", "detects endogenous cascade risks"],
                "quote": "detects endogenous cascade risks generated by mutual order cancellation feedback"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "mod_01",
                "dst_local_id": "phe_01",
                "relation_type": "EXPLAINS",
                "quote": "multivariate Hawkes point processes to model high-frequency flash crash propagation"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "mod_01",
                "relation_type": "PROVIDED_BY",
                "quote": "The system detects endogenous cascade risks generated by mutual order cancellation feedback"
            }
        ]
    },
    {
        "source_id": "github_30101",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "oss",
        "title": "nautilus_trader: Production-grade algorithmic trading platform in Rust and Python",
        "readme_text": "NautilusTrader is a production-grade algorithmic trading platform written in Rust and Python. Executes low-latency event-driven strategies on tick-by-tick financial market data with native WebSocket exchange connectors.",
        "entities": [
            {
                "local_id": "soft_01",
                "entity_type": "software",
                "name": "NautilusTrader Trading Platform",
                "category": "execution_platform",
                "synonyms": ["NautilusTrader", "algorithmic trading platform"],
                "quote": "NautilusTrader is a production-grade algorithmic trading platform"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Execute Low-Latency Event-Driven Strategies",
                "operation": "transform",
                "synonyms": ["execute trading strategies", "Executes low-latency event-driven strategies"],
                "quote": "Executes low-latency event-driven strategies on tick-by-tick financial market data"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "cap_01",
                "dst_local_id": "soft_01",
                "relation_type": "PROVIDED_BY",
                "quote": "NautilusTrader is a production-grade algorithmic trading platform written in Rust and Python. Executes low-latency event-driven strategies"
            }
        ]
    },
    {
        "source_id": "github_30102",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "oss",
        "title": "arch: Autoregressive Conditional Heteroskedasticity and Volatility Modeling in Python",
        "readme_text": "arch is a Python library for autoregressive conditional heteroskedasticity models. Implements GARCH parameter estimation and hypothesis testing routines for financial return series.",
        "entities": [
            {
                "local_id": "soft_01",
                "entity_type": "software",
                "name": "ARCH Python Library",
                "category": "analytics_library",
                "synonyms": ["arch", "arch library"],
                "quote": "arch is a Python library"
            },
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "GARCH Parameter Estimation",
                "category": "parameter_inference",
                "synonyms": ["GARCH parameter estimation", "volatility estimation"],
                "quote": "GARCH parameter estimation and hypothesis testing routines"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Estimate Volatility Models",
                "operation": "infer",
                "synonyms": ["GARCH estimation", "parameter estimation"],
                "quote": "GARCH parameter estimation and hypothesis testing routines for financial return series"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "soft_01",
                "dst_local_id": "met_01",
                "relation_type": "IMPLEMENTS",
                "quote": "arch is a Python library for autoregressive conditional heteroskedasticity models. Implements GARCH parameter estimation"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "soft_01",
                "relation_type": "PROVIDED_BY",
                "quote": "arch is a Python library for autoregressive conditional heteroskedasticity models. Implements GARCH parameter estimation and hypothesis testing routines for financial return series"
            }
        ]
    },
    {
        "source_id": "github_30103",
        "version_id": "v1",
        "split": "holdout_expansion",
        "source_type": "oss",
        "title": "cvxpy: Domain-specific language for convex optimization problems in Python",
        "readme_text": "CVXPY is a domain-specific language for convex optimization embedded in Python. Solves constrained portfolio optimization and risk parity problems with disciplined convex programming.",
        "entities": [
            {
                "local_id": "soft_01",
                "entity_type": "software",
                "name": "CVXPY Convex Optimization Language",
                "category": "analytics_library",
                "synonyms": ["CVXPY", "convex optimization library"],
                "quote": "CVXPY is a domain-specific language for convex optimization embedded in Python"
            },
            {
                "local_id": "met_01",
                "entity_type": "method",
                "name": "Disciplined Convex Programming",
                "category": "optimization",
                "synonyms": ["disciplined convex programming", "DCP"],
                "quote": "disciplined convex programming"
            },
            {
                "local_id": "cap_01",
                "entity_type": "capability",
                "name": "Solve Constrained Portfolio Optimization",
                "operation": "optimize",
                "synonyms": ["portfolio optimization", "Solves constrained portfolio optimization"],
                "quote": "Solves constrained portfolio optimization and risk parity problems"
            }
        ],
        "relations": [
            {
                "relation_id": "rel_01",
                "src_local_id": "soft_01",
                "dst_local_id": "met_01",
                "relation_type": "IMPLEMENTS",
                "quote": "CVXPY is a domain-specific language for convex optimization embedded in Python. Solves constrained portfolio optimization and risk parity problems with disciplined convex programming"
            },
            {
                "relation_id": "rel_02",
                "src_local_id": "cap_01",
                "dst_local_id": "soft_01",
                "relation_type": "PROVIDED_BY",
                "quote": "CVXPY is a domain-specific language for convex optimization embedded in Python. Solves constrained portfolio optimization"
            }
        ]
    }
]

# Verify all quotes and build verified records
verified_docs = []
all_valid = True

for doc in expansion_docs:
    sid = doc["source_id"]
    tf_name = "readme_text" if doc["source_type"] == "oss" else "abstract_text"
    text = doc[tf_name]
    title = doc["title"]
    
    fields = {"title_text": title, tf_name: text}
    hashes = {k: compute_canonical_hash(v) for k, v in fields.items()}
    
    verified_entities = []
    for ent in doc["entities"]:
        q = ent["quote"]
        idx = text.find(q)
        if idx == -1:
            print(f"[ERROR] Quote '{q}' not found in {sid}!")
            all_valid = False
            continue
        verified_entities.append({
            "local_id": ent["local_id"],
            "entity_type": ent["entity_type"],
            "name": ent.get("name"),
            "statement": ent.get("statement"),
            "category": ent.get("category"),
            "synonyms": ent.get("synonyms", []),
            "evidence": {
                "target_field_name": tf_name,
                "target_field_hash": hashes[tf_name],
                "verbatim_quote": q,
                "start_char": idx,
                "end_char": idx + len(q),
                "raw_start_char": idx,
                "raw_end_char": idx + len(q),
                "raw_offset_valid": True,
                "resolved_offset_valid": True,
                "offset_resolution_status": "unchanged"
            }
        })
        
    verified_relations = []
    for rel in doc["relations"]:
        q = rel["quote"]
        idx = text.find(q)
        if idx == -1:
            print(f"[ERROR] Rel quote '{q}' not found in {sid}!")
            all_valid = False
            continue
        verified_relations.append({
            "relation_id": rel["relation_id"],
            "src_local_id": rel["src_local_id"],
            "dst_local_id": rel["dst_local_id"],
            "relation_type": rel["relation_type"],
            "evidence": {
                "target_field_name": tf_name,
                "target_field_hash": hashes[tf_name],
                "verbatim_quote": q,
                "start_char": idx,
                "end_char": idx + len(q),
                "raw_start_char": idx,
                "raw_end_char": idx + len(q),
                "raw_offset_valid": True,
                "resolved_offset_valid": True,
                "offset_resolution_status": "unchanged"
            }
        })
        
    doc_record = {
        "source_id": sid,
        "version_id": doc["version_id"],
        "split": doc["split"],
        "source_type": doc["source_type"],
        "title": title,
        "target_fields": fields,
        "target_field_hashes": hashes,
        "entities": verified_entities,
        "relations": verified_relations
    }
    verified_docs.append(doc_record)
    print(f"[VERIFIED] {sid}: {len(verified_entities)} entities, {len(verified_relations)} relations")

if all_valid:
    out_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/holdout_expansion_ground_truth.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verified_docs, f, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully created sealed Holdout Expansion Ground Truth: {out_path}")
    print(f"Total documents: {len(verified_docs)}")
    print(f"Total entities: {sum(len(d['entities']) for d in verified_docs)}")
    print(f"Total relations: {sum(len(d['relations']) for d in verified_docs)}")
else:
    print("\nErrors encountered, file not written.")
