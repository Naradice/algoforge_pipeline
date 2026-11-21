import json
import hashlib
from pathlib import Path

base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
out_path = base_dir / "tests/fixtures/gold_standard/holdout_expansion_er_benchmark.json"

er_benchmark = {
    "description": "Sealed Holdout Expansion Entity Resolution Benchmark (10 pairs: 3 SAME_AS, 3 VARIANT_OF, 4 DISTINCT)",
    "pairs": [
        # --- SAME_AS (3 pairs) ---
        {
            "pair_id": "er_exp_same_01",
            "entity_type": "software",
            "ground_truth": "SAME_AS",
            "mention_a": {
                "source_id": "github_30101",
                "mention_text": "NautilusTrader Trading Platform"
            },
            "mention_b": {
                "source_id": "synthetic_expansion_alias",
                "mention_text": "NautilusTrader Platform"
            },
            "direction": "bidirectional",
            "rationale": "Exact identical algorithmic trading platform referenced with slight variation in modifier."
        },
        {
            "pair_id": "er_exp_same_02",
            "entity_type": "model",
            "ground_truth": "SAME_AS",
            "mention_a": {
                "source_id": "arxiv_2303_01001",
                "mention_text": "Deep Equilibrium Model"
            },
            "mention_b": {
                "source_id": "synthetic_expansion_alias",
                "mention_text": "Deep Equilibrium Models (DEQ)"
            },
            "direction": "bidirectional",
            "rationale": "Exact identical model class with acronym expansion."
        },
        {
            "pair_id": "er_exp_same_03",
            "entity_type": "method",
            "ground_truth": "SAME_AS",
            "mention_a": {
                "source_id": "arxiv_2303_01005",
                "mention_text": "Physics-Informed Neural Networks"
            },
            "mention_b": {
                "source_id": "synthetic_expansion_alias",
                "mention_text": "Physics-Informed Neural Network (PINN)"
            },
            "direction": "bidirectional",
            "rationale": "Exact identical machine learning method singular/plural acronym variation."
        },

        # --- VARIANT_OF (3 pairs) ---
        {
            "pair_id": "er_exp_var_01",
            "entity_type": "model",
            "ground_truth": "VARIANT_OF",
            "mention_a": {
                "source_id": "arxiv_2303_01007",
                "mention_text": "Multivariate Hawkes Point Process"
            },
            "mention_b": {
                "source_id": "external_prior_concept",
                "mention_text": "Hawkes Point Process"
            },
            "direction": "a_is_variant_of_b",
            "rationale": "Multivariate Hawkes Point Process is a multi-dimensional extension/specialization of univariate Hawkes process."
        },
        {
            "pair_id": "er_exp_var_02",
            "entity_type": "model",
            "ground_truth": "VARIANT_OF",
            "mention_a": {
                "source_id": "arxiv_2303_01002",
                "mention_text": "Almgren-Chriss Transient Impact Model"
            },
            "mention_b": {
                "source_id": "external_prior_concept",
                "mention_text": "Almgren-Chriss Liquidation Framework"
            },
            "direction": "a_is_variant_of_b",
            "rationale": "Transient impact extension is a specialized variant of the canonical linear permanent impact Almgren-Chriss framework."
        },
        {
            "pair_id": "er_exp_var_03",
            "entity_type": "model",
            "ground_truth": "VARIANT_OF",
            "mention_a": {
                "source_id": "arxiv_2303_01004",
                "mention_text": "Regime-Switching Jump Diffusion Model"
            },
            "mention_b": {
                "source_id": "external_prior_concept",
                "mention_text": "Jump Diffusion Model"
            },
            "direction": "a_is_variant_of_b",
            "rationale": "Regime-switching jump diffusion incorporates hidden Markov states to modulate base jump diffusion parameters."
        },

        # --- DISTINCT Hard Negatives (4 pairs) ---
        {
            "pair_id": "er_exp_dist_01",
            "entity_type": "model",
            "ground_truth": "DISTINCT",
            "mention_a": {
                "source_id": "arxiv_2303_01001",
                "mention_text": "Deep Equilibrium Model"
            },
            "mention_b": {
                "source_id": "arxiv_2303_01003",
                "mention_text": "Graph Attention Network"
            },
            "direction": "none",
            "rationale": "Both are deep learning architectures, but DEQ is root-finding implicit layers while GAT is masked self-attention over graphs."
        },
        {
            "pair_id": "er_exp_dist_02",
            "entity_type": "method",
            "ground_truth": "DISTINCT",
            "mention_a": {
                "source_id": "arxiv_2303_01001",
                "mention_text": "Implicit Layer Root-Finding Calibration"
            },
            "mention_b": {
                "source_id": "github_30102",
                "mention_text": "GARCH Parameter Estimation"
            },
            "direction": "none",
            "rationale": "Both are volatility estimation methods, but implicit layer calibration solves non-linear fixed-point systems while GARCH uses MLE on return series."
        },
        {
            "pair_id": "er_exp_dist_03",
            "entity_type": "software",
            "ground_truth": "DISTINCT",
            "mention_a": {
                "source_id": "github_30102",
                "mention_text": "ARCH Python Library"
            },
            "mention_b": {
                "source_id": "github_30103",
                "mention_text": "CVXPY Convex Optimization Language"
            },
            "direction": "none",
            "rationale": "Both are scientific Python packages, but ARCH models econometric time series while CVXPY compiles convex conic optimization programs."
        },
        {
            "pair_id": "er_exp_dist_04",
            "entity_type": "method",
            "ground_truth": "DISTINCT",
            "mention_a": {
                "source_id": "arxiv_2303_01005",
                "mention_text": "Physics-Informed Neural Networks"
            },
            "mention_b": {
                "source_id": "arxiv_2303_01006",
                "mention_text": "Invariant Risk Minimization"
            },
            "direction": "none",
            "rationale": "Both are machine learning methodologies, but PINN embeds PDE differential operators in loss functions while IRM enforces causal invariance across training environments."
        }
    ]
}

content = json.dumps(er_benchmark, indent=2, ensure_ascii=False)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content)

sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
print(f"Created Holdout Expansion ER Benchmark: {out_path}")
print(f"Total pairs: {len(er_benchmark['pairs'])}")
print(f"SHA-256: {sha}")
