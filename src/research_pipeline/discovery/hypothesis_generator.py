"""Hypothesis Generator supporting revised experimental conditions H2-C0 through H2-C3 across 3 tasks."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from research_pipeline.discovery.models import GraphPath, HypothesisCandidate, AlgoForgeBacktestSpec


H2_SYSTEM_INSTRUCTION = """You are a senior quantitative finance researcher and algorithmic trading architect working on Study Manager Goal #774 ('FX Trade Strategy Discovery').
Your objective is to formulate a single novel, mathematically rigorous, and empirically testable algorithmic trading hypothesis for USDJPY execution in AlgoForge.

A valid scientific trading hypothesis must strictly satisfy:
1. Target Phenomenon: Concrete, documented financial market inefficiency or phenomenon.
2. Proposed Mechanism: Non-trivial mathematical, statistical, or computational mechanism grounded in the provided literature.
3. Analogical Source: Clear discipline, foundational model, or tool transferring the insight.
4. Core Claim: Specific causal proposition explaining how the mechanism addresses the phenomenon.
5. Grounding Rationale: Explanation of literature mechanisms or relational facts supporting the claim.
6. Testable Falsification Condition: Explicit quantitative metric threshold (e.g. Sharpe ratio, max drawdown, p-value, basis points) that disproves the hypothesis if failed.
7. AlgoForge Backtest Spec: Unambiguous operational specification including target asset ('USDJPY'), timeframe, feature engineering, entry condition, exit condition, risk management, benchmark baseline to beat, and falsification metric.

Output MUST be a single valid JSON object strictly matching the HypothesisCandidate schema.
"""


class TaskSpecification(BaseModel):
    """Specification of an independent hypothesis discovery task."""
    task_id: str
    task_name: str
    target_asset: str = "USDJPY"
    doc_ids: List[str]
    goal_description: str
    facts_nl: List[str]
    facts_kg: List[str]
    facts_evidence: List[Dict[str, str]]


TASKS: Dict[str, TaskSpecification] = {
    "task_1": TaskSpecification(
        task_id="task_1",
        task_name="Short-Term Price Movement & Order Flow Prediction",
        target_asset="USDJPY",
        doc_ids=["arxiv_2303_01006", "arxiv_2303_01007", "github_30101"],
        goal_description=(
            "Formulate a quantitative trading hypothesis for short-term USDJPY price movement and order flow prediction. "
            "Focus on distinguishing endogenous liquidity cascades from exogenous shocks and maintaining invariant predictability across market regimes."
        ),
        facts_nl=[
            "Fact 1: The capability to Predict Non-Stationary Order Flow is PROVIDED_BY Invariant Risk Minimization.",
            "Fact 2: The capability to Detect Endogenous Cascade Risks is PROVIDED_BY Multivariate Hawkes Point Process, which is a VARIANT_OF Hawkes Point Process.",
            "Fact 3: The capability to Execute Low-Latency Event-Driven Strategies is PROVIDED_BY NautilusTrader Trading Platform."
        ],
        facts_kg=[
            "[Predict Non-Stationary Order Flow (capability)] --(PROVIDED_BY)--> [Invariant Risk Minimization (method)]",
            "[Detect Endogenous Cascade Risks (capability)] --(PROVIDED_BY)--> [Multivariate Hawkes Point Process (model)] --(VARIANT_OF)--> [Hawkes Point Process (model)]",
            "[Execute Low-Latency Event-Driven Strategies (capability)] --(PROVIDED_BY)--> [NautilusTrader Trading Platform (software)]"
        ],
        facts_evidence=[
            {
                "fact": "IRM order flow predictability",
                "quote": "We evaluate invariant risk minimization for predicting non-stationary order flow, demonstrating stable out-of-distribution predictive performance across volatile market sessions."
            },
            {
                "fact": "Multivariate Hawkes cascade detection",
                "quote": "We apply multivariate Hawkes point processes to model high-frequency flash crash propagation in equity and foreign exchange limit order books."
            },
            {
                "fact": "NautilusTrader low-latency execution",
                "quote": "NautilusTrader is a production-grade algorithmic trading platform... Executes low-latency event-driven strategies with high-precision time synchronization."
            }
        ]
    ),
    "task_2": TaskSpecification(
        task_id="task_2",
        task_name="Volatility & Regime Adaptation",
        target_asset="USDJPY",
        doc_ids=["arxiv_2303_01004", "arxiv_2303_01001", "github_30102"],
        goal_description=(
            "Formulate a quantitative trading hypothesis for USDJPY volatility and regime adaptation. "
            "Focus on inferring latent jump probabilities and dynamic volatility surface recalibration during regime shifts."
        ),
        facts_nl=[
            "Fact 1: The capability to Infer Latent Jump Intensities is PROVIDED_BY Regime-Switching Jump Diffusion Model, which EXPLAINS Currency Rate Jump Dynamics.",
            "Fact 2: The Regime-Switching Jump Diffusion Model is a VARIANT_OF the Jump Diffusion Model family.",
            "Fact 3: The capability to Calibrate Volatility Surfaces is PROVIDED_BY Implicit Layer Root-Finding Calibration, which APPLIES_TO Deep Equilibrium Model.",
            "Fact 4: The ARCH Python Library IMPLEMENTS GARCH Parameter Estimation."
        ],
        facts_kg=[
            "[Infer Latent Jump Intensities (capability)] --(PROVIDED_BY)--> [Regime-Switching Jump Diffusion Model (model)] --(EXPLAINS)--> [Currency Rate Jump Dynamics (phenomenon)]",
            "[Regime-Switching Jump Diffusion Model (model)] --(VARIANT_OF)--> [Jump Diffusion Model (model)]",
            "[Calibrate Volatility Surfaces (capability)] --(PROVIDED_BY)--> [Implicit Layer Root-Finding Calibration (method)] --(APPLIES_TO)--> [Deep Equilibrium Model (model)]",
            "[ARCH Python Library (software)] --(IMPLEMENTS)--> [GARCH Parameter Estimation (method)]"
        ],
        facts_evidence=[
            {
                "fact": "Regime-switching jump diffusion in FX",
                "quote": "We develop a regime-switching jump diffusion model to characterize currency rate jump dynamics across crisis periods and infer latent jump intensities."
            },
            {
                "fact": "Jump diffusion model family",
                "quote": "We develop a regime-switching jump diffusion model to characterize currency rate jump dynamics..."
            },
            {
                "fact": "Deep equilibrium volatility calibration",
                "quote": "The proposed method enables gradient-based calibration of rough volatility surfaces with implicit layer root-finding without unrolling network iterations."
            },
            {
                "fact": "ARCH library GARCH estimation",
                "quote": "Implements GARCH parameter estimation and hypothesis testing routines for financial return series and volatility forecasting."
            }
        ]
    ),
    "task_3": TaskSpecification(
        task_id="task_3",
        task_name="Execution & Liquidation Cost Optimization",
        target_asset="USDJPY",
        doc_ids=["arxiv_2303_01002", "arxiv_2303_01005", "github_30103"],
        goal_description=(
            "Formulate a quantitative trading hypothesis for USDJPY execution and liquidation cost optimization. "
            "Focus on optimal liquidation trajectories under transient price impact and high-dimensional dynamic programming."
        ),
        facts_nl=[
            "Fact 1: The capability to Compute Optimal Liquidation Trajectories is PROVIDED_BY Almgren-Chriss Transient Impact Model, which EXPLAINS Transient Price Impact.",
            "Fact 2: The Almgren-Chriss Transient Impact Model is a VARIANT_OF the Almgren-Chriss Liquidation Framework.",
            "Fact 3: The capability to Calculate Equilibrium Policy Functions is PROVIDED_BY Physics-Informed Neural Networks, which APPLIES_TO Hamilton-Jacobi-Bellman Equation.",
            "Fact 4: The capability to Solve Constrained Portfolio Optimization is PROVIDED_BY CVXPY Convex Optimization Language, which IMPLEMENTS Disciplined Convex Programming."
        ],
        facts_kg=[
            "[Compute Optimal Liquidation Trajectories (capability)] --(PROVIDED_BY)--> [Almgren-Chriss Transient Impact Model (model)] --(EXPLAINS)--> [Transient Price Impact (phenomenon)]",
            "[Almgren-Chriss Transient Impact Model (model)] --(VARIANT_OF)--> [Almgren-Chriss Liquidation Framework (model)]",
            "[Calculate Equilibrium Policy Functions (capability)] --(PROVIDED_BY)--> [Physics-Informed Neural Networks (method)] --(APPLIES_TO)--> [Hamilton-Jacobi-Bellman Equation (model)]",
            "[Solve Constrained Portfolio Optimization (capability)] --(PROVIDED_BY)--> [CVXPY Convex Optimization Language (software)] --(IMPLEMENTS)--> [Disciplined Convex Programming (method)]"
        ],
        facts_evidence=[
            {
                "fact": "Almgren-Chriss transient liquidation",
                "quote": "We investigate optimal execution under transient price impact using an Almgren-Chriss framework... computes optimal liquidation trajectories balancing market risk against execution slippage."
            },
            {
                "fact": "Almgren-Chriss liquidation framework",
                "quote": "We investigate optimal execution under transient price impact using an Almgren-Chriss framework..."
            },
            {
                "fact": "PINN solving HJB equations",
                "quote": "We apply physics-informed neural networks to solve high-dimensional Hamilton-Jacobi-Bellman partial differential equations... calculates equilibrium policy functions without spatial discretization."
            },
            {
                "fact": "CVXPY disciplined convex programming",
                "quote": "CVXPY is a domain-specific language for convex optimization... Solves constrained portfolio optimization and trajectory problems."
            }
        ]
    )
}


def format_doc_content(doc: Dict[str, Any]) -> str:
    sid = doc.get("source_id", "")
    stype = doc.get("source_type", "")
    title = doc.get("title", doc.get("target_fields", {}).get("title_text", ""))
    tf_name = "readme_text" if stype == "oss" else "abstract_text"
    text = doc.get("target_fields", {}).get(tf_name, "")
    return f"Document [{sid}] ({stype}):\nTitle: {title}\nSummary Content:\n{text}\n"


def build_task_prompt(
    task: TaskSpecification,
    condition_id: str,
    docs: List[Dict[str, Any]],
) -> str:
    """Build standardized prompt for task across conditions H2-C0, H2-C1, H2-C2, H2-C3."""
    docs_text = "\n---\n".join(format_doc_content(d) for d in docs)
    
    sections = [
        f"TARGET GOAL: Study Manager Goal #774 - {task.task_name} (Asset: {task.target_asset})",
        f"DESCRIPTION:\n{task.goal_description}",
        f"PRIMARY LITERATURE SOURCES:\n\n{docs_text}",
    ]

    if condition_id == "H2-C0":
        # Raw source literature only
        pass

    elif condition_id == "H2-C1":
        # Same facts in unstructured natural language
        facts_str = "\n".join(task.facts_nl)
        sections.append(f"EXTRACTED DOMAIN FACTS (Unstructured Natural Language):\n{facts_str}")

    elif condition_id == "H2-C2":
        # Same facts structured as knowledge graph paths
        kg_str = "\n".join(f"- Path: {p}" for p in task.facts_kg)
        sections.append(f"STRUCTURED KNOWLEDGE GRAPH PATHS (Relational Graph Structure):\n{kg_str}")

    elif condition_id == "H2-C3":
        # Same facts structured as KG paths + verbatim quotes
        items = []
        for p, ev in zip(task.facts_kg, task.facts_evidence):
            items.append(f"- Path: {p}\n  Verbatim Source Evidence: \"{ev['quote']}\"")
        ev_str = "\n\n".join(items)
        sections.append(f"STRUCTURED KNOWLEDGE GRAPH PATHS & VERBATIM EVIDENCE:\n{ev_str}")

    else:
        raise ValueError(f"Unknown condition_id: {condition_id}")

    sections.append(
        "INSTRUCTIONS:\n"
        f"Synthesize the provided information to formulate a novel trading hypothesis for {task.target_asset} ({task.task_name}).\n"
        "Your output must be a single valid JSON object strictly conforming to the HypothesisCandidate schema:\n"
        "{\n"
        '  "hypothesis_id": "...",\n'
        f'  "condition_id": "{condition_id}",\n'
        f'  "task_id": "{task.task_id}",\n'
        '  "title": "...",\n'
        '  "target_phenomenon": "...",\n'
        '  "proposed_mechanism": "...",\n'
        '  "analogical_source": "...",\n'
        '  "claim": "...",\n'
        '  "grounding_rationale": "...",\n'
        '  "testable_falsification_condition": "...",\n'
        '  "backtest_spec": {\n'
        '    "target_asset": "USDJPY",\n'
        '    "timeframe": "M5",\n'
        '    "feature_engineering": "...",\n'
        '    "entry_condition": "...",\n'
        '    "exit_condition": "...",\n'
        '    "risk_management": "...",\n'
        '    "benchmark_baseline": "...",\n'
        '    "falsification_metric": "..."\n'
        '  }\n'
        "}"
    )

    return "\n\n========================================\n\n".join(sections)
