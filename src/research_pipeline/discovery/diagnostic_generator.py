"""Prompt Generator for HDE Step H2-D1 Diagnostic Experiment.

Guarantees:
- C0 (NL) and C1 (KG) share 100% identical concepts, relations, and subjects.
- C2 augments C1 with mathematically accurate Capability IO, state/control variables, and assumptions.
- Prompt structure, system instruction, and output schema are identical across all conditions.
"""

from typing import Dict, Any, List
from research_pipeline.discovery.diagnostic_models import ConceptSpecification


DIAGNOSTIC_SYSTEM_INSTRUCTION = """You are a senior quantitative finance researcher and algorithmic trading architect working on Study Manager Goal #774 ('FX Trade Strategy Discovery').
Your objective is to formulate a single mathematically rigorous, empirically testable, and operable algorithmic trading hypothesis for USDJPY execution in AlgoForge.

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


CONCEPTS: Dict[str, ConceptSpecification] = {
    "almgren_chriss": ConceptSpecification(
        concept_id="almgren_chriss",
        concept_name="Almgren-Chriss Transient Impact Model",
        category="execution_control",
        doc_id="arxiv_2303_01002",
        facts_nl=[
            "Fact 1: The Almgren-Chriss Transient Impact Model EXPLAINS Transient Price Impact in financial markets.",
            "Fact 2: The capability to Compute Optimal Liquidation Trajectories is PROVIDED_BY Almgren-Chriss Transient Impact Model.",
            "Fact 3: Almgren-Chriss Transient Impact Model is a VARIANT_OF Almgren-Chriss Liquidation Framework."
        ],
        facts_kg=[
            "[Almgren-Chriss Transient Impact Model (model)] --(EXPLAINS)--> [Transient Price Impact (phenomenon)]",
            "[Compute Optimal Liquidation Trajectories (capability)] --(PROVIDED_BY)--> [Almgren-Chriss Transient Impact Model (model)]",
            "[Almgren-Chriss Transient Impact Model (model)] --(VARIANT_OF)--> [Almgren-Chriss Liquidation Framework (model)]"
        ],
        math_io_specification=(
            "- Capability: Compute Optimal Liquidation Trajectories\n"
            "- Mathematical Inputs: Initial inventory quantity X_0 (scalar shares), target liquidation horizon T (scalar time), risk aversion parameter lambda > 0, price volatility sigma, transient impact kernel parameters (impact scale eta, decay rate rho).\n"
            "- Mathematical Outputs: Inventory trajectory x(t) in [0, X_0] representing remaining position size over time t in [0, T], and trading speed schedule v(t) = -dx(t)/dt.\n"
            "- Objective Functional: Minimize expected execution cost plus risk variance: E[Cost] + lambda * Var[Cost]."
        ),
        state_control_variables=(
            "- State Variable: Inventory remaining x(t) in [0, X_0], and accumulated transient impact state I(t) = integral_0^t eta * exp(-rho*(t-s)) * v(s) ds.\n"
            "- Control Variable: Liquidation speed v(t) >= 0 (rate of order execution at time t).\n"
            "- Exogenous Process: Mid-market price S(t) = S_0 + sigma * W(t) - integral_0^t gamma * v(s) ds - I(t).\n"
            "- Crucial Semantic Distinction: The trajectory x(t) is an inventory schedule of remaining shares to execute, NOT a price forecast S(t). This model executes pre-existing positions and is not a speculative trade entry signal."
        ),
        assumptions_and_limitations=(
            "- Pure Liquidation Assumption: In the standard liquidation setting, inventory x(t) decreases monotonically from X_0 to 0 (no speculative round-trip buying).\n"
            "- Transient Impact Kernel: Price impact decays exponentially with rate rho; calibrating (eta, rho) requires transaction fill logs or order book event data, not raw aggregated OHLC bars alone."
        )
    ),
    "hawkes_process": ConceptSpecification(
        concept_id="hawkes_process",
        concept_name="Multivariate Hawkes Point Process",
        category="point_process",
        doc_id="arxiv_2303_01007",
        facts_nl=[
            "Fact 1: Multivariate Hawkes Point Processes EXPLAIN High-Frequency Flash Crash Propagation in limit order books.",
            "Fact 2: The capability to Detect Endogenous Cascade Risks is PROVIDED_BY Multivariate Hawkes Point Process.",
            "Fact 3: Multivariate Hawkes Point Process is a VARIANT_OF Hawkes Point Process."
        ],
        facts_kg=[
            "[Multivariate Hawkes Point Process (model)] --(EXPLAINS)--> [High-Frequency Flash Crash Propagation (phenomenon)]",
            "[Detect Endogenous Cascade Risks (capability)] --(PROVIDED_BY)--> [Multivariate Hawkes Point Process (model)]",
            "[Multivariate Hawkes Point Process (model)] --(VARIANT_OF)--> [Hawkes Point Process (model)]"
        ],
        math_io_specification=(
            "- Capability: Detect Endogenous Cascade Risks\n"
            "- Mathematical Inputs: Discrete event arrival sequence with timestamps {t_i}_{i=1}^N and event marks m_i in {1, ..., D} (e.g. order arrivals, trades, cancellations), having sufficient timestamp resolution to distinguish discrete event ordering.\n"
            "- Mathematical Outputs: Conditional event intensity vector lambda(t) = [lambda_1(t), ..., lambda_D(t)]^T, and cross-excitation branching matrix Gamma = [alpha_{jk} / beta_{jk}].\n"
            "- Mathematical Stability Condition: Spectral radius rho(Gamma) < 1 ensures subcritical stationarity (no infinite cascading explosion)."
        ),
        state_control_variables=(
            "- State Variable: Instantaneous conditional intensity lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta*(t - t_i)).\n"
            "- Control Variable: None (Hawkes process is a point process statistical estimation model, not an optimal control or execution policy).\n"
            "- Crucial Semantic Distinction: Intensity lambda(t) represents the instantaneous rate of event arrivals (cascading risk), NOT future price return mu_R or price level S(t)."
        ),
        assumptions_and_limitations=(
            "- Event Sequence Requirement: Mathematically requires timestamped discrete event sequences. A fixed-interval M5 OHLC bar cannot uniquely identify discrete order arrivals without an intraday event reconstruction mechanism or microstructure proxy.\n"
            "- Parametric Assumption: Exponential memory decay kernel is assumed; estimation is sensitive to parameter initialization."
        )
    ),
    "irm": ConceptSpecification(
        concept_id="irm",
        concept_name="Invariant Risk Minimization (IRM)",
        category="robust_prediction",
        doc_id="arxiv_2303_01006",
        facts_nl=[
            "Fact 1: Invariant Risk Minimization ADDRESSES Out-of-distribution regime shifts degrade out-of-sample predictability.",
            "Fact 2: The capability to Predict Non-Stationary Order Flow is PROVIDED_BY Invariant Risk Minimization."
        ],
        facts_kg=[
            "[Invariant Risk Minimization (method)] --(ADDRESSES)--> [Out-of-distribution regime shifts degrade out-of-sample predictability (limitation)]",
            "[Predict Non-Stationary Order Flow (capability)] --(PROVIDED_BY)--> [Invariant Risk Minimization (method)]"
        ],
        math_io_specification=(
            "- Capability: Predict Non-Stationary Order Flow\n"
            "- Mathematical Inputs: Multi-environment training dataset D = {(x_i^e, y_i^e)}_{e in E}, where E represents distinct market regimes or environments (e.g. trading sessions, volatility regimes).\n"
            "- Mathematical Outputs: Feature representation mapping Phi(x) and invariant linear predictor w such that w o Phi minimizes empirical risk while simultaneously satisfying invariance across environments.\n"
            "- Objective Formulation: min_{Phi, w} sum_{e in E} R^e(w o Phi) + lambda * ||grad_{w|w=1.0} R^e(w o Phi)||^2."
        ),
        state_control_variables=(
            "- State Variable: Extracted feature representation Phi(x) and linear predictor weights w.\n"
            "- Control Variable: None (IRM is a supervised representation learning / regression objective, not an execution scheduler).\n"
            "- Crucial Semantic Distinction: IRM produces directional predictions robust across training environments E; it is not an execution algorithm and does not eliminate market risk."
        ),
        assumptions_and_limitations=(
            "- Scope of Invariance: IRM optimizes for invariance across the observed training environments E; it does NOT mathematically guarantee causal truth or universal out-of-distribution invariance under arbitrary, unseen structural regime shifts.\n"
            "- Environment Partitioning Requirement: Requires explicit definition of environment indices E; cannot operate on an unpartitioned stationary time series."
        )
    ),
    "jump_diffusion": ConceptSpecification(
        concept_id="jump_diffusion",
        concept_name="Regime-Switching Jump Diffusion Model",
        category="state_space_jump",
        doc_id="arxiv_2303_01004",
        facts_nl=[
            "Fact 1: Regime-Switching Jump Diffusion Model EXPLAINS Currency Rate Jump Dynamics in foreign exchange markets.",
            "Fact 2: The capability to Infer Latent Jump Intensities is PROVIDED_BY Regime-Switching Jump Diffusion Model.",
            "Fact 3: Regime-Switching Jump Diffusion Model is a VARIANT_OF Jump Diffusion Model."
        ],
        facts_kg=[
            "[Regime-Switching Jump Diffusion Model (model)] --(EXPLAINS)--> [Currency Rate Jump Dynamics (phenomenon)]",
            "[Infer Latent Jump Intensities (capability)] --(PROVIDED_BY)--> [Regime-Switching Jump Diffusion Model (model)]",
            "[Regime-Switching Jump Diffusion Model (model)] --(VARIANT_OF)--> [Jump Diffusion Model (model)]"
        ],
        math_io_specification=(
            "- Capability: Infer Latent Jump Intensities\n"
            "- Mathematical Inputs: Currency log-return time series r_t = ln(S_t / S_{t-1}).\n"
            "- Mathematical Outputs: Filtered latent regime posterior probabilities P(s_t = k | F_t) for k in {1, ..., K}, regime-dependent jump intensities lambda_k, and jump size distribution parameters (mu_{J,k}, sigma_{J,k}).\n"
            "- Process Stochastic Formulation: dS_t / S_t = mu_{s_t} * dt + sigma_{s_t} * dW_t + J_{s_t} * dN_t, where s_t is a discrete Markov chain and N_t is a Poisson process."
        ),
        state_control_variables=(
            "- State Variable: Discrete latent regime state s_t in {1, ..., K}, and continuous mid-price S_t.\n"
            "- Control Variable: None (continuous-time stochastic process / probabilistic filtering model).\n"
            "- Crucial Semantic Distinction: Filtered regime probability P(s_t = k | F_t) is a probability distribution over market states, NOT a deterministic future price path or trend signal."
        ),
        assumptions_and_limitations=(
            "- Identifiability Assumption: Maximum likelihood estimation via Hamilton filter / EM algorithm assumes regime persistence and sufficient observations per regime; M5 return series alone do not guarantee unique parameter identification without convergence constraints.\n"
            "- Jump Structure: Jumps are compound Poisson arrivals with random amplitudes, not predictable calendar events."
        )
    )
}


def format_doc_text(doc: Dict[str, Any]) -> str:
    sid = doc.get("source_id", "")
    stype = doc.get("source_type", "")
    title = doc.get("title", doc.get("target_fields", {}).get("title_text", ""))
    tf_name = "readme_text" if stype == "oss" else "abstract_text"
    text = doc.get("target_fields", {}).get(tf_name, "")
    return f"Document [{sid}] ({stype}):\nTitle: {title}\nAbstract/Summary:\n{text}\n"


def build_diagnostic_prompt(
    concept: ConceptSpecification,
    condition_id: str,
    source_doc: Dict[str, Any],
) -> str:
    """Build standardized prompt for H2-D1 diagnostic experiment."""
    doc_text = format_doc_text(source_doc)

    sections = [
        f"TARGET GOAL: Study Manager Goal #774 - Formulate Trading Hypothesis for USDJPY leveraging '{concept.concept_name}'",
        f"PRIMARY LITERATURE SOURCE:\n\n{doc_text}",
    ]

    if condition_id == "D1-C0":
        # Baseline: Identical facts presented as unstructured natural language
        facts_str = "\n".join(concept.facts_nl)
        sections.append(f"EXTRACTED DOMAIN FACTS (Unstructured Natural Language):\n{facts_str}")

    elif condition_id == "D1-C1":
        # Structure effect: Identical facts presented as explicit KG relations
        kg_str = "\n".join(f"- Path: {p}" for p in concept.facts_kg)
        sections.append(f"STRUCTURED KNOWLEDGE GRAPH RELATIONS (Typed Graph Paths):\n{kg_str}")

    elif condition_id == "D1-C2":
        # Role definition effect: KG relations + Math IO, Control/State variables, Assumptions
        kg_str = "\n".join(f"- Path: {p}" for p in concept.facts_kg)
        math_block = (
            f"{kg_str}\n\n"
            f"CAPABILITY INPUT/OUTPUT SPECIFICATION (IO CONTRACT):\n{concept.math_io_specification}\n\n"
            f"STATE AND CONTROL VARIABLES (ROLE DEFINITION):\n{concept.state_control_variables}\n\n"
            f"MATHEMATICAL ASSUMPTIONS & DATA LIMITATIONS:\n{concept.assumptions_and_limitations}"
        )
        sections.append(f"STRUCTURED KNOWLEDGE GRAPH & MATHEMATICAL ROLE DEFINITION:\n{math_block}")

    else:
        raise ValueError(f"Unknown condition_id: {condition_id}")

    sections.append(
        "INSTRUCTIONS:\n"
        f"Formulate a complete, testable, and operable algorithmic trading strategy hypothesis for USDJPY leveraging '{concept.concept_name}'.\n"
        "Your output must be a single valid JSON object strictly conforming to the HypothesisCandidate schema:\n"
        "{\n"
        f'  "hypothesis_id": "hyp_d1_{concept.concept_id}_{condition_id.lower().replace("-", "_")}",\n'
        f'  "condition_id": "{condition_id}",\n'
        f'  "task_id": "d1_{concept.concept_id}",\n'
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
