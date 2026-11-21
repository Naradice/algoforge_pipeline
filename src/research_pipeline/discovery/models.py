"""Domain models and schemas for Step H2 Hypothesis Discovery Engine."""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class PathHop(BaseModel):
    """A single directed relation hop in a knowledge graph path."""
    hop_index: int = Field(..., ge=1)
    src_entity_id: str
    src_name: str
    src_type: str
    relation_type: str
    dst_entity_id: str
    dst_name: str
    dst_type: str
    origin_type: str = Field("text_supported", description="text_supported | rule_inferred | hierarchical_variant")
    evidence_quote: Optional[str] = None


class GraphPath(BaseModel):
    """A multi-hop relational path connecting concepts across papers/OSS."""
    path_id: str
    hops: List[PathHop]
    hop_count: int
    source_concept: str
    target_concept: str
    path_summary: str


class AlgoForgeBacktestSpec(BaseModel):
    """Concrete algorithmic trading specification implementable in AlgoForge."""
    target_asset: str = Field("USDJPY", description="Primary currency pair or ticker")
    timeframe: str = Field("M5", description="Bar timeframe: M1, M5, M15, H1, D1")
    feature_engineering: str = Field(..., description="Mathematical features computed from order book or price series")
    entry_condition: str = Field(..., description="Exact signal condition triggering position entry")
    exit_condition: str = Field(..., description="Stop loss, take profit, or time-stop exit rule")
    risk_management: str = Field(..., description="Position sizing, max drawdown limit, volatility scaling")
    benchmark_baseline: str = Field(..., description="Standard baseline model to beat (e.g. static Bollinger/GARCH)")
    falsification_metric: str = Field(..., description="Quantitative metric threshold that falsifies the hypothesis")


class HypothesisCandidate(BaseModel):
    """A complete, structured, and testable financial market hypothesis."""
    hypothesis_id: str
    condition_id: Optional[str] = Field(
        None, description="Condition ID under which this candidate was generated (blinded during evaluation)"
    )
    task_id: Optional[str] = Field(None, description="Task identifier (task_1, task_2, task_3)")
    title: str = Field(..., description="Concise academic title of the proposed hypothesis")
    target_phenomenon: str = Field(..., description="Market inefficiency or phenomenon being addressed")
    proposed_mechanism: str = Field(..., description="Core mathematical/algorithmic mechanism proposed")
    analogical_source: str = Field(..., description="Origin discipline, model, or tool transferring the insight")
    claim: str = Field(..., description="Core scientific claim stating why this mechanism solves the problem")
    grounding_rationale: str = Field(..., description="Explanation of literature mechanisms or relational facts supporting the claim")
    supporting_path: Optional[GraphPath] = None
    supporting_evidence_quotes: List[str] = Field(default_factory=list)
    testable_falsification_condition: str = Field(..., description="Exact quantitative condition that proves the claim false")
    backtest_spec: Optional[AlgoForgeBacktestSpec] = None


class HypothesisEvaluationResult(BaseModel):
    """Standardized multi-dimensional evaluation rubric for generated hypotheses."""
    hypothesis_id: str
    is_disqualified: bool = Field(False, description="Whether the hypothesis is disqualified by failing gating criteria")
    disqualification_reason: Optional[str] = Field(None, description="Reason for disqualification if disqualified")
    testability_score: float = Field(..., ge=0.0, le=1.0, description="Is the hypothesis empirical and falsifiable?")
    grounding_validity_score: float = Field(..., ge=0.0, le=1.0, description="Is the logical derivation structurally sound and consistent with literature?")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Does it bridge concepts beyond single-paper triviality?")
    operational_concreteness: float = Field(..., ge=0.0, le=1.0, description="Can it be immediately backtested in AlgoForge?")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Secondary weighted overall score (0.0 if disqualified)")
    evaluator_feedback: str = Field(..., description="Detailed strengths, weaknesses, and flaws")
