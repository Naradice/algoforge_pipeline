"""Domain models and schemas for HDE Step H2-D3 Data Feasibility & Semantic Fidelity Gate."""

from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field


DataFeasibilityStatus = Literal["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]
SemanticFidelityStatus = Literal["EXACT", "PROXY", "INVALID", "UNKNOWN"]
EstimationSpecStatus = Literal["COMPLETE", "INCOMPLETE", "NOT_REQUIRED"]
IdentifiabilityStatus = Literal["IDENTIFIED", "PROXY_IDENTIFIED", "UNKNOWN", "NOT_APPLICABLE"]
TemporalValidityStatus = Literal["VALID", "LOOKAHEAD_LEAK", "LOOKBACK_UNDEFINED"]
GateDecision = Literal["PASS", "BLOCK", "REVIEW"]


class ProxySpecification(BaseModel):
    """5-point explicit specification required for PROXY -> PASS qualification."""
    original_target: str = Field(..., description="Theoretical target quantity in original literature")
    proxy_target: str = Field(..., description="Actual surrogate quantity computed from M5 data")
    transformed_hypothesis: str = Field(..., description="Consistent re-scoped hypothesis under proxy")
    permitted_interpretations: str = Field(..., description="Scientifically valid conclusions permitted")
    prohibited_interpretations: str = Field(..., description="Forbidden conclusions or claims of measuring original target")


class GateEvaluationResult(BaseModel):
    """Complete 7-dimensional evaluation result from Data Feasibility & Semantic Fidelity Gate."""
    hypothesis_id: str
    case_id: Optional[str] = None
    # 7 Core Evaluative Dimensions
    data_feasibility: DataFeasibilityStatus
    semantic_fidelity: SemanticFidelityStatus
    estimation_spec: EstimationSpecStatus
    identifiability: IdentifiabilityStatus
    temporal_validity: TemporalValidityStatus
    gate_decision: GateDecision
    reason: str
    # Detailed Evidence & Audit Trail
    proxy_spec: Optional[ProxySpecification] = None
    semantic_shift_documented: bool = False
    evidence_details: Dict[str, Any] = Field(default_factory=dict)
