"""Domain models and schemas for HDE Step H2-D1 Diagnostic Experiment."""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from research_pipeline.discovery.models import HypothesisCandidate


class ConceptSpecification(BaseModel):
    """Specification of a quantitative finance concept archetypes tested in H2-D1."""
    concept_id: str
    concept_name: str
    category: Literal["execution_control", "point_process", "robust_prediction", "state_space_jump"]
    doc_id: str
    # Exact facts for C0 (natural language) and C1 (KG relation) - guaranteed identical
    facts_nl: List[str]
    facts_kg: List[str]
    # Detailed mathematical specifications for C2
    math_io_specification: str
    state_control_variables: str
    assumptions_and_limitations: str


class DiagnosticEvaluationItem(BaseModel):
    """Detailed objective evaluation of a hypothesis in H2-D1 against the Independent Gold Spec."""
    concept_id: str
    condition_id: Optional[str] = Field(None, description="Condition ID if applicable (blinded during grading)")
    hypothesis_id: str
    # 5 Objective Error Flags (True = Error Present, False = Error Free)
    e_role: bool = Field(False, description="Role Misconception: Misusing model purpose (e.g. execution as alpha entry)")
    e_math: bool = Field(False, description="Math/State Variable Misconception: Conflating state/control variables (e.g. x(t) vs S(t))")
    e_data: bool = Field(False, description="Data Availability Inconsistency: Requiring unavailable resolution without proxy")
    e_leap: bool = Field(False, description="Inferential Leap: Stating unproven profitability as established fact")
    e_inop: bool = Field(False, description="Inoperability: Entry/exit conditions lack programmable mathematical definition")
    # Tri-State Data Feasibility Model
    feasibility_status: Literal["SUPPORTED", "UNSUPPORTED", "UNKNOWN"] = Field("SUPPORTED", description="Tri-state data contract feasibility")
    # Verbatim Evidence & Reasoning for each flag
    error_evidence: Dict[str, Optional[str]] = Field(default_factory=dict)
    # Overall operability (all 5 errors false)
    is_operable: bool = Field(False, description="True if sum of all 5 error flags is 0")
    evaluator_notes: str = Field("", description="Detailed explanation of findings")
