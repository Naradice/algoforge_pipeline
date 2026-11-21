"""Raw artifacts, evaluation candidate representations, and rejection bundles for H1a.

Preserves all unparsed strings, parse errors, and validator rejections.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

from research_pipeline.extractors.models import EvidenceSpan, ExtractedKnowledgeRecord


class RawExtractionArtifact(BaseModel):
    """Raw LLM/Rule-based response saved before syntax parsing and schema validation."""
    extraction_run_id: str = Field(..., description="Unique extraction run/trial ID")
    source_id: str = Field(..., description="Logical source ID")
    version_id: str = Field(..., description="Source version ID")
    condition_id: Literal["C0", "C1", "C2", "C3", "C4", "C5"] = Field(..., description="Experimental condition")
    raw_response_text: str = Field(..., description="Raw text returned by model/extractor")
    http_status: int = Field(200, description="HTTP status code")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="prompt_tokens, completion_tokens, total_tokens")
    executed_at: str = Field(..., description="ISO 8601 execution timestamp")
    model_checkpoint: str = Field(..., description="Model identifier / checkpoint name")
    prompt_hash: str = Field(..., description="SHA-256 of prompt template + inputs")
    target_field_hashes: Dict[str, str] = Field(default_factory=dict, description="Map of target field name to SHA-256")

    # Parsing audit trail
    json_parse_status: Literal["success", "invalid_json", "empty_response", "not_applicable"] = Field(...)
    parse_error: Optional[str] = Field(None, description="Detailed syntax or decoding error if any")
    parsed_candidates: Optional[Dict[str, Any]] = Field(None, description="Decoded dictionary before model validation")


class EvaluationCandidate(BaseModel):
    """Intermediate candidate representation for fair evaluation across all conditions (C0〜C4).
    
    C0〜C2 allow evidence=None. C3〜C4 flag missing_evidence if evidence was omitted.
    """
    candidate_local_id: str = Field(..., description="Local ID, e.g. 'cap_01'")
    entity_type: Literal[
        "phenomenon",
        "model",
        "method",
        "software",
        "capability",
        "assumption",
        "limitation",
    ] = Field(..., description="Target entity category")
    name_or_statement: str = Field(..., description="Entity name or assumption/limitation statement")
    description: Optional[str] = Field(None, description="Detailed description")
    evidence: Optional[EvidenceSpan] = Field(None, description="Evidence span (None permitted for C0-C2)")
    condition_id: Literal["C0", "C1", "C2", "C3", "C4", "C5"] = Field(..., description="Condition ID")
    extraction_run_id: str = Field(..., description="Trial run ID")
    conversion_status: Literal["success", "partial", "failed"] = Field(
        "success", description="Status of normalization conversion"
    )
    missing_evidence: bool = Field(False, description="True if condition required evidence but it was missing")
    raw_payload: Dict[str, Any] = Field(default_factory=dict, description="Raw dictionary payload from extractor")


class RejectionRecord(BaseModel):
    """Record of an entity or relation rejected by validator with explicit causal category."""
    local_id: str = Field(..., description="Local ID of rejected item")
    entity_type: str = Field(..., description="Entity type or 'relation'")
    rejection_stage: Literal[
        "syntax_parse",
        "quote_validity",
        "claim_support",
        "claim_completeness",
        "io_consistency",
        "schema_violation",
        "dangling_reference",
        "invalid_relation_type",
        "missing_evidence",
    ] = Field(..., description="Validation stage where rejection occurred")
    rejection_reason: str = Field(..., description="Specific descriptive reason for rejection")
    raw_entity_data: Dict[str, Any] = Field(default_factory=dict, description="Original candidate payload")


class ExtractionTrioBundle(BaseModel):
    """Three-tier bundle preserving Raw, Accepted, and Rejected items per document version trial."""
    extraction_run_id: str
    source_id: str
    version_id: str
    raw_artifact: RawExtractionArtifact
    evaluation_candidates: List[EvaluationCandidate] = Field(default_factory=list)
    accepted_record: Optional[ExtractedKnowledgeRecord] = None
    rejected_items: List[RejectionRecord] = Field(default_factory=list)
