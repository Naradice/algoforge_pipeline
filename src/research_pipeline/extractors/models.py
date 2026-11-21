"""Pydantic schemas and domain models for HDE H1a Capability Extraction.

Complies strictly with H1a scientific experiment plan v0.5.
"""

from typing import List, Optional, Literal, Dict, Any
import unicodedata
import hashlib
from pydantic import BaseModel, Field, field_validator, model_validator

from research_pipeline.extractors.vocab import L1_VALID_TYPES


def canonicalize_text(text: str) -> str:
    """Deterministic text canonicalization for citation and offset indexing.
    
    1. LF line endings unification
    2. Unicode NFC normalization
    3. Strip outer whitespace
    4. Preserve internal spacing, newlines, and indentation
    """
    if not text:
        return ""
    # 1. Unify line endings to LF
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", unified)
    # 3. Strip outer whitespace
    return normalized.strip()


def compute_canonical_hash(canonical_text: str) -> str:
    """Compute deterministic SHA-256 hash of canonical UTF-8 text."""
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


class EvidenceSpan(BaseModel):
    """Verbatim citation span and positional coordinates into canonicalized text.
    
    Tracks both raw model-generated offsets and resolved offsets for scientific auditing.
    """
    target_field_name: Literal["abstract_text", "readme_text", "title_text"] = Field(
        ..., description="Name of the canonicalized target text field"
    )
    target_field_hash: Optional[str] = Field(
        None, pattern=r"^[0-9a-f]{64}$", description="SHA-256 hash of target field canonical text"
    )
    verbatim_quote: str = Field(
        ..., min_length=1, description="Verbatim quote extracted from the target field"
    )
    start_char: int = Field(..., ge=0, description="Unicode codepoint start index (resolved/effective)")
    end_char: int = Field(..., ge=1, description="Unicode codepoint end index (resolved/effective)")
    
    # Audit trail for raw vs resolved offset resolution (User Instruction 1)
    raw_start_char: Optional[int] = Field(None, description="Original start_char emitted by LLM")
    raw_end_char: Optional[int] = Field(None, description="Original end_char emitted by LLM")
    raw_offset_valid: Optional[bool] = Field(None, description="Whether the raw LLM offset was strictly valid")
    resolved_offset_valid: Optional[bool] = Field(None, description="Whether the resolved offset is strictly valid")
    offset_resolution_status: Literal["unchanged", "unique_match", "ambiguous", "not_found"] = Field(
        "unchanged", description="Classification of deterministic offset resolution"
    )

    @model_validator(mode="after")
    def validate_indices(self) -> "EvidenceSpan":
        if self.end_char <= self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be strictly greater than start_char ({self.start_char})"
            )
        return self


class PhenomenonEntity(BaseModel):
    """Subject phenomenon entity described or explained in the research."""
    local_id: str = Field(..., description="Document-local unique identifier, e.g. 'phe_01'")
    name: str = Field(..., min_length=1, description="Canonical name of the phenomenon")
    description: str = Field(..., description="Detailed description of the phenomenon")
    evidence: EvidenceSpan


class ModelEntity(BaseModel):
    """Mathematical, computational, or simulation framework entity."""
    local_id: str = Field(..., description="Document-local unique identifier, e.g. 'mod_01'")
    name: str = Field(..., min_length=1, description="Model framework name")
    category: Literal["agent_based", "statistical_physics", "state_space", "deep_learning", "stochastic_process", "other"]
    description: str = Field(..., description="Mathematical/structural description")
    evidence: EvidenceSpan


class MethodEntity(BaseModel):
    """Algorithm, estimation, or optimization procedure entity."""
    local_id: str = Field(..., description="Document-local unique identifier, e.g. 'met_01'")
    name: str = Field(..., min_length=1, description="Method/algorithm name")
    category: Literal[
        "parameter_inference",
        "anomaly_detection",
        "state_estimation",
        "representation_learning",
        "optimization",
        "hypothesis_testing",
        "other"
    ]
    description: str = Field(..., description="Computational or inferential procedure description")
    evidence: EvidenceSpan


class SoftwareEntity(BaseModel):
    """Reusable codebase, execution framework, simulator, or software tool entity."""
    local_id: str = Field(..., description="Document-local unique identifier, e.g. 'soft_01'")
    name: str = Field(..., min_length=1, description="Software tool or framework name")
    category: Literal[
        "backtesting_framework",
        "matching_engine",
        "simulation_environment",
        "data_collector",
        "analytics_library",
        "execution_platform",
        "other"
    ] = Field("other", description="Category of software artifact")
    description: str = Field(..., description="Description of the codebase or framework")
    evidence: EvidenceSpan


class CapabilityIO(BaseModel):
    """Input or output specification for a capability with strict state consistency (R8)."""
    data_type_l1: Literal[
        "timeseries_1d",
        "timeseries_multivariate",
        "tabular_record",
        "graph_network",
        "discrete_sequence",
        "probability_distribution",
        "parameter_vector",
        "scalar_value",
        "event_stream",
        "unknown"
    ] = Field(..., description="L1 Universal Type")
    data_type_l2: Optional[str] = Field(None, description="L2 Domain Specialization")
    description: Optional[str] = Field(None, description="Semantic description of the data artifact")
    time_resolution: str = Field("unspecified", description="e.g. tick, minute, daily, static, unspecified")
    dimensionality: Optional[str] = Field(None, description="e.g. 1D, 5D, unknown")
    is_unspecified: bool = Field(False, description="True if the input/output specification is omitted in source text")
    evidence: Optional[EvidenceSpan] = Field(None, description="Evidence span (forbidden if is_unspecified is True)")

    @field_validator("data_type_l1", mode="before")
    @classmethod
    def normalize_l1(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            alias_map = {
                "timeseries": "timeseries_1d",
                "time_series": "timeseries_1d",
                "series": "timeseries_1d",
                "dataset": "tabular_record",
                "table": "tabular_record",
                "boolean": "scalar_value",
                "scalar": "scalar_value",
                "vector": "parameter_vector",
                "distribution": "probability_distribution",
                "graph": "graph_network",
                "network": "graph_network",
                "sequence": "discrete_sequence",
                "stream": "event_stream",
            }
            if v_clean in alias_map:
                return alias_map[v_clean]
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "CapabilityIO":
        if self.data_type_l1 not in L1_VALID_TYPES:
            raise ValueError(f"Invalid L1 data type: '{self.data_type_l1}'. Must be one of {sorted(L1_VALID_TYPES)}")

        if self.is_unspecified:
            if self.data_type_l1 != "unknown":
                raise ValueError("When is_unspecified is True, data_type_l1 must be 'unknown'")
            # Normalize trivial placeholder values emitted by LLMs to None
            if self.data_type_l2 in ("unknown", "unspecified", "", "none", "None"):
                self.data_type_l2 = None
            elif self.data_type_l2 is not None:
                raise ValueError("When is_unspecified is True, data_type_l2 must be None")
            if self.description in ("unknown", "unspecified", "", "none", "None"):
                self.description = None
            elif self.description is not None:
                raise ValueError("When is_unspecified is True, description must be None")
            if self.dimensionality in ("unknown", "unspecified", "", "none", "None"):
                self.dimensionality = None
            elif self.dimensionality is not None:
                raise ValueError("When is_unspecified is True, dimensionality must be None")
            if self.time_resolution in ("unspecified", "unknown", ""):
                self.time_resolution = "unspecified"
            elif self.time_resolution != "unspecified":
                raise ValueError("When is_unspecified is True, time_resolution must be 'unspecified'")
            if self.evidence is not None:
                raise ValueError("When is_unspecified is True, evidence must not be fabricated (must be None)")
        else:
            if self.data_type_l1 == "unknown":
                raise ValueError("When is_unspecified is False, data_type_l1 cannot be 'unknown'")
            if self.evidence is None:
                raise ValueError("When is_unspecified is False, explicit IO requires evidence")
        return self


class AssumptionItem(BaseModel):
    """Theoretical assumption or environmental condition entity."""
    local_id: str = Field(..., description="Document-local unique identifier, e.g. 'asm_01'")
    statement: str = Field(..., min_length=1, description="Statement of assumption")
    is_explicit: bool = Field(..., description="Whether explicitly stated in text vs implied")
    evidence: EvidenceSpan


class LimitationItem(BaseModel):
    """Applicability constraint, scaling bottleneck, or failure condition entity."""
    local_id: str = Field(..., description="Document-local unique identifier, e.g. 'lim_01'")
    statement: str = Field(..., min_length=1, description="Statement of limitation")
    evidence: EvidenceSpan


class CapabilityEntity(BaseModel):
    """Actionable operational capability provided by a model or method."""
    local_id: str = Field(..., description="Document-local unique identifier, e.g. 'cap_01'")
    name: str = Field(..., min_length=1, description="Descriptive capability name")
    operation: Literal["generate", "infer", "detect", "predict", "transform", "optimize"] = Field(
        ..., description="Standardized operational category"
    )
    inputs: List[CapabilityIO] = Field(..., min_length=1, description="Input data requirements")
    outputs: List[CapabilityIO] = Field(..., min_length=1, description="Output data artifacts")
    preconditions: List[str] = Field(default_factory=list, description="Preconditions required for execution")
    limitations: List[str] = Field(default_factory=list, description="Specific known limitations")
    evidence: EvidenceSpan


class LocalRelation(BaseModel):
    """Directed relation between entities within a single document (R9)."""
    relation_id: str = Field(..., description="Relation unique ID within trial, e.g. 'rel_01'")
    src_local_id: str = Field(..., description="Source entity local_id")
    dst_local_id: str = Field(..., description="Destination entity local_id")
    relation_type: Literal[
        "EXPLAINS",       # Model/Method -> Phenomenon
        "APPLIES_TO",     # Method -> Model
        "PROVIDED_BY",    # Capability -> Model/Method/Software
        "REQUIRES",       # Capability/Model -> Assumption
        "CONSTRAINED_BY", # Capability/Model/Method/Software -> Limitation
        "IMPLEMENTS"      # Software -> Model/Method
    ] = Field(..., description="Semantic relation type")
    evidence: EvidenceSpan = Field(..., description="Verbatim evidence supporting this relation")


class ExtractedKnowledgeRecord(BaseModel):
    """Complete structured record of accepted knowledge extracted from a document version."""
    source_id: str = Field(..., description="H0 logical source ID")
    version_id: str = Field(..., description="H0 source version ID")
    version_hash: str = Field(..., description="Content SHA-256 hash")
    extraction_run_id: str = Field(..., description="Unique trial run identifier")
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"]
    evidence_status: Literal["reported", "unverified"] = Field("reported")

    phenomena: List[PhenomenonEntity] = Field(default_factory=list)
    models: List[ModelEntity] = Field(default_factory=list)
    methods: List[MethodEntity] = Field(default_factory=list)
    softwares: List[SoftwareEntity] = Field(default_factory=list)
    capabilities: List[CapabilityEntity] = Field(default_factory=list)
    assumptions: List[AssumptionItem] = Field(default_factory=list)
    limitations: List[LimitationItem] = Field(default_factory=list)
    local_relations: List[LocalRelation] = Field(default_factory=list)

    def generate_candidate_entity_id(self, entity_type: str, local_id: str) -> str:
        """Construct candidate entity ID conforming to R10."""
        return f"{self.source_id}:{self.version_hash}:{self.extraction_run_id}:{entity_type}:{local_id}"

    def generate_candidate_edge_id(self, src_candidate_id: str, relation_type: str, dst_candidate_id: str) -> str:
        """Construct candidate edge ID conforming to R10."""
        return f"{src_candidate_id}->{relation_type}->{dst_candidate_id}"


class CandidateEntityRef(BaseModel):
    """Exported candidate entity mention for H1b entity resolution."""
    candidate_id: str
    source_id: str
    version_id: str


class C2GenericEntity(BaseModel):
    """Entity representation for C2 condition (no evidence)."""
    local_id: str = Field(..., description="Document-local identifier, e.g. 'mod_01'")
    name: Optional[str] = Field(None, description="Entity name")
    statement: Optional[str] = Field(None, description="Entity statement if assumption/limitation")
    description: Optional[str] = Field(None, description="Entity description")


class C2Record(BaseModel):
    """C2 Condition: Structured JSON without evidence spans or offsets."""
    phenomena: List[C2GenericEntity] = Field(default_factory=list)
    models: List[C2GenericEntity] = Field(default_factory=list)
    methods: List[C2GenericEntity] = Field(default_factory=list)
    softwares: List[C2GenericEntity] = Field(default_factory=list)
    capabilities: List[C2GenericEntity] = Field(default_factory=list)
    assumptions: List[C2GenericEntity] = Field(default_factory=list)
    limitations: List[C2GenericEntity] = Field(default_factory=list)


class C3Record(BaseModel):
    """C3 Condition: Structured JSON with evidence grounding, without local relations."""
    phenomena: List[PhenomenonEntity] = Field(default_factory=list)
    models: List[ModelEntity] = Field(default_factory=list)
    methods: List[MethodEntity] = Field(default_factory=list)
    softwares: List[SoftwareEntity] = Field(default_factory=list)
    capabilities: List[CapabilityEntity] = Field(default_factory=list)
    assumptions: List[AssumptionItem] = Field(default_factory=list)
    limitations: List[LimitationItem] = Field(default_factory=list)


# --- Two-Stage Extraction Schemas (H1a-R2 / Condition C5) ---

class Stage1MentionItem(BaseModel):
    local_id: str = Field(..., description="Local ID, e.g. 'ent_01'")
    entity_type: Literal["model", "method", "software", "capability", "phenomenon", "limitation", "assumption"]
    name_or_statement: str = Field(..., description="Name or descriptive statement")
    category: Optional[str] = Field(None, description="Optional sub-category")
    verbatim_quote: str = Field(..., description="Exact verbatim quote from text")


class Stage1MentionRecord(BaseModel):
    """Stage 1: High-recall mention detection and classification without IO typing."""
    mentions: List[Stage1MentionItem] = Field(default_factory=list)


class Stage2CapabilityIODetail(BaseModel):
    """Stage 2: Focused Capability IO typing and operational specification."""
    capability_name: str
    operation: Literal["generate", "infer", "detect", "predict", "transform", "optimize"] = "infer"
    inputs: List[CapabilityIO] = Field(default_factory=list)
    outputs: List[CapabilityIO] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


# --- H1b Knowledge Graph Multi-Evidence Schema ---

class MentionProvenance(BaseModel):
    """Cryptographic provenance of an individual entity mention back to H0 storage."""
    mention_id: str = Field(..., description="Unique mention ID e.g. 'doc_id:hash:entity_type:local_id'")
    source_id: str = Field(..., description="Logical source ID e.g. arXiv ID or repo name")
    version_hash: str = Field(..., description="H0 content SHA-256 hash")
    verbatim_quote: str = Field(..., description="Exact verbatim quote in text")
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=1)
    target_field_name: str = Field("abstract_text")


class CanonicalEntity(BaseModel):
    """Canonical resolved entity in H1b Knowledge Graph."""
    entity_id: str = Field(..., description="UUID or canonical slug")
    canonical_name: str = Field(..., min_length=1)
    entity_type: Literal["phenomenon", "model", "method", "software", "capability", "assumption", "limitation"]
    origin_type: Literal["extracted", "inferred"] = Field("extracted", description="extracted: directly from H0 text; inferred: derived via ER/reasoning")
    derivation_rationale: Optional[str] = Field(None, description="Explanation for inferred entities")
    aliases: List[str] = Field(default_factory=list)
    provenances: List[MentionProvenance] = Field(default_factory=list)


class RelationEvidence(BaseModel):
    """Document-specific supporting evidence for a knowledge graph relation edge."""
    evidence_id: str = Field(..., description="Evidence unique identifier")
    edge_id: str = Field(..., description="Target relation edge ID: 'src_id->type->dst_id'")
    source_id: str = Field(..., description="Document source ID")
    version_hash: str = Field(..., description="H0 content SHA-256 hash")
    verbatim_quote: str = Field(..., description="Sentence or clause evidence quote")
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=1)
    reasoning: Optional[str] = Field(None, description="Brief justification of the relation")


class RelationEdge(BaseModel):
    """Typed relation edge in H1b Knowledge Graph (normalized, separates edge from evidence)."""
    edge_id: str = Field(..., description="'src_id->relation_type->dst_id'")
    src_entity_id: str = Field(..., description="Canonical source entity ID")
    dst_entity_id: str = Field(..., description="Canonical destination entity ID")
    relation_type: Literal[
        "EXPLAINS",
        "APPLIES_TO",
        "PROVIDED_BY",
        "REQUIRES",
        "CONSTRAINED_BY",
        "IMPLEMENTS",
        "VARIANT_OF"
    ]
    confidence_score: float = Field(1.0, ge=0.0, le=1.0)
    origin_type: Literal["text_supported", "rule_inferred", "hierarchical_variant"] = Field(
        "text_supported", description="text_supported: verified by textual evidence; rule_inferred: inferred by type co-occurrence; hierarchical_variant: ER taxonomy"
    )
    evidences: List[RelationEvidence] = Field(default_factory=list)


class CandidateEdgeRef(BaseModel):
    """Exported candidate local relation edge for H1b knowledge graph."""
    candidate_edge_id: str
    src_candidate_id: str
    dst_candidate_id: str
    relation_type: str
    evidence_quote: str


class H1bDocumentPackage(BaseModel):
    """Document package delivered from H1a to H1b."""
    source_id: str
    version_id: str
    version_hash: str
    extraction_run_id: str
    source_type: Literal["paper", "oss"]
    primary_name: str
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"]
    evidence_status: Literal["reported", "unverified"]

    entities: List[CandidateEntityRef]
    local_relations: List[CandidateEdgeRef]
    accepted_record: ExtractedKnowledgeRecord
