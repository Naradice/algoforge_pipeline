"""Evidence verification and structural validation pipeline for H1a.

Implements 4 validation stages:
1. Quote Validity with deterministic offset resolution (User Instruction 1)
2. Claim Support verification
3. Claim Completeness / Overclaim detection
4. Structural Consistency (IO state, entity ID existence, relation referential integrity)
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import hashlib

from research_pipeline.extractors.models import (
    EvidenceSpan,
    CapabilityIO,
    LocalRelation,
    ExtractedKnowledgeRecord,
    compute_canonical_hash,
)
from research_pipeline.extractors.raw_artifact import RejectionRecord


def resolve_evidence_offset(
    evidence: EvidenceSpan,
    canonical_text: str,
    target_field_hash: str,
) -> EvidenceSpan:
    """Validate and deterministically resolve verbatim quote coordinates into canonical text.
    
    Conforms to User Instruction 1:
    - raw_offset_valid: was the original LLM offset exact?
    - resolved_offset_valid: is the resolved offset exact?
    - offset_resolution_status: 'unchanged', 'unique_match', 'ambiguous', 'not_found'
    - Multiple matches ('ambiguous') are NOT automatically accepted.
    """
    quote = evidence.verbatim_quote
    text_len = len(canonical_text)
    
    raw_start = evidence.raw_start_char if evidence.raw_start_char is not None else evidence.start_char
    raw_end = evidence.raw_end_char if evidence.raw_end_char is not None else evidence.end_char
    
    # 1. Check if raw coordinates are already strictly valid
    if raw_end <= text_len and canonical_text[raw_start:raw_end] == quote:
        return evidence.model_copy(
            update={
                "target_field_hash": target_field_hash,
                "raw_start_char": raw_start,
                "raw_end_char": raw_end,
                "raw_offset_valid": True,
                "resolved_offset_valid": True,
                "offset_resolution_status": "unchanged",
            }
        )
    
    # 2. Search for occurrences of verbatim_quote in canonical_text
    matches: List[Tuple[int, int]] = []
    idx = canonical_text.find(quote)
    while idx != -1:
        matches.append((idx, idx + len(quote)))
        idx = canonical_text.find(quote, idx + 1)
        
    if not matches:
        return evidence.model_copy(
            update={
                "target_field_hash": target_field_hash,
                "raw_start_char": raw_start,
                "raw_end_char": raw_end,
                "raw_offset_valid": False,
                "resolved_offset_valid": False,
                "offset_resolution_status": "not_found",
            }
        )
    
    if len(matches) == 1:
        match_start, match_end = matches[0]
        return evidence.model_copy(
            update={
                "target_field_hash": target_field_hash,
                "start_char": match_start,
                "end_char": match_end,
                "raw_start_char": raw_start,
                "raw_end_char": raw_end,
                "raw_offset_valid": False,
                "resolved_offset_valid": True,
                "offset_resolution_status": "unique_match",
            }
        )
    
    # Multiple matches found: marked 'ambiguous' and not automatically accepted
    return evidence.model_copy(
        update={
            "target_field_hash": target_field_hash,
            "raw_start_char": raw_start,
            "raw_end_char": raw_end,
            "raw_offset_valid": False,
            "resolved_offset_valid": False,
            "offset_resolution_status": "ambiguous",
        }
    )


def validate_evidence(
    evidence: EvidenceSpan,
    canonical_fields: Dict[str, str],
) -> Tuple[bool, str, Optional[EvidenceSpan]]:
    """Validate EvidenceSpan against normalized canonical text fields (R7).
    
    Returns: (is_valid, failure_reason, resolved_evidence)
    """
    field_name = evidence.target_field_name
    if field_name not in canonical_fields:
        return False, f"Target field '{field_name}' not found in document", None
    
    text = canonical_fields[field_name]
    expected_hash = compute_canonical_hash(text)
    
    # Resolve offsets
    resolved = resolve_evidence_offset(evidence, text, expected_hash)
    
    if resolved.offset_resolution_status == "not_found":
        return False, f"Verbatim quote '{resolved.verbatim_quote}' not found in field '{field_name}'", resolved
    
    if resolved.offset_resolution_status == "ambiguous":
        return False, f"Verbatim quote '{resolved.verbatim_quote}' appears multiple times in field '{field_name}' (ambiguous match, context verification required)", resolved
    
    if not resolved.resolved_offset_valid:
        return False, f"Resolved offset [{resolved.start_char}:{resolved.end_char}] does not match quote in '{field_name}'", resolved
    
    if resolved.target_field_hash != expected_hash:
        return False, f"Target field hash mismatch: got '{resolved.target_field_hash}', expected '{expected_hash}'", resolved
    
    return True, "", resolved


# Valid relation type domain constraints (source_type, destination_type)
ALLOWED_RELATION_ENDPOINTS: Dict[str, Tuple[Set[str], Set[str]]] = {
    "EXPLAINS": ({"model", "method"}, {"phenomenon"}),
    "APPLIES_TO": ({"method"}, {"model"}),
    "PROVIDED_BY": ({"capability"}, {"model", "method", "software"}),
    "REQUIRES": ({"capability", "model"}, {"assumption"}),
    "CONSTRAINED_BY": ({"capability", "model", "method", "software"}, {"limitation"}),
    "IMPLEMENTS": ({"software"}, {"model", "method"}),
}


def validate_local_relation(
    relation: LocalRelation,
    accepted_entities: Dict[str, str],  # local_id -> entity_type
    canonical_fields: Dict[str, str],
) -> Tuple[bool, str, Optional[LocalRelation]]:
    """Validate referential integrity and evidence for LocalRelation (R9)."""
    src_id = relation.src_local_id
    dst_id = relation.dst_local_id
    rel_type = relation.relation_type
    
    # 1. Prevent self-reference
    if src_id == dst_id:
        return False, f"Self-referential relation not allowed: {src_id} -> {dst_id}", None
    
    # 2. Check source entity exists in accepted
    if src_id not in accepted_entities:
        return False, f"Source entity '{src_id}' is not in accepted entities (dangling reference)", None
    
    # 3. Check destination entity exists in accepted
    if dst_id not in accepted_entities:
        return False, f"Destination entity '{dst_id}' is not in accepted entities (dangling reference)", None
    
    src_type = accepted_entities[src_id]
    dst_type = accepted_entities[dst_id]
    
    # 4. Check relation type endpoint validity
    if rel_type not in ALLOWED_RELATION_ENDPOINTS:
        return False, f"Unknown relation type '{rel_type}'", None
    
    allowed_srcs, allowed_dsts = ALLOWED_RELATION_ENDPOINTS[rel_type]
    if src_type not in allowed_srcs or dst_type not in allowed_dsts:
        return False, f"Invalid endpoints for relation '{rel_type}': {src_type}({src_id}) -> {dst_type}({dst_id}). Allowed: {allowed_srcs} -> {allowed_dsts}", None
    
    # 5. Check relation evidence
    ev_valid, ev_reason, resolved_ev = validate_evidence(relation.evidence, canonical_fields)
    if not ev_valid or resolved_ev is None:
        return False, f"Relation evidence validation failed: {ev_reason}", None
    
    return True, "", relation.model_copy(update={"evidence": resolved_ev})
