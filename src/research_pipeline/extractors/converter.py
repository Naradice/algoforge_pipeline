"""Deterministic Normalization Converters for Experimental Conditions C0〜C4.

Converts diverse output formats (Rule-based, Free-text LLM, Schema-only, Full Evidence)
into unified EvaluationCandidate and ExtractedKnowledgeRecord objects.
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import json

from research_pipeline.extractors.models import (
    EvidenceSpan,
    CapabilityIO,
    PhenomenonEntity,
    ModelEntity,
    MethodEntity,
    CapabilityEntity,
    AssumptionItem,
    LimitationItem,
    LocalRelation,
    ExtractedKnowledgeRecord,
)
from research_pipeline.extractors.raw_artifact import (
    EvaluationCandidate,
    RawExtractionArtifact,
    RejectionRecord,
)


def get_entity_items(parsed: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
    """Robust lookup for entity list handling irregular plurals (capabilities, phenomena)."""
    key_map = {
        "phenomenon": ["phenomenas", "phenomena", "phenomenon"],
        "model": ["models", "model"],
        "method": ["methods", "method"],
        "software": ["softwares", "software", "tools", "tool"],
        "capability": ["capabilities", "capabilitys", "capability"],
        "assumption": ["assumptions", "assumption"],
        "limitation": ["limitations", "limitation"],
    }
    for k in key_map.get(entity_type, [f"{entity_type}s", entity_type]):
        val = parsed.get(k)
        if isinstance(val, list):
            return val
    return []


class StandardizationConverter:
    """Converts condition outputs C0〜C4 into canonical evaluation representations."""

    @staticmethod
    def convert_c0_rule_based(
        raw_artifact: RawExtractionArtifact,
    ) -> List[EvaluationCandidate]:
        """Convert C0 (Rule-based baseline) dictionary into EvaluationCandidates.
        
        C0 candidates have no evidence spans (evidence=None).
        """
        candidates: List[EvaluationCandidate] = []
        parsed = raw_artifact.parsed_candidates or {}
        
        for entity_type in ["phenomenon", "model", "method", "software", "capability", "assumption", "limitation"]:
            items = get_entity_items(parsed, entity_type)
            for idx, item in enumerate(items):
                local_id = item.get("local_id", f"{entity_type[:3]}_{idx+1:02d}")
                name_or_stmt = item.get("name") or item.get("statement") or ""
                if name_or_stmt:
                    candidates.append(
                        EvaluationCandidate(
                            candidate_local_id=local_id,
                            entity_type=entity_type,  # type: ignore
                            name_or_statement=name_or_stmt,
                            description=item.get("description"),
                            evidence=None,
                            condition_id="C0",
                            extraction_run_id=raw_artifact.extraction_run_id,
                            conversion_status="success",
                            missing_evidence=False,
                            raw_payload=item,
                        )
                    )
        return candidates

    @staticmethod
    def convert_c1_free_text(
        raw_artifact: RawExtractionArtifact,
    ) -> Tuple[List[EvaluationCandidate], bool, str]:
        """Convert C1 (Free-text LLM) unstructured response into EvaluationCandidates.
        
        Returns: (candidates, is_conversion_success, failure_reason)
        Tracks conversion failure independently as per v0.5 Section 7.2.
        """
        text = raw_artifact.raw_response_text.strip()
        if not text:
            return [], False, "Empty LLM response text"

        candidates: List[EvaluationCandidate] = []
        section_pattern = re.compile(
            r"###?\s*(Phenomenon|Model|Method|Software|Capability|Assumption|Limitation)s?[\r\n]+(.*?)(?=(###?\s*(?:Phenomenon|Model|Method|Software|Capability|Assumption|Limitation)s?|\Z))",
            re.IGNORECASE | re.DOTALL,
        )

        matches = list(section_pattern.finditer(text))
        if not matches:
            # Fallback: check bullet points or numbered lists
            bullet_pattern = re.compile(
                r"[-*]\s*(Phenomenon|Model|Method|Software|Capability|Assumption|Limitation):\s*([^\r\n]+)",
                re.IGNORECASE,
            )
            b_matches = list(bullet_pattern.finditer(text))
            if not b_matches:
                return [], False, "Failed to parse section headers or structured bullets from free text"

            for idx, bm in enumerate(b_matches):
                etype = bm.group(1).lower()
                content = bm.group(2).strip()
                candidates.append(
                    EvaluationCandidate(
                        candidate_local_id=f"{etype[:3]}_{idx+1:02d}",
                        entity_type=etype,  # type: ignore
                        name_or_statement=content,
                        evidence=None,
                        condition_id="C1",
                        extraction_run_id=raw_artifact.extraction_run_id,
                        conversion_status="partial",
                        missing_evidence=False,
                        raw_payload={"bullet_match": content},
                    )
                )
            return candidates, True, ""

        entity_counters: Dict[str, int] = {}
        for match in matches:
            header_type = match.group(1).lower()
            body = match.group(2).strip()

            # Split lines into items
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            for line in lines:
                cleaned = re.sub(r"^[-*\d.)\s]+", "", line).strip()
                if cleaned:
                    entity_counters[header_type] = entity_counters.get(header_type, 0) + 1
                    candidates.append(
                        EvaluationCandidate(
                            candidate_local_id=f"{header_type[:3]}_{entity_counters[header_type]:02d}",
                            entity_type=header_type,  # type: ignore
                            name_or_statement=cleaned,
                            evidence=None,
                            condition_id="C1",
                            extraction_run_id=raw_artifact.extraction_run_id,
                            conversion_status="success",
                            missing_evidence=False,
                            raw_payload={"raw_line": line},
                        )
                    )

        if not candidates:
            return [], False, "Headers found but no valid candidate lines extracted"

        return candidates, True, ""

    @staticmethod
    def convert_c2_structured_no_evidence(
        raw_artifact: RawExtractionArtifact,
    ) -> List[EvaluationCandidate]:
        """Convert C2 (Structured output without evidence) candidates."""
        candidates: List[EvaluationCandidate] = []
        parsed = raw_artifact.parsed_candidates or {}

        for entity_type in ["phenomenon", "model", "method", "software", "capability", "assumption", "limitation"]:
            items = get_entity_items(parsed, entity_type)
            for idx, item in enumerate(items):
                local_id = item.get("local_id", f"{entity_type[:3]}_{idx+1:02d}")
                name_or_stmt = item.get("name") or item.get("statement") or ""
                if name_or_stmt:
                    candidates.append(
                        EvaluationCandidate(
                            candidate_local_id=local_id,
                            entity_type=entity_type,  # type: ignore
                            name_or_statement=name_or_stmt,
                            description=item.get("description"),
                            evidence=None,
                            condition_id="C2",
                            extraction_run_id=raw_artifact.extraction_run_id,
                            conversion_status="success",
                            missing_evidence=False,
                            raw_payload=item,
                        )
                    )
        return candidates

    @staticmethod
    def convert_c3_c4_with_evidence(
        raw_artifact: RawExtractionArtifact,
    ) -> Tuple[List[EvaluationCandidate], List[RejectionRecord]]:
        """Convert C3 (Evidence Grounding) or C4 (Proposed) dictionary into EvaluationCandidates.
        
        Validates evidence presence and creates RejectionRecords if missing.
        """
        candidates: List[EvaluationCandidate] = []
        rejections: List[RejectionRecord] = []
        parsed = raw_artifact.parsed_candidates or {}
        cond_id = raw_artifact.condition_id

        for entity_type in ["phenomenon", "model", "method", "software", "capability", "assumption", "limitation"]:
            items = get_entity_items(parsed, entity_type)
            for idx, item in enumerate(items):
                local_id = item.get("local_id", f"{entity_type[:3]}_{idx+1:02d}")
                name_or_stmt = item.get("name") or item.get("statement") or ""

                ev_dict = item.get("evidence")
                evidence_span: Optional[EvidenceSpan] = None
                missing_evidence = False

                if ev_dict and isinstance(ev_dict, dict):
                    # Deterministically bind target_field_hash if omitted or dummy by LLM (R7)
                    f_name = ev_dict.get("target_field_name") or "abstract_text"
                    if "target_field_name" not in ev_dict:
                        ev_dict["target_field_name"] = f_name
                    h = str(ev_dict.get("target_field_hash") or "")
                    if len(h) != 64 or not all(c in "0123456789abcdefABCDEF" for c in h):
                        if raw_artifact.target_field_hashes and f_name in raw_artifact.target_field_hashes:
                            ev_dict["target_field_hash"] = raw_artifact.target_field_hashes[f_name]
                    start = ev_dict.get("start_char", 0)
                    end = ev_dict.get("end_char", 0)
                    quote = str(ev_dict.get("verbatim_quote") or "")
                    if end <= start:
                        ev_dict["start_char"] = start
                        ev_dict["end_char"] = max(start + len(quote), start + 1)
                    try:
                        evidence_span = EvidenceSpan(**ev_dict)
                    except Exception as e:
                        rejections.append(
                            RejectionRecord(
                                local_id=local_id,
                                entity_type=entity_type,
                                rejection_stage="schema_violation",
                                rejection_reason=f"Malformed evidence span: {str(e)}",
                                raw_entity_data=item,
                            )
                        )
                        missing_evidence = True
                else:
                    missing_evidence = True
                    rejections.append(
                        RejectionRecord(
                            local_id=local_id,
                            entity_type=entity_type,
                            rejection_stage="missing_evidence",
                            rejection_reason="Evidence span omitted in C3/C4 output",
                            raw_entity_data=item,
                        )
                    )

                if name_or_stmt:
                    candidates.append(
                        EvaluationCandidate(
                            candidate_local_id=local_id,
                            entity_type=entity_type,  # type: ignore
                            name_or_statement=name_or_stmt,
                            description=item.get("description"),
                            evidence=evidence_span,
                            condition_id=cond_id,
                            extraction_run_id=raw_artifact.extraction_run_id,
                            conversion_status="success" if not missing_evidence else "partial",
                            missing_evidence=missing_evidence,
                            raw_payload=item,
                        )
                    )

        return candidates, rejections
