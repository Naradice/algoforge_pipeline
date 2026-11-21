"""Semantic Entity Matching via Maximum Bipartite Matching (Hungarian algorithm).

Implements Section 8.1:
- Primary metric: Kuhn-Munkres 1-to-1 maximum weight bipartite matching based on semantic equivalence.
- Auxiliary metrics: Strict String Match and Embedding Cosine Similarity.
"""

from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from pydantic import BaseModel, Field

from research_pipeline.extractors.raw_artifact import EvaluationCandidate


class MatchPair(BaseModel):
    gold_id: str
    predicted_id: str
    entity_type: str
    similarity_score: float
    is_semantic_match: bool


class BipartiteMatchingResult(BaseModel):
    matched_pairs: List[MatchPair]
    unmatched_gold_ids: List[str]
    unmatched_predicted_ids: List[str]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def normalize_entity_name(name: str) -> str:
    """Normalize string for lexical matching: lowercase, strip, collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", name.strip().lower())


def are_semantically_equivalent(
    gold_name: str,
    gold_synonyms: List[str],
    pred_name: str,
) -> bool:
    """Determine semantic equivalence based on expert gold synonym registry and substring rules."""
    norm_gold = normalize_entity_name(gold_name)
    norm_pred = normalize_entity_name(pred_name)
    
    if norm_gold == norm_pred:
        return True
    
    # Check synonym cluster
    for syn in gold_synonyms:
        if normalize_entity_name(syn) == norm_pred:
            return True
        
    # Acronym containment (e.g. "Neural Posterior Estimation (NPE)" vs "NPE")
    if f"({norm_pred})" in norm_gold or f"({norm_gold})" in norm_pred:
        return True
    if len(norm_pred) >= 4 and norm_pred in norm_gold:
        return True
    if len(norm_gold) >= 4 and norm_gold in norm_pred:
        return True
        
    return False


def match_entities_bipartite(
    gold_entities: List[Dict[str, Any]],
    predicted_candidates: List[EvaluationCandidate],
    target_entity_type: Optional[str] = None,
) -> BipartiteMatchingResult:
    """Execute 1-to-1 maximum weight bipartite matching between gold and predicted candidates.
    
    If target_entity_type is specified, filters both sets to that type.
    """
    golds = [g for g in gold_entities if target_entity_type is None or g["entity_type"] == target_entity_type]
    preds = [p for p in predicted_candidates if target_entity_type is None or p.entity_type == target_entity_type]

    n_gold = len(golds)
    n_pred = len(preds)

    if n_gold == 0 and n_pred == 0:
        return BipartiteMatchingResult(
            matched_pairs=[],
            unmatched_gold_ids=[],
            unmatched_predicted_ids=[],
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
        )

    if n_gold == 0:
        return BipartiteMatchingResult(
            matched_pairs=[],
            unmatched_gold_ids=[],
            unmatched_predicted_ids=[p.candidate_local_id for p in preds],
            true_positives=0,
            false_positives=n_pred,
            false_negatives=0,
            precision=0.0,
            recall=1.0,
            f1=0.0,
        )

    if n_pred == 0:
        return BipartiteMatchingResult(
            matched_pairs=[],
            unmatched_gold_ids=[g["local_id"] for g in golds],
            unmatched_predicted_ids=[],
            true_positives=0,
            false_positives=0,
            false_negatives=n_gold,
            precision=1.0,
            recall=0.0,
            f1=0.0,
        )

    # Cost matrix for Hungarian algorithm (we maximize similarity, so cost = 1.0 - sim)
    cost_matrix = np.ones((n_gold, n_pred), dtype=float)

    for i, g in enumerate(golds):
        g_name = g.get("name") or g.get("statement") or ""
        g_synonyms = g.get("synonyms", [])
        g_type = g["entity_type"]

        for j, p in enumerate(preds):
            if p.entity_type != g_type:
                continue

            p_name = p.name_or_statement
            if are_semantically_equivalent(g_name, g_synonyms, p_name):
                # 1.0 weight for semantic equivalence
                cost_matrix[i, j] = 0.0

    # Hungarian algorithm: finds minimum weight assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: List[MatchPair] = []
    matched_gold_indices: Set[int] = set()
    matched_pred_indices: Set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] == 0.0:  # Matches semantic equivalence
            matched_gold_indices.add(r)
            matched_pred_indices.add(c)
            matched_pairs.append(
                MatchPair(
                    gold_id=golds[r]["local_id"],
                    predicted_id=preds[c].candidate_local_id,
                    entity_type=golds[r]["entity_type"],
                    similarity_score=1.0,
                    is_semantic_match=True,
                )
            )

    unmatched_gold_ids = [golds[i]["local_id"] for i in range(n_gold) if i not in matched_gold_indices]
    unmatched_pred_ids = [preds[j].candidate_local_id for j in range(n_pred) if j not in matched_pred_indices]

    tp = len(matched_pairs)
    fp = len(unmatched_pred_ids)
    fn = len(unmatched_gold_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return BipartiteMatchingResult(
        matched_pairs=matched_pairs,
        unmatched_gold_ids=unmatched_gold_ids,
        unmatched_predicted_ids=unmatched_pred_ids,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )
