"""Document-Level Cluster Bootstrap and Small-Sample Confidence Intervals.

Implements:
- Document-level Cluster Bootstrap (B=1000) for 95% Confidence Intervals (Primary CI)
- Wilson score interval for proportions
- Multi-label Cohen's Kappa for sentence annotations across 6 concept classes + NONE
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel

from research_pipeline.extractors.raw_artifact import ExtractionTrioBundle
from research_pipeline.evaluation.metrics import evaluate_run, FullEvaluationReport


class BootstrapConfidenceInterval(BaseModel):
    metric_name: str
    point_estimate: float
    ci_lower_2_5: float
    ci_upper_97_5: float


def calculate_wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate exact Wilson score confidence interval for a single binomial proportion."""
    if total == 0:
        return 0.0, 1.0

    z = 1.95996  # 95% confidence z-score
    p_hat = successes / total

    denominator = 1 + (z**2) / total
    center_adjusted = (p_hat + (z**2) / (2 * total)) / denominator
    margin = (
        z * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2) / (4 * (total**2)))
    ) / denominator

    lower = max(0.0, center_adjusted - margin)
    upper = min(1.0, center_adjusted + margin)
    return round(lower, 4), round(upper, 4)


def run_cluster_bootstrap(
    condition_id: str,
    gold_records: List[Dict[str, Any]],
    bundles: List[ExtractionTrioBundle],
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, BootstrapConfidenceInterval]:
    """Execute Document-Level Cluster Bootstrap resampling (B=1000).
    
    Resamples whole documents with replacement to preserve intra-document correlation.
    """
    rng = np.random.default_rng(seed)
    n_docs = len(gold_records)
    if n_docs == 0:
        return {}

    # Initial point estimates
    point_report = evaluate_run(condition_id, gold_records, bundles)

    bootstrap_overall_f1: List[float] = []
    bootstrap_cap_f1: List[float] = []
    bootstrap_mod_f1: List[float] = []
    bootstrap_macro_f1: List[float] = []
    bootstrap_qv: List[float] = []

    bundle_by_source = {b.source_id: b for b in bundles}

    for _ in range(n_iterations):
        sample_indices = rng.integers(0, n_docs, size=n_docs)
        sampled_golds = [gold_records[i] for i in sample_indices]
        sampled_bundles: List[ExtractionTrioBundle] = []

        for g in sampled_golds:
            b = bundle_by_source.get(g["source_id"])
            if b is not None:
                sampled_bundles.append(b)

        sample_report = evaluate_run(condition_id, sampled_golds, sampled_bundles)
        bootstrap_overall_f1.append(sample_report.overall_f1)
        bootstrap_macro_f1.append(sample_report.macro_f1)
        bootstrap_qv.append(sample_report.resolved_quote_validity_rate)

        cap_m = sample_report.class_metrics.get("capability")
        if cap_m:
            bootstrap_cap_f1.append(cap_m.f1)

        mod_m = sample_report.class_metrics.get("model")
        if mod_m:
            bootstrap_mod_f1.append(mod_m.f1)

    ci_results: Dict[str, BootstrapConfidenceInterval] = {}

    def get_ci(name: str, point: float, values: List[float]) -> BootstrapConfidenceInterval:
        lo = float(np.percentile(values, 2.5))
        hi = float(np.percentile(values, 97.5))
        return BootstrapConfidenceInterval(
            metric_name=name,
            point_estimate=point,
            ci_lower_2_5=round(lo, 4),
            ci_upper_97_5=round(hi, 4),
        )

    ci_results["overall_f1"] = get_ci("overall_f1", point_report.overall_f1, bootstrap_overall_f1)
    ci_results["macro_f1"] = get_ci("macro_f1", point_report.macro_f1, bootstrap_macro_f1)
    ci_results["resolved_quote_validity_rate"] = get_ci("resolved_quote_validity_rate", point_report.resolved_quote_validity_rate, bootstrap_qv)

    if bootstrap_cap_f1:
        cap_point = point_report.class_metrics.get("capability")
        ci_results["capability_f1"] = get_ci("capability_f1", cap_point.f1 if cap_point else 0.0, bootstrap_cap_f1)

    if bootstrap_mod_f1:
        mod_point = point_report.class_metrics.get("model")
        ci_results["model_f1"] = get_ci("model_f1", mod_point.f1 if mod_point else 0.0, bootstrap_mod_f1)

    return ci_results


def calculate_multilabel_cohens_kappa(
    rater1_labels: List[Dict[str, int]],  # List of {class: 0 or 1} for each sentence
    rater2_labels: List[Dict[str, int]],
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate Cohen's Kappa per concept class and macro-average for multi-label sentence annotations.
    
    Includes negative class 'NONE' where all class flags are 0.
    """
    if len(rater1_labels) != len(rater2_labels):
        raise ValueError("Raters must annotate the exact same number of sentences")

    target_classes = classes or ["phenomenon", "model", "method", "capability", "assumption", "limitation"]
    n_sentences = len(rater1_labels)
    if n_sentences == 0:
        return {}

    kappa_per_class: Dict[str, float] = {}

    for c in target_classes:
        # Binary 2x2 contingency table for class c
        # a: r1=1, r2=1
        # b: r1=1, r2=0
        # c_: r1=0, r2=1
        # d: r1=0, r2=0
        a = sum(1 for i in range(n_sentences) if rater1_labels[i].get(c, 0) == 1 and rater2_labels[i].get(c, 0) == 1)
        b = sum(1 for i in range(n_sentences) if rater1_labels[i].get(c, 0) == 1 and rater2_labels[i].get(c, 0) == 0)
        c_ = sum(1 for i in range(n_sentences) if rater1_labels[i].get(c, 0) == 0 and rater2_labels[i].get(c, 0) == 1)
        d = sum(1 for i in range(n_sentences) if rater1_labels[i].get(c, 0) == 0 and rater2_labels[i].get(c, 0) == 0)

        p_o = (a + d) / n_sentences
        p_yes = ((a + b) / n_sentences) * ((a + c_) / n_sentences)
        p_no = ((c_ + d) / n_sentences) * ((b + d) / n_sentences)
        p_e = p_yes + p_no

        if p_e == 1.0:
            kappa = 1.0
        else:
            kappa = (p_o - p_e) / (1.0 - p_e)

        kappa_per_class[c] = round(kappa, 4)

    # Macro average
    macro_kappa = sum(kappa_per_class.values()) / len(kappa_per_class) if kappa_per_class else 0.0
    kappa_per_class["macro_average"] = round(macro_kappa, 4)

    return kappa_per_class
