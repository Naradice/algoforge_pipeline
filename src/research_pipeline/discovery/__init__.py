"""Step H2: Hypothesis Discovery Engine (HDE-H2).

Discovers cross-domain algorithmic trading hypotheses by traversing
relational knowledge graphs, finding multi-hop analogy paths, and formulating
falsifiable, backtestable trading strategy hypotheses for AlgoForge.
"""

from research_pipeline.discovery.models import (
    GraphPath,
    PathHop,
    HypothesisCandidate,
    HypothesisEvaluationResult,
    AlgoForgeBacktestSpec,
)
from research_pipeline.discovery.path_finder import MultiHopPathFinder

__all__ = [
    "GraphPath",
    "PathHop",
    "HypothesisCandidate",
    "HypothesisEvaluationResult",
    "AlgoForgeBacktestSpec",
    "MultiHopPathFinder",
]
