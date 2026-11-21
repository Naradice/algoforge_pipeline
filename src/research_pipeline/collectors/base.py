"""Abstract base class for all research source collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from research_pipeline.collectors.models import SearchQuery, CandidateRef, CollectedContent


class BaseCollector(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return 'paper' or 'oss'."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[CandidateRef]:
        """Perform search and return list of candidate references."""

    @abstractmethod
    async def fetch(self, candidate: CandidateRef) -> CollectedContent:
        """Fetch full metadata and text content for a candidate reference."""
