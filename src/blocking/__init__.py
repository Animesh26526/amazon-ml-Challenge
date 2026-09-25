"""Blocking and Candidate Generation Interfaces.

Defines the core data structures and protocols for generating candidate pairs
from multiple independent retrieval routes while tracking discovery provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Set


@dataclass
class CandidatePair:
    """Represents a candidate pair between a Source 1 entity and a target (S2/S3) entity."""

    source1_id: str
    target_id: str
    routes: Set[str] = field(default_factory=set)
    route_scores: Dict[str, float] = field(default_factory=dict)

    def add_route(self, route_name: str, score: float = 1.0) -> None:
        """Record discovery provenance from a blocking route."""
        self.routes.add(route_name)
        if route_name not in self.route_scores or score > self.route_scores[route_name]:
            self.route_scores[route_name] = score


class CandidateSet:
    """Container for managing candidate pairs grouped by Source 1 entity ID."""

    def __init__(self) -> None:
        self._candidates: Dict[str, Dict[str, CandidatePair]] = {}

    def add(self, source1_id: str, target_id: str, route_name: str, score: float = 1.0) -> None:
        """Add or update a candidate pair."""
        s1 = source1_id.strip()
        tgt = target_id.strip()
        if not s1 or not tgt:
            return

        if s1 not in self._candidates:
            self._candidates[s1] = {}

        if tgt not in self._candidates[s1]:
            self._candidates[s1][tgt] = CandidatePair(source1_id=s1, target_id=tgt)

        self._candidates[s1][tgt].add_route(route_name, score)

    def get_candidates_for(self, source1_id: str) -> Dict[str, CandidatePair]:
        """Return all candidate pairs for a given Source 1 ID."""
        return self._candidates.get(source1_id.strip(), {})

    def get_target_ids(self, source1_id: str) -> List[str]:
        """Return list of candidate target IDs for a given Source 1 ID."""
        return list(self._candidates.get(source1_id.strip(), {}).keys())

    def s1_ids(self) -> List[str]:
        """Return list of all registered Source 1 entity IDs."""
        return list(self._candidates.keys())

    def total_pairs(self) -> int:
        """Return total number of candidate pairs across all entities."""
        return sum(len(pairs) for pairs in self._candidates.values())

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to {source1_id: [target_ids]} mapping."""
        return {s1: list(pairs.keys()) for s1, pairs in self._candidates.items()}

    def merge(self, other: CandidateSet) -> None:
        """Merge another CandidateSet into this one, combining route metadata."""
        for s1, pairs in other._candidates.items():
            for tgt, pair in pairs.items():
                for route_name in pair.routes:
                    score = pair.route_scores.get(route_name, 1.0)
                    self.add(s1, tgt, route_name, score)

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self) -> Iterator[str]:
        return iter(self._candidates)


class BaseBlockingRule(ABC):
    """Abstract base class for a candidate generation / blocking route."""

    name: str = "base_blocking"

    @abstractmethod
    def build_index(self, target_records: Iterable[Any]) -> None:
        """Build the blocking index over target records (S2 and S3)."""
        pass

    @abstractmethod
    def query(self, query_record: Any) -> List[str]:
        """Retrieve candidate target IDs for a Source 1 query record."""
        pass
