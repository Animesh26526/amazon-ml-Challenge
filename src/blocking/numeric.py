"""Numeric Anchor Blocking Route.

Indexes records by numeric tokens found in addresses (e.g. house numbers, PIN/ZIP codes).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set

from src.blocking import BaseBlockingRule
from src.normalization import NormalizedRecord


class NumericAnchorBlocking(BaseBlockingRule):
    """Retrieves candidates sharing numeric tokens combined with primary name prefixes."""

    name: str = "numeric_anchors"

    def __init__(self, max_candidates_per_key: int = 100) -> None:
        self.max_candidates_per_key = max_candidates_per_key
        self._index: Dict[str, List[str]] = defaultdict(list)

    def _make_keys(self, record: NormalizedRecord) -> List[str]:
        keys = []
        # Prefix of the name combined with numeric anchor
        name_prefix = record.business_name_norm[:4] if len(record.business_name_norm) >= 4 else ""
        for num in record.numeric_anchors:
            if len(num) >= 3:  # meaningful number like house number or PIN code
                if name_prefix:
                    keys.append(f"{name_prefix}#{num}")
                keys.append(f"NUM_{num}")
        return keys

    def build_index(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Build index from numeric anchors."""
        self._index.clear()
        for rec in target_records:
            keys = self._make_keys(rec)
            for k in keys:
                if len(self._index[k]) < self.max_candidates_per_key:
                    self._index[k].append(rec.entity_id)

    def query(self, query_record: NormalizedRecord) -> List[str]:
        """Query candidates sharing numeric anchor keys."""
        keys = self._make_keys(query_record)
        candidates: Set[str] = set()
        for k in keys:
            candidates.update(self._index.get(k, []))
        return list(candidates)
