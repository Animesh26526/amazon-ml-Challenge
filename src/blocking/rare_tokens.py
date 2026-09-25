"""Rare Token Inverted Index Blocking Route.

Identifies discriminative tokens (e.g. unique business name tokens) and indexes
records sharing these rare identifiers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set

from src.blocking import BaseBlockingRule
from src.normalization import NormalizedRecord


class RareTokenBlocking(BaseBlockingRule):
    """Retrieves candidates sharing rare, highly discriminative tokens."""

    name: str = "rare_tokens"

    def __init__(
        self,
        min_doc_freq: int = 2,
        max_doc_freq: int = 50,
        max_candidates: int = 50,
    ) -> None:
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq = max_doc_freq
        self.max_candidates = max_candidates
        self._doc_frequencies: Counter = Counter()
        self._index: Dict[str, List[str]] = defaultdict(list)

    def build_index(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Count document frequencies and index records by rare tokens."""
        self._doc_frequencies.clear()
        self._index.clear()

        # Temporary buffer of records
        records_list: List[NormalizedRecord] = list(target_records)

        # 1. Count frequencies
        for rec in records_list:
            seen_tokens: Set[str] = set(rec.name_tokens)
            for tok in seen_tokens:
                if len(tok) >= 4 and not tok.isdigit():
                    self._doc_frequencies[tok] += 1

        # 2. Index rare tokens
        for rec in records_list:
            for tok in rec.name_tokens:
                freq = self._doc_frequencies[tok]
                if self.min_doc_freq <= freq <= self.max_doc_freq:
                    if len(self._index[tok]) < self.max_candidates:
                        self._index[tok].append(rec.entity_id)

    def query(self, query_record: NormalizedRecord) -> List[str]:
        """Query candidates matching any rare tokens from the query record."""
        candidates: Set[str] = set()
        for tok in query_record.name_tokens:
            freq = self._doc_frequencies.get(tok, 0)
            if self.min_doc_freq <= freq <= self.max_doc_freq:
                postings = self._index.get(tok, [])
                candidates.update(postings)
                if len(candidates) >= self.max_candidates:
                    break
        return list(candidates)
