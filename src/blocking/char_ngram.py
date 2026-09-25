"""Character N-gram Blocking Route.

Retrieves candidates sharing significant character n-gram overlaps to tolerate
spelling variations, typos, and minor OCR/formatting errors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set

from src.blocking import BaseBlockingRule
from src.normalization import NormalizedRecord, extract_char_ngrams


class CharNgramBlocking(BaseBlockingRule):
    """Indexes records by character n-grams and retrieves candidates with high overlap."""

    name: str = "char_ngram"

    def __init__(
        self,
        n: int = 3,
        min_overlap: int = 3,
        max_candidates_per_key: int = 500,
    ) -> None:
        self.n = n
        self.min_overlap = min_overlap
        self.max_candidates_per_key = max_candidates_per_key
        self._inverted_index: Dict[str, List[str]] = defaultdict(list)

    def build_index(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Build character n-gram inverted index over normalized names."""
        self._inverted_index.clear()
        for rec in target_records:
            if not rec.business_name_norm:
                continue
            ngrams = extract_char_ngrams(rec.business_name_norm, self.n)
            for gram in ngrams:
                postings = self._inverted_index[gram]
                if len(postings) < self.max_candidates_per_key:
                    postings.append(rec.entity_id)

    def query(self, query_record: NormalizedRecord) -> List[str]:
        """Retrieve candidate IDs sharing at least min_overlap character n-grams."""
        if not query_record.business_name_norm:
            return []

        ngrams = extract_char_ngrams(query_record.business_name_norm, self.n)
        counts: Dict[str, int] = defaultdict(int)

        for gram in ngrams:
            if gram in self._inverted_index:
                for tgt_id in self._inverted_index[gram]:
                    counts[tgt_id] += 1

        return [tgt_id for tgt_id, cnt in counts.items() if cnt >= self.min_overlap]
