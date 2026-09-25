"""TF-IDF Lexical Blocking Route.

Retrieves top-k candidates per entity using sparse TF-IDF cosine similarity.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.blocking import BaseBlockingRule
from src.normalization import NormalizedRecord


class TfidfBlocking(BaseBlockingRule):
    """Retrieves candidates using sparse TF-IDF representation and cosine similarity."""

    name: str = "tfidf"

    def __init__(
        self,
        top_k: int = 15,
        threshold: float = 0.4,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
    ) -> None:
        self.top_k = top_k
        self.threshold = threshold
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            analyzer="word",
        )
        self._target_ids: List[str] = []
        self._target_matrix = None

    def build_index(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Fit vectorizer and index target record text representations."""
        self._target_ids = []
        corpus: List[str] = []
        for rec in target_records:
            # Combine name and address for comprehensive retrieval
            text = f"{rec.business_name_norm} {rec.business_address_norm}".strip()
            self._target_ids.append(rec.entity_id)
            corpus.append(text)

        if corpus:
            self._target_matrix = self.vectorizer.fit_transform(corpus)
        else:
            self._target_matrix = None

    def query(self, query_record: NormalizedRecord) -> List[str]:
        """Query top-k candidate IDs with cosine similarity >= threshold."""
        if self._target_matrix is None or not self._target_ids:
            return []

        text = f"{query_record.business_name_norm} {query_record.business_address_norm}".strip()
        if not text:
            return []

        q_vec = self.vectorizer.transform([text])
        scores = linear_kernel(q_vec, self._target_matrix).flatten()

        if len(scores) == 0:
            return []

        # Find indices above threshold
        above_mask = scores >= self.threshold
        indices = np.where(above_mask)[0]

        if len(indices) == 0:
            return []

        # Sort top-k
        if len(indices) > self.top_k:
            top_local = np.argpartition(scores[indices], -self.top_k)[-self.top_k :]
            indices = indices[top_local]

        sorted_indices = indices[np.argsort(-scores[indices])]
        return [self._target_ids[idx] for idx in sorted_indices]
