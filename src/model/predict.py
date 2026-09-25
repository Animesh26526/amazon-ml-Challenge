"""Pairwise Inference Pipeline.

Scores candidate pairs using a trained matching model.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.model import BaseMatcher


def score_candidate_pairs(
    matcher: BaseMatcher,
    features: np.ndarray,
    candidate_keys: Sequence[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Score candidate pairs and group results by Source 1 entity ID.

    Args:
        matcher: Trained BaseMatcher instance.
        features: 2D numpy array of shape (N, D).
        candidate_keys: Sequence of (s1_id, target_id) corresponding to each row.

    Returns:
        Mapping from s1_id to list of (target_id, score) pairs.
    """
    if len(features) == 0:
        return {}

    scores = matcher.predict_proba(features)
    grouped: Dict[str, List[Tuple[str, float]]] = {}

    for (s1_id, tgt_id), score in zip(candidate_keys, scores):
        if s1_id not in grouped:
            grouped[s1_id] = []
        grouped[s1_id].append((tgt_id, float(score)))

    return grouped
