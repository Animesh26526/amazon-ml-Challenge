"""Retrieval Route Provenance Features.

Transforms candidate discovery route metadata into ML model features.
"""

from __future__ import annotations

from typing import Dict

from src.blocking import CandidatePair


def compute_retrieval_features(pair: CandidatePair) -> Dict[str, float]:
    """Extract features from the discovery routes of a candidate pair.

    Args:
        pair: CandidatePair instance containing routes and route scores.

    Returns:
        Dictionary of numeric retrieval features.
    """
    known_routes = [
        "exact_name",
        "exact_address",
        "exact_name_address",
        "char_ngram",
        "tfidf",
        "phonetic",
        "rare_tokens",
        "numeric_anchors",
    ]

    features: Dict[str, float] = {}
    for r in known_routes:
        features[f"route_{r}"] = 1.0 if r in pair.routes else 0.0

    features["num_routes_retrieved"] = float(len(pair.routes))
    max_score = max(pair.route_scores.values()) if pair.route_scores else 1.0
    features["max_route_score"] = float(max_score)

    return features
