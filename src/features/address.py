"""Pairwise Address Similarity Features.

Computes lexical, token, numeric anchor overlap, and explicit missing-value features.

PRD Constraint:
Missing address != contradiction. Missingness is captured as an explicit indicator
so the gradient-boosted matcher can learn distinct representations.
"""

from __future__ import annotations

from typing import Dict

from src.features import jaccard_similarity, normalized_edit_similarity, token_overlap_count
from src.normalization import NormalizedRecord


def compute_address_features(rec1: NormalizedRecord, rec2: NormalizedRecord) -> Dict[str, float]:
    """Compute pairwise address similarity and missingness features.

    Args:
        rec1: Source 1 normalized record.
        rec2: Target (S2/S3) normalized record.

    Returns:
        Dictionary of numeric feature values.
    """
    addr1 = rec1.business_address_norm
    addr2 = rec2.business_address_norm

    missing_s1 = 1.0 if rec1.is_address_missing else 0.0
    missing_s2_s3 = 1.0 if rec2.is_address_missing else 0.0
    missing_either = 1.0 if (rec1.is_address_missing or rec2.is_address_missing) else 0.0

    # If either address is missing, similarity is set to default neutral (0.0)
    # but the explicit missing indicator alerts the model that it's unobserved.
    if rec1.is_address_missing or rec2.is_address_missing:
        return {
            "addr_exact": 0.0,
            "addr_token_jaccard": 0.0,
            "addr_token_overlap": 0.0,
            "addr_edit_similarity": 0.0,
            "addr_numeric_overlap": 0.0,
            "addr_missing_s1": missing_s1,
            "addr_missing_target": missing_s2_s3,
            "addr_missing_either": missing_either,
        }

    exact_match = 1.0 if addr1 == addr2 else 0.0
    tok_jaccard = jaccard_similarity(rec1.address_tokens, rec2.address_tokens)
    tok_overlap = float(token_overlap_count(rec1.address_tokens, rec2.address_tokens))
    edit_sim = normalized_edit_similarity(addr1, addr2)

    # Numeric token overlap (e.g. house number, PIN code match)
    num1 = set(rec1.numeric_anchors)
    num2 = set(rec2.numeric_anchors)
    num_overlap = float(len(num1 & num2))

    return {
        "addr_exact": exact_match,
        "addr_token_jaccard": tok_jaccard,
        "addr_token_overlap": tok_overlap,
        "addr_edit_similarity": edit_sim,
        "addr_numeric_overlap": num_overlap,
        "addr_missing_s1": missing_s1,
        "addr_missing_target": missing_s2_s3,
        "addr_missing_either": missing_either,
    }
