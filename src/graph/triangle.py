"""Triangle Evidence Aggregation.

Assesses mutual 3-way consistency between (S1, S2, S3) triplets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def compute_triangle_support(
    s1_s2_score: float,
    s1_s3_score: float,
    s2_s3_score: float,
    threshold: float = 0.65,
) -> Dict[str, float]:
    """Calculate triangle closure metrics for a triplet (S1, S2, S3).

    Args:
        s1_s2_score: Direct pairwise score between S1 and S2.
        s1_s3_score: Direct pairwise score between S1 and S3.
        s2_s3_score: Cross-witness pairwise score between S2 and S3.
        threshold: Minimum score threshold to consider an edge active.

    Returns:
        Dictionary containing triangle consistency features.
    """
    closed = 1.0 if (s1_s2_score >= threshold and s1_s3_score >= threshold and s2_s3_score >= threshold) else 0.0
    min_edge = min(s1_s2_score, s1_s3_score, s2_s3_score)
    mean_edge = (s1_s2_score + s1_s3_score + s2_s3_score) / 3.0

    return {
        "triangle_closed": closed,
        "triangle_min_edge": min_edge,
        "triangle_mean_edge": mean_edge,
    }
