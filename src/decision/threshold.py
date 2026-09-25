"""Threshold Filtering Utilities.

Applies global and source-specific thresholds to candidate predictions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def apply_source_specific_thresholds(
    scored_candidates: Sequence[Tuple[str, float]],
    s2_threshold: float = 0.50,
    s3_threshold: float = 0.50,
) -> List[Tuple[str, float]]:
    """Filter candidates using source-specific thresholds for S2 vs S3."""
    accepted = []
    for tgt_id, score in scored_candidates:
        if tgt_id.startswith("S2-") and score >= s2_threshold:
            accepted.append((tgt_id, score))
        elif tgt_id.startswith("S3-") and score >= s3_threshold:
            accepted.append((tgt_id, score))
        elif not tgt_id.startswith(("S2-", "S3-")) and score >= min(s2_threshold, s3_threshold):
            accepted.append((tgt_id, score))
    return accepted
