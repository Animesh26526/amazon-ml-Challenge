"""S1-Centered Cross-Source Witness Evidence.

Examines whether candidate S2 and S3 records for a common S1 entity exhibit
mutual consistency, acting as mutual witness evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.features import normalized_edit_similarity
from src.normalization import NormalizedRecord


def compute_witness_score(rec_s2: NormalizedRecord, rec_s3: NormalizedRecord) -> float:
    """Compute pairwise witness agreement score between candidate S2 and S3 records."""
    name_sim = normalized_edit_similarity(
        rec_s2.business_name_norm, rec_s3.business_name_norm
    )
    if not rec_s2.is_address_missing and not rec_s3.is_address_missing:
        addr_sim = normalized_edit_similarity(
            rec_s2.business_address_norm, rec_s3.business_address_norm
        )
        return (name_sim * 0.6) + (addr_sim * 0.4)
    return name_sim


def find_strongest_witness(
    target_rec: NormalizedRecord,
    other_candidates: List[NormalizedRecord],
    min_witness_threshold: float = 0.70,
) -> Tuple[Optional[str], float]:
    """Find the highest-scoring witness candidate of the opposing source (S2 vs S3).

    Args:
        target_rec: Candidate record being evaluated.
        other_candidates: S1's candidates from the complementary source.
        min_witness_threshold: Minimum agreement score to qualify as a valid witness.

    Returns:
        (best_witness_id, best_score)
    """
    best_id = None
    best_score = 0.0

    for cand in other_candidates:
        # Cross-source check: S2 pairs with S3
        if (target_rec.entity_id.startswith("S2-") and cand.entity_id.startswith("S3-")) or (
            target_rec.entity_id.startswith("S3-") and cand.entity_id.startswith("S2-")
        ):
            score = compute_witness_score(target_rec, cand)
            if score > best_score:
                best_score = score
                best_id = cand.entity_id

    if best_score >= min_witness_threshold:
        return best_id, best_score
    return None, 0.0
