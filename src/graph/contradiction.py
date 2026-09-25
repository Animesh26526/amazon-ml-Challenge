"""Contradiction vs Neutral vs Support Analysis.

Explicitly categorizes candidate relationships into SUPPORT, NEUTRAL, or CONTRADICTION.

PRD Guideline:
Weak evidence is NOT automatically contradiction. Missing addresses or partial
information must remain NEUTRAL / UNKNOWN. Only explicit, high-confidence
incompatibility constitutes CONTRADICTION.
"""

from __future__ import annotations

from src.features import normalized_edit_similarity
from src.graph import CrossSourceEvidenceType
from src.normalization import NormalizedRecord


def classify_relationship(
    rec1: NormalizedRecord,
    rec2: NormalizedRecord,
    name_threshold_support: float = 0.85,
    name_threshold_contradiction: float = 0.30,
) -> CrossSourceEvidenceType:
    """Classify the relationship between two records into SUPPORT, NEUTRAL, or CONTRADICTION.

    Args:
        rec1: Normalized record 1.
        rec2: Normalized record 2.
        name_threshold_support: Minimum name edit similarity for direct support.
        name_threshold_contradiction: Upper bound on name similarity for hard contradiction.

    Returns:
        CrossSourceEvidenceType enum value.
    """
    name_sim = normalized_edit_similarity(
        rec1.business_name_norm, rec2.business_name_norm
    )

    # If country labels are known, non-empty, and explicitly conflict -> Contradiction
    if rec1.country and rec2.country and rec1.country != rec2.country:
        return CrossSourceEvidenceType.CONTRADICTION

    # If both addresses are non-empty and have zero token overlap while names are also distant
    if (
        not rec1.is_address_missing
        and not rec2.is_address_missing
        and name_sim < name_threshold_contradiction
    ):
        return CrossSourceEvidenceType.CONTRADICTION

    if name_sim >= name_threshold_support:
        return CrossSourceEvidenceType.SUPPORT

    # Default conservative position: NEUTRAL
    return CrossSourceEvidenceType.NEUTRAL
