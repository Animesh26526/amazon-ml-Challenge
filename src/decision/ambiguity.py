"""Ambiguity Detection and Suppression.

Detects ambiguous clusters where multiple candidate matches have nearly identical scores,
indicating uncertain identity or duplicate confusion.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def compute_top_margin(scored_candidates: Sequence[Tuple[str, float]]) -> float:
    """Calculate the margin between the highest and second-highest candidate score."""
    if len(scored_candidates) < 2:
        return 1.0 if len(scored_candidates) == 1 else 0.0
    sorted_scores = sorted([score for _, score in scored_candidates], reverse=True)
    return sorted_scores[0] - sorted_scores[1]


def is_ambiguous(
    scored_candidates: Sequence[Tuple[str, float]],
    margin_threshold: float = 0.03,
    score_threshold: float = 0.60,
) -> bool:
    """Determine if a candidate set is critically ambiguous.

    Occurs when top candidates are above threshold but have tiny margins between them.
    """
    if len(scored_candidates) < 2:
        return False
    sorted_scores = sorted([score for _, score in scored_candidates], reverse=True)
    top1, top2 = sorted_scores[0], sorted_scores[1]
    if top1 >= score_threshold and (top1 - top2) < margin_threshold:
        return True
    return False
