"""Evaluation and Metric Infrastructure.

Computes macro F_0.5, candidate recall, and subgroup diagnostic cohorts.
"""

from __future__ import annotations

from src.evaluation.metrics import (
    compute_macro_f05,
    compute_per_entity_f05,
    evaluate_predictions,
)
from src.evaluation.candidate_recall import compute_candidate_recall

__all__ = [
    "compute_macro_f05",
    "compute_per_entity_f05",
    "evaluate_predictions",
    "compute_candidate_recall",
]
