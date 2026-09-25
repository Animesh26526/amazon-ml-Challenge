"""Candidate Recall and Volume Evaluation.

Measures candidate generation quality against ground truth.

Candidate Recall = (true matches present in candidate set) / (total true matches)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple


def compute_candidate_recall(
    candidates: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
) -> Dict[str, float]:
    """Compute candidate recall, total pairs, and reduction metrics.

    Args:
        candidates: Mapping from s1_id to sequence of candidate target IDs.
        ground_truth: Mapping from s1_id to sequence of true target IDs.

    Returns:
        Dictionary of candidate generation metrics.
    """
    total_true_matches = 0
    covered_true_matches = 0
    total_candidate_pairs = 0

    for s1_id, true_matches in ground_truth.items():
        true_set = {t.strip() for t in true_matches if t.strip()}
        total_true_matches += len(true_set)

        cand_set = {c.strip() for c in candidates.get(s1_id, []) if c.strip()}
        total_candidate_pairs += len(cand_set)
        covered_true_matches += len(true_set & cand_set)

    recall = (
        (covered_true_matches / total_true_matches)
        if total_true_matches > 0
        else 1.0
    )
    avg_candidates = (
        (total_candidate_pairs / len(ground_truth))
        if ground_truth
        else 0.0
    )

    return {
        "candidate_recall": recall,
        "covered_true_matches": float(covered_true_matches),
        "total_true_matches": float(total_true_matches),
        "total_candidate_pairs": float(total_candidate_pairs),
        "avg_candidates_per_s1": avg_candidates,
    }
