"""Subgroup and Cohort Diagnostic Evaluation.

Evaluates performance across key cohorts:
- singletons (0 matches)
- single-match (1 match)
- multi-match (>1 matches)
- S2-only matches
- S3-only matches
- both-source matches
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set

from src.evaluation.metrics import compute_per_entity_f05


def evaluate_cohorts(
    predictions: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by cardinality and source composition cohorts.

    Args:
        predictions: Mapping from s1_id to sequence of predicted target IDs.
        ground_truth: Mapping from s1_id to sequence of true target IDs.

    Returns:
        Mapping from cohort name to dictionary of metric values.
    """
    cohorts: Dict[str, List[str]] = {
        "all": list(ground_truth.keys()),
        "singletons": [],
        "one_match": [],
        "multi_match": [],
        "s2_only": [],
        "s3_only": [],
        "both_sources": [],
    }

    for s1_id, matches in ground_truth.items():
        clean_matches = [m.strip() for m in matches if m.strip()]
        n_matches = len(clean_matches)

        if n_matches == 0:
            cohorts["singletons"].append(s1_id)
        elif n_matches == 1:
            cohorts["one_match"].append(s1_id)
        else:
            cohorts["multi_match"].append(s1_id)

        has_s2 = any(m.startswith("S2-") for m in clean_matches)
        has_s3 = any(m.startswith("S3-") for m in clean_matches)

        if has_s2 and not has_s3:
            cohorts["s2_only"].append(s1_id)
        elif has_s3 and not has_s2:
            cohorts["s3_only"].append(s1_id)
        elif has_s2 and has_s3:
            cohorts["both_sources"].append(s1_id)

    results: Dict[str, Dict[str, float]] = {}

    for cohort_name, s1_ids in cohorts.items():
        if not s1_ids:
            results[cohort_name] = {
                "count": 0.0,
                "macro_f0_5": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
            }
            continue

        tot_f05 = tot_p = tot_r = 0.0
        for s1_id in s1_ids:
            p, r, f05 = compute_per_entity_f05(
                predictions.get(s1_id, []), ground_truth.get(s1_id, [])
            )
            tot_p += p
            tot_r += r
            tot_f05 += f05

        n = len(s1_ids)
        results[cohort_name] = {
            "count": float(n),
            "macro_f0_5": tot_f05 / n,
            "mean_precision": tot_p / n,
            "mean_recall": tot_r / n,
        }

    return results
