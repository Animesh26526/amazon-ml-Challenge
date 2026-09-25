"""Evaluation Metrics for Entity Resolution.

Implements macro-averaged F_0.5 per Source 1 entity, strictly adhering to the
competition formula and singleton credit rules.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple


def compute_per_entity_f05(
    predicted_ids: Iterable[str],
    true_ids: Iterable[str],
) -> Tuple[float, float, float]:
    """Compute (precision, recall, F_0.5) for a single Source 1 entity.

    Rules:
    - If true_ids is empty (true singleton):
        - If predicted_ids is empty: F_0.5 = 1.0, precision = 1.0, recall = 1.0
        - If predicted_ids is non-empty: F_0.5 = 0.0, precision = 0.0, recall = 0.0
    - If true_ids is non-empty:
        - If predicted_ids is empty: F_0.5 = 0.0, precision = 0.0, recall = 0.0
        - Standard precision and recall:
            TP = len(predicted & true)
            Precision = TP / len(predicted)
            Recall = TP / len(true)
            F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall) if denom > 0 else 0.0

    Args:
        predicted_ids: Iterable of predicted entity IDs.
        true_ids: Iterable of ground truth entity IDs.

    Returns:
        (precision, recall, f05)
    """
    pred_set = {p.strip() for p in predicted_ids if p.strip()}
    true_set = {t.strip() for t in true_ids if t.strip()}

    # Case 1: True singleton (no true matches)
    if not true_set:
        if not pred_set:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    # Case 2: True non-singleton, but predicted empty
    if not pred_set:
        return 0.0, 0.0, 0.0

    # Case 3: Both non-empty
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = (0.25 * precision) + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0.0 else 0.0

    return precision, recall, f05


def compute_macro_f05(
    predictions: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
) -> float:
    """Compute macro-averaged F_0.5 across all Source 1 entities in ground_truth.

    Args:
        predictions: Mapping from s1_id to sequence of predicted target IDs.
        ground_truth: Mapping from s1_id to sequence of true target IDs.

    Returns:
        Macro-averaged F_0.5 score.
    """
    if not ground_truth:
        return 0.0

    total_f05 = 0.0
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        _, _, f05 = compute_per_entity_f05(pred_matches, true_matches)
        total_f05 += f05

    return total_f05 / len(ground_truth)


def evaluate_predictions(
    predictions: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics across all entities.

    Returns:
        Dictionary with 'macro_f0_5', 'mean_precision', 'mean_recall',
        'singleton_accuracy', and 'total_entities'.
    """
    if not ground_truth:
        return {
            "macro_f0_5": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "singleton_accuracy": 0.0,
            "total_entities": 0,
        }

    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    singleton_count = 0
    singleton_correct = 0

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        p, r, f05 = compute_per_entity_f05(pred_matches, true_matches)
        total_prec += p
        total_rec += r
        total_f05 += f05

        if len(true_matches) == 0:
            singleton_count += 1
            if len(pred_matches) == 0:
                singleton_correct += 1

    n = len(ground_truth)
    singleton_acc = (
        (singleton_correct / singleton_count) if singleton_count > 0 else 1.0
    )

    return {
        "macro_f0_5": total_f05 / n,
        "mean_precision": total_prec / n,
        "mean_recall": total_rec / n,
        "singleton_accuracy": singleton_acc,
        "total_entities": float(n),
    }
