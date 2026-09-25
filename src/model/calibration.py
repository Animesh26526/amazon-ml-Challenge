"""Score Calibration and Threshold Optimization.

Searches for optimal global and source-specific decision thresholds
specifically targeting validation macro F_0.5.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def find_optimal_threshold(
    eval_fn: Callable[[float], float],
    min_threshold: float = 0.20,
    max_threshold: float = 0.90,
    num_steps: int = 71,
) -> Tuple[float, float]:
    """Search for the scalar threshold that maximizes the evaluation metric.

    Args:
        eval_fn: Function mapping candidate threshold float to macro F_0.5 score.
        min_threshold: Lower bound for search grid.
        max_threshold: Upper bound for search grid.
        num_steps: Number of grid evaluation points.

    Returns:
        (best_threshold, best_score)
    """
    thresholds = np.linspace(min_threshold, max_threshold, num_steps)
    best_thresh = 0.50
    best_score = -1.0

    for thresh in thresholds:
        score = eval_fn(float(thresh))
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh, best_score
