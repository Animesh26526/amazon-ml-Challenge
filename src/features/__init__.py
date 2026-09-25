"""Pairwise Feature Engineering Interfaces.

Defines the feature extraction protocols and combines name, address, interaction,
and retrieval route metadata features for candidate pairs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.blocking import CandidatePair
from src.normalization import NormalizedRecord


def jaccard_similarity(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Compute Jaccard similarity between two token sequences."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def token_overlap_count(tokens1: Sequence[str], tokens2: Sequence[str]) -> int:
    """Compute count of overlapping tokens between two sequences."""
    return len(set(tokens1) & set(tokens2))


import numpy as np

try:
    import numba

    @numba.njit(fastmath=True)
    def _numba_levenshtein(a: np.ndarray, b: np.ndarray) -> int:
        la, lb = len(a), len(b)
        dp = np.empty((la + 1, lb + 1), dtype=np.int32)
        for i in range(la + 1):
            dp[i, 0] = i
        for j in range(lb + 1):
            dp[0, j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
        return dp[la, lb]

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def normalized_edit_similarity(s1: str, s2: str) -> float:
    """Compute normalized edit similarity in range [0.0, 1.0].

    Uses Numba JIT when available, falling back to pure-Python dynamic programming.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if abs(len1 - len2) / max_len > 0.6:
        return 0.0

    if _HAS_NUMBA:
        a = np.frombuffer(s1.encode("latin1", "replace"), dtype=np.uint8)
        b = np.frombuffer(s2.encode("latin1", "replace"), dtype=np.uint8)
        dist = _numba_levenshtein(a, b)
        return 1.0 - (dist / max_len)

    # Two-row pure-Python fallback Levenshtein distance
    prev_row = list(range(len2 + 1))
    curr_row = [0] * (len2 + 1)

    for i, c1 in enumerate(s1):
        curr_row[0] = i + 1
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row[j + 1] = min(
                curr_row[j] + 1,        # insertion
                prev_row[j + 1] + 1,    # deletion
                prev_row[j] + cost,     # substitution
            )
        prev_row, curr_row = curr_row, prev_row

    dist = prev_row[len2]
    return 1.0 - (dist / max_len)
