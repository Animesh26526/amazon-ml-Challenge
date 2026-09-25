"""Pairwise Name Similarity Features.

Calculates lexical, token, edit, and n-gram similarity metrics between business names.
"""

from __future__ import annotations

from typing import Dict

from src.features import jaccard_similarity, normalized_edit_similarity, token_overlap_count
from src.normalization import NormalizedRecord, extract_char_ngrams


def compute_name_features(rec1: NormalizedRecord, rec2: NormalizedRecord) -> Dict[str, float]:
    """Compute pairwise business name similarity features.

    Args:
        rec1: Source 1 normalized record.
        rec2: Target (S2/S3) normalized record.

    Returns:
        Dictionary of numeric feature values.
    """
    name1 = rec1.business_name_norm
    name2 = rec2.business_name_norm

    exact_match = 1.0 if name1 and name2 and name1 == name2 else 0.0
    tok_jaccard = jaccard_similarity(rec1.name_tokens, rec2.name_tokens)
    tok_overlap = float(token_overlap_count(rec1.name_tokens, rec2.name_tokens))
    edit_sim = normalized_edit_similarity(name1, name2)

    # Character 3-gram Jaccard
    ngrams1 = extract_char_ngrams(name1, 3)
    ngrams2 = extract_char_ngrams(name2, 3)
    ngram_sim = (
        len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
        if (ngrams1 or ngrams2)
        else 0.0
    )

    len1 = len(name1)
    len2 = len(name2)
    len_diff = abs(len1 - len2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    return {
        "name_exact": exact_match,
        "name_token_jaccard": tok_jaccard,
        "name_token_overlap": tok_overlap,
        "name_edit_similarity": edit_sim,
        "name_char_ngram_similarity": ngram_sim,
        "name_length_diff": float(len_diff),
        "name_length_ratio": float(len_ratio),
    }
