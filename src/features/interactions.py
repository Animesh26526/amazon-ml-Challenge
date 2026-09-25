"""Cross-Field Interaction Features.

Combines name and address signals to detect strong agreement, partial matches,
or asymmetric confidence.
"""

from __future__ import annotations

from typing import Dict


def compute_interaction_features(
    name_features: Dict[str, float],
    address_features: Dict[str, float],
) -> Dict[str, float]:
    """Compute interaction features between name and address similarities.

    Args:
        name_features: Precomputed name feature dictionary.
        address_features: Precomputed address feature dictionary.

    Returns:
        Dictionary of interaction feature values.
    """
    name_sim = name_features.get("name_token_jaccard", 0.0)
    addr_sim = address_features.get("addr_token_jaccard", 0.0)
    name_edit = name_features.get("name_edit_similarity", 0.0)
    addr_edit = address_features.get("addr_edit_similarity", 0.0)
    missing_addr = address_features.get("addr_missing_either", 0.0)

    product_sim = name_sim * addr_sim
    min_sim = min(name_sim, addr_sim)
    mean_sim = (name_sim + addr_sim) / 2.0

    # Strong name with weak or unobserved address
    name_strong_addr_weak = 1.0 if (name_edit > 0.85 and addr_sim < 0.2) else 0.0
    # Strong address with weak name (e.g. different business at same location)
    addr_strong_name_weak = 1.0 if (addr_edit > 0.85 and name_sim < 0.2) else 0.0
    # Both fields strongly agree
    both_strong = 1.0 if (name_edit > 0.80 and addr_edit > 0.80) else 0.0

    return {
        "interaction_product": product_sim,
        "interaction_min": min_sim,
        "interaction_mean": mean_sim,
        "interaction_name_strong_addr_weak": name_strong_addr_weak,
        "interaction_addr_strong_name_weak": addr_strong_name_weak,
        "interaction_both_strong": both_strong,
        "interaction_name_strong_addr_missing": 1.0 if (name_edit > 0.85 and missing_addr > 0.5) else 0.0,
    }
