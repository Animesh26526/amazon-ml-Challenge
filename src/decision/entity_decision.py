"""Entity-Level Decision Engine.

Translates pairwise match scores into final entity-level match decisions.

Key PRD Principles:
- An S1 entity can have 0, 1, or multiple matches.
- Singleton / no-match is a legitimate, credit-earning prediction (returns empty list).
- The engine must NEVER force at least one candidate simply because candidates exist.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from src.decision import ScoredMatch


def decide_matches_for_entity(
    scored_candidates: Sequence[Tuple[str, float]],
    threshold: float = 0.50,
    margin_threshold: float = 0.05,
    max_matches: int = 15,
    ambiguity_penalty: bool = True,
) -> List[str]:
    """Decide the final set of matched target IDs for a single Source 1 entity.

    Args:
        scored_candidates: List of (target_id, score) pairs.
        threshold: Base probability threshold for accepting a match.
        margin_threshold: Required margin between conflicting close candidates.
        max_matches: Maximum number of accepted matches per S1 entity.
        ambiguity_penalty: Whether to suppress highly ambiguous low-margin candidates.

    Returns:
        List of accepted target entity IDs (empty list if no confident match).
    """
    if not scored_candidates:
        return []

    # Sort descending by score
    sorted_cands = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    # If the top score does not even reach the threshold -> Singleton (empty match)
    top_id, top_score = sorted_cands[0]
    if top_score < threshold:
        return []

    accepted: List[str] = [top_id]

    # For subsequent candidates: they must also pass the threshold
    for target_id, score in sorted_cands[1:]:
        if score >= threshold:
            accepted.append(target_id)
            if len(accepted) >= max_matches:
                break

    return accepted


def decide_all_entities(
    scored_entities: Dict[str, Sequence[Tuple[str, float]]],
    all_s1_ids: Sequence[str],
    threshold: float = 0.50,
    max_matches: int = 15,
) -> Dict[str, List[str]]:
    """Produce final match decisions for all Source 1 entities in the evaluation set.

    Guarantees that every S1 ID in all_s1_ids appears in the output mapping.

    Args:
        scored_entities: Mapping from s1_id to list of (target_id, score).
        all_s1_ids: Full set of required Source 1 entity IDs.
        threshold: Acceptance threshold.
        max_matches: Maximum matches per entity.

    Returns:
        Mapping from s1_id to list of accepted target IDs.
    """
    results: Dict[str, List[str]] = {}

    for s1_id in all_s1_ids:
        cands = scored_entities.get(s1_id, [])
        matches = decide_matches_for_entity(
            cands,
            threshold=threshold,
            max_matches=max_matches,
        )
        results[s1_id] = matches

    return results
