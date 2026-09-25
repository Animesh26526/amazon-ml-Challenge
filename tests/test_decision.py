"""Unit Tests for Entity-Level Decision Engine and Ambiguity Handling."""

from __future__ import annotations

import pytest

from src.decision.ambiguity import compute_top_margin, is_ambiguous
from src.decision.entity_decision import decide_all_entities, decide_matches_for_entity
from src.decision.threshold import apply_source_specific_thresholds


def test_decide_matches_empty_when_below_threshold():
    cands = [("S2-100", 0.45), ("S3-200", 0.30)]
    # Threshold 0.50 -> should return empty list (singleton)
    matches = decide_matches_for_entity(cands, threshold=0.50)
    assert matches == []


def test_decide_matches_multi_match():
    cands = [("S2-100", 0.92), ("S3-200", 0.88), ("S2-300", 0.40)]
    matches = decide_matches_for_entity(cands, threshold=0.50)
    # Both S2-100 and S3-200 pass
    assert matches == ["S2-100", "S3-200"]


def test_decide_matches_max_limit():
    cands = [(f"S2-{i}", 0.90) for i in range(10)]
    matches = decide_matches_for_entity(cands, threshold=0.50, max_matches=3)
    assert len(matches) == 3


def test_decide_all_entities_covers_all_s1():
    scored = {
        "S1-01": [("S2-10", 0.9)],
    }
    all_s1 = ["S1-01", "S1-02", "S1-03"]
    results = decide_all_entities(scored, all_s1, threshold=0.50)

    # Every S1 must be present in the output
    assert set(results.keys()) == set(all_s1)
    assert results["S1-01"] == ["S2-10"]
    assert results["S1-02"] == []
    assert results["S1-03"] == []


def test_ambiguity_detection():
    # Tight scores above threshold -> ambiguous
    tight = [("S2-1", 0.82), ("S2-2", 0.81)]
    assert is_ambiguous(tight, margin_threshold=0.03, score_threshold=0.60)

    # Clear winner -> not ambiguous
    clear = [("S2-1", 0.95), ("S2-2", 0.60)]
    assert not is_ambiguous(clear, margin_threshold=0.03, score_threshold=0.60)
    assert compute_top_margin(clear) == pytest.approx(0.35)
