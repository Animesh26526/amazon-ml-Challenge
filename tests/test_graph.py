"""Unit Tests for Local S1-Centered Cross-Source Evidence and Graph Modules."""

from __future__ import annotations

import pytest

from src.graph import CrossSourceEvidenceType
from src.graph.contradiction import classify_relationship
from src.graph.triangle import compute_triangle_support
from src.graph.witnesses import compute_witness_score, find_strongest_witness
from src.normalization import NormalizedRecord


def test_witness_scoring():
    rec_s2 = NormalizedRecord.from_raw("S2-01", "Acme Corporation", "123 Main St", "US")
    rec_s3_match = NormalizedRecord.from_raw("S3-01", "Acme Corp", "123 Main Street", "US")
    rec_s3_diff = NormalizedRecord.from_raw("S3-02", "Unrelated Business", "999 Other St", "US")

    score_match = compute_witness_score(rec_s2, rec_s3_match)
    assert score_match > 0.8

    score_diff = compute_witness_score(rec_s2, rec_s3_diff)
    assert score_diff < 0.3


def test_triangle_support():
    tri = compute_triangle_support(
        s1_s2_score=0.90,
        s1_s3_score=0.85,
        s2_s3_score=0.88,
        threshold=0.70,
    )
    assert tri["triangle_closed"] == 1.0
    assert tri["triangle_min_edge"] == 0.85

    tri_broken = compute_triangle_support(
        s1_s2_score=0.90,
        s1_s3_score=0.30,
        s2_s3_score=0.20,
        threshold=0.70,
    )
    assert tri_broken["triangle_closed"] == 0.0


def test_contradiction_vs_neutral_vs_support():
    r1 = NormalizedRecord.from_raw("S1-01", "Acme Corp", "123 Main St", "US")
    r_support = NormalizedRecord.from_raw("S2-01", "Acme Corp", "123 Main St", "US")
    r_diff_country = NormalizedRecord.from_raw("S2-02", "Acme Corp", "123 Main St", "India")
    r_neutral_addr = NormalizedRecord.from_raw("S2-03", "Acme Corp", None, "US")  # missing address

    # High similarity, same country -> SUPPORT
    assert classify_relationship(r1, r_support) == CrossSourceEvidenceType.SUPPORT

    # Conflicting country -> CONTRADICTION
    assert classify_relationship(r1, r_diff_country) == CrossSourceEvidenceType.CONTRADICTION

    # Missing address with moderate match -> NEUTRAL
    r_moderate = NormalizedRecord.from_raw("S2-04", "Acme International", None, "US")
    assert classify_relationship(r1, r_moderate) == CrossSourceEvidenceType.NEUTRAL
