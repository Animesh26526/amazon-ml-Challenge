"""Unit Tests for Pairwise Feature Calculations and Missingness Handling."""

from __future__ import annotations

import pytest

from src.blocking import CandidatePair
from src.features.address import compute_address_features
from src.features.interactions import compute_interaction_features
from src.features.name import compute_name_features
from src.features.retrieval import compute_retrieval_features
from src.normalization import NormalizedRecord


def test_name_features():
    rec1 = NormalizedRecord.from_raw("S1-1", "Acme Corporation", "123 Main St", "US")
    rec2 = NormalizedRecord.from_raw("S2-1", "Acme Corp.", "123 Main St", "US")
    rec_diff = NormalizedRecord.from_raw("S2-2", "Global Logistics", "999 Other St", "US")

    feats_match = compute_name_features(rec1, rec2)
    assert feats_match["name_exact"] == 1.0
    assert feats_match["name_token_jaccard"] == 1.0
    assert feats_match["name_edit_similarity"] == 1.0

    feats_diff = compute_name_features(rec1, rec_diff)
    assert feats_diff["name_exact"] == 0.0
    assert feats_diff["name_token_jaccard"] == 0.0
    assert feats_diff["name_edit_similarity"] < 0.3


def test_address_features_missing_handling():
    rec1 = NormalizedRecord.from_raw("S1-1", "Acme Corp", "123 Main Street", "US")
    rec_missing = NormalizedRecord.from_raw("S2-1", "Acme Corp", None, "US")

    feats = compute_address_features(rec1, rec_missing)
    # Missing address must be explicitly flagged
    assert feats["addr_missing_target"] == 1.0
    assert feats["addr_missing_either"] == 1.0
    assert feats["addr_missing_s1"] == 0.0
    # Similarity should be neutral (0.0), not falsely indicating conflict
    assert feats["addr_exact"] == 0.0
    assert feats["addr_token_jaccard"] == 0.0


def test_interaction_features():
    rec1 = NormalizedRecord.from_raw("S1-1", "Acme Corp", "123 Main Street", "US")
    rec2 = NormalizedRecord.from_raw("S2-1", "Acme Corp", "123 Main St", "US")

    nf = compute_name_features(rec1, rec2)
    af = compute_address_features(rec1, rec2)
    inter = compute_interaction_features(nf, af)

    assert inter["interaction_both_strong"] == 1.0
    assert inter["interaction_product"] > 0.8
    assert inter["interaction_name_strong_addr_weak"] == 0.0


def test_retrieval_features():
    pair = CandidatePair(source1_id="S1-1", target_id="S2-1")
    pair.add_route("exact_name", score=1.0)
    pair.add_route("tfidf", score=0.88)

    feats = compute_retrieval_features(pair)
    assert feats["route_exact_name"] == 1.0
    assert feats["route_tfidf"] == 1.0
    assert feats["route_phonetic"] == 0.0
    assert feats["num_routes_retrieved"] == 2.0
    assert feats["max_route_score"] == 1.0
