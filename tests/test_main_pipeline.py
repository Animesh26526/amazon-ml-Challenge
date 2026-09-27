"""Unit tests for the integrated V0->V5 entity resolution pipeline in main.py."""

from __future__ import annotations

import numpy as np
import pytest

from main import (
    CandidatePair,
    GBDTMatcher,
    MultiPassCandidateGenerator,
    NormalizedRecord,
    clean_unicode,
    compute_per_entity_f05,
    decide_matches_for_entity,
    evaluate_candidate_recall,
    evaluate_predictions,
    extract_pairwise_feature_vector,
    normalize_address,
    normalize_business_name,
)


def test_clean_unicode():
    assert clean_unicode("Café & Co.") == "Cafe & Co."
    assert clean_unicode("Prabhav™") == "PrabhavTM"
    assert clean_unicode("") == ""


def test_normalize_business_name():
    assert normalize_business_name("Acme Corporation") == "acme corp"
    assert normalize_business_name("Nyasa Nursing Private Limited") == "nyasa nursing pvt ltd"
    assert normalize_business_name("Bio-Tech, Inc.") == "bio tech inc"


def test_normalize_address():
    assert normalize_address("123 Main Street, Suite 400") == "123 main st ste 400"
    assert normalize_address("4001 E. Baseline Road") == "4001 e baseline rd"
    assert normalize_address("27 Rue Jean Bart Lille") == "27 r jean bart lille"


def test_normalized_record_creation():
    rec = NormalizedRecord.from_raw(
        entity_id="S1-001",
        name="Urgent Care Clinic, LLC",
        address="309 N Main St, Vici, OK",
        country="US",
    )
    assert rec.entity_id == "S1-001"
    assert rec.source == "S1"
    assert rec.country == "US"
    assert "urgent" in rec.name_tokens
    assert not rec.addr_missing
    assert "309" in rec.addr_numbers


def test_candidate_generation_and_provenance():
    t1 = NormalizedRecord.from_raw("S2-101", "Urgent Care Clinic", "309 N Main St Vici OK", "US")
    t2 = NormalizedRecord.from_raw("S2-102", "Completely Unrelated Inc", "100 Other Road", "US")

    gen = MultiPassCandidateGenerator(max_candidates_per_key=10, max_candidates_per_s1=10)
    gen.build_indexes([t1, t2])

    query = NormalizedRecord.from_raw("S1-001", "Urgent Care Clinic", "309 N Main St Vici OK", "US")
    cands = gen.generate_candidates_for_record(query)

    assert "S2-101" in cands
    pair = cands["S2-101"]
    assert pair.route_exact_name == 1
    assert pair.route_exact_addr == 1
    assert pair.route_count >= 2


def test_feature_vector_dimension():
    s1 = NormalizedRecord.from_raw("S1-001", "Acme Health", "123 Main St", "US")
    tgt = NormalizedRecord.from_raw("S2-001", "Acme Health Group", "123 Main St Ste 1", "US")
    cp = CandidatePair(s1_id="S1-001", target_id="S2-001", route_exact_name=1, route_count=1)

    feats = extract_pairwise_feature_vector(s1, tgt, cp)
    assert len(feats) == 33
    assert all(isinstance(x, (int, float)) for x in feats)


def test_gbdt_matcher_train_and_predict():
    X = np.random.randn(50, 33).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.int32)
    y[0] = 1
    y[1] = 0

    cfg = {"model": {"algorithm": "lightgbm", "n_estimators": 5, "random_state": 42}}
    matcher = GBDTMatcher(cfg)
    matcher.fit(X, y)

    probs = matcher.predict_proba(X)
    assert len(probs) == 50
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_per_entity_f05():
    # Singleton correctly predicted
    p, r, f = compute_per_entity_f05(set(), set())
    assert f == 1.0

    # Singleton predicted false positive
    p, r, f = compute_per_entity_f05({"S2-01"}, set())
    assert f == 0.0

    # True match predicted empty (false negative)
    p, r, f = compute_per_entity_f05(set(), {"S2-01"})
    assert f == 0.0

    # Exact match
    p, r, f = compute_per_entity_f05({"S2-01", "S3-01"}, {"S2-01", "S3-01"})
    assert f == 1.0

    # Partial match: 1 TP, 1 FP, 1 FN
    # True = {A, B}, Pred = {A, C} -> TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5 -> F0.5=0.5
    p, r, f = compute_per_entity_f05({"S2-01", "S3-02"}, {"S2-01", "S3-01"})
    assert p == 0.5 and r == 0.5 and f == 0.5


def test_decide_matches_singleton_policy():
    tgt = NormalizedRecord.from_raw("S2-001", "Some Name", "Some Address", "US")
    # Low score below singleton threshold (0.32)
    scored = [("S2-001", 0.25, tgt)]
    res = decide_matches_for_entity(scored, base_threshold=0.50, singleton_threshold=0.32)
    assert res == []

    # High score above threshold
    scored_high = [("S2-001", 0.85, tgt)]
    res_high = decide_matches_for_entity(scored_high, base_threshold=0.50, singleton_threshold=0.32)
    assert res_high == ["S2-001"]
