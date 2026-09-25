"""Unit Tests for Candidate Blocking Routes and CandidateSet Data Structure."""

from __future__ import annotations

import pytest

from src.blocking import CandidatePair, CandidateSet
from src.blocking.char_ngram import CharNgramBlocking
from src.blocking.exact import ExactAddressBlocking, ExactNameAndAddressBlocking, ExactNameBlocking
from src.blocking.phonetic import PhoneticBlocking, soundex
from src.blocking.tfidf import TfidfBlocking
from src.normalization import NormalizedRecord


def test_candidate_set_operations():
    cand_set = CandidateSet()
    cand_set.add("S1-001", "S2-100", route_name="exact_name", score=1.0)
    cand_set.add("S1-001", "S2-100", route_name="tfidf", score=0.85)
    cand_set.add("S1-001", "S3-200", route_name="exact_address", score=1.0)

    # Verify routes combined
    pair = cand_set.get_candidates_for("S1-001")["S2-100"]
    assert "exact_name" in pair.routes
    assert "tfidf" in pair.routes
    assert pair.route_scores["tfidf"] == 0.85

    # Verify target IDs
    assert set(cand_set.get_target_ids("S1-001")) == {"S2-100", "S3-200"}
    assert cand_set.total_pairs() == 2


def test_exact_blocking(synthetic_records):
    s1_records = [r for r in synthetic_records if r.entity_id.startswith("S1-")]
    target_records = [r for r in synthetic_records if not r.entity_id.startswith("S1-")]

    # Exact name
    name_blocker = ExactNameBlocking()
    name_blocker.build_index(target_records)

    # "S1-100": "Acme Corporation" -> norm is "acme corp"
    # Target "S2-101" norm is "acme corp", "S3-102" norm is "acme corp"
    s1_acme = next(r for r in s1_records if r.entity_id == "S1-100")
    matches = name_blocker.query(s1_acme)
    assert "S2-101" in matches
    assert "S3-102" in matches
    assert "S2-999" not in matches


def test_char_ngram_blocking(synthetic_records):
    target_records = [r for r in synthetic_records if not r.entity_id.startswith("S1-")]
    s1_records = [r for r in synthetic_records if r.entity_id.startswith("S1-")]

    blocker = CharNgramBlocking(n=3, min_overlap=3)
    blocker.build_index(target_records)

    s1_acme = next(r for r in s1_records if r.entity_id == "S1-100")
    results = blocker.query(s1_acme)
    assert "S2-101" in results or "S3-102" in results


def test_tfidf_blocking(synthetic_records):
    target_records = [r for r in synthetic_records if not r.entity_id.startswith("S1-")]
    s1_records = [r for r in synthetic_records if r.entity_id.startswith("S1-")]

    blocker = TfidfBlocking(top_k=5, threshold=0.2)
    blocker.build_index(target_records)

    s1_acme = next(r for r in s1_records if r.entity_id == "S1-100")
    results = blocker.query(s1_acme)
    assert len(results) > 0
    assert any(eid.startswith(("S2-", "S3-")) for eid in results)


def test_soundex():
    assert soundex("Robert") == "R163"
    assert soundex("Rupert") == "R163"
    assert soundex("Smith") == "S530"
    assert soundex("Smythe") == "S530"
