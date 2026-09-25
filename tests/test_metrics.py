"""Unit Tests for Evaluation Metrics (Macro F_0.5, Singletons, Candidate Recall)."""

from __future__ import annotations

import pytest

from src.evaluation.candidate_recall import compute_candidate_recall
from src.evaluation.diagnostics import evaluate_cohorts
from src.evaluation.metrics import compute_macro_f05, compute_per_entity_f05, evaluate_predictions


def test_official_problem_statement_example():
    """Verify exact formula from problem statement section:

    - Model predicts: [S2-00047, S2-00193, S3-00812]
    - Ground truth: [S2-00047, S3-00812]
    - Precision: 2/3 (0.667), Recall: 1.0
    - F_0.5: (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0) = (5/6)/(7/6) = 5/7 = ~0.714
    """
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    truth = ["S2-00047", "S3-00812"]

    p, r, f05 = compute_per_entity_f05(pred, truth)
    assert abs(p - (2.0 / 3.0)) < 1e-4
    assert abs(r - 1.0) < 1e-4
    assert abs(f05 - (5.0 / 7.0)) < 1e-4
    assert round(f05, 3) == 0.714


def test_singleton_scoring():
    # Correctly predicted empty singleton -> 1.0
    p, r, f05 = compute_per_entity_f05([], [])
    assert f05 == 1.0
    assert p == 1.0
    assert r == 1.0

    # False merge on singleton -> 0.0
    p, r, f05 = compute_per_entity_f05(["S2-999"], [])
    assert f05 == 0.0
    assert p == 0.0
    assert r == 0.0

    # Missed match on true entity -> 0.0
    p, r, f05 = compute_per_entity_f05([], ["S2-100"])
    assert f05 == 0.0


def test_macro_f05():
    gt = {
        "S1-01": ["S2-10", "S3-10"],  # perfect match -> 1.0
        "S1-02": [],                  # perfect singleton -> 1.0
        "S1-03": ["S2-20"],           # missed match -> 0.0
    }
    preds = {
        "S1-01": ["S2-10", "S3-10"],
        "S1-02": [],
        "S1-03": [],
    }

    macro = compute_macro_f05(preds, gt)
    # (1.0 + 1.0 + 0.0) / 3 = 0.6667
    assert abs(macro - (2.0 / 3.0)) < 1e-4


def test_candidate_recall():
    gt = {
        "S1-01": ["S2-10", "S3-10"],
        "S1-02": ["S2-20"],
    }
    candidates = {
        "S1-01": ["S2-10", "S2-99"],  # 1 of 2 covered
        "S1-02": ["S2-20"],           # 1 of 1 covered
    }
    res = compute_candidate_recall(candidates, gt)
    # Covered 2 out of 3 = 66.67%
    assert abs(res["candidate_recall"] - (2.0 / 3.0)) < 1e-4
    assert res["total_candidate_pairs"] == 3.0
