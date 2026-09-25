"""Unit Tests for I/O and Submission Invariant Verification."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.io import (
    CANDIDATE_PAIRS_COLUMNS,
    MATCHING_RESULTS_COLUMNS,
    format_id_list,
    load_config,
    parse_id_list,
    verify_submission_invariants,
    write_candidate_pairs,
    write_matching_results,
)


def test_parse_and_format_id_list():
    raw_str = "S2-001, S3-002, S2-001, S2-003"
    parsed = parse_id_list(raw_str)
    assert parsed == ["S2-001", "S3-002", "S2-001", "S2-003"]

    formatted = format_id_list(parsed)
    # Deduplication while preserving order
    assert formatted == "S2-001,S3-002,S2-003"

    assert parse_id_list(None) == []
    assert parse_id_list("") == []
    assert format_id_list([]) == ""


def test_write_and_verify_submission_invariants():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        matching_file = tmp_path / "matching_results.tsv"
        candidate_file = tmp_path / "candidate_pairs.tsv"

        matching_data = {
            "S1-001": ["S2-101", "S3-102"],
            "S1-002": ["S2-201"],
            "S1-003": [],  # singleton
        }
        candidate_data = {
            "S1-001": ["S2-101", "S3-102", "S2-999"],
            "S1-002": ["S2-201", "S3-202"],
            "S1-003": ["S2-999"],
        }

        write_matching_results(matching_data, matching_file)
        write_candidate_pairs(candidate_data, candidate_file)

        assert matching_file.is_file()
        assert candidate_file.is_file()

        # Invariant check must pass
        valid, errors = verify_submission_invariants(matching_file, candidate_file)
        assert valid, f"Invariant verification failed: {errors}"
        assert len(errors) == 0


def test_verify_submission_invariants_catches_violations():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        matching_file = tmp_path / "matching_results.tsv"
        candidate_file = tmp_path / "candidate_pairs.tsv"

        # Violation 1: matched ID (S2-101) not in candidate set
        # Violation 2: self-match (S1-002) in matched IDs
        matching_data = {
            "S1-001": ["S2-101"],
            "S1-002": ["S1-002"],  # Self match!
        }
        candidate_data = {
            "S1-001": ["S2-999"],  # Missing S2-101!
            "S1-002": ["S2-201"],
        }

        write_matching_results(matching_data, matching_file)
        write_candidate_pairs(candidate_data, candidate_file)

        valid, errors = verify_submission_invariants(matching_file, candidate_file)
        assert not valid
        assert any("Candidate invariant violated" in err for err in errors)
        assert any("Self-match detected" in err for err in errors)
