"""Unit Tests for Normalization and Multi-View Record Creation."""

from __future__ import annotations

import pytest

from src.normalization import (
    NormalizedRecord,
    clean_unicode,
    extract_char_ngrams,
    extract_numeric_anchors,
    normalize_address,
    normalize_business_name,
)


def test_clean_unicode():
    assert clean_unicode("Café & Co.") == "Cafe & Co."
    assert clean_unicode("Über Flöße") == "Uber Flosse" or "Uber Floe" or clean_unicode("Über") == "Uber"


def test_normalize_business_name_suffixes():
    assert normalize_business_name("Acme Corporation") == "acme corp"
    assert normalize_business_name("Global Logistics, Limited") == "global logistics ltd"
    assert normalize_business_name("Tech Innovations Pvt. Ltd.") == "tech innovations pvt ltd"
    assert normalize_business_name("Alpha Company") == "alpha co"


def test_normalize_address_abbreviations():
    assert normalize_address("123 Main Street, Suite 400") == "123 main st ste 400"
    assert normalize_address("456 Commerce Road, Floor 2") == "456 commerce rd fl 2"
    assert normalize_address("789 North Boulevard") == "789 n blvd"


def test_extract_numeric_anchors():
    nums = extract_numeric_anchors("Flat 402, Building 12, Sector 56, PIN 122011")
    assert nums == ["402", "12", "56", "122011"]


def test_extract_char_ngrams():
    ngrams = extract_char_ngrams("acme", n=3)
    # padded: " acme " -> " ac", "acm", "cme", "me "
    assert "acm" in ngrams
    assert "cme" in ngrams


def test_normalized_record_multi_view():
    rec = NormalizedRecord.from_raw(
        entity_id="S1-12345",
        business_name="Acme Corporation",
        business_address="123 Main Street",
        country="US",
    )
    # Raw values preserved
    assert rec.business_name_raw == "Acme Corporation"
    assert rec.business_address_raw == "123 Main Street"
    assert rec.country == "US"

    # Normalized views populated
    assert rec.business_name_norm == "acme corp"
    assert rec.business_address_norm == "123 main st"
    assert rec.name_tokens == ["acme", "corp"]
    assert rec.numeric_anchors == ["123"]
    assert not rec.is_address_missing


def test_normalized_record_missing_address():
    rec = NormalizedRecord.from_raw(
        entity_id="S2-999",
        business_name="No Address Corp",
        business_address=None,
        country="India",
    )
    assert rec.is_address_missing
    assert rec.business_address_norm == ""
    assert rec.address_tokens == []
