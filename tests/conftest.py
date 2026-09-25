"""Pytest Fixtures for Unit Testing.

Provides lightweight synthetic records and temporary sample files.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Ensure project root is always in sys.path when running pytest directly
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from src.normalization import NormalizedRecord



@pytest.fixture
def synthetic_records() -> List[NormalizedRecord]:
    """Return small list of synthetic normalized records."""
    raw_data = [
        ("S1-100", "Acme Corporation", "123 Main Street, Suite 400", "US"),
        ("S1-200", "Global Logistics Ltd", "456 Commerce Road", "US"),
        ("S1-300", "Lone Star Cafe", "789 Elm Boulevard", "US"),
        ("S1-400", "Empty Match Enterprise", "100 Industrial Parkway", "India"),
        ("S2-101", "Acme Corp.", "123 Main St, Ste 400", "US"),
        ("S2-201", "Global Logistics", "456 Commerce Rd", "US"),
        ("S2-999", "Completely Unrelated Inc", "999 Nowhere Lane", "US"),
        ("S3-102", "Acme Corp", "123 Main Street", "US"),
        ("S3-202", "Global Logistics Limited", "", "US"),  # missing address
    ]
    return [
        NormalizedRecord.from_raw(eid, name, addr, country)
        for eid, name, addr, country in raw_data
    ]
