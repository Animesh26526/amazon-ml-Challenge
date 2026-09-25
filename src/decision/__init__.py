"""Entity-Level Decision Engine Interfaces.

Provides post-pairwise decision logic, ambiguity resolution, and singleton handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ScoredMatch:
    """Represents a scored candidate match."""
    target_id: str
    score: float
    is_accepted: bool = False
