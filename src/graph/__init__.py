"""Local S1-Centered Graph and Cross-Source Evidence Module.

Provides witness verification, triangle consistency checks, and
Support / Neutral / Contradiction classification for candidate pairs.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class CrossSourceEvidenceType(Enum):
    """Categorization of cross-source relationship evidence."""
    SUPPORT = "support"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"
