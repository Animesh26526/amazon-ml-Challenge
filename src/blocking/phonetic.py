"""Phonetic Blocking Route.

Retrieves candidates sharing phonetic encoding (Soundex) on the primary business name tokens.
Useful for phonetic and transliteration variations without external dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from src.blocking import BaseBlockingRule
from src.normalization import NormalizedRecord


def soundex(token: str) -> str:
    """Compute American Soundex code for a single word token."""
    if not token or not token.isalpha():
        return ""

    token = token.upper()
    first_letter = token[0]

    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    digits: List[str] = []
    prev_code = mapping.get(first_letter, "")

    for char in token[1:]:
        code = mapping.get(char, "")
        if code != prev_code:
            if code:
                digits.append(code)
            prev_code = code

    code_str = first_letter + "".join(digits)
    return (code_str + "000")[:4]


class PhoneticBlocking(BaseBlockingRule):
    """Indexes records by Soundex code of the primary business name token."""

    name: str = "phonetic"

    def __init__(self, max_candidates_per_key: int = 200) -> None:
        self.max_candidates_per_key = max_candidates_per_key
        self._index: Dict[str, List[str]] = defaultdict(list)

    def _get_key(self, record: NormalizedRecord) -> str:
        if not record.name_tokens:
            return ""
        # Use first non-trivial token
        for tok in record.name_tokens:
            if len(tok) >= 3:
                return soundex(tok)
        return soundex(record.name_tokens[0])

    def build_index(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Build inverted index on phonetic Soundex code."""
        self._index.clear()
        for rec in target_records:
            key = self._get_key(rec)
            if key and len(self._index[key]) < self.max_candidates_per_key:
                self._index[key].append(rec.entity_id)

    def query(self, query_record: NormalizedRecord) -> List[str]:
        """Query candidates sharing the phonetic Soundex key."""
        key = self._get_key(query_record)
        if not key:
            return []
        return self._index.get(key, [])
