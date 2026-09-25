"""Text Normalization and Multi-View Record Representation.

Implements conservative text transformations, legal entity suffix handling,
address abbreviation standardizations, tokenizations, and numeric anchor extractions.

PRD Guideline:
Multi-view representations coexist; raw values must never be destroyed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Canonical mapping for common legal business suffixes
LEGAL_SUFFIX_MAP: Dict[str, str] = {
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "limited": "ltd",
    "private": "pvt",
    "llc": "llc",
    "llp": "llp",
    "l.l.c.": "llc",
    "l.l.p.": "llp",
    "corp.": "corp",
    "inc.": "inc",
    "co.": "co",
    "ltd.": "ltd",
    "pvt.": "pvt",
    "sarl": "sarl",
    "s.a.r.l.": "sarl",
    "sas": "sas",
    "s.a.s.": "sas",
    "sa": "sa",
    "s.a.": "sa",
    "eurl": "eurl",
    "sci": "sci",
    "snc": "snc",
}

# Canonical mapping for common address component abbreviations
ADDRESS_ABBREV_MAP: Dict[str, str] = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "floor": "fl",
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "rue": "r",
    "chemin": "chem",
}


# Regex pre-compilations
RE_NON_ALPHANUM_SPACE = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\b\d+\b")


def clean_unicode(text: str) -> str:
    """Normalize Unicode characters using NFKD decomposition."""
    if not text:
        return ""
    # Normalize unicode and strip non-ASCII combining marks if applicable
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim edges."""
    if not text:
        return ""
    return RE_WHITESPACE.sub(" ", text).strip()


def normalize_business_name(
    name: Optional[str],
    normalize_suffixes: bool = True,
) -> str:
    """Conservatively normalize a business name string.

    Preserves essential tokens while removing extraneous punctuation and
    standardizing common legal suffixes.

    Args:
        name: Raw business name.
        normalize_suffixes: Whether to map legal suffixes to standard tokens.

    Returns:
        Normalized business name string.
    """
    if not name or not isinstance(name, str):
        return ""

    text = clean_unicode(name).lower()
    text = RE_NON_ALPHANUM_SPACE.sub(" ", text)
    tokens = text.split()

    if normalize_suffixes and tokens:
        normalized_tokens = [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in tokens]
        tokens = normalized_tokens

    return normalize_whitespace(" ".join(tokens))


def normalize_address(
    address: Optional[str],
    normalize_abbrevs: bool = True,
) -> str:
    """Conservatively normalize a business address string.

    Args:
        address: Raw address string.
        normalize_abbrevs: Whether to normalize common address abbreviations.

    Returns:
        Normalized address string.
    """
    if not address or not isinstance(address, str):
        return ""

    text = clean_unicode(address).lower()
    text = RE_NON_ALPHANUM_SPACE.sub(" ", text)
    tokens = text.split()

    if normalize_abbrevs and tokens:
        tokens = [ADDRESS_ABBREV_MAP.get(tok, tok) for tok in tokens]

    return normalize_whitespace(" ".join(tokens))


def extract_numeric_anchors(text: Optional[str]) -> List[str]:
    """Extract numeric sequences from text (e.g., street numbers, postal codes).

    Args:
        text: Input string.

    Returns:
        List of numeric tokens found in order of appearance.
    """
    if not text or not isinstance(text, str):
        return []
    return RE_DIGITS.findall(text)


def extract_char_ngrams(text: Optional[str], n: int = 3) -> Set[str]:
    """Extract character n-grams from normalized text with word boundary padding.

    Args:
        text: Input string.
        n: N-gram length.

    Returns:
        Set of character n-grams.
    """
    if not text or not isinstance(text, str):
        return set()
    cleaned = f" {normalize_whitespace(text.lower())} "
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    """Multi-view representation of a business record.

    Keeps the raw fields completely intact while attaching computed views.
    """

    entity_id: str
    business_name_raw: str
    business_address_raw: str
    country: str

    business_name_norm: str = field(default="")
    business_address_norm: str = field(default="")
    name_tokens: List[str] = field(default_factory=list)
    address_tokens: List[str] = field(default_factory=list)
    numeric_anchors: List[str] = field(default_factory=list)
    is_address_missing: bool = field(default=False)

    @classmethod
    def from_raw(
        cls,
        entity_id: str,
        business_name: Optional[str],
        business_address: Optional[str],
        country: Optional[str],
        store_raw: bool = True,
    ) -> NormalizedRecord:
        """Construct multi-view record from raw input attributes."""
        raw_name = business_name or ""
        raw_addr = business_address or ""
        raw_country = country or ""

        norm_name = normalize_business_name(raw_name)
        norm_addr = normalize_address(raw_addr)
        name_tokens = norm_name.split() if norm_name else []
        addr_tokens = norm_addr.split() if norm_addr else []
        num_anchors = extract_numeric_anchors(raw_addr)
        is_missing = not bool(norm_addr.strip())

        return cls(
            entity_id=entity_id.strip(),
            business_name_raw=raw_name if store_raw else "",
            business_address_raw=raw_addr if store_raw else "",
            country=raw_country.strip(),
            business_name_norm=norm_name,
            business_address_norm=norm_addr,
            name_tokens=name_tokens,
            address_tokens=addr_tokens,
            numeric_anchors=num_anchors,
            is_address_missing=is_missing,
        )
