"""Exact Matching Blocking Routes.

Blocks candidate pairs on exact normalized name, exact normalized address,
or the composite (name + address) key.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set

from src.blocking import BaseBlockingRule
from src.normalization import NormalizedRecord


class ExactNameBlocking(BaseBlockingRule):
    """Retrieves candidates sharing the exact normalized business name."""

    name: str = "exact_name"

    def __init__(self, country_partition: bool = False, max_candidates_per_key: int = 50) -> None:
        self.country_partition = country_partition
        self.max_candidates_per_key = max_candidates_per_key
        self._index: Dict[str, List[str]] = defaultdict(list)

    def _make_key(self, name_norm: str, country: str) -> str:
        if self.country_partition and country:
            return f"{country}::{name_norm}"
        return name_norm

    def build_index(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Index target records by exact normalized name."""
        self._index.clear()
        for rec in target_records:
            if rec.business_name_norm:
                key = self._make_key(rec.business_name_norm, rec.country)
                if len(self._index[key]) < self.max_candidates_per_key:
                    self._index[key].append(rec.entity_id)

    def query(self, query_record: NormalizedRecord) -> List[str]:
        """Query matching target IDs for the given query record."""
        if not query_record.business_name_norm:
            return []
        key = self._make_key(query_record.business_name_norm, query_record.country)
        return self._index.get(key, [])



class ExactAddressBlocking(BaseBlockingRule):
    """Retrieves candidates sharing the exact normalized address."""

    name: str = "exact_address"

    def __init__(self, country_partition: bool = False, max_candidates_per_key: int = 50) -> None:
        self.country_partition = country_partition
        self.max_candidates_per_key = max_candidates_per_key
        self._index: Dict[str, List[str]] = defaultdict(list)

    def _make_key(self, address_norm: str, country: str) -> str:
        if self.country_partition and country:
            return f"{country}::{address_norm}"
        return address_norm

    def build_index(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Index target records by exact normalized address."""
        self._index.clear()
        for rec in target_records:
            if rec.business_address_norm:
                key = self._make_key(rec.business_address_norm, rec.country)
                if len(self._index[key]) < self.max_candidates_per_key:
                    self._index[key].append(rec.entity_id)

    def query(self, query_record: NormalizedRecord) -> List[str]:
        """Query matching target IDs for the given query record."""
        if not query_record.business_address_norm:
            return []
        key = self._make_key(query_record.business_address_norm, query_record.country)
        return self._index.get(key, [])


class ExactNameAndAddressBlocking(BaseBlockingRule):
    """Retrieves candidates matching both exact normalized name and address."""

    name: str = "exact_name_address"

    def __init__(self, max_candidates_per_key: int = 50) -> None:
        self.max_candidates_per_key = max_candidates_per_key
        self._index: Dict[str, List[str]] = defaultdict(list)

    def build_index(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Index target records by joint (name, address) key."""
        self._index.clear()
        for rec in target_records:
            if rec.business_name_norm and rec.business_address_norm:
                key = f"{rec.business_name_norm}__##__{rec.business_address_norm}"
                if len(self._index[key]) < self.max_candidates_per_key:
                    self._index[key].append(rec.entity_id)

    def query(self, query_record: NormalizedRecord) -> List[str]:
        """Query matching target IDs."""
        if not query_record.business_name_norm or not query_record.business_address_norm:
            return []
        key = f"{query_record.business_name_norm}__##__{query_record.business_address_norm}"
        return self._index.get(key, [])

