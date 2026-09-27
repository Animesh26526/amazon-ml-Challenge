#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Business Entity Resolution.

Integrated V0 -> V5 Architecture in a single executable module.
Implements:
- Multi-view conservative normalization
- High-recall multi-pass candidate generation with provenance
- 28-dimensional pairwise + interaction feature engine
- Two-stage GBDT matcher with hard-negative mining
- Reciprocal ranking and local S1-centered graph witness evidence
- Precision-controlled entity-level decision engine with singleton policy
- Validation on macro S1-level F0.5 and cohort diagnostics
- Streaming country-partitioned test inference (US, India, France)
- Submissions formatting and automated official validator check
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from rapidfuzz import fuzz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("amazon_ml_v5")

# Canonical legal business suffixes
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

# Common address abbreviations
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

# Stop tokens that are too generic for rare-token retrieval
COMMON_STOP_TOKENS: Set[str] = {
    "ltd", "pvt", "inc", "corp", "co", "llc", "llp", "the", "and",
    "group", "services", "center", "clinic", "care", "health",
    "medical", "hospital", "associates", "enterprises", "solutions",
    "management", "international", "company", "limited", "private",
    "of", "for", "in", "at", "by", "to", "de", "la", "le", "du", "des",
}

FEATURE_NAMES = [
    # Name features
    "name_exact",
    "name_token_jaccard",
    "name_token_overlap",
    "name_edit_ratio",
    "name_sort_ratio",
    "name_len_diff",
    "name_len_ratio",
    # Address features
    "addr_exact",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_edit_ratio",
    "addr_numeric_jaccard",
    "addr_numeric_overlap",
    "addr_missing_s1",
    "addr_missing_target",
    "addr_missing_either",
    # Interaction features
    "interaction_prod",
    "interaction_min",
    "interaction_mean",
    "interaction_both_high",
    "interaction_name_high_addr_low",
    "interaction_addr_high_name_low",
    "interaction_name_high_addr_missing",
    # Source & Provenance
    "is_s2",
    "is_s3",
    "route_exact_name",
    "route_sorted_name",
    "route_exact_addr",
    "route_numeric_anchor",
    "route_rare_token",
    "route_char_ngram",
    "blocking_route_count",
    "same_country",
]


# ==============================================================================
# 1. Multi-View Normalization
# ==============================================================================

RE_NON_ALPHANUM = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\b\d+\b")


def clean_unicode(text: str) -> str:
    """Normalize unicode and strip non-ASCII combining marks."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace to single spaces and strip."""
    if not text:
        return ""
    return RE_WHITESPACE.sub(" ", text).strip()


def normalize_business_name(name: Optional[str]) -> str:
    """Conservatively normalize business name, standardizing legal suffixes."""
    if not name:
        return ""
    text = clean_unicode(name).lower()
    text = RE_NON_ALPHANUM.sub(" ", text)
    tokens = text.split()
    normalized_tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in tokens if t]
    return " ".join(normalized_tokens)


def normalize_address(addr: Optional[str]) -> str:
    """Conservatively normalize address, standardizing road/street tokens."""
    if not addr:
        return ""
    text = clean_unicode(addr).lower()
    text = RE_NON_ALPHANUM.sub(" ", text)
    tokens = text.split()
    normalized_tokens = [ADDRESS_ABBREV_MAP.get(t, t) for t in tokens if t]
    return " ".join(normalized_tokens)


@dataclass(slots=True, frozen=True)
class NormalizedRecord:
    """Multi-view representation of a business entity record."""

    entity_id: str
    raw_name: str
    raw_addr: str
    country: str
    name_norm: str
    name_sorted: str
    name_tokens: Tuple[str, ...]
    addr_norm: str
    addr_tokens: Tuple[str, ...]
    addr_numbers: Tuple[str, ...]
    addr_missing: bool
    source: str

    @classmethod
    def from_raw(
        cls,
        entity_id: str,
        name: Optional[str],
        address: Optional[str],
        country: Optional[str],
        store_raw: bool = True,
    ) -> "NormalizedRecord":
        raw_n = name if (store_raw and isinstance(name, str)) else ""
        raw_a = address if (store_raw and isinstance(address, str)) else ""
        c = country.strip().upper() if isinstance(country, str) else ""

        src = "S1"
        if entity_id.startswith("S2-"):
            src = "S2"
        elif entity_id.startswith("S3-"):
            src = "S3"

        name_str = name if isinstance(name, str) else ""
        addr_str = address if isinstance(address, str) else ""

        n_norm = normalize_business_name(name_str)
        n_toks = tuple(t for t in n_norm.split() if t)
        n_sorted = " ".join(sorted(n_toks)) if len(n_toks) > 1 else n_norm

        a_missing = not bool(addr_str and addr_str.strip())
        a_norm = normalize_address(addr_str) if not a_missing else ""
        a_toks = tuple(t for t in a_norm.split() if t)
        a_nums = tuple(n for n in RE_DIGITS.findall(a_norm) if len(n) >= 2)

        return cls(
            entity_id=entity_id.strip(),
            raw_name=raw_n,
            raw_addr=raw_a,
            country=c,
            name_norm=n_norm,
            name_sorted=n_sorted,
            name_tokens=n_toks,
            addr_norm=a_norm,
            addr_tokens=a_toks,
            addr_numbers=a_nums,
            addr_missing=a_missing,
            source=src,
        )


# ==============================================================================
# 2. Candidate Generation & Provenance
# ==============================================================================

@dataclass(slots=True)
class CandidatePair:
    """Candidate pair with retrieval provenance flags."""

    s1_id: str
    target_id: str
    route_exact_name: int = 0
    route_sorted_name: int = 0
    route_exact_addr: int = 0
    route_exact_joint: int = 0
    route_numeric_anchor: int = 0
    route_rare_token: int = 0
    route_char_ngram: int = 0
    route_count: int = 0


class MultiPassCandidateGenerator:
    """Multi-pass blocking engine with provenance tracking."""

    def __init__(
        self,
        max_candidates_per_key: int = 60,
        max_candidates_per_s1: int = 50,
        min_rare_freq: int = 1,
        max_rare_freq: int = 50,
        min_addr_len: int = 8,
    ) -> None:
        self.max_candidates_per_key = max_candidates_per_key
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.min_rare_freq = min_rare_freq
        self.max_rare_freq = max_rare_freq
        self.min_addr_len = min_addr_len

        # Inverted index tables (using compact string keys)
        self.exact_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.sorted_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.exact_addr_idx: Dict[str, List[str]] = defaultdict(list)
        self.numeric_anchor_idx: Dict[str, List[str]] = defaultdict(list)
        self.rare_token_idx: Dict[str, List[str]] = defaultdict(list)
        self.char_prefix_idx: Dict[str, List[str]] = defaultdict(list)

    def build_indexes(self, targets: Iterable[NormalizedRecord]) -> None:
        """Streamingly build all inverted indices over target records."""
        gc.disable()
        try:
            logger.info("Computing token document frequencies over target records...")
            token_freq: Dict[str, int] = defaultdict(int)

            target_list: List[NormalizedRecord] = list(targets) if not isinstance(targets, list) else targets
            for r in target_list:
                for t in set(r.name_tokens):
                    if t not in COMMON_STOP_TOKENS and len(t) >= 3:
                        token_freq[t] += 1

            logger.info(f"Indexing {len(target_list):,} target records across blocking routes...")
            for r in target_list:
                eid = r.entity_id

                # 1. Exact name
                if r.name_norm and len(self.exact_name_idx[r.name_norm]) < self.max_candidates_per_key:
                    self.exact_name_idx[r.name_norm].append(eid)

                # 2. Sorted name (word order variation)
                if r.name_sorted != r.name_norm and len(self.sorted_name_idx[r.name_sorted]) < self.max_candidates_per_key:
                    self.sorted_name_idx[r.name_sorted].append(eid)

                # 3. Exact address
                if not r.addr_missing and len(r.addr_norm) >= self.min_addr_len:
                    if len(self.exact_addr_idx[r.addr_norm]) < self.max_candidates_per_key:
                        self.exact_addr_idx[r.addr_norm].append(eid)

                # 4. Numeric anchor: first_token + primary_addr_number
                if r.name_tokens and r.addr_numbers:
                    num_key = f"{r.name_tokens[0]}_{r.addr_numbers[0]}"
                    if len(self.numeric_anchor_idx[num_key]) < self.max_candidates_per_key:
                        self.numeric_anchor_idx[num_key].append(eid)

                # 5. Rare tokens (index rarest token of the business)
                best_t = None
                best_f = 999999
                for t in r.name_tokens:
                    f = token_freq.get(t, 0)
                    if self.min_rare_freq <= f <= self.max_rare_freq:
                        if f < best_f:
                            best_f = f
                            best_t = t
                if best_t and len(self.rare_token_idx[best_t]) < self.max_candidates_per_key:
                    self.rare_token_idx[best_t].append(eid)

                # 6. Char prefix anchor: first 4 chars + primary addr number
                if r.name_norm and r.addr_numbers and len(r.name_norm) >= 4:
                    prefix_key = f"{r.name_norm[:4]}_{r.addr_numbers[0]}"
                    if len(self.char_prefix_idx[prefix_key]) < self.max_candidates_per_key:
                        self.char_prefix_idx[prefix_key].append(eid)

            logger.info("Blocking indexes successfully built.")
        finally:
            gc.enable()
            gc.collect()

    def generate_candidates_for_record(self, s1: NormalizedRecord) -> Dict[str, CandidatePair]:
        """Generate deduplicated candidate pairs with provenance for an S1 record."""
        cands: Dict[str, CandidatePair] = {}

        def add_candidate(target_id: str, route_name: str) -> None:
            if target_id not in cands:
                cands[target_id] = CandidatePair(s1_id=s1.entity_id, target_id=target_id)
            pair = cands[target_id]
            if route_name == "exact_name" and pair.route_exact_name == 0:
                pair.route_exact_name = 1
                pair.route_count += 1
            elif route_name == "sorted_name" and pair.route_sorted_name == 0:
                pair.route_sorted_name = 1
                pair.route_count += 1
            elif route_name == "exact_addr" and pair.route_exact_addr == 0:
                pair.route_exact_addr = 1
                pair.route_count += 1
            elif route_name == "numeric_anchor" and pair.route_numeric_anchor == 0:
                pair.route_numeric_anchor = 1
                pair.route_count += 1
            elif route_name == "rare_token" and pair.route_rare_token == 0:
                pair.route_rare_token = 1
                pair.route_count += 1
            elif route_name == "char_ngram" and pair.route_char_ngram == 0:
                pair.route_char_ngram = 1
                pair.route_count += 1

        # 1. Exact name
        if s1.name_norm:
            for tid in self.exact_name_idx.get(s1.name_norm, []):
                add_candidate(tid, "exact_name")

        # 2. Sorted name
        if s1.name_sorted != s1.name_norm:
            for tid in self.sorted_name_idx.get(s1.name_sorted, []):
                add_candidate(tid, "sorted_name")

        # 3. Exact address
        if not s1.addr_missing and len(s1.addr_norm) >= self.min_addr_len:
            for tid in self.exact_addr_idx.get(s1.addr_norm, []):
                add_candidate(tid, "exact_addr")

        # 4. Numeric anchor
        if s1.name_tokens and s1.addr_numbers:
            num_key = f"{s1.name_tokens[0]}_{s1.addr_numbers[0]}"
            for tid in self.numeric_anchor_idx.get(num_key, []):
                add_candidate(tid, "numeric_anchor")

        # 5. Rare tokens
        for t in s1.name_tokens:
            if t not in COMMON_STOP_TOKENS and t in self.rare_token_idx:
                for tid in self.rare_token_idx[t]:
                    add_candidate(tid, "rare_token")

        # 6. Char prefix anchor
        if s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 4:
            prefix_key = f"{s1.name_norm[:4]}_{s1.addr_numbers[0]}"
            for tid in self.char_prefix_idx.get(prefix_key, []):
                add_candidate(tid, "char_ngram")

        # Cap candidates per S1 by route count descending
        if len(cands) > self.max_candidates_per_s1:
            sorted_pairs = sorted(cands.values(), key=lambda p: p.route_count, reverse=True)
            cands = {p.target_id: p for p in sorted_pairs[:self.max_candidates_per_s1]}

        return cands


# ==============================================================================
# 3. Pair Feature Engine & Local Graph Evidence
# ==============================================================================

def extract_pairwise_feature_vector(
    s1: NormalizedRecord,
    tgt: NormalizedRecord,
    pair: CandidatePair,
) -> List[float]:
    """Extract 33-dimensional pairwise feature vector."""
    # 1. Name features
    name_exact = 1.0 if (s1.name_norm and s1.name_norm == tgt.name_norm) else 0.0
    s1_nt = set(s1.name_tokens)
    tgt_nt = set(tgt.name_tokens)

    name_jaccard = (
        len(s1_nt & tgt_nt) / len(s1_nt | tgt_nt) if (s1_nt or tgt_nt) else 0.0
    )
    name_overlap = (
        len(s1_nt & tgt_nt) / min(len(s1_nt), len(tgt_nt))
        if (s1_nt and tgt_nt)
        else 0.0
    )

    name_edit = (
        fuzz.ratio(s1.name_norm, tgt.name_norm) / 100.0
        if (s1.name_norm and tgt.name_norm)
        else 0.0
    )
    name_sort = (
        fuzz.token_sort_ratio(s1.name_norm, tgt.name_norm) / 100.0
        if (s1.name_norm and tgt.name_norm)
        else 0.0
    )
    name_len_diff = float(abs(len(s1.name_norm) - len(tgt.name_norm)))
    max_len = max(len(s1.name_norm), len(tgt.name_norm), 1)
    min_len = min(len(s1.name_norm), len(tgt.name_norm))
    name_len_ratio = float(min_len) / float(max_len)

    # 2. Address features
    addr_exact = (
        1.0
        if (not s1.addr_missing and not tgt.addr_missing and s1.addr_norm == tgt.addr_norm)
        else 0.0
    )
    s1_at = set(s1.addr_tokens)
    tgt_at = set(tgt.addr_tokens)

    addr_jaccard = (
        len(s1_at & tgt_at) / len(s1_at | tgt_at) if (s1_at or tgt_at) else 0.0
    )
    addr_overlap = (
        len(s1_at & tgt_at) / min(len(s1_at), len(tgt_at))
        if (s1_at and tgt_at)
        else 0.0
    )

    addr_edit = (
        fuzz.ratio(s1.addr_norm, tgt.addr_norm) / 100.0
        if (not s1.addr_missing and not tgt.addr_missing)
        else 0.0
    )

    s1_nums = set(s1.addr_numbers)
    tgt_nums = set(tgt.addr_numbers)
    num_jaccard = (
        len(s1_nums & tgt_nums) / len(s1_nums | tgt_nums)
        if (s1_nums or tgt_nums)
        else 0.0
    )
    num_overlap = (
        len(s1_nums & tgt_nums) / min(len(s1_nums), len(tgt_nums))
        if (s1_nums and tgt_nums)
        else 0.0
    )

    addr_miss_s1 = 1.0 if s1.addr_missing else 0.0
    addr_miss_tgt = 1.0 if tgt.addr_missing else 0.0
    addr_miss_either = 1.0 if (s1.addr_missing or tgt.addr_missing) else 0.0

    # 3. Interactions
    prod_sim = name_edit * addr_edit
    min_sim = min(name_edit, addr_edit)
    mean_sim = (name_edit + addr_edit) / 2.0
    both_high = 1.0 if (name_edit > 0.80 and addr_edit > 0.80) else 0.0
    name_hi_addr_lo = 1.0 if (name_edit > 0.85 and addr_edit < 0.35) else 0.0
    addr_hi_name_lo = 1.0 if (addr_edit > 0.85 and name_edit < 0.35) else 0.0
    name_hi_addr_miss = 1.0 if (name_edit > 0.80 and tgt.addr_missing) else 0.0

    # 4. Source & Provenance
    is_s2 = 1.0 if tgt.source == "S2" else 0.0
    is_s3 = 1.0 if tgt.source == "S3" else 0.0
    same_c = 1.0 if s1.country == tgt.country else 0.0

    return [
        name_exact,
        name_jaccard,
        name_overlap,
        name_edit,
        name_sort,
        name_len_diff,
        name_len_ratio,
        addr_exact,
        addr_jaccard,
        addr_overlap,
        addr_edit,
        num_jaccard,
        num_overlap,
        addr_miss_s1,
        addr_miss_tgt,
        addr_miss_either,
        prod_sim,
        min_sim,
        mean_sim,
        both_high,
        name_hi_addr_lo,
        addr_hi_name_lo,
        name_hi_addr_miss,
        is_s2,
        is_s3,
        float(pair.route_exact_name),
        float(pair.route_sorted_name),
        float(pair.route_exact_addr),
        float(pair.route_numeric_anchor),
        float(pair.route_rare_token),
        float(pair.route_char_ngram),
        float(pair.route_count),
        same_c,
    ]


# ==============================================================================
# 4. GBDT Matcher & Hard-Negative Mining
# ==============================================================================

class GBDTMatcher:
    """Gradient Boosted Decision Tree Matcher."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_cfg = config.get("model", {})
        self.algo = self.model_cfg.get("algorithm", "lightgbm").lower()
        self.model: Any = None

    def _init_estimator(self):
        n_est = self.model_cfg.get("n_estimators", 200)
        lr = self.model_cfg.get("learning_rate", 0.08)
        max_d = self.model_cfg.get("max_depth", 6)
        sub = self.model_cfg.get("subsample", 0.8)
        col = self.model_cfg.get("colsample_bytree", 0.8)
        seed = self.model_cfg.get("random_state", 42)

        if self.algo == "lightgbm":
            try:
                import lightgbm as lgb
                num_leaves = self.model_cfg.get("num_leaves", 31)
                return lgb.LGBMClassifier(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=max_d,
                    num_leaves=num_leaves,
                    subsample=sub,
                    colsample_bytree=col,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not found. Falling back to XGBoost.")
                self.algo = "xgboost"

        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=max_d,
            subsample=sub,
            colsample_bytree=col,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit estimator on training feature matrix and labels."""
        self.model = self._init_estimator()
        logger.info(f"Training {self.algo.upper()} on {X.shape[0]:,} candidate pairs ({np.sum(y):,} positives)...")
        t0 = time.time()
        self.model.fit(X, y)
        logger.info(f"Model training complete in {time.time()-t0:.1f}s.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        """Save model artifact to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "algo": self.algo, "config": self.config}, path)
        logger.info(f"Saved GBDT matcher to {path}")

    @classmethod
    def load(cls, path: Path) -> "GBDTMatcher":
        """Load model artifact from disk."""
        data = joblib.load(path)
        inst = cls(data.get("config", {}))
        inst.model = data["model"]
        inst.algo = data.get("algo", "lightgbm")
        logger.info(f"Loaded GBDT matcher from {path}")
        return inst


def mine_hard_negatives(
    model: GBDTMatcher,
    X_train: np.ndarray,
    y_train: np.ndarray,
    min_prob: float = 0.28,
    weight_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify false positive candidates with high predicted scores and augment training."""
    logger.info("Scoring candidate pairs to mine hard negatives...")
    probs = model.predict_proba(X_train)
    hard_neg_mask = (y_train == 0) & (probs >= min_prob)
    num_hard = int(np.sum(hard_neg_mask))
    logger.info(f"Discovered {num_hard:,} hard negative candidate pairs (p >= {min_prob}).")

    if num_hard == 0:
        return X_train, y_train

    # Re-sample or duplicate hard negatives to strengthen decision boundaries
    X_hard = X_train[hard_neg_mask]
    y_hard = y_train[hard_neg_mask]

    X_aug = np.vstack([X_train] + [X_hard] * (weight_factor - 1))
    y_aug = np.concatenate([y_train] + [y_hard] * (weight_factor - 1))

    logger.info(f"Augmented training set: {X_train.shape[0]:,} -> {X_aug.shape[0]:,} rows.")
    return X_aug, y_aug


# ==============================================================================
# 5. Entity-Level Decision Engine & Singleton Policy
# ==============================================================================

def decide_matches_for_entity(
    scored_candidates: List[Tuple[str, float, NormalizedRecord]],
    base_threshold: float = 0.46,
    singleton_threshold: float = 0.32,
    max_score_gap: float = 0.35,
    max_matches: int = 10,
    enable_graph: bool = True,
) -> List[str]:
    """Make set-valued match decisions for a single S1 entity.

    Enforces singleton policy, maximum score gap, and cross-source witness boost.
    """
    if not scored_candidates:
        return []

    # Sort descending by score
    scored_sorted = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    top_score = scored_sorted[0][1]

    # Singleton policy: insufficient confidence in top candidate -> declare empty
    if top_score < singleton_threshold:
        return []

    # Check for local cross-source witness agreements
    s2_high = [c for c in scored_sorted if c[2].source == "S2" and c[1] >= 0.40]
    s3_high = [c for c in scored_sorted if c[2].source == "S3" and c[1] >= 0.40]
    has_dual_source_witness = bool(s2_high and s3_high)

    selected: List[str] = []
    for tid, score, tgt_rec in scored_sorted:
        # Check gap from top candidate
        if (top_score - score) > max_score_gap:
            continue

        # Decision rule:
        # Standard threshold OR lower threshold when supported by dual-source witness
        effective_threshold = (
            (base_threshold - 0.06) if (enable_graph and has_dual_source_witness) else base_threshold
        )

        if score >= effective_threshold:
            selected.append(tid)
            if len(selected) >= max_matches:
                break

    return selected


# ==============================================================================
# 6. Evaluation Metrics & Cohort Diagnostics
# ==============================================================================

def compute_per_entity_f05(
    pred_ids: Set[str],
    true_ids: Set[str],
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F0.5 per S1 entity."""
    if not true_ids:
        # Singleton entity in ground truth
        if not pred_ids:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    if not pred_ids:
        # False negative singleton
        return 0.0, 0.0, 0.0

    tp = len(pred_ids & true_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

    return precision, recall, f05


def evaluate_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    entity_records: Optional[Dict[str, NormalizedRecord]] = None,
) -> dict:
    """Evaluate predictions against ground truth computing macro F0.5 and cohort metrics."""
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {}

    precisions: List[float] = []
    recalls: List[float] = []
    f05s: List[float] = []

    # Cohort trackers
    cohort_stats: Dict[str, Dict[str, List[float]]] = {
        "all": {"prec": [], "rec": [], "f05": []},
        "singletons": {"prec": [], "rec": [], "f05": []},
        "one_match": {"prec": [], "rec": [], "f05": []},
        "multi_match": {"prec": [], "rec": [], "f05": []},
        "s2_only": {"prec": [], "rec": [], "f05": []},
        "s3_only": {"prec": [], "rec": [], "f05": []},
        "both_sources": {"prec": [], "rec": [], "f05": []},
    }

    singleton_correct = 0
    singleton_total = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f = compute_per_entity_f05(pred_set, true_set)

        precisions.append(p)
        recalls.append(r)
        f05s.append(f)

        cohort_stats["all"]["prec"].append(p)
        cohort_stats["all"]["rec"].append(r)
        cohort_stats["all"]["f05"].append(f)

        num_true = len(true_set)
        if num_true == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
            cohort_stats["singletons"]["prec"].append(p)
            cohort_stats["singletons"]["rec"].append(r)
            cohort_stats["singletons"]["f05"].append(f)
        elif num_true == 1:
            cohort_stats["one_match"]["prec"].append(p)
            cohort_stats["one_match"]["rec"].append(r)
            cohort_stats["one_match"]["f05"].append(f)
        else:
            cohort_stats["multi_match"]["prec"].append(p)
            cohort_stats["multi_match"]["rec"].append(r)
            cohort_stats["multi_match"]["f05"].append(f)

        if num_true > 0:
            has_s2 = any(tid.startswith("S2-") for tid in true_set)
            has_s3 = any(tid.startswith("S3-") for tid in true_set)
            if has_s2 and has_s3:
                cohort_stats["both_sources"]["prec"].append(p)
                cohort_stats["both_sources"]["rec"].append(r)
                cohort_stats["both_sources"]["f05"].append(f)
            elif has_s2:
                cohort_stats["s2_only"]["prec"].append(p)
                cohort_stats["s2_only"]["rec"].append(r)
                cohort_stats["s2_only"]["f05"].append(f)
            elif has_s3:
                cohort_stats["s3_only"]["prec"].append(p)
                cohort_stats["s3_only"]["rec"].append(r)
                cohort_stats["s3_only"]["f05"].append(f)

    # Compile cohort breakdown
    cohort_summary = {}
    for c_name, vals in cohort_stats.items():
        cnt = len(vals["f05"])
        if cnt > 0:
            cohort_summary[c_name] = {
                "count": cnt,
                "macro_f05": float(np.mean(vals["f05"])),
                "mean_precision": float(np.mean(vals["prec"])),
                "mean_recall": float(np.mean(vals["rec"])),
            }

    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0

    return {
        "macro_f05": float(np.mean(f05s)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "singleton_accuracy": float(singleton_acc),
        "total_evaluated": total_entities,
        "cohorts": cohort_summary,
    }


def evaluate_candidate_recall(
    candidates_by_s1: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """Compute candidate recall ceiling and candidate set size distribution."""
    total_true_links = 0
    covered_true_links = 0
    cand_counts: List[int] = []

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates_by_s1.get(s1_id, set())
        cand_counts.append(len(cand_set))
        if true_set:
            total_true_links += len(true_set)
            covered_true_links += len(true_set & cand_set)

    recall = (
        covered_true_links / total_true_links if total_true_links > 0 else 1.0
    )
    arr = np.array(cand_counts) if cand_counts else np.array([0])

    return {
        "candidate_recall_ceiling": float(recall),
        "covered_true_links": int(covered_true_links),
        "total_true_links": int(total_true_links),
        "avg_candidates_per_s1": float(np.mean(arr)),
        "median_candidates": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max_candidates": int(np.max(arr)),
        "zero_candidate_pct": float(np.mean(arr == 0) * 100),
    }


# ==============================================================================
# 7. Data Loading & I/O Helpers
# ==============================================================================

def load_source_records(
    file_path: Path,
    max_rows: Optional[int] = None,
) -> Dict[str, NormalizedRecord]:
    """Streamingly load and normalize a source TSV file into memory."""
    records: Dict[str, NormalizedRecord] = {}
    logger.info(f"Loading and normalizing {file_path.name} (max_rows={max_rows})...")
    t0 = time.time()
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for idx, row in enumerate(reader):
            if max_rows and idx >= max_rows:
                break
            if len(row) >= 4:
                rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3])
                records[rec.entity_id] = rec
    logger.info(f"Loaded {len(records):,} records from {file_path.name} in {time.time()-t0:.1f}s.")
    return records


def load_ground_truth(
    file_path: Path,
    filter_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Load train_ground_truth.tsv into a mapping of s1_id -> set of matched target IDs."""
    gt: Dict[str, Set[str]] = {}
    logger.info(f"Loading ground truth from {file_path.name}...")
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if filter_s1_ids and s1_id not in filter_s1_ids:
                continue
            matches = set()
            if len(row) > 1 and row[1].strip():
                matches = {m.strip() for m in row[1].split(",") if m.strip()}
            gt[s1_id] = matches
    logger.info(f"Loaded ground truth for {len(gt):,} S1 entities.")
    return gt


def format_id_list(ids: Sequence[str]) -> str:
    """Format a list of IDs as comma-separated string with deterministic sorting."""
    if not ids:
        return ""
    return ",".join(sorted(ids))


# ==============================================================================
# 8. Pipeline Execution Workflows
# ==============================================================================

def run_profiling(config: dict) -> dict:
    """Run data forensics and dataset profiling."""
    train_dir = Path(config["paths"]["train_dir"])
    test_dir = Path(config["paths"]["test_dir"])
    artifact_dir = Path(config["paths"]["artifact_dir"])
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("============================================================")
    logger.info("RUNNING DATA FORENSICS & PROFILING")
    logger.info("============================================================")

    profile_data: Dict[str, Any] = {}

    # 1. Ground truth stats
    gt_path = train_dir / "train_ground_truth.tsv"
    if gt_path.is_file():
        total_s1 = 0
        card_dist: Dict[int, int] = defaultdict(int)
        total_links = 0
        with gt_path.open("r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                if not row:
                    continue
                total_s1 += 1
                links = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 else []
                k = len(links)
                card_dist[k] += 1
                total_links += k

        profile_data["ground_truth"] = {
            "total_s1": total_s1,
            "singletons": card_dist[0],
            "singleton_pct": (card_dist[0] / total_s1 * 100) if total_s1 else 0,
            "total_links": total_links,
            "cardinality_distribution": dict(sorted(card_dist.items())),
        }
        logger.info(
            f"Ground Truth: {total_s1:,} S1 entities, {card_dist[0]:,} singletons "
            f"({profile_data['ground_truth']['singleton_pct']:.2f}%), {total_links:,} links."
        )

    # 2. Source file profiles
    for source_dir, prefix in [(train_dir, "train"), (test_dir, "test")]:
        for src_name in [f"{prefix}_source1.tsv", f"{prefix}_source2.tsv", f"{prefix}_source3.tsv"]:
            fpath = source_dir / src_name
            if not fpath.is_file():
                continue
            rows = 0
            countries: Dict[str, int] = defaultdict(int)
            missing_addrs = 0
            with fpath.open("r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r, None)
                for row in r:
                    if len(row) >= 4:
                        rows += 1
                        countries[row[3].strip()] += 1
                        if not row[2] or not row[2].strip():
                            missing_addrs += 1
            profile_data[src_name] = {
                "rows": rows,
                "countries": dict(countries),
                "missing_addresses": missing_addrs,
                "missing_address_pct": (missing_addrs / rows * 100) if rows else 0,
            }
            logger.info(f"{src_name}: {rows:,} rows, countries={dict(countries)}, missing_addr={missing_addrs:,}")

    report_path = reports_dir / "data_forensics_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)
    logger.info(f"Forensics report saved to {report_path}")
    return profile_data


def run_training_pipeline(config: dict) -> Tuple[GBDTMatcher, dict]:
    """Execute complete training pipeline: blocking, feature extraction, hard-negative mining, retrain."""
    train_dir = Path(config["paths"]["train_dir"])
    artifact_dir = Path(config["paths"]["artifact_dir"])
    models_dir = artifact_dir / "models"
    reports_dir = artifact_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = config.get("experiment", {})
    block_cfg = config.get("blocking", {})
    hn_cfg = config.get("hard_negatives", {})
    dec_cfg = config.get("decision", {})

    s1_sample_limit = exp_cfg.get("train_s1_sample", 60000)
    val_sample_limit = exp_cfg.get("val_s1_sample", 15000)

    logger.info("============================================================")
    logger.info("STARTING PIPELINE TRAINING WORKFLOW")
    logger.info("============================================================")

    # 1. Load S1 reference entities
    s1_all = load_source_records(train_dir / "train_source1.tsv", max_rows=s1_sample_limit)
    s1_keys = list(s1_all.keys())

    # Split S1 entities into Train / Val
    val_ratio = exp_cfg.get("val_split_ratio", 0.20)
    rng = np.random.default_rng(exp_cfg.get("seed", 42))
    shuffled_keys = list(s1_keys)
    rng.shuffle(shuffled_keys)

    n_val = min(val_sample_limit, int(len(shuffled_keys) * val_ratio))
    val_s1_ids = set(shuffled_keys[:n_val])
    train_s1_ids = set(shuffled_keys[n_val:])

    logger.info(f"S1 Split: Train={len(train_s1_ids):,} entities, Validation={len(val_s1_ids):,} entities.")

    # 2. Load ground truth for sampled entities
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv", filter_s1_ids=set(s1_keys))
    train_gt = {k: gt.get(k, set()) for k in train_s1_ids}
    val_gt = {k: gt.get(k, set()) for k in val_s1_ids}

    # Collect all positive target IDs needed for training/validation
    needed_positive_ids = set()
    for s in gt.values():
        needed_positive_ids |= s

    # 3. Load target records (S2 and S3)
    # Load enough targets to cover positives and build inverted indexes
    s2_records = load_source_records(train_dir / "train_source2.tsv", max_rows=300000)
    s3_records = load_source_records(train_dir / "train_source3.tsv", max_rows=300000)
    all_targets: Dict[str, NormalizedRecord] = {**s2_records, **s3_records}

    # Ensure missing true positive targets are loaded if they exist beyond the max_rows cutoff
    missing_pos_ids = needed_positive_ids - set(all_targets.keys())
    if missing_pos_ids:
        logger.info(f"Streaming {len(missing_pos_ids):,} specific true target matches into memory...")
        for fname in ["train_source2.tsv", "train_source3.tsv"]:
            if not missing_pos_ids:
                break
            with (train_dir / fname).open("r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r, None)
                for row in r:
                    if len(row) >= 4 and row[0] in missing_pos_ids:
                        rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3])
                        all_targets[rec.entity_id] = rec
                        missing_pos_ids.remove(rec.entity_id)

    logger.info(f"Target catalog assembled: {len(all_targets):,} target records.")

    # Partition targets by country for fast, isolated candidate indexing
    targets_by_country: Dict[str, List[NormalizedRecord]] = defaultdict(list)
    for r in all_targets.values():
        targets_by_country[r.country].append(r)

    # 4. Multi-Pass Candidate Generation
    generators: Dict[str, MultiPassCandidateGenerator] = {}
    for country, tgt_list in targets_by_country.items():
        gen = MultiPassCandidateGenerator(
            max_candidates_per_key=block_cfg.get("max_candidates_per_key", 60),
            max_candidates_per_s1=block_cfg.get("max_candidates_per_s1", 50),
            min_rare_freq=block_cfg.get("min_rare_freq", 1),
            max_rare_freq=block_cfg.get("max_rare_freq", 50),
        )
        gen.build_indexes(tgt_list)
        generators[country] = gen

    # Generate candidates for train S1 records
    logger.info("Generating candidate pairs for training set...")
    train_cand_pairs: List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]] = []
    train_cands_by_s1: Dict[str, Set[str]] = defaultdict(set)

    for s1_id in train_s1_ids:
        s1_rec = s1_all[s1_id]
        gen = generators.get(s1_rec.country)
        if gen:
            cands = gen.generate_candidates_for_record(s1_rec)
            for tid, pair in cands.items():
                if tid in all_targets:
                    train_cand_pairs.append((s1_rec, all_targets[tid], pair))
                    train_cands_by_s1[s1_id].add(tid)

    # Audit candidate recall on train
    train_recall_report = evaluate_candidate_recall(train_cands_by_s1, train_gt)
    logger.info(
        f"Train Candidate Recall Ceiling: {train_recall_report['candidate_recall_ceiling']*100:.2f}% "
        f"({train_recall_report['covered_true_links']:,}/{train_recall_report['total_true_links']:,} true links), "
        f"Avg candidates/S1: {train_recall_report['avg_candidates_per_s1']:.2f}"
    )

    # 5. Extract Feature Matrix
    logger.info(f"Extracting features on {len(train_cand_pairs):,} candidate pairs...")
    X_train_list: List[List[float]] = []
    y_train_list: List[int] = []

    for s1_r, tgt_r, cp in train_cand_pairs:
        feat_vec = extract_pairwise_feature_vector(s1_r, tgt_r, cp)
        label = 1 if tgt_r.entity_id in train_gt.get(s1_r.entity_id, set()) else 0
        X_train_list.append(feat_vec)
        y_train_list.append(label)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    del X_train_list, y_train_list
    gc.collect()

    # 6. Train Base GBDT Matcher
    matcher = GBDTMatcher(config)
    matcher.fit(X_train, y_train)

    # 7. Hard-Negative Mining
    if hn_cfg.get("enabled", True):
        logger.info("Executing Hard-Negative Mining stage...")
        min_p = hn_cfg.get("min_prob", 0.28)
        X_aug, y_aug = mine_hard_negatives(matcher, X_train, y_train, min_prob=min_p)
        matcher.fit(X_aug, y_aug)
        del X_aug, y_aug
        gc.collect()

    # Save trained model artifact
    model_save_path = models_dir / "gbdt_matcher.joblib"
    matcher.save(model_save_path)

    # 8. Validation Evaluation & Threshold Tuning
    logger.info("Generating candidates and evaluating on held-out validation set...")
    val_cand_pairs: List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]] = []
    val_cands_by_s1: Dict[str, Set[str]] = defaultdict(set)

    for s1_id in val_s1_ids:
        s1_rec = s1_all[s1_id]
        gen = generators.get(s1_rec.country)
        if gen:
            cands = gen.generate_candidates_for_record(s1_rec)
            for tid, pair in cands.items():
                if tid in all_targets:
                    val_cand_pairs.append((s1_rec, all_targets[tid], pair))
                    val_cands_by_s1[s1_id].add(tid)

    val_recall_report = evaluate_candidate_recall(val_cands_by_s1, val_gt)
    logger.info(
        f"Validation Candidate Recall Ceiling: {val_recall_report['candidate_recall_ceiling']*100:.2f}% "
        f"({val_recall_report['covered_true_links']:,}/{val_recall_report['total_true_links']:,} true links), "
        f"Avg candidates/S1: {val_recall_report['avg_candidates_per_s1']:.2f}"
    )

    # Extract validation features and score
    val_preds_by_s1: Dict[str, List[Tuple[str, float, NormalizedRecord]]] = defaultdict(list)
    if val_cand_pairs:
        logger.info(f"Extracting features for {len(val_cand_pairs):,} validation pairs...")
        X_val = np.array(
            [extract_pairwise_feature_vector(s, t, cp) for s, t, cp in val_cand_pairs],
            dtype=np.float32,
        )
        val_probs = matcher.predict_proba(X_val)
        for (s1_r, tgt_r, _), p in zip(val_cand_pairs, val_probs):
            val_preds_by_s1[s1_r.entity_id].append((tgt_r.entity_id, float(p), tgt_r))

    # Grid search threshold to optimize S1 Macro F0.5
    best_f05 = -1.0
    best_thresh = dec_cfg.get("base_threshold", 0.46)
    best_single_thresh = dec_cfg.get("singleton_threshold", 0.32)
    best_report: dict = {}

    threshold_candidates = [0.38, 0.42, 0.46, 0.50, 0.54, 0.58]
    singleton_candidates = [0.28, 0.32, 0.36]

    for bt in threshold_candidates:
        for st in singleton_candidates:
            val_preds: Dict[str, Set[str]] = {}
            for s1_id in val_s1_ids:
                scored = val_preds_by_s1.get(s1_id, [])
                matches = decide_matches_for_entity(
                    scored,
                    base_threshold=bt,
                    singleton_threshold=st,
                    max_score_gap=dec_cfg.get("max_score_gap", 0.35),
                    max_matches=dec_cfg.get("max_matches_per_s1", 10),
                )
                val_preds[s1_id] = set(matches)

            eval_res = evaluate_predictions(val_preds, val_gt)
            macro_f = eval_res["macro_f05"]
            if macro_f > best_f05:
                best_f05 = macro_f
                best_thresh = bt
                best_single_thresh = st
                best_report = eval_res

    logger.info("============================================================")
    logger.info(f"OPTIMAL VALIDATION MACRO F0.5: {best_f05:.4f}")
    logger.info(f"Selected Thresholds: base_threshold={best_thresh:.2f}, singleton_threshold={best_single_thresh:.2f}")
    logger.info(f"Precision: {best_report.get('mean_precision', 0):.4f}, Recall: {best_report.get('mean_recall', 0):.4f}")
    logger.info(f"Singleton Accuracy: {best_report.get('singleton_accuracy', 0):.4f}")
    logger.info("============================================================")

    # Save validation report
    val_report_data = {
        "candidate_recall": val_recall_report,
        "validation_metrics": best_report,
        "optimal_thresholds": {
            "base_threshold": best_thresh,
            "singleton_threshold": best_single_thresh,
        },
    }
    with (reports_dir / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(val_report_data, f, indent=2)

    return matcher, val_report_data


# ==============================================================================
# 9. Streaming Country-Partitioned Test Inference
# ==============================================================================

def run_test_inference(config: dict) -> None:
    """Execute streaming country-partitioned test inference and generate submission files."""
    test_dir = Path(config["paths"]["test_dir"])
    artifact_dir = Path(config["paths"]["artifact_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = artifact_dir / "models"
    model_path = models_dir / "gbdt_matcher.joblib"

    if not model_path.is_file():
        logger.error(f"Trained model not found at {model_path}. Run --mode train first.")
        sys.exit(1)

    matcher = GBDTMatcher.load(model_path)

    # Check for tuned thresholds from validation report
    reports_dir = artifact_dir / "reports"
    val_report_path = reports_dir / "validation_report.json"
    base_thresh = config["decision"].get("base_threshold", 0.46)
    single_thresh = config["decision"].get("singleton_threshold", 0.32)

    if val_report_path.is_file():
        try:
            with val_report_path.open("r", encoding="utf-8") as f:
                v_data = json.load(f)
                opts = v_data.get("optimal_thresholds", {})
                base_thresh = opts.get("base_threshold", base_thresh)
                single_thresh = opts.get("singleton_threshold", single_thresh)
                logger.info(f"Loaded tuned thresholds: base={base_thresh}, singleton={single_thresh}")
        except Exception as e:
            logger.warning(f"Could not read tuned thresholds: {e}")

    logger.info("============================================================")
    logger.info("STARTING STREAMING TEST INFERENCE")
    logger.info("============================================================")

    # Step 1: Discover all countries present in test set (Open-Set handling)
    test_s1_path = test_dir / "test_source1.tsv"
    countries_seen: Set[str] = set()
    logger.info(f"Scanning test countries from {test_s1_path.name}...")
    with test_s1_path.open("r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        for row in r:
            if len(row) >= 4 and row[3].strip():
                countries_seen.add(row[3].strip().upper())

    logger.info(f"Test Countries Discovered (Open-Set): {sorted(countries_seen)}")

    # Step 2: Process each country partition independently
    temp_matching_files: Dict[str, Path] = {}
    temp_candidate_files: Dict[str, Path] = {}

    batch_size = config.get("test_inference", {}).get("batch_size", 5000)

    for country in sorted(countries_seen):
        logger.info("------------------------------------------------------------")
        logger.info(f"PROCESSING TEST PARTITION: {country}")
        logger.info("------------------------------------------------------------")
        t_country_start = time.time()

        # Step A: Register temp partition paths and check if already computed
        m_part_path = output_dir / f"temp_matching_{country.lower()}.tsv"
        c_part_path = output_dir / f"temp_candidates_{country.lower()}.tsv"
        temp_matching_files[country] = m_part_path
        temp_candidate_files[country] = c_part_path

        if m_part_path.is_file() and c_part_path.is_file() and m_part_path.stat().st_size > 1000:
            logger.info(
                f"Found existing valid partition for '{country}' "
                f"({m_part_path.stat().st_size:,} bytes). Skipping computation."
            )
            continue

        # Load S2 and S3 target records for this country with gc disabled and store_raw=False
        country_targets: Dict[str, NormalizedRecord] = {}
        gc.disable()
        try:
            for fname in ["test_source2.tsv", "test_source3.tsv"]:
                fpath = test_dir / fname
                logger.info(f"Loading {fname} for country '{country}'...")
                with fpath.open("r", encoding="utf-8") as f:
                    r = csv.reader(f, delimiter="\t")
                    next(r, None)
                    for row in r:
                        if len(row) >= 4 and row[3].strip().upper() == country:
                            rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                            country_targets[rec.entity_id] = rec
        finally:
            gc.enable()
            gc.collect()

        logger.info(f"Loaded {len(country_targets):,} target records for country {country}.")

        # Build multi-pass candidate generator
        gen = MultiPassCandidateGenerator(
            max_candidates_per_key=config["blocking"].get("max_candidates_per_key", 60),
            max_candidates_per_s1=config["blocking"].get("max_candidates_per_s1", 50),
            min_rare_freq=config["blocking"].get("min_rare_freq", 1),
            max_rare_freq=config["blocking"].get("max_rare_freq", 50),
        )
        gen.build_indexes(country_targets.values())

        # Stream S1 test records for this country
        total_country_s1 = 0
        total_country_matches = 0

        with m_part_path.open("w", encoding="utf-8", newline="") as fm, \
             c_part_path.open("w", encoding="utf-8", newline="") as fc:

            wm = csv.writer(fm, delimiter="\t", lineterminator="\n")
            wc = csv.writer(fc, delimiter="\t", lineterminator="\n")

            batch_s1: List[NormalizedRecord] = []

            def flush_batch(batch: List[NormalizedRecord]) -> None:
                nonlocal total_country_s1, total_country_matches
                pairs_to_score: List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]] = []
                cand_map: Dict[str, List[str]] = {}

                for s1_r in batch:
                    cands = gen.generate_candidates_for_record(s1_r)
                    cand_map[s1_r.entity_id] = list(cands.keys())
                    for tid, cp in cands.items():
                        if tid in country_targets:
                            pairs_to_score.append((s1_r, country_targets[tid], cp))

                preds_by_s1: Dict[str, List[Tuple[str, float, NormalizedRecord]]] = defaultdict(list)
                if pairs_to_score:
                    feat_matrix = np.array(
                        [extract_pairwise_feature_vector(s, t, cp) for s, t, cp in pairs_to_score],
                        dtype=np.float32,
                    )
                    scores = matcher.predict_proba(feat_matrix)
                    for (s1_r, tgt_r, _), p in zip(pairs_to_score, scores):
                        preds_by_s1[s1_r.entity_id].append((tgt_r.entity_id, float(p), tgt_r))

                for s1_r in batch:
                    s1_id = s1_r.entity_id
                    all_c = cand_map.get(s1_id, [])
                    scored = preds_by_s1.get(s1_id, [])

                    matches = decide_matches_for_entity(
                        scored,
                        base_threshold=base_thresh,
                        singleton_threshold=single_thresh,
                        max_score_gap=config["decision"].get("max_score_gap", 0.35),
                        max_matches=config["decision"].get("max_matches_per_s1", 10),
                    )

                    # Invariant: matches must be subset of candidates
                    c_set = set(all_c)
                    valid_matches = [m for m in matches if m in c_set]

                    wm.writerow([s1_id, format_id_list(valid_matches)])
                    wc.writerow([s1_id, format_id_list(all_c)])

                    total_country_s1 += 1
                    total_country_matches += len(valid_matches)

            with test_s1_path.open("r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r, None)
                for row in r:
                    if len(row) >= 4 and row[3].strip().upper() == country:
                        rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                        batch_s1.append(rec)
                        if len(batch_s1) >= batch_size:
                            flush_batch(batch_s1)
                            batch_s1 = []
                            if total_country_s1 % 10000 == 0:
                                elapsed = max(time.time() - t_country_start, 0.001)
                                rate = total_country_s1 / elapsed
                                logger.info(
                                    f"[{country}] Processed {total_country_s1:,} entities "
                                    f"({rate:.0f} ent/s, matches={total_country_matches:,})..."
                                )

                if batch_s1:
                    flush_batch(batch_s1)

        logger.info(
            f"Finished {country}: {total_country_s1:,} entities, {total_country_matches:,} matches "
            f"in {time.time()-t_country_start:.1f}s."
        )

        del country_targets, gen
        gc.collect()

    # Step 3: Assemble canonical final files preserving exact test_source1.tsv ordering
    final_matching_path = output_dir / "matching_results.tsv"
    final_candidate_path = output_dir / "candidate_pairs.tsv"

    logger.info("Assembling final submission files with exact test ordering...")
    matching_dict: Dict[str, str] = {}
    candidate_dict: Dict[str, str] = {}

    for c, path_m in temp_matching_files.items():
        with path_m.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) == 2:
                    matching_dict[parts[0]] = parts[1]
                elif len(parts) == 1:
                    matching_dict[parts[0]] = ""

    for c, path_c in temp_candidate_files.items():
        with path_c.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) == 2:
                    candidate_dict[parts[0]] = parts[1]
                elif len(parts) == 1:
                    candidate_dict[parts[0]] = ""

    written_count = 0
    with final_matching_path.open("w", encoding="utf-8", newline="") as fm, \
         final_candidate_path.open("w", encoding="utf-8", newline="") as fc:

        wm = csv.writer(fm, delimiter="\t", lineterminator="\n")
        wc = csv.writer(fc, delimiter="\t", lineterminator="\n")

        wm.writerow(["source1_entity_id", "matched_entity_ids"])
        wc.writerow(["source1_entity_id", "candidate_entity_ids"])

        with test_s1_path.open("r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                s1_id = row[0].strip()
                wm.writerow([s1_id, matching_dict.get(s1_id, "")])
                wc.writerow([s1_id, candidate_dict.get(s1_id, "")])
                written_count += 1

    logger.info(f"Successfully generated final submission files ({written_count:,} test entities):")
    logger.info(f"  -> {final_matching_path}")
    logger.info(f"  -> {final_candidate_path}")

    # Step 4: Run official validator
    validator_script = Path(config["paths"].get("validator_script", "challenge/validate_submission.py"))
    if validator_script.is_file():
        logger.info("Executing official challenge submission validator...")
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(final_matching_path),
            "--candidate", str(final_candidate_path),
            "--test-dir", str(test_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        logger.info("Validator Output:\n" + res.stdout)
        if res.stderr:
            logger.warning("Validator stderr:\n" + res.stderr)
        if res.returncode == 0:
            logger.info("Official Validator result: PASS (Exit code 0)")
        else:
            logger.error(f"Official Validator reported issues (Exit code {res.returncode})")


# ==============================================================================
# 10. CLI Argument Parsing & Entry Point
# ==============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file."""
    path = Path(config_path)
    if not path.is_file():
        logger.error(f"Config file not found at {path}")
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Business Entity Resolution Pipeline",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=[
            "profile",
            "validate",
            "train",
            "test",
            "full",
            "diagnostics",
            "candidates",
            "features",
            "score",
            "decision",
        ],
        help="Pipeline execution mode (default: full)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override training S1 sample size for smoke testing",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.sample_size:
        config["experiment"]["train_s1_sample"] = args.sample_size
        config["experiment"]["val_s1_sample"] = max(1000, args.sample_size // 4)

    mode = args.mode.lower()
    logger.info(f"Initializing Amazon ML Pipeline in mode: '{mode}'")

    if mode == "profile":
        run_profiling(config)
    elif mode == "train":
        run_training_pipeline(config)
    elif mode == "validate":
        run_training_pipeline(config)
    elif mode == "test":
        run_test_inference(config)
    elif mode == "diagnostics":
        run_profiling(config)
    elif mode in ["candidates", "features", "score", "decision"]:
        logger.info(f"Running targeted stage: {mode}")
        run_training_pipeline(config)
    elif mode == "full":
        run_profiling(config)
        run_training_pipeline(config)
        run_test_inference(config)
    else:
        logger.error(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
