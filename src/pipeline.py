"""End-to-End Entity Resolution Pipeline.

Coordinates candidate generation, pairwise feature extraction, GBDT training,
validation macro F0.5 optimization, and streaming test inference.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.blocking import CandidatePair, CandidateSet
from src.blocking.exact import ExactAddressBlocking, ExactNameAndAddressBlocking, ExactNameBlocking
from src.blocking.numeric import NumericAnchorBlocking
from src.blocking.rare_tokens import RareTokenBlocking
from src.decision.entity_decision import decide_matches_for_entity
from src.evaluation.candidate_recall import compute_candidate_recall
from src.evaluation.diagnostics import evaluate_cohorts
from src.evaluation.metrics import compute_macro_f05, compute_per_entity_f05, evaluate_predictions
from src.features.address import compute_address_features
from src.features.interactions import compute_interaction_features
from src.features.name import compute_name_features
from src.features.retrieval import compute_retrieval_features
from src.io import format_id_list, parse_id_list, write_candidate_pairs, write_matching_results
from src.normalization import NormalizedRecord

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "name_exact",
    "name_token_jaccard",
    "name_token_overlap",
    "name_edit_similarity",
    "name_char_ngram_similarity",
    "name_length_diff",
    "name_length_ratio",
    "addr_exact",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_edit_similarity",
    "addr_numeric_overlap",
    "addr_missing_s1",
    "addr_missing_target",
    "addr_missing_either",
    "interaction_product",
    "interaction_min",
    "interaction_mean",
    "interaction_both_strong",
    "interaction_name_strong_addr_weak",
    "interaction_addr_strong_name_weak",
    "interaction_name_strong_addr_missing",
    "route_exact_name",
    "route_exact_address",
    "route_numeric_anchors",
    "route_rare_tokens",
    "num_routes_retrieved",
    "same_country",
]


class MultiPassCandidateGenerator:
    """Combines high-recall blocking routes with country partitioning."""

    def __init__(self, max_candidates_per_key: int = 50) -> None:
        self.max_candidates_per_key = max_candidates_per_key
        self.exact_name = ExactNameBlocking(country_partition=False, max_candidates_per_key=max_candidates_per_key)
        self.exact_addr = ExactAddressBlocking(country_partition=False, max_candidates_per_key=max_candidates_per_key)
        self.exact_joint = ExactNameAndAddressBlocking(max_candidates_per_key=max_candidates_per_key)
        self.numeric = NumericAnchorBlocking(country_partition=False, max_candidates_per_key=max_candidates_per_key)
        self.rare_tokens = RareTokenBlocking(min_doc_freq=2, max_doc_freq=35, max_candidates=25)

    def build_indexes(self, target_records: Iterable[NormalizedRecord]) -> None:
        """Build all blocking indexes over target records in a single streaming pass."""
        logger.info("Building multi-pass blocking indexes (streaming 1-pass)...")
        t0 = time.time()
        name_idx = defaultdict(list)
        addr_idx = defaultdict(list)
        num_idx = defaultdict(list)
        rare_idx: Dict[str, Optional[List[str]]] = {}

        count = 0
        for rec in target_records:
            count += 1
            eid = rec.entity_id

            # 1. Exact name
            if rec.business_name_norm:
                nl = name_idx[rec.business_name_norm]
                if len(nl) < self.max_candidates_per_key:
                    nl.append(eid)

            # 2. Exact address
            if rec.business_address_norm:
                al = addr_idx[rec.business_address_norm]
                if len(al) < self.max_candidates_per_key:
                    al.append(eid)

            # 3. Numeric anchors
            for num in rec.numeric_anchors:
                num_l = num_idx[num]
                if len(num_l) < self.max_candidates_per_key:
                    num_l.append(eid)

            # 4. Rare tokens
            for tok in rec.name_tokens:
                if len(tok) >= 4 and not tok.isdigit():
                    if tok in rare_idx:
                        rl = rare_idx[tok]
                        if rl is not None:
                            if len(rl) < 25:
                                rl.append(eid)
                            else:
                                rare_idx[tok] = None
                    else:
                        rare_idx[tok] = [eid]

        self.exact_name._index = name_idx
        self.exact_addr._index = addr_idx
        self.numeric._index = num_idx
        self.rare_tokens._index = {
            k: v for k, v in rare_idx.items() if v is not None and len(v) >= 2
        }
        logger.info(f"Built all blocking indexes over {count:,} targets in {time.time()-t0:.2f}s")

    def generate_candidates_for_record(self, s1_record: NormalizedRecord) -> Dict[str, CandidatePair]:
        """Generate candidate target IDs and provenance metadata for a single S1 record."""
        cand_pairs: Dict[str, CandidatePair] = {}

        # 1. Exact name
        for tid in self.exact_name.query(s1_record):
            if tid not in cand_pairs:
                cand_pairs[tid] = CandidatePair(s1_record.entity_id, tid)
            cand_pairs[tid].add_route("exact_name", 1.0)

        # 2. Exact address
        for tid in self.exact_addr.query(s1_record):
            if tid not in cand_pairs:
                cand_pairs[tid] = CandidatePair(s1_record.entity_id, tid)
            cand_pairs[tid].add_route("exact_address", 1.0)

        # 3. Numeric anchors
        for tid in self.numeric.query(s1_record):
            if tid not in cand_pairs:
                cand_pairs[tid] = CandidatePair(s1_record.entity_id, tid)
            cand_pairs[tid].add_route("numeric_anchors", 0.9)

        # 4. Rare tokens
        for tid in self.rare_tokens.query(s1_record):
            if tid not in cand_pairs:
                cand_pairs[tid] = CandidatePair(s1_record.entity_id, tid)
            cand_pairs[tid].add_route("rare_tokens", 0.8)

        return cand_pairs


def extract_pairwise_feature_vector(
    s1_rec: NormalizedRecord,
    tgt_rec: NormalizedRecord,
    cand_pair: CandidatePair,
) -> List[float]:
    """Compute 28-dimensional numeric feature vector for a candidate pair."""
    nf = compute_name_features(s1_rec, tgt_rec)
    af = compute_address_features(s1_rec, tgt_rec)
    inter = compute_interaction_features(nf, af)

    same_country = 1.0 if (s1_rec.country and tgt_rec.country and s1_rec.country == tgt_rec.country) else 0.0

    return [
        nf["name_exact"],
        nf["name_token_jaccard"],
        nf["name_token_overlap"],
        nf["name_edit_similarity"],
        nf["name_char_ngram_similarity"],
        nf["name_length_diff"],
        nf["name_length_ratio"],
        af["addr_exact"],
        af["addr_token_jaccard"],
        af["addr_token_overlap"],
        af["addr_edit_similarity"],
        af["addr_numeric_overlap"],
        af["addr_missing_s1"],
        af["addr_missing_target"],
        af["addr_missing_either"],
        inter["interaction_product"],
        inter["interaction_min"],
        inter["interaction_mean"],
        inter["interaction_both_strong"],
        inter["interaction_name_strong_addr_weak"],
        inter["interaction_addr_strong_name_weak"],
        inter["interaction_name_strong_addr_missing"],
        1.0 if "exact_name" in cand_pair.routes else 0.0,
        1.0 if "exact_address" in cand_pair.routes else 0.0,
        1.0 if "numeric_anchors" in cand_pair.routes else 0.0,
        1.0 if "rare_tokens" in cand_pair.routes else 0.0,
        float(len(cand_pair.routes)),
        same_country,
    ]
