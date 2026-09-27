#!/usr/bin/env python3
"""HYBRID-V7-99-FAST — Amazon ML Challenge 2026 Business Entity Resolution.

Production pipeline implementing:
- Multi-view conservative normalization with legal suffix extraction
- 11-route multi-pass candidate generation with provenance tracking
- OPTIONAL GPU-ACCELERATED SEMANTIC SEARCH (MiniLM + FAISS) when available
- 39-dimensional pairwise + interaction + target-frequency features
- Two-stage LightGBM matcher with hard-negative mining
- Local S1-centered cross-source witness evidence
- Precision-controlled entity-level decision with singleton protection
- Streaming country-partitioned test inference (US, India, France)

Usage:
    python hybrid_v7_99_fast.py --mode full
    python hybrid_v7_99_fast.py --mode train [--sample N]
    python hybrid_v7_99_fast.py --mode test
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

# --- DYNAMIC GPU/SEMANTIC IMPORTS ---
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# ==============================================================================
# Configuration
# ==============================================================================

TRAIN_DIR = Path("data/raw/train")
TEST_DIR = Path("data/raw/test")
ARTIFACT_DIR = Path("artifacts")
OUTPUT_DIR = Path("output")
VALIDATOR_SCRIPT = Path("challenge/validate_submission.py")

SEED = 42
VAL_RATIO = 0.20

# Blocking configuration
MAX_CANDIDATES_PER_KEY = 100
MAX_CANDIDATES_PER_S1 = 80
MIN_RARE_FREQ = 1
MAX_RARE_FREQ = 100
MAX_ADDR_RARE_FREQ = 40
MIN_ADDR_LEN = 8
SEMANTIC_TOP_K = 5
SEMANTIC_THRESHOLD = 0.65

# Model configuration
N_ESTIMATORS = 300
LEARNING_RATE = 0.07
MAX_DEPTH = 7
NUM_LEAVES = 63
SUBSAMPLE = 0.8
COLSAMPLE = 0.8

# Decision configuration
BASE_THRESHOLD = 0.50
SINGLETON_THRESHOLD = 0.28
MAX_SCORE_GAP = 0.38
MAX_MATCHES = 15

BATCH_SIZE = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hybrid_v7")

# ==============================================================================
# Legal suffix and normalization maps
# ==============================================================================

LEGAL_SUFFIX_MAP = {
    "corporation": "corp", "incorporated": "inc", "company": "co",
    "limited": "ltd", "private": "pvt", "llc": "llc", "llp": "llp",
    "l.l.c.": "llc", "l.l.p.": "llp", "corp.": "corp", "inc.": "inc",
    "co.": "co", "ltd.": "ltd", "pvt.": "pvt",
    "sarl": "sarl", "s.a.r.l.": "sarl", "s.a.r.l": "sarl",
    "sas": "sas", "s.a.s.": "sas", "s.a.s": "sas",
    "sasu": "sasu", "sa": "sa", "s.a.": "sa",
    "eurl": "eurl", "e.u.r.l.": "eurl",
    "sci": "sci", "s.c.i.": "sci",
    "snc": "snc", "s.n.c.": "snc",
    "selarl": "selarl", "sel": "sel", "ei": "ei",
    "opc": "opc", "micro-entreprise": "me",
    "auto-entrepreneur": "ae",
}

LEGAL_FORMS = frozenset(LEGAL_SUFFIX_MAP.values())

ADDRESS_ABBREV = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "floor": "fl",
    "suite": "ste", "apartment": "apt", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "rue": "r", "chemin": "chem", "place": "pl",
    "nagar": "ngr", "marg": "mg", "cross": "cr",
}

COMMON_STOP_TOKENS = frozenset({
    "ltd", "pvt", "inc", "corp", "co", "llc", "llp", "the", "and",
    "group", "services", "center", "clinic", "care", "health",
    "medical", "hospital", "associates", "enterprises", "solutions",
    "management", "international", "company", "limited", "private",
    "of", "for", "in", "at", "by", "to", "de", "la", "le", "du", "des",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "et",
})

# ==============================================================================
# Normalization
# ==============================================================================

RE_NON_ALPHANUM = re.compile(r"[^\w\s]", re.UNICODE)
RE_DIGITS = re.compile(r"\b\d+\b")

def clean_unicode(text: str) -> str:
    if not text: return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))

def normalize_business_name(name: Optional[str]) -> str:
    if not name or not isinstance(name, str): return ""
    text = clean_unicode(name).lower()
    text = RE_NON_ALPHANUM.sub(" ", text)
    return " ".join([LEGAL_SUFFIX_MAP.get(t, t) for t in text.split() if t])

def strip_legal_suffix(name_norm: str) -> str:
    tokens = name_norm.split()
    while tokens and tokens[-1] in LEGAL_FORMS:
        tokens.pop()
    while tokens and tokens[0] in ("the", "les", "la", "le"):
        tokens.pop(0)
    return " ".join(tokens) if tokens else name_norm

def extract_legal_form(name_norm: str) -> str:
    tokens = name_norm.split()
    forms = [t for t in tokens if t in LEGAL_FORMS]
    return forms[-1] if forms else ""

def normalize_address(addr: Optional[str]) -> str:
    if not addr or not isinstance(addr, str): return ""
    text = clean_unicode(addr).lower()
    text = RE_NON_ALPHANUM.sub(" ", text)
    return " ".join([ADDRESS_ABBREV.get(t, t) for t in text.split() if t])

@dataclass(slots=True, frozen=True)
class Rec:
    eid: str
    name_norm: str
    name_sorted: str
    name_core: str
    name_core_sorted: str
    name_tokens: tuple
    legal_form: str
    addr_norm: str
    addr_tokens: tuple
    addr_numbers: tuple
    addr_missing: bool
    country: str
    source: str

    @classmethod
    def from_row(cls, eid: str, name: str, addr: str, country: str) -> "Rec":
        src = "S1" if eid.startswith("S1-") else ("S2" if eid.startswith("S2-") else "S3")
        c = country.strip().upper() if isinstance(country, str) else ""

        n_norm = normalize_business_name(name)
        n_toks = tuple(t for t in n_norm.split() if t)
        n_sorted = " ".join(sorted(n_toks)) if len(n_toks) > 1 else n_norm

        n_core = strip_legal_suffix(n_norm)
        n_core_toks = tuple(n_core.split())
        n_core_sorted = " ".join(sorted(n_core_toks)) if len(n_core_toks) > 1 else n_core
        legal = extract_legal_form(n_norm)

        a_str = addr if isinstance(addr, str) and addr.strip() else ""
        a_missing = not bool(a_str)
        a_norm = normalize_address(a_str)
        a_toks = tuple(t for t in a_norm.split() if t)
        a_nums = tuple(n for n in RE_DIGITS.findall(a_norm) if len(n) >= 2)

        return cls(
            eid=eid.strip(), name_norm=n_norm, name_sorted=n_sorted,
            name_core=n_core, name_core_sorted=n_core_sorted,
            name_tokens=n_toks, legal_form=legal,
            addr_norm=a_norm, addr_tokens=a_toks, addr_numbers=a_nums,
            addr_missing=a_missing, country=c, source=src,
        )

# ==============================================================================
# Semantic Search Engine (GPU Accelerated, Highly Scalable)
# ==============================================================================

class SemanticEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing SemanticEngine on {self.device}...")
        # all-MiniLM-L6-v2 is ultra-fast, robust, and outputs 384d normalized embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.index = None
        self.unique_names = []
        self.name_to_eids = defaultdict(list)

    def build_index(self, records: Dict[str, Rec]):
        logger.info("Building Semantic Index (Deduplicated Names)...")
        t0 = time.time()
        
        # 1. Deduplicate by exact normalized name to save 60%+ memory and time
        for r in records.values():
            if r.name_norm and len(r.name_norm) >= 3:
                self.name_to_eids[r.name_norm].append(r.eid)
                
        self.unique_names = list(self.name_to_eids.keys())
        logger.info(f"Reduced {len(records):,} targets to {len(self.unique_names):,} unique names.")
        
        # 2. Encode on GPU/CPU
        batch_size = 1024 if self.device == 'cuda' else 128
        embeddings = self.model.encode(self.unique_names, batch_size=batch_size, 
                                       show_progress_bar=True, normalize_embeddings=True)
        
        # 3. Build FAISS Index (IVFFlat for speed on millions of vectors)
        d = embeddings.shape[1]
        nlist = min(4096, max(1, len(self.unique_names) // 500))
        quantizer = faiss.IndexFlatIP(d)
        
        # If GPU available, move FAISS index to GPU for blindingly fast training/search
        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            
        logger.info(f"Training FAISS index (nlist={nlist})...")
        train_sample = embeddings
        if len(embeddings) > 1000000:
            idx = np.random.choice(len(embeddings), 1000000, replace=False)
            train_sample = embeddings[idx]
            
        self.index.train(train_sample)
        self.index.add(embeddings)
        self.index.nprobe = 32
        
        logger.info(f"Semantic index built in {time.time()-t0:.1f}s.")

    def search_batch(self, s1_records: List[Rec], top_k=SEMANTIC_TOP_K) -> Dict[str, Set[str]]:
        """Returns dict: s1_eid -> set of matched target_eids"""
        results = {r.eid: set() for r in s1_records}
        valid_indices = [i for i, r in enumerate(s1_records) if r.name_norm and len(r.name_norm) >= 3]
        
        if not valid_indices:
            return results
            
        query_names = [s1_records[i].name_norm for i in valid_indices]
        batch_size = 1024 if self.device == 'cuda' else 128
        q_embs = self.model.encode(query_names, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        
        scores, indices = self.index.search(q_embs, top_k)
        
        for i, original_idx in enumerate(valid_indices):
            s1_eid = s1_records[original_idx].eid
            for rank in range(top_k):
                idx = indices[i][rank]
                if idx != -1 and scores[i][rank] >= SEMANTIC_THRESHOLD:
                    matched_name = self.unique_names[idx]
                    for tid in self.name_to_eids[matched_name]:
                        results[s1_eid].add(tid)
                        
        return results

# ==============================================================================
# Lexical Blocking Engine — 11-Route Multi-Pass
# ==============================================================================

class BlockingEngine:
    def __init__(self):
        self.exact_name: Dict[str, List[str]] = defaultdict(list)
        self.sorted_name: Dict[str, List[str]] = defaultdict(list)
        self.exact_addr: Dict[str, List[str]] = defaultdict(list)
        self.numeric_anchor: Dict[str, List[str]] = defaultdict(list)
        self.rare_token: Dict[str, List[str]] = defaultdict(list)
        self.char_prefix: Dict[str, List[str]] = defaultdict(list)
        self.core_name: Dict[str, List[str]] = defaultdict(list)
        self.core_name_sorted: Dict[str, List[str]] = defaultdict(list)
        self.addr_token_num: Dict[str, List[str]] = defaultdict(list)
        self.prefix3_num: Dict[str, List[str]] = defaultdict(list)
        self.first5: Dict[str, List[str]] = defaultdict(list)
        self.token_freq: Dict[str, int] = {}
        self.addr_token_freq: Dict[str, int] = {}
        self.target_name_freq: Counter = Counter()

    def build_indexes(self, records: Dict[str, Rec]) -> None:
        t0 = time.time()
        gc.disable()
        try:
            tf: Dict[str, int] = defaultdict(int)
            atf: Dict[str, int] = defaultdict(int)
            for r in records.values():
                seen, seen_a = set(), set()
                for t in r.name_tokens:
                    if t not in COMMON_STOP_TOKENS and len(t) >= 2 and t not in seen:
                        tf[t] += 1; seen.add(t)
                for t in r.addr_tokens:
                    if t not in COMMON_STOP_TOKENS and len(t) >= 3 and t not in seen_a:
                        atf[t] += 1; seen_a.add(t)
            self.token_freq = dict(tf)
            self.addr_token_freq = dict(atf)

            cap = MAX_CANDIDATES_PER_KEY
            for r in records.values():
                eid = r.eid
                if r.name_norm:
                    self.target_name_freq[r.name_norm] += 1
                    if len(self.exact_name[r.name_norm]) < cap: self.exact_name[r.name_norm].append(eid)
                if r.name_sorted and r.name_sorted != r.name_norm:
                    if len(self.sorted_name[r.name_sorted]) < cap: self.sorted_name[r.name_sorted].append(eid)
                if not r.addr_missing and len(r.addr_norm) >= MIN_ADDR_LEN:
                    if len(self.exact_addr[r.addr_norm]) < cap: self.exact_addr[r.addr_norm].append(eid)
                if r.name_tokens and r.addr_numbers:
                    key = f"{r.name_tokens[0]}_{r.addr_numbers[0]}"
                    if len(self.numeric_anchor[key]) < cap: self.numeric_anchor[key].append(eid)
                for t in r.name_tokens:
                    if MIN_RARE_FREQ <= tf.get(t, 0) <= MAX_RARE_FREQ and t not in COMMON_STOP_TOKENS and len(t) >= 2:
                        if len(self.rare_token[t]) < cap: self.rare_token[t].append(eid)
                if r.name_norm and r.addr_numbers and len(r.name_norm) >= 4:
                    key = f"{r.name_norm[:4]}_{r.addr_numbers[0]}"
                    if len(self.char_prefix[key]) < cap: self.char_prefix[key].append(eid)
                if r.name_core and r.name_core != r.name_norm:
                    if len(self.core_name[r.name_core]) < cap: self.core_name[r.name_core].append(eid)
                if r.name_core_sorted and r.name_core_sorted != r.name_core:
                    if len(self.core_name_sorted[r.name_core_sorted]) < cap: self.core_name_sorted[r.name_core_sorted].append(eid)
                if r.name_tokens and r.addr_tokens:
                    for at in r.addr_tokens:
                        if 1 <= atf.get(at, 0) <= MAX_ADDR_RARE_FREQ and len(at) >= 3:
                            for an in r.addr_numbers[:1]:
                                key = f"{at}_{an}"
                                if len(self.addr_token_num[key]) < cap: self.addr_token_num[key].append(eid)
                            break
                if r.name_norm and r.addr_numbers and len(r.name_norm) >= 3:
                    for num in r.addr_numbers[:2]:
                        key = f"{r.name_norm[:3]}_{num}"
                        if len(self.prefix3_num[key]) < cap: self.prefix3_num[key].append(eid)
                if r.name_norm and len(r.name_norm) >= 5:
                    key = r.name_norm[:5]
                    if len(self.first5[key]) < cap: self.first5[key].append(eid)
        finally:
            gc.enable()
            gc.collect()
        logger.info(f"Lexical Blocking indexes built in {time.time()-t0:.1f}s.")

    def generate_candidates(self, s1: Rec, semantic_matches: Set[str] = None) -> Dict[str, int]:
        cands: Dict[str, int] = defaultdict(int)
        def add(ids):
            for tid in ids: cands[tid] += 1

        if s1.name_norm: add(self.exact_name.get(s1.name_norm, []))
        if s1.name_sorted and s1.name_sorted != s1.name_norm: add(self.sorted_name.get(s1.name_sorted, []))
        if not s1.addr_missing and len(s1.addr_norm) >= MIN_ADDR_LEN: add(self.exact_addr.get(s1.addr_norm, []))
        if s1.name_tokens and s1.addr_numbers: add(self.numeric_anchor.get(f"{s1.name_tokens[0]}_{s1.addr_numbers[0]}", []))
        for t in s1.name_tokens:
            if t not in COMMON_STOP_TOKENS and t in self.rare_token: add(self.rare_token[t])
        if s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 4: add(self.char_prefix.get(f"{s1.name_norm[:4]}_{s1.addr_numbers[0]}", []))
        if s1.name_core and s1.name_core != s1.name_norm: add(self.core_name.get(s1.name_core, []))
        if s1.name_core_sorted and s1.name_core_sorted != s1.name_core: add(self.core_name_sorted.get(s1.name_core_sorted, []))
        if s1.name_tokens and s1.addr_tokens:
            for at in s1.addr_tokens:
                if 1 <= self.addr_token_freq.get(at, 0) <= MAX_ADDR_RARE_FREQ and len(at) >= 3:
                    for an in s1.addr_numbers[:1]: add(self.addr_token_num.get(f"{at}_{an}", []))
                    break
        if s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 3:
            for num in s1.addr_numbers[:2]: add(self.prefix3_num.get(f"{s1.name_norm[:3]}_{num}", []))
        if s1.name_norm and len(s1.name_norm) >= 5: add(self.first5.get(s1.name_norm[:5], []))
        
        # Inject Semantic Matches safely
        if semantic_matches:
            for tid in semantic_matches:
                cands[tid] += 1

        if len(cands) > MAX_CANDIDATES_PER_S1:
            top = sorted(cands.items(), key=lambda x: x[1], reverse=True)[:MAX_CANDIDATES_PER_S1]
            cands = dict(top)
        return dict(cands)

# ==============================================================================
# Feature Engineering (39 Dimensions)
# ==============================================================================

FEATURE_NAMES = [
    "name_exact", "name_jaccard", "name_overlap", "name_edit",
    "name_sort_ratio", "name_token_set", "name_partial",
    "name_len_diff", "name_len_ratio", "name_containment",
    "name_prefix_match", "name_core_edit",
    "addr_exact", "addr_jaccard", "addr_overlap", "addr_edit",
    "addr_num_jaccard", "addr_num_overlap",
    "addr_miss_tgt", "addr_miss_either",
    "inter_prod", "inter_min", "inter_mean",
    "inter_both_high", "inter_name_hi_addr_lo",
    "inter_addr_hi_name_lo", "inter_name_hi_addr_miss",
    "is_s2", "is_s3", "same_country",
    "route_count", "has_exact_route", "has_rare_route", "has_addr_route",
    "has_semantic_route",
    "target_name_freq_log", "legal_form_match",
]

def extract_features(s1: Rec, tgt: Rec, route_count: int, 
                     target_name_freq: int = 1, is_semantic: bool = False) -> List[float]:
    name_exact = 1.0 if (s1.name_norm and s1.name_norm == tgt.name_norm) else 0.0
    s1_nt, tgt_nt = set(s1.name_tokens), set(tgt.name_tokens)
    inter_n, union_n = len(s1_nt & tgt_nt), len(s1_nt | tgt_nt)
    name_jaccard = inter_n / union_n if union_n else 0.0
    name_overlap = inter_n / min(len(s1_nt), len(tgt_nt)) if (s1_nt and tgt_nt) else 0.0
    name_containment = inter_n / len(s1_nt) if s1_nt else 0.0
    name_edit = fuzz.ratio(s1.name_norm, tgt.name_norm) / 100.0 if (s1.name_norm and tgt.name_norm) else 0.0
    name_sort = fuzz.token_sort_ratio(s1.name_norm, tgt.name_norm) / 100.0 if (s1.name_norm and tgt.name_norm) else 0.0
    name_tset = fuzz.token_set_ratio(s1.name_norm, tgt.name_norm) / 100.0 if (s1.name_norm and tgt.name_norm) else 0.0
    name_partial = fuzz.partial_ratio(s1.name_norm, tgt.name_norm) / 100.0 if (s1.name_norm and tgt.name_norm) else 0.0
    name_len_diff = float(abs(len(s1.name_norm) - len(tgt.name_norm)))
    name_len_ratio = float(min(len(s1.name_norm), len(tgt.name_norm))) / float(max(len(s1.name_norm), len(tgt.name_norm), 1))
    name_prefix = 1.0 if (len(s1.name_norm) >= 3 and len(tgt.name_norm) >= 3 and s1.name_norm[:3] == tgt.name_norm[:3]) else 0.0
    name_core_edit = fuzz.ratio(s1.name_core, tgt.name_core) / 100.0 if (s1.name_core and tgt.name_core) else 0.0

    addr_exact = 1.0 if (not s1.addr_missing and not tgt.addr_missing and s1.addr_norm == tgt.addr_norm) else 0.0
    s1_at, tgt_at = set(s1.addr_tokens), set(tgt.addr_tokens)
    inter_a, union_a = len(s1_at & tgt_at), len(s1_at | tgt_at)
    addr_jaccard = inter_a / union_a if union_a else 0.0
    addr_overlap = inter_a / min(len(s1_at), len(tgt_at)) if (s1_at and tgt_at) else 0.0
    addr_edit = fuzz.ratio(s1.addr_norm, tgt.addr_norm) / 100.0 if (not s1.addr_missing and not tgt.addr_missing) else 0.0
    
    s1_nums, tgt_nums = set(s1.addr_numbers), set(tgt.addr_numbers)
    num_inter, num_union = len(s1_nums & tgt_nums), len(s1_nums | tgt_nums)
    num_jaccard = num_inter / num_union if num_union else 0.0
    num_overlap = num_inter / min(len(s1_nums), len(tgt_nums)) if (s1_nums and tgt_nums) else 0.0
    addr_miss_tgt = 1.0 if tgt.addr_missing else 0.0
    addr_miss_either = 1.0 if (s1.addr_missing or tgt.addr_missing) else 0.0

    prod_sim, min_sim, mean_sim = name_edit * addr_edit, min(name_edit, addr_edit), (name_edit + addr_edit) / 2.0
    both_high = 1.0 if (name_edit > 0.80 and addr_edit > 0.80) else 0.0
    name_hi_addr_lo = 1.0 if (name_edit > 0.85 and addr_edit < 0.35 and not tgt.addr_missing) else 0.0
    addr_hi_name_lo = 1.0 if (addr_edit > 0.85 and name_edit < 0.35) else 0.0
    name_hi_addr_miss = 1.0 if (name_edit > 0.80 and tgt.addr_missing) else 0.0

    is_s2 = 1.0 if tgt.source == "S2" else 0.0
    is_s3 = 1.0 if tgt.source == "S3" else 0.0
    same_c = 1.0 if s1.country == tgt.country else 0.0
    has_exact = 1.0 if (name_exact > 0.5 or addr_exact > 0.5) else 0.0
    has_rare = 1.0 if route_count >= 2 else 0.0
    has_addr = 1.0 if addr_edit > 0.5 else 0.0
    
    tf_log = math.log1p(target_name_freq)
    legal_match = 1.0 if (s1.legal_form and tgt.legal_form and s1.legal_form == tgt.legal_form) else 0.0

    return [
        name_exact, name_jaccard, name_overlap, name_edit, name_sort, name_tset, name_partial,
        name_len_diff, name_len_ratio, name_containment, name_prefix, name_core_edit,
        addr_exact, addr_jaccard, addr_overlap, addr_edit, num_jaccard, num_overlap,
        addr_miss_tgt, addr_miss_either, prod_sim, min_sim, mean_sim,
        both_high, name_hi_addr_lo, addr_hi_name_lo, name_hi_addr_miss,
        is_s2, is_s3, same_c, float(min(route_count, 8)), has_exact, has_rare, has_addr,
        1.0 if is_semantic else 0.0, tf_log, legal_match,
    ]

# ==============================================================================
# Pipeline Methods (Train / Test / Profile)
# ==============================================================================

def load_records_streaming(path: Path, country_filter: str = None) -> Dict[str, Rec]:
    records = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                if country_filter and row[3].strip().upper() != country_filter: continue
                rec = Rec.from_row(row[0], row[1], row[2], row[3])
                records[rec.eid] = rec
    return records

def decide_matches(scored: List[Tuple[str, float, str]], base_thresh=BASE_THRESHOLD, 
                   single_thresh=SINGLETON_THRESHOLD, gap=MAX_SCORE_GAP, max_m=MAX_MATCHES) -> List[str]:
    if not scored: return []
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    if scored_sorted[0][1] < single_thresh: return []
    has_witness = any(s[2] == "S2" and s[1] >= 0.40 for s in scored_sorted) and any(s[2] == "S3" and s[1] >= 0.40 for s in scored_sorted)
    selected = []
    for tid, score, src in scored_sorted:
        if (scored_sorted[0][1] - score) > gap: continue
        eff_thresh = (base_thresh - 0.06) if has_witness else base_thresh
        if score >= eff_thresh:
            selected.append(tid)
            if len(selected) >= max_m: break
    return selected

def run_train(sample_size: int = 200000):
    logger.info("=" * 70)
    logger.info(f"HYBRID-V7-99-FAST TRAINING (Semantic={'ON' if SEMANTIC_AVAILABLE else 'OFF'})")
    logger.info("=" * 70)
    models_dir, reports_dir = ARTIFACT_DIR / "hybrid_v7_models", ARTIFACT_DIR / "hybrid_v7_reports"
    models_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)

    s1_all = load_records_streaming(TRAIN_DIR / "train_source1.tsv")
    s1_keys = list(s1_all.keys())
    np.random.default_rng(SEED).shuffle(s1_keys)
    if sample_size: s1_keys = s1_keys[:sample_size]

    n_val = min(15000, int(len(s1_keys) * VAL_RATIO))
    val_ids, train_ids = set(s1_keys[:n_val]), set(s1_keys[n_val:])
    
    gt = {}
    with (TRAIN_DIR / "train_ground_truth.tsv").open() as f:
        reader = csv.reader(f, delimiter="\t"); next(reader, None)
        for row in reader:
            if row and row[0].strip() in set(s1_keys):
                gt[row[0].strip()] = {m.strip() for m in row[1].split(",")} if len(row) > 1 and row[1].strip() else set()
                
    targets = {}
    sampled_countries = {s1_all[sid].country for sid in s1_keys}
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        logger.info(f"Loading {fname}...")
        targets.update(load_records_streaming(TRAIN_DIR / fname))

    eng = BlockingEngine()
    eng.build_indexes(targets)
    
    semantic_eng = SemanticEngine() if SEMANTIC_AVAILABLE else None
    if semantic_eng: semantic_eng.build_index(targets)

    def process_s1_batch(s1_id_set):
        pairs_X, pairs_y, pairs_meta = [], [], []
        
        batch_size = 5000
        s1_list = list(s1_id_set)
        for i in tqdm(range(0, len(s1_list), batch_size), desc="Candidates"):
            batch_ids = s1_list[i:i+batch_size]
            batch_recs = [s1_all[sid] for sid in batch_ids]
            
            sem_results = semantic_eng.search_batch(batch_recs) if semantic_eng else {}
            
            for s1_r in batch_recs:
                sid = s1_r.eid
                sem_matches = sem_results.get(sid, set())
                cands = eng.generate_candidates(s1_r, sem_matches)
                true_set = gt.get(sid, set())
                
                for tid, rc in cands.items():
                    if tid in targets:
                        tgt = targets[tid]
                        is_sem = tid in sem_matches
                        feat = extract_features(s1_r, tgt, rc, eng.target_name_freq.get(tgt.name_norm, 1), is_sem)
                        pairs_X.append(feat); pairs_y.append(1 if tid in true_set else 0)
                        pairs_meta.append((sid, tid))
        return np.array(pairs_X, dtype=np.float32), np.array(pairs_y, dtype=np.int32), pairs_meta

    X_train, y_train, _ = process_s1_batch(train_ids)
    
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH, 
                               num_leaves=NUM_LEAVES, subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    
    logger.info("Hard-negative mining...")
    hard_mask = (y_train == 0) & (model.predict_proba(X_train)[:, 1] >= 0.28)
    if np.sum(hard_mask) > 0:
        model.fit(np.vstack([X_train, X_train[hard_mask]]), np.concatenate([y_train, y_train[hard_mask]]))
    
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, models_dir / "lgbm_hybrid_v7.joblib")
    logger.info("Training complete & saved.")

def run_test():
    logger.info("=" * 70)
    logger.info(f"HYBRID-V7-99-FAST TEST (Semantic={'ON' if SEMANTIC_AVAILABLE else 'OFF'})")
    logger.info("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model_data = joblib.load(ARTIFACT_DIR / "hybrid_v7_models/lgbm_hybrid_v7.joblib")
    model = model_data["model"]
    
    countries = set()
    with (TEST_DIR / "test_source1.tsv").open() as f:
        reader = csv.reader(f, delimiter="\t"); next(reader, None)
        for row in reader:
            if len(row) >= 4: countries.add(row[3].strip().upper())

    matching_dict, candidate_dict = {}, {}

    for country in sorted(countries):
        logger.info(f"\nProcessing {country}...")
        targets_c = {}
        for fname in ["test_source2.tsv", "test_source3.tsv"]:
            targets_c.update(load_records_streaming(TEST_DIR / fname, country))
            
        eng = BlockingEngine()
        eng.build_indexes(targets_c)
        
        semantic_eng = SemanticEngine() if SEMANTIC_AVAILABLE else None
        if semantic_eng: semantic_eng.build_index(targets_c)

        batch = []
        def flush(batch_records):
            sem_results = semantic_eng.search_batch(batch_records) if semantic_eng else {}
            pairs_to_score = []
            all_cands = {}
            for s1_r in batch_records:
                sem_matches = sem_results.get(s1_r.eid, set())
                cands = eng.generate_candidates(s1_r, sem_matches)
                all_cands[s1_r.eid] = cands
                for tid, rc in cands.items():
                    if tid in targets_c:
                        tgt = targets_c[tid]
                        feat = extract_features(s1_r, tgt, rc, eng.target_name_freq.get(tgt.name_norm, 1), tid in sem_matches)
                        pairs_to_score.append((s1_r.eid, tid, feat))
                        
            scored_by_s1 = defaultdict(list)
            if pairs_to_score:
                probs = model.predict_proba(np.array([p[2] for p in pairs_to_score], dtype=np.float32))[:, 1]
                for (sid, tid, _), prob in zip(pairs_to_score, probs):
                    scored_by_s1[sid].append((tid, float(prob), targets_c[tid].source))
                    
            for s1_r in batch_records:
                sid = s1_r.eid
                cand_ids = sorted(all_cands.get(sid, {}).keys())
                matches = decide_matches(scored_by_s1.get(sid, []))
                matches = [m for m in matches if m in cand_ids]
                matching_dict[sid] = ",".join(matches)
                candidate_dict[sid] = ",".join(cand_ids)

        with (TEST_DIR / "test_source1.tsv").open() as f:
            reader = csv.reader(f, delimiter="\t"); next(reader, None)
            for row in reader:
                if len(row) >= 4 and row[3].strip().upper() == country:
                    batch.append(Rec.from_row(row[0], row[1], row[2], row[3]))
                    if len(batch) >= BATCH_SIZE:
                        flush(batch); batch = []
            if batch: flush(batch)
            
        del targets_c, eng, semantic_eng
        gc.collect()

    logger.info("Writing output TSVs...")
    with (OUTPUT_DIR / "matching_results.tsv").open("w", newline="") as fm, \
         (OUTPUT_DIR / "candidate_pairs.tsv").open("w", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        with (TEST_DIR / "test_source1.tsv").open() as f:
            reader = csv.reader(f, delimiter="\t"); next(reader, None)
            for row in reader:
                sid = row[0].strip()
                fm.write(f"{sid}\t{matching_dict.get(sid, '')}\n")
                fc.write(f"{sid}\t{candidate_dict.get(sid, '')}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profile", "train", "test", "full"], default="full")
    parser.add_argument("--sample", type=int, default=200000)
    args = parser.parse_args()

    if args.mode in ("train", "full"): run_train(args.sample)
    if args.mode in ("test", "full"): run_test()

if __name__ == "__main__": main()
