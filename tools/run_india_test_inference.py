#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Production Memory-Bounded India Test Inference Pipeline

Executes India test inference across all 809,986 India test S1 records:
- Existing trained/validated LightGBM matcher (artifacts/models/gbdt_matcher.joblib)
- Validated Recommended Multi-Modal Candidate Blocker with STRICT early candidate cap = 30 per pass
- Precomputed target token sets for zero-allocation feature extraction
- Two-pass memory-bounded streaming (Pass A: S2 targets, Pass B: S3 targets)
- Native empirical LightGBM decision engine (base_th=0.46, single_th=0.32, gap=0.35, max_matches=15)
- Automated assembly of France + US + India in exact canonical test_source1.tsv order
- Submission validation via challenge/validate_submission.py
"""

import csv
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import yaml
from rapidfuzz import fuzz

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    COMMON_STOP_TOKENS,
    FEATURE_NAMES,
    CandidatePair,
    GBDTMatcher,
    NormalizedRecord,
    decide_matches_for_entity,
    format_id_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("india_inference")

DummyRecord = namedtuple("DummyRecord", ["source"])

LEGAL_SUFFIXES_EXT = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "gmbh", "bv", "sa",
    "sas", "sarl", "spa", "pllc", "limited", "corporation", "incorporated",
    "private", "company", "cie", "nv", "ag", "sl", "srl",
}

def strip_legal_suffixes(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    res = list(tokens)
    while res and res[-1] in LEGAL_SUFFIXES_EXT:
        res.pop()
    return tuple(res)


class CompactTargetRecord:
    __slots__ = (
        "entity_id", "name_norm", "name_sorted", "addr_norm", "addr_missing",
        "name_tokens", "addr_tokens", "addr_numbers",
        "name_tokens_set", "addr_tokens_set", "addr_numbers_set", "source"
    )
    def __init__(self, rec: NormalizedRecord):
        self.entity_id = rec.entity_id
        self.name_norm = rec.name_norm
        self.name_sorted = rec.name_sorted
        self.addr_norm = rec.addr_norm
        self.addr_missing = rec.addr_missing
        self.name_tokens = rec.name_tokens
        self.addr_tokens = rec.addr_tokens
        self.addr_numbers = rec.addr_numbers
        self.name_tokens_set = set(rec.name_tokens)
        self.addr_tokens_set = set(rec.addr_tokens)
        self.addr_numbers_set = set(rec.addr_numbers)
        self.source = rec.source


class RecommendedMultiModalBlockerStrict:
    """Validated high-recall multi-modal candidate generator with strict early-stopping."""

    def __init__(
        self,
        max_candidates_per_key: int = 60,
        max_candidates_per_s1: int = 30,
        min_rare_freq: int = 1,
        max_rare_freq: int = 50,
        min_addr_len: int = 8,
    ):
        self.max_candidates_per_key = max_candidates_per_key
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.min_rare_freq = min_rare_freq
        self.max_rare_freq = max_rare_freq
        self.min_addr_len = min_addr_len

        # Indices
        self.exact_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.sorted_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.exact_addr_idx: Dict[str, List[str]] = defaultdict(list)
        self.numeric_anchor_idx: Dict[str, List[str]] = defaultdict(list)
        self.single_rare_idx: Dict[str, List[str]] = defaultdict(list)
        self.char_prefix4_idx: Dict[str, List[str]] = defaultdict(list)

        # New validated additions
        self.all_rare_df50_idx: Dict[str, List[str]] = defaultdict(list)
        self.core_name_exact_idx: Dict[str, List[str]] = defaultdict(list)
        self.core_name_sorted_idx: Dict[str, List[str]] = defaultdict(list)
        self.prefix3_any_num_idx: Dict[str, List[str]] = defaultdict(list)
        self.addr_token_num_idx: Dict[str, List[str]] = defaultdict(list)

    def build_indexes(self, targets: List[CompactTargetRecord]) -> None:
        gc.disable()
        try:
            name_tf: Dict[str, int] = defaultdict(int)
            addr_tf: Dict[str, int] = defaultdict(int)

            for r in targets:
                for t in r.name_tokens_set:
                    if t not in COMMON_STOP_TOKENS and len(t) >= 3:
                        name_tf[t] += 1
                for t in r.addr_tokens_set:
                    if len(t) >= 4:
                        addr_tf[t] += 1

            cap = self.max_candidates_per_key
            for r in targets:
                eid = r.entity_id

                # 1. exact name
                if r.name_norm and len(self.exact_name_idx[r.name_norm]) < cap:
                    self.exact_name_idx[r.name_norm].append(eid)

                # 2. sorted name
                if r.name_sorted != r.name_norm and len(self.sorted_name_idx[r.name_sorted]) < cap:
                    self.sorted_name_idx[r.name_sorted].append(eid)

                # 3. exact addr
                if not r.addr_missing and len(r.addr_norm) >= self.min_addr_len:
                    if len(self.exact_addr_idx[r.addr_norm]) < cap:
                        self.exact_addr_idx[r.addr_norm].append(eid)

                # 4. numeric anchor
                if r.name_tokens and r.addr_numbers:
                    k = f"{r.name_tokens[0]}_{r.addr_numbers[0]}"
                    if len(self.numeric_anchor_idx[k]) < cap:
                        self.numeric_anchor_idx[k].append(eid)

                # 5. single rarest token
                best_t = None
                best_f = 999999
                for t in r.name_tokens:
                    f = name_tf.get(t, 0)
                    if self.min_rare_freq <= f <= self.max_rare_freq and f < best_f:
                        best_f = f
                        best_t = t
                if best_t and len(self.single_rare_idx[best_t]) < cap:
                    self.single_rare_idx[best_t].append(eid)

                # 6. char prefix 4 + primary number
                if r.name_norm and r.addr_numbers and len(r.name_norm) >= 4:
                    k = f"{r.name_norm[:4]}_{r.addr_numbers[0]}"
                    if len(self.char_prefix4_idx[k]) < cap:
                        self.char_prefix4_idx[k].append(eid)

                # 7. all_rare_df50
                for t in r.name_tokens_set:
                    f = name_tf.get(t, 0)
                    if 1 <= f <= 50 and len(self.all_rare_df50_idx[t]) < cap:
                        self.all_rare_df50_idx[t].append(eid)

                # 8 & 9. core name exact & sorted
                core_toks = strip_legal_suffixes(r.name_tokens)
                core_n = " ".join(core_toks)
                if core_n and core_n != r.name_norm and len(self.core_name_exact_idx[core_n]) < cap:
                    self.core_name_exact_idx[core_n].append(eid)

                core_s = " ".join(sorted(core_toks)) if len(core_toks) > 1 else core_n
                if core_s and core_s != r.name_sorted and core_s != core_n:
                    if len(self.core_name_sorted_idx[core_s]) < cap:
                        self.core_name_sorted_idx[core_s].append(eid)

                # 10. prefix 3 + any address number
                if len(r.name_norm) >= 3 and r.addr_numbers:
                    p3 = r.name_norm[:3]
                    for num in r.addr_numbers[:2]:
                        k = f"{p3}_{num}"
                        if len(self.prefix3_any_num_idx[k]) < cap:
                            self.prefix3_any_num_idx[k].append(eid)

                # 11. addr_token_num (rare address token df <= 40 + number)
                if r.addr_tokens and r.addr_numbers:
                    best_at = None
                    best_af = 999999
                    for at in r.addr_tokens:
                        af = addr_tf.get(at, 0)
                        if 1 <= af <= 40 and af < best_af:
                            best_af = af
                            best_at = at
                    if best_at:
                        k = f"{best_at}_{r.addr_numbers[0]}"
                        if len(self.addr_token_num_idx[k]) < cap:
                            self.addr_token_num_idx[k].append(eid)
        finally:
            gc.enable()
            del name_tf, addr_tf
            gc.collect()

    def generate_candidates(self, s1: NormalizedRecord) -> Dict[str, CandidatePair]:
        cands: Dict[str, CandidatePair] = {}
        cap = self.max_candidates_per_s1

        def add_c(tid: str, rt: str) -> bool:
            if tid not in cands:
                if len(cands) >= cap:
                    return False
                cands[tid] = CandidatePair(s1_id=s1.entity_id, target_id=tid)
            p = cands[tid]
            if rt == "exact_name" and p.route_exact_name == 0:
                p.route_exact_name = 1
                p.route_count += 1
            elif rt == "sorted_name" and p.route_sorted_name == 0:
                p.route_sorted_name = 1
                p.route_count += 1
            elif rt == "exact_addr" and p.route_exact_addr == 0:
                p.route_exact_addr = 1
                p.route_count += 1
            elif rt == "numeric_anchor" and p.route_numeric_anchor == 0:
                p.route_numeric_anchor = 1
                p.route_count += 1
            elif rt == "rare_token" and p.route_rare_token == 0:
                p.route_rare_token = 1
                p.route_count += 1
            elif rt == "char_ngram" and p.route_char_ngram == 0:
                p.route_char_ngram = 1
                p.route_count += 1
            else:
                p.route_count += 1
            return True

        # 1. exact name
        if s1.name_norm:
            for tid in self.exact_name_idx.get(s1.name_norm, []):
                if not add_c(tid, "exact_name"): break

        # 2. sorted name
        if len(cands) < cap and s1.name_sorted != s1.name_norm:
            for tid in self.sorted_name_idx.get(s1.name_sorted, []):
                if not add_c(tid, "sorted_name"): break

        # 3. exact addr
        if len(cands) < cap and not s1.addr_missing and len(s1.addr_norm) >= self.min_addr_len:
            for tid in self.exact_addr_idx.get(s1.addr_norm, []):
                if not add_c(tid, "exact_addr"): break

        # 4. numeric anchor
        if len(cands) < cap and s1.name_tokens and s1.addr_numbers:
            k = f"{s1.name_tokens[0]}_{s1.addr_numbers[0]}"
            for tid in self.numeric_anchor_idx.get(k, []):
                if not add_c(tid, "numeric_anchor"): break

        # 5. single rarest token
        if len(cands) < cap:
            for t in s1.name_tokens:
                if t not in COMMON_STOP_TOKENS and t in self.single_rare_idx:
                    for tid in self.single_rare_idx[t]:
                        if not add_c(tid, "rare_token"): break
                    if len(cands) >= cap: break

        # 6. char prefix 4
        if len(cands) < cap and s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 4:
            k = f"{s1.name_norm[:4]}_{s1.addr_numbers[0]}"
            for tid in self.char_prefix4_idx.get(k, []):
                if not add_c(tid, "char_ngram"): break

        # 7. core name exact & sorted
        if len(cands) < cap:
            core_toks = strip_legal_suffixes(s1.name_tokens)
            cn = " ".join(core_toks)
            if cn:
                for tid in self.core_name_exact_idx.get(cn, []):
                    if not add_c(tid, "exact_name"): break
            if len(cands) < cap:
                cs = " ".join(sorted(core_toks)) if len(core_toks) > 1 else cn
                if cs and cs != cn:
                    for tid in self.core_name_sorted_idx.get(cs, []):
                        if not add_c(tid, "sorted_name"): break

        # 8. prefix 3 any num
        if len(cands) < cap and len(s1.name_norm) >= 3 and s1.addr_numbers:
            p3 = s1.name_norm[:3]
            for num in s1.addr_numbers[:2]:
                k = f"{p3}_{num}"
                for tid in self.prefix3_any_num_idx.get(k, []):
                    if not add_c(tid, "char_ngram"): break
                if len(cands) >= cap: break

        # 9. all rare df 50
        if len(cands) < cap:
            for t in s1.name_tokens:
                if t not in COMMON_STOP_TOKENS and t in self.all_rare_df50_idx:
                    for tid in self.all_rare_df50_idx[t]:
                        if not add_c(tid, "rare_token"): break
                    if len(cands) >= cap: break

        # 10. addr token num
        if len(cands) < cap and s1.addr_tokens and s1.addr_numbers:
            for at in s1.addr_tokens:
                k = f"{at}_{s1.addr_numbers[0]}"
                for tid in self.addr_token_num_idx.get(k, []):
                    if not add_c(tid, "exact_addr"): break
                if len(cands) >= cap: break

        return cands


def extract_pairwise_feature_vector_fast(
    s1: NormalizedRecord,
    tgt: CompactTargetRecord,
    pair: CandidatePair,
    s1_nt: Set[str],
    s1_at: Set[str],
    s1_nums: Set[str],
) -> List[float]:
    """Ultra-fast pairwise feature extraction using precomputed target sets."""
    # 1. Name features
    name_exact = 1.0 if (s1.name_norm and s1.name_norm == tgt.name_norm) else 0.0
    tgt_nt = tgt.name_tokens_set

    if name_exact == 1.0:
        name_jaccard = 1.0
        name_overlap = 1.0
        name_edit = 1.0
        name_sort = 1.0
        name_len_diff = 0.0
        name_len_ratio = 1.0
    else:
        inter = len(s1_nt & tgt_nt)
        name_jaccard = inter / len(s1_nt | tgt_nt) if (s1_nt or tgt_nt) else 0.0
        name_overlap = inter / min(len(s1_nt), len(tgt_nt)) if (s1_nt and tgt_nt) else 0.0
        name_edit = fuzz.ratio(s1.name_norm, tgt.name_norm) / 100.0 if (s1.name_norm and tgt.name_norm) else 0.0
        name_sort = fuzz.token_sort_ratio(s1.name_norm, tgt.name_norm) / 100.0 if (s1.name_norm and tgt.name_norm) else 0.0
        name_len_diff = float(abs(len(s1.name_norm) - len(tgt.name_norm)))
        max_len = max(len(s1.name_norm), len(tgt.name_norm), 1)
        name_len_ratio = float(min(len(s1.name_norm), len(tgt.name_norm))) / float(max_len)

    # 2. Address features
    addr_exact = (
        1.0
        if (not s1.addr_missing and not tgt.addr_missing and s1.addr_norm == tgt.addr_norm)
        else 0.0
    )
    tgt_at = tgt.addr_tokens_set

    if addr_exact == 1.0:
        addr_jaccard = 1.0
        addr_overlap = 1.0
        addr_edit = 1.0
    elif s1.addr_missing or tgt.addr_missing:
        addr_jaccard = 0.0
        addr_overlap = 0.0
        addr_edit = 0.0
    else:
        inter_a = len(s1_at & tgt_at)
        addr_jaccard = inter_a / len(s1_at | tgt_at) if (s1_at or tgt_at) else 0.0
        addr_overlap = inter_a / min(len(s1_at), len(tgt_at)) if (s1_at and tgt_at) else 0.0
        addr_edit = fuzz.ratio(s1.addr_norm, tgt.addr_norm) / 100.0

    tgt_nums = tgt.addr_numbers_set
    inter_num = len(s1_nums & tgt_nums)
    num_jaccard = inter_num / len(s1_nums | tgt_nums) if (s1_nums or tgt_nums) else 0.0
    num_overlap = inter_num / min(len(s1_nums), len(tgt_nums)) if (s1_nums and tgt_nums) else 0.0

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
    r_exact_n = float(pair.route_exact_name)
    r_sort_n = float(pair.route_sorted_name)
    r_exact_a = float(pair.route_exact_addr)
    r_num_a = float(pair.route_numeric_anchor)
    r_rare_t = float(pair.route_rare_token)
    r_char_n = float(pair.route_char_ngram)
    r_count = float(pair.route_count)
    same_c = 1.0

    return [
        name_exact, name_jaccard, name_overlap, name_edit, name_sort, name_len_diff, name_len_ratio,
        addr_exact, addr_jaccard, addr_overlap, addr_edit, num_jaccard, num_overlap,
        addr_miss_s1, addr_miss_tgt, addr_miss_either,
        prod_sim, min_sim, mean_sim, both_high, name_hi_addr_lo, addr_hi_name_lo, name_hi_addr_miss,
        is_s2, is_s3, r_exact_n, r_sort_n, r_exact_a, r_num_a, r_rare_t, r_char_n, r_count, same_c
    ]


def run_india_pipeline():
    t_start = time.time()
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    test_dir = Path(paths_cfg["test_dir"])
    artifact_dir = Path(paths_cfg["artifact_dir"])
    output_dir = Path(paths_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = artifact_dir / "final_candidates"
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_source_path = artifact_dir / "models" / "gbdt_matcher.joblib"
    matcher = GBDTMatcher.load(model_source_path)
    logger.info(f"Loaded validated LightGBM matcher from {model_source_path}")

    test_s1_path = test_dir / "test_source1.tsv"
    test_s2_path = test_dir / "test_source2.tsv"
    test_s3_path = test_dir / "test_source3.tsv"

    base_thresh = 0.46
    single_thresh = 0.32
    max_score_gap = 0.35
    max_matches = 15
    batch_size = 10000

    path_s2_scored = temp_dir / "temp_s2_india.tsv"
    path_m = temp_dir / "temp_matching_india.tsv"
    path_c = temp_dir / "temp_candidates_india.tsv"

    # Clean intermediate files for India
    path_s2_scored.unlink(missing_ok=True)
    path_m.unlink(missing_ok=True)
    path_c.unlink(missing_ok=True)

    # ------------------------------------------------------------
    # PASS A: Process India S2 Targets (Cap = 30)
    # ------------------------------------------------------------
    logger.info("============================================================")
    logger.info("[INDIA] PASS A: Loading S2 targets from test_source2.tsv...")
    logger.info("============================================================")
    t0 = time.time()
    s2_targets: Dict[str, CompactTargetRecord] = {}
    with test_s2_path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            row = line.rstrip("\r\n").split("\t")
            if len(row) >= 4 and row[3].strip().upper() == "INDIA":
                rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                s2_targets[rec.entity_id] = CompactTargetRecord(rec)
    logger.info(f"[INDIA] Loaded {len(s2_targets):,} S2 targets in {time.time()-t0:.1f}s.")

    s2_blocker = RecommendedMultiModalBlockerStrict(max_candidates_per_key=60, max_candidates_per_s1=30)
    t_idx = time.time()
    s2_blocker.build_indexes(list(s2_targets.values()))
    logger.info(f"[INDIA] S2 blocker built in {time.time()-t_idx:.1f}s.")

    logger.info("[INDIA] PASS A: Streaming S1 entities against S2 targets...")
    t_pass_a_stream = time.time()
    with path_s2_scored.open("w", encoding="utf-8", newline="") as fs2:
        ws2 = csv.writer(fs2, delimiter="\t", lineterminator="\n")

        batch_s1: List[NormalizedRecord] = []
        s1_count_s2 = 0

        def flush_batch_s2(records: List[NormalizedRecord]):
            nonlocal s1_count_s2
            cand_pairs: List[Tuple[NormalizedRecord, CompactTargetRecord, CandidatePair, Set[str], Set[str], Set[str]]] = []
            c_map: Dict[str, List[str]] = {}

            for s1 in records:
                cands = s2_blocker.generate_candidates(s1)
                c_keys = list(cands.keys())
                c_map[s1.entity_id] = c_keys

                s1_nt = set(s1.name_tokens)
                s1_at = set(s1.addr_tokens)
                s1_nums = set(s1.addr_numbers)

                for tid, p in cands.items():
                    if tid in s2_targets:
                        cand_pairs.append((s1, s2_targets[tid], p, s1_nt, s1_at, s1_nums))

            scored_s2: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
            if cand_pairs:
                X_b = np.array(
                    [extract_pairwise_feature_vector_fast(s, t, cp, n, a, nums) for s, t, cp, n, a, nums in cand_pairs],
                    dtype=np.float32,
                )
                probs_b = matcher.predict_proba(X_b)
                for (s1_r, tgt_r, _, _, _, _), prob in zip(cand_pairs, probs_b):
                    scored_s2[s1_r.entity_id].append((tgt_r.entity_id, float(prob)))

            for s1 in records:
                s1_id = s1.entity_id
                c_list = c_map.get(s1_id, [])
                sc_list = scored_s2.get(s1_id, [])
                sc_str = ",".join(f"{tid}:{pr:.4f}" for tid, pr in sc_list)
                ws2.writerow([s1_id, format_id_list(c_list), sc_str])
                s1_count_s2 += 1

            if s1_count_s2 % 50000 == 0 or s1_count_s2 >= 800000:
                el = max(time.time() - t_pass_a_stream, 0.001)
                logger.info(f"[INDIA Pass A] Streamed {s1_count_s2:,} / 809,986 S1 ({s1_count_s2/el:.0f} ent/s)...")

        with test_s1_path.open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                row = line.rstrip("\r\n").split("\t")
                if len(row) >= 4 and row[3].strip().upper() == "INDIA":
                    rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                    batch_s1.append(rec)
                    if len(batch_s1) >= batch_size:
                        flush_batch_s2(batch_s1)
                        batch_s1 = []

            if batch_s1:
                flush_batch_s2(batch_s1)

    logger.info(f"[INDIA] PASS A completed: {s1_count_s2:,} entities scored in {time.time()-t_pass_a_stream:.1f}s.")
    del s2_targets, s2_blocker
    gc.collect()

    # ------------------------------------------------------------
    # PASS B: Process India S3 Targets (Cap = 30) & Synchronize
    # ------------------------------------------------------------
    logger.info("============================================================")
    logger.info("[INDIA] PASS B: Loading S3 targets from test_source3.tsv...")
    logger.info("============================================================")
    t0 = time.time()
    s3_targets: Dict[str, CompactTargetRecord] = {}
    with test_s3_path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            row = line.rstrip("\r\n").split("\t")
            if len(row) >= 4 and row[3].strip().upper() == "INDIA":
                rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                s3_targets[rec.entity_id] = CompactTargetRecord(rec)
    logger.info(f"[INDIA] Loaded {len(s3_targets):,} S3 targets in {time.time()-t0:.1f}s.")

    s3_blocker = RecommendedMultiModalBlockerStrict(max_candidates_per_key=60, max_candidates_per_s1=30)
    t_idx = time.time()
    s3_blocker.build_indexes(list(s3_targets.values()))
    logger.info(f"[INDIA] S3 blocker built in {time.time()-t_idx:.1f}s.")

    logger.info("[INDIA] PASS B: Streaming S1 entities & merging S2 + S3 candidates...")
    t_pass_b_stream = time.time()

    total_india_s1 = 0
    total_india_matches = 0
    total_india_cands = 0
    india_zero_matches = 0

    with path_m.open("w", encoding="utf-8", newline="") as fm, \
         path_c.open("w", encoding="utf-8", newline="") as fc, \
         path_s2_scored.open("r", encoding="utf-8") as fs2:

        wm = csv.writer(fm, delimiter="\t", lineterminator="\n")
        wc = csv.writer(fc, delimiter="\t", lineterminator="\n")

        batch_s1 = []
        batch_s2_lines = []

        def flush_batch_s3(records: List[NormalizedRecord], s2_lines: List[str]):
            nonlocal total_india_s1, total_india_matches, total_india_cands, india_zero_matches

            cand_pairs: List[Tuple[NormalizedRecord, CompactTargetRecord, CandidatePair, Set[str], Set[str], Set[str]]] = []
            c3_map: Dict[str, List[str]] = {}

            for s1 in records:
                cands = s3_blocker.generate_candidates(s1)
                c_keys = list(cands.keys())
                c3_map[s1.entity_id] = c_keys

                s1_nt = set(s1.name_tokens)
                s1_at = set(s1.addr_tokens)
                s1_nums = set(s1.addr_numbers)

                for tid, p in cands.items():
                    if tid in s3_targets:
                        cand_pairs.append((s1, s3_targets[tid], p, s1_nt, s1_at, s1_nums))

            scored_s3: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
            if cand_pairs:
                X_b = np.array(
                    [extract_pairwise_feature_vector_fast(s, t, cp, n, a, nums) for s, t, cp, n, a, nums in cand_pairs],
                    dtype=np.float32,
                )
                probs_b = matcher.predict_proba(X_b)
                for (s1_r, tgt_r, _, _, _, _), prob in zip(cand_pairs, probs_b):
                    scored_s3[s1_r.entity_id].append((tgt_r.entity_id, float(prob)))

            # Merge S2 + S3 for each S1 in batch
            for s1, s2_line in zip(records, s2_lines):
                s1_id = s1.entity_id
                parts = s2_line.rstrip("\r\n").split("\t")
                s2_c_list = parts[1].split(",") if len(parts) > 1 and parts[1] else []
                s2_scored_str = parts[2] if len(parts) > 2 else ""

                scored_candidates = []
                if s2_scored_str:
                    for item in s2_scored_str.split(","):
                        if ":" in item:
                            tid, sc_str = item.split(":", 1)
                            scored_candidates.append((tid, float(sc_str), DummyRecord(source="S2")))

                for tid, prob in scored_s3.get(s1_id, []):
                    scored_candidates.append((tid, prob, DummyRecord(source="S3")))

                combined_cands = set(s2_c_list) | set(c3_map.get(s1_id, []))
                final_cands_list = list(combined_cands)
                total_india_cands += len(final_cands_list)

                matches = decide_matches_for_entity(
                    scored_candidates,
                    base_threshold=base_thresh,
                    singleton_threshold=single_thresh,
                    max_score_gap=max_score_gap,
                    max_matches=max_matches,
                    enable_graph=True,
                )

                c_set = set(final_cands_list)
                valid_matches = [m for m in matches if m in c_set]

                wm.writerow([s1_id, format_id_list(valid_matches)])
                wc.writerow([s1_id, format_id_list(final_cands_list)])

                total_india_s1 += 1
                total_india_matches += len(valid_matches)
                if len(valid_matches) == 0:
                    india_zero_matches += 1

            if total_india_s1 % 50000 == 0 or total_india_s1 >= 800000:
                el = max(time.time() - t_pass_b_stream, 0.001)
                logger.info(
                    f"[INDIA Pass B] Processed {total_india_s1:,} / 809,986 ({total_india_s1/el:.0f} ent/s, "
                    f"matches={total_india_matches:,}, cands={total_india_cands:,})..."
                )

        with test_s1_path.open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                row = line.rstrip("\r\n").split("\t")
                if len(row) >= 4 and row[3].strip().upper() == "INDIA":
                    rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                    s2_line = fs2.readline()
                    batch_s1.append(rec)
                    batch_s2_lines.append(s2_line)

                    if len(batch_s1) >= batch_size:
                        flush_batch_s3(batch_s1, batch_s2_lines)
                        batch_s1 = []
                        batch_s2_lines = []

            if batch_s1:
                flush_batch_s3(batch_s1, batch_s2_lines)

    logger.info(
        f"[INDIA] Finished: {total_india_s1:,} entities, {total_india_matches:,} matches, "
        f"{total_india_cands:,} candidates in {time.time()-t_pass_b_stream:.1f}s."
    )
    del s3_targets, s3_blocker
    path_s2_scored.unlink(missing_ok=True)
    gc.collect()

    # ------------------------------------------------------------
    # 5. ASSEMBLE ALL 3 COUNTRIES IN CANONICAL TEST_SOURCE1 ORDER
    # ------------------------------------------------------------
    logger.info("============================================================")
    logger.info("ASSEMBLING FINAL CANONICAL TEST FILES ACROSS ALL 3 COUNTRIES")
    logger.info("============================================================")

    final_matching_path = output_dir / "matching_results.tsv"
    final_candidate_path = output_dir / "candidate_pairs.tsv"

    matching_files = {
        "FRANCE": temp_dir / "temp_matching_france.tsv",
        "US": temp_dir / "temp_matching_us.tsv",
        "INDIA": temp_dir / "temp_matching_india.tsv",
    }
    candidate_files = {
        "FRANCE": temp_dir / "temp_candidates_france.tsv",
        "US": temp_dir / "temp_candidates_us.tsv",
        "INDIA": temp_dir / "temp_candidates_india.tsv",
    }

    # Verify all files exist
    for c, p in matching_files.items():
        if not p.is_file():
            logger.error(f"Missing matching file for {c}: {p}")
            sys.exit(1)
    for c, p in candidate_files.items():
        if not p.is_file():
            logger.error(f"Missing candidate file for {c}: {p}")
            sys.exit(1)

    # Open readers for all 3 countries
    m_readers = {}
    c_readers = {}
    m_fps = {}
    c_fps = {}

    for c in ["FRANCE", "US", "INDIA"]:
        m_fps[c] = matching_files[c].open("r", encoding="utf-8")
        c_fps[c] = candidate_files[c].open("r", encoding="utf-8")
        m_readers[c] = csv.reader(m_fps[c], delimiter="\t")
        c_readers[c] = csv.reader(c_fps[c], delimiter="\t")

    total_rows = 0
    total_final_matches = 0
    total_final_candidates = 0
    zero_match_s1 = 0
    single_match_s1 = 0
    multi_match_s1 = 0
    cand_counts = []

    logger.info("Streaming assembled rows into output/matching_results.tsv & output/candidate_pairs.tsv...")
    with final_matching_path.open("w", encoding="utf-8", newline="") as fm, \
         final_candidate_path.open("w", encoding="utf-8", newline="") as fc, \
         test_s1_path.open("r", encoding="utf-8") as f_s1:

        wm = csv.writer(fm, delimiter="\t", lineterminator="\n")
        wc = csv.writer(fc, delimiter="\t", lineterminator="\n")

        wm.writerow(["source1_entity_id", "matched_entity_ids"])
        wc.writerow(["source1_entity_id", "candidate_entity_ids"])

        next(f_s1, None)
        for line in f_s1:
            row = line.rstrip("\r\n").split("\t")
            if len(row) < 4:
                continue
            s1_id = row[0].strip()
            country = row[3].strip().upper()

            if country not in m_readers:
                wm.writerow([s1_id, ""])
                wc.writerow([s1_id, ""])
                total_rows += 1
                zero_match_s1 += 1
                continue

            m_row = next(m_readers[country])
            c_row = next(c_readers[country])

            assert m_row[0] == s1_id, f"Matching ID mismatch: expected {s1_id}, got {m_row[0]}"
            assert c_row[0] == s1_id, f"Candidate ID mismatch: expected {s1_id}, got {c_row[0]}"

            m_str = m_row[1] if len(m_row) > 1 else ""
            c_str = c_row[1] if len(c_row) > 1 else ""

            wm.writerow([s1_id, m_str])
            wc.writerow([s1_id, c_str])

            m_list = m_str.split(",") if m_str else []
            c_list = c_str.split(",") if c_str else []

            total_rows += 1
            n_m = len(m_list)
            n_c = len(c_list)
            total_final_matches += n_m
            total_final_candidates += n_c

            if n_m == 0:
                zero_match_s1 += 1
            elif n_m == 1:
                single_match_s1 += 1
            else:
                multi_match_s1 += 1

            if len(cand_counts) < 100000:
                cand_counts.append(n_c)

    for c in ["FRANCE", "US", "INDIA"]:
        m_fps[c].close()
        c_fps[c].close()

    logger.info(f"Assembly completed: {total_rows:,} total rows written in {time.time()-t_start:.1f}s.")
    logger.info(f"Total Matches: {total_final_matches:,} (mean: {total_final_matches/total_rows:.2f}/S1)")
    logger.info(f"Total Candidates: {total_final_candidates:,} (mean: {total_final_candidates/total_rows:.2f}/S1)")
    logger.info(f"Zero-Match S1: {zero_match_s1:,} ({zero_match_s1/total_rows*100:.1f}%)")
    logger.info(f"Single-Match S1: {single_match_s1:,} ({single_match_s1/total_rows*100:.1f}%)")
    logger.info(f"Multi-Match S1: {multi_match_s1:,} ({multi_match_s1/total_rows*100:.1f}%)")

    # ------------------------------------------------------------
    # 6. OFFICIAL VALIDATION
    # ------------------------------------------------------------
    logger.info("============================================================")
    logger.info("RUNNING OFFICIAL CHALLENGE VALIDATION SCRIPT")
    logger.info("============================================================")
    val_script = REPO_ROOT / "challenge" / "validate_submission.py"
    if val_script.is_file():
        cmd = [sys.executable, str(val_script), "--test-dir", str(test_dir), "--submission-dir", str(output_dir)]
        logger.info(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"Validator return code: {res.returncode}")
        logger.info(f"Validator output:\n{res.stdout}")
        if res.stderr:
            logger.warning(f"Validator stderr:\n{res.stderr}")
    else:
        logger.warning(f"Validator script not found at {val_script}")

    logger.info(f"PRODUCTION TEST INFERENCE 100% COMPLETE in {time.time()-t_start:.1f}s!")

if __name__ == "__main__":
    run_india_pipeline()
