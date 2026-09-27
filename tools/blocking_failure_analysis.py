#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Blocking Failure Decomposition & Candidate Generation Engine

Performs deep analysis of all 10,339 blocking misses on the validation set,
evaluates existing block size distributions and truncation losses,
tests high-recall targeted blocker designs, and measures final F0.5 impact.
"""

import csv
import gc
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
    MultiPassCandidateGenerator,
    NormalizedRecord,
    compute_per_entity_f05,
    decide_matches_for_entity,
    evaluate_candidate_recall,
    evaluate_predictions,
    extract_pairwise_feature_vector,
    load_ground_truth,
    load_source_records,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("blocking_decomposition")


def main():
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    train_dir = Path(paths_cfg["train_dir"])
    artifact_dir = Path(paths_cfg["artifact_dir"])
    models_dir = artifact_dir / "models"
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "gbdt_matcher.joblib"
    matcher = GBDTMatcher.load(model_path)

    # 1. Load Validation Data exactly as in audit_validation.py
    logger.info("Loading S1 records...")
    exp_cfg = config.get("experiment", {})
    train_sample_limit = exp_cfg.get("train_s1_sample", 60000)
    val_sample_limit = exp_cfg.get("val_s1_sample", 15000)
    val_ratio = exp_cfg.get("val_split_ratio", 0.20)
    seed = exp_cfg.get("seed", 42)

    s1_all = load_source_records(train_dir / "train_source1.tsv", max_rows=train_sample_limit)
    s1_keys = list(s1_all.keys())
    rng = np.random.default_rng(seed)
    shuffled_keys = list(s1_keys)
    rng.shuffle(shuffled_keys)

    n_val = min(val_sample_limit, int(len(shuffled_keys) * val_ratio))
    val_s1_ids = set(shuffled_keys[:n_val])

    logger.info(f"Loaded {len(s1_all):,} S1 records. Validation S1 set = {len(val_s1_ids):,} entities.")

    # 2. Load Ground Truth
    gt_all = load_ground_truth(train_dir / "train_ground_truth.tsv", filter_s1_ids=set(s1_keys))
    val_gt = {k: gt_all.get(k, set()) for k in val_s1_ids}
    total_true_links = sum(len(v) for v in val_gt.values())
    logger.info(f"Total true validation links: {total_true_links:,}")

    needed_positive_ids = set()
    for s1_id in val_s1_ids:
        needed_positive_ids |= val_gt.get(s1_id, set())

    # 3. Load Targets (300k S2 + 300k S3 + missing positives)
    logger.info("Loading target records...")
    s2_records = load_source_records(train_dir / "train_source2.tsv", max_rows=300000)
    s3_records = load_source_records(train_dir / "train_source3.tsv", max_rows=300000)
    all_targets: Dict[str, NormalizedRecord] = {**s2_records, **s3_records}

    missing_pos_ids = needed_positive_ids - set(all_targets.keys())
    if missing_pos_ids:
        logger.info(f"Loading {len(missing_pos_ids):,} specific true target matches into memory...")
        for fname in ["train_source2.tsv", "train_source3.tsv"]:
            if not missing_pos_ids:
                break
            with (train_dir / fname).open("r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r, None)
                for row in r:
                    if len(row) >= 4 and row[0] in missing_pos_ids:
                        rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                        all_targets[rec.entity_id] = rec
                        missing_pos_ids.remove(rec.entity_id)

    logger.info(f"Target catalog ready: {len(all_targets):,} target records.")

    # 4. Detailed Block Size Analysis & Baseline Candidate Generation
    logger.info("============================================================")
    logger.info("PHASE 1: EXISTING BLOCKER SIZES & BASELINE RETRIEVAL")
    logger.info("============================================================")
    
    # We will build un-truncated indexes to measure true block sizes,
    # and also simulate the truncated candidate generation.
    block_cfg = config.get("blocking", {})
    max_candidates_per_key = block_cfg.get("max_candidates_per_key", 60)
    max_candidates_per_s1 = block_cfg.get("max_candidates_per_s1", 50)
    min_rare_freq = block_cfg.get("min_rare_freq", 1)
    max_rare_freq = block_cfg.get("max_rare_freq", 50)
    min_addr_len = block_cfg.get("min_addr_len", 8)

    targets_by_country: Dict[str, List[NormalizedRecord]] = defaultdict(list)
    for r in all_targets.values():
        targets_by_country[r.country].append(r)

    # Compute target token frequencies per country
    token_freqs: Dict[str, Dict[str, int]] = {}
    for country, tgt_list in targets_by_country.items():
        tf = defaultdict(int)
        for r in tgt_list:
            for t in set(r.name_tokens):
                if t not in COMMON_STOP_TOKENS and len(t) >= 3:
                    tf[t] += 1
        token_freqs[country] = tf

    # Build unconstrained & constrained indexes to measure sizes and truncation loss
    # Per country indexes:
    # route -> dict(key -> list of eids)
    raw_indexes: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    truncated_indexes: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    routes = ["exact_name", "sorted_name", "exact_addr", "numeric_anchor", "rare_token", "char_prefix"]

    for country, tgt_list in targets_by_country.items():
        raw_idx = {rt: defaultdict(list) for rt in routes}
        trunc_idx = {rt: defaultdict(list) for rt in routes}
        tf = token_freqs[country]

        for r in tgt_list:
            eid = r.entity_id
            # 1. exact name
            if r.name_norm:
                raw_idx["exact_name"][r.name_norm].append(eid)
                if len(trunc_idx["exact_name"][r.name_norm]) < max_candidates_per_key:
                    trunc_idx["exact_name"][r.name_norm].append(eid)

            # 2. sorted name
            if r.name_sorted != r.name_norm:
                raw_idx["sorted_name"][r.name_sorted].append(eid)
                if len(trunc_idx["sorted_name"][r.name_sorted]) < max_candidates_per_key:
                    trunc_idx["sorted_name"][r.name_sorted].append(eid)

            # 3. exact addr
            if not r.addr_missing and len(r.addr_norm) >= min_addr_len:
                raw_idx["exact_addr"][r.addr_norm].append(eid)
                if len(trunc_idx["exact_addr"][r.addr_norm]) < max_candidates_per_key:
                    trunc_idx["exact_addr"][r.addr_norm].append(eid)

            # 4. numeric anchor
            if r.name_tokens and r.addr_numbers:
                num_key = f"{r.name_tokens[0]}_{r.addr_numbers[0]}"
                raw_idx["numeric_anchor"][num_key].append(eid)
                if len(trunc_idx["numeric_anchor"][num_key]) < max_candidates_per_key:
                    trunc_idx["numeric_anchor"][num_key].append(eid)

            # 5. rare token (rarest token)
            best_t = None
            best_f = 999999
            for t in r.name_tokens:
                f = tf.get(t, 0)
                if min_rare_freq <= f <= max_rare_freq:
                    if f < best_f:
                        best_f = f
                        best_t = t
            if best_t:
                raw_idx["rare_token"][best_t].append(eid)
                if len(trunc_idx["rare_token"][best_t]) < max_candidates_per_key:
                    trunc_idx["rare_token"][best_t].append(eid)

            # 6. char prefix
            if r.name_norm and r.addr_numbers and len(r.name_norm) >= 4:
                prefix_key = f"{r.name_norm[:4]}_{r.addr_numbers[0]}"
                raw_idx["char_prefix"][prefix_key].append(eid)
                if len(trunc_idx["char_prefix"][prefix_key]) < max_candidates_per_key:
                    trunc_idx["char_prefix"][prefix_key].append(eid)

        raw_indexes[country] = raw_idx
        truncated_indexes[country] = trunc_idx

    # Block size statistics across all countries
    block_size_stats = {}
    for rt in routes:
        all_sizes = []
        for country in targets_by_country:
            for k, elist in raw_indexes[country][rt].items():
                all_sizes.append(len(elist))
        all_sizes = np.array(all_sizes, dtype=np.int32)
        if len(all_sizes) > 0:
            block_size_stats[rt] = {
                "total_keys": int(len(all_sizes)),
                "p50": float(np.percentile(all_sizes, 50)),
                "p90": float(np.percentile(all_sizes, 90)),
                "p95": float(np.percentile(all_sizes, 95)),
                "p99": float(np.percentile(all_sizes, 99)),
                "p99.9": float(np.percentile(all_sizes, 99.9)),
                "max": int(np.max(all_sizes)),
                "keys_truncated": int(np.sum(all_sizes > max_candidates_per_key)),
                "pct_keys_truncated": float(np.mean(all_sizes > max_candidates_per_key) * 100),
            }
        else:
            block_size_stats[rt] = {}

    logger.info("Existing Blocker Block-Size Distribution (Raw Unconstrained):")
    for rt, stats in block_size_stats.items():
        logger.info(f"  {rt:15s}: Keys={stats['total_keys']:,}, P50={stats['p50']:.1f}, P90={stats['p90']:.1f}, P99={stats['p99']:.1f}, Max={stats['max']}, TruncatedKeys={stats['keys_truncated']} ({stats['pct_keys_truncated']:.2f}%)")

    # Generate baseline candidates and track:
    # 1. Retrieved candidates before per-S1 cap
    # 2. Retrieved candidates after per-S1 cap
    # 3. Exactly which route retrieved which true matches
    # 4. True matches lost specifically to:
    #    - key truncation (target was in raw_idx[key] but not in trunc_idx[key])
    #    - S1 cap (candidate pair was in uncapped cands, but dropped by top-50 cap)
    
    val_cands_capped: Dict[str, Set[str]] = defaultdict(set)
    val_cands_uncapped: Dict[str, Set[str]] = defaultdict(set)
    val_cands_raw_no_trunc: Dict[str, Set[str]] = defaultdict(set)
    route_true_hits: Dict[str, int] = defaultdict(int)

    for s1_id in val_s1_ids:
        s1 = s1_all[s1_id]
        country = s1.country
        trunc_idx = truncated_indexes.get(country, {})
        raw_idx = raw_indexes.get(country, {})
        tf = token_freqs.get(country, {})
        true_tgts = val_gt.get(s1_id, set())

        # Uncapped candidate map: tid -> route_count
        cands_uncapped: Dict[str, int] = defaultdict(int)
        cands_raw: Set[str] = set()

        # 1. exact name
        if s1.name_norm:
            for tid in trunc_idx.get("exact_name", {}).get(s1.name_norm, []):
                cands_uncapped[tid] += 1
                if tid in true_tgts:
                    route_true_hits["exact_name"] += 1
            for tid in raw_idx.get("exact_name", {}).get(s1.name_norm, []):
                cands_raw.add(tid)

        # 2. sorted name
        if s1.name_sorted != s1.name_norm:
            for tid in trunc_idx.get("sorted_name", {}).get(s1.name_sorted, []):
                cands_uncapped[tid] += 1
                if tid in true_tgts:
                    route_true_hits["sorted_name"] += 1
            for tid in raw_idx.get("sorted_name", {}).get(s1.name_sorted, []):
                cands_raw.add(tid)

        # 3. exact addr
        if not s1.addr_missing and len(s1.addr_norm) >= min_addr_len:
            for tid in trunc_idx.get("exact_addr", {}).get(s1.addr_norm, []):
                cands_uncapped[tid] += 1
                if tid in true_tgts:
                    route_true_hits["exact_addr"] += 1
            for tid in raw_idx.get("exact_addr", {}).get(s1.addr_norm, []):
                cands_raw.add(tid)

        # 4. numeric anchor
        if s1.name_tokens and s1.addr_numbers:
            num_key = f"{s1.name_tokens[0]}_{s1.addr_numbers[0]}"
            for tid in trunc_idx.get("numeric_anchor", {}).get(num_key, []):
                cands_uncapped[tid] += 1
                if tid in true_tgts:
                    route_true_hits["numeric_anchor"] += 1
            for tid in raw_idx.get("numeric_anchor", {}).get(num_key, []):
                cands_raw.add(tid)

        # 5. rare tokens
        for t in s1.name_tokens:
            if t not in COMMON_STOP_TOKENS and t in trunc_idx.get("rare_token", {}):
                for tid in trunc_idx["rare_token"][t]:
                    cands_uncapped[tid] += 1
                    if tid in true_tgts:
                        route_true_hits["rare_token"] += 1
            if t not in COMMON_STOP_TOKENS and t in raw_idx.get("rare_token", {}):
                for tid in raw_idx["rare_token"][t]:
                    cands_raw.add(tid)

        # 6. char prefix
        if s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 4:
            prefix_key = f"{s1.name_norm[:4]}_{s1.addr_numbers[0]}"
            for tid in trunc_idx.get("char_prefix", {}).get(prefix_key, []):
                cands_uncapped[tid] += 1
                if tid in true_tgts:
                    route_true_hits["char_prefix"] += 1
            for tid in raw_idx.get("char_prefix", {}).get(prefix_key, []):
                cands_raw.add(tid)

        val_cands_uncapped[s1_id] = set(cands_uncapped.keys())
        val_cands_raw_no_trunc[s1_id] = cands_raw

        # Apply per-S1 cap (max_candidates_per_s1 = 50)
        if len(cands_uncapped) > max_candidates_per_s1:
            sorted_items = sorted(cands_uncapped.items(), key=lambda x: x[1], reverse=True)
            val_cands_capped[s1_id] = set(k for k, _ in sorted_items[:max_candidates_per_s1])
        else:
            val_cands_capped[s1_id] = set(cands_uncapped.keys())

    # Verify baseline candidate coverage
    covered_links_capped = 0
    covered_links_uncapped = 0
    covered_links_raw = 0
    missed_pairs_capped = []

    for s1_id, true_set in val_gt.items():
        cov_capped = true_set & val_cands_capped[s1_id]
        cov_uncapped = true_set & val_cands_uncapped[s1_id]
        cov_raw = true_set & val_cands_raw_no_trunc[s1_id]

        covered_links_capped += len(cov_capped)
        covered_links_uncapped += len(cov_uncapped)
        covered_links_raw += len(cov_raw)

        for tid in (true_set - val_cands_capped[s1_id]):
            missed_pairs_capped.append((s1_id, tid))

    logger.info(f"Baseline Candidate Coverage:")
    logger.info(f"  Total true links: {total_true_links:,}")
    logger.info(f"  Candidate covered (capped at 50/S1): {covered_links_capped:,} ({covered_links_capped/total_true_links*100:.2f}%)")
    logger.info(f"  Candidate covered (uncapped per S1): {covered_links_uncapped:,} ({covered_links_uncapped/total_true_links*100:.2f}%)")
    logger.info(f"  Candidate covered (raw, no key truncation): {covered_links_raw:,} ({covered_links_raw/total_true_links*100:.2f}%)")
    logger.info(f"  Total missed true links: {len(missed_pairs_capped):,}")

    loss_to_s1_cap = covered_links_uncapped - covered_links_capped
    loss_to_key_trunc = covered_links_raw - covered_links_uncapped
    logger.info(f"  Matches lost specifically to per-S1 cap (50): {loss_to_s1_cap:,} ({loss_to_s1_cap/len(missed_pairs_capped)*100:.2f}% of misses)")
    logger.info(f"  Matches lost specifically to key truncation (60): {loss_to_key_trunc:,} ({loss_to_key_trunc/len(missed_pairs_capped)*100:.2f}% of misses)")

    # ============================================================
    # PHASE 2: FAILURE DECOMPOSITION ON ALL 10,339 MISSED LINKS
    # ============================================================
    logger.info("============================================================")
    logger.info(f"PHASE 2: DEEP DECOMPOSITION OF ALL {len(missed_pairs_capped):,} MISSED PAIRS")
    logger.info("============================================================")

    # Extended legal suffixes
    EXT_LEGAL = {"gmbh", "bv", "sa", "sas", "sarl", "spa", "pllc", "llc", "llp", "ltd", "inc", "corp", "co", "pvt"}

    # Common business abbreviations
    ABBREV_PAIRS = {
        ("mgmt", "management"), ("intl", "international"), ("tech", "technology"), ("technologies", "tech"),
        ("serv", "services"), ("svc", "services"), ("svcs", "services"), ("dept", "department"),
        ("univ", "university"), ("inst", "institute"), ("assoc", "associates"), ("ctr", "center"),
        ("clnc", "clinic"), ("grp", "group"), ("mfg", "manufacturing"), ("distr", "distribution"),
        ("hosp", "hospital"), ("med", "medical"), ("lab", "laboratory"), ("labs", "laboratories"),
        ("fed", "federal"), ("natl", "national"), ("comm", "communications"), ("corp", "corporation"),
    }
    abbrev_dict = {}
    for a, b in ABBREV_PAIRS:
        abbrev_dict[a] = b
        abbrev_dict[b] = a

    category_counts = Counter()
    pairwise_analytics = []

    # Potential retrievability counts for missed pairs
    retrievability_counts = Counter()

    for s1_id, tgt_id in missed_pairs_capped:
        s1 = s1_all[s1_id]
        tgt = all_targets[tgt_id]

        s1_nt = set(s1.name_tokens)
        tgt_nt = set(tgt.name_tokens)
        shared_tokens = s1_nt & tgt_nt
        tok_jaccard = len(shared_tokens) / len(s1_nt | tgt_nt) if (s1_nt or tgt_nt) else 0.0
        tok_overlap = len(shared_tokens) / min(len(s1_nt), len(tgt_nt)) if (s1_nt and tgt_nt) else 0.0

        name_ratio = fuzz.ratio(s1.name_norm, tgt.name_norm)
        name_sort_ratio = fuzz.token_sort_ratio(s1.name_norm, tgt.name_norm)
        
        addr_ratio = fuzz.ratio(s1.addr_norm, tgt.addr_norm) if (not s1.addr_missing and not tgt.addr_missing) else 0.0
        addr_num_s1 = set(s1.addr_numbers)
        addr_num_tgt = set(tgt.addr_numbers)
        shared_nums = addr_num_s1 & addr_num_tgt
        num_jaccard = len(shared_nums) / len(addr_num_s1 | addr_num_tgt) if (addr_num_s1 or addr_num_tgt) else 0.0

        # Classification into A-R
        reasons = set()

        # M. country restriction
        if s1.country != tgt.country:
            reasons.add("M_country_restriction")

        # O. candidate top-K / truncation restriction (retrieved uncapped, dropped by S1 cap)
        if tgt_id in val_cands_uncapped[s1_id]:
            reasons.add("O_candidate_topK_cap_restriction")

        # N. candidate block size / control restriction (in raw index, dropped by key cap)
        if tgt_id in val_cands_raw_no_trunc[s1_id] and tgt_id not in val_cands_uncapped[s1_id]:
            reasons.add("N_block_size_truncation_restriction")

        # H. missing target address
        if tgt.addr_missing:
            reasons.add("H_missing_target_address")

        # I. missing useful numeric anchor
        if not s1.addr_numbers or not tgt.addr_numbers or not shared_nums:
            reasons.add("I_missing_useful_numeric_anchor")

        # P. no useful shared token
        if len(shared_tokens) == 0:
            reasons.add("P_no_useful_shared_token")

        # C. word-order mismatch
        if (s1.name_sorted == tgt.name_sorted or tok_jaccard >= 0.8) and s1.name_norm != tgt.name_norm:
            reasons.add("C_word_order_mismatch")

        # G. typo / edit-distance mismatch
        if 75 <= name_ratio < 100:
            reasons.add("G_typo_edit_distance_mismatch")

        # A. name normalization mismatch
        if name_ratio >= 85 and s1.name_norm != tgt.name_norm:
            reasons.add("A_name_normalization_mismatch")

        # B. address normalization mismatch
        if not s1.addr_missing and not tgt.addr_missing and 75 <= addr_ratio < 100:
            reasons.add("B_address_normalization_mismatch")

        # D. abbreviation mismatch
        has_abbrev = False
        for t1 in s1_nt:
            if t1 in abbrev_dict and abbrev_dict[t1] in tgt_nt:
                has_abbrev = True
                break
        if has_abbrev:
            reasons.add("D_abbreviation_mismatch")

        # E. legal-suffix mismatch
        s1_suff = s1.name_tokens[-1] if s1.name_tokens else ""
        tgt_suff = tgt.name_tokens[-1] if tgt.name_tokens else ""
        if (s1_suff in EXT_LEGAL or tgt_suff in EXT_LEGAL) and (s1_suff != tgt_suff):
            reasons.add("E_legal_suffix_mismatch")

        # F. transliteration mismatch
        if s1.raw_name != tgt.raw_name and s1.country in ("FR", "IN") and name_ratio >= 70:
            reasons.add("F_transliteration_mismatch")

        # J. generic / common name
        tf = token_freqs.get(s1.country, {})
        is_generic = bool(s1.name_tokens) and all(t in COMMON_STOP_TOKENS or tf.get(t, 0) > 50 for t in s1.name_tokens)
        if is_generic:
            reasons.add("J_generic_common_name")

        # K. rare-token route failed
        # Shares a token, but neither had it indexed (e.g. freq not in [1, 50] or wasn't rarest)
        if shared_tokens and "N_block_size_truncation_restriction" not in reasons and "O_candidate_topK_cap_restriction" not in reasons:
            reasons.add("K_rare_token_route_failed")

        # L. character-prefix route failed
        if s1.name_norm[:4] != tgt.name_norm[:4] or not shared_nums:
            reasons.add("L_char_prefix_route_failed")

        # Q. multiple-field weak similarity
        if name_ratio < 60 and addr_ratio < 60:
            reasons.add("Q_multiple_field_weak_similarity")

        if not reasons:
            reasons.add("R_other")

        for r_cat in reasons:
            category_counts[r_cat] += 1

        # ============================================================
        # Retrievability analysis on missed true link
        # ============================================================
        n1 = s1.name_norm
        n2 = tgt.name_norm

        # char prefix 2
        if len(n1) >= 2 and len(n2) >= 2 and n1[:2] == n2[:2]:
            retrievability_counts["char_prefix_2"] += 1

        # char prefix 3
        if len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3]:
            retrievability_counts["char_prefix_3"] += 1

        # char prefix 4
        if len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4]:
            retrievability_counts["char_prefix_4"] += 1

        # char prefix 5
        if len(n1) >= 5 and len(n2) >= 5 and n1[:5] == n2[:5]:
            retrievability_counts["char_prefix_5"] += 1

        # token prefix (first token)
        if s1.name_tokens and tgt.name_tokens and s1.name_tokens[0] == tgt.name_tokens[0]:
            retrievability_counts["token_prefix"] += 1

        # suffix anchor (last token)
        if s1.name_tokens and tgt.name_tokens and s1.name_tokens[-1] == tgt.name_tokens[-1]:
            retrievability_counts["suffix_anchor"] += 1

        # numeric anchor (any shared address number)
        if shared_nums:
            retrievability_counts["any_shared_number"] += 1

        # normalized token intersection (any shared token >= 3 chars not in stop words)
        valid_shared = [t for t in shared_tokens if t not in COMMON_STOP_TOKENS and len(t) >= 3]
        if valid_shared:
            retrievability_counts["normalized_token_intersection"] += 1

        # rare token expanded (any shared token with freq <= 200)
        has_rare_200 = any(tf.get(t, 9999) <= 200 for t in valid_shared)
        if has_rare_200:
            retrievability_counts["rare_token_freq_le_200"] += 1

        has_rare_500 = any(tf.get(t, 9999) <= 500 for t in valid_shared)
        if has_rare_500:
            retrievability_counts["rare_token_freq_le_500"] += 1

        # first 3 chars + primary number
        if len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3] and shared_nums:
            retrievability_counts["prefix3_plus_number"] += 1

        # first token + any shared number
        if s1.name_tokens and tgt.name_tokens and s1.name_tokens[0] == tgt.name_tokens[0] and shared_nums:
            retrievability_counts["token_prefix_plus_number"] += 1

        # Record pairwise analytics
        pairwise_analytics.append({
            "s1_id": s1_id,
            "tgt_id": tgt_id,
            "country": s1.country,
            "source": tgt.source,
            "s1_name_len": len(s1.name_norm),
            "tgt_name_len": len(tgt.name_norm),
            "s1_addr_len": len(s1.addr_norm),
            "tgt_addr_len": len(tgt.addr_norm),
            "s1_addr_missing": s1.addr_missing,
            "tgt_addr_missing": tgt.addr_missing,
            "name_ratio": name_ratio,
            "name_sort_ratio": name_sort_ratio,
            "addr_ratio": addr_ratio,
            "token_jaccard": tok_jaccard,
            "token_overlap": tok_overlap,
            "shared_tokens": list(shared_tokens),
            "shared_nums": list(shared_nums),
            "reasons": list(reasons),
        })

    logger.info("Blocking Failure Categories among 10,339 Missed True Links:")
    total_misses = len(missed_pairs_capped)
    for cat, count in category_counts.most_common():
        pct = count / total_misses * 100
        logger.info(f"  {cat:38s}: {count:5,d} ({pct:5.2f}%)")

    logger.info("Retrievability Analysis on Missed True Links:")
    for method, count in retrievability_counts.most_common():
        pct = count / total_misses * 100
        logger.info(f"  {method:32s}: {count:5,d} ({pct:5.2f}%)")

    # Save Phase 1 and Phase 2 reports to JSON
    decomposition_report = {
        "total_true_links": total_true_links,
        "covered_links_baseline": covered_links_capped,
        "missed_links_baseline": total_misses,
        "block_size_stats": block_size_stats,
        "route_true_hits": dict(route_true_hits),
        "loss_to_s1_cap": loss_to_s1_cap,
        "loss_to_key_trunc": loss_to_key_trunc,
        "category_counts": dict(category_counts),
        "retrievability_counts": dict(retrievability_counts),
    }

    with open(reports_dir / "blocking_decomposition.json", "w", encoding="utf-8") as f:
        json.dump(decomposition_report, f, indent=2)

    logger.info("Decomposition report written to artifacts/reports/blocking_decomposition.json")


if __name__ == "__main__":
    main()
