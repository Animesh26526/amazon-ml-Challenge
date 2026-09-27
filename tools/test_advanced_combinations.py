#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Advanced Blocker Combinations & Model Evaluation

Tests the combination of the highest-yield, high-precision candidate blockers:
- addr_token_num (Rare address token + house number)
- all_rare_tokens_df50 (All name tokens with df <= 50)
- core_name_exact (Suffix-stripped exact name)
- core_name_sorted (Suffix-stripped sorted name)
- prefix_3_any_num (3-char name prefix + address number)

Measures:
1. Block size distributions for all new blockers (P50, P90, P95, P99, Max, Truncation)
2. Incremental candidate recall & candidate set statistics (avg, median, P95, P99, max)
3. End-to-end model evaluation:
   - Features extraction using existing 33 pairwise features
   - GBDT scoring with existing trained LightGBM model
   - Decision engine with missing-address relief rule:
     target.addr_missing == True and name_edit >= 0.85 -> threshold relief
   - Final Macro F0.5, Precision, Recall
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
    CandidatePair,
    GBDTMatcher,
    NormalizedRecord,
    compute_per_entity_f05,
    decide_matches_for_entity,
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
logger = logging.getLogger("advanced_blocker_eval")

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

    exp_cfg = config.get("experiment", {})
    train_sample_limit = exp_cfg.get("train_s1_sample", 60000)
    val_sample_limit = exp_cfg.get("val_s1_sample", 15000)
    val_ratio = exp_cfg.get("val_split_ratio", 0.20)
    seed = exp_cfg.get("seed", 42)

    logger.info("Loading S1 and Ground Truth...")
    s1_all = load_source_records(train_dir / "train_source1.tsv", max_rows=train_sample_limit)
    s1_keys = list(s1_all.keys())
    rng = np.random.default_rng(seed)
    shuffled_keys = list(s1_keys)
    rng.shuffle(shuffled_keys)

    n_val = min(val_sample_limit, int(len(shuffled_keys) * val_ratio))
    val_s1_ids = set(shuffled_keys[:n_val])

    gt_all = load_ground_truth(train_dir / "train_ground_truth.tsv", filter_s1_ids=set(s1_keys))
    val_gt = {k: gt_all.get(k, set()) for k in val_s1_ids}
    total_true_links = sum(len(v) for v in val_gt.values())
    logger.info(f"Loaded {len(val_s1_ids):,} Val S1 entities. Total true links: {total_true_links:,}")

    needed_positive_ids = set()
    for s1_id in val_s1_ids:
        needed_positive_ids |= val_gt.get(s1_id, set())

    # Load targets
    logger.info("Loading target records...")
    s2_records = load_source_records(train_dir / "train_source2.tsv", max_rows=300000)
    s3_records = load_source_records(train_dir / "train_source3.tsv", max_rows=300000)
    all_targets: Dict[str, NormalizedRecord] = {**s2_records, **s3_records}

    missing_pos_ids = needed_positive_ids - set(all_targets.keys())
    if missing_pos_ids:
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

    logger.info(f"Targets ready: {len(all_targets):,} records.")

    targets_by_country: Dict[str, List[NormalizedRecord]] = defaultdict(list)
    for r in all_targets.values():
        targets_by_country[r.country].append(r)

    name_token_freqs: Dict[str, Dict[str, int]] = {}
    addr_token_freqs: Dict[str, Dict[str, int]] = {}

    for country, tgt_list in targets_by_country.items():
        ntf = defaultdict(int)
        atf = defaultdict(int)
        for r in tgt_list:
            for t in set(r.name_tokens):
                if t not in COMMON_STOP_TOKENS and len(t) >= 3:
                    ntf[t] += 1
            for t in set(r.addr_tokens):
                if len(t) >= 4:
                    atf[t] += 1
        name_token_freqs[country] = ntf
        addr_token_freqs[country] = atf

    # Build Baseline MultiPassCandidateGenerator indexes
    logger.info("Building Baseline Candidate Generator...")
    max_candidates_per_key = 60
    base_indexes: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for country, tgt_list in targets_by_country.items():
        b_idx = {
            "exact_name": defaultdict(list),
            "sorted_name": defaultdict(list),
            "exact_addr": defaultdict(list),
            "numeric_anchor": defaultdict(list),
            "rare_token": defaultdict(list),
            "char_prefix": defaultdict(list),
        }
        tf = name_token_freqs[country]
        for r in tgt_list:
            eid = r.entity_id
            if r.name_norm and len(b_idx["exact_name"][r.name_norm]) < max_candidates_per_key:
                b_idx["exact_name"][r.name_norm].append(eid)
            if r.name_sorted != r.name_norm and len(b_idx["sorted_name"][r.name_sorted]) < max_candidates_per_key:
                b_idx["sorted_name"][r.name_sorted].append(eid)
            if not r.addr_missing and len(r.addr_norm) >= 8 and len(b_idx["exact_addr"][r.addr_norm]) < max_candidates_per_key:
                b_idx["exact_addr"][r.addr_norm].append(eid)
            if r.name_tokens and r.addr_numbers:
                num_key = f"{r.name_tokens[0]}_{r.addr_numbers[0]}"
                if len(b_idx["numeric_anchor"][num_key]) < max_candidates_per_key:
                    b_idx["numeric_anchor"][num_key].append(eid)
            # Baseline single rarest token
            best_t = None
            best_f = 999999
            for t in r.name_tokens:
                f = tf.get(t, 0)
                if 1 <= f <= 50 and f < best_f:
                    best_f = f
                    best_t = t
            if best_t and len(b_idx["rare_token"][best_t]) < max_candidates_per_key:
                b_idx["rare_token"][best_t].append(eid)
            if r.name_norm and r.addr_numbers and len(r.name_norm) >= 4:
                prefix_key = f"{r.name_norm[:4]}_{r.addr_numbers[0]}"
                if len(b_idx["char_prefix"][prefix_key]) < max_candidates_per_key:
                    b_idx["char_prefix"][prefix_key].append(eid)

        base_indexes[country] = b_idx

    # ============================================================
    # BUILD NEW TARGETED BLOCKER INDICES WITH BLOCK-SIZE TRACKING
    # ============================================================
    logger.info("Building new targeted blocker indices and measuring block sizes...")

    new_blockers = [
        "all_rare_df50",
        "core_name_exact",
        "core_name_sorted",
        "prefix_3_any_num",
        "addr_token_num_df40",
    ]

    raw_new_indices: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        b: {c: defaultdict(list) for c in targets_by_country} for b in new_blockers
    }
    trunc_new_indices: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        b: {c: defaultdict(list) for c in targets_by_country} for b in new_blockers
    }

    NEW_KEY_CAP = 60

    for country, tgt_list in targets_by_country.items():
        ntf = name_token_freqs[country]
        atf = addr_token_freqs[country]

        for r in tgt_list:
            eid = r.entity_id
            core_toks = strip_legal_suffixes(r.name_tokens)
            core_name = " ".join(core_toks)
            core_sorted = " ".join(sorted(core_toks)) if len(core_toks) > 1 else core_name

            # 1. all_rare_df50
            for t in set(r.name_tokens):
                f = ntf.get(t, 0)
                if 1 <= f <= 50:
                    raw_new_indices["all_rare_df50"][country][t].append(eid)
                    if len(trunc_new_indices["all_rare_df50"][country][t]) < NEW_KEY_CAP:
                        trunc_new_indices["all_rare_df50"][country][t].append(eid)

            # 2. core_name_exact
            if core_name and core_name != r.name_norm:
                raw_new_indices["core_name_exact"][country][core_name].append(eid)
                if len(trunc_new_indices["core_name_exact"][country][core_name]) < NEW_KEY_CAP:
                    trunc_new_indices["core_name_exact"][country][core_name].append(eid)

            # 3. core_name_sorted
            if core_sorted and core_sorted != r.name_sorted and core_sorted != core_name:
                raw_new_indices["core_name_sorted"][country][core_sorted].append(eid)
                if len(trunc_new_indices["core_name_sorted"][country][core_sorted]) < NEW_KEY_CAP:
                    trunc_new_indices["core_name_sorted"][country][core_sorted].append(eid)

            # 4. prefix_3_any_num
            if len(r.name_norm) >= 3 and r.addr_numbers:
                p3 = r.name_norm[:3]
                for num in r.addr_numbers[:2]:
                    k = f"{p3}_{num}"
                    raw_new_indices["prefix_3_any_num"][country][k].append(eid)
                    if len(trunc_new_indices["prefix_3_any_num"][country][k]) < NEW_KEY_CAP:
                        trunc_new_indices["prefix_3_any_num"][country][k].append(eid)

            # 5. addr_token_num_df40
            if r.addr_tokens and r.addr_numbers:
                best_at = None
                best_af = 999999
                for at in r.addr_tokens:
                    af = atf.get(at, 0)
                    if 1 <= af <= 40 and af < best_af:
                        best_af = af
                        best_at = at
                if best_at:
                    k = f"{best_at}_{r.addr_numbers[0]}"
                    raw_new_indices["addr_token_num_df40"][country][k].append(eid)
                    if len(trunc_new_indices["addr_token_num_df40"][country][k]) < NEW_KEY_CAP:
                        trunc_new_indices["addr_token_num_df40"][country][k].append(eid)

    # Measure block sizes for new blockers
    new_block_size_stats = {}
    for b in new_blockers:
        all_sizes = []
        for country in targets_by_country:
            for k, elist in raw_new_indices[b][country].items():
                all_sizes.append(len(elist))
        all_sizes = np.array(all_sizes, dtype=np.int32)
        if len(all_sizes) > 0:
            new_block_size_stats[b] = {
                "total_keys": int(len(all_sizes)),
                "p50": float(np.percentile(all_sizes, 50)),
                "p90": float(np.percentile(all_sizes, 90)),
                "p95": float(np.percentile(all_sizes, 95)),
                "p99": float(np.percentile(all_sizes, 99)),
                "p99.9": float(np.percentile(all_sizes, 99.9)),
                "max": int(np.max(all_sizes)),
                "keys_truncated": int(np.sum(all_sizes > NEW_KEY_CAP)),
                "pct_keys_truncated": float(np.mean(all_sizes > NEW_KEY_CAP) * 100),
            }
        logger.info(f"Blocker {b:20s}: Keys={new_block_size_stats[b]['total_keys']:,}, P50={new_block_size_stats[b]['p50']:.1f}, P90={new_block_size_stats[b]['p90']:.1f}, P99={new_block_size_stats[b]['p99']:.1f}, Max={new_block_size_stats[b]['max']}, Truncated={new_block_size_stats[b]['keys_truncated']} ({new_block_size_stats[b]['pct_keys_truncated']:.2f}%)")

    # ============================================================
    # DEFINE RECOMMENDED MULTI-PASS RETRIEVAL
    # ============================================================
    # We will test candidate generation configurations with different caps (50, 60, 75)
    # and measure the full pipeline through model scoring.

    def generate_candidates_combined(
        per_s1_cap: int = 60,
        include_addr_token: bool = True,
    ) -> Tuple[Dict[str, Set[str]], List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]], float]:
        t0 = time.time()
        cands_by_s1: Dict[str, Set[str]] = {}
        pair_objects: List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]] = []

        for s1_id in val_s1_ids:
            s1 = s1_all[s1_id]
            country = s1.country
            b_idx = base_indexes.get(country, {})
            cands_dict: Dict[str, CandidatePair] = {}

            def add_c(target_id: str, rt: str):
                if target_id not in cands_dict:
                    cands_dict[target_id] = CandidatePair(s1_id=s1.entity_id, target_id=target_id)
                p = cands_dict[target_id]
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

            # 1. Baseline routes
            if s1.name_norm:
                for tid in b_idx.get("exact_name", {}).get(s1.name_norm, []):
                    add_c(tid, "exact_name")
            if s1.name_sorted != s1.name_norm:
                for tid in b_idx.get("sorted_name", {}).get(s1.name_sorted, []):
                    add_c(tid, "sorted_name")
            if not s1.addr_missing and len(s1.addr_norm) >= 8:
                for tid in b_idx.get("exact_addr", {}).get(s1.addr_norm, []):
                    add_c(tid, "exact_addr")
            if s1.name_tokens and s1.addr_numbers:
                num_key = f"{s1.name_tokens[0]}_{s1.addr_numbers[0]}"
                for tid in b_idx.get("numeric_anchor", {}).get(num_key, []):
                    add_c(tid, "numeric_anchor")
            for t in s1.name_tokens:
                if t not in COMMON_STOP_TOKENS and t in b_idx.get("rare_token", {}):
                    for tid in b_idx["rare_token"][t]:
                        add_c(tid, "rare_token")
            if s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 4:
                prefix_key = f"{s1.name_norm[:4]}_{s1.addr_numbers[0]}"
                for tid in b_idx.get("char_prefix", {}).get(prefix_key, []):
                    add_c(tid, "char_ngram")

            # 2. Targeted Additions
            # 2a. all_rare_tokens_df50
            for t in s1.name_tokens:
                if t not in COMMON_STOP_TOKENS and t in trunc_new_indices["all_rare_df50"].get(country, {}):
                    for tid in trunc_new_indices["all_rare_df50"][country][t]:
                        add_c(tid, "rare_token")

            # 2b. core_name_exact
            core_toks = strip_legal_suffixes(s1.name_tokens)
            cn = " ".join(core_toks)
            if cn:
                for tid in trunc_new_indices["core_name_exact"].get(country, {}).get(cn, []):
                    add_c(tid, "exact_name")

            # 2c. core_name_sorted
            cs = " ".join(sorted(core_toks)) if len(core_toks) > 1 else cn
            if cs and cs != cn:
                for tid in trunc_new_indices["core_name_sorted"].get(country, {}).get(cs, []):
                    add_c(tid, "sorted_name")

            # 2d. prefix_3_any_num
            if len(s1.name_norm) >= 3 and s1.addr_numbers:
                p3 = s1.name_norm[:3]
                p3_idx = trunc_new_indices["prefix_3_any_num"].get(country, {})
                for num in s1.addr_numbers[:2]:
                    k = f"{p3}_{num}"
                    if k in p3_idx:
                        for tid in p3_idx[k]:
                            add_c(tid, "char_ngram")

            # 2e. addr_token_num_df40 (optional high yield)
            if include_addr_token and s1.addr_tokens and s1.addr_numbers:
                at_idx = trunc_new_indices["addr_token_num_df40"].get(country, {})
                for at in s1.addr_tokens:
                    k = f"{at}_{s1.addr_numbers[0]}"
                    if k in at_idx:
                        for tid in at_idx[k]:
                            add_c(tid, "exact_addr")

            # Cap candidates per S1 by route_count descending
            if len(cands_dict) > per_s1_cap:
                sorted_pairs = sorted(cands_dict.values(), key=lambda p: p.route_count, reverse=True)
                final_pairs = sorted_pairs[:per_s1_cap]
            else:
                final_pairs = list(cands_dict.values())

            cands_by_s1[s1_id] = set(p.target_id for p in final_pairs)
            for p in final_pairs:
                if p.target_id in all_targets:
                    pair_objects.append((s1, all_targets[p.target_id], p))

        elapsed = time.time() - t0
        return cands_by_s1, pair_objects, elapsed

    # Test two configurations:
    # Config 1: Without addr_token (Name-focused conservative expansion)
    # Config 2: With addr_token (Full multi-modal high-recall expansion)

    configs_to_test = [
        ("Conservative Name Expansion (cap=50)", 50, False),
        ("Conservative Name Expansion (cap=60)", 60, False),
        ("Recommended Multi-Modal Expansion (cap=50)", 50, True),
        ("Recommended Multi-Modal Expansion (cap=60)", 60, True),
        ("Recommended Multi-Modal Expansion (cap=70)", 70, True),
    ]

    pipeline_results = []

    logger.info("============================================================")
    logger.info("RUNNING END-TO-END CANDIDATE GENERATION & MODEL EVALUATION")
    logger.info("============================================================")

    for cfg_name, cap, inc_addr in configs_to_test:
        cands_by_s1, pair_objects, gen_time = generate_candidates_combined(
            per_s1_cap=cap,
            include_addr_token=inc_addr,
        )

        covered = sum(len(val_gt[s] & cands_by_s1[s]) for s in val_s1_ids)
        cand_recall = covered / total_true_links
        cands_counts = [len(cands_by_s1[s]) for s in val_s1_ids]
        avg_cands = float(np.mean(cands_counts))
        med_cands = float(np.median(cands_counts))
        p90_cands = float(np.percentile(cands_counts, 90))
        p95_cands = float(np.percentile(cands_counts, 95))
        p99_cands = float(np.percentile(cands_counts, 99))
        max_cands = int(np.max(cands_counts))
        total_pairs = len(pair_objects)

        logger.info(f"--- Evaluated: {cfg_name} ---")
        logger.info(f"  Candidate Recall: {cand_recall*100:.2f}% ({covered:,}/{total_true_links:,})")
        logger.info(f"  Candidates/S1: Avg={avg_cands:.2f}, Med={med_cands:.0f}, P95={p95_cands:.0f}, P99={p99_cands:.0f}, Max={max_cands}")
        logger.info(f"  Total Pairs: {total_pairs:,}, GenTime={gen_time:.2f}s")

        # Extract features & predict with GBDT
        logger.info(f"  Extracting 33 pairwise features for {total_pairs:,} pairs...")
        t_feat = time.time()
        X_val = np.array(
            [extract_pairwise_feature_vector(s, t, cp) for s, t, cp in pair_objects],
            dtype=np.float32,
        )
        t_feat_elapsed = time.time() - t_feat
        logger.info(f"  Feature extraction completed in {t_feat_elapsed:.2f}s.")

        t_pred = time.time()
        probs = matcher.predict_proba(X_val)
        t_pred_elapsed = time.time() - t_pred
        logger.info(f"  Model scoring completed in {t_pred_elapsed:.2f}s.")

        # Group scored candidates by S1
        scored_by_s1: Dict[str, List[Tuple[str, float, NormalizedRecord]]] = defaultdict(list)
        for (s1_r, tgt_r, _), p in zip(pair_objects, probs):
            scored_by_s1[s1_r.entity_id].append((tgt_r.entity_id, float(p), tgt_r))

        # Decision Policy with missing-address relief rule:
        # target.addr_missing == True and name_edit >= 0.85 -> relief threshold
        base_thresh = config["decision"].get("base_threshold", 0.46)
        single_thresh = config["decision"].get("singleton_threshold", 0.32)
        max_score_gap = config["decision"].get("max_score_gap", 0.35)
        max_matches = config["decision"].get("max_matches_per_s1", 10)

        preds_with_relief: Dict[str, Set[str]] = {}
        preds_without_relief: Dict[str, Set[str]] = {}

        for s1_id in val_s1_ids:
            scored = scored_by_s1.get(s1_id, [])
            s1_r = s1_all[s1_id]

            # 1. Standard decision
            standard_matches = decide_matches_for_entity(
                scored,
                base_threshold=base_thresh,
                singleton_threshold=single_thresh,
                max_score_gap=max_score_gap,
                max_matches=max_matches,
            )
            preds_without_relief[s1_id] = set(standard_matches)

            # 2. Decision with Missing Address Relief
            # If target.addr_missing and name_edit >= 0.85, allow acceptance at relaxed threshold (0.32)
            adjusted_scored = []
            for tid, prob, tgt_r in scored:
                p_eff = prob
                if tgt_r.addr_missing:
                    n_edit = fuzz.ratio(s1_r.name_norm, tgt_r.name_norm) / 100.0
                    if n_edit >= 0.85 and p_eff < base_thresh and p_eff >= 0.28:
                        # Relief boost: elevate probability to base_thresh
                        p_eff = max(p_eff, base_thresh)
                adjusted_scored.append((tid, p_eff, tgt_r))

            relief_matches = decide_matches_for_entity(
                adjusted_scored,
                base_threshold=base_thresh,
                singleton_threshold=single_thresh,
                max_score_gap=max_score_gap,
                max_matches=max_matches,
            )
            preds_with_relief[s1_id] = set(relief_matches)

        # Evaluate final macro F0.5
        m_std = evaluate_predictions(preds_without_relief, val_gt)
        m_rel = evaluate_predictions(preds_with_relief, val_gt)
        f05_std, p_std, r_std = m_std["macro_f05"], m_std["mean_precision"], m_std["mean_recall"]
        f05_rel, p_rel, r_rel = m_rel["macro_f05"], m_rel["mean_precision"], m_rel["mean_recall"]

        logger.info(f"  Standard Decision: Macro F0.5={f05_std:.4f} | Prec={p_std:.4f} | Rec={r_std:.4f}")
        logger.info(f"  With Missing-Addr Relief: Macro F0.5={f05_rel:.4f} | Prec={p_rel:.4f} | Rec={r_rel:.4f}")

        pipeline_results.append({
            "name": cfg_name,
            "cap": cap,
            "include_addr_token": inc_addr,
            "candidate_recall": float(cand_recall),
            "candidates_covered": covered,
            "avg_candidates_per_s1": avg_cands,
            "median_candidates_per_s1": med_cands,
            "p95": p95_cands,
            "p99": p99_cands,
            "max": max_cands,
            "total_pairs": total_pairs,
            "gen_time_s": gen_time,
            "feat_time_s": t_feat_elapsed,
            "pred_time_s": t_pred_elapsed,
            "standard_metrics": {
                "macro_f05": float(f05_std),
                "precision": float(p_std),
                "recall": float(r_std),
            },
            "relief_metrics": {
                "macro_f05": float(f05_rel),
                "precision": float(p_rel),
                "recall": float(r_rel),
            },
        })

    # Save complete evaluation report
    eval_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_true_links": total_true_links,
        "new_blocker_block_sizes": new_block_size_stats,
        "evaluations": pipeline_results,
    }

    with open(reports_dir / "advanced_blocking_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    logger.info("Saved advanced evaluation report to artifacts/reports/advanced_blocking_evaluation.json")


if __name__ == "__main__":
    main()
