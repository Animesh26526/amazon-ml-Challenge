#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Retrain LightGBM on New Multi-Modal Candidates & Compare Validation

1. Builds Recommended Multi-Modal candidate blocker on training & validation targets.
2. Generates candidate pairs for training set (48,000 S1 entities).
3. Extracts 33 pairwise features.
4. Trains Base LightGBM on new candidate distribution.
5. Performs hard-negative mining (mines difficult non-matches with p >= 0.28).
6. Retrains Final LightGBM on augmented data with hard negatives.
7. Saves retrained model into artifacts/final_retrained/gbdt_matcher_retrained.joblib.
8. Evaluates both models (OLD existing LightGBM vs NEW retrained LightGBM) on canonical validation set.
9. Compares Macro F0.5, Precision, Recall, Singleton Accuracy, FP, FN, etc.
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
logger = logging.getLogger("retrain_and_validate")

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


class RecommendedMultiModalBlocker:
    """Validated high-recall multi-modal candidate generator."""

    def __init__(
        self,
        max_candidates_per_key: int = 60,
        max_candidates_per_s1: int = 60,
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

    def build_indexes(self, targets: List[NormalizedRecord]) -> None:
        gc.disable()
        try:
            name_tf: Dict[str, int] = defaultdict(int)
            addr_tf: Dict[str, int] = defaultdict(int)

            for r in targets:
                for t in set(r.name_tokens):
                    if t not in COMMON_STOP_TOKENS and len(t) >= 3:
                        name_tf[t] += 1
                for t in set(r.addr_tokens):
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
                for t in set(r.name_tokens):
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
            gc.collect()

    def generate_candidates(self, s1: NormalizedRecord) -> Dict[str, CandidatePair]:
        cands: Dict[str, CandidatePair] = {}

        def add_c(tid: str, rt: str):
            if tid not in cands:
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

        # 1. exact name
        if s1.name_norm:
            for tid in self.exact_name_idx.get(s1.name_norm, []):
                add_c(tid, "exact_name")

        # 2. sorted name
        if s1.name_sorted != s1.name_norm:
            for tid in self.sorted_name_idx.get(s1.name_sorted, []):
                add_c(tid, "sorted_name")

        # 3. exact addr
        if not s1.addr_missing and len(s1.addr_norm) >= self.min_addr_len:
            for tid in self.exact_addr_idx.get(s1.addr_norm, []):
                add_c(tid, "exact_addr")

        # 4. numeric anchor
        if s1.name_tokens and s1.addr_numbers:
            k = f"{s1.name_tokens[0]}_{s1.addr_numbers[0]}"
            for tid in self.numeric_anchor_idx.get(k, []):
                add_c(tid, "numeric_anchor")

        # 5. single rarest token
        for t in s1.name_tokens:
            if t not in COMMON_STOP_TOKENS and t in self.single_rare_idx:
                for tid in self.single_rare_idx[t]:
                    add_c(tid, "rare_token")

        # 6. char prefix 4
        if s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 4:
            k = f"{s1.name_norm[:4]}_{s1.addr_numbers[0]}"
            for tid in self.char_prefix4_idx.get(k, []):
                add_c(tid, "char_ngram")

        # 7. all rare df 50
        for t in s1.name_tokens:
            if t not in COMMON_STOP_TOKENS and t in self.all_rare_df50_idx:
                for tid in self.all_rare_df50_idx[t]:
                    add_c(tid, "rare_token")

        # 8 & 9. core name exact & sorted
        core_toks = strip_legal_suffixes(s1.name_tokens)
        cn = " ".join(core_toks)
        if cn:
            for tid in self.core_name_exact_idx.get(cn, []):
                add_c(tid, "exact_name")
        cs = " ".join(sorted(core_toks)) if len(core_toks) > 1 else cn
        if cs and cs != cn:
            for tid in self.core_name_sorted_idx.get(cs, []):
                add_c(tid, "sorted_name")

        # 10. prefix 3 any num
        if len(s1.name_norm) >= 3 and s1.addr_numbers:
            p3 = s1.name_norm[:3]
            for num in s1.addr_numbers[:2]:
                k = f"{p3}_{num}"
                for tid in self.prefix3_any_num_idx.get(k, []):
                    add_c(tid, "char_ngram")

        # 11. addr token num
        if s1.addr_tokens and s1.addr_numbers:
            for at in s1.addr_tokens:
                k = f"{at}_{s1.addr_numbers[0]}"
                for tid in self.addr_token_num_idx.get(k, []):
                    add_c(tid, "exact_addr")

        # Cap candidates per S1 by route count descending
        if len(cands) > self.max_candidates_per_s1:
            sorted_pairs = sorted(cands.values(), key=lambda p: p.route_count, reverse=True)
            cands = {p.target_id: p for p in sorted_pairs[:self.max_candidates_per_s1]}

        return cands


def main():
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    train_dir = Path(paths_cfg["train_dir"])
    artifact_dir = Path(paths_cfg["artifact_dir"])
    
    # Versioned artifact directory
    retrain_dir = artifact_dir / "final_retrained"
    retrain_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = config.get("experiment", {})
    train_sample_limit = exp_cfg.get("train_s1_sample", 60000)
    val_sample_limit = exp_cfg.get("val_s1_sample", 15000)
    val_ratio = exp_cfg.get("val_split_ratio", 0.20)
    seed = exp_cfg.get("seed", 42)

    logger.info("Loading S1 records and Ground Truth...")
    s1_all = load_source_records(train_dir / "train_source1.tsv", max_rows=train_sample_limit)
    s1_keys = list(s1_all.keys())
    rng = np.random.default_rng(seed)
    shuffled_keys = list(s1_keys)
    rng.shuffle(shuffled_keys)

    n_val = min(val_sample_limit, int(len(shuffled_keys) * val_ratio))
    val_s1_ids = set(shuffled_keys[:n_val])
    train_s1_ids = set(shuffled_keys[n_val:])

    gt_all = load_ground_truth(train_dir / "train_ground_truth.tsv", filter_s1_ids=set(s1_keys))
    train_gt = {k: gt_all.get(k, set()) for k in train_s1_ids}
    val_gt = {k: gt_all.get(k, set()) for k in val_s1_ids}

    logger.info(f"S1 Split: Train={len(train_s1_ids):,} entities, Val={len(val_s1_ids):,} entities.")

    needed_positive_ids = set()
    for s in gt_all.values():
        needed_positive_ids |= s

    logger.info("Loading Target Records...")
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

    logger.info(f"Target catalog ready: {len(all_targets):,} records.")

    # Partition targets by country
    targets_by_country: Dict[str, List[NormalizedRecord]] = defaultdict(list)
    for r in all_targets.values():
        targets_by_country[r.country].append(r)

    # Build Recommended Multi-Modal Blockers
    logger.info("Building Recommended Multi-Modal Blockers per country (cap=60)...")
    blockers: Dict[str, RecommendedMultiModalBlocker] = {}
    for country, tgt_list in targets_by_country.items():
        blk = RecommendedMultiModalBlocker(max_candidates_per_key=60, max_candidates_per_s1=60)
        blk.build_indexes(tgt_list)
        blockers[country] = blk

    # ============================================================
    # 1. GENERATE TRAINING CANDIDATES WITH NEW BLOCKER
    # ============================================================
    logger.info("Generating candidate pairs for training set under NEW blocker...")
    train_cand_pairs: List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]] = []
    train_cands_by_s1: Dict[str, Set[str]] = defaultdict(set)

    for s1_id in train_s1_ids:
        s1_rec = s1_all[s1_id]
        blk = blockers.get(s1_rec.country)
        if blk:
            cands = blk.generate_candidates(s1_rec)
            for tid, pair in cands.items():
                if tid in all_targets:
                    train_cand_pairs.append((s1_rec, all_targets[tid], pair))
                    train_cands_by_s1[s1_id].add(tid)

    train_recall_report = evaluate_candidate_recall(train_cands_by_s1, train_gt)
    logger.info(
        f"Train Candidate Recall under New Blocker: {train_recall_report['candidate_recall_ceiling']*100:.2f}% "
        f"({train_recall_report['covered_true_links']:,}/{train_recall_report['total_true_links']:,}), "
        f"Avg Cands/S1: {train_recall_report['avg_candidates_per_s1']:.2f}"
    )

    # Extract 33 pairwise features for training pairs
    logger.info(f"Extracting 33 pairwise features for {len(train_cand_pairs):,} training pairs...")
    X_train_list = []
    y_train_list = []
    for s1_r, tgt_r, cp in train_cand_pairs:
        feat = extract_pairwise_feature_vector(s1_r, tgt_r, cp)
        is_true = 1 if tgt_r.entity_id in train_gt.get(s1_r.entity_id, set()) else 0
        X_train_list.append(feat)
        y_train_list.append(is_true)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    del X_train_list, y_train_list
    gc.collect()

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    logger.info(f"Training Feature Matrix: {X_train.shape[0]:,} rows (Pos={pos_count:,}, Neg={neg_count:,}).")

    # ============================================================
    # 2. TRAIN INITIAL LIGHTGBM & MINE HARD NEGATIVES
    # ============================================================
    logger.info("Training initial LightGBM matcher on new candidate distribution...")
    model_cfg = dict(config)
    # Ensure optimal parameters
    matcher_new = GBDTMatcher(model_cfg)
    matcher_new.fit(X_train, y_train)

    logger.info("Scoring training candidates to identify boundary hard negatives...")
    train_probs = matcher_new.predict_proba(X_train)

    # Hard-negative selection: non-matching pairs with probability >= 0.28
    hard_neg_mask = (y_train == 0) & (train_probs >= 0.28)
    num_hard = int(np.sum(hard_neg_mask))
    logger.info(f"Discovered {num_hard:,} hard negative candidates (prob >= 0.28).")

    if num_hard > 0:
        # Augment training set with hard negatives (weight factor = 3 for boundary sharpening)
        X_hard = X_train[hard_neg_mask]
        y_hard = y_train[hard_neg_mask]

        X_aug = np.vstack([X_train, X_hard, X_hard])
        y_aug = np.concatenate([y_train, y_hard, y_hard])
        logger.info(f"Augmented training set: {X_train.shape[0]:,} -> {X_aug.shape[0]:,} rows.")

        logger.info("Refitting Final LightGBM Matcher on augmented dataset...")
        matcher_new.fit(X_aug, y_aug)
        del X_aug, y_aug, X_hard, y_hard
        gc.collect()

    # Save retrained model
    new_model_path = retrain_dir / "gbdt_matcher_retrained.joblib"
    matcher_new.save(new_model_path)
    logger.info(f"Saved retrained model to {new_model_path}")

    # ============================================================
    # 3. CANONICAL VALIDATION COMPARISON: OLD vs NEW
    # ============================================================
    logger.info("============================================================")
    logger.info("CANONICAL VALIDATION COMPARISON: OLD vs RETRAINED MODEL")
    logger.info("============================================================")

    # Load existing model (OLD)
    old_model_path = artifact_dir / "models" / "gbdt_matcher.joblib"
    matcher_old = GBDTMatcher.load(old_model_path)

    # Generate validation candidate pairs under new blocker
    val_cand_pairs: List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]] = []
    val_cands_by_s1: Dict[str, Set[str]] = defaultdict(set)

    for s1_id in val_s1_ids:
        s1_rec = s1_all[s1_id]
        blk = blockers.get(s1_rec.country)
        if blk:
            cands = blk.generate_candidates(s1_rec)
            for tid, pair in cands.items():
                if tid in all_targets:
                    val_cand_pairs.append((s1_rec, all_targets[tid], pair))
                    val_cands_by_s1[s1_id].add(tid)

    val_cand_eval = evaluate_candidate_recall(val_cands_by_s1, val_gt)
    total_val_links = val_cand_eval["total_true_links"]
    covered_val_links = val_cand_eval["covered_true_links"]
    cand_recall = val_cand_eval["candidate_recall_ceiling"]

    logger.info(f"Validation Candidate Recall: {cand_recall*100:.2f}% ({covered_val_links:,}/{total_val_links:,})")

    # Extract features for validation pairs
    logger.info(f"Extracting features for {len(val_cand_pairs):,} validation pairs...")
    X_val = np.array(
        [extract_pairwise_feature_vector(s, t, cp) for s, t, cp in val_cand_pairs],
        dtype=np.float32,
    )

    # Score with OLD model
    logger.info("Scoring validation set with OLD LightGBM model...")
    old_probs = matcher_old.predict_proba(X_val)

    # Score with NEW retrained model
    logger.info("Scoring validation set with NEW retrained LightGBM model...")
    new_probs = matcher_new.predict_proba(X_val)

    def evaluate_model_pipeline(
        probs_array: np.ndarray,
        base_th: float = 0.46,
        single_th: float = 0.32,
        max_matches: int = 15,
    ) -> Dict[str, Any]:
        scored_by_s1 = defaultdict(list)
        for (s1_r, tgt_r, _), p in zip(val_cand_pairs, probs_array):
            scored_by_s1[s1_r.entity_id].append((tgt_r.entity_id, float(p), tgt_r))

        predictions = {}
        total_pred_matches = 0
        matches_per_s1 = []

        for s1_id in val_s1_ids:
            scored = scored_by_s1.get(s1_id, [])
            matches = decide_matches_for_entity(
                scored,
                base_threshold=base_th,
                singleton_threshold=single_th,
                max_score_gap=0.35,
                max_matches=max_matches,
            )
            # Ensure matches are subset of candidates
            valid_m = set(matches) & val_cands_by_s1.get(s1_id, set())
            predictions[s1_id] = valid_m
            k = len(valid_m)
            matches_per_s1.append(k)
            total_pred_matches += k

        metrics = evaluate_predictions(predictions, val_gt)

        # Count total false positives and false negatives across all validation entities
        total_fp = 0
        total_fn = 0
        total_tp = 0
        for s1_id in val_s1_ids:
            p_set = predictions.get(s1_id, set())
            t_set = val_gt.get(s1_id, set())
            total_tp += len(p_set & t_set)
            total_fp += len(p_set - t_set)
            total_fn += len(t_set - p_set)

        arr = np.array(matches_per_s1, dtype=np.int32)

        return {
            "macro_f05": metrics["macro_f05"],
            "precision": metrics["mean_precision"],
            "recall": metrics["mean_recall"],
            "singleton_accuracy": metrics["singleton_accuracy"],
            "candidate_recall": cand_recall,
            "total_predicted_matches": total_pred_matches,
            "avg_matches_per_s1": float(np.mean(arr)),
            "max_matches_per_s1": int(np.max(arr)),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        }

    # Evaluate OLD model on validation
    old_eval = evaluate_model_pipeline(old_probs, base_th=0.46, single_th=0.32, max_matches=15)
    # Evaluate NEW retrained model on validation
    new_eval = evaluate_model_pipeline(new_probs, base_th=0.46, single_th=0.32, max_matches=15)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL VALIDATION COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':25s} | {'OLD (Existing Model)':20s} | {'NEW (Retrained Model)':20s}")
    logger.info("-" * 71)
    logger.info(f"{'Macro F0.5':25s} | {old_eval['macro_f05']:20.4f} | {new_eval['macro_f05']:20.4f}")
    logger.info(f"{'Precision':25s} | {old_eval['precision']:20.4f} | {new_eval['precision']:20.4f}")
    logger.info(f"{'Recall':25s} | {old_eval['recall']:20.4f} | {new_eval['recall']:20.4f}")
    logger.info(f"{'Singleton Accuracy':25s} | {old_eval['singleton_accuracy']:20.4f} | {new_eval['singleton_accuracy']:20.4f}")
    logger.info(f"{'Candidate Recall':25s} | {old_eval['candidate_recall']*100:19.2f}% | {new_eval['candidate_recall']*100:19.2f}%")
    logger.info(f"{'Total Predicted Matches':25s} | {old_eval['total_predicted_matches']:20,d} | {new_eval['total_predicted_matches']:20,d}")
    logger.info(f"{'Avg Matches / S1':25s} | {old_eval['avg_matches_per_s1']:20.2f} | {new_eval['avg_matches_per_s1']:20.2f}")
    logger.info(f"{'Max Matches / S1':25s} | {old_eval['max_matches_per_s1']:20d} | {new_eval['max_matches_per_s1']:20d}")
    logger.info(f"{'False Positives':25s} | {old_eval['false_positives']:20,d} | {new_eval['false_positives']:20,d}")
    logger.info(f"{'False Negatives':25s} | {old_eval['false_negatives']:20,d} | {new_eval['false_negatives']:20,d}")
    logger.info("=" * 60)

    # Determine final selected model
    if new_eval["macro_f05"] >= old_eval["macro_f05"]:
        selected_model_name = "NEW_RETRAINED"
        selected_model_path = new_model_path
        logger.info("Selection Verdict: NEW RETRAINED MODEL selected for test inference.")
    else:
        selected_model_name = "OLD_EXISTING"
        selected_model_path = old_model_path
        logger.info("Selection Verdict: OLD EXISTING MODEL selected as fallback for test inference.")

    # Save comparison report
    comp_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_model": selected_model_name,
        "selected_model_path": str(selected_model_path),
        "old_model_validation": old_eval,
        "new_model_validation": new_eval,
    }
    with open(reports_dir / "model_retraining_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comp_report, f, indent=2)

    logger.info("Saved comparison report to artifacts/reports/model_retraining_comparison.json")


if __name__ == "__main__":
    main()
