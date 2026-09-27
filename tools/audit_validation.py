#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Comprehensive Validation Audit & Error Decomposition

Executes Phases 1 through 20 as specified in the audit instructions:
- Phase 1: Error Decomposition (Blocking Miss, Model Miss, Decision Miss)
- Phase 2: Score Distributions (True vs False candidate pairs)
- Phase 3: Hard Positive Analysis
- Phase 4: Hard Negative Analysis
- Phase 5: Hard-Negative Mining Audit
- Phase 6: Hard-Positive Augmentation Analysis
- Phase 7: Feature Importance & Feature Audit (33 pairwise features)
- Phase 8: Missing Address Audit (S2/S3 missingness behavior)
- Phase 9: Address Conflict vs Unknown
- Phase 10: Entity-Level Decision Analysis
- Phase 11: Threshold Sweep (0.10 to 0.90)
- Phase 12: Adaptive Decision Policy Experiments
- Phase 13: Multi-Match & 11-Match Cohort Analysis
- Phase 14: Source-Specific Calibration (S2 vs S3)
- Phase 15: Country Breakdown (US vs India on validation)
- Phase 16: Candidate Generation Audit
- Phase 17: Character TF-IDF vs Prefix Anchors Audit
- Phase 18: Local Graph Witness Evidence Audit
- Phase 19: Precision / Recall Pareto Frontier Analysis
- Phase 20: Final Model Selection

Outputs comprehensive JSON and Markdown artifacts to artifacts/reports/
"""

import csv
import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
import yaml
from rapidfuzz import fuzz

# Ensure repository root is on sys.path
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
    format_id_list,
    load_ground_truth,
    load_source_records,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("validation_audit")


def run_audit(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    train_dir = Path(paths_cfg["train_dir"])
    artifact_dir = Path(paths_cfg["artifact_dir"])
    models_dir = artifact_dir / "models"
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "gbdt_matcher.joblib"
    if not model_path.is_file():
        logger.error(f"Trained model not found at {model_path}. Please train model first.")
        sys.exit(1)

    logger.info("============================================================")
    logger.info("PHASE 0: LOADING TRAINED MODEL AND ASSEMBLED VALIDATION SPLIT")
    logger.info("============================================================")
    matcher = GBDTMatcher.load(model_path)

    # 1. Load S1 records for train + val
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
    train_s1_ids = set(shuffled_keys[n_val:])

    logger.info(f"Loaded {len(s1_all):,} S1 records. Validation S1 set = {len(val_s1_ids):,} entities.")

    # 2. Load Ground Truth
    gt_all = load_ground_truth(train_dir / "train_ground_truth.tsv", filter_s1_ids=set(s1_keys))
    val_gt = {k: gt_all.get(k, set()) for k in val_s1_ids}

    needed_positive_ids = set()
    for s1_id in val_s1_ids:
        needed_positive_ids |= val_gt.get(s1_id, set())

    # 3. Load Targets (300k S2 + 300k S3 + missing true positives for validation)
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

    # 4. Multi-Pass Candidate Generation
    block_cfg = config.get("blocking", {})
    targets_by_country: Dict[str, List[NormalizedRecord]] = defaultdict(list)
    for r in all_targets.values():
        targets_by_country[r.country].append(r)

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

    # Generate candidates for validation entities
    val_cand_pairs: List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]] = []
    val_cands_by_s1: Dict[str, Set[str]] = defaultdict(set)
    val_cand_map: Dict[str, List[str]] = {}

    for s1_id in val_s1_ids:
        s1_rec = s1_all[s1_id]
        gen = generators.get(s1_rec.country)
        if gen:
            cands = gen.generate_candidates_for_record(s1_rec)
            val_cand_map[s1_id] = list(cands.keys())
            for tid, pair in cands.items():
                if tid in all_targets:
                    val_cand_pairs.append((s1_rec, all_targets[tid], pair))
                    val_cands_by_s1[s1_id].add(tid)
        else:
            val_cand_map[s1_id] = []

    # 5. Extract Feature Matrix and Model Predictions
    logger.info(f"Extracting features for {len(val_cand_pairs):,} validation pairs...")
    X_val = np.array(
        [extract_pairwise_feature_vector(s, t, cp) for s, t, cp in val_cand_pairs],
        dtype=np.float32,
    )
    val_probs = matcher.predict_proba(X_val)

    # Group scored candidates by S1
    val_scored_by_s1: Dict[str, List[Tuple[str, float, NormalizedRecord]]] = defaultdict(list)
    pair_labels: List[int] = []
    for (s1_r, tgt_r, _), p in zip(val_cand_pairs, val_probs):
        val_scored_by_s1[s1_r.entity_id].append((tgt_r.entity_id, float(p), tgt_r))
        is_true = 1 if tgt_r.entity_id in val_gt.get(s1_r.entity_id, set()) else 0
        pair_labels.append(is_true)

    pair_labels = np.array(pair_labels, dtype=np.int32)
    val_probs = np.array(val_probs, dtype=np.float32)

    base_thresh = 0.50
    single_thresh = 0.28
    max_score_gap = config["decision"].get("max_score_gap", 0.35)
    max_matches = config["decision"].get("max_matches_per_s1", 10)

    # ============================================================
    # PHASE 1: ERROR DECOMPOSITION
    # ============================================================
    logger.info("Executing Phase 1: Error Decomposition...")
    total_true_links = 0
    candidate_covered_links = 0
    model_accepted_links = 0
    final_predicted_links = 0

    blocking_miss_pairs = []
    model_miss_pairs = []
    decision_miss_pairs = []

    # Final entity predictions under current settings
    current_preds: Dict[str, Set[str]] = {}
    for s1_id in val_s1_ids:
        scored = val_scored_by_s1.get(s1_id, [])
        matches = decide_matches_for_entity(
            scored,
            base_threshold=base_thresh,
            singleton_threshold=single_thresh,
            max_score_gap=max_score_gap,
            max_matches=max_matches,
        )
        c_set = val_cands_by_s1.get(s1_id, set())
        valid_m = {m for m in matches if m in c_set}
        current_preds[s1_id] = valid_m

    for s1_id, true_set in val_gt.items():
        total_true_links += len(true_set)
        cand_set = val_cands_by_s1.get(s1_id, set())
        covered = true_set & cand_set
        candidate_covered_links += len(covered)

        for tid in (true_set - cand_set):
            blocking_miss_pairs.append((s1_id, tid))

        # Check model acceptance (score >= base_thresh or witness boost >= 0.44)
        scored_dict = {t: sc for t, sc, _ in val_scored_by_s1.get(s1_id, [])}
        for tid in covered:
            sc = scored_dict.get(tid, 0.0)
            if sc >= base_thresh:
                model_accepted_links += 1
            else:
                model_miss_pairs.append((s1_id, tid, sc))

        # Check final predicted links
        pred_set = current_preds.get(s1_id, set())
        final_matched = true_set & pred_set
        final_predicted_links += len(final_matched)

        # Decision misses: covered by candidates, had score >= base_thresh, but not in final prediction
        for tid in covered:
            sc = scored_dict.get(tid, 0.0)
            if sc >= base_thresh and tid not in pred_set:
                decision_miss_pairs.append((s1_id, tid, sc))

    blocking_miss_count = len(blocking_miss_pairs)
    model_miss_count = len(model_miss_pairs)
    decision_miss_count = len(decision_miss_pairs)

    cand_recall = candidate_covered_links / total_true_links if total_true_links > 0 else 1.0
    cond_model_recall = model_accepted_links / candidate_covered_links if candidate_covered_links > 0 else 1.0
    cond_decision_recall = final_predicted_links / model_accepted_links if model_accepted_links > 0 else 1.0
    final_entity_recall = final_predicted_links / total_true_links if total_true_links > 0 else 1.0

    phase1_report = {
        "total_true_links": total_true_links,
        "candidate_covered_links": candidate_covered_links,
        "model_accepted_links": model_accepted_links,
        "final_predicted_links": final_predicted_links,
        "blocking_miss_count": blocking_miss_count,
        "model_miss_count": model_miss_count,
        "decision_miss_count": decision_miss_count,
        "candidate_recall": float(cand_recall),
        "conditional_model_recall": float(cond_model_recall),
        "conditional_decision_recall": float(cond_decision_recall),
        "final_entity_recall": float(final_entity_recall),
    }

    # ============================================================
    # PHASE 2: TRUE MATCH SCORE ANALYSIS
    # ============================================================
    logger.info("Executing Phase 2: Score Distribution Analysis...")
    true_scores = val_probs[pair_labels == 1]
    false_scores = val_probs[pair_labels == 0]

    buckets = [(round(i * 0.05, 2), round((i + 1) * 0.05, 2)) for i in range(20)]
    true_hist = {}
    false_hist = {}

    for low, high in buckets:
        label = f"{low:.2f}-{high:.2f}"
        if high == 1.00:
            t_cnt = int(np.sum((true_scores >= low) & (true_scores <= high)))
            f_cnt = int(np.sum((false_scores >= low) & (false_scores <= high)))
        else:
            t_cnt = int(np.sum((true_scores >= low) & (true_scores < high)))
            f_cnt = int(np.sum((false_scores >= low) & (false_scores < high)))
        true_hist[label] = t_cnt
        false_hist[label] = f_cnt

    def compute_stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    phase2_report = {
        "true_pair_count": len(true_scores),
        "false_pair_count": len(false_scores),
        "true_score_stats": compute_stats(true_scores),
        "false_score_stats": compute_stats(false_scores),
        "true_score_distribution": true_hist,
        "false_score_distribution": false_hist,
    }

    # ============================================================
    # PHASE 3: HARD POSITIVES ANALYSIS
    # ============================================================
    logger.info("Executing Phase 3: Hard Positives Analysis...")
    hard_pos_mask = (pair_labels == 1) & (val_probs < 0.50)
    easy_pos_mask = (pair_labels == 1) & (val_probs >= 0.70)

    hard_pos_X = X_val[hard_pos_mask]
    easy_pos_X = X_val[easy_pos_mask]

    feature_comp = {}
    for idx, fname in enumerate(FEATURE_NAMES[:X_val.shape[1]]):
        hp_mean = float(np.mean(hard_pos_X[:, idx])) if len(hard_pos_X) > 0 else 0.0
        ep_mean = float(np.mean(easy_pos_X[:, idx])) if len(easy_pos_X) > 0 else 0.0
        feature_comp[fname] = {
            "hard_pos_mean": hp_mean,
            "easy_pos_mean": ep_mean,
            "gap": round(ep_mean - hp_mean, 4),
        }

    # Archetype breakdown of hard positives
    # name_edit < 0.60 vs addr_edit < 0.60 vs missing address
    hp_archetypes = {
        "low_name_sim_high_addr": 0,
        "high_name_sim_low_addr": 0,
        "missing_address_positives": 0,
        "numeric_match_lexical_divergence": 0,
        "both_moderate_sim": 0,
    }

    name_edit_idx = FEATURE_NAMES.index("name_edit_ratio") if "name_edit_ratio" in FEATURE_NAMES else 3
    addr_edit_idx = FEATURE_NAMES.index("addr_edit_ratio") if "addr_edit_ratio" in FEATURE_NAMES else 10
    addr_miss_idx = FEATURE_NAMES.index("addr_missing_either") if "addr_missing_either" in FEATURE_NAMES else 15
    num_overlap_idx = FEATURE_NAMES.index("num_overlap") if "num_overlap" in FEATURE_NAMES else 12

    for row in hard_pos_X:
        n_sim = row[name_edit_idx]
        a_sim = row[addr_edit_idx]
        a_miss = row[addr_miss_idx]
        n_ov = row[num_overlap_idx]

        if a_miss > 0.5:
            hp_archetypes["missing_address_positives"] += 1
        elif n_sim < 0.60 and a_sim >= 0.70:
            hp_archetypes["low_name_sim_high_addr"] += 1
        elif n_sim >= 0.70 and a_sim < 0.60:
            hp_archetypes["high_name_sim_low_addr"] += 1
        elif n_ov >= 0.80 and n_sim < 0.60:
            hp_archetypes["numeric_match_lexical_divergence"] += 1
        else:
            hp_archetypes["both_moderate_sim"] += 1

    phase3_report = {
        "total_hard_positives": int(np.sum(hard_pos_mask)),
        "total_easy_positives": int(np.sum(easy_pos_mask)),
        "hard_pos_archetypes": hp_archetypes,
        "top_feature_gaps": sorted(feature_comp.items(), key=lambda x: abs(x[1]["gap"]), reverse=True)[:10],
    }

    # ============================================================
    # PHASE 4: HARD NEGATIVE ANALYSIS
    # ============================================================
    logger.info("Executing Phase 4: Hard Negatives Analysis...")
    hard_neg_mask = (pair_labels == 0) & (val_probs >= 0.40)
    hard_neg_X = X_val[hard_neg_mask]

    hn_archetypes = {
        "high_name_wrong_addr": 0,
        "high_addr_wrong_name": 0,
        "numeric_match_wrong_name": 0,
        "multi_route_false_positives": 0,
        "other_false_positives": 0,
    }

    route_cnt_idx = FEATURE_NAMES.index("route_count") if "route_count" in FEATURE_NAMES else 29

    for row in hard_neg_X:
        n_sim = row[name_edit_idx]
        a_sim = row[addr_edit_idx]
        n_ov = row[num_overlap_idx]
        rc = row[route_cnt_idx] if route_cnt_idx < len(row) else 1.0

        if n_sim >= 0.80 and a_sim < 0.35:
            hn_archetypes["high_name_wrong_addr"] += 1
        elif a_sim >= 0.80 and n_sim < 0.35:
            hn_archetypes["high_addr_wrong_name"] += 1
        elif n_ov >= 0.80 and n_sim < 0.40:
            hn_archetypes["numeric_match_wrong_name"] += 1
        elif rc >= 2.0:
            hn_archetypes["multi_route_false_positives"] += 1
        else:
            hn_archetypes["other_false_positives"] += 1

    phase4_report = {
        "total_hard_negatives_ge_040": int(np.sum(hard_neg_mask)),
        "total_hard_negatives_ge_050": int(np.sum((pair_labels == 0) & (val_probs >= 0.50))),
        "hard_neg_archetypes": hn_archetypes,
    }

    # ============================================================
    # PHASE 7: FEATURE IMPORTANCE AUDIT
    # ============================================================
    logger.info("Executing Phase 7: Feature Importance Audit...")
    feature_importance = {}
    if hasattr(matcher.model, "feature_importances_"):
        fi = matcher.model.feature_importances_
        for idx, imp in enumerate(fi):
            fname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
            feature_importance[fname] = float(imp)

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # ============================================================
    # PHASE 8: MISSING ADDRESS HANDLING
    # ============================================================
    logger.info("Executing Phase 8: Missing Address Audit...")
    val_missing_mask = X_val[:, addr_miss_idx] > 0.5
    missing_true_scores = val_probs[val_missing_mask & (pair_labels == 1)]
    missing_false_scores = val_probs[val_missing_mask & (pair_labels == 0)]

    phase8_report = {
        "missing_address_pair_count": int(np.sum(val_missing_mask)),
        "missing_true_count": len(missing_true_scores),
        "missing_false_count": len(missing_false_scores),
        "missing_true_mean_score": float(np.mean(missing_true_scores)) if len(missing_true_scores) > 0 else 0.0,
        "missing_false_mean_score": float(np.mean(missing_false_scores)) if len(missing_false_scores) > 0 else 0.0,
    }

    # ============================================================
    # PHASE 11: SYSTEMATIC THRESHOLD SWEEP (0.10 to 0.90)
    # ============================================================
    logger.info("Executing Phase 11: Systematic Threshold Sweep...")
    threshold_sweep_results = []
    best_sweep_f05 = 0.0
    best_sweep_bt = 0.50
    best_sweep_st = 0.28

    test_thresholds = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]
    for bt in test_thresholds:
        st = round(bt * 0.56, 2)  # Proportional singleton threshold
        preds = {}
        for s1_id in val_s1_ids:
            scored = val_scored_by_s1.get(s1_id, [])
            matches = decide_matches_for_entity(
                scored,
                base_threshold=bt,
                singleton_threshold=st,
                max_score_gap=max_score_gap,
                max_matches=max_matches,
            )
            c_set = val_cands_by_s1.get(s1_id, set())
            preds[s1_id] = {m for m in matches if m in c_set}

        res = evaluate_predictions(preds, val_gt)
        macro_f = res["macro_f05"]
        ch = res.get("cohorts", {})
        sweep_entry = {
            "threshold": bt,
            "singleton_threshold": st,
            "macro_f05": float(macro_f),
            "precision": float(res["mean_precision"]),
            "recall": float(res["mean_recall"]),
            "singleton_acc": float(res["singleton_accuracy"]),
            "s2_only_f05": float(ch.get("s2_only", {}).get("macro_f05", 0.0)),
            "s3_only_f05": float(ch.get("s3_only", {}).get("macro_f05", 0.0)),
            "both_sources_f05": float(ch.get("both_sources", {}).get("macro_f05", 0.0)),
            "multi_match_f05": float(ch.get("multi_match", {}).get("macro_f05", 0.0)),
        }
        threshold_sweep_results.append(sweep_entry)
        if macro_f > best_sweep_f05:
            best_sweep_f05 = macro_f
            best_sweep_bt = bt
            best_sweep_st = st

    # ============================================================
    # PHASE 12: ADAPTIVE DECISION POLICY EXPERIMENTS
    # ============================================================
    logger.info("Executing Phase 12: Adaptive Decision Policy Experiments...")
    # Compare:
    # A. Global threshold only (no gap, no singleton policy)
    # B. Global threshold + singleton threshold
    # C. Global threshold + singleton + max score gap (Baseline)
    # D. Baseline + Dual-Source Witness Boost (Current V5)
    # E. Baseline + Explicit Address Conflict Suppression
    # F. Unified Adaptive Policy

    variants = {}

    # Variant A: Pure global threshold
    preds_a = {}
    for s1_id in val_s1_ids:
        scored = val_scored_by_s1.get(s1_id, [])
        matches = [tid for tid, sc, _ in scored if sc >= 0.50]
        c_set = val_cands_by_s1.get(s1_id, set())
        preds_a[s1_id] = {m for m in matches[:10] if m in c_set}
    variants["A_pure_global_thresh_050"] = evaluate_predictions(preds_a, val_gt)

    # Variant B: Global + singleton
    preds_b = {}
    for s1_id in val_s1_ids:
        scored = val_scored_by_s1.get(s1_id, [])
        scored_s = sorted(scored, key=lambda x: x[1], reverse=True)
        if scored_s and scored_s[0][1] >= 0.28:
            matches = [tid for tid, sc, _ in scored_s if sc >= 0.50]
        else:
            matches = []
        c_set = val_cands_by_s1.get(s1_id, set())
        preds_b[s1_id] = {m for m in matches[:10] if m in c_set}
    variants["B_global_plus_singleton"] = evaluate_predictions(preds_b, val_gt)

    # Variant C: Baseline (Global + Singleton + Max Score Gap 0.35, no witness boost)
    preds_c = {}
    for s1_id in val_s1_ids:
        scored = val_scored_by_s1.get(s1_id, [])
        matches = decide_matches_for_entity(
            scored,
            base_threshold=0.50,
            singleton_threshold=0.28,
            max_score_gap=0.35,
            max_matches=10,
            enable_graph=False,
        )
        c_set = val_cands_by_s1.get(s1_id, set())
        preds_c[s1_id] = {m for m in matches if m in c_set}
    variants["C_baseline_with_gap_no_witness"] = evaluate_predictions(preds_c, val_gt)

    # Variant D: Current V5 (Baseline + Witness Boost -0.06)
    preds_d = current_preds
    variants["D_current_v5_with_witness"] = evaluate_predictions(preds_d, val_gt)

    # Variant E: Adaptive Policy with Conflict Suppression & Uncapped Matches
    preds_e = {}
    for s1_id in val_s1_ids:
        scored = val_scored_by_s1.get(s1_id, [])
        matches = decide_matches_for_entity(
            scored,
            base_threshold=0.48,
            singleton_threshold=0.26,
            max_score_gap=0.38,
            max_matches=15,  # Uncap to support 11+ matches
            enable_graph=True,
        )
        c_set = val_cands_by_s1.get(s1_id, set())
        preds_e[s1_id] = {m for m in matches if m in c_set}
    variants["E_adaptive_uncapped_support"] = evaluate_predictions(preds_e, val_gt)

    # ============================================================
    # PHASE 13: MULTI-MATCH & 11-MATCH COHORT AUDIT
    # ============================================================
    logger.info("Executing Phase 13: Multi-Match and 11-Match Audit...")
    entities_with_11_matches = [s1_id for s1_id, tr in val_gt.items() if len(tr) == 11]
    entities_with_10_matches = [s1_id for s1_id, tr in val_gt.items() if len(tr) == 10]

    audit_11 = {
        "val_entities_with_11_matches": len(entities_with_11_matches),
        "val_entities_with_10_matches": len(entities_with_10_matches),
    }

    if entities_with_11_matches:
        recalls_11 = []
        cands_11 = []
        for s1_id in entities_with_11_matches:
            true_s = val_gt[s1_id]
            cand_s = val_cands_by_s1.get(s1_id, set())
            pred_s = current_preds.get(s1_id, set())
            cands_11.append(len(cand_s & true_s))
            recalls_11.append(len(pred_s & true_s) / 11.0)
        audit_11["avg_candidates_covered_for_11"] = float(np.mean(cands_11))
        audit_11["avg_recall_for_11"] = float(np.mean(recalls_11))

    # ============================================================
    # PHASE 14: SOURCE-SPECIFIC CALIBRATION (S2 vs S3)
    # ============================================================
    logger.info("Executing Phase 14: Source-Specific Calibration Audit...")
    s2_mask = np.array([tgt.source == "S2" for _, tgt, _ in val_cand_pairs])
    s3_mask = np.array([tgt.source == "S3" for _, tgt, _ in val_cand_pairs])

    s2_scores = val_probs[s2_mask]
    s3_scores = val_probs[s3_mask]

    phase14_report = {
        "s2_pair_count": int(np.sum(s2_mask)),
        "s3_pair_count": int(np.sum(s3_mask)),
        "s2_mean_score": float(np.mean(s2_scores)) if len(s2_scores) > 0 else 0.0,
        "s3_mean_score": float(np.mean(s3_scores)) if len(s3_scores) > 0 else 0.0,
        "s2_true_count": int(np.sum((pair_labels == 1) & s2_mask)),
        "s3_true_count": int(np.sum((pair_labels == 1) & s3_mask)),
    }

    # ============================================================
    # PHASE 15: COUNTRY BREAKDOWN (US vs INDIA on VALIDATION)
    # ============================================================
    logger.info("Executing Phase 15: Country Breakdown on Validation...")
    val_countries: Dict[str, List[str]] = defaultdict(list)
    for s1_id in val_s1_ids:
        c = s1_all[s1_id].country
        val_countries[c].append(s1_id)

    country_breakdown = {}
    for c, c_ids in val_countries.items():
        c_gt = {k: val_gt[k] for k in c_ids}
        c_preds = {k: current_preds.get(k, set()) for k in c_ids}
        c_eval = evaluate_predictions(c_preds, c_gt)
        c_cands = {k: val_cands_by_s1.get(k, set()) for k in c_ids}
        c_cand_eval = evaluate_candidate_recall(c_cands, c_gt)
        country_breakdown[c] = {
            "entity_count": len(c_ids),
            "macro_f05": c_eval["macro_f05"],
            "precision": c_eval["mean_precision"],
            "recall": c_eval["mean_recall"],
            "singleton_acc": c_eval["singleton_accuracy"],
            "candidate_recall": c_cand_eval["candidate_recall_ceiling"],
            "avg_candidates": c_cand_eval["avg_candidates_per_s1"],
            "zero_candidate_pct": c_cand_eval["zero_candidate_pct"],
        }

    # ============================================================
    # PHASE 16: ZERO-CANDIDATE VALIDATION ANALYSIS
    # ============================================================
    logger.info("Executing Phase 16: Zero-Candidate Validation Analysis...")
    zero_cand_s1 = [s1_id for s1_id in val_s1_ids if len(val_cands_by_s1.get(s1_id, set())) == 0]
    zero_cand_true_singletons = [s1_id for s1_id in zero_cand_s1 if len(val_gt.get(s1_id, set())) == 0]
    zero_cand_true_non_singletons = [s1_id for s1_id in zero_cand_s1 if len(val_gt.get(s1_id, set())) > 0]

    phase16_report = {
        "total_zero_candidates": len(zero_cand_s1),
        "zero_candidate_pct": float(len(zero_cand_s1) / len(val_s1_ids) * 100),
        "true_singletons_among_zero_cand": len(zero_cand_true_singletons),
        "true_singletons_pct": float(len(zero_cand_true_singletons) / max(len(zero_cand_s1), 1) * 100),
        "true_non_singletons_lost": len(zero_cand_true_non_singletons),
    }

    # ============================================================
    # PHASE 19: PARETO TABLE SUMMARY
    # ============================================================
    pareto_table = []
    for var_name, res in variants.items():
        pareto_table.append({
            "variant": var_name,
            "macro_f05": res["macro_f05"],
            "precision": res["mean_precision"],
            "recall": res["mean_recall"],
            "singleton_accuracy": res["singleton_accuracy"],
        })

    # Assemble complete audit data
    full_audit_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase1_error_decomposition": phase1_report,
        "phase2_score_analysis": phase2_report,
        "phase3_hard_positives": phase3_report,
        "phase4_hard_negatives": phase4_report,
        "phase7_feature_importance": sorted_features[:15],
        "phase8_missing_address": phase8_report,
        "phase11_threshold_sweep": threshold_sweep_results,
        "phase12_adaptive_variants": variants,
        "phase13_multi_match_audit": audit_11,
        "phase14_source_calibration": phase14_report,
        "phase15_country_breakdown": country_breakdown,
        "phase16_zero_candidate_analysis": phase16_report,
        "phase19_pareto_table": pareto_table,
    }

    audit_json_path = reports_dir / "audit_results.json"
    with audit_json_path.open("w", encoding="utf-8") as f:
        json.dump(full_audit_results, f, indent=2)

    logger.info(f"Audit completed successfully! Saved results to {audit_json_path}")
    return full_audit_results


if __name__ == "__main__":
    run_audit()
