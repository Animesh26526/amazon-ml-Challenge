"""Train Baseline GBDT Model and Optimize Decision Threshold on Validation Cohort."""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision.entity_decision import decide_matches_for_entity
from src.evaluation.candidate_recall import compute_candidate_recall
from src.evaluation.diagnostics import evaluate_cohorts
from src.evaluation.metrics import compute_macro_f05, compute_per_entity_f05, evaluate_predictions
from src.normalization import NormalizedRecord
from src.pipeline import FEATURE_NAMES, MultiPassCandidateGenerator, extract_pairwise_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_training_and_validation(
    train_s1_count: int = 28000,
    val_s1_count: int = 12000,
    target_pool_size: int = 350000,
) -> Dict[str, Any]:
    """Execute end-to-end training and threshold calibration."""
    total_s1_needed = train_s1_count + val_s1_count
    logger.info(f"Loading {total_s1_needed:,} S1 entities and ground truth...")

    gt: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()

    with open("data/raw/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1_id = row[0].strip()
            matches = [m.strip() for m in row[1].split(",") if m.strip()]
            gt[s1_id] = set(matches)
            needed_targets.update(matches)
            if len(gt) >= total_s1_needed:
                break

    # Load S1 records
    s1_all: List[NormalizedRecord] = []
    with open("data/raw/train/train_source1.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1_id = row[0].strip()
            if s1_id in gt:
                s1_all.append(NormalizedRecord.from_raw(s1_id, row[1], row[2], row[3]))
                if len(s1_all) >= total_s1_needed:
                    break

    # Entity-level deterministic split
    train_s1 = s1_all[:train_s1_count]
    val_s1 = s1_all[train_s1_count:total_s1_needed]
    train_gt = {r.entity_id: gt[r.entity_id] for r in train_s1}
    val_gt = {r.entity_id: gt[r.entity_id] for r in val_s1}

    logger.info(f"Split: {len(train_s1):,} Train S1 entities, {len(val_s1):,} Validation S1 entities.")

    # Load target records (true targets + background distractors)
    target_dict: Dict[str, NormalizedRecord] = {}
    remaining_needed = set(needed_targets)

    for path in ["data/raw/train/train_source2.tsv", "data/raw/train/train_source3.tsv"]:
        logger.info(f"Scanning {Path(path).name} for target records...")
        with open(path, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                tid = row[0].strip()
                is_needed = tid in remaining_needed
                if is_needed or len(target_dict) < target_pool_size:
                    if tid not in target_dict:
                        target_dict[tid] = NormalizedRecord.from_raw(tid, row[1], row[2], row[3])
                    if is_needed:
                        remaining_needed.remove(tid)
                if len(target_dict) >= target_pool_size and len(remaining_needed) == 0:
                    break


    logger.info(f"Target universe loaded: {len(target_dict):,} records (contains {len(needed_targets & set(target_dict.keys())):,} / {len(needed_targets):,} true targets).")

    # Build Candidate Generator
    generator = MultiPassCandidateGenerator(max_candidates_per_key=50)
    generator.build_indexes(target_dict.values())

    # Generate training pairs
    logger.info("Extracting features for training pairs...")
    X_train_list = []
    y_train_list = []

    for s1 in train_s1:
        cands = generator.generate_candidates_for_record(s1)
        true_set = train_gt.get(s1.entity_id, set())

        # Include all generated candidates
        for tid, cand_pair in cands.items():
            if tid in target_dict:
                tgt_rec = target_dict[tid]
                feat_vec = extract_pairwise_feature_vector(s1, tgt_rec, cand_pair)
                label = 1 if tid in true_set else 0
                X_train_list.append(feat_vec)
                y_train_list.append(label)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)

    logger.info(f"Training dataset: {len(X_train):,} pairs ({np.sum(y_train):,} positives, {len(y_train)-np.sum(y_train):,} negatives).")

    # Train XGBoost Matcher
    logger.info("Training XGBoost GBDT matcher...")
    t_train0 = time.time()
    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t_train0
    logger.info(f"Trained XGBoost in {train_time:.2f}s.")

    # Save model artifact
    model_path = Path("artifacts/baseline_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Generate validation candidates & features
    logger.info("Evaluating on held-out validation cohort...")
    t_val0 = time.time()
    val_candidates: Dict[str, List[str]] = {}
    val_scored_pairs: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    val_pairs_features = []
    val_pair_keys = []

    for s1 in val_s1:
        cands = generator.generate_candidates_for_record(s1)
        val_candidates[s1.entity_id] = list(cands.keys())
        for tid, cand_pair in cands.items():
            if tid in target_dict:
                tgt_rec = target_dict[tid]
                val_pairs_features.append(extract_pairwise_feature_vector(s1, tgt_rec, cand_pair))
                val_pair_keys.append((s1.entity_id, tid))

    if val_pairs_features:
        X_val = np.array(val_pairs_features, dtype=np.float32)
        val_probs = model.predict_proba(X_val)[:, 1]
        for (s1_id, tid), prob in zip(val_pair_keys, val_probs):
            val_scored_pairs[s1_id].append((tid, float(prob)))

    val_inference_time = time.time() - t_val0

    # Candidate recall on validation
    cand_metrics = compute_candidate_recall(val_candidates, val_gt)
    logger.info(f"Validation Candidate Recall: {cand_metrics['candidate_recall']*100:.2f}% ({cand_metrics['covered_true_matches']:,} / {cand_metrics['total_true_matches']:,})")

    # Grid search decision threshold for macro F0.5
    best_thresh = 0.50
    best_macro_f05 = -1.0
    best_eval = {}

    threshold_grid = np.linspace(0.35, 0.85, 51)
    for t in threshold_grid:
        thresh = round(float(t), 3)
        preds = {}
        for s1 in val_s1:
            scored = val_scored_pairs.get(s1.entity_id, [])
            preds[s1.entity_id] = decide_matches_for_entity(scored, threshold=thresh, max_matches=10)

        ev = evaluate_predictions(preds, val_gt)
        if ev["macro_f0_5"] > best_macro_f05:
            best_macro_f05 = ev["macro_f0_5"]
            best_thresh = thresh
            best_eval = ev

    logger.info(f"Optimal Decision Threshold: {best_thresh:.3f} -> Validation Macro F0.5: {best_macro_f05:.4f}")
    logger.info(f"Precision: {best_eval['mean_precision']:.4f}, Recall: {best_eval['mean_recall']:.4f}, Singleton Accuracy: {best_eval['singleton_accuracy']:.4f}")

    # Generate best predictions for cohort evaluation
    final_val_preds = {}
    for s1 in val_s1:
        scored = val_scored_pairs.get(s1.entity_id, [])
        final_val_preds[s1.entity_id] = decide_matches_for_entity(scored, threshold=best_thresh, max_matches=10)

    cohort_results = evaluate_cohorts(final_val_preds, val_gt)

    report = {
        "experiment_id": "EXP001",
        "description": "Baseline V0 - Multi-Pass Blocking + XGBoost GBDT Matcher",
        "train_s1_count": train_s1_count,
        "val_s1_count": val_s1_count,
        "training_pairs": len(X_train),
        "validation_pairs": len(val_pair_keys),
        "train_time_s": round(train_time, 2),
        "val_inference_time_s": round(val_inference_time, 2),
        "candidate_recall": round(cand_metrics["candidate_recall"], 4),
        "avg_candidates_per_s1": round(cand_metrics["avg_candidates_per_s1"], 2),
        "optimal_threshold": best_thresh,
        "validation_macro_f0_5": round(best_macro_f05, 4),
        "validation_precision": round(best_eval["mean_precision"], 4),
        "validation_recall": round(best_eval["mean_recall"], 4),
        "singleton_accuracy": round(best_eval["singleton_accuracy"], 4),
        "cohort_breakdown": cohort_results,
    }

    report_path = Path("artifacts/baseline_validation_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation report saved to {report_path}")

    # Record in experiments/experiment_log.csv
    log_file = Path("experiments/experiment_log.csv")
    with log_file.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "EXP001",
            time.strftime("%Y-%m-%d"),
            "baseline-v0",
            "Baseline classical ER: multi-pass blocking + 28 pairwise features + XGBoost GBDT",
            "Initial full baseline pipeline",
            f"threshold={best_thresh}, max_cands_per_key=50, n_est=250",
            round(cand_metrics["candidate_recall"], 4),
            int(cand_metrics["total_candidate_pairs"]),
            round(best_eval["mean_precision"], 4),
            round(best_eval["mean_recall"], 4),
            round(best_macro_f05, 4),
            round(best_eval["singleton_accuracy"], 4),
            f"{train_time + val_inference_time:.1f}s",
            "VALIDATED",
            "Baseline model trained and threshold calibrated on validation split",
        ])

    return report


if __name__ == "__main__":
    run_training_and_validation()
