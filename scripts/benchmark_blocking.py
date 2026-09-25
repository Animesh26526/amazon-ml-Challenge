"""Benchmark Blocking Strategies on Held-out Ground Truth.

Measures individual and combined blocking strategies for:
- candidate recall
- avg/p95/p99/max candidates per S1
- reduction ratio
- runtime
- incremental unique true matches recovered
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking.char_ngram import CharNgramBlocking
from src.blocking.exact import ExactAddressBlocking, ExactNameAndAddressBlocking, ExactNameBlocking
from src.blocking.numeric import NumericAnchorBlocking
from src.blocking.phonetic import PhoneticBlocking
from src.blocking.rare_tokens import RareTokenBlocking
from src.normalization import NormalizedRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_benchmark_sample(
    s1_sample_size: int = 30000,
    target_sample_size: int = 400000,
) -> Tuple[List[NormalizedRecord], List[NormalizedRecord], Dict[str, Set[str]]]:
    """Load a representative sample of S1 and target records with ground truth."""
    logger.info("Loading benchmark sample...")

    # 1. Load ground truth for sample
    gt: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()

    with open("data/raw/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            s1_id = row[0].strip()
            matches = [m.strip() for m in row[1].split(",") if m.strip()]
            gt[s1_id] = set(matches)
            needed_targets.update(matches)
            if len(gt) >= s1_sample_size:
                break

    # 2. Load S1 records corresponding to ground truth
    s1_records: List[NormalizedRecord] = []
    with open("data/raw/train/train_source1.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1_id = row[0].strip()
            if s1_id in gt:
                s1_records.append(
                    NormalizedRecord.from_raw(s1_id, row[1], row[2], row[3])
                )
                if len(s1_records) >= len(gt):
                    break

    # 3. Load target records: all true targets + background distractors up to target_sample_size
    target_records: List[NormalizedRecord] = []
    found_targets: Set[str] = set()

    for path in ["data/raw/train/train_source2.tsv", "data/raw/train/train_source3.tsv"]:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for i, row in enumerate(r):
                tgt_id = row[0].strip()
                if tgt_id in needed_targets or len(target_records) < target_sample_size:
                    target_records.append(
                        NormalizedRecord.from_raw(tgt_id, row[1], row[2], row[3])
                    )
                    found_targets.add(tgt_id)
                if len(target_records) >= target_sample_size and needed_targets.issubset(found_targets):
                    break

    logger.info(
        f"Benchmark sample loaded: {len(s1_records):,} S1 queries, {len(target_records):,} target records, {len(needed_targets):,} true targets."
    )
    return s1_records, target_records, gt


def evaluate_blocker(
    blocker_name: str,
    blocker_instance: Any,
    s1_records: List[NormalizedRecord],
    target_records: List[NormalizedRecord],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """Evaluate a single blocking strategy."""
    logger.info(f"Evaluating {blocker_name}...")
    t0 = time.time()
    blocker_instance.build_index(target_records)
    build_time = time.time() - t0

    t1 = time.time()
    candidate_counts = []
    total_true_matches = sum(len(m) for m in ground_truth.values())
    covered_matches = 0
    s1_candidates_map: Dict[str, Set[str]] = {}

    for s1 in s1_records:
        cands = set(blocker_instance.query(s1))
        candidate_counts.append(len(cands))
        s1_candidates_map[s1.entity_id] = cands

        true_m = ground_truth.get(s1.entity_id, set())
        if true_m:
            covered_matches += len(true_m & cands)

    query_time = time.time() - t1
    counts_arr = np.array(candidate_counts)

    recall = covered_matches / total_true_matches if total_true_matches else 0.0
    avg_cands = float(np.mean(counts_arr))
    p90 = float(np.percentile(counts_arr, 90))
    p95 = float(np.percentile(counts_arr, 95))
    p99 = float(np.percentile(counts_arr, 99))
    max_cands = int(np.max(counts_arr))
    reduction_ratio = 1.0 - (np.sum(counts_arr) / (len(s1_records) * len(target_records)))

    return {
        "blocker": blocker_name,
        "recall": round(recall, 4),
        "covered_matches": covered_matches,
        "total_true_matches": total_true_matches,
        "avg_candidates": round(avg_cands, 2),
        "median_candidates": float(np.median(counts_arr)),
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max_candidates": max_cands,
        "reduction_ratio": round(reduction_ratio, 6),
        "build_time_s": round(build_time, 2),
        "query_time_s": round(query_time, 2),
        "total_time_s": round(build_time + query_time, 2),
        "candidates_map": s1_candidates_map,
    }


def main() -> None:
    s1_records, target_records, gt = load_benchmark_sample(s1_sample_size=30000, target_sample_size=300000)

    blockers = [
        ("exact_name_and_address", ExactNameAndAddressBlocking()),
        ("exact_name", ExactNameBlocking(country_partition=True)),
        ("exact_address", ExactAddressBlocking(country_partition=True)),
        ("numeric_anchors", NumericAnchorBlocking(max_candidates_per_key=50)),
        ("phonetic", PhoneticBlocking(max_candidates_per_key=50)),
        ("rare_tokens", RareTokenBlocking(min_doc_freq=2, max_doc_freq=30, max_candidates=25)),
    ]

    results = []
    maps = {}

    for name, blocker in blockers:
        res = evaluate_blocker(name, blocker, s1_records, target_records, gt)
        maps[name] = res.pop("candidates_map")
        results.append(res)
        logger.info(
            f"[{name}] Recall: {res['recall']*100:.2f}%, Avg Cands: {res['avg_candidates']:.1f}, Max: {res['max_candidates']}, Time: {res['total_time_s']}s"
        )

    # Union experiments
    combos = [
        ("Union: Name + Address", ["exact_name", "exact_address"]),
        ("Union: Name + Address + ExactJoint", ["exact_name", "exact_address", "exact_name_and_address"]),
        ("Union: Name + Address + Numeric", ["exact_name", "exact_address", "numeric_anchors"]),
        ("Union: Name + Address + Numeric + RareTokens", ["exact_name", "exact_address", "numeric_anchors", "rare_tokens"]),
        ("Union: All Blockers", ["exact_name", "exact_address", "exact_name_and_address", "numeric_anchors", "rare_tokens", "phonetic"]),
    ]

    total_true = sum(len(m) for m in gt.values())
    for combo_name, b_names in combos:
        union_counts = []
        covered = 0
        for s1 in s1_records:
            s1_id = s1.entity_id
            union_set: Set[str] = set()
            for b_name in b_names:
                union_set.update(maps[b_name].get(s1_id, set()))
            union_counts.append(len(union_set))
            true_m = gt.get(s1_id, set())
            if true_m:
                covered += len(true_m & union_set)

        u_arr = np.array(union_counts)
        rec = covered / total_true if total_true else 0.0
        results.append({
            "blocker": combo_name,
            "recall": round(rec, 4),
            "covered_matches": covered,
            "total_true_matches": total_true,
            "avg_candidates": round(float(np.mean(u_arr)), 2),
            "median_candidates": float(np.median(u_arr)),
            "p90": float(np.percentile(u_arr, 90)),
            "p95": float(np.percentile(u_arr, 95)),
            "p99": float(np.percentile(u_arr, 99)),
            "max_candidates": int(np.max(u_arr)),
            "reduction_ratio": round(1.0 - (np.sum(u_arr) / (len(s1_records) * len(target_records))), 6),
            "build_time_s": 0.0,
            "query_time_s": 0.0,
            "total_time_s": 0.0,
        })
        logger.info(
            f"[{combo_name}] Recall: {rec*100:.2f}%, Avg Cands: {np.mean(u_arr):.1f}, P95: {np.percentile(u_arr, 95)}, Max: {np.max(u_arr)}"
        )

    out_file = Path("artifacts/blocking_benchmark_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved blocking benchmark report to {out_file}")


if __name__ == "__main__":
    main()
