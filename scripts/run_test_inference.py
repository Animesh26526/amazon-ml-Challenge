"""Full Test Inference Pipeline.

Streams test_source1.tsv partitioned by country (France, US, India),
generates candidate pairs using multi-pass blocking, extracts pairwise
features, applies the calibrated XGBoost baseline matcher with optimal
decision threshold, enforces the candidate subset invariant, and writes
competition-compliant matching_results.tsv and candidate_pairs.tsv.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import joblib
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import CandidatePair
from src.decision.entity_decision import decide_matches_for_entity
from src.io import (
    CANDIDATE_PAIRS_COLUMNS,
    MATCHING_RESULTS_COLUMNS,
    format_id_list,
    verify_submission_invariants,
)
from src.normalization import NormalizedRecord
from src.pipeline import MultiPassCandidateGenerator, extract_pairwise_feature_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_inference")


from scripts.partition_test_data import partition_test_sources


def stream_country_s1(
    cache_path: Path,
) -> Iterator[NormalizedRecord]:
    """Stream NormalizedRecord from pre-partitioned country S1 cache."""
    with cache_path.open("r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if len(row) >= 4:
                yield NormalizedRecord.from_raw(row[0], row[1], row[2], row[3])


def load_country_targets(
    cache_path: Path,
    target_country: str,
) -> Dict[str, NormalizedRecord]:
    """Load target records (S2 and S3) for a given country from cache."""
    targets: Dict[str, NormalizedRecord] = {}
    logger.info(f"Loading pre-partitioned targets from {cache_path.name} for country '{target_country}'...")
    with cache_path.open("r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if len(row) >= 4:
                tid = row[0].strip()
                targets[tid] = NormalizedRecord.from_raw(tid, row[1], row[2], row[3], store_raw=False)

    logger.info(f"Total targets loaded for '{target_country}': {len(targets):,}")
    return targets


def run_country_inference(
    country: str,
    cache_dir: Path,
    model,
    threshold: float,
    output_dir: Path,
    batch_size: int = 5000,
) -> Tuple[int, int]:
    """Run candidate generation, feature extraction, and prediction for one country."""
    logger.info(f"============================================================")
    logger.info(f"STARTING COUNTRY PARTITION: {country.upper()}")
    logger.info(f"============================================================")
    t0 = time.time()

    matching_part_path = output_dir / f"temp_matching_{country}.tsv"
    candidate_part_path = output_dir / f"temp_candidates_{country}.tsv"
    s1_cache_path = cache_dir / f"s1_{country}.tsv"

    if matching_part_path.is_file() and matching_part_path.stat().st_size > 1000:
        logger.info(f"Country partition {country.upper()} output already exists ({matching_part_path.name}). Skipping re-computation.")
        return 0, 0

    # 1. Load target records for country
    target_cache_path = cache_dir / f"targets_{country}.tsv"
    targets = load_country_targets(target_cache_path, country)

    # 2. Build multi-pass candidate generator
    generator = MultiPassCandidateGenerator(max_candidates_per_key=50)
    generator.build_indexes(targets.values())

    # 3. Stream S1 entities in batches
    total_s1 = 0
    total_matches = 0

    with matching_part_path.open("w", encoding="utf-8", newline="") as f_m, \
         candidate_part_path.open("w", encoding="utf-8", newline="") as f_c:

        w_m = csv.writer(f_m, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")
        w_c = csv.writer(f_c, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")

        batch_s1: List[NormalizedRecord] = []

        def process_batch(batch: List[NormalizedRecord]):
            nonlocal total_s1, total_matches
            # Collect all candidate pairs for batch
            pairs_to_score: List[Tuple[NormalizedRecord, NormalizedRecord, CandidatePair]] = []
            cand_lists: Dict[str, List[str]] = {}

            for s1_rec in batch:
                cands = generator.generate_candidates_for_record(s1_rec)
                cand_lists[s1_rec.entity_id] = list(cands.keys())
                for tid, cand_pair in cands.items():
                    if tid in targets:
                        pairs_to_score.append((s1_rec, targets[tid], cand_pair))

            # Feature extraction & scoring
            preds_by_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
            if pairs_to_score:
                feat_matrix = [
                    extract_pairwise_feature_vector(s1_r, tgt_r, cp)
                    for s1_r, tgt_r, cp in pairs_to_score
                ]
                X = np.array(feat_matrix, dtype=np.float32)
                probs = model.predict_proba(X)[:, 1]

                for (s1_r, tgt_r, _), p in zip(pairs_to_score, probs):
                    preds_by_s1[s1_r.entity_id].append((tgt_r.entity_id, float(p)))

            # Decision making and writing
            for s1_rec in batch:
                s1_id = s1_rec.entity_id
                all_cands = cand_lists.get(s1_id, [])
                scored = preds_by_s1.get(s1_id, [])

                # Apply calibrated threshold decision
                matches = decide_matches_for_entity(scored, threshold=threshold, max_matches=10)

                # Hard invariant: matches <= candidates
                cand_set = set(all_cands)
                valid_matches = [m for m in matches if m in cand_set]

                w_m.writerow([s1_id, format_id_list(valid_matches)])
                w_c.writerow([s1_id, format_id_list(all_cands)])

                total_s1 += 1
                total_matches += len(valid_matches)

        for s1_rec in stream_country_s1(s1_cache_path):
            batch_s1.append(s1_rec)
            if len(batch_s1) >= batch_size:
                process_batch(batch_s1)
                batch_s1 = []
                if total_s1 % 25000 == 0:
                    elapsed = time.time() - t0
                    rate = total_s1 / elapsed if elapsed > 0 else 0
                    logger.info(f"[{country}] Processed {total_s1:,} entities ({rate:.0f} ent/s, matches: {total_matches:,})...")

        if batch_s1:
            process_batch(batch_s1)

    elapsed = time.time() - t0
    logger.info(f"Completed {country.upper()}: {total_s1:,} entities, {total_matches:,} matches in {elapsed:.1f}s.")

    # Explicit memory cleanup
    del targets
    del generator
    gc.collect()

    return total_s1, total_matches


def assemble_final_submission(
    countries: List[str],
    test_dir: Path,
    output_dir: Path,
) -> None:
    """Concatenate or re-order country partitions to form canonical submission files."""
    final_matching_path = output_dir / "matching_results.tsv"
    final_candidate_path = output_dir / "candidate_pairs.tsv"

    logger.info("Assembling final submission files preserving exact test_source1.tsv order...")
    t0 = time.time()

    # Load all country parts into memory for fast ordered lookup
    # 1.73M mapping of id -> formatted string
    matching_dict: Dict[str, str] = {}
    candidate_dict: Dict[str, str] = {}

    for c in countries:
        m_part = output_dir / f"temp_matching_{c}.tsv"
        c_part = output_dir / f"temp_candidates_{c}.tsv"

        logger.info(f"Reading temporary parts for {c}...")
        with m_part.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) == 2:
                    matching_dict[parts[0]] = parts[1]
                elif len(parts) == 1:
                    matching_dict[parts[0]] = ""

        with c_part.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) == 2:
                    candidate_dict[parts[0]] = parts[1]
                elif len(parts) == 1:
                    candidate_dict[parts[0]] = ""

    # Stream test_source1.tsv to guarantee 100% order preservation and coverage
    s1_path = test_dir / "test_source1.tsv"
    written_count = 0

    with final_matching_path.open("w", encoding="utf-8", newline="") as f_m, \
         final_candidate_path.open("w", encoding="utf-8", newline="") as f_c:

        w_m = csv.writer(f_m, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")
        w_c = csv.writer(f_c, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")

        w_m.writerow(MATCHING_RESULTS_COLUMNS)
        w_c.writerow(CANDIDATE_PAIRS_COLUMNS)

        with s1_path.open("r", encoding="utf-8") as f_s1:
            r = csv.reader(f_s1, delimiter="\t")
            next(r, None)
            for row in r:
                s1_id = row[0].strip()
                m_str = matching_dict.get(s1_id, "")
                c_str = candidate_dict.get(s1_id, "")
                w_m.writerow([s1_id, m_str])
                w_c.writerow([s1_id, c_str])
                written_count += 1

    # Cleanup temporary part files
    for c in countries:
        (output_dir / f"temp_matching_{c}.tsv").unlink(missing_ok=True)
        (output_dir / f"temp_candidates_{c}.tsv").unlink(missing_ok=True)

    logger.info(f"Successfully assembled {written_count:,} rows into {final_matching_path} and {final_candidate_path} in {time.time()-t0:.1f}s.")


def main():
    test_dir = Path("data/raw/test")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained model artifact
    model_path = Path("artifacts/baseline_model.joblib")
    if not model_path.is_file():
        logger.error(f"Trained model not found at {model_path}! Run training first.")
        sys.exit(1)
    logger.info(f"Loading trained matcher from {model_path}...")
    model = joblib.load(model_path)

    # 2. Load optimal threshold from validation report
    report_path = Path("artifacts/baseline_validation_report.json")
    threshold = 0.55
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as f:
            rep = json.load(f)
            threshold = float(rep.get("optimal_threshold", 0.55))
            logger.info(f"Loaded calibrated threshold from report: {threshold:.3f}")
    else:
        logger.warning(f"Validation report not found at {report_path}, using default threshold {threshold:.3f}")

    cache_dir = Path("data/cache")
    if not (cache_dir / "targets_france.tsv").is_file():
        logger.info("Partition caches not found. Running one-pass partitioner...")
        partition_test_sources()

    # 3. Process countries sequentially to keep peak memory < 1.5 GB
    countries = ["france", "us", "india"]
    for c in countries:
        run_country_inference(c, cache_dir, model, threshold, output_dir)

    # 4. Assemble final submission TSVs
    assemble_final_submission(countries, test_dir, output_dir)

    # 5. Invariant check
    logger.info("Running internal submission invariant verification...")
    valid, errors = verify_submission_invariants(
        matching_path=output_dir / "matching_results.tsv",
        candidate_path=output_dir / "candidate_pairs.tsv",
        test_source1_path=test_dir / "test_source1.tsv",
    )
    if not valid:
        logger.error(f"Internal submission invariant verification FAILED with {len(errors)} errors:")
        for err in errors[:10]:
            logger.error(f"  - {err}")
        sys.exit(1)
    logger.info("Submission invariants verified successfully!")


if __name__ == "__main__":
    main()
