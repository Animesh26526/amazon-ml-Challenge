"""Instant Assembly, Validation, and Packaging Script.

Assembles all available test partitions (France 100%, US 100%, India available),
fills any remaining entities safely with empty match lists (valid singletons),
runs the official challenge/validate_submission.py, and packages the final
output/submission_baseline_v0.zip for immediate upload before the deadline.
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("emergency_package")

MATCHING_RESULTS_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_PAIRS_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def main():
    t0 = time.time()
    test_dir = Path("data/raw/test")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_matching_path = output_dir / "matching_results.tsv"
    final_candidate_path = output_dir / "candidate_pairs.tsv"
    zip_path = output_dir / "submission_baseline_v0.zip"

    countries = ["france", "us", "india"]
    matching_dict: Dict[str, str] = {}
    candidate_dict: Dict[str, str] = {}

    for c in countries:
        m_part = output_dir / f"temp_matching_{c}.tsv"
        c_part = output_dir / f"temp_candidates_{c}.tsv"

        if m_part.is_file():
            logger.info(f"Loading matches from {m_part.name}...")
            with m_part.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\r\n").split("\t", 1)
                    if len(parts) == 2:
                        matching_dict[parts[0]] = parts[1]
                    elif len(parts) == 1:
                        matching_dict[parts[0]] = ""

        if c_part.is_file():
            logger.info(f"Loading candidates from {c_part.name}...")
            with c_part.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\r\n").split("\t", 1)
                    if len(parts) == 2:
                        candidate_dict[parts[0]] = parts[1]
                    elif len(parts) == 1:
                        candidate_dict[parts[0]] = ""

    logger.info(f"Total scored entities loaded into memory: {len(matching_dict):,}")

    # Stream test_source1.tsv to guarantee 100% order preservation and coverage
    s1_path = test_dir / "test_source1.tsv"
    if not s1_path.is_file():
        logger.error(f"test_source1.tsv not found at {s1_path}!")
        sys.exit(1)

    logger.info("Writing complete 1.73M canonical output files...")
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

    logger.info(f"Successfully assembled {written_count:,} rows into {final_matching_path.name} and {final_candidate_path.name} in {time.time()-t0:.1f}s.")

    # Run official organizer validator
    validator_path = Path("challenge/validate_submission.py")
    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(final_matching_path),
        "--candidate", str(final_candidate_path),
        "--test-dir", str(test_dir),
    ]
    logger.info(f"Executing official validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        logger.error(f"Official validator FAILED with exit code {result.returncode}!")
        sys.exit(1)

    logger.info("Official validator PASSED with exit code 0 (PASS)!")

    # Package into submission zip
    logger.info(f"Packaging {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(final_matching_path, arcname="matching_results.tsv")
        zf.write(final_candidate_path, arcname="candidate_pairs.tsv")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"============================================================")
    logger.info(f"SUBMISSION 1 READY FOR UPLOAD: {zip_path} ({zip_size_mb:.2f} MB)")
    logger.info(f"Leaderboard file: {final_matching_path}")
    logger.info(f"Total time elapsed: {time.time()-t0:.1f}s")
    logger.info(f"============================================================")


if __name__ == "__main__":
    main()
