"""Packaging and Submission Validation Script.

Verifies submission TSVs against official organizer rules, runs the official
challenge/validate_submission.py script, and packages the outputs into
output/submission_baseline_v0.zip.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("package_submission")


def verify_and_package(
    matching_path: Path = Path("output/matching_results.tsv"),
    candidate_path: Path = Path("output/candidate_pairs.tsv"),
    test_dir: Path = Path("data/raw/test"),
    zip_path: Path = Path("output/submission_baseline_v0.zip"),
) -> bool:
    logger.info("============================================================")
    logger.info("VERIFYING SUBMISSION FILES BEFORE PACKAGING")
    logger.info("============================================================")

    if not matching_path.is_file():
        logger.error(f"Missing matching results file: {matching_path}")
        return False
    if not candidate_path.is_file():
        logger.error(f"Missing candidate pairs file: {candidate_path}")
        return False

    # Check file sizes
    logger.info(f"matching_results.tsv size: {matching_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"candidate_pairs.tsv size: {candidate_path.stat().st_size / (1024*1024):.2f} MB")

    # Run official organizer validator
    validator_path = Path("challenge/validate_submission.py")
    if not validator_path.is_file():
        logger.error(f"Official validator not found at {validator_path}")
        return False

    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
    logger.info(f"Executing official validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        logger.error(f"Official validator FAILED with exit code {result.returncode}!")
        return False

    logger.info("Official validator PASSED with exit code 0!")

    # Create submission zip
    logger.info(f"Creating submission package at {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_path, arcname="matching_results.tsv")
        zf.write(candidate_path, arcname="candidate_pairs.tsv")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully created {zip_path} ({zip_size_mb:.2f} MB)")
    logger.info("Submission package is ready for upload!")
    return True


if __name__ == "__main__":
    success = verify_and_package()
    if not success:
        sys.exit(1)
