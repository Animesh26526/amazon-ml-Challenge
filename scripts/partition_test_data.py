"""Partition test datasets by country in a single sequential streaming pass.

Enables fast, low-memory independent processing of each country cohort
without redundant 10-million-row scans.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("partition_test")


def partition_test_sources():
    test_dir = Path("data/raw/test")
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    countries = ["france", "us", "india"]

    # 1. Partition Target Sources (S2 and S3)
    target_files = {
        c: (cache_dir / f"targets_{c}.tsv").open("w", encoding="utf-8", newline="")
        for c in countries
    }
    target_writers = {c: csv.writer(f, delimiter="\t", lineterminator="\n") for c, f in target_files.items()}

    t0 = time.time()
    logger.info("Partitioning test_source2.tsv and test_source3.tsv...")
    counts = {c: 0 for c in countries}

    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = test_dir / fname
        logger.info(f"Streaming {fname}...")
        with path.open("r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)  # skip header
            for row in r:
                if len(row) >= 4:
                    c = row[3].strip().lower()
                    if c in target_writers:
                        target_writers[c].writerow(row)
                        counts[c] += 1

    for f in target_files.values():
        f.close()

    logger.info(f"Target partitioning completed in {time.time()-t0:.1f}s: {counts}")

    # 2. Partition Source 1 (S1 queries)
    t1 = time.time()
    s1_files = {
        c: (cache_dir / f"s1_{c}.tsv").open("w", encoding="utf-8", newline="")
        for c in countries
    }
    s1_writers = {c: csv.writer(f, delimiter="\t", lineterminator="\n") for c, f in s1_files.items()}
    s1_counts = {c: 0 for c in countries}

    s1_path = test_dir / "test_source1.tsv"
    logger.info("Streaming test_source1.tsv...")
    with s1_path.open("r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)  # skip header
        for row in r:
            if len(row) >= 4:
                c = row[3].strip().lower()
                if c in s1_writers:
                    s1_writers[c].writerow(row)
                    s1_counts[c] += 1

    for f in s1_files.values():
        f.close()

    logger.info(f"S1 partitioning completed in {time.time()-t1:.1f}s: {s1_counts}")
    logger.info("All test partitions successfully created in data/cache/")


if __name__ == "__main__":
    partition_test_sources()
