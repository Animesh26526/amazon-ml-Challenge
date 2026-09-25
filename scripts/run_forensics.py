"""Comprehensive Streaming Data Forensics Script.

Profiles data integrity, name/address distributions, missingness,
and ground-truth characteristics without loading full multi-gigabyte datasets into memory.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalization import (
    clean_unicode,
    extract_numeric_anchors,
    normalize_address,
    normalize_business_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def profile_source_file(file_path: Path, max_rows: int | None = None) -> Dict[str, Any]:
    """Stream a source TSV file and compute structural forensics."""
    logger.info(f"Profiling {file_path.name}...")
    start_time = time.time()

    total_rows = 0
    missing_names = 0
    missing_addrs = 0
    empty_countries = 0
    country_counts = Counter()
    malformed_ids = 0

    name_len_sum = 0
    addr_len_sum = 0
    valid_addr_count = 0

    # Sample top frequent names/addresses using top Counter
    top_names = Counter()
    top_addrs = Counter()
    top_tokens = Counter()

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)

        for row in reader:
            if not row:
                continue
            total_rows += 1

            eid = row[0].strip() if len(row) > 0 else ""
            bname = row[1].strip() if len(row) > 1 else ""
            baddr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip() if len(row) > 3 else ""

            # ID check
            if not (eid.startswith("S1-") or eid.startswith("S2-") or eid.startswith("S3-")):
                malformed_ids += 1

            # Name stats
            if not bname:
                missing_names += 1
            else:
                name_len_sum += len(bname)
                if total_rows <= 500000:  # Sample for top frequent
                    norm_n = normalize_business_name(bname)
                    top_names[norm_n] += 1
                    for tok in norm_n.split():
                        if len(tok) >= 3:
                            top_tokens[tok] += 1

            # Address stats
            if not baddr:
                missing_addrs += 1
            else:
                addr_len_sum += len(baddr)
                valid_addr_count += 1
                if total_rows <= 500000:
                    norm_a = normalize_address(baddr)
                    top_addrs[norm_a] += 1

            # Country
            if not country:
                empty_countries += 1
            else:
                country_counts[country] += 1

            if max_rows and total_rows >= max_rows:
                break

    elapsed = time.time() - start_time
    logger.info(f"Finished {file_path.name} in {elapsed:.1f}s ({total_rows:,} rows)")

    return {
        "file": file_path.name,
        "total_rows": total_rows,
        "header": header,
        "missing_names": missing_names,
        "missing_addrs": missing_addrs,
        "missing_addrs_pct": round((missing_addrs / total_rows * 100), 3) if total_rows else 0.0,
        "empty_countries": empty_countries,
        "country_distribution": dict(country_counts),
        "malformed_ids": malformed_ids,
        "avg_name_length": round(name_len_sum / (total_rows - missing_names), 2) if (total_rows - missing_names) else 0.0,
        "avg_addr_length": round(addr_len_sum / valid_addr_count, 2) if valid_addr_count else 0.0,
        "top_5_names": top_names.most_common(5),
        "top_5_addrs": top_addrs.most_common(5),
        "top_5_tokens": top_tokens.most_common(5),
    }


def profile_ground_truth(file_path: Path, max_rows: int | None = None) -> Dict[str, Any]:
    """Profile ground truth matching relationships and cardinality."""
    logger.info(f"Profiling ground truth {file_path.name}...")
    start_time = time.time()

    total_s1 = 0
    zero_matches = 0
    one_match = 0
    multi_match = 0
    cardinality_counts = Counter()

    s2_only = 0
    s3_only = 0
    both_sources = 0

    total_true_links = 0
    max_matches_seen = 0

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)

        for row in reader:
            if not row:
                continue
            total_s1 += 1
            s1_id = row[0].strip() if len(row) > 0 else ""
            match_str = row[1].strip() if len(row) > 1 else ""

            matches = [m.strip() for m in match_str.split(",") if m.strip()] if match_str else []
            k = len(matches)
            cardinality_counts[k] += 1
            total_true_links += k
            if k > max_matches_seen:
                max_matches_seen = k

            if k == 0:
                zero_matches += 1
            elif k == 1:
                one_match += 1
            else:
                multi_match += 1

            if k > 0:
                has_s2 = any(m.startswith("S2-") for m in matches)
                has_s3 = any(m.startswith("S3-") for m in matches)
                if has_s2 and not has_s3:
                    s2_only += 1
                elif has_s3 and not has_s2:
                    s3_only += 1
                elif has_s2 and has_s3:
                    both_sources += 1

            if max_rows and total_s1 >= max_rows:
                break

    elapsed = time.time() - start_time
    logger.info(f"Finished ground truth in {elapsed:.1f}s ({total_s1:,} S1 rows)")

    return {
        "file": file_path.name,
        "total_s1": total_s1,
        "zero_matches": zero_matches,
        "zero_matches_pct": round(zero_matches / total_s1 * 100, 3) if total_s1 else 0.0,
        "one_match": one_match,
        "one_match_pct": round(one_match / total_s1 * 100, 3) if total_s1 else 0.0,
        "multi_match": multi_match,
        "multi_match_pct": round(multi_match / total_s1 * 100, 3) if total_s1 else 0.0,
        "s2_only": s2_only,
        "s3_only": s3_only,
        "both_sources": both_sources,
        "total_true_links": total_true_links,
        "avg_matches_per_s1": round(total_true_links / total_s1, 3) if total_s1 else 0.0,
        "max_matches_seen": max_matches_seen,
        "cardinality_distribution": dict(sorted(cardinality_counts.items())),
    }


def main() -> None:
    data_dir = Path("data/raw")
    reports = {}

    # 1. Profile Train Ground Truth
    gt_file = data_dir / "train" / "train_ground_truth.tsv"
    if gt_file.is_file():
        reports["ground_truth"] = profile_ground_truth(gt_file)

    # 2. Profile Source files
    source_files = [
        data_dir / "train" / "train_source1.tsv",
        data_dir / "train" / "train_source2.tsv",
        data_dir / "train" / "train_source3.tsv",
        data_dir / "test" / "test_source1.tsv",
        data_dir / "test" / "test_source2.tsv",
        data_dir / "test" / "test_source3.tsv",
    ]

    for sf in source_files:
        if sf.is_file():
            reports[sf.name] = profile_source_file(sf)

    output_path = Path("artifacts/data_forensics_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    logger.info(f"Data forensics report saved to {output_path}")


if __name__ == "__main__":
    main()
