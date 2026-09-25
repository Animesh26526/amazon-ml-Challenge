#!/usr/bin/env python3
"""Data Profiling and Forensics Script.

Performs lightweight schema, row count, and missing-value inspections across
raw competition TSV datasets without loading multi-gigabyte files entirely into memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add repository root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io import inspect_file_metadata, load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile dataset files and verify integrity.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional data directory override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    data_files = cfg.get("data", {})

    logger.info("Starting Phase 0 metadata inspection of dataset files...")

    report: Dict[str, Any] = {}
    for key, rel_path in data_files.items():
        p = Path(rel_path)
        if p.is_file():
            meta = inspect_file_metadata(p)
            report[key] = meta
            logger.info(f"Verified {key}: {meta['size_mb']} MB, header={meta['header']}")
        else:
            logger.warning(f"File {key} not found at {p}")
            report[key] = {"path": str(p), "status": "missing"}

    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
