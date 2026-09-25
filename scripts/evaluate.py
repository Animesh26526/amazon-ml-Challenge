#!/usr/bin/env python3
"""Evaluation Script Interface.

Computes macro F_0.5, candidate recall, and subgroup diagnostic cohorts on validation data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add repository root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io import load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    parser.add_argument("--predictions", type=str, default=None, help="Path to matching_results.tsv")
    parser.add_argument("--candidates", type=str, default=None, help="Path to candidate_pairs.tsv")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to ground truth TSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Loaded evaluation configuration for experiment {cfg.get('experiment', {}).get('id')}")
    logger.info("Evaluation script interface ready.")


if __name__ == "__main__":
    main()
