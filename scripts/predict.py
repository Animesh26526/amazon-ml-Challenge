#!/usr/bin/env python3
"""Inference and Submission File Generation Script Interface.

Runs pairwise scoring, entity-level decisions, and writes:
- output/matching_results.tsv
- output/candidate_pairs.tsv
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
    parser = argparse.ArgumentParser(description="Run inference and generate submission files.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing test TSVs")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for TSVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Loaded predict configuration for experiment {cfg.get('experiment', {}).get('id')}")
    logger.info("Predict script interface ready.")


if __name__ == "__main__":
    main()
