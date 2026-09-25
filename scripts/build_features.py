#!/usr/bin/env python3
"""Pairwise Feature Engineering Script Interface.

Computes pairwise name, address, interaction, and route provenance features for candidate pairs.
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
    parser = argparse.ArgumentParser(description="Extract pairwise features for candidate pairs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    parser.add_argument("--candidates", type=str, default=None, help="Candidate pairs input file")
    parser.add_argument("--output", type=str, default=None, help="Destination feature matrix path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Loaded feature configuration for experiment {cfg.get('experiment', {}).get('id')}")
    logger.info("Feature builder interface ready for feature generation phase.")


if __name__ == "__main__":
    main()
