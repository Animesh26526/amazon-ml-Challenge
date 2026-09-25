#!/usr/bin/env python3
"""Pairwise Matching Model Training Script Interface.

Trains pairwise gradient-boosted matching model (XGBoost / CatBoost / HistGBDT).
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
    parser = argparse.ArgumentParser(description="Train pairwise matching model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    parser.add_argument("--features", type=str, default=None, help="Path to training feature matrix")
    parser.add_argument("--model-output", type=str, default=None, help="Path to save trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Loaded training configuration for experiment {cfg.get('experiment', {}).get('id')}")
    logger.info("Train script interface ready.")


if __name__ == "__main__":
    main()
