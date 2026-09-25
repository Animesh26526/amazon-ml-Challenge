"""Matching Model Interfaces and Base Classes.

Defines the core model protocol for scoring candidate pairs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseMatcher(ABC):
    """Abstract base class for pairwise matching models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> BaseMatcher:
        """Train the pairwise matching model on feature matrix X and binary labels y."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probability for candidate pairs. Returns 1D array of floats."""
        pass

    @abstractmethod
    def save(self, model_path: str) -> None:
        """Save model checkpoint to disk."""
        pass

    @abstractmethod
    def load(self, model_path: str) -> BaseMatcher:
        """Load model checkpoint from disk."""
        pass
