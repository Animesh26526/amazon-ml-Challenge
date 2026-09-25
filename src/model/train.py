"""Pairwise Matching Model Training.

Implements gradient-boosted decision tree training for pairwise candidate matching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.model import BaseMatcher

logger = logging.getLogger(__name__)


class GradientBoostedMatcher(BaseMatcher):
    """Gradient boosted decision tree pairwise matcher."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.model: Optional[Any] = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                random_state=self.random_state,
                eval_metric="logloss",
                tree_method="hist",
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> GradientBoostedMatcher:
        """Train the matcher."""
        if self.model is None:
            self._init_model()
        self.model.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of match."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Binary classification probability of class 1
            return probs[:, 1] if probs.ndim == 2 else probs
        return self.model.predict(X).astype(float)

    def save(self, model_path: str) -> None:
        """Save model artifact to disk."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, model_path: str) -> GradientBoostedMatcher:
        """Load model artifact from disk."""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self
