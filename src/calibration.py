"""Validation-fitted probability calibration for held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class ProbabilityCalibrator:
    method: str
    estimator: Any

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if self.method == "platt":
            clipped = np.clip(values, 1e-7, 1 - 1e-7)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            return self.estimator.predict_proba(logits)[:, 1]
        if self.method == "isotonic":
            return np.clip(self.estimator.predict(values), 0.0, 1.0)
        raise ValueError(f"Unknown calibration method: {self.method}")


def fit_calibrator(y_validation: Sequence[int], scores_validation: Sequence[float], method: str) -> ProbabilityCalibrator:
    """Fit Platt or isotonic calibration on validation observations only."""
    if len(y_validation) != len(scores_validation) or len(y_validation) < 2:
        raise ValueError("Validation labels and scores must have equal length >= 2")
    if len(set(map(int, y_validation))) != 2:
        raise ValueError("Calibration requires both label classes in validation")
    y = np.asarray(y_validation, dtype=int)
    scores = np.asarray(scores_validation, dtype=float)
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Calibration scores must be finite probabilities in [0,1]")
    if method == "platt":
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise RuntimeError("Platt calibration requires scikit-learn") from exc
        clipped = np.clip(scores, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=0).fit(logits, y)
    elif method == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise RuntimeError("Isotonic calibration requires scikit-learn") from exc
        estimator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, y)
    else:
        raise ValueError("method must be 'platt' or 'isotonic'")
    return ProbabilityCalibrator(method, estimator)


def calibration_curve_data(y_true: Sequence[int], probabilities: Sequence[float], *, n_bins: int = 10) -> list[dict[str, float | int]]:
    """Equal-width reliability bins; empty bins are omitted."""
    if len(y_true) != len(probabilities) or n_bins < 1:
        raise ValueError("Inputs must have equal length and n_bins must be positive")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Calibration data must be non-empty with probabilities in [0,1]")
    index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    output = []
    for bin_number in range(n_bins):
        mask = index == bin_number
        if np.any(mask):
            output.append({"bin": bin_number, "count": int(mask.sum()),
                           "mean_predicted_probability": float(p[mask].mean()),
                           "observed_positive_rate": float(y[mask].mean())})
    return output
