"""Leakage-aware pair-level logistic models for synthetic ASR error risk.

All preprocessing is fitted on training data only. Identifiers, ASR text,
reference text, labels, filenames, and requested DSP settings are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .statistical_analysis import binary_metrics


FEATURE_GROUPS = {
    "phonetic": ("phoneme_similarity", "phoneme_edit_distance", "syllable_difference",
                 "stress_pattern_similarity", "articulatory_place_mismatch",
                 "articulatory_manner_mismatch", "articulatory_voicing_mismatch",
                 "soundex_similarity_proxy"),
    "lexical": ("jaro_winkler", "character_similarity", "normalized_character_edit_distance",
                "absolute_length_difference", "common_prefix_fraction"),
    "contextual": ("context_available", "context_pseudo_nll_a", "context_pseudo_nll_b",
                   "context_pseudo_nll_difference", "context_embedding_cosine"),
    "acoustic": ("measured_snr_db", "output_snr_db", "measured_rt60_s", "duration_s",
                 "clipped_sample_fraction", "rms_dbfs", "peak_dbfs"),
    "recognizer": ("asr_confidence",),
}
CATEGORICAL_FEATURES = ("recognizer", "model_id", "condition_family", "context_condition", "voice")


@dataclass
class FittedPairRiskModel:
    feature_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    categories: dict[str, tuple[str, ...]]
    estimator: Any
    feature_group_names: tuple[str, ...]

    def _matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        values = []
        for row in rows:
            numeric = []
            for name in self.feature_names:
                try:
                    value = float(row.get(name))
                    numeric.append(value if np.isfinite(value) else np.nan)
                except (TypeError, ValueError):
                    numeric.append(np.nan)
            values.append(numeric)
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2:
            matrix = np.empty((0, len(self.feature_names)), dtype=float)
        for column in range(matrix.shape[1]):
            matrix[~np.isfinite(matrix[:, column]), column] = self.medians[column]
        if matrix.shape[1]:
            matrix = (matrix - self.means) / self.scales
        one_hot = []
        for name, categories in self.categories.items():
            observed = [str(row.get(name) or "__MISSING__") for row in rows]
            for category in categories:
                one_hot.append([float(value == category) for value in observed])
        if one_hot:
            matrix = np.column_stack((matrix, np.asarray(one_hot, dtype=float).T))
        return matrix

    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=float)
        return self.estimator.predict_proba(self._matrix(rows))[:, 1]

    def coefficients(self) -> dict[str, float]:
        numeric_count = len(self.feature_names)
        return {name: float(value) for name, value in zip(
            list(self.feature_names) + [f"{field}={category}" for field, cats in self.categories.items() for category in cats],
            self.estimator.coef_[0][:numeric_count + sum(map(len, self.categories.values()))])}


def fit_pair_risk_model(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[int], *,
    feature_groups: Sequence[str] = ("phonetic", "lexical", "contextual", "acoustic", "recognizer"),
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    c: float = 1.0,
) -> FittedPairRiskModel:
    """Fit regularized logistic regression; this module only sees training rows."""
    if len(rows) != len(labels) or not rows:
        raise ValueError("rows and labels must have equal non-zero length")
    if len(set(map(int, labels))) != 2:
        raise ValueError("Training requires examples from both classes")
    unknown = set(feature_groups) - FEATURE_GROUPS.keys()
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    names = tuple(dict.fromkeys(name for group in feature_groups for name in FEATURE_GROUPS[group]))
    medians = []
    for name in names:
        vals = []
        for row in rows:
            try:
                value = float(row.get(name))
                if np.isfinite(value):
                    vals.append(value)
            except (TypeError, ValueError):
                pass
        medians.append(float(np.median(vals)) if vals else 0.0)
    categories = {
        name: tuple(sorted({str(row.get(name) or "__MISSING__") for row in rows}))
        for name in categorical_features
    }
    numeric_matrix = np.asarray([
        [float(row[name]) if row.get(name) is not None and np.isfinite(float(row[name])) else medians[i]
         for i, name in enumerate(names)] for row in rows
    ], dtype=float)
    means = numeric_matrix.mean(axis=0) if names else np.asarray([], dtype=float)
    scales = numeric_matrix.std(axis=0) if names else np.asarray([], dtype=float)
    scales[scales < 1e-12] = 1.0
    model = FittedPairRiskModel(names, np.asarray(medians), means, scales, categories, None, tuple(feature_groups))
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("Model fitting requires scikit-learn; install it with `python -m pip install scikit-learn`.") from exc
    model.estimator = LogisticRegression(C=c, max_iter=2000, solver="liblinear", random_state=0)
    model.estimator.fit(model._matrix(rows), np.asarray(labels, dtype=int))
    return model


def similarity_baseline(rows: Sequence[Mapping[str, Any]], feature: str = "soundex_similarity_proxy") -> np.ndarray:
    """Probability-like ranking baseline; scores are normalized to [0,1]."""
    vals = []
    for row in rows:
        value = row.get(feature)
        try:
            score = float(value)
            vals.append(float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else 0.0)
        except (TypeError, ValueError):
            vals.append(0.0)
    return np.asarray(vals)


def evaluate_model(model: FittedPairRiskModel, rows: Sequence[Mapping[str, Any]], labels: Sequence[int]) -> dict[str, Any]:
    probabilities = model.predict_proba(rows)
    return {"metrics": binary_metrics(labels, probabilities), "probabilities": probabilities.tolist(),
            "coefficients": model.coefficients(), "feature_groups": list(model.feature_group_names)}
