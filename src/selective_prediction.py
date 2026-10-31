"""Validation-only threshold selection and selective medication auditing.

The threshold is selected on validation scores, then frozen for test use. The
finite-sample bound controls the false-positive rate among negative mentions;
the reported false alerts per 100 mentions also accounts for validation
prevalence. This is an auditing aid and never rewrites an ASR transcript.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    budget_per_100_mentions: float
    alpha: float
    validation_n: int
    validation_negatives: int
    validation_false_positives: int
    upper_false_positive_rate: float
    upper_false_alerts_per_100: float
    validation_recall: float
    guarantee_met: bool


def _cp_upper(false_positives: int, negatives: int, alpha: float) -> float:
    if negatives <= 0:
        return 1.0
    if false_positives >= negatives:
        return 1.0
    if false_positives == 0:
        # Exact one-sided Clopper-Pearson upper confidence bound.
        return 1.0 - alpha ** (1.0 / negatives)
    try:
        from scipy.stats import beta
        return float(beta.ppf(1.0 - alpha, false_positives + 1, negatives - false_positives))
    except ImportError as exc:
        raise RuntimeError("Finite-sample selective thresholding requires scipy.") from exc


def select_threshold(
    y_true: Sequence[int], scores: Sequence[float], *,
    budget_per_100_mentions: float, alpha: float = 0.05,
) -> ThresholdSelection:
    """Choose the highest-recall empirical threshold meeting an upper risk bound.

    A union-bound adjustment over candidate thresholds keeps the overall
    one-sided confidence level at least ``1-alpha``. Threshold candidates are
    observed validation scores plus an always-negative threshold.
    """
    if len(y_true) != len(scores) or not y_true:
        raise ValueError("y_true and scores must have equal non-zero length")
    if budget_per_100_mentions < 0 or not 0 < alpha < 1:
        raise ValueError("budget must be >= 0 and alpha must be in (0, 1)")
    labels = [int(value) for value in y_true]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("Labels must be binary 0/1")
    values = [float(value) for value in scores]
    if any(not (value >= 0 and value <= 1) for value in values):
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    positives = sum(labels)
    negatives = len(labels) - positives
    prevalence = positives / len(labels)
    candidates = sorted(set(values) | {1.000000000001})
    adjusted_alpha = alpha / len(candidates)
    feasible: list[tuple[float, float, int, int, float, float]] = []
    for threshold in candidates:
        predicted = [score >= threshold for score in values]
        fp = sum(flag and label == 0 for flag, label in zip(predicted, labels))
        tp = sum(flag and label == 1 for flag, label in zip(predicted, labels))
        upper_fpr = _cp_upper(fp, negatives, adjusted_alpha)
        upper_alerts = 100.0 * (1.0 - prevalence) * upper_fpr
        recall = tp / positives if positives else 0.0
        if upper_alerts <= budget_per_100_mentions:
            feasible.append((recall, threshold, fp, tp, upper_fpr, upper_alerts))
    if feasible:
        recall, threshold, fp, tp, upper_fpr, upper_alerts = max(feasible, key=lambda row: (row[0], row[1]))
        met = True
    else:
        threshold, fp, tp = 1.000000000001, 0, 0
        upper_fpr = _cp_upper(0, negatives, adjusted_alpha)
        upper_alerts = 100.0 * (1.0 - prevalence) * upper_fpr
        recall, met = 0.0, False
    return ThresholdSelection(threshold, budget_per_100_mentions, alpha, len(labels), negatives,
                              fp, upper_fpr, upper_alerts, recall, met)


def apply_threshold(scores: Sequence[float], selection: ThresholdSelection | Mapping[str, Any]) -> list[int]:
    threshold = selection.threshold if isinstance(selection, ThresholdSelection) else float(selection["threshold"])
    return [int(float(score) >= threshold) for score in scores]


def risk_coverage_curve(y_true: Sequence[int], scores: Sequence[float]) -> list[dict[str, float]]:
    """Return empirical alert precision/risk across increasing coverage levels."""
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have equal length")
    ranked = sorted(zip(map(int, y_true), map(float, scores)), key=lambda row: row[1], reverse=True)
    positives = sum(label for label, _ in ranked)
    output = []
    errors = 0
    for index, (label, _) in enumerate(ranked, start=1):
        errors += 1 - label
        output.append({"coverage": index / len(ranked), "alert_risk": errors / index,
                       "error_detection_rate": (index - errors) / positives if positives else 0.0})
    return output


def selection_as_dict(selection: ThresholdSelection) -> dict[str, Any]:
    return asdict(selection)
