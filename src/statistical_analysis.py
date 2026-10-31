"""Pair-level metrics and ingredient-clustered inferential summaries."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _binary_inputs(y_true: Sequence[int], scores: Sequence[float]):
    if len(y_true) != len(scores) or not y_true:
        raise ValueError("y_true and scores must have equal non-zero length")
    y = [int(value) for value in y_true]
    p = [float(value) for value in scores]
    if any(value not in (0, 1) for value in y):
        raise ValueError("Labels must be binary 0/1")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in p):
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return y, p


def auprc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision (step-wise precision-recall area); prevalence if tied."""
    y, p = _binary_inputs(y_true, scores)
    positives = sum(y)
    if not positives:
        return 0.0
    order = sorted(range(len(y)), key=lambda i: (-p[i], i))
    tp = fp = area = 0
    previous_recall = 0.0
    cursor = 0
    while cursor < len(order):
        score = p[order[cursor]]
        end = cursor
        while end < len(order) and p[order[end]] == score:
            label = y[order[end]]
            tp += label
            fp += 1 - label
            end += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end
    return area


def binary_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict[str, float | int]:
    y, p = _binary_inputs(y_true, scores)
    pred = [int(value >= threshold) for value in p]
    tp = sum(a == b == 1 for a, b in zip(y, pred))
    tn = sum(a == b == 0 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    try:
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    except ImportError:
        roc = _auroc_rank(y, p)
    return {"n": len(y), "positives": sum(y), "prevalence": sum(y) / len(y),
            "auprc": auprc(y, p), "auroc": roc,
            "brier": sum((a - b) ** 2 for a, b in zip(y, p)) / len(y),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _auroc_rank(y: Sequence[int], scores: Sequence[float]) -> float:
    positives, negatives = sum(y), len(y) - sum(y)
    if not positives or not negatives:
        return float("nan")
    ranks = [0.0] * len(y)
    order = sorted(range(len(y)), key=lambda i: scores[i])
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return (sum(r for r, label in zip(ranks, y) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def mcnemar_test(y_true: Sequence[int], predictions_a: Sequence[int], predictions_b: Sequence[int]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired binary predictions."""
    if not (len(y_true) == len(predictions_a) == len(predictions_b)) or not y_true:
        raise ValueError("Inputs must have equal non-zero length")
    a_correct = [int(a) == int(y) for y, a in zip(y_true, predictions_a)]
    b_correct = [int(b) == int(y) for y, b in zip(y_true, predictions_b)]
    a_only = sum(a and not b for a, b in zip(a_correct, b_correct))
    b_only = sum(b and not a for a, b in zip(a_correct, b_correct))
    discordant = a_only + b_only
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2 ** discordant)
        p = min(1.0, 2.0 * tail)
    return {"method": "exact two-sided McNemar binomial test", "n": len(y_true),
            "a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only,
            "discordant": discordant, "p_value": p}


def grouped_bootstrap(
    y_true: Sequence[int], scores: Sequence[float], groups: Sequence[str], *,
    n_bootstrap: int = 2000, seed: int = 20261030, metric: str = "auprc",
) -> dict[str, Any]:
    """Percentile interval resampling whole ingredient/component groups."""
    if not (len(y_true) == len(scores) == len(groups)) or not y_true:
        raise ValueError("Inputs must have equal non-zero length")
    if n_bootstrap < 1 or metric not in {"auprc", "auroc", "brier"}:
        raise ValueError("n_bootstrap must be positive and metric must be auprc/auroc/brier")
    import random
    rng = random.Random(seed)
    indices: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        indices.setdefault(str(group), []).append(idx)
    keys = list(indices)
    observed = _metric(metric, y_true, scores)
    values = []
    for _ in range(n_bootstrap):
        sample = [idx for group in (rng.choice(keys) for _ in keys) for idx in indices[group]]
        labels = [int(y_true[i]) for i in sample]
        if metric in ("auprc", "auroc") and len(set(labels)) < 2:
            continue
        values.append(_metric(metric, labels, [scores[i] for i in sample]))
    values.sort()
    return {"metric": metric, "estimate": observed, "lower_95": _quantile(values, 0.025),
            "upper_95": _quantile(values, 0.975), "valid_replicates": len(values),
            "requested_replicates": n_bootstrap, "groups": len(keys), "resampling_unit": "ingredient_component"}


def paired_grouped_bootstrap_delta(
    y_true: Sequence[int], scores_a: Sequence[float], scores_b: Sequence[float], groups: Sequence[str], *,
    n_bootstrap: int = 2000, seed: int = 20261030, metric: str = "auprc",
) -> dict[str, Any]:
    """Paired group bootstrap for metric(A)-metric(B), with shared group draws."""
    if not (len(y_true) == len(scores_a) == len(scores_b) == len(groups)) or not y_true:
        raise ValueError("Inputs must have equal non-zero length")
    import random
    rng = random.Random(seed)
    members: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(str(group), []).append(i)
    keys = list(members)
    observed = _metric(metric, y_true, scores_a) - _metric(metric, y_true, scores_b)
    deltas = []
    for _ in range(n_bootstrap):
        sample = [i for key in (rng.choice(keys) for _ in keys) for i in members[key]]
        labels = [int(y_true[i]) for i in sample]
        if metric in ("auprc", "auroc") and len(set(labels)) < 2:
            continue
        deltas.append(_metric(metric, labels, [scores_a[i] for i in sample]) -
                      _metric(metric, labels, [scores_b[i] for i in sample]))
    deltas.sort()
    return {"metric": metric, "delta_a_minus_b": observed, "lower_95": _quantile(deltas, 0.025),
            "upper_95": _quantile(deltas, 0.975), "valid_replicates": len(deltas),
            "requested_replicates": n_bootstrap, "groups": len(keys), "resampling_unit": "ingredient_component"}


def _metric(name: str, y: Sequence[int], scores: Sequence[float]) -> float:
    if name == "auprc":
        return auprc(y, scores)
    if name == "auroc":
        return _auroc_rank(y, scores)
    return sum((int(a) - float(b)) ** 2 for a, b in zip(y, scores)) / len(y)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * q
    lo, hi = int(position), math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)
