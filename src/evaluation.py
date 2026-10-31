"""Error-auditor evaluation with correct-mention FAR and cluster-aware intervals."""

from __future__ import annotations

import csv
import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _validate_binary(y_true: Sequence[int], scores: Sequence[float]) -> tuple[list[int], list[float]]:
    labels, values = list(map(int, y_true)), list(map(float, scores))
    if len(labels) != len(values) or not labels:
        raise ValueError("Labels and scores must have equal, non-zero lengths")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Labels must be binary: 1=transcription error, 0=correct mention")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Scores must be finite")
    if not 0 in labels or not 1 in labels:
        raise ValueError("Evaluation requires both correct mentions and transcription errors")
    return labels, values


def evaluate_at_threshold(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, float | int]:
    """Compute FAR=FP/all correct mentions and miss rate=FN/all true errors."""
    labels, values = _validate_binary(y_true, scores)
    predicted = [int(value >= threshold) for value in values]
    correct = sum(label == 0 for label in labels)
    errors = sum(label == 1 for label in labels)
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predicted))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predicted))
    tp = errors - fn
    return {
        "threshold": float(threshold), "n_mentions": len(labels), "n_correct_mentions": correct,
        "n_true_errors": errors, "false_alerts": fp, "missed_errors": fn,
        "false_alert_rate": fp / correct, "missed_error_rate": fn / errors,
        "precision": tp / max(tp + fp, 1), "recall": tp / errors,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def select_threshold_at_far_budget(
    y_true: Sequence[int], scores: Sequence[float], far_budget: float,
) -> dict[str, float | int]:
    """Choose a score threshold maximizing error recall under validation FAR budget."""
    if not 0 <= far_budget <= 1:
        raise ValueError("FAR budget must be a fraction in [0, 1]")
    labels, values = _validate_binary(y_true, scores)
    candidates = [math.nextafter(max(values), math.inf), *sorted(set(values), reverse=True)]
    feasible = [evaluate_at_threshold(labels, values, threshold) for threshold in candidates]
    feasible = [row for row in feasible if float(row["false_alert_rate"]) <= far_budget + 1e-12]
    return max(feasible, key=lambda row: (float(row["recall"]), float(row["threshold"])))


def roc_pr_curves(y_true: Sequence[int], scores: Sequence[float]) -> dict[str, list[float]]:
    """Generate ROC and precision-recall coordinates without requiring sklearn."""
    labels, values = _validate_binary(y_true, scores)
    thresholds = [math.nextafter(max(values), math.inf), *sorted(set(values), reverse=True)]
    roc_fpr, roc_tpr, precision, recall = [], [], [], []
    n_correct, n_error = sum(y == 0 for y in labels), sum(labels)
    for threshold in thresholds:
        pred = [score >= threshold for score in values]
        tp = sum(p and y == 1 for p, y in zip(pred, labels))
        fp = sum(p and y == 0 for p, y in zip(pred, labels))
        roc_tpr.append(tp / n_error)
        roc_fpr.append(fp / n_correct)
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / n_error)
    return {"thresholds": thresholds, "roc_false_positive_rate": roc_fpr,
            "roc_true_positive_rate": roc_tpr, "pr_precision": precision, "pr_recall": recall}


def curve_summary(y_true: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    """Return AUROC and non-interpolated average precision from ranked scores."""
    labels, values = _validate_binary(y_true, scores)
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    positives, negatives = sum(labels), len(labels) - sum(labels)
    tp = fp = 0
    previous_recall = 0.0
    average_precision = 0.0
    points = [(0.0, 0.0)]
    index = 0
    while index < len(order):
        score = values[order[index]]
        end = index
        while end < len(order) and values[order[end]] == score:
            if labels[order[end]]:
                tp += 1
            else:
                fp += 1
            end += 1
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        average_precision += (recall - previous_recall) * precision
        points.append((fp / negatives, recall))
        previous_recall = recall
        index = end
    auroc = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(points, points[1:]))
    return {"auroc": auroc, "auprc_average_precision": average_precision}


def summary_table(
    y_true: Sequence[int], scores: Sequence[float], far_budgets: Sequence[float] = (0.05, 0.10),
) -> list[dict[str, float | int]]:
    """Return fixed-FAR operating points; threshold selection is on supplied dev data."""
    return [{"far_budget": float(budget), **select_threshold_at_far_budget(y_true, scores, float(budget))}
            for budget in far_budgets]


def evaluate_review_budgets(
    records: Sequence[Mapping[str, Any]], *, score_field: str = "risk_score",
    label_field: str = "is_medication_error", budgets: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    duration_field: str = "audio_duration_s", correct_mentions_field: str = "correct_medication_mentions",
    error_mentions_field: str = "medication_error_mentions",
) -> dict[str, Any]:
    """Measure error concentration when the top-ranked utterances are reviewed.

    Each budget is a fraction of utterances. Error capture is the fraction of
    all medication errors contained in that review subset. The false-alert
    denominator is all correctly recognized medication mentions, as distinct
    from correct utterances. Oracle capture ranks by gold error counts and is a
    theoretical ceiling, never an achievable model result.
    """
    if not records:
        raise ValueError("Review-budget evaluation requires labeled records")
    labels = [int(row[label_field]) for row in records]
    scores = [float(row[score_field]) for row in records]
    if any(label not in (0, 1) for label in labels):
        raise ValueError(f"{label_field} must be binary (1=contains a medication error)")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Risk scores must be finite")
    missing_counts = [index for index, row in enumerate(records)
                      if error_mentions_field not in row or correct_mentions_field not in row
                      or duration_field not in row]
    if missing_counts:
        raise ValueError(
            f"Exact FAR/error-capture denominators require {error_mentions_field!r} and "
            f"{correct_mentions_field!r} and {duration_field!r} on every row; missing at {missing_counts[:5]}"
        )
    error_counts = [int(row[error_mentions_field]) for row in records]
    correct_counts = [int(row[correct_mentions_field]) for row in records]
    if any((error_counts[index] > 0) != bool(labels[index]) for index in range(len(records))):
        raise ValueError("Binary utterance error labels must agree with medication_error_mentions counts")
    durations = [float(row[duration_field]) for row in records]
    if any(count < 0 for count in error_counts + correct_counts) or any(
        not math.isfinite(duration) or duration < 0 for duration in durations
    ):
        raise ValueError("Mention counts and audio durations must be nonnegative")
    total_errors, total_correct = sum(error_counts), sum(correct_counts)
    if total_errors <= 0 or total_correct <= 0:
        raise ValueError("Evaluation requires at least one medication error and one correct mention")
    ranked = sorted(range(len(records)), key=lambda index: (-scores[index], index))
    rows = []
    for budget in budgets:
        if not 0 < float(budget) <= 1:
            raise ValueError("Review budgets must be fractions in (0, 1]")
        selected_count = min(len(records), max(1, math.ceil(float(budget) * len(records))))
        selected = ranked[:selected_count]
        captured = sum(error_counts[index] for index in selected)
        false_alerts = sum(correct_counts[index] for index in selected)
        flagged_errors = sum(labels[index] == 1 for index in selected)
        # Oracle ceiling under the same number of reviewed utterances.
        oracle = sorted(range(len(records)), key=lambda index: (-error_counts[index], index))[:selected_count]
        oracle_captured = sum(error_counts[index] for index in oracle)
        rows.append({
            "review_budget_fraction": float(budget), "review_budget_percent": 100 * float(budget),
            "utterances_total": len(records), "utterances_flagged": selected_count,
            "errors_total": total_errors, "errors_captured": captured,
            "error_capture": captured / total_errors,
            "correct_medication_mentions_total": total_correct,
            "correct_medication_mentions_in_flagged_subset": false_alerts,
            "false_alert_rate": false_alerts / total_correct,
            "flagged_utterance_precision": flagged_errors / selected_count,
            "missed_errors": total_errors - captured,
            "flagged_audio_duration_s": sum(durations[index] for index in selected),
            "oracle_error_capture_ceiling": oracle_captured / total_errors,
        })
    curves = {"review_budget_fraction": [row["review_budget_fraction"] for row in rows],
              "error_capture": [row["error_capture"] for row in rows],
              "oracle_error_capture_ceiling": [row["oracle_error_capture_ceiling"] for row in rows]}
    return {"evaluation_type": "offline_review_budget_error_prioritization",
            "interpretation": "Error concentration and prioritization only; not workload reduction or harm prevention.",
            "budget_summary": rows, "error_capture_curve": curves}


def clustered_bootstrap_review_budgets(
    records: Sequence[Mapping[str, Any]], *, budgets: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    score_field: str = "risk_score", label_field: str = "is_medication_error",
    speaker_field: str = "speaker_id", ingredient_field: str = "ingredient_id",
    n_bootstrap: int = 1000, seed: int = 20261030,
) -> dict[str, Any]:
    """Two-way speaker/ingredient pigeonhole bootstrap of error capture curves.

    A row's resampling weight is its sampled speaker multiplicity multiplied
    by the sum of sampled ingredient multiplicities for its (possibly several)
    ingredient IDs. Rows missing either grouping key are rejected.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    speakers = sorted({str(row[speaker_field]) for row in records if row.get(speaker_field) not in (None, "")})
    ingredients = sorted({ingredient for row in records for ingredient in _ingredients_from_row(row, ingredient_field)})
    if not speakers or not ingredients:
        raise ValueError("Cluster bootstrap requires speaker and ingredient identifiers")
    if any(row.get(speaker_field) in (None, "") or not _ingredients_from_row(row, ingredient_field)
           for row in records):
        raise ValueError("Every evaluation row must carry speaker and ingredient grouping IDs")
    rng = random.Random(seed)
    samples: dict[float, list[float]] = {float(budget): [] for budget in budgets}
    for _ in range(n_bootstrap):
        speaker_multiplicity, ingredient_multiplicity = CounterLike(), CounterLike()
        for _ in speakers:
            speaker_multiplicity.add(rng.choice(speakers))
        for _ in ingredients:
            ingredient_multiplicity.add(rng.choice(ingredients))
        bootstrap_rows = []
        for row in records:
            ids = _ingredients_from_row(row, ingredient_field)
            mult = speaker_multiplicity.get(str(row.get(speaker_field))) * sum(ingredient_multiplicity.get(i) for i in ids)
            bootstrap_rows.extend([row] * mult)
        if not bootstrap_rows or sum(int(row[label_field]) for row in bootstrap_rows) == 0:
            continue
        try:
            result = evaluate_review_budgets(bootstrap_rows, score_field=score_field,
                                             label_field=label_field, budgets=budgets)
        except ValueError:
            continue
        for row in result["budget_summary"]:
            samples[float(row["review_budget_fraction"])].append(float(row["error_capture"]))
    intervals = {}
    for budget, values in samples.items():
        values.sort()
        intervals[str(budget)] = ([values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]
                                  if values else None)
    return {"valid_replicates": max((len(values) for values in samples.values()), default=0),
            "error_capture_ci_95_by_budget": intervals}


def cluster_bootstrap_review_budgets(
    records: Sequence[Mapping[str, Any]], *, cluster_fields: Sequence[str] = ("speaker_id", "consultation_id"),
    budgets: Sequence[float] = (.01, .05, .10, .20), score_field: str = "risk_score",
    label_field: str = "is_medication_error", n_bootstrap: int = 1000, seed: int = 20261030,
) -> dict[str, Any]:
    """Pigeonhole bootstrap over declared speaker/consultation clusters.

    Each row receives the product of sampled cluster multiplicities across
    levels. The procedure preserves the dependence structure better than an
    utterance bootstrap; it does not claim independent corpus sampling.
    """
    if not records or not cluster_fields or n_bootstrap < 1:
        raise ValueError("Records, cluster fields, and positive replicate count are required")
    clusters: dict[str, list[str]] = {}
    for field in cluster_fields:
        values = sorted({str(row.get(field)) for row in records if row.get(field) not in (None, "")})
        if not values or any(row.get(field) in (None, "") for row in records):
            raise ValueError(f"Every row must have a non-empty {field!r} for cluster bootstrap")
        clusters[field] = values
    rng = random.Random(seed)
    samples: dict[float, list[float]] = {float(budget): [] for budget in budgets}
    for _ in range(n_bootstrap):
        multiplicities: dict[str, dict[str, int]] = {}
        for field, values in clusters.items():
            count: dict[str, int] = {}
            for _ in values:
                chosen = rng.choice(values)
                count[chosen] = count.get(chosen, 0) + 1
            multiplicities[field] = count
        sample: list[Mapping[str, Any]] = []
        for row in records:
            weight = 1
            for field in cluster_fields:
                weight *= multiplicities[field].get(str(row[field]), 0)
            sample.extend([row] * weight)
        if not sample or {int(row[label_field]) for row in sample} != {0, 1}:
            continue
        try:
            result = evaluate_review_budgets(sample, score_field=score_field, label_field=label_field,
                                             budgets=budgets)
        except ValueError:
            continue
        for row in result["budget_summary"]:
            samples[float(row["review_budget_fraction"])].append(float(row["error_capture"]))
    intervals = {}
    for budget, values in samples.items():
        values.sort()
        intervals[str(budget)] = ([values[int(.025 * (len(values) - 1))],
                                   values[int(.975 * (len(values) - 1))]] if values else None)
    return {"cluster_fields": list(cluster_fields), "valid_replicates": min(map(len, samples.values()), default=0),
            "error_capture_ci_95_by_budget": intervals}


def evaluate_subgroups(
    records: Sequence[Mapping[str, Any]], *, subgroup_fields: Sequence[str] =
    ("accent_group", "speaker_seen_status", "ingredient_seen_status", "corpus"),
    score_field: str = "risk_score", label_field: str = "is_medication_error",
    budgets: Sequence[float] = (.01, .05, .10, .20),
) -> dict[str, Any]:
    """Report subgroup review concentration while retaining group identities."""
    result: dict[str, Any] = {}
    for field in subgroup_fields:
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for row in records:
            groups.setdefault(str(row.get(field) or "unknown"), []).append(row)
        result[field] = {}
        for group, rows in sorted(groups.items()):
            item: dict[str, Any] = {"n_utterances": len(rows), "n_speakers": len({r.get("speaker_id") for r in rows if r.get("speaker_id")}),
                                    "n_consultations": len({r.get("consultation_id") for r in rows if r.get("consultation_id")}),
                                    "status": "scored"}
            try:
                metrics = evaluate_review_budgets(rows, score_field=score_field, label_field=label_field, budgets=budgets)
                item["budget_summary"] = metrics["budget_summary"]
                item["curve_summary"] = curve_summary([int(r[label_field]) for r in rows], [float(r[score_field]) for r in rows])
            except (ValueError, KeyError) as exc:
                item["status"] = "insufficient_or_invalid_labels"
                item["reason"] = str(exc)
            result[field][group] = item
    return result


def _ingredients_from_row(row: Mapping[str, Any], field: str) -> set[str]:
    raw = row.get(field) or row.get("ingredient_ids") or row.get("ingredient_rxcui") or row.get("ingredient")
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw}
    return {item.strip() for item in str(raw).replace(";", "|").replace(",", "|").split("|") if item.strip()}


class CounterLike:
    """Tiny multiplicity map, avoiding implicit defaultdict behavior."""
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
    def add(self, key: str) -> None:
        self.values[key] = self.values.get(key, 0) + 1
    def get(self, key: str) -> int:
        return self.values.get(key, 0)


def clustered_bootstrap(
    records: Sequence[Mapping[str, Any]], *, label_field: str, score_field: str,
    speaker_field: str = "speaker_id", ingredient_field: str = "ingredient_id",
    n_bootstrap: int = 1000, seed: int = 20261030, far_budget: float = 0.05,
) -> dict[str, Any]:
    """Two-way pigeonhole bootstrap over speakers and ingredients.

    Multiplicity weights for a row are the product of its sampled speaker and
    ingredient cluster counts, so recordings are not treated as independent.
    Threshold is selected within each resample at the requested FAR budget.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    speakers = sorted({str(row[speaker_field]) for row in records if row.get(speaker_field) not in (None, "")})
    ingredients = sorted({ingredient for row in records for ingredient in _ingredients_from_row(row, ingredient_field)})
    if not speakers or not ingredients:
        raise ValueError("Cluster bootstrap requires speaker and ingredient identifiers on all usable rows")
    if any(row.get(speaker_field) in (None, "") or not _ingredients_from_row(row, ingredient_field) for row in records):
        raise ValueError("Every bootstrap row must carry speaker and ingredient grouping IDs")
    rng = random.Random(seed)
    recalls: list[float] = []
    fars: list[float] = []
    misses: list[float] = []
    for _ in range(n_bootstrap):
        speaker_counts: dict[str, int] = {}
        ingredient_counts: dict[str, int] = {}
        for _ in speakers:
            key = rng.choice(speakers)
            speaker_counts[key] = speaker_counts.get(key, 0) + 1
        for _ in ingredients:
            key = rng.choice(ingredients)
            ingredient_counts[key] = ingredient_counts.get(key, 0) + 1
        sample: list[Mapping[str, Any]] = []
        for row in records:
            ingredient_multiplicity = sum(ingredient_counts.get(ingredient, 0)
                                          for ingredient in _ingredients_from_row(row, ingredient_field))
            multiplicity = speaker_counts.get(str(row[speaker_field]), 0) * ingredient_multiplicity
            sample.extend([row] * multiplicity)
        if not sample or {int(row[label_field]) for row in sample} != {0, 1}:
            continue
        result = select_threshold_at_far_budget([int(row[label_field]) for row in sample],
                                                [float(row[score_field]) for row in sample], far_budget)
        recalls.append(float(result["recall"]))
        fars.append(float(result["false_alert_rate"]))
        misses.append(float(result["missed_error_rate"]))
    def interval(values: Sequence[float]) -> list[float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]
    return {"valid_replicates": len(recalls), "recall_ci_95": interval(recalls),
            "far_ci_95": interval(fars), "missed_error_rate_ci_95": interval(misses),
            "bootstrap_recall": recalls, "bootstrap_far": fars}


def write_summary_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write summary rows to a CSV, creating parent directories as needed."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Cannot write an empty evaluation summary")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def evaluate_score_columns(
    records: Sequence[Mapping[str, Any]], score_fields: Mapping[str, str], *,
    label_field: str, budgets: Sequence[float] = (.01, .05, .10, .20),
    bootstrap_replicates: int = 0, seed: int = 20261030,
) -> dict[str, Any]:
    """Evaluate multiple frozen model-score columns on one held-out manifest."""
    output: dict[str, Any] = {"systems": {}}
    for system, score_field in score_fields.items():
        prepared = [dict(row, risk_score=float(row[score_field])) for row in records]
        metrics = evaluate_review_budgets(prepared, score_field="risk_score", label_field=label_field,
                                          budgets=budgets)
        result: dict[str, Any] = {**metrics, "curve_summary": curve_summary(
            [int(row[label_field]) for row in prepared], [float(row["risk_score"]) for row in prepared])}
        if bootstrap_replicates:
            result["clustered_bootstrap"] = clustered_bootstrap_review_budgets(
                prepared, budgets=budgets, score_field="risk_score", label_field=label_field,
                n_bootstrap=bootstrap_replicates, seed=seed)
        output["systems"][system] = result
    return output


def main() -> None:
    """Evaluate frozen JSONL predictions using a YAML evaluation manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/evaluation_config.yaml")
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to load evaluation config") from exc
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    input_path = Path(config["evaluation_input_jsonl"])
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate_score_columns(
        records, config["score_fields"], label_field=config.get("label_field", "is_medication_error"),
        budgets=config.get("review_budgets", [.01, .05, .10, .20]),
        bootstrap_replicates=int(config.get("bootstrap_replicates", 0)),
        seed=int(config.get("seed", 20261030)),
    )
    result.update({"corpus": config.get("corpus", "unspecified"),
                   "partition": config.get("partition", "unspecified"),
                   "evaluation_file": str(input_path)})
    output_path = Path(config.get("output_json", "results/human_speech/evaluation_test.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote offline review-prioritization metrics to {output_path}")


if __name__ == "__main__":
    main()

