"""Development-only feature ablations using offline review-budget metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config_loader import load_yaml_config
from .evaluation import cluster_bootstrap_review_budgets, evaluate_review_budgets, write_summary_csv

_FORBIDDEN_FEATURE_MARKERS = ("reference", "gold", "correct", "label", "truth", "intended",
                              "drug_name", "timestamp", "error", "annotation",
                              "speaker_id", "ingredient_id", "consultation_id", "recording_id",
                              "transcript", "audio_path", "risk_score")


def default_ablation_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Build prespecified full and leave-one-signal-family-out feature sets."""
    metadata = {"label", "label_error", "is_error", "is_medication_error", "recording_id", "audio_path", "speaker_id",
                "ingredient_id", "split", "pair_id", "transcript", "reference_text"}
    fields = sorted(key for key in ({key for row in rows for key in row} - metadata)
                    if not any(marker in key.casefold() for marker in _FORBIDDEN_FEATURE_MARKERS))
    groups = {
        "audio": {key for key in fields if any(tag in key.casefold() for tag in ("audio", "acoustic", "snr", "rt60", "spectral"))},
        "disagreement": {key for key in fields if "disagree" in key.casefold() or "recognizer_distance" in key.casefold()},
        "phonetic_retrieval": {key for key in fields if any(tag in key.casefold() for tag in ("phonetic", "phoneme", "soundex", "retrieval", "lexical_similarity"))},
    }
    all_features = set(fields)
    return {
        "full": sorted(all_features),
        "text_only_no_audio": sorted(all_features - groups["audio"]),
        "no_disagreement": sorted(all_features - groups["disagreement"]),
        "no_phonetic_retrieval": sorted(all_features - groups["phonetic_retrieval"]),
    }


def _feature_dict(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    result = {}
    for field in fields:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            result[field] = float(value)
        except (TypeError, ValueError):
            result[field] = str(value)
    return result


def run_ablation_experiments(
    training: Sequence[Mapping[str, Any]], development: Sequence[Mapping[str, Any]], *,
    label_field: str = "is_medication_error", review_budgets: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    feature_sets: Mapping[str, Sequence[str]] | None = None,
    estimator_factory: Callable[[], Any] | None = None,
    bootstrap_replicates: int = 1000,
    cluster_fields: Sequence[str] = ("speaker_id", "consultation_id"),
) -> list[dict[str, Any]]:
    """Fit on training rows and compare ranked error capture on development only."""
    if not training or not development:
        raise ValueError("Both training and development records are required")
    if estimator_factory is None:
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
        except ImportError as exc:
            raise RuntimeError("Ablations require scikit-learn") from exc
        estimator_factory = lambda: make_pipeline(
            DictVectorizer(sparse=False), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
        )
    feature_sets = feature_sets or default_ablation_features([*training, *development])
    forbidden = {field for fields in feature_sets.values() for field in fields
                 if any(marker in field.casefold() for marker in _FORBIDDEN_FEATURE_MARKERS)}
    if forbidden:
        raise ValueError(f"Ablation feature lists contain reference/label-derived fields: {sorted(forbidden)}")
    train_y = [int(row[label_field]) for row in training]
    if len(set(train_y)) != 2:
        raise ValueError("Training split must contain both error and correct-mention classes")
    output: list[dict[str, Any]] = []
    for name, fields in feature_sets.items():
        if not fields:
            raise ValueError(f"Ablation {name!r} has no selected features")
        model = estimator_factory()
        model.fit([_feature_dict(row, fields) for row in training], train_y)
        dev_scores = model.predict_proba([_feature_dict(row, fields) for row in development])[:, 1].tolist()
        scored = [dict(row, risk_score=score) for row, score in zip(development, dev_scores)]
        metrics = evaluate_review_budgets(scored, score_field="risk_score", label_field=label_field,
                                          budgets=review_budgets)
        try:
            bootstrap = cluster_bootstrap_review_budgets(
                scored, budgets=review_budgets, score_field="risk_score", label_field=label_field,
                cluster_fields=cluster_fields, n_bootstrap=bootstrap_replicates,
            ) if bootstrap_replicates else None
        except ValueError as exc:
            bootstrap = {"error": str(exc), "error_capture_ci_95_by_budget": {}}
        for row in metrics["budget_summary"]:
            ci = (bootstrap or {}).get("error_capture_ci_95_by_budget", {}).get(str(row["review_budget_fraction"]))
            output.append({"ablation": name, "n_features": len(fields), **row,
                           "error_capture_ci95_low": ci[0] if ci else None,
                           "error_capture_ci95_high": ci[1] if ci else None,
                           "bootstrap_valid_replicates": (bootstrap or {}).get("valid_replicates", 0),
                           "bootstrap_status": "available" if ci else (bootstrap or {}).get("error", "disabled")})
    return output


def main() -> None:
    """Run from YAML naming training/development CSVs and label field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML with training_csv/development_csv and output_csv")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    import csv
    def read_csv(path: str) -> list[dict[str, Any]]:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    rows = run_ablation_experiments(
        read_csv(config["training_csv"]), read_csv(config["development_csv"]),
        label_field=config.get("label_field", "is_medication_error"),
        review_budgets=config.get("review_budgets", [0.01, 0.05, 0.10, 0.20]),
        bootstrap_replicates=int(config.get("bootstrap_replicates", 1000)),
        cluster_fields=config.get("cluster_fields", ["speaker_id", "consultation_id"]),
    )
    write_summary_csv(rows, config.get("output_csv", "results/ablations.csv"))


if __name__ == "__main__":
    main()

