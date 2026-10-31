"""Summarize Phase 6 pipeline artifacts for results review and submission drafting.

Example: ``python -m src.generate_report --results-dir results/phase6``
The script reads saved outputs only; it never refits models or tunes thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration import calibration_curve_data


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _fmt(value: Any) -> str:
    if value is None:
        return "not reported"
    try:
        number = float(value)
        return "not estimable" if number != number else f"{number:.4f}"
    except (TypeError, ValueError):
        return str(value)


def _append_metric(rows: list[dict[str, Any]], phase: str, model: str, metrics: Mapping[str, Any] | None) -> None:
    if not metrics:
        return
    for metric in ("n", "positives", "prevalence", "auprc", "auroc", "brier", "precision", "recall", "f1"):
        if metric in metrics:
            rows.append({"phase": phase, "model": model, "metric": metric, "value": metrics[metric]})


def _calibration_rows(predictions: Sequence[Mapping[str, str]], n_bins: int) -> list[dict[str, Any]]:
    if not predictions:
        return []
    labels = [int(float(row["label"])) for row in predictions]
    output = []
    for method, field in (("raw", "raw_probability"), ("platt", "platt_probability"),
                          ("isotonic", "isotonic_probability")):
        paired = [(label, float(row[field])) for label, row in zip(labels, predictions)
                  if row.get(field) not in (None, "")]
        if not paired:
            continue
        curve = calibration_curve_data([item[0] for item in paired], [item[1] for item in paired], n_bins=n_bins)
        output.extend({"method": method, **point} for point in curve)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_calibration(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Calibration plot generation requires matplotlib; install requirements.txt") from exc
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    for method in ("raw", "platt", "isotonic"):
        curve = [row for row in rows if row["method"] == method]
        if curve:
            axis.plot([float(row["mean_predicted_probability"]) for row in curve],
                      [float(row["observed_positive_rate"]) for row in curve],
                      marker="o", label=method)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
             ylabel="Observed positive rate", title="H2: calibration on held-out acoustic conditions")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_report(results_dir: str | Path, *, calibration_bins: int = 10) -> dict[str, Path]:
    root = Path(results_dir)
    phase5 = _read_json(root / "model_report.json")
    h4 = _read_json(root / "h4_transfer_report.json")
    if phase5 is None:
        raise FileNotFoundError(f"Missing pipeline summary: {root / 'model_report.json'}")
    metric_rows: list[dict[str, Any]] = []
    _append_metric(metric_rows, "H1 ingredient-disjoint", "full model", phase5.get("test_full_model"))
    _append_metric(metric_rows, "H1 ingredient-disjoint", "phonetic-only model", phase5.get("test_phonetic_only"))
    _append_metric(metric_rows, "H1 ingredient-disjoint", "Soundex similarity", phase5.get("test_similarity_baseline"))
    for split, count in phase5.get("independent_components", {}).items():
        metric_rows.append({"phase": "H1 ingredient-disjoint", "model": "split size", "metric": f"{split} ingredient components", "value": count})
    for name, key in (("full minus phonetic", "paired_group_bootstrap_auprc_delta_full_minus_phonetic"),
                      ("full minus Soundex", "paired_group_bootstrap_auprc_delta_full_minus_similarity")):
        delta = phase5.get(key, {})
        if delta:
            metric_rows.extend([
                {"phase": "H1 ingredient-disjoint", "model": name, "metric": "AUPRC delta", "value": delta.get("delta_a_minus_b")},
                {"phase": "H1 ingredient-disjoint", "model": name, "metric": "bootstrap 95% lower", "value": delta.get("lower_95")},
                {"phase": "H1 ingredient-disjoint", "model": name, "metric": "bootstrap 95% upper", "value": delta.get("upper_95")},
            ])
    h2 = phase5.get("H2_unseen_acoustic_conditions", {})
    if h2.get("test_raw_model"):
        _append_metric(metric_rows, "H2 unseen conditions", "raw full model", h2.get("test_raw_model"))
        for split, count in h2.get("independent_components", {}).items():
            metric_rows.append({"phase": "H2 unseen conditions", "model": "split size", "metric": f"{split} ingredient components", "value": count})
        metric_rows.append({"phase": "H2 unseen conditions", "model": "prevalence baseline",
                            "metric": "AUPRC", "value": h2.get("test_prevalence_baseline_auprc")})
        delta = h2.get("paired_group_bootstrap_auprc_delta_raw_vs_prevalence", {})
        if delta:
            metric_rows.extend([
                {"phase": "H2 unseen conditions", "model": "raw full minus prevalence", "metric": "AUPRC delta", "value": delta.get("delta_a_minus_b")},
                {"phase": "H2 unseen conditions", "model": "raw full minus prevalence", "metric": "bootstrap 95% lower", "value": delta.get("lower_95")},
                {"phase": "H2 unseen conditions", "model": "raw full minus prevalence", "metric": "bootstrap 95% upper", "value": delta.get("upper_95")},
            ])
        for method, detail in h2.get("calibration_comparison", {}).items():
            _append_metric(metric_rows, "H2 unseen conditions", f"{method} calibrated", detail.get("test_metrics"))
    if h4:
        _append_metric(metric_rows, "H4 cross-recognizer transfer", str(h4.get("secondary_recognizer", "secondary")),
                       h4.get("test_metrics"))
        metric_rows.append({"phase": "H4 cross-recognizer transfer", "model": "split size",
                            "metric": "test ingredient components", "value": h4.get("independent_test_components")})
        interval = h4.get("grouped_bootstrap_auprc", {})
        if interval:
            metric_rows.extend([
                {"phase": "H4 cross-recognizer transfer", "model": "secondary recognizer", "metric": "AUPRC bootstrap 95% lower", "value": interval.get("lower_95")},
                {"phase": "H4 cross-recognizer transfer", "model": "secondary recognizer", "metric": "AUPRC bootstrap 95% upper", "value": interval.get("upper_95")},
            ])
    metric_csv = root / "submission" / "evaluation_metrics.csv"
    _write_csv(metric_csv, metric_rows)

    h2_predictions = _read_csv(root / "h2_calibration_predictions.csv")
    curve_rows = _calibration_rows(h2_predictions, calibration_bins)
    curve_csv = root / "submission" / "h2_calibration_curve_data.csv"
    _write_csv(curve_csv, curve_rows)
    curve_png = root / "submission" / "h2_calibration_curves.png"
    _plot_calibration(curve_rows, curve_png)

    h1_delta = phase5.get("paired_group_bootstrap_auprc_delta_full_minus_phonetic", {})
    h1_similarity_delta = phase5.get("paired_group_bootstrap_auprc_delta_full_minus_similarity", {})
    h2_cal = h2.get("calibration_comparison", {})
    h2_status = h2.get("status", "evaluated" if h2 else "not run (no H2 result artifact found)")
    h4_status = "evaluated" if h4 else "not run (no secondary recognizer report found)"
    h1_metrics = phase5.get("test_full_model", {})
    h1_groups = phase5.get("independent_components", {})
    h2_metrics = h2.get("test_raw_model", {})
    h4_metrics = h4.get("test_metrics", {}) if h4 else {}
    lines = [
        "# MISHEAR Results Summary Draft",
        "",
        "This is a results-writing aid for the Davidson Fellows submission. Check the current program prompt, eligibility rules, and word limits before using it. Replace bracketed fields with project-specific details and retain negative or inconclusive findings.",
        "",
        "## Research question and design",
        "",
        "MISHEAR evaluates whether interpretable models can identify medication-name substitutions made by ASR on a controlled synthetic-speech benchmark. Candidate medication pairs were generated from the configured vocabulary; fictional utterances were synthesized, acoustically augmented, transcribed, and aligned to intended names. Models use phonetic, lexical, contextual, measured acoustic, and recognizer features where available. Ingredient-linked groups are kept together for H1; H2 reserves configured acoustic conditions; H4 evaluates the primary-recognizer model on matched examples from a second recognizer.",
        "",
        "## Results to report",
        "",
        f"- **H1, ingredient-disjoint test:** full-model AUPRC {_fmt(h1_metrics.get('auprc'))} (n={h1_metrics.get('n', 'not reported')}, ingredient components={h1_groups.get('test', 'not reported')}, prevalence {_fmt(h1_metrics.get('prevalence'))}); AUROC {_fmt(h1_metrics.get('auroc'))}. The grouped-bootstrap AUPRC difference versus the phonetic-only model was {_fmt(h1_delta.get('delta_a_minus_b'))} (95% interval {_fmt(h1_delta.get('lower_95'))} to {_fmt(h1_delta.get('upper_95'))}); versus the Soundex ranking baseline it was {_fmt(h1_similarity_delta.get('delta_a_minus_b'))} ({_fmt(h1_similarity_delta.get('lower_95'))} to {_fmt(h1_similarity_delta.get('upper_95'))}).",
        f"- **H2, unseen acoustic conditions:** status {h2_status}; raw-model AUPRC {_fmt(h2_metrics.get('auprc'))} (n={h2_metrics.get('n', 'not reported')}, ingredient components={h2.get('independent_components', {}).get('test', 'not reported')}), compared with held-out prevalence {_fmt(h2.get('test_prevalence_baseline_auprc'))}. Calibration methods evaluated: {', '.join(h2_cal) if h2_cal else 'none'}.",
        f"- **H4, cross-recognizer transfer:** {h4_status}; secondary recognizer {h4.get('secondary_recognizer', 'not reported')}; AUPRC {_fmt(h4_metrics.get('auprc'))}, AUROC {_fmt(h4_metrics.get('auroc'))} (n={h4_metrics.get('n', 'not reported')}, ingredient components={h4.get('independent_test_components', 'not reported')}).",
        "- **Selective alerting:** report the validation-selected alert threshold and its finite-sample bound from `model_report.json`, plus the held-out false-alert count and missed substitutions. Do not describe a validation bound as a guarantee under distribution shift.",
        "",
        "## Interpretation and limits",
        "",
        "These results characterize the configured synthetic benchmark. Synthetic voices, generated carrier sentences, selected medication terms, the chosen ASR checkpoints, and simulated room/noise conditions do not establish performance on real clinical speech or clinical benefit. Report the number of independent ingredient-pair components and class prevalence alongside clip counts; repeated clips are not independent observations. If split construction leaves too few independent components or a class absent, label that analysis infeasible rather than relaxing the split after viewing test results.",
        "",
        "## Suggested figures and tables",
        "",
        "- `evaluation_metrics.csv`: model-by-hypothesis metrics table.",
        "- `h2_calibration_curves.png` and `h2_calibration_curve_data.csv`: equal-width-bin reliability curves on held-out acoustic conditions.",
        "- `../test_predictions.csv`, `../h2_calibration_predictions.csv`, and `../h4_transfer_predictions.csv`: per-example predictions for audit and supplementary analysis.",
        "",
        "## Items to complete before submission",
        "",
        "Insert the final sample/group counts, exact ASR and TTS versions, prespecified thresholds, confidence intervals, error counts, limitations, and the project's novel contribution after checking the saved artifacts. Keep synthetic benchmark claims separate from patient-safety claims.",
    ]
    draft_path = root / "submission" / "DAVIDSON_FELLOWS_RESULTS_DRAFT.md"
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"metrics_csv": metric_csv, "calibration_csv": curve_csv,
            "calibration_png": curve_png, "submission_draft": draft_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/phase6")
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()
    outputs = generate_report(args.results_dir, calibration_bins=args.calibration_bins)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
