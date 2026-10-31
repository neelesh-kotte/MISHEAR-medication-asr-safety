"""Generate paper-facing summaries for cross-corpus offline review prioritization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _extract_rows(payload: Mapping[str, Any], source: Path) -> list[dict[str, Any]]:
    rows = []
    systems = payload.get("systems")
    if isinstance(systems, Mapping):
        for system, result in systems.items():
            for metric in result.get("budget_summary", []):
                rows.append({"source": str(source), "corpus": payload.get("corpus", "unspecified"),
                             "partition": payload.get("partition", "unspecified"), "system": system, **metric})
    elif isinstance(payload.get("budget_summary"), list):
        for metric in payload["budget_summary"]:
            rows.append({"source": str(source), "corpus": payload.get("corpus", "unspecified"),
                         "partition": payload.get("partition", "unspecified"),
                         "system": payload.get("system", source.stem), **metric})
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (list(dict.fromkeys(key for row in rows for key in row)) if rows else [
        "source", "corpus", "partition", "system", "review_budget_fraction", "error_capture",
        "false_alert_rate", "flagged_utterance_precision", "missed_errors",
        "flagged_audio_duration_s", "oracle_error_capture_ceiling",
    ])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(rows: Sequence[Mapping[str, Any]], path: Path) -> bool:
    if not rows:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axis = plt.subplots(figsize=(7.4, 5.0))
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["corpus"]), str(row["partition"]), str(row["system"]))
        groups.setdefault(key, []).append(row)
    for key, points in sorted(groups.items()):
        points.sort(key=lambda item: float(item["review_budget_fraction"]))
        axis.plot([100 * float(point["review_budget_fraction"]) for point in points],
                  [float(point["error_capture"]) for point in points], marker="o", label=" / ".join(key))
    axis.set(xlabel="Top-ranked utterances selected for review (%)", ylabel="Medication-error capture",
             title="Error concentration by offline review budget", xlim=(0, 21), ylim=(0, 1.02))
    axis.grid(alpha=.25)
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def generate_report(results_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Summarize saved human-corpus budget-evaluation JSON without refitting."""
    root = Path(results_dir)
    destination = Path(output_dir) if output_dir else root / "submission"
    destination.mkdir(parents=True, exist_ok=True)
    inventory_path = root / "inventory.json"
    evaluation_files = sorted(path for path in root.rglob("evaluation*.json")
                              if destination not in path.parents)
    rows = []
    for path in evaluation_files:
        rows.extend(_extract_rows(_load_json(path), path))
    table_path = destination / "review_budget_metrics.csv"
    curve_path = destination / "error_capture_curves.png"
    report_path = destination / "CROSS_CORPUS_RESULTS_DRAFT.md"
    _write_csv(table_path, rows)
    plotted = _plot_curves(rows, curve_path)

    lines = ["# MISHEAR cross-corpus results summary", "",
             "This report describes offline error concentration and prioritization only. It does not establish workload reduction, harm prevention, clinical effectiveness, or deployment safety.", ""]
    if inventory_path.is_file():
        inventory = _load_json(inventory_path)
        lines += [f"**Feasibility status:** {inventory.get('status', 'not recorded')}", ""]
        if inventory.get("stopping_reasons"):
            lines += ["Inventory stop reasons:", ""] + [f"- {reason}" for reason in inventory["stopping_reasons"]] + [""]
    if rows:
        lines += ["## Error capture at fixed review budgets", "",
                  "| Corpus | Partition | System | Budget | Error capture | False-alert rate | Flagged precision | Missed errors | Audio seconds flagged | Oracle ceiling |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append("| {corpus} | {partition} | {system} | {budget:.0%} | {capture:.3f} | {far:.3f} | {precision:.3f} | {missed} | {duration:.1f} | {oracle:.3f} |".format(
                corpus=row["corpus"], partition=row["partition"], system=row["system"],
                budget=float(row["review_budget_fraction"]), capture=float(row["error_capture"]),
                far=float(row["false_alert_rate"]), precision=float(row["flagged_utterance_precision"]),
                missed=int(row["missed_errors"]), duration=float(row["flagged_audio_duration_s"]),
                oracle=float(row["oracle_error_capture_ceiling"])))
        lines += ["", "The false-alert rate uses correctly recognized medication mentions as its denominator. The oracle column is a theoretical ceiling based on gold error labels and is not an implementable detector.", ""]
        if plotted:
            lines += ["![Error capture curves](error_capture_curves.png)", ""]
    else:
        lines += ["No cross-corpus evaluation JSON artifacts were found. No empirical detector results are available.", ""]
    lines += ["## Provenance", "", f"Results directory: `{root}`", f"Evaluation files summarized: {len(evaluation_files)}", "",
              "Report error denominators, corpus revisions, split sizes, speaker/ingredient/consultation counts, and cluster-aware intervals alongside any paper claim.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report": report_path, "metrics_csv": table_path, **({"curve_png": curve_path} if plotted else {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/human_speech")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    outputs = generate_report(args.results_dir, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
