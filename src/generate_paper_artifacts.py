"""Build paper tables and error-capture figures from frozen human-speech results.

All reported metrics describe offline error concentration/prioritization. This
module does not infer clinical outcomes or workflow effects.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import (cluster_bootstrap_review_budgets, evaluate_review_budgets,
                         evaluate_subgroups, write_summary_csv)

BUDGETS = (0.01, 0.05, 0.10, 0.20)


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_table(rows: Sequence[Mapping[str, Any]], markdown: Path, latex: Path) -> None:
    if not rows:
        markdown.write_text("No verified inventory rows are available.\n", encoding="utf-8")
        latex.write_text("% No verified inventory rows are available.\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    md = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    md += ["| " + " | ".join(str(row.get(key, "")) for key in columns) + " |" for row in rows]
    markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    def esc(value: Any) -> str:
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
    tex = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
           " & ".join(map(esc, columns)) + " " + chr(92) * 2, r"\hline"]
    tex += [" & ".join(esc(row.get(key, "")) for key in columns) + " " + chr(92) * 2 for row in rows]
    tex += [r"\hline", r"\end{tabular}"]
    latex.write_text("\n".join(tex) + "\n", encoding="utf-8")


def _inventory_table(inventory_json: str | None) -> list[dict[str, Any]]:
    if not inventory_json or not Path(inventory_json).is_file():
        return []
    report = json.loads(Path(inventory_json).read_text(encoding="utf-8"))
    result = []
    for corpus, row in report.get("corpora", {}).items():
        result.append({"Corpus": corpus, "Records": row.get("records_scanned", 0),
                       "Medication utterances": row.get("medication_utterances", 0),
                       "Mentions": row.get("medication_mentions", 0),
                       "Resolved ingredients": row.get("resolved_ingredient_ids", 0),
                       "Speakers": row.get("distinct_speakers", 0),
                       "Consultations": row.get("distinct_consultations", 0),
                       "Partition counts": json.dumps(row.get("partition_breakdown", {}), sort_keys=True),
                       "Accent counts": json.dumps(row.get("mentions_by_accent", {}), sort_keys=True),
                       "Feasibility": report.get("status", "unknown")})
    return result


def run_artifact_generation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run configured development ablations and summarize frozen evaluation rows."""
    out = Path(config.get("output_dir", "results/paper_artifacts"))
    out.mkdir(parents=True, exist_ok=True)
    data_kind = str(config.get("data_kind", "real evaluation rows; verify provenance"))
    inventory_rows = _inventory_table(config.get("inventory_json"))
    _write_table(inventory_rows, out / "table1_dataset_inventory.md", out / "table1_dataset_inventory.tex")

    ablation_rows: list[dict[str, Any]] = []
    if config.get("ablation"):
        from .ablations import run_ablation_experiments
        spec = config["ablation"]
        ablation_rows = run_ablation_experiments(
            _read_csv(spec["training_csv"]), _read_csv(spec["development_csv"]),
            label_field=spec.get("label_field", "is_medication_error"),
            review_budgets=spec.get("review_budgets", BUDGETS),
            bootstrap_replicates=int(spec.get("bootstrap_replicates", 1000)),
            cluster_fields=spec.get("cluster_fields", ["speaker_id", "consultation_id"]),
        )
        for row in ablation_rows:
            row["data_kind"] = data_kind
        write_summary_csv(ablation_rows, out / "table2_ablation_summary.csv")
        _write_table(ablation_rows, out / "table2_ablation_summary.md", out / "table2_ablation_summary.tex")
    else:
        (out / "table2_ablation_summary.md").write_text(
            "Ablation results unavailable: provide development-only train/dev feature manifests.\n", encoding="utf-8")
        (out / "table2_ablation_summary.tex").write_text("% Ablation results unavailable.\n", encoding="utf-8")

    evaluations: dict[str, Any] = {}
    curve_rows: list[dict[str, Any]] = []
    for name, spec in config.get("evaluation_sets", {}).items():
        rows = _read_jsonl(spec["predictions_jsonl"])
        fixture_only = bool(rows) and all(bool(row.get("synthetic_fixture")) for row in rows)
        set_data_kind = "synthetic fixtures only (not corpus results)" if fixture_only else data_kind
        score_fields = spec["score_fields"]
        corpus_result: dict[str, Any] = {"systems": {}}
        for system, score_field in score_fields.items():
            scored = [dict(row, risk_score=float(row[score_field])) for row in rows]
            metrics = evaluate_review_budgets(scored, score_field="risk_score",
                label_field=spec.get("label_field", "is_medication_error"), budgets=spec.get("review_budgets", BUDGETS))
            bootstrap = cluster_bootstrap_review_budgets(
                scored, cluster_fields=spec.get("cluster_fields", ["speaker_id", "consultation_id"]),
                budgets=spec.get("review_budgets", BUDGETS), score_field="risk_score",
                label_field=spec.get("label_field", "is_medication_error"),
                n_bootstrap=int(spec.get("bootstrap_replicates", 1000)), seed=int(config.get("seed", 20261030)))
            corpus_result["systems"][system] = {**metrics, "clustered_bootstrap": bootstrap}
            for point in metrics["budget_summary"]:
                ci = bootstrap["error_capture_ci_95_by_budget"].get(str(point["review_budget_fraction"]))
                curve_rows.append({"evaluation_set": name, "system": system,
                    "data_kind": set_data_kind,
                    "review_budget_fraction": point["review_budget_fraction"],
                    "error_capture": point["error_capture"], "ci95_low": ci[0] if ci else None,
                    "ci95_high": ci[1] if ci else None,
                    "oracle_error_capture_ceiling": point["oracle_error_capture_ceiling"]})
        subgroup_rows = []
        for row in rows:
            row = dict(row)
            row.setdefault("corpus", spec.get("corpus", name))
            subgroup_rows.append(row)
        # Report subgroup results per frozen system score to keep each ranking independent.
        corpus_result["subgroups"] = {}
        for system, score_field in score_fields.items():
            system_rows = [dict(row, risk_score=float(row[score_field])) for row in subgroup_rows]
            corpus_result["subgroups"][system] = evaluate_subgroups(
                system_rows, subgroup_fields=spec.get("subgroup_fields", [
                    "accent_group", "speaker_seen_status", "ingredient_seen_status", "corpus"]),
                score_field="risk_score", label_field=spec.get("label_field", "is_medication_error"),
                budgets=spec.get("review_budgets", BUDGETS))
        corpus_result["data_kind"] = set_data_kind
        evaluations[name] = corpus_result
    (out / "evaluation_results.json").write_text(json.dumps({"data_kind": data_kind, "interpretation":
        "Error concentration and prioritization only; no workflow reduction or harm-prevention claims.",
        "evaluation_sets": evaluations}, indent=2, allow_nan=False), encoding="utf-8")
    if curve_rows:
        write_summary_csv(curve_rows, out / "error_capture_curves.csv")
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Install matplotlib to produce the required publication figure") from exc
        sets = list(config.get("evaluation_sets", {}))
        fig, axes = plt.subplots(1, max(1, len(sets)), figsize=(5.2 * max(1, len(sets)), 4.2), squeeze=False)
        for axis, set_name in zip(axes[0], sets):
            subset = [row for row in curve_rows if row["evaluation_set"] == set_name]
            for system in dict.fromkeys(row["system"] for row in subset):
                data = [row for row in subset if row["system"] == system]
                x = [100 * row["review_budget_fraction"] for row in data]
                y = [row["error_capture"] for row in data]
                axis.plot(x, y, marker="o", label=system)
                low = [row["ci95_low"] for row in data]
                high = [row["ci95_high"] for row in data]
                if all(value is not None for value in low + high):
                    axis.fill_between(x, low, high, alpha=.15)
            fixture_label = "\nSYNTHETIC FIXTURE — NOT CORPUS RESULTS" if "synthetic" in str(
                next(row["data_kind"] for row in subset)
            ).casefold() else ""
            axis.set(title=set_name + fixture_label, xlabel="Review budget (% of utterances)", ylabel="Medication errors captured")
            axis.set_ylim(0, 1); axis.grid(alpha=.25); axis.legend(fontsize="small")
        title_prefix = "SYNTHETIC SMOKE TEST — " if "synthetic" in data_kind.casefold() else ""
        fig.suptitle(title_prefix + "Error concentration by offline review budget")
        fig.tight_layout()
        fig.savefig(out / "figure1_error_capture.png", dpi=400, bbox_inches="tight")
        fig.savefig(out / "figure1_error_capture.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        (out / "figure1_status.md").write_text("Figure unavailable: no labeled evaluation prediction sets configured.\n", encoding="utf-8")
    return {"output_dir": str(out), "inventory_rows": len(inventory_rows),
            "ablation_rows": len(ablation_rows), "evaluation_sets": list(evaluations),
            "figure_generated": bool(curve_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/paper_artifacts.yaml")
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required") from exc
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    print(json.dumps(run_artifact_generation(config), indent=2))


if __name__ == "__main__":
    main()
