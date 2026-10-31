"""Corpus-first CLI for MISHEAR's human-speech protocol.

The default action is the inventory stop/go report. This entrypoint does not
launch training unless corpus feasibility, blinding, and frozen partitions are
handled explicitly by later experiment stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config_loader import load_yaml_config
from src.data_inventory import build_feasibility_report, load_vocabulary, normalize_name, write_report
from src.data_splits import (
    create_ingredient_holdout_split,
    create_primock_consultation_splits,
    load_manifest,
    verify_official_speaker_partitions,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/corpus_config.yaml")
    parser.add_argument("--stage", choices=("inventory", "verify-afrispeech", "ingredient-holdout", "primock-split", "all"),
                        default="inventory")
    parser.add_argument("--manifest", help="Normalized local CSV/JSON/JSONL manifest for a split stage")
    parser.add_argument("--held-out-ingredients", help="Comma-separated stable ingredient IDs")
    parser.add_argument("--source-split-field", default="split",
                        help="Official AfriSpeech partition field; auto-detected when present")
    parser.add_argument("--output", default="results/human_speech/splits.jsonl")
    parser.add_argument("--inventory-json", default="results/human_speech/inventory.json")
    parser.add_argument("--inventory-markdown", default="results/human_speech/feasibility.md")
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    if args.stage == "all":
        # `all` is only an end-to-end synthetic-fixture smoke run. It is not
        # the production ASR/train/test orchestrator.
        synthetic_fixture = str(config.get("purpose", "")).casefold().startswith("synthetic")
        if not synthetic_fixture:
            output_root = Path("results/human_speech")
            output_root.mkdir(parents=True, exist_ok=True)
            report = build_feasibility_report(config)
            write_report(report, output_root / "inventory.json", output_root / "feasibility.md")
            print(json.dumps({"status": report["status"], "reasons": report["stopping_reasons"]}, indent=2))
            if report["status"] == "STOP_BEFORE_TRAINING":
                raise SystemExit(2)
            raise SystemExit(
                "The real-corpus all-stage ASR/model orchestration is not implemented; refusing to report a smoke split as a production evaluation."
            )
        dry = config.get("dry_run", {})
        output_root = Path(dry.get("output_dir", "results/stub_dry_run"))
        output_root.mkdir(parents=True, exist_ok=True)
        report = build_feasibility_report(config)
        write_report(report, output_root / "feasibility.json", output_root / "feasibility.md")
        if report["status"] == "STOP_BEFORE_TRAINING":
            print(json.dumps({"status": report["status"], "reasons": report["stopping_reasons"]}, indent=2))
            raise SystemExit(2)

        datasets = config.get("datasets", {})
        afri_spec = next((spec for name, spec in datasets.items() if "afri" in name.casefold()), None)
        primock_spec = next((spec for name, spec in datasets.items() if "primock" in name.casefold()), None)
        if not afri_spec or not primock_spec:
            raise SystemExit("stage all requires both AfriSpeech and PriMock57 manifest entries")
        afri = load_manifest(afri_spec["manifest"])
        primock = load_manifest(primock_spec["manifest"])
        split_audit = verify_official_speaker_partitions(
            afri, split_field=afri_spec.get("split_field", "split"),
            speaker_field=afri_spec.get("speaker_field", "speaker_id"))
        aliases, _, _ = load_vocabulary(config.get("vocabulary_csv", "data/vocabulary/medication_vocabulary.csv"))
        enriched_afri = []
        for row in afri:
            term = str(row.get(afri_spec.get("medication_field", "medication_name"), ""))
            ids = sorted(aliases.get(normalize_name(term), set()) - {""})
            if not ids:
                raise ValueError(f"Stub manifest contains vocabulary-unresolved term {term!r}")
            enriched_afri.append({**row, "ingredient_ids": ids})
        held = dry.get("held_out_ingredients", ["RXNORM:5553"])
        holdout = create_ingredient_holdout_split(
            enriched_afri, held, ingredient_field="ingredient_ids",
            official_split_field=afri_spec.get("split_field", "split"))
        mock_splits = create_primock_consultation_splits(primock, seed=int(dry.get("seed", 20261030)))
        split_dir = output_root / "splits"
        for name, subset in holdout.items():
            _write_jsonl(split_dir / f"afrispeech_ingredient_{name}.jsonl", subset)
        for name, subset in mock_splits.items():
            _write_jsonl(split_dir / f"primock_consultation_{name}.jsonl", subset)

        artifact_summary = None
        artifact_config = dry.get("paper_artifacts_config")
        if artifact_config:
            from src.generate_paper_artifacts import run_artifact_generation
            artifact_summary = run_artifact_generation(load_yaml_config(artifact_config))
        summary = {"status": "SMOKE_TEST_COMPLETE", "data_kind": "synthetic fixtures only",
                   "inventory_status": report["status"], "training_authorized": report["training_authorized"],
                   "official_speaker_split_audit": split_audit,
                   "ingredient_holdout_rows": {key: len(rows) for key, rows in holdout.items()},
                   "primock_consultation_split_rows": {key: len(rows) for key, rows in mock_splits.items()},
                   "paper_artifacts": artifact_summary,
                   "warning": "No ASR inference or real-corpus result is produced by this smoke run."}
        (output_root / "all_stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    if args.stage == "inventory":
        report = build_feasibility_report(config)
        write_report(report, args.inventory_json, args.inventory_markdown)
        print(json.dumps({"status": report["status"], "reasons": report["stopping_reasons"]}, indent=2))
        return
    if not args.manifest:
        parser.error(f"--manifest is required for --stage {args.stage}")
    rows = load_manifest(args.manifest)
    if args.stage == "verify-afrispeech":
        audit = verify_official_speaker_partitions(rows)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps(audit, indent=2))
        return
    if args.stage == "ingredient-holdout":
        if not args.held_out_ingredients:
            parser.error("--held-out-ingredients is required for ingredient-holdout")
        official_field = args.source_split_field if any(
            row.get(args.source_split_field) or row.get("dataset_split") for row in rows
        ) else None
        split = create_ingredient_holdout_split(
            rows, args.held_out_ingredients.split(","), official_split_field=official_field
        )
        output_path = Path(args.output)
        for name, subset in split.items():
            _write_jsonl(output_path.with_name(f"ingredient_holdout_{name}.jsonl"), subset)
        print(json.dumps({key: len(value) for key, value in split.items()}, indent=2))
        return
    partitions = create_primock_consultation_splits(rows)
    output_path = Path(args.output)
    for name, subset in partitions.items():
        _write_jsonl(output_path.with_name(f"primock_{name}.jsonl"), subset)
    print(json.dumps({key: len(value) for key, value in partitions.items()}, indent=2))


if __name__ == "__main__":
    main()
