"""Corpus-aware split utilities for AfriSpeech-200 and PriMock57.

Reference annotations are permitted here for dataset partition construction,
but this module's split outputs are metadata only. Gold drug names, spans, and
timestamps must never be copied into ASR or auditor feature inputs.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data_inventory import load_dataset_records, load_vocabulary


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a normalized CSV, JSON, or JSONL corpus manifest."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".csv", ".tsv"}:
        with source.open(newline="", encoding="utf-8-sig") as stream:
            delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
            return [dict(row) for row in csv.DictReader(stream, delimiter=delimiter)]
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else value.get("records", [])
    raise ValueError(f"Unsupported manifest format: {source.suffix}")


def load_afrispeech_manifest(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load AfriSpeech metadata in the dataset's original train/dev/test partitions."""
    return load_dataset_records(config)


def load_primock57_manifest(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load normalized PriMock57 utterances or parse upstream Praat TextGrids."""
    rows = load_dataset_records(config)
    if not rows:
        raise FileNotFoundError("No PriMock57 manifest/TextGrid records found; check corpus_config.yaml paths")
    for row in rows:
        if not (row.get("consultation_id") or row.get("consultation")):
            raise ValueError("PriMock57 records must retain consultation IDs")
    if config.get("require_audio", True) and any(not row.get("audio_path") for row in rows):
        raise FileNotFoundError("PriMock57 transcript rows are missing matched audio_path values")
    return rows


def verify_official_speaker_partitions(
    records: Sequence[Mapping[str, Any]], *, split_field: str = "split",
    speaker_field: str = "speaker_id", required_splits: Sequence[str] = ("train", "dev", "test"),
) -> dict[str, Any]:
    """Verify required official partitions exist and share no speaker IDs."""
    by_split: dict[str, set[str]] = defaultdict(set)
    def split_name(row: Mapping[str, Any]) -> str:
        split = str(row.get(split_field) or row.get("dataset_split") or "").casefold()
        return "dev" if split in {"validation", "valid"} else split
    for index, row in enumerate(records):
        split = split_name(row)
        if split not in required_splits:
            raise ValueError(f"Unknown official partition {split!r} at record {index}")
        speaker = row.get(speaker_field) or row.get("speaker")
        if not split or not speaker:
            raise ValueError(f"Missing official split or speaker ID at record {index}")
        by_split[split].add(str(speaker))
    missing = [part for part in required_splits if not by_split.get(part)]
    if missing:
        raise ValueError(f"Official split metadata missing required partitions: {missing}")
    overlaps: dict[str, list[str]] = {}
    for i, left in enumerate(required_splits):
        for right in required_splits[i + 1:]:
            shared = sorted(by_split[left] & by_split[right])
            if shared:
                overlaps[f"{left}:{right}"] = shared
    if overlaps:
        raise ValueError(f"Official AfriSpeech partitions are not speaker-disjoint: {overlaps}")
    return {"speaker_disjoint": True, "speaker_count_by_split": {part: len(by_split[part]) for part in required_splits},
            "record_count_by_split": {part: sum(split_name(row) == part for row in records) for part in required_splits}}


def preserve_official_partitions(
    records: Sequence[Mapping[str, Any]], *, split_field: str = "split", speaker_field: str = "speaker_id",
) -> dict[str, list[dict[str, Any]]]:
    """Return official AfriSpeech train/dev/test records without re-randomizing."""
    verify_official_speaker_partitions(records, split_field=split_field, speaker_field=speaker_field)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for source in records:
        row = dict(source)
        split = str(row.get(split_field) or row.get("dataset_split") or "").casefold()
        split = "dev" if split in {"validation", "valid"} else split
        row["official_split"] = split
        output[split].append(row)
    return output


def _ingredients(row: Mapping[str, Any], field: str) -> set[str]:
    value = row.get(field) or row.get("ingredient_ids") or row.get("ingredient_id") or row.get("ingredient_rxcui") or row.get("ingredient")
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(part).strip() for part in value if str(part).strip()}
    return {part.strip() for part in str(value).replace(";", "|").replace(",", "|").split("|") if part.strip()}


def create_ingredient_holdout_split(
    records: Sequence[Mapping[str, Any]], held_out_ingredients: Iterable[str], *,
    ingredient_field: str = "ingredient_id", mixed_mention_policy: str = "test",
    official_split_field: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Hold selected ingredients out of supervised training, including combo rows.

    ``mixed_mention_policy='test'`` sends any utterance mentioning a held-out
    ingredient to test. ``exclude`` drops such mixed-ingredient rows from both
    partitions so the held-out evaluation contains only pure mentions.
    """
    held = {str(value) for value in held_out_ingredients}
    if not held:
        raise ValueError("At least one held-out ingredient must be specified")
    if mixed_mention_policy not in {"test", "exclude"}:
        raise ValueError("mixed_mention_policy must be 'test' or 'exclude'")
    output: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": [], "excluded_mixed": [],
        "excluded_heldout_development": [], "unused_nonheldout_test": [],
    }
    found: set[str] = set()
    found_in_test: set[str] = set()
    for row in records:
        ids = _ingredients(row, ingredient_field)
        if not ids:
            raise ValueError("Every labeled utterance must include resolved ingredient ID(s)")
        selected = ids & held
        found.update(selected)
        source_split = str(row.get(official_split_field) or row.get("dataset_split") or "").casefold() if official_split_field else ""
        source_split = "validation" if source_split in {"dev", "valid"} else source_split
        if official_split_field and source_split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown official source partition {source_split!r}")
        if not selected:
            if official_split_field and source_split == "test":
                output["unused_nonheldout_test"].append(dict(row, ingredient_holdout_split="unused_nonheldout_test"))
            else:
                destination = "validation" if official_split_field and source_split == "validation" else "train"
                output[destination].append(dict(row, ingredient_holdout_split=destination))
        elif mixed_mention_policy == "exclude" and ids - held:
            output["excluded_mixed"].append(dict(row, ingredient_holdout_split="excluded_mixed"))
        elif official_split_field and source_split != "test":
            # Any held ingredient from official train/dev is kept out of fitting
            # and tuning, even though it is not part of the official test score.
            output["excluded_heldout_development"].append(
                dict(row, ingredient_holdout_split="excluded_heldout_development"))
        else:
            if source_split == "test":
                found_in_test.update(selected)
            output["test"].append(dict(row, ingredient_holdout_split="test"))
    absent = held - found
    if absent:
        raise ValueError(f"Held-out ingredients are absent from this corpus: {sorted(absent)}")
    if official_split_field and held - found_in_test:
        raise ValueError(f"Held-out ingredients lack official test examples: {sorted(held - found_in_test)}")
    return output


def create_primock_consultation_splits(
    records: Sequence[Mapping[str, Any]], *, consultation_field: str = "consultation_id",
    fractions: Mapping[str, float] | None = None, seed: int = 20261030,
) -> dict[str, list[dict[str, Any]]]:
    """Assign whole PriMock57 consultations to train/dev/test without utterance leakage."""
    fractions = dict(fractions or {"train": 0.6, "dev": 0.2, "test": 0.2})
    if set(fractions) != {"train", "dev", "test"} or any(float(x) <= 0 for x in fractions.values()):
        raise ValueError("fractions must define positive train/dev/test proportions")
    total = sum(map(float, fractions.values()))
    fractions = {key: float(value) / total for key, value in fractions.items()}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        consultation = row.get(consultation_field) or row.get("consultation") or row.get("recording_id")
        if not consultation:
            raise ValueError("Each PriMock57 utterance needs a consultation ID")
        groups[str(consultation)].append(row)
    if len(groups) < 3:
        raise ValueError("At least three consultations are required for train/dev/test splitting")
    names = list(groups)
    random.Random(seed).shuffle(names)
    target_counts = {key: fractions[key] * len(names) for key in fractions}
    assigned: dict[str, list[str]] = {key: [] for key in fractions}
    counts = dict.fromkeys(fractions, 0)
    for name in names:
        split = min(fractions, key=lambda key: (counts[key] - target_counts[key], counts[key]))
        assigned[split].append(name)
        counts[split] += 1
    return {split: [dict(row, primock_split=split) for name in assigned[split] for row in groups[name]]
            for split in fractions}


def resolve_vocabulary_terms(text_terms: Iterable[str], vocabulary_csv: str | Path) -> dict[str, Any]:
    """Resolve corpus medication strings through vocabulary brand/generic aliases."""
    aliases, _, metadata = load_vocabulary(vocabulary_csv)
    resolved, unresolved = {}, []
    from .data_inventory import normalize_name
    for term in text_terms:
        key = normalize_name(str(term))
        ids = sorted(aliases.get(key, set()) - {""})
        if ids:
            resolved[str(term)] = ids
        else:
            unresolved.append(str(term))
    return {"resolved": resolved, "unresolved": unresolved, "vocabulary": metadata}

