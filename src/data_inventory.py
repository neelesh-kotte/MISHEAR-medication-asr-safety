"""Inventory medication annotations in licensed human-speech corpus manifests.

This stage is intentionally lightweight and must run before ASR/model training.
It counts reference mentions solely for feasibility and data partition design;
inventory labels are never exposed to ``auditor`` or baseline feature APIs.
"""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config_loader import load_yaml_config
from .vocabulary import validate_vocabulary_csv


def normalize_name(value: str) -> str:
    """Normalize a medication alias while retaining word boundaries."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def _split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[|;,]", str(value)) if part.strip()]


def load_vocabulary(path: str | Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    """Load aliases→ingredient IDs and equivalent brand/generic terms from CSV."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Medication vocabulary is empty: {source}")
    aliases: dict[str, set[str]] = defaultdict(set)
    surfaces: dict[str, set[str]] = defaultdict(set)
    ingredients: set[str] = set()
    terms_by_ingredient: dict[str, set[str]] = defaultdict(set)
    combination_aliases: set[str] = set()
    for row in rows:
        # Names in an ``ingredient`` column are not treated as resolved IDs.
        # Only stable identifier fields qualify; combination IDs may be listed
        # in a dedicated combination_ingredients column.
        ingredient_values = _split_values(row.get("ingredient_id") or row.get("ingredient_rxcui")
                                          or row.get("ingredient_ids") or row.get("active_ingredient_ids"))
        ingredient_values += _split_values(row.get("combination_ingredient_ids") or row.get("combination_ingredients_ids"))
        ingredient_values = list(dict.fromkeys(ingredient_values))
        if not ingredient_values:
            ingredient_values = [""]
        ingredients.update(value for value in ingredient_values if value)
        terms: list[tuple[str, list[str]]] = []
        general_columns = ("drug_name", "name", "medication_name", "brand_name", "brand_names",
                           "generic_name", "aliases", "medication_aliases",
                           "brand_generic_equivalents", "equivalent_names")
        for column in general_columns:
            terms.extend((term, ingredient_values) for term in _split_values(row.get(column)))
        inn_terms = _split_values(row.get("inn"))
        if len(ingredient_values) > 1 and len(inn_terms) == len(ingredient_values):
            terms.extend((term, [ingredient]) for term, ingredient in zip(inn_terms, ingredient_values))
        else:
            terms.extend((term, ingredient_values) for term in inn_terms)
        for term, term_ingredients in terms:
            normalized = normalize_name(term)
            if normalized:
                if len(ingredient_values) > 1 and len(term_ingredients) > 1:
                    combination_aliases.add(normalized)
                aliases[normalized].update(term_ingredients)
                surfaces[normalized].add(term.strip())
                for ingredient in term_ingredients:
                    if ingredient:
                        terms_by_ingredient[ingredient].add(term.strip())
    ambiguous = {term: sorted(ids) for term, ids in aliases.items()
                 if len(ids - {""}) > 1 and term not in combination_aliases}
    variants: dict[str, list[str]] = {
        f"normalized:{term}": sorted(names) for term, names in surfaces.items()
        if len({normalize_name(name) for name in names}) > 1
    }
    for ingredient, names in terms_by_ingredient.items():
        values = sorted(names)
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                ratio = SequenceMatcher(None, normalize_name(left).replace(" ", ""),
                                        normalize_name(right).replace(" ", "")).ratio()
                if 0.84 <= ratio < 1.0:
                    variants[f"ingredient:{ingredient}:{left}|{right}"] = [left, right]
    version = next((row.get("vocabulary_release") or row.get("vocabulary_version") or row.get("version") for row in rows
                    if row.get("vocabulary_release") or row.get("vocabulary_version") or row.get("version")), None)
    schema_validation = validate_vocabulary_csv(source, require_resolved=False)
    release_validation = validate_vocabulary_csv(source, require_resolved=True)
    schema_version, vocabulary_status = schema_validation.schema_version, "unknown"
    if not version:
        for sidecar in (source.with_name(source.stem + ".metadata.yaml"),
                        source.with_name(source.stem + ".metadata.json")):
            if sidecar.is_file():
                if sidecar.suffix == ".yaml":
                    try:
                        import yaml
                    except ImportError as exc:
                        raise RuntimeError("Reading vocabulary YAML sidecars requires PyYAML") from exc
                    sidecar_info = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
                else:
                    sidecar_info = json.loads(sidecar.read_text(encoding="utf-8"))
                version = sidecar_info.get("vocabulary_release") or sidecar_info.get("vocabulary_version") or sidecar_info.get("version")
                schema_version = sidecar_info.get("schema_version")
                vocabulary_status = sidecar_info.get("status")
                break
        else:
            schema_version, vocabulary_status = schema_validation.schema_version, "unknown"
    else:
        schema_version, vocabulary_status = schema_validation.schema_version, "unknown"
    metadata = {"path": str(source), "rows": len(rows), "version": version,
                "schema_version": schema_version, "status": vocabulary_status,
                "schema_valid": schema_validation.valid, "schema_errors": list(schema_validation.errors),
                "release_valid": release_validation.valid,
                "release_errors": list(release_validation.errors),
                "resolved_ingredient_count": len(ingredients), "ambiguous_names": ambiguous,
                "spelling_variants": variants,
                "unresolved_vocabulary_rows": sum(
                    not (row.get("ingredient_id") or row.get("ingredient_rxcui") or row.get("ingredient_ids")
                         or row.get("active_ingredient_ids") or row.get("combination_ingredient_ids")
                         or row.get("combination_ingredients_ids"))
                    for row in rows
                )}
    return aliases, surfaces, metadata


def _load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() in {".csv", ".tsv"}:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
            return [dict(row) for row in csv.DictReader(stream, delimiter=delimiter)]
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.casefold() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else value.get("records", [])
    raise ValueError(f"Unsupported manifest format: {path}")


def _load_afrispeech_hub(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Stream AfriSpeech metadata with audio decoding disabled for inventory."""
    if not bool(dataset.get("access_terms_accepted", False)):
        raise PermissionError("Accept AfriSpeech-200 access terms upstream and set access_terms_accepted: true locally")
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError("AfriSpeech Hub ingestion requires `pip install datasets`") from exc
    dataset_id = str(dataset.get("dataset_id", "intronhealth/afrispeech-200"))
    configuration = dataset.get("configuration")
    revision = dataset.get("revision")
    max_records = int(dataset.get("inventory_max_records", 0))
    rows: list[dict[str, Any]] = []
    for split in dataset.get("splits", ("train", "dev", "test")):
        options: dict[str, Any] = {"split": str(split), "streaming": True}
        if revision:
            options["revision"] = str(revision)
        options["trust_remote_code"] = bool(dataset.get("trust_remote_code", True))
        if configuration:
            stream = load_dataset(dataset_id, str(configuration), **options)
        else:
            stream = load_dataset(dataset_id, **options)
        try:
            stream = stream.cast_column("audio", Audio(decode=False))
        except (KeyError, ValueError):
            pass
        for item in stream:
            audio_meta = item.get("audio") or {}
            rows.append({
                "transcript": item.get("transcript", item.get("sentence", "")),
                "speaker_id": item.get("speaker_id", ""),
                "recording_id": item.get("audio_id", item.get("path", "")),
                "audio_path": audio_meta.get("path") if isinstance(audio_meta, Mapping) else item.get("path"),
                "sample_rate_hz": audio_meta.get("sampling_rate") if isinstance(audio_meta, Mapping) else None,
                "duration_s": item.get("duration"), "dataset_split": str(split),
                "domain": item.get("domain"), "accent": item.get("accent"),
            })
            if max_records and len(rows) >= max_records:
                return rows
    return rows


def _load_textgrid(path: Path, dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read Praat long-format TextGrid intervals without requiring Praat."""
    content = path.read_text(encoding="utf-8", errors="replace")
    # Parse tiers and intervals in long TextGrid representation.
    tier_pattern = re.compile(r"item \[\d+\]:\s*(.*?)(?=\n\s*item \[\d+\]:|\Z)", re.S)
    interval_pattern = re.compile(
        r"intervals \[\d+\]:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*\"(.*?)\"",
        re.S,
    )
    rows = []
    consultation = str(dataset.get("consultation_id") or path.stem)
    id_pattern = dataset.get("consultation_id_pattern")
    if id_pattern:
        match = re.search(str(id_pattern), path.stem)
        if match:
            consultation = str(match.group(1) if match.groups() else match.group(0))
    audio_root_value = dataset.get("audio_root")
    audio_root = Path(str(audio_root_value)) if audio_root_value else None
    audio_path = None
    if audio_root and audio_root.is_dir():
        for extension in (".wav", ".flac", ".mp3", ".m4a"):
            exact = audio_root / f"{consultation}{extension}"
            if exact.is_file():
                audio_path = str(exact)
                break
        if audio_path is None:
            matches = sorted(path for path in audio_root.rglob("*")
                             if path.suffix.casefold() in {".wav", ".flac", ".mp3", ".m4a"}
                             and consultation.casefold() in path.stem.casefold())
            if len(matches) == 1:
                audio_path = str(matches[0])
    for tier in tier_pattern.findall(content):
        name_match = re.search(r'name\s*=\s*"(.*?)"', tier)
        tier_name = name_match.group(1) if name_match else "speaker_unknown"
        for start, end, text in interval_pattern.findall(tier):
            text = text.replace('""', '"').strip()
            if not text:
                continue
            rows.append({"transcript": text, "speaker_id": f"{consultation}:{tier_name}",
                         "speaker_role": tier_name, "consultation_id": consultation,
                         "recording_id": path.stem, "audio_path": audio_path,
                         "start_s": float(start), "end_s": float(end),
                         "dataset_split": dataset.get("split", "unknown")})
    return rows


def load_dataset_records(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load configured annotation manifest or recursively read TextGrid files."""
    local_manifest = Path(str(dataset.get("manifest", ""))) if dataset.get("manifest") else None
    if dataset.get("source") == "huggingface" and not (local_manifest and local_manifest.is_file()):
        return _load_afrispeech_hub(dataset)
    manifest = dataset.get("manifest")
    if manifest:
        path = Path(str(manifest))
        if path.is_file():
            return _load_table(path)
        if path.is_dir():
            files = sorted(path.rglob("*.TextGrid")) + sorted(path.rglob("*.textgrid"))
            return [row for file in files for row in _load_textgrid(file, dataset)]
        # Fall back to a corpus root of TextGrids when the optional normalized
        # manifest has not yet been built.
        root = Path(str(dataset.get("root", "")))
        if root.is_dir():
            files = sorted(root.rglob("*.TextGrid")) + sorted(root.rglob("*.textgrid"))
            return [row for file in files for row in _load_textgrid(file, dataset)]
        return []
    root = Path(str(dataset.get("root", "")))
    if not root.is_dir():
        return []
    files = sorted(root.rglob("*.TextGrid")) + sorted(root.rglob("*.textgrid"))
    return [row for file in files for row in _load_textgrid(file, dataset)]


def inventory_dataset(
    name: str, dataset: Mapping[str, Any], aliases: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Count exact vocabulary mentions and unresolved explicit medication labels."""
    records = load_dataset_records(dataset)
    text_field = str(dataset.get("text_field", "transcript"))
    medication_field = str(dataset.get("medication_field", "medication_name"))
    speaker_field = str(dataset.get("speaker_field", "speaker_id"))
    consultation_field = str(dataset.get("consultation_field", "consultation_id"))
    # Longest-first matching over token boundaries; do not count a name as a
    # substring of another word. Combo product aliases resolve to every mapped ID.
    phrases = sorted(((key, key.split(), ids) for key, ids in aliases.items()),
                     key=lambda item: len(item[1]), reverse=True)
    ingredient_counts: Counter[str] = Counter()
    medication_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    unresolved: Counter[str] = Counter()
    split_mentions: Counter[str] = Counter()
    accent_mentions: Counter[str] = Counter()
    ingredient_speakers: dict[str, set[str]] = defaultdict(set)
    partition_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "utterance_ids": set(), "ingredient_ids": set(), "speaker_ids": set(), "consultation_ids": set(),
        "medication_mentions": 0,
    })
    medication_utterances = 0
    total_mentions = 0
    speakers: set[str] = set()
    consultations: set[str] = set()
    rows_with_mentions = []
    for row_index, row in enumerate(records):
        text = str(row.get(text_field, row.get("text", "")) or "")
        explicit = _split_values(row.get(medication_field) or row.get("drug_name") or row.get("medication"))
        found: list[tuple[str, set[str]]] = []
        if text:
            tokens = normalize_name(text).split()
            i = 0
            while i < len(tokens):
                match = next(((key, ids) for key, phrase, ids in phrases
                              if tokens[i:i + len(phrase)] == phrase), None)
                if match:
                    found.append(match)
                    i += len(match[0].split())
                else:
                    i += 1
        terms = explicit or [item[0] for item in found]
        unresolved_terms = [term for term in explicit if normalize_name(term) not in aliases]
        unresolved.update(unresolved_terms)
        if terms:
            medication_utterances += 1
            speaker_value = row.get(speaker_field) or row.get("speaker")
            consultation_value = row.get(consultation_field) or row.get("consultation")
            speaker = str(speaker_value) if speaker_value else ""
            consultation = str(consultation_value) if consultation_value else ""
            if speaker:
                speakers.add(speaker)
                speaker_counts[speaker] += len(terms)
            if consultation:
                consultations.add(consultation)
            partition = str(row.get(dataset.get("split_field", "split")) or row.get("dataset_split") or "unknown")
            partition_record = partition_stats[partition]
            partition_record["utterance_ids"].add(row_index)
            partition_record["medication_mentions"] += len(terms)
            split_mentions[partition] += len(terms)
            if speaker:
                partition_record["speaker_ids"].add(speaker)
            if consultation:
                partition_record["consultation_ids"].add(consultation)
            for term in terms:
                normalized = normalize_name(term)
                medication_counts[normalized] += 1
                accent_mentions[str(row.get("accent") or row.get("accent_group") or "unknown")] += 1
                ids = aliases.get(normalized, set())
                if not (ids - {""}) and term not in unresolved_terms:
                    unresolved[term] += 1
                for ingredient in ids - {""}:
                    ingredient_counts[ingredient] += 1
                    partition_record["ingredient_ids"].add(ingredient)
                    if speaker:
                        ingredient_speakers[ingredient].add(speaker)
            total_mentions += len(terms)
            rows_with_mentions.append({"row_index": row_index, "recording_id": row.get("recording_id"),
                                       "speaker_id": speaker or None, "consultation_id": consultation or None,
                                       "medication_mentions": terms})
    known_terms = list(aliases)
    oov_near_matches = {}
    for term, count in unresolved.items():
        normalized = normalize_name(term).replace(" ", "")
        candidates = sorted(((SequenceMatcher(None, normalized, candidate.replace(" ", "")).ratio(), candidate)
                             for candidate in known_terms), reverse=True)[:3]
        oov_near_matches[term] = {"count": count, "candidate_aliases": [
            {"alias": candidate, "similarity": round(score, 4)} for score, candidate in candidates if score >= .70
        ]}
    return {
        "dataset": name, "license": dataset.get("license", "not_recorded"),
        "source": dataset.get("dataset_id", dataset.get("source", "local_manifest")),
        "source_revision": dataset.get("revision") or dataset.get("source_revision") or dataset.get("commit"),
        "manifest_path": dataset.get("manifest"),
        "records_scanned": len(records), "medication_utterances": medication_utterances,
        "medication_mentions": total_mentions, "distinct_medication_names": len(medication_counts),
        "resolved_ingredient_ids": len(ingredient_counts), "distinct_speakers": len(speakers),
        "distinct_consultations": len(consultations), "mentions_per_ingredient": dict(ingredient_counts),
        "mentions_per_speaker": dict(speaker_counts), "mentions_per_medication": dict(medication_counts),
        "mentions_by_partition": dict(split_mentions), "mentions_by_accent": dict(accent_mentions),
        "partition_breakdown": {key: {
            "medication_utterances": len(value["utterance_ids"]),
            "medication_mentions": value["medication_mentions"],
            "distinct_ingredients": len(value["ingredient_ids"]),
            "distinct_speakers": len(value["speaker_ids"]),
            "distinct_consultations": len(value["consultation_ids"]),
        } for key, value in partition_stats.items()},
        "speakers_per_ingredient": {key: len(value) for key, value in ingredient_speakers.items()},
        "sparse_ingredients_below_5_mentions": sorted(key for key, count in ingredient_counts.items() if count < 5),
        "mention_skew": {"max_ingredient_share": max(ingredient_counts.values(), default=0) / max(sum(ingredient_counts.values()), 1),
                         "max_speaker_share": max(speaker_counts.values(), default=0) / max(sum(speaker_counts.values()), 1)},
        "mentions_missing_speaker_id": sum(bool(row.get("medication_mentions")) and not row.get("speaker_id") for row in rows_with_mentions),
        "mentions_missing_consultation_id": sum(bool(row.get("medication_mentions")) and not row.get("consultation_id") for row in rows_with_mentions),
        "official_partition_audit": _audit_official_partitions(
            records, dataset.get("split_field", "split"), speaker_field
        ) if dataset.get("official_speaker_split") else None,
        "unresolved_terms": dict(unresolved),
        "oov_near_matches": oov_near_matches,
    }


def _audit_official_partitions(records: Sequence[Mapping[str, Any]], split_field: str, speaker_field: str) -> dict[str, Any]:
    """Return official train/dev/test speaker-overlap diagnostics."""
    by_split: dict[str, set[str]] = defaultdict(set)
    missing = 0
    for row in records:
        split = str(row.get(split_field) or row.get("dataset_split") or "").casefold()
        split = "dev" if split in {"validation", "valid"} else split
        speaker = row.get(speaker_field) or row.get("speaker")
        if not split or not speaker:
            missing += 1
        elif split in {"train", "dev", "test"}:
            by_split[split].add(str(speaker))
    overlaps = {f"{a}:{b}": sorted(by_split[a] & by_split[b])
                for i, a in enumerate(("train", "dev", "test"))
                for b in ("train", "dev", "test")[i + 1:] if by_split[a] & by_split[b]}
    complete = all(by_split.get(key) for key in ("train", "dev", "test"))
    return {"complete": complete, "speaker_disjoint": complete and not overlaps,
            "speaker_count_by_split": {key: len(by_split[key]) for key in ("train", "dev", "test")},
            "missing_split_or_speaker_rows": missing, "overlaps": overlaps}


def build_feasibility_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory configured corpora and apply stop rules before model training."""
    vocabulary_path = config.get("vocabulary_csv", "data/vocabulary/medication_vocabulary.csv")
    try:
        aliases, _, vocabulary_metadata = load_vocabulary(vocabulary_path)
    except Exception as exc:
        aliases = {}
        vocabulary_metadata = {"path": str(vocabulary_path), "rows": 0, "version": None,
                               "resolved_ingredient_count": 0, "ambiguous_names": {},
                               "spelling_variants": {}, "unresolved_vocabulary_rows": 0,
                               "inventory_error": f"{type(exc).__name__}: {exc}"}
    corpus_reports: dict[str, dict[str, Any]] = {}
    for name, spec in config.get("datasets", {}).items():
        try:
            corpus_reports[name] = inventory_dataset(name, spec, aliases)
        except Exception as exc:
            corpus_reports[name] = {
                "dataset": name, "license": spec.get("license", "not_recorded"),
                "source": spec.get("dataset_id", spec.get("source", "local_manifest")),
                "source_revision": spec.get("revision") or spec.get("source_revision") or spec.get("commit"),
                "manifest_path": spec.get("manifest"),
                "records_scanned": 0, "medication_utterances": 0, "medication_mentions": 0,
                "distinct_medication_names": 0, "resolved_ingredient_ids": 0,
                "distinct_speakers": 0, "distinct_consultations": 0,
                "mentions_per_ingredient": {}, "mentions_per_speaker": {},
                "mentions_per_medication": {}, "unresolved_terms": {},
                "inventory_error": f"{type(exc).__name__}: {exc}",
            }
    thresholds = config.get("stopping_rules", {})
    synthetic_fixture = str(config.get("purpose", "")).casefold().startswith("synthetic")
    total_mentions = sum(row["medication_mentions"] for row in corpus_reports.values())
    total_ingredients = len({ingredient for row in corpus_reports.values()
                             for ingredient, count in row["mentions_per_ingredient"].items() if count})
    total_speakers = len({speaker for row in corpus_reports.values()
                          for speaker, count in row["mentions_per_speaker"].items() if count})
    reasons = []
    if vocabulary_metadata.get("inventory_error"):
        reasons.append(f"Vocabulary: {vocabulary_metadata['inventory_error']}")
    for field, actual in (("minimum_mentions", total_mentions), ("minimum_ingredients", total_ingredients),
                          ("minimum_speakers", total_speakers)):
        if actual < int(thresholds.get(field, 1)):
            reasons.append(f"{field}: observed {actual}, required {thresholds.get(field, 1)}")
    if not any(report["records_scanned"] for report in corpus_reports.values()):
        reasons.append("No corpus manifests/audio annotations were found; configure dataset paths")
    if thresholds.get("require_versioned_vocabulary", True) and not vocabulary_metadata.get("version"):
        reasons.append("Medication vocabulary has no source-audited release identifier")
    if thresholds.get("require_versioned_vocabulary", True):
        if vocabulary_metadata.get("schema_version") != "1.0" or not vocabulary_metadata.get("schema_valid"):
            reasons.append("Medication vocabulary does not pass the v1.0 schema validation")
        if not vocabulary_metadata.get("release_valid"):
            reasons.append("Medication vocabulary is not a complete source-audited release")
        if vocabulary_metadata.get("unresolved_vocabulary_rows", 0):
            reasons.append(f"Medication vocabulary has {vocabulary_metadata['unresolved_vocabulary_rows']} unresolved product rows")
    for name, report in corpus_reports.items():
        specification = config.get("datasets", {}).get(name, {})
        revision = str(specification.get("revision") or specification.get("source_revision") or specification.get("commit") or "")
        revision_is_pinned = bool(re.fullmatch(r"[0-9a-fA-F]{40}", revision))
        if synthetic_fixture:
            revision_is_pinned = revision.startswith("fixture-")
        if thresholds.get("require_dataset_revision", True) and not revision_is_pinned:
            reasons.append(f"{name}: exact 40-character upstream commit is missing or still a placeholder")
        if not synthetic_fixture and ("afrispeech" in name.casefold() or "afrispeech" in str(specification.get("dataset_id", "")).casefold()):
            if not bool(specification.get("access_terms_accepted", False)):
                reasons.append(f"{name}: confirm AfriSpeech access terms were accepted before corpus use")
        if report.get("inventory_error"):
            reasons.append(f"{name}: {report['inventory_error']}")
        if report["medication_mentions"] and report["mentions_missing_speaker_id"]:
            reasons.append(f"{name}: {report['mentions_missing_speaker_id']} medication mentions lack speaker IDs")
        if report["medication_mentions"] and report["mentions_missing_consultation_id"]:
            reasons.append(f"{name}: {report['mentions_missing_consultation_id']} medication mentions lack consultation IDs")
        partition_audit = report.get("official_partition_audit")
        if partition_audit and not partition_audit["speaker_disjoint"]:
            reasons.append(f"{name}: official speaker-disjoint train/dev/test metadata failed or was incomplete")
    status = ("FEASIBLE_FOR_PIPELINE_SMOKE_TEST" if synthetic_fixture else "FEASIBLE_FOR_PILOT") if not reasons else "STOP_BEFORE_TRAINING"
    return {"project": "MISHEAR cross-corpus medication transcription auditing",
            "status": status, "purpose": config.get("purpose", "real_human_speech_research"),
            "training_authorized": bool(not reasons and not synthetic_fixture),
            "stopping_reasons": reasons, "vocabulary": vocabulary_metadata,
            "total_medication_mentions": total_mentions, "total_resolved_ingredients": total_ingredients,
            "total_speakers": total_speakers, "corpora": corpus_reports,
            "note": ("Synthetic stubs validate pipeline mechanics only; they do not establish corpus feasibility, ASR errors, or model performance."
                     if synthetic_fixture else
                     "Inventory is descriptive. It does not establish ASR error labels or model performance.")}


def write_report(report: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    """Write machine-readable JSON and concise feasibility Markdown."""
    json_target, md_target = Path(json_path), Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# MISHEAR corpus feasibility inventory", "", f"**Status: {report['status']}**",
             f"**Purpose: {report.get('purpose', 'unspecified')}**",
             f"**Training authorized: {report.get('training_authorized', False)}**", "",
             "This inventory is a stop/go check before model training. Counts do not imply ASR errors or clinical validity.", "",
             "| Corpus | License | Source revision | Records scanned | Medication utterances | Mentions | Ingredients | Speakers | Consultations |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["corpora"].items():
        lines.append(f"| {name} | {row['license']} | {row.get('source_revision') or 'not pinned'} | {row['records_scanned']} | {row['medication_utterances']} | {row['medication_mentions']} | {row['resolved_ingredient_ids']} | {row['distinct_speakers']} | {row['distinct_consultations']} |")
    lines += ["", f"Vocabulary version: {report['vocabulary'].get('version') or 'not recorded'}",
              f"Total mentions / ingredients / speakers: {report['total_medication_mentions']} / {report['total_resolved_ingredients']} / {report['total_speakers']}", ""]
    if report["stopping_reasons"]:
        lines += ["## Stop reasons", ""] + [f"- {reason}" for reason in report["stopping_reasons"]]
    md_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/corpus_config.yaml")
    parser.add_argument("--json", default="results/inventory/feasibility.json")
    parser.add_argument("--markdown", default="results/inventory/feasibility.md")
    args = parser.parse_args()
    report = build_feasibility_report(load_yaml_config(args.config))
    write_report(report, args.json, args.markdown)
    print(f"{report['status']}: wrote {args.markdown}")


if __name__ == "__main__":
    main()

