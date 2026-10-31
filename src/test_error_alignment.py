"""Reference-to-ASR alignment, medication event labels, and stratified metrics.

This is a reusable evaluation module despite the ``test_`` filename. It runs
only after ASR predictions have been generated; reference labels are joined here
for scoring and are never passed to an ASR runner or predictive feature builder.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def tokenize(text: str) -> list[str]:
    """Case-fold text and retain alphanumeric word tokens for WER alignment."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[\w]+", normalized, flags=re.UNICODE)


def normalize_drug_name(name: str) -> str:
    return " ".join(tokenize(name))


def _align_tokens(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Minimum-edit alignment with explicit reference/hypothesis token indices."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0], back[i][0] = i, "deletion"
    for j in range(1, cols):
        cost[0][j], back[0][j] = j, "insertion"
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j], back[i][j] = cost[i - 1][j - 1], "match"
                continue
            # Deterministic tie preference: substitution, deletion, insertion.
            choices = (
                (cost[i - 1][j - 1] + 1, "substitution"),
                (cost[i - 1][j] + 1, "deletion"),
                (cost[i][j - 1] + 1, "insertion"),
            )
            cost[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    alignment: list[dict[str, Any]] = []
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0, "matches": 0}
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = back[i][j]
        if operation in ("match", "substitution"):
            alignment.append({
                "operation": operation,
                "reference_index": i - 1,
                "hypothesis_index": j - 1,
                "reference_token": reference[i - 1],
                "hypothesis_token": hypothesis[j - 1],
            })
            counts["matches" if operation == "match" else "substitutions"] += int(operation == "match" or operation == "substitution")
            i -= 1
            j -= 1
        elif operation == "deletion":
            alignment.append({
                "operation": operation,
                "reference_index": i - 1,
                "hypothesis_index": None,
                "reference_token": reference[i - 1],
                "hypothesis_token": None,
            })
            counts["deletions"] += 1
            i -= 1
        else:
            alignment.append({
                "operation": operation,
                "reference_index": None,
                "hypothesis_index": j - 1,
                "reference_token": None,
                "hypothesis_token": hypothesis[j - 1],
            })
            counts["insertions"] += 1
            j -= 1
    alignment.reverse()
    return alignment, counts


def _drug_map(drug_vocabulary: Mapping[str, str] | Iterable[str] | None) -> dict[str, str]:
    if drug_vocabulary is None:
        return {}
    if isinstance(drug_vocabulary, Mapping):
        items = drug_vocabulary.items()
    else:
        items = ((name, name) for name in drug_vocabulary)
    result: dict[str, str] = {}
    for name, ingredient in items:
        key = normalize_drug_name(str(name))
        if key:
            result[key] = str(ingredient)
    return result


def find_drug_mentions(text: str, drug_vocabulary: Mapping[str, str] | Iterable[str]) -> list[dict[str, Any]]:
    """Find longest non-overlapping vocabulary mentions in a tokenized string."""
    tokens = tokenize(text)
    vocabulary = _drug_map(drug_vocabulary)
    forms = sorted(((key.split(), key, ingredient) for key, ingredient in vocabulary.items()),
                   key=lambda item: (len(item[0]), len(item[1])), reverse=True)
    mentions: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for start in range(len(tokens)):
        if start in occupied:
            continue
        for form_tokens, canonical, ingredient in forms:
            end = start + len(form_tokens)
            if end <= len(tokens) and tokens[start:end] == form_tokens and not any(i in occupied for i in range(start, end)):
                mentions.append({
                    "start": start,
                    "end": end,
                    "text": " ".join(tokens[start:end]),
                    "canonical_name": canonical,
                    "ingredient_id": ingredient,
                })
                occupied.update(range(start, end))
                break
    return mentions


def _find_reference_drug_span(reference_tokens: Sequence[str], intended_drug: str) -> tuple[int, int] | None:
    expected = tokenize(intended_drug)
    if not expected:
        return None
    for start in range(len(reference_tokens) - len(expected) + 1):
        if list(reference_tokens[start:start + len(expected)]) == expected:
            return start, start + len(expected)
    return None


def align_transcription(
    reference_text: str,
    hypothesis_text: str,
    *,
    intended_drug: str,
    drug_vocabulary: Mapping[str, str] | Iterable[str] | None = None,
    accepted_drug_forms: Iterable[str] = (),
    strict_reference_match: bool = True,
) -> dict[str, Any]:
    """Align one transcript and classify the intended medication event.

    Medication event classes are `correct`, `substitution`, and `deletion` for
    the intended mention, plus countable extra `insertion` mentions. A wrong
    word replacing the expected name is a substitution even if it is not in the
    candidate medication vocabulary. An extra recognized drug is an insertion.
    """
    reference_tokens = tokenize(reference_text)
    hypothesis_tokens = tokenize(hypothesis_text)
    intended = normalize_drug_name(intended_drug)
    reference_span = _find_reference_drug_span(reference_tokens, intended_drug)
    if reference_span is None and strict_reference_match:
        raise ValueError(f"Intended drug {intended_drug!r} was not found in reference: {reference_text!r}")
    alignment, word_counts = _align_tokens(reference_tokens, hypothesis_tokens)
    accepted = {intended, *(normalize_drug_name(name) for name in accepted_drug_forms)}
    vocabulary = _drug_map(drug_vocabulary)
    expected_ingredient = vocabulary.get(intended)
    accepted_ingredients = {
        vocabulary[form] for form in accepted if form in vocabulary
    }
    if expected_ingredient:
        accepted_ingredients.add(expected_ingredient)

    mapped_hypothesis_indices: set[int] = set()
    if reference_span is not None:
        ref_start, ref_end = reference_span
        for step in alignment:
            hyp_idx = step["hypothesis_index"]
            ref_idx = step["reference_index"]
            if hyp_idx is not None and ref_idx is not None and ref_start <= ref_idx < ref_end:
                mapped_hypothesis_indices.add(hyp_idx)
            elif hyp_idx is not None and ref_idx is None:
                # Include insertions anchored immediately within the drug span.
                prior_refs = [item["reference_index"] for item in alignment
                              if item["hypothesis_index"] is not None
                              and item["hypothesis_index"] < hyp_idx
                              and item["reference_index"] is not None]
                next_refs = [item["reference_index"] for item in alignment
                             if item["hypothesis_index"] is not None
                             and item["hypothesis_index"] > hyp_idx
                             and item["reference_index"] is not None]
                prior = max(prior_refs) if prior_refs else -1
                following = min(next_refs) if next_refs else len(reference_tokens)
                if ref_start <= prior + 1 <= ref_end or ref_start <= following <= ref_end:
                    mapped_hypothesis_indices.add(hyp_idx)

    predicted_slot = " ".join(hypothesis_tokens[index] for index in sorted(mapped_hypothesis_indices))
    all_mentions = find_drug_mentions(hypothesis_text, vocabulary) if vocabulary else []
    slot_mentions = [mention for mention in all_mentions
                     if any(index in mapped_hypothesis_indices for index in range(mention["start"], mention["end"]))]
    expected_found_in_slot = any(
        mention["canonical_name"] in accepted
        or (mention["ingredient_id"] in accepted_ingredients if accepted_ingredients else False)
        for mention in slot_mentions
    )
    if expected_found_in_slot or predicted_slot in accepted:
        event = "correct"
    elif not mapped_hypothesis_indices:
        # A wrong drug inserted exactly at the deleted reference slot is a substitution.
        nearby_wrong = [mention for mention in all_mentions
                        if mention["ingredient_id"] not in accepted_ingredients
                        and abs(mention["start"] - ref_start) <= 2] if reference_span else []
        event = "substitution" if nearby_wrong else "deletion"
        if nearby_wrong:
            slot_mentions = nearby_wrong
    else:
        event = "substitution"

    predicted_drug = slot_mentions[0]["canonical_name"] if slot_mentions else predicted_slot
    if event == "correct" and not slot_mentions:
        predicted_drug = predicted_slot or intended

    # Count extra vocabulary mentions, excluding one wrong mention used as the
    # replacement in a substitution; same-ingredient mentions count as correct.
    selected_replacement = slot_mentions[0] if event == "substitution" and slot_mentions else None
    extra_mentions = []
    for mention in all_mentions:
        if event == "correct" and (
            mention["canonical_name"] in accepted
            or (mention["ingredient_id"] in accepted_ingredients if accepted_ingredients else False)
        ):
            continue
        if selected_replacement is mention:
            continue
        extra_mentions.append(mention)

    return {
        "reference_text": reference_text,
        "hypothesis_text": hypothesis_text,
        "reference_tokens": reference_tokens,
        "hypothesis_tokens": hypothesis_tokens,
        "word_alignment": alignment,
        "word_substitutions": word_counts["substitutions"],
        "word_deletions": word_counts["deletions"],
        "word_insertions": word_counts["insertions"],
        "reference_word_count": len(reference_tokens),
        "wer": (word_counts["substitutions"] + word_counts["deletions"] + word_counts["insertions"])
               / max(len(reference_tokens), 1),
        "intended_drug": intended_drug,
        "predicted_drug": predicted_drug,
        "medication_event": event,
        "medication_substitutions": int(event == "substitution"),
        "medication_deletions": int(event == "deletion"),
        "medication_insertions": len(extra_mentions),
        "reference_drug_mentions": int(reference_span is not None),
        "extra_drug_mentions": [mention["canonical_name"] for mention in extra_mentions],
    }


def _recording_key(record: Mapping[str, Any]) -> str:
    for field in ("recording_id", "utterance_id", "audio_id"):
        if record.get(field):
            return str(record[field])
    for field in ("output_path", "audio_path", "path"):
        if record.get(field):
            return Path(str(record[field])).stem
    raise ValueError(f"Could not determine recording ID from record: {record!r}")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV, JSON array/object, or JSONL record file."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if suffix in (".jsonl", ".ndjson"):
        with source.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return [dict(row) for row in payload["records"]]
        if isinstance(payload, dict):
            return [payload]
    raise ValueError(f"Unsupported record file extension: {source.suffix}")


def attach_reference_metadata(
    clean_audio_metadata: Iterable[Mapping[str, Any]],
    augmented_audio_metadata: Iterable[Mapping[str, Any]],
    *,
    clean_path_fields: Sequence[str] = ("audio_path", "path"),
    augmented_input_field: str = "input_path",
) -> list[dict[str, Any]]:
    """Attach original TTS references/drug labels to DSP-derived audio rows.

    DSP metadata points from ``input_path`` (clean TTS clip) to ``output_path``
    (augmented recording). The merge preserves the original fictional sentence
    and drug ID while giving the augmented output its own recording ID.
    """
    reference_by_key: dict[str, dict[str, Any]] = {}
    for item in clean_audio_metadata:
        row = dict(item)
        keys = {_recording_key(row)}
        for field in clean_path_fields:
            if row.get(field):
                keys.add(Path(str(row[field])).stem)
        for key in keys:
            reference_by_key[key] = row
    output = []
    for item in augmented_audio_metadata:
        augmented = dict(item)
        if not augmented.get(augmented_input_field):
            raise ValueError(f"Augmentation row has no {augmented_input_field!r}: {augmented!r}")
        clean_key = Path(str(augmented[augmented_input_field])).stem
        reference = reference_by_key.get(clean_key)
        if reference is None:
            raise KeyError(f"No clean TTS metadata matched DSP input {clean_key!r}")
        output_path = augmented.get("output_path")
        recording_id = Path(str(output_path)).stem if output_path else str(augmented.get("recording_id", clean_key))
        output.append({
            **reference,
            **augmented,
            "recording_id": recording_id,
            "reference_text": reference.get("sentence", reference.get("reference_text", "")),
            "intended_drug": reference.get("drug_name", reference.get("intended_drug", "")),
            "ingredient_id": reference.get("ingredient_id", reference.get("ingredient_rxcui", "")),
        })
    return output


def pair_audio_metadata_with_asr(
    audio_metadata: Iterable[Mapping[str, Any]],
    asr_records: Iterable[Mapping[str, Any]],
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Join augmented audio metadata to one or more ASR records by recording ID.

    This is the post-inference point where generated reference text and intended
    drug labels may be attached to ASR predictions for scoring.
    """
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for item in audio_metadata:
        row = dict(item)
        key = _recording_key(row)
        if key in metadata_by_id:
            raise ValueError(f"Duplicate audio metadata key {key!r}")
        metadata_by_id[key] = row
    output: list[dict[str, Any]] = []
    matched: set[str] = set()
    for item in asr_records:
        asr = dict(item)
        key = _recording_key(asr)
        metadata = metadata_by_id.get(key)
        if metadata is None:
            if strict:
                raise KeyError(f"ASR recording {key!r} has no matching generated-audio metadata.")
            continue
        matched.add(key)
        output.append({**metadata, **asr, "recording_id": key})
    if strict:
        unmatched = set(metadata_by_id) - matched
        if unmatched:
            raise KeyError(f"Generated audio has no ASR result for IDs: {sorted(unmatched)[:10]}")
    return output


def align_evaluation_records(
    records: Iterable[Mapping[str, Any]],
    *,
    drug_vocabulary: Mapping[str, str] | Iterable[str] | None = None,
    accepted_forms_by_ingredient: Mapping[str, Iterable[str]] | None = None,
    reference_field: str = "sentence",
    hypothesis_field: str = "transcript",
    intended_drug_field: str = "drug_name",
) -> list[dict[str, Any]]:
    """Align joined rows and carry condition/recognizer metadata into labels."""
    output = []
    vocabulary_map = _drug_map(drug_vocabulary)
    for record in records:
        row = dict(record)
        reference = str(row.get(reference_field, row.get("reference_text", "")))
        hypothesis = str(row.get(hypothesis_field, row.get("asr_transcript", "")))
        intended = str(row.get(intended_drug_field, row.get("intended_drug", "")))
        if not reference or not intended:
            raise ValueError("Each evaluation record must contain reference text and intended drug name.")
        ingredient = str(row.get("ingredient_id") or row.get("ingredient_rxcui") or vocabulary_map.get(normalize_drug_name(intended), ""))
        accepted = (accepted_forms_by_ingredient or {}).get(ingredient, ())
        alignment = align_transcription(
            reference,
            hypothesis,
            intended_drug=intended,
            drug_vocabulary=drug_vocabulary,
            accepted_drug_forms=accepted,
        )
        output.append({**row, **alignment})
    return output


def compute_error_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Calculate WER and medication substitution/deletion/insertion rates."""
    rows = list(records)
    word_sub = sum(int(row.get("word_substitutions", 0)) for row in rows)
    word_del = sum(int(row.get("word_deletions", 0)) for row in rows)
    word_ins = sum(int(row.get("word_insertions", 0)) for row in rows)
    ref_words = sum(int(row.get("reference_word_count", len(tokenize(str(row.get("reference_text", "")))))) for row in rows)
    drug_ref = sum(int(row.get("reference_drug_mentions", 1)) for row in rows)
    drug_sub = sum(int(row.get("medication_substitutions", 0)) for row in rows)
    drug_del = sum(int(row.get("medication_deletions", 0)) for row in rows)
    drug_ins = sum(int(row.get("medication_insertions", 0)) for row in rows)
    return {
        "recording_count": len(rows),
        "reference_word_count": ref_words,
        "word_substitutions": word_sub,
        "word_deletions": word_del,
        "word_insertions": word_ins,
        "wer": (word_sub + word_del + word_ins) / max(ref_words, 1),
        "reference_drug_mentions": drug_ref,
        "medication_substitutions": drug_sub,
        "medication_deletions": drug_del,
        "medication_insertions": drug_ins,
        "drug_token_error_rate": (drug_sub + drug_del + drug_ins) / max(drug_ref, 1),
        "medication_substitution_rate": drug_sub / max(drug_ref, 1),
        "medication_deletion_rate": drug_del / max(drug_ref, 1),
        "medication_insertion_rate": drug_ins / max(drug_ref, 1),
        "medication_error_record_rate": sum(
            row.get("medication_event") in {"substitution", "deletion"}
            or int(row.get("medication_insertions", 0)) > 0
            for row in rows
        ) / max(len(rows), 1),
    }


DEFAULT_STRATIFICATION_FIELDS = (
    "recognizer", "acoustic_condition_id", "condition_family", "noise_kind",
    "target_snr_db", "measured_snr_db", "rt60_target_s", "measured_rt60_s",
    "clipping_threshold", "context_condition", "voice", "rate_wpm",
)


def stratified_error_metrics(
    records: Iterable[Mapping[str, Any]],
    group_fields: Sequence[str] = DEFAULT_STRATIFICATION_FIELDS,
) -> list[dict[str, Any]]:
    """Return metric rows grouped by whichever DSP/model fields are present."""
    rows = [dict(row) for row in records]
    fields = [field for field in group_fields if any(field in row for row in rows)]
    if not fields:
        return [{**compute_error_metrics(rows)}]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        output.append({**dict(zip(fields, key)), **compute_error_metrics(group)})
    return output


def save_alignment_outputs(
    aligned_records: Sequence[Mapping[str, Any]],
    *,
    records_jsonl: str | Path,
    metrics_csv: str | Path,
) -> None:
    """Write per-record alignments and condition-stratified metric table."""
    records_path, metrics_path = Path(records_jsonl), Path(metrics_csv)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as stream:
        for row in aligned_records:
            stream.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
    metrics = stratified_error_metrics(aligned_records)
    if metrics:
        fieldnames = list(dict.fromkeys(key for row in metrics for key in row))
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
