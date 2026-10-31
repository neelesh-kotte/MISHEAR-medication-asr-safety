"""Word/character error metrics and condition-stratified evaluation tables."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .test_error_alignment import tokenize


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return Levenshtein distance using O(min(n,m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute token WER; empty references use denominator 1 by convention."""
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    return edit_distance(ref, hyp) / max(len(ref), 1)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute case-folded character error rate, ignoring whitespace."""
    ref = "".join(tokenize(reference))
    hyp = "".join(tokenize(hypothesis))
    return edit_distance(ref, hyp) / max(len(ref), 1)


def record_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Aggregate WER/CER from aligned transcript rows without averaging rates."""
    rows = list(records)
    word_edits = char_edits = words = chars = 0
    for row in rows:
        reference = str(row.get("reference_text", row.get("sentence", "")))
        hypothesis = str(row.get("hypothesis_text", row.get("transcript", "")))
        ref_words, hyp_words = tokenize(reference), tokenize(hypothesis)
        ref_chars, hyp_chars = "".join(ref_words), "".join(hyp_words)
        word_edits += edit_distance(ref_words, hyp_words)
        char_edits += edit_distance(ref_chars, hyp_chars)
        words += len(ref_words)
        chars += len(ref_chars)
    return {
        "recording_count": len(rows),
        "reference_word_count": words,
        "word_edits": word_edits,
        "wer": word_edits / max(words, 1),
        "reference_character_count": chars,
        "character_edits": char_edits,
        "cer": char_edits / max(chars, 1),
    }


DEFAULT_GROUP_FIELDS = (
    "phonological_similarity", "phonetic_similarity", "dosage_form",
    "condition_family", "condition_id", "noise_kind", "target_snr_db",
    "measured_snr_db", "rt60_target_s", "measured_rt60_s", "recognizer",
)


def stratified_metrics(
    records: Iterable[Mapping[str, Any]],
    group_fields: Sequence[str] = DEFAULT_GROUP_FIELDS,
) -> list[dict[str, Any]]:
    """Aggregate error metrics for each present similarity/form/condition key."""
    rows = [dict(row) for row in records]
    fields = [key for key in group_fields if any(key in row for row in rows)]
    if not fields:
        return [record_metrics(rows)] if rows else []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    return [
        {**dict(zip(fields, key)), **record_metrics(group)}
        for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))
    ]


__all__ = ["edit_distance", "word_error_rate", "character_error_rate", "record_metrics", "stratified_metrics"]
