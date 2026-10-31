"""Transcript alignment helpers and JSONL/CSV artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .test_error_alignment import (
    align_evaluation_records,
    align_transcription,
    load_records,
    pair_audio_metadata_with_asr,
    save_alignment_outputs,
    stratified_error_metrics,
    tokenize,
)
from .vocabulary import load_vocabulary_csv


def vocabulary_ingredient_map(csv_path: str | Path) -> dict[str, str]:
    """Load medication names and their ingredient identifiers from the CSV."""
    mapping: dict[str, str] = {}
    for drug in load_vocabulary_csv(csv_path):
        name = str(getattr(drug, "name", "") or "")
        ingredient = str(
            getattr(drug, "ingredient_rxcui", "")
            or getattr(drug, "ingredient_id", "")
            or getattr(drug, "ingredient", "")
            or name
        )
        if name:
            mapping[name] = ingredient
    return mapping


def align_files(
    audio_metadata_path: str | Path,
    asr_jsonl_path: str | Path,
    vocabulary_csv: str | Path,
    records_jsonl: str | Path,
    metrics_csv: str | Path,
) -> list[dict[str, Any]]:
    """Join ASR and audio metadata, align against references, and save outputs."""
    audio_metadata = load_records(audio_metadata_path)
    asr_records = load_records(asr_jsonl_path)
    joined = pair_audio_metadata_with_asr(audio_metadata, asr_records)
    aligned = align_evaluation_records(
        joined, drug_vocabulary=vocabulary_ingredient_map(vocabulary_csv)
    )
    save_alignment_outputs(aligned, records_jsonl=records_jsonl, metrics_csv=metrics_csv)
    return aligned


__all__ = [
    "align_evaluation_records", "align_files", "align_transcription", "load_records",
    "pair_audio_metadata_with_asr", "save_alignment_outputs", "stratified_error_metrics", "tokenize",
]
