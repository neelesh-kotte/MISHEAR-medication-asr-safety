"""Tests for corpus partition boundaries, blinded interfaces, and budget metrics."""

import inspect

import pytest

from src.auditor import audit_segment
from src.auditor import ctc_log_likelihood
from src.data_splits import (
    create_ingredient_holdout_split,
    create_primock_consultation_splits,
    verify_official_speaker_partitions,
)
from src.data_inventory import inventory_dataset, load_vocabulary
from src.evaluation import evaluate_review_budgets


def test_official_speaker_partitions_are_verified() -> None:
    records = [
        {"split": "train", "speaker_id": "s1"},
        {"split": "dev", "speaker_id": "s2"},
        {"split": "test", "speaker_id": "s3"},
    ]
    assert verify_official_speaker_partitions(records)["speaker_disjoint"]
    with pytest.raises(ValueError, match="not speaker-disjoint"):
        verify_official_speaker_partitions(records + [{"split": "test", "speaker_id": "s1"}])


def test_ingredient_holdout_keeps_mixed_utterance_whole() -> None:
    rows = [
        {"id": "a", "ingredient_ids": ["i1"]},
        {"id": "b", "ingredient_ids": ["i2", "i3"]},
    ]
    split = create_ingredient_holdout_split(rows, ["i2"])
    assert [row["id"] for row in split["train"]] == ["a"]
    assert [row["id"] for row in split["test"]] == ["b"]


def test_primock_splits_preserve_consultations() -> None:
    records = [{"consultation_id": c, "utterance_id": f"{c}-{i}"}
               for c in ("c1", "c2", "c3", "c4", "c5", "c6") for i in range(3)]
    partitions = create_primock_consultation_splits(records, seed=2)
    consultations = [{row["consultation_id"] for row in partition} for partition in partitions.values()]
    assert all(not consultations[i] & consultations[j] for i in range(3) for j in range(i + 1, 3))


def test_far_uses_correct_medication_mentions_and_budget_capture() -> None:
    records = [
        {"is_medication_error": 1, "risk_score": .9, "medication_error_mentions": 1,
         "correct_medication_mentions": 0, "audio_duration_s": 2},
        {"is_medication_error": 0, "risk_score": .8, "medication_error_mentions": 0,
         "correct_medication_mentions": 2, "audio_duration_s": 3},
        {"is_medication_error": 1, "risk_score": .1, "medication_error_mentions": 1,
         "correct_medication_mentions": 0, "audio_duration_s": 1},
    ]
    result = evaluate_review_budgets(records, budgets=[1 / 3])
    row = result["budget_summary"][0]
    assert row["error_capture"] == .5
    assert row["false_alert_rate"] == 0
    assert row["flagged_audio_duration_s"] == 2
    assert "Error concentration and prioritization" in result["interpretation"]


def test_inference_api_does_not_accept_gold_reference_fields() -> None:
    params = set(inspect.signature(audit_segment).parameters)
    assert not {"reference", "reference_drug", "gold_timestamps", "correctness"} & params


def test_ctc_candidate_support_is_finite() -> None:
    import numpy as np
    log_probs = np.log(np.asarray([[.8, .2], [.7, .3], [.6, .4]], dtype=float))
    assert np.isfinite(ctc_log_likelihood(log_probs, [1], blank_id=0))


def test_inventory_resolves_brand_and_combination_components(tmp_path) -> None:
    vocab_path = tmp_path / "vocabulary.csv"
    vocab_path.write_text(
        "drug_name,brand_name,generic_name,ingredient_id,combination_ingredient_ids,vocabulary_version\n"
        "brandone,Brand One,Generic One,i1,,v1\n"
        "combo,Combo Product,,i2,i3|i4,v1\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "transcript,speaker_id,consultation_id,medication_name\n"
        "Please take Brand One,s1,c1,\n"
        "Start Combo Product,s2,c2,\n",
        encoding="utf-8",
    )
    aliases, _, metadata = load_vocabulary(vocab_path)
    result = inventory_dataset(
        "toy", {"manifest": str(manifest_path), "text_field": "transcript"}, aliases,
    )
    assert metadata["version"] == "v1"
    assert result["medication_mentions"] == 2
    assert set(result["mentions_per_ingredient"]) == {"i1", "i2", "i3", "i4"}

