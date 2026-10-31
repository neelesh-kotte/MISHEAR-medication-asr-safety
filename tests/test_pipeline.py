"""Fast unit tests for the public MISHEAR pipeline interfaces."""

from pathlib import Path

import pytest

from src.alignment import align_transcription
from src.config_loader import load_yaml_config
from src.metrics import character_error_rate, word_error_rate


ROOT = Path(__file__).resolve().parents[1]


def test_load_pipeline_config_and_validate_backend() -> None:
    config = load_yaml_config(ROOT / "config" / "pipeline_config.yaml")
    assert config["asr"]["backend"]
    assert 0 <= float(config["false_alert_budget"]) <= 1


def test_error_rates() -> None:
    assert word_error_rate("take hydroxyzine today", "take hydralazine today") == pytest.approx(1 / 3)
    assert character_error_rate("abc", "adc") == pytest.approx(1 / 3)


def test_alignment_marks_sound_alike_drug_as_substitution() -> None:
    aligned = align_transcription(
        "Take hydroxyzine today",
        "Take hydralazine today",
        intended_drug="hydroxyzine",
        drug_vocabulary={"hydroxyzine": "ingredient-a", "hydralazine": "ingredient-b"},
    )
    assert aligned["medication_event"] == "substitution"
    assert aligned["predicted_drug"] == "hydralazine"
