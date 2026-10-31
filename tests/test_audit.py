"""Tests for normalization, label boundaries, equivalence, and resampling."""

import numpy as np
import pytest

from src.audit_utils import build_equivalence_map, classify_alignment_error, normalize_text, resample_waveform


def test_text_normalization_is_unicode_aware_and_tokenized() -> None:
    assert normalize_text("  HYDROXYZINE—XR ") == "hydroxyzine xr"


def test_brand_and_generic_names_share_ingredient_identity() -> None:
    equivalence = build_equivalence_map([
        {"brand_name": "BrandMed", "generic_name": "generic drug", "ingredient_id": "ingredient-1"}
    ])
    result = classify_alignment_error(
        "generic drug", "BrandMed", medication_vocabulary=["BrandMed", "generic drug"],
        equivalence_map=equivalence,
    )
    assert result == "correct"


def test_resampling_updates_waveform_length_and_rate() -> None:
    waveform = np.ones(800, dtype=np.float32)
    converted, rate = resample_waveform(waveform, 8000, 16000)
    assert rate == 16000
    assert len(converted) == 1600


def test_medication_matching_respects_complete_token_boundaries() -> None:
    # The shorter string must not count as a substring match for the full name.
    result = classify_alignment_error("metformin", "met", medication_vocabulary=["metformin"])
    assert result != "correct"


def test_multitoken_splinter_is_not_counted_as_exact_name() -> None:
    result = classify_alignment_error("digoxin", "dikod sin", medication_vocabulary=["digoxin"])
    assert result == "non_drug_rendering"

