"""Reference/hypothesis audit labels and audio sample-rate utilities."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from math import gcd
from typing import Mapping, Sequence


ERROR_TYPES = (
    "correct", "paired_drug_substitution", "other_drug_substitution",
    "non_drug_rendering", "complete_deletion", "insertion_hallucination",
)


def normalize_text(text: str) -> str:
    """Unicode-normalize and casefold text while preserving token boundaries."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def _tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def _contains_phrase(text: str, phrase: str) -> bool:
    tokens, target = _tokens(text), _tokens(phrase)
    return bool(target) and any(tokens[i:i + len(target)] == target for i in range(len(tokens) - len(target) + 1))


def build_equivalence_map(entries: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Map brand/generic name variants to a shared ingredient identity."""
    result: dict[str, str] = {}
    for row in entries:
        identity = (row.get("ingredient_id") or row.get("ingredient_rxcui")
                    or row.get("ingredient_ids") or row.get("active_ingredient_ids")
                    or row.get("ingredient") or row.get("name") or row.get("drug_name"))
        if not identity:
            continue
        if isinstance(identity, (list, tuple, set)):
            identity = "|".join(sorted(str(item).strip() for item in identity if str(item).strip()))
        else:
            identity = "|".join(sorted(part.strip() for part in re.split(r"[|;,]", str(identity)) if part.strip()))
        for key in ("name", "drug_name", "brand_name", "generic_name", "aliases",
                    "medication_aliases", "brand_generic_equivalents", "equivalent_names"):
            value = row.get(key)
            if not value:
                continue
            values = value.split("|") if key == "aliases" else [value]
            for alias in values:
                normalized = normalize_text(alias)
                if normalized:
                    result[normalized] = str(identity)
    return result


def classify_alignment_error(
    intended_drug: str,
    hypothesis: str,
    *,
    paired_drugs: Sequence[str] = (),
    medication_vocabulary: Sequence[str] = (),
    equivalence_map: Mapping[str, str] | None = None,
    splinter_similarity_threshold: float = 0.62,
) -> str:
    """Classify one intended mention's ASR rendering into a discrete audit type.

    Multiword exact phrases use token boundaries (so ``metformin`` does not
    match ``metformin-like`` after normalization by accident). A compacted
    token comparison catches plausible phonetic splinterings such as
    ``digoxin`` → ``dikod sin`` without labeling unrelated text as a drug.
    """
    expected = normalize_text(intended_drug)
    observed = normalize_text(hypothesis)
    equivalences = {normalize_text(k): str(v) for k, v in (equivalence_map or {}).items()}
    expected_found = _contains_phrase(hypothesis, intended_drug)
    expected_id = equivalences.get(expected)
    for known in medication_vocabulary:
        if _contains_phrase(hypothesis, known) and expected_id and equivalences.get(normalize_text(known)) == expected_id:
            expected_found = True
    observed_drugs = [drug for drug in medication_vocabulary if _contains_phrase(hypothesis, drug)]
    if expected_found:
        extra = [drug for drug in observed_drugs
                 if equivalences.get(normalize_text(drug), normalize_text(drug)) !=
                 (expected_id or expected)]
        return "insertion_hallucination" if extra else "correct"
    for alternative in paired_drugs:
        if _contains_phrase(hypothesis, alternative):
            return "paired_drug_substitution"
    for drug in medication_vocabulary:
        if normalize_text(drug) != expected and _contains_phrase(hypothesis, drug):
            return "other_drug_substitution"
    # Match against compacted characters to catch token splitting while still
    # requiring the hypothesis to contain a plausible rendering.
    observed_tokens = _tokens(hypothesis)
    expected_compact = expected.replace(" ", "")
    if expected_compact and observed_tokens:
        max_window = min(len(observed_tokens), max(len(_tokens(intended_drug)) + 2, 1))
        for width in range(1, max_window + 1):
            for start in range(len(observed_tokens) - width + 1):
                fragment = "".join(observed_tokens[start:start + width])
                if SequenceMatcher(None, expected_compact, fragment).ratio() >= splinter_similarity_threshold:
                    return "non_drug_rendering"
    if not observed or not observed_drugs:
        return "complete_deletion"
    return "other_drug_substitution"


def resample_waveform(waveform, source_rate_hz: int, target_rate_hz: int):
    """Polyphase-resample a waveform and return ``(audio, target_rate_hz)``."""
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate_hz == target_rate_hz:
        return waveform, target_rate_hz
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("Sample-rate conversion requires scipy") from exc
    divisor = gcd(int(source_rate_hz), int(target_rate_hz))
    converted = resample_poly(waveform, target_rate_hz // divisor, source_rate_hz // divisor, axis=0)
    return converted, target_rate_hz

