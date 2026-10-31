"""Pairwise phonetic, lexical, and fictional-context feature extraction.

CMUdict provides pronunciation-based features when installed and when both
medication names occur in its lexicon. Otherwise those fields stay missing and
a clearly labeled Soundex string-similarity proxy is returned. Soundex is not
a pronunciation dictionary and should not be reported as phoneme distance.

Context features are optional. ``MaskedLMContextScorer`` computes candidate
pseudo-log-likelihood and contextual embedding similarity from a model supplied
by the caller and a fictional sentence template containing ``{drug}``. Do not
pass patient text or reference labels. Review the chosen model's provenance and
terms before use (the project audit flags ClinicalBERT/MIMIC provenance).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from .vocabulary import jaro_winkler, normalize_name, phonetic_similarity, soundex


_VOWEL_CLASS = {
    "IY": "vowel_high_front", "IH": "vowel_high_front", "EY": "vowel_mid_front",
    "EH": "vowel_mid_front", "AE": "vowel_low_front", "AA": "vowel_low_back",
    "AO": "vowel_mid_back", "OW": "vowel_mid_back", "UH": "vowel_high_back",
    "UW": "vowel_high_back", "AH": "vowel_central", "ER": "vowel_central",
}

# ARPAbet place, manner, voicing; features are intentionally broad categories.
_CONSONANT_CLASS = {
    "P": ("bilabial", "stop", "voiceless"), "B": ("bilabial", "stop", "voiced"),
    "T": ("alveolar", "stop", "voiceless"), "D": ("alveolar", "stop", "voiced"),
    "K": ("velar", "stop", "voiceless"), "G": ("velar", "stop", "voiced"),
    "CH": ("postalveolar", "affricate", "voiceless"), "JH": ("postalveolar", "affricate", "voiced"),
    "F": ("labiodental", "fricative", "voiceless"), "V": ("labiodental", "fricative", "voiced"),
    "TH": ("dental", "fricative", "voiceless"), "DH": ("dental", "fricative", "voiced"),
    "S": ("alveolar", "fricative", "voiceless"), "Z": ("alveolar", "fricative", "voiced"),
    "SH": ("postalveolar", "fricative", "voiceless"), "ZH": ("postalveolar", "fricative", "voiced"),
    "HH": ("glottal", "fricative", "voiceless"), "M": ("bilabial", "nasal", "voiced"),
    "N": ("alveolar", "nasal", "voiced"), "NG": ("velar", "nasal", "voiced"),
    "L": ("alveolar", "lateral_approximant", "voiced"), "R": ("postalveolar", "approximant", "voiced"),
    "W": ("labial_velar", "approximant", "voiced"), "Y": ("palatal", "approximant", "voiced"),
}


@lru_cache(maxsize=1)
def _cmudict_entries() -> Mapping[str, Sequence[Sequence[str]]]:
    try:
        import cmudict
    except ImportError:
        return {}
    return cmudict.dict()


def _arpabet_for_name(name: str, dictionary: Mapping[str, Sequence[Sequence[str]]] | None = None) -> list[str] | None:
    entries = dictionary if dictionary is not None else _cmudict_entries()
    words = re.findall(r"[A-Za-z]+", name.lower())
    if not words or not entries:
        return None
    phones: list[str] = []
    for word in words:
        pronunciations = entries.get(word)
        if not pronunciations:
            return None
        phones.extend(pronunciations[0])
    return phones


def _edit_alignment(left: Sequence[str], right: Sequence[str]):
    rows, cols = len(left) + 1, len(right) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0], back[i][0] = i, "delete"
    for j in range(1, cols):
        dp[0][j], back[0][j] = j, "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = dp[i - 1][j - 1] + (left[i - 1] != right[j - 1])
            choices = ((substitution, "match" if left[i - 1] == right[j - 1] else "substitute"),
                       (dp[i - 1][j] + 1, "delete"), (dp[i][j - 1] + 1, "insert"))
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])
    aligned = []
    i, j = len(left), len(right)
    while i or j:
        action = back[i][j]
        if action in ("match", "substitute"):
            aligned.append((left[i - 1], right[j - 1]))
            i -= 1
            j -= 1
        elif action == "delete":
            aligned.append((left[i - 1], None))
            i -= 1
        else:
            aligned.append((None, right[j - 1]))
            j -= 1
    return dp[-1][-1], list(reversed(aligned))


def _base_phone(phone: str) -> str:
    return re.sub(r"[0-2]$", "", phone.upper())


def _phone_class(phone: str):
    base = _base_phone(phone)
    if base in _CONSONANT_CLASS:
        return _CONSONANT_CLASS[base]
    if base in _VOWEL_CLASS:
        return (_VOWEL_CLASS[base], "vowel", "voiced")
    return None


def _articulatory_distances(aligned):
    mismatches = {"place": [], "manner": [], "voicing": []}
    applicable = {key: 0 for key in mismatches}
    for phone_a, phone_b in aligned:
        if not phone_a or not phone_b:
            continue
        class_a, class_b = _phone_class(phone_a), _phone_class(phone_b)
        if not class_a or not class_b:
            continue
        for idx, key in enumerate(("place", "manner", "voicing")):
            applicable[key] += 1
            mismatches[key].append(float(class_a[idx] != class_b[idx]))
    return {
        f"articulatory_{key}_mismatch": sum(values) / applicable[key] if applicable[key] else None
        for key, values in mismatches.items()
    }


def _lexical_features(name_a: str, name_b: str) -> dict[str, float]:
    a, b = normalize_name(name_a), normalize_name(name_b)
    distance, _ = _edit_alignment(list(a), list(b))
    width = max(len(a), len(b), 1)
    prefix = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        prefix += 1
    return {
        "jaro_winkler": jaro_winkler(a, b),
        "character_edit_distance": float(distance),
        "normalized_character_edit_distance": distance / width,
        "character_similarity": 1.0 - distance / width,
        "name_length_a": float(len(a)),
        "name_length_b": float(len(b)),
        "absolute_length_difference": float(abs(len(a) - len(b))),
        "common_prefix_length": float(prefix),
        "common_prefix_fraction": prefix / width,
    }


def extract_pair_features(
    name_a: str,
    name_b: str,
    *,
    context_template: str = "",
    pronunciation_dictionary: Mapping[str, Sequence[Sequence[str]]] | None = None,
    g2p_provider: Callable[[str], Sequence[str]] | None = None,
    contextual_scorer: "MaskedLMContextScorer | Any | None" = None,
) -> dict[str, Any]:
    """Create one fixed-schema feature row for a medication-name pair.

    ``context_template`` must be fictional text with a ``{drug}`` slot to score
    context. Use no reference transcript, ASR output, or test-derived label here.
    """
    if not name_a.strip() or not name_b.strip():
        raise ValueError("Both medication names must be non-empty.")
    features: dict[str, Any] = {
        "name_a": name_a,
        "name_b": name_b,
        **_lexical_features(name_a, name_b),
        "soundex_a": soundex(name_a),
        "soundex_b": soundex(name_b),
        "soundex_similarity_proxy": phonetic_similarity(name_a, name_b),
    }

    phones_a = _arpabet_for_name(name_a, pronunciation_dictionary)
    phones_b = _arpabet_for_name(name_b, pronunciation_dictionary)
    pronunciation_method = "cmudict" if phones_a and phones_b else "missing"
    if (not phones_a or not phones_b) and g2p_provider is not None:
        try:
            phones_a = list(g2p_provider(name_a))
            phones_b = list(g2p_provider(name_b))
            pronunciation_method = "g2p_provider"
        except Exception:
            phones_a = phones_b = None
            pronunciation_method = "missing"

    if phones_a and phones_b:
        phoneme_distance, alignment = _edit_alignment(phones_a, phones_b)
        phone_width = max(len(phones_a), len(phones_b), 1)
        stress_a = [re.search(r"[0-2]$", phone).group(0) for phone in phones_a
                    if _base_phone(phone) in _VOWEL_CLASS and re.search(r"[0-2]$", phone)]
        stress_b = [re.search(r"[0-2]$", phone).group(0) for phone in phones_b
                    if _base_phone(phone) in _VOWEL_CLASS and re.search(r"[0-2]$", phone)]
        syllables_a = sum(_base_phone(phone) in _VOWEL_CLASS for phone in phones_a)
        syllables_b = sum(_base_phone(phone) in _VOWEL_CLASS for phone in phones_b)
        stress_dist, _ = _edit_alignment(stress_a, stress_b)
        features.update({
            "pronunciation_method": pronunciation_method,
            "phoneme_edit_distance": float(phoneme_distance),
            "phoneme_similarity": 1.0 - phoneme_distance / phone_width,
            "syllables_a": float(syllables_a),
            "syllables_b": float(syllables_b),
            "syllable_difference": float(abs(syllables_a - syllables_b)),
            "stress_pattern_edit_distance": float(stress_dist),
            "stress_pattern_similarity": 1.0 - stress_dist / max(len(stress_a), len(stress_b), 1),
            **_articulatory_distances(alignment),
        })
    else:
        features.update({
            "pronunciation_method": pronunciation_method,
            "phoneme_edit_distance": None,
            "phoneme_similarity": None,
            "syllables_a": None,
            "syllables_b": None,
            "syllable_difference": None,
            "stress_pattern_edit_distance": None,
            "stress_pattern_similarity": None,
            "articulatory_place_mismatch": None,
            "articulatory_manner_mismatch": None,
            "articulatory_voicing_mismatch": None,
        })

    context_result = None
    if context_template and contextual_scorer is not None:
        if "{drug}" not in context_template:
            raise ValueError("context_template must contain the literal '{drug}' slot")
        context_result = contextual_scorer.score_pair(context_template, name_a, name_b)
    if context_result:
        features.update(context_result)
    else:
        features.update({
            "context_available": 0,
            "context_pseudo_nll_a": None,
            "context_pseudo_nll_b": None,
            "context_pseudo_perplexity_a": None,
            "context_pseudo_perplexity_b": None,
            "context_candidate_embedding_cosine": None,
        })
    return features


class MaskedLMContextScorer:
    """Optional masked-LM pseudo-likelihood and contextual similarity scorer.

    Scoring masks each candidate wordpiece in turn and evaluates its original
    token probability. This is a pseudo-likelihood feature, not a calibrated
    probability or a clinical judgment. Long templates are truncated to the
    selected model's maximum length; ensure candidate pieces remain in range.
    """

    def __init__(self, model_name: str, *, device: str | None = None, revision: str | None = None):
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install `torch` and `transformers` to use a contextual language model.") from exc
        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model.to(device).eval()
        self.model.config.output_hidden_states = True

    def _score(self, context_template: str, candidate: str) -> tuple[float, Any]:
        torch = self.torch
        rendered = context_template.replace("{drug}", candidate, 1)
        char_start = len(context_template.split("{drug}", 1)[0])
        char_end = char_start + len(candidate)
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=getattr(self.model.config, "max_position_embeddings", 512),
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        positions = [
            index for index, (start, end) in enumerate(offsets)
            if end > start and start < char_end and end > char_start
        ]
        if not positions:
            raise ValueError(f"Candidate {candidate!r} was truncated or tokenized to no wordpieces.")
        original_ids = input_ids[0, positions].clone()
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError("The selected tokenizer has no mask token; choose a masked-language model.")
        losses = []
        for position, token_id in zip(positions, original_ids):
            masked = input_ids.clone()
            masked[0, position] = mask_id
            with torch.inference_mode():
                logits = self.model(input_ids=masked, attention_mask=attention_mask).logits[0, position]
                losses.append(-torch.log_softmax(logits, dim=-1)[token_id])
        pseudo_nll = float(torch.stack(losses).mean().detach().cpu())
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            vector = output.hidden_states[-1][0, positions].mean(dim=0).detach().float().cpu()
        return pseudo_nll, vector

    def score_pair(self, context_template: str, name_a: str, name_b: str) -> dict[str, Any]:
        torch = self.torch
        nll_a, vector_a = self._score(context_template, name_a)
        nll_b, vector_b = self._score(context_template, name_b)
        denominator = torch.linalg.vector_norm(vector_a) * torch.linalg.vector_norm(vector_b)
        cosine = float(torch.dot(vector_a, vector_b) / denominator) if denominator > 0 else 0.0
        return {
            "context_available": 1,
            "context_model_id": self.model_name,
            "context_model_revision": self.revision or "un-pinned",
            "context_pseudo_nll_a": nll_a,
            "context_pseudo_nll_b": nll_b,
            "context_pseudo_perplexity_a": math.exp(min(nll_a, 20.0)),
            "context_pseudo_perplexity_b": math.exp(min(nll_b, 20.0)),
            "context_candidate_embedding_cosine": cosine,
            "context_relative_score_a_minus_b": nll_b - nll_a,
        }
