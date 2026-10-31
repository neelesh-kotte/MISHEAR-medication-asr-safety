"""Strong transcript-only error-review baselines (no intended-drug labels)."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .vocabulary import jaro_winkler, normalize_name, phonetic_similarity, soundex


def token_confidence_score(token_confidences: Sequence[float | Mapping[str, Any]]) -> float:
    """Return one minus mean token probability, accepting log probabilities."""
    values = []
    for item in token_confidences:
        if isinstance(item, Mapping):
            if "log_probability" in item or "logprob" in item:
                log_probability = float(item.get("log_probability", item.get("logprob")))
                value = math.exp(max(-745.0, min(0.0, log_probability)))
            else:
                value = float(item.get("probability", item.get("confidence", item.get("score", 0.0))))
        else:
            value = float(item)
        values.append(value)
    if not values:
        raise ValueError("At least one token confidence is required")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("Token probabilities must be finite and lie in [0, 1]")
    return 1.0 - sum(values) / len(values)


def nbest_uncertainty(alternatives: Sequence[Mapping[str, Any]], *, score_field: str = "score") -> dict[str, float]:
    """Compute normalized entropy and top-two probability margin from N-best scores.

    Scores are interpreted as log scores/log probabilities and softmaxed for
    numerical stability. The returned entropy lies in [0, 1] after normalization.
    """
    if not alternatives:
        raise ValueError("N-best alternatives cannot be empty")
    scores = []
    for item in alternatives:
        if score_field in item:
            scores.append(float(item[score_field]))
        elif "log_probability" in item or "logprob" in item:
            scores.append(float(item.get("log_probability", item.get("logprob"))))
        elif "probability" in item:
            probability = float(item["probability"])
            if not 0 <= probability <= 1:
                raise ValueError("N-best probabilities must lie in [0, 1]")
            scores.append(math.log(max(probability, 1e-300)))
        else:
            raise ValueError(f"N-best row lacks {score_field!r}, log_probability, or probability")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("N-best scores must be finite")
    maximum = max(scores)
    exp_scores = [math.exp(score - maximum) for score in scores]
    total = sum(exp_scores)
    probs = sorted((value / total for value in exp_scores), reverse=True)
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    normalized = entropy / math.log(len(probs)) if len(probs) > 1 else 0.0
    return {"entropy": normalized, "top2_margin": probs[0] - (probs[1] if len(probs) > 1 else 0.0),
            "risk_score": normalized}


def normalized_transcript_distance(first: str, second: str, *, unit: str = "word") -> float:
    """Normalized Levenshtein distance between independent ASR transcripts."""
    if unit == "char":
        left, right = normalize_name(first), normalize_name(second)
    elif unit == "word":
        left = re.findall(r"\w+", first.casefold())
        right = re.findall(r"\w+", second.casefold())
    else:
        raise ValueError("unit must be 'word' or 'char'")
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def disagreement_score(first: str, second: str, *, unit: str = "word") -> float:
    """Alias for the normalized disagreement risk between recognizer outputs."""
    return normalized_transcript_distance(first, second, unit=unit)


def extract_baseline_features(
    transcript: str, vocabulary: Sequence[str], *,
    token_confidences: Sequence[float | Mapping[str, Any]] = (),
    nbest: Sequence[Mapping[str, Any]] = (), alternate_transcript: str | None = None,
) -> dict[str, float]:
    """Create the allowed transcript-side combined-baseline feature row."""
    anomaly = lexical_phonetic_anomaly(transcript, vocabulary)
    spans = anomaly["candidate_spans"]
    lexical = max((float(row["lexical_similarity"]) for row in spans), default=0.0)
    phonetic = max((float(row["phonetic_similarity"]) for row in spans), default=0.0)
    uncertainty = nbest_uncertainty(nbest) if nbest else {"entropy": 0.0, "top2_margin": 1.0}
    return {
        "token_confidence_risk": token_confidence_score(token_confidences) if token_confidences else 0.0,
        "nbest_entropy": float(uncertainty["entropy"]),
        "nbest_margin_risk": 1.0 - float(uncertainty["top2_margin"]),
        "recognizer_disagreement": disagreement_score(transcript, alternate_transcript)
        if alternate_transcript is not None else 0.0,
        "lexical_anomaly_score": lexical,
        "phonetic_anomaly_score": phonetic,
    }


def lexical_phonetic_anomaly(
    transcript: str, vocabulary: Sequence[str], *, max_token_chars: int = 32,
) -> dict[str, Any]:
    """Score transcript tokens by similarity to authorized vocabulary entries.

    No true medication identity is required. A high score indicates that a
    token resembles a medication name enough to deserve review; it is not a
    claim that the token is wrong or a proposed correction.
    """
    if not vocabulary:
        raise ValueError("Medication vocabulary must not be empty")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", transcript)
    candidates: list[dict[str, Any]] = []
    normalized_vocab = [(name, normalize_name(name)) for name in vocabulary]
    for index, token in enumerate(tokens):
        normalized = normalize_name(token)
        if not normalized or len(normalized) > max_token_chars:
            continue
        ranked = []
        for name, normalized_name in normalized_vocab:
            lexical = jaro_winkler(normalized, normalized_name)
            phonetic = phonetic_similarity(token, name)
            score = max(lexical, phonetic)
            ranked.append((score, name, lexical, phonetic))
        score, name, lexical, phonetic = max(ranked, key=lambda item: item[0])
        candidates.append({"token": token, "token_index": index, "closest_vocabulary_name": name,
                           "lexical_similarity": lexical, "phonetic_similarity": phonetic,
                           "soundex": soundex(token), "risk_score": score})
    return {"risk_score": max((item["risk_score"] for item in candidates), default=0.0),
            "candidate_spans": candidates}


def fit_combined_baseline(features: Sequence[Mapping[str, Any]], labels: Sequence[int]):
    """Fit a small regularized classifier over baseline scores only.

    Callers should supply only ASR-side confidence, disagreement, uncertainty,
    and lexical/phonetic anomaly features. Reference identity, correctness,
    reference timestamps, and recording-specific gold candidates are forbidden.
    """
    if len(features) != len(labels) or len(features) == 0:
        raise ValueError("Features and labels must have equal non-zero lengths")
    if len(set(map(int, labels))) != 2:
        raise ValueError("Combined baseline training requires both outcome classes")
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError("The combined baseline requires scikit-learn") from exc
    sanitized = [_blinded_score_features(row) for row in features]
    if not any(sanitized):
        raise ValueError("No approved ASR-side baseline score features were supplied")
    model = make_pipeline(DictVectorizer(sparse=False),
                          LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0))
    model.fit(sanitized, list(map(int, labels)))
    return model


def score_combined_baseline(model: Any, features: Sequence[Mapping[str, Any]]) -> list[float]:
    """Score feature rows with a model trained by :func:`fit_combined_baseline`."""
    sanitized = [_blinded_score_features(row) for row in features]
    return model.predict_proba(sanitized)[:, 1].tolist()


def _blinded_score_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Allow only named ASR-side scores and reject gold/reference-derived keys."""
    allowed = {
        "asr_confidence", "token_confidence_risk", "span_confidence_risk",
        "nbest_entropy", "alternative_hypothesis_entropy", "nbest_margin_risk",
        "recognizer_disagreement", "word_disagreement", "character_disagreement",
        "lexical_anomaly_score", "phonetic_anomaly_score", "soundex_anomaly_score",
        "acoustic_support_gap", "acoustic_candidate_margin", "combined_risk_score",
    }
    forbidden_markers = ("reference", "gold", "correct", "label", "truth", "timestamp", "intended_drug")
    return {str(key): value for key, value in row.items()
            if str(key).casefold() in allowed
            and not any(marker in str(key).casefold() for marker in forbidden_markers)}

