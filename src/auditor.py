"""Audio-aware review prioritization; this module never auto-corrects ASR text."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .audit_utils import normalize_text
from .baselines import (disagreement_score, lexical_phonetic_anomaly, nbest_uncertainty,
                        token_confidence_score)
from .vocabulary import jaro_winkler, normalize_name, phonetic_similarity

AcousticScorer = Callable[[Any, int, str], float]
_MODEL_FEATURES = (
    "acoustic_risk", "confidence_risk", "nbest_uncertainty", "recognizer_disagreement",
    "lexical_phonetic_context_risk", "nbest_entropy", "token_confidence_risk",
    "acoustic_support_gap", "character_disagreement", "semantic_disagreement",
)


class DevelopmentCalibratedRiskModel:
    """Whitelisted risk model with calibration fitted on development data only."""

    def __init__(self, vectorizer: Any, estimator: Any, calibrator: Any) -> None:
        self.vectorizer, self.estimator, self.calibrator = vectorizer, estimator, calibrator

    def predict_risk(self, feature_rows: Sequence[Mapping[str, Any]]) -> list[float]:
        """Return calibrated risk probabilities; gold/reference keys are ignored."""
        import numpy as np
        matrix = self.vectorizer.transform([_sanitize_auditor_features(row) for row in feature_rows])
        raw = self.estimator.predict_proba(matrix)[:, 1]
        logits = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / np.clip(1 - raw, 1e-7, 1))
        return self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1].tolist()


def _sanitize_auditor_features(row: Mapping[str, Any]) -> dict[str, float]:
    """Accept only predeclared inference-time features, excluding all targets/IDs."""
    output: dict[str, float] = {}
    for key in _MODEL_FEATURES:
        if key not in row or row[key] in (None, ""):
            continue
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite auditor feature {key!r}")
        output[key] = value
    if not output:
        raise ValueError("No approved auditor inference features supplied")
    return output


def fit_development_risk_model(
    training_features: Sequence[Mapping[str, Any]], training_labels: Sequence[int],
    development_features: Sequence[Mapping[str, Any]], development_labels: Sequence[int], *,
    estimator: str = "logistic", training_partition: str = "train", calibration_partition: str = "dev",
) -> DevelopmentCalibratedRiskModel:
    """Fit the risk estimator on train and a Platt calibrator on dev only.

    Test/external partition labels are intentionally not accepted by this API.
    ``training_partition`` and ``calibration_partition`` are explicit audit
    tags so accidental held-out fitting fails closed.
    """
    if training_partition.casefold() not in {"train", "training", "development_train"}:
        raise ValueError("Risk model fitting is restricted to development/training partitions")
    if calibration_partition.casefold() not in {"dev", "development", "validation"}:
        raise ValueError("Calibration is restricted to development/validation partitions")
    if len(training_features) != len(training_labels) or len(development_features) != len(development_labels):
        raise ValueError("Feature and label lengths must match in each development split")
    if len(set(map(int, training_labels))) != 2 or len(set(map(int, development_labels))) != 2:
        raise ValueError("Training and calibration splits each require both classes")
    try:
        import numpy as np
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError("Calibrated auditor fitting requires scikit-learn and numpy") from exc
    train = [_sanitize_auditor_features(row) for row in training_features]
    development = [_sanitize_auditor_features(row) for row in development_features]
    vectorizer = DictVectorizer(sparse=False)
    train_matrix = vectorizer.fit_transform(train)
    dev_matrix = vectorizer.transform(development)
    if estimator == "logistic":
        fitted = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    elif estimator == "hist_gradient_boosting":
        fitted = HistGradientBoostingClassifier(max_iter=100, l2_regularization=1.0, random_state=0)
    else:
        raise ValueError("estimator must be logistic or hist_gradient_boosting")
    fitted.fit(train_matrix, list(map(int, training_labels)))
    dev_prob = np.clip(fitted.predict_proba(dev_matrix)[:, 1], 1e-7, 1 - 1e-7)
    dev_logits = np.log(dev_prob / (1 - dev_prob)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=0)
    calibrator.fit(dev_logits, list(map(int, development_labels)))
    return DevelopmentCalibratedRiskModel(vectorizer, fitted, calibrator)


def load_auditor_config(path: str | Path) -> dict[str, Any]:
    """Load optional auditor weights/config from YAML."""
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Auditor YAML must contain a mapping")
    return value


def find_medication_spans(transcript: str, vocabulary: Sequence[str]) -> list[dict[str, Any]]:
    """Find longest exact, token-boundary medication mentions in ASR text."""
    tokens = normalize_text(transcript).split()
    normalized_tokens = tokens
    terms = sorted(((name, normalize_text(name).split()) for name in vocabulary),
                   key=lambda item: len(item[1]), reverse=True)
    found: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for start in range(len(tokens)):
        if start in occupied:
            continue
        for name, term_tokens in terms:
            end = start + len(term_tokens)
            if term_tokens and end <= len(tokens) and normalized_tokens[start:end] == term_tokens:
                found.append({"start_token": start, "end_token": end, "surface": " ".join(tokens[start:end]),
                              "vocabulary_name": name})
                occupied.update(range(start, end))
                break
    return found


def retrieve_plausible_alternatives(name: str, vocabulary: Sequence[str], *, top_k: int = 5) -> list[dict[str, Any]]:
    """Rank other authorized names by orthographic/phonetic similarity."""
    ranked = []
    for candidate in vocabulary:
        if normalize_name(candidate) == normalize_name(name):
            continue
        lexical = jaro_winkler(normalize_name(name), normalize_name(candidate))
        phonetic = phonetic_similarity(name, candidate)
        ranked.append({"name": candidate, "lexical_similarity": lexical,
                       "phonetic_similarity": phonetic, "retrieval_score": max(lexical, phonetic)})
    return sorted(ranked, key=lambda row: row["retrieval_score"], reverse=True)[:top_k]


def ctc_log_likelihood(frame_log_probs: Any, token_ids: Sequence[int], *, blank_id: int) -> float:
    """Compute CTC sequence log-likelihood for candidate tokens and audio frames.

    ``frame_log_probs`` is a T×V matrix produced by an acoustic model from the
    current waveform. The dynamic program sums over valid CTC alignments,
    including blanks and repeated-token constraints. This is candidate support,
    not a transcript correction decision.
    """
    import numpy as np

    matrix = np.asarray(frame_log_probs, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("frame_log_probs must be a non-empty T×V matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("frame_log_probs must contain only finite values")
    if not 0 <= int(blank_id) < matrix.shape[1] or any(not 0 <= int(token) < matrix.shape[1] for token in token_ids):
        raise ValueError("blank/token ID is outside the model vocabulary")

    extended = [int(blank_id)]
    for token in token_ids:
        extended.extend((int(token), int(blank_id)))

    def logadd(*values: float) -> float:
        maximum = max(values)
        if maximum == -math.inf:
            return maximum
        return maximum + math.log(sum(math.exp(value - maximum) for value in values))

    previous = [-math.inf] * len(extended)
    previous[0] = float(matrix[0, blank_id])
    if len(extended) > 1:
        previous[1] = float(matrix[0, extended[1]])
    for frame in range(1, matrix.shape[0]):
        current = [-math.inf] * len(extended)
        for state, token in enumerate(extended):
            transitions = [previous[state]]
            if state > 0:
                transitions.append(previous[state - 1])
            if state > 1 and token != blank_id and token != extended[state - 2]:
                transitions.append(previous[state - 2])
            current[state] = logadd(*transitions) + float(matrix[frame, token])
        previous = current
    if len(extended) == 1:
        return previous[0]
    return logadd(previous[-1], previous[-2])


def make_ctc_acoustic_scorer(frame_log_probs: Any, encode_candidate: Callable[[str], Sequence[int]], *, blank_id: int) -> AcousticScorer:
    """Adapt current-utterance CTC frame scores into the auditor callback API."""
    def score(_waveform: Any, _sample_rate_hz: int, candidate: str) -> float:
        return ctc_log_likelihood(frame_log_probs, encode_candidate(candidate), blank_id=blank_id)
    return score


def _softmax_support(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    maximum = max(scores)
    weights = [math.exp(max(-60.0, min(60.0, score - maximum))) for score in scores]
    total = sum(weights)
    return [value / total for value in weights]


def audit_segment(
    waveform: Any,
    sample_rate_hz: int,
    transcript: str,
    vocabulary: Sequence[str],
    *,
    nbest: Sequence[Mapping[str, Any]] = (),
    token_confidences: Sequence[float | Mapping[str, Any]] = (),
    alternate_transcript: str | None = None,
    semantic_distance_fn: Callable[[str, str], float] | None = None,
    acoustic_scorer: AcousticScorer | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return spans, evidence features, and a review-priority score in [0,1].

    ``acoustic_scorer`` optionally returns a candidate's acoustic log-likelihood
    or comparable support score. The function produces no replacement text;
    ``recommended_action`` is always human review when the score is elevated.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if not vocabulary:
        raise ValueError("An authorized medication vocabulary subset is required")
    settings = dict(config or {})
    weights = {"acoustic": 0.40, "confidence": 0.20, "uncertainty": 0.10,
               "disagreement": 0.15, "context": 0.15}
    weights.update(settings.get("weights", {}))
    if set(weights) != {"acoustic", "confidence", "uncertainty", "disagreement", "context"}:
        raise ValueError("weights may override only acoustic, confidence, uncertainty, disagreement, and context")
    if any(float(weight) < 0 for weight in weights.values()) or sum(map(float, weights.values())) <= 0:
        raise ValueError("Auditor weights must be non-negative with a positive sum")

    spans = find_medication_spans(transcript, vocabulary)
    anomaly = lexical_phonetic_anomaly(transcript, vocabulary)
    confidence_risk = token_confidence_score(token_confidences) if token_confidences else 0.0
    uncertainty = nbest_uncertainty(nbest) if nbest else {"entropy": 0.0, "top2_margin": 1.0, "risk_score": 0.0}
    word_disagreement = disagreement_score(transcript, alternate_transcript, unit="word") if alternate_transcript is not None else 0.0
    character_disagreement = disagreement_score(transcript, alternate_transcript, unit="char") if alternate_transcript is not None else 0.0
    semantic_disagreement = float(semantic_distance_fn(transcript, alternate_transcript)) if (
        alternate_transcript is not None and semantic_distance_fn is not None) else 0.0
    if not 0.0 <= semantic_disagreement <= 1.0 or not math.isfinite(semantic_disagreement):
        raise ValueError("semantic_distance_fn must return a finite distance in [0, 1]")
    disagreement = max(word_disagreement, character_disagreement, semantic_disagreement)
    span_results = []
    acoustic_risks = []
    scoring_spans = list(spans)
    if not scoring_spans:
        # For misspelled/fragmented ASR tokens, form candidates from general
        # vocabulary similarity rather than a gold reference list.
        scoring_spans = [
            {"start_token": item["token_index"], "end_token": item["token_index"] + 1,
             "surface": item["token"], "vocabulary_name": item["closest_vocabulary_name"],
             "retrieved_from_asr_token": True}
            for item in anomaly["candidate_spans"] if float(item["risk_score"]) >= float(settings.get("retrieval_min_similarity", 0.35))
        ]
    for span in scoring_spans:
        alternatives = retrieve_plausible_alternatives(span["vocabulary_name"], vocabulary,
                                                         top_k=int(settings.get("candidate_top_k", 5)))
        candidate_names = [span["vocabulary_name"], *(item["name"] for item in alternatives)]
        candidate_scores = [float(acoustic_scorer(waveform, sample_rate_hz, name))
                            for name in candidate_names] if acoustic_scorer else []
        if any(not math.isfinite(score) for score in candidate_scores):
            raise ValueError("acoustic_scorer must return finite candidate scores")
        supports = _softmax_support(candidate_scores)
        acoustic_risk = 1.0 - supports[0] if supports else 0.0
        acoustic_risks.append(acoustic_risk)
        span_results.append({**span, "alternatives": alternatives,
                             "acoustic_candidate_support": dict(zip(candidate_names, supports)),
                             "acoustic_risk": acoustic_risk})
    acoustic_risk = max(acoustic_risks, default=0.0)
    # A valid exact medication token is not itself anomalous. For a recognized
    # medication span, context risk is its similarity to a competing name;
    # for an unrecognized token, retain the transcript-to-vocabulary anomaly.
    context_risk = max((
        float(anomaly["risk_score"]) if span.get("retrieved_from_asr_token") else
        max((item["retrieval_score"] for item in span["alternatives"]), default=0.0)
        for span in span_results
    ), default=0.0)
    total_weight = sum(float(value) for value in weights.values())
    priority = sum(float(weights[key]) * value for key, value in (
        ("acoustic", acoustic_risk), ("confidence", confidence_risk),
        ("uncertainty", float(uncertainty["risk_score"])),
        ("disagreement", disagreement), ("context", context_risk),
    )) / total_weight
    threshold = float(settings.get("review_threshold", 0.5))
    return {
        "transcript": transcript,
        "medication_spans": span_results,
        "features": {"acoustic_risk": acoustic_risk, "confidence_risk": confidence_risk,
                     "token_confidence_risk": confidence_risk,
                     "nbest_uncertainty": float(uncertainty["risk_score"]),
                     "nbest_entropy": float(uncertainty["entropy"]),
                     "recognizer_disagreement": disagreement,
                     "character_disagreement": character_disagreement,
                     "semantic_disagreement": semantic_disagreement,
                     "acoustic_support_gap": acoustic_risk,
                     "lexical_phonetic_context_risk": context_risk},
        "review_priority_score": min(1.0, max(0.0, priority)),
        "review_recommended": priority >= threshold,
        "recommended_action": "human_review" if priority >= threshold else "no_alert",
        "automatic_correction": None,
    }

