"""Configured end-to-end synthetic MISHEAR validation pipeline.

Run with ``python -m src.run_validation_pipeline --config config/pipeline_config.yaml``.
The ``all`` stage synthesizes the configured candidate pairs, applies fixed DSP
conditions, runs the configured local ASR backend, aligns errors, creates a
grouped holdout and evaluates prespecified logistic baselines. It may be costly
and requires all local system/model dependencies to be installed first.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audio_generation import build_synthesis_jobs, generate_jobs
from .asr_evaluation import OpenAIWhisperRunner, WhisperCppRunner, run_batch
from .asr_evaluation import Wav2Vec2CTCRunner
from .calibration import calibration_curve_data, fit_calibrator
from .context_features import extract_pair_features
from .dsp_augmentation import AugmentationConfig, augment_wav
from .pair_risk_model import evaluate_model, fit_pair_risk_model, similarity_baseline
from .selective_prediction import select_threshold, selection_as_dict
from .statistical_analysis import binary_metrics, grouped_bootstrap, mcnemar_test, paired_grouped_bootstrap_delta
from .test_data_splits import (build_ingredient_component_map, create_ingredient_disjoint_splits,
                               create_unseen_acoustic_condition_split, write_split_manifest)
from .test_error_alignment import align_evaluation_records, pair_audio_metadata_with_asr, save_alignment_outputs
from .vocabulary import generate_candidate_pairs, load_vocabulary_csv


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:60] or "none"


def generate_pairs(config: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    vocab = load_vocabulary_csv(config["vocabulary_csv"])
    pairs = generate_candidate_pairs(vocab,
        min_jaro_winkler=float(config.get("min_jaro_winkler", 0.72)),
        min_phonetic_similarity=float(config.get("min_phonetic_similarity", 0.66)))
    limit = config.get("max_pairs")
    if limit:
        pairs = pairs[:int(limit)]
    rows = []
    for i, pair in enumerate(pairs):
        row = asdict(pair)
        row["pair_id"] = f"pair_{i:05d}"
        rows.append(row)
    _write_csv(out / "candidate_pairs.csv", rows)
    return rows


def synthesize(config: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]], out: Path) -> list[dict[str, Any]]:
    vocab = _read_csv(config["vocabulary_csv"])
    jobs = build_synthesis_jobs(vocab, pairs, config["templates"], config["voices"], config["rates_wpm"])
    return generate_jobs(jobs, out / "synthesis", int(config.get("sample_rate_hz", 16000)))


def augment(config: Mapping[str, Any], clean: Sequence[Mapping[str, Any]], out: Path) -> list[dict[str, Any]]:
    audio_root = out / "augmented"
    conditions = config["dsp_conditions"]
    records = []
    for index, source in enumerate(clean):
        clean_path = Path(str(source["audio_path"]))
        for condition in conditions:
            condition_id = str(condition.get("condition_id", f"condition_{conditions.index(condition):02d}"))
            output = audio_root / condition_id / f"{clean_path.stem}__{_safe(condition_id)}.wav"
            noise_kind = condition.get("noise_kind")
            babble_paths: list[str] = []
            if noise_kind == "babble":
                # Other distinct synthetic utterances only; deterministic circular source selection.
                candidates = [str(row["audio_path"]) for j, row in enumerate(clean)
                              if j != index and row.get("utterance_id") != source.get("utterance_id")]
                count = min(len(candidates), int(condition.get("babble_speakers", 3)))
                babble_paths = candidates[index % len(candidates):] + candidates[:index % len(candidates)] if candidates else []
                babble_paths = babble_paths[:count]
                if len(babble_paths) < 2:
                    raise ValueError("Babble condition needs at least two other synthesized utterances")
            dsp_config = AugmentationConfig(
                noise_kind=noise_kind,
                target_snr_db=float(condition["target_snr_db"]) if condition.get("target_snr_db") is not None else None,
                rt60_target_s=float(condition["rt60_target_s"]) if condition.get("rt60_target_s") is not None else None,
                clipping_threshold=float(condition["clipping_threshold"]) if condition.get("clipping_threshold") is not None else None,
                seed=int(config.get("seed", 20261030)) + index * 1009 + conditions.index(condition),
            )
            meta = augment_wav(clean_path, output, dsp_config, babble_paths=babble_paths)
            records.append({**source, **meta, "recording_id": output.stem,
                            "condition_id": condition_id,
                            "condition_family": condition.get("condition_family", "clean" if not noise_kind and not dsp_config.rt60_target_s and not dsp_config.clipping_threshold else condition_id)})
    _write_csv(audio_root / "augmentation_metadata.csv", records)
    return records


def run_asr(config: Mapping[str, Any], audio: Sequence[Mapping[str, Any]], out: Path,
            asr: Mapping[str, Any] | None = None, *, run_name: str = "primary") -> list[dict[str, Any]]:
    asr = asr or config["asr"]
    output_dir = out / f"asr_raw_{run_name}"
    if asr["backend"] == "whisper_cpp":
        runner = WhisperCppRunner(asr["model_path"], executable=asr.get("executable", "whisper-cli"),
                                  output_dir=output_dir, threads=int(asr.get("threads", 4)),
                                  beam_size=int(asr.get("beam_size", 5)), no_gpu=bool(asr.get("no_gpu", False)))
    elif asr["backend"] == "openai_whisper":
        runner = OpenAIWhisperRunner(asr.get("model_name", "base"), device=asr.get("device"))
    elif asr["backend"] == "wav2vec2_ctc":
        runner = Wav2Vec2CTCRunner(asr.get("model_id", "facebook/wav2vec2-base-960h"),
                                   device=asr.get("device"),
                                   save_frame_confidence=bool(asr.get("save_frame_confidence", True)),
                                   revision=asr.get("revision"))
    else:
        raise ValueError("backend must be whisper_cpp or openai_whisper")
    records = run_batch(runner, [row["output_path"] for row in audio], out / f"asr_runs_{run_name}.jsonl",
                        continue_on_error=bool(asr.get("continue_on_error", False)))
    return records


def build_prediction_rows(aligned: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]], *,
                          include_all_outcomes: bool = True) -> list[dict[str, Any]]:
    pair_index = {str(row["pair_id"]): row for row in pair_rows}
    result = []
    for row in aligned:
        pair = pair_index.get(str(row.get("pair_id", "")))
        if not pair:
            continue
        expected = str(row.get("drug_name", ""))
        other = str(pair["name_b"] if expected.casefold() == str(pair["name_a"]).casefold() else pair["name_a"])
        predicted = str(row.get("predicted_drug", "")).casefold()
        is_pair_substitution = row.get("medication_event") == "substitution" and predicted == other.casefold()
        # The positive event is confusion with the other pair member. Correct
        # names, deletions, and substitutions to third names are negatives for
        # this pair-specific target; full event labels remain in alignment data.
        if not include_all_outcomes and row.get("medication_event") != "correct" and not is_pair_substitution:
            continue
        aligned_features = extract_pair_features(str(pair["name_a"]), str(pair["name_b"]))
        # Only measured outcomes/features from pre-inference audio metadata enter X.
        acoustic = {
            "measured_snr_db": row.get("measured_snr_db"),
            "output_snr_db": row.get("output_snr_including_distortion_db"),
            "measured_rt60_s": row.get("measured_rt60_s"),
            "duration_s": row.get("duration_s"),
            "clipped_sample_fraction": row.get("clipped_sample_fraction"),
            "recognizer": row.get("recognizer"), "model_id": row.get("model_id"),
            "model_version": row.get("model_version"),
            "condition_family": row.get("condition_family"), "context_condition": row.get("context_condition"),
            "condition_id": row.get("condition_id"), "acoustic_condition_id": row.get("condition_id"),
            "voice": row.get("voice"),
        }
        target = int(is_pair_substitution)
        result.append({**aligned_features, **acoustic,
                       "recording_id": row.get("recording_id"), "pair_id": pair["pair_id"],
                       "ingredient_a": pair.get("ingredient_a", ""), "ingredient_b": pair.get("ingredient_b", ""),
                       "component_id": row.get("component_id") or pair.get("pair_id"), "label": target,
                       "event_label": row.get("medication_event")})
    return result


def evaluate(config: Mapping[str, Any], features: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    # Exclude graph connectivity for a row only if ingredient IDs are missing; that fails explicitly.
    splits = create_ingredient_disjoint_splits(features, pair_edges=features,
                                               seed=int(config.get("seed", 20261030)))
    write_split_manifest(splits, out / "split_manifest.json")
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    if not train or not validation or not test:
        raise ValueError("Ingredient grouped split produced an empty partition; inspect data/group counts.")
    train_y = [int(row["label"]) for row in train]
    if len(set(train_y)) != 2:
        raise ValueError("Training split needs both positive and negative pair-confusion labels")
    y_val = [int(row["label"]) for row in validation]
    y_test = [int(row["label"]) for row in test]
    model_phonetic = fit_pair_risk_model(train, train_y, feature_groups=("phonetic",))
    model_full = fit_pair_risk_model(train, train_y)
    val_phonetic = model_phonetic.predict_proba(validation)
    val_full = model_full.predict_proba(validation)
    test_phonetic = model_phonetic.predict_proba(test)
    test_full = model_full.predict_proba(test)
    test_similarity = similarity_baseline(test)
    budget = config.get("false_alert_budget_per_100_mentions")
    if budget is None:
        raise ValueError("Set false_alert_budget_per_100_mentions in config before running the frozen test evaluation")
    threshold = select_threshold(y_val, val_full, budget_per_100_mentions=float(budget),
                                 alpha=float(config.get("selective_alpha", 0.05)))
    threshold_phonetic = select_threshold(y_val, val_phonetic, budget_per_100_mentions=float(budget),
                                          alpha=float(config.get("selective_alpha", 0.05)))
    prediction_full = [int(value >= threshold.threshold) for value in test_full]
    prediction_phonetic = [int(value >= threshold_phonetic.threshold) for value in test_phonetic]
    groups = [str(row["component_id"]) for row in test]
    boot = paired_grouped_bootstrap_delta(y_test, test_full.tolist(), test_phonetic.tolist(), groups,
                                          n_bootstrap=int(config.get("bootstrap_replicates", 2000)),
                                          seed=int(config.get("seed", 20261030)))
    report = {
        "protocol": "ingredient_connected_component_holdout",
        "n": {"train": len(train), "validation": len(validation), "test": len(test)},
        "independent_components": {"train": len({str(row.get("component_id")) for row in train}),
                                   "validation": len({str(row.get("component_id")) for row in validation}),
                                   "test": len({str(row.get("component_id")) for row in test})},
        "test_full_model": binary_metrics(y_test, test_full),
        "test_phonetic_only": binary_metrics(y_test, test_phonetic),
        "test_similarity_baseline": binary_metrics(y_test, test_similarity),
        "full_model_coefficients_standardized": model_full.coefficients(),
        "phonetic_model_coefficients_standardized": model_phonetic.coefficients(),
        "paired_group_bootstrap_auprc_delta_full_minus_phonetic": boot,
        "paired_group_bootstrap_auprc_delta_full_minus_similarity": paired_grouped_bootstrap_delta(
            y_test, test_full.tolist(), test_similarity.tolist(), groups,
            n_bootstrap=int(config.get("bootstrap_replicates", 2000)), seed=int(config.get("seed", 20261030))),
        "validation_selected_threshold": selection_as_dict(threshold),
        "test_selective_metrics": binary_metrics(y_test, test_full, threshold.threshold),
        "mcnemar_full_vs_phonetic_at_validation_selected_thresholds": mcnemar_test(
            y_test, prediction_full, prediction_phonetic),
        "warning": "Synthetic benchmark estimates only; no clinical effectiveness inference.",
    }
    # H2 reserves entire prespecified acoustic conditions before fitting.
    h2_config = config.get("h2", {})
    held_ids = h2_config.get("held_out_condition_ids", [])
    held_families = h2_config.get("held_out_families", [])
    if held_ids or held_families:
        h2_splits = create_unseen_acoustic_condition_split(
            features, held_out_condition_ids=held_ids, held_out_families=held_families,
            pair_edges=features, validation_fraction=float(h2_config.get("validation_fraction", 0.2)),
            seed=int(config.get("seed", 20261030)))
        component_map = build_ingredient_component_map(features, features)
        for row in h2_splits["test"]:
            ingredient_a, ingredient_b = str(row.get("ingredient_a", "")), str(row.get("ingredient_b", ""))
            if ingredient_a and ingredient_b:
                row["component_id"] = component_map[ingredient_a]
        h2_train, h2_validation, h2_test = (h2_splits[name] for name in ("train", "validation", "test"))
        if not h2_train or not h2_validation or not h2_test:
            raise ValueError("H2 held-out condition split has an empty partition")
        h2_y_train = [int(row["label"]) for row in h2_train]
        h2_y_validation = [int(row["label"]) for row in h2_validation]
        h2_y_test = [int(row["label"]) for row in h2_test]
        if len(set(h2_y_train)) != 2 or len(set(h2_y_validation)) != 2:
            raise ValueError("H2 training and validation must each contain both classes")
        h2_model = fit_pair_risk_model(h2_train, h2_y_train)
        h2_validation_raw = h2_model.predict_proba(h2_validation)
        h2_test_raw = h2_model.predict_proba(h2_test)
        calibration_results = {}
        calibration_rows = []
        for method in ("platt", "isotonic"):
            calibrator = fit_calibrator(h2_y_validation, h2_validation_raw, method)
            calibrated_validation = calibrator.predict(h2_validation_raw)
            calibrated_test = calibrator.predict(h2_test_raw)
            calibration_results[method] = {
                "validation_metrics": binary_metrics(h2_y_validation, calibrated_validation),
                "test_metrics": binary_metrics(h2_y_test, calibrated_test),
                "test_calibration_curve": calibration_curve_data(h2_y_test, calibrated_test,
                    n_bins=int(h2_config.get("calibration_bins", 10))),
            }
            calibration_rows.extend({
                **row, "label": label, "raw_probability": float(raw),
                "platt_probability": float(calibrated_test[i]) if method == "platt" else None,
                "isotonic_probability": float(calibrated_test[i]) if method == "isotonic" else None,
            } for i, (row, label, raw) in enumerate(zip(h2_test, h2_y_test, h2_test_raw)))
        h2_groups = [str(row.get("component_id") or row.get("pair_id")) for row in h2_test]
        report["H2_unseen_acoustic_conditions"] = {
            "held_out_condition_ids": list(held_ids), "held_out_families": list(held_families),
            "n": {"train": len(h2_train), "validation": len(h2_validation), "test": len(h2_test)},
            "independent_components": {"train": len({str(row.get("component_id")) for row in h2_train}),
                                       "validation": len({str(row.get("component_id")) for row in h2_validation}),
                                       "test": len(set(h2_groups))},
            "test_raw_model": binary_metrics(h2_y_test, h2_test_raw),
            "test_prevalence_baseline_auprc": sum(h2_y_test) / len(h2_y_test),
            "calibration_comparison": calibration_results,
            "paired_group_bootstrap_auprc_delta_raw_vs_prevalence": paired_grouped_bootstrap_delta(
                h2_y_test, h2_test_raw.tolist(), [sum(h2_y_test) / len(h2_y_test)] * len(h2_test), h2_groups,
                n_bootstrap=int(config.get("bootstrap_replicates", 2000)),
                seed=int(config.get("seed", 20261030))),
        }
        # Persist a single tidy row per held-out example with all score variants.
        # The loop above accumulates each method; collapse by stable recording ID.
        calibration_by_id = {}
        for row in calibration_rows:
            key = str(row.get("recording_id"))
            dest = calibration_by_id.setdefault(key, {k: v for k, v in row.items()
                                                       if k not in {"platt_probability", "isotonic_probability"}})
            for score_name in ("platt_probability", "isotonic_probability"):
                if row.get(score_name) is not None:
                    dest[score_name] = row[score_name]
        _write_csv(out / "h2_calibration_predictions.csv", list(calibration_by_id.values()))
        write_split_manifest(h2_splits, out / "h2_split_manifest.json")
    else:
        report["H2_unseen_acoustic_conditions"] = {"status": "not_run", "reason": "No H2 held-out conditions configured"}
    (out / "model_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    _write_csv(out / "test_predictions.csv", [dict(row, phonetic_score=float(p), full_model_score=float(f),
                                                   similarity_score=float(s), full_model_alert=alert)
                                               for row, p, f, s, alert in zip(test, test_phonetic, test_full, test_similarity, prediction_full)])
    return report


def evaluate_h4_transfer(config: Mapping[str, Any], primary_features: Sequence[Mapping[str, Any]],
                         secondary_features: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    """Train on primary recognizer development data; evaluate the same recordings with recognizer two."""
    splits = create_ingredient_disjoint_splits(primary_features, pair_edges=primary_features,
                                               seed=int(config.get("seed", 20261030)))
    train = splits["train"]
    test_by_id = {str(row.get("recording_id")): row for row in splits["test"]}
    test_ids = set(test_by_id)
    secondary_by_id = {str(row.get("recording_id")): row for row in secondary_features}
    missing = test_ids - set(secondary_by_id)
    if missing:
        raise ValueError(f"H4 secondary recognizer lacks {len(missing)} ingredient-test recordings")
    transfer_test = [dict(secondary_by_id[key], component_id=test_by_id[key].get("component_id"))
                     for key in sorted(test_ids)]
    if not train or not transfer_test:
        raise ValueError("H4 requires non-empty primary training data and matched secondary-recognizer test records")
    labels_train = [int(row["label"]) for row in train]
    labels_test = [int(row["label"]) for row in transfer_test]
    if len(set(labels_train)) != 2:
        raise ValueError("H4 primary training split must include both classes")
    model = fit_pair_risk_model(train, labels_train)
    scores = model.predict_proba(transfer_test)
    groups = [str(row["component_id"]) for row in transfer_test]
    result = {
        "protocol": "train_primary_recognizer_test_secondary_recognizer_same_ingredient_holdout",
        "secondary_recognizer": (transfer_test[0].get("recognizer") if transfer_test else None),
        "secondary_model_id": (transfer_test[0].get("model_id") if transfer_test else None),
        "secondary_model_version": (transfer_test[0].get("model_version") if transfer_test else None),
        "n_train_primary": len(train), "n_test_secondary": len(transfer_test),
        "independent_test_components": len(set(groups)),
        "test_metrics": binary_metrics(labels_test, scores),
        "grouped_bootstrap_auprc": grouped_bootstrap(
            labels_test, scores.tolist(), groups, n_bootstrap=int(config.get("bootstrap_replicates", 2000)),
            seed=int(config.get("seed", 20261030))),
        "warning": "Recognizer transfer is evaluated on synthetic recordings only.",
    }
    (out / "h4_transfer_report.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    _write_csv(out / "h4_transfer_predictions.csv", [dict(row, transfer_score=float(score))
                                                       for row, score in zip(transfer_test, scores)])
    return result


def run(config: Mapping[str, Any], stage: str = "all") -> None:
    out = Path(config.get("output_dir", "results/phase5")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pair_path = out / "candidate_pairs.csv"
    pairs = _read_csv(pair_path) if stage in {"asr", "model"} and pair_path.exists() else generate_pairs(config, out)
    synthesis_meta_path = out / "synthesis" / "synthetic_audio_metadata.csv"
    clean = _read_csv(synthesis_meta_path) if stage in {"asr", "model"} and synthesis_meta_path.exists() else synthesize(config, pairs, out)
    aug_path = out / "augmented" / "augmentation_metadata.csv"
    audio = _read_csv(aug_path) if stage == "model" and aug_path.exists() else augment(config, clean, out)
    asr_path = out / "asr_runs_primary.jsonl"
    if not asr_path.exists():
        asr_path = out / "asr_runs.jsonl"  # compatibility with Phase 5 run artifacts
    if stage == "model" and asr_path.exists():
        asr_records = [json.loads(line) for line in asr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        asr_records = run_asr(config, audio, out)
    joined = pair_audio_metadata_with_asr(audio, asr_records)
    ingredients = {str(row.get("name", "")).casefold(): str(row.get("ingredient_rxcui") or row.get("ingredient_id") or "")
                   for row in _read_csv(config["vocabulary_csv"])}
    aligned = align_evaluation_records(joined, drug_vocabulary=ingredients)
    save_alignment_outputs(aligned,
                           records_jsonl=out / "alignment" / "aligned_records.jsonl",
                           metrics_csv=out / "alignment" / "stratified_error_metrics.csv")
    if stage == "all" or stage == "model":
        features = build_prediction_rows(aligned, pairs)
        _write_csv(out / "pair_features_and_labels.csv", features)
        report = evaluate(config, features, out)
        secondary_config = config.get("secondary_asr")
        if secondary_config:
            secondary_path = out / "asr_runs_secondary.jsonl"
            if stage == "model" and secondary_path.exists():
                secondary_records = [json.loads(line) for line in secondary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                secondary_records = run_asr(config, audio, out, secondary_config, run_name="secondary")
            secondary_joined = pair_audio_metadata_with_asr(audio, secondary_records)
            secondary_aligned = align_evaluation_records(secondary_joined, drug_vocabulary=ingredients)
            save_alignment_outputs(secondary_aligned,
                                   records_jsonl=out / "alignment_secondary" / "aligned_records.jsonl",
                                   metrics_csv=out / "alignment_secondary" / "stratified_error_metrics.csv")
            primary_h4_features = build_prediction_rows(aligned, pairs, include_all_outcomes=True)
            secondary_features = build_prediction_rows(secondary_aligned, pairs, include_all_outcomes=True)
            report["H4_cross_recognizer_transfer"] = evaluate_h4_transfer(
                config, primary_h4_features, secondary_features, out)
            (out / "model_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
        print(json.dumps(report, indent=2, allow_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--stage", choices=("all", "asr", "model"), default="all",
                        help="all executes the pipeline; asr/model resume from saved upstream artifacts")
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: python -m pip install PyYAML") from exc
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    run(config, args.stage)


if __name__ == "__main__":
    main()
