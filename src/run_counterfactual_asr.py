"""Run pinned local ASR on generated stimuli without passing target labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .asr_evaluation import OpenAIWhisperRunner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_segment_scores(raw_output_path: str | None) -> tuple[list[dict[str, Any]], str | None]:
    """Extract only scalar fields actually emitted by the Whisper decoder."""
    if not raw_output_path:
        return [], None
    path = Path(raw_output_path)
    if not path.is_file():
        return [], None
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")
    scores = [{key: segment[key] for key in fields if key in segment}
              for segment in raw.get("segments", []) if isinstance(segment, dict)]
    return scores, sha256(path)


def run(manifest: Path, output_dir: Path, *, model: str = "base", device: str = "cpu",
        decoding: dict[str, Any] | None = None, runner: Any | None = None,
        checkpoint_path: Path | None = None) -> dict[str, Any]:
    out = output_dir.resolve()
    if "results" not in out.parts or out.exists():
        raise ValueError("Output must be a new run-specific directory under results/")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Stimulus manifest is empty")
    decoding_config = decoding or {
        "language": "en", "temperature": 0.0, "fp16": False,
        "condition_on_previous_text": False, "word_timestamps": False,
        "verbose": False, "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0, "compression_ratio_threshold": 2.4,
    }
    if runner is None:
        if checkpoint_path is None:
            checkpoint_path = Path.home() / ".cache" / "whisper" / f"{model}.pt"
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Pinned local Whisper checkpoint is absent; no model download is attempted")
        runner = OpenAIWhisperRunner(str(checkpoint_path), device=device, **decoding_config)
    out.mkdir(parents=True)
    predictions = []
    errors = []
    for row in rows:
        audio = (manifest.parent / row["audio_path"]).resolve()
        try:
            if not audio.is_file() or sha256(audio) != row["audio_sha256"]:
                raise ValueError("Stimulus audio is missing or its SHA-256 hash differs from the generation manifest")
            # Only the audio path is supplied to recognizer inference. Target,
            # condition text, and target boundaries are excluded.
            # The backend writes raw JSON beside its input. Copy into this run
            # so reruns cannot overwrite previous raw evidence.
            isolated = out / "inputs" / (row["audio_sha256"] + ".wav")
            isolated.parent.mkdir(exist_ok=True)
            if not isolated.exists():
                shutil.copyfile(audio, isolated)
            result = runner.transcribe(isolated)
            if result.get("audio_sha256") != row["audio_sha256"]:
                raise ValueError("Recognizer output is not bound to stimulus audio")
            observed_scores, raw_hash = observed_segment_scores(result.get("raw_output_path"))
            predictions.append({
                "stimulus_id": row["audio_sha256"],
                "stimulus_audio_sha256": row["audio_sha256"],
                "recognizer": result.get("recognizer"), "model_id": result.get("model_id"),
                "model_version": result.get("model_version"),
                "decoding_config": result.get("decoding_config"),
                "transcript": result.get("transcript"),
                "raw_output_path": result.get("raw_output_path"),
                "raw_output_sha256": raw_hash,
                "observed_decoder_segment_scores": observed_scores,
                "runtime_seconds": result.get("runtime_s"),
                "error": result.get("error", ""),
                "inference_inputs": ["stimulus waveform"],
                "excluded_inputs": ["target_text", "context text", "condition label", "reference", "correctness label", "risk score"],
                "score_limit": "Decoder avg_logprob/no_speech_prob/compression_ratio are raw emitted signals, not calibrated correctness probabilities.",
                "evaluation_status": "UNLABELED_SYNTHETIC_ENGINEERING_RUN",
            })
        except Exception as exc:
            failure = {"stimulus_id": row.get("case_id"), "error": f"{type(exc).__name__}: {exc}"}
            errors.append(failure)
            predictions.append(failure)
    pred_path = out / "predictions.jsonl"
    pred_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions),
                         encoding="utf-8")
    summary = {
        "status": "COMPLETED_WITH_ERRORS" if errors else "COMPLETED_UNLABELED_SYNTHETIC_INFERENCE",
        "purpose": "exploratory local synthetic audio inference; not performance evaluation",
        "stimulus_manifest": str(manifest.resolve()), "stimulus_manifest_sha256": sha256(manifest),
        "stimuli_attempted": len(rows), "successful_predictions": len(predictions) - len(errors),
        "failed_predictions": len(errors), "prediction_path": str(pred_path),
        "prediction_sha256": sha256(pred_path), "model_requested": model,
        "model_checkpoint_path": str(checkpoint_path) if runner.__class__ is OpenAIWhisperRunner else str(getattr(runner, "model_path", "")),
        "model_checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path and checkpoint_path.is_file()
                                   else getattr(runner, "_model_hash", None),
        "model_version": getattr(runner, "model_version", None),
        "decoding_config": getattr(runner, "decoding_config", decoding or {}),
        "errors": errors,
        "interpretation_limit": "No target pronunciations were human-verified; no errors or context effects are counted.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default="en")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = run(args.manifest, args.output_dir, model=args.model, device=args.device,
                 decoding={"language": args.language, "temperature": args.temperature, "fp16": False,
                           "condition_on_previous_text": False, "word_timestamps": False,
                           "verbose": False, "no_speech_threshold": 0.6,
                           "logprob_threshold": -1.0, "compression_ratio_threshold": 2.4},
                 checkpoint_path=args.checkpoint)
    print(json.dumps(result, indent=2))
    return 0 if result["failed_predictions"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
