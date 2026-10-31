"""Config-driven wrappers for the repository's local ASR backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .asr_evaluation import (
    OpenAIWhisperRunner,
    Wav2Vec2CTCRunner,
    WhisperCppRunner,
    run_batch,
)


def build_asr_runner(config: Mapping[str, Any], *, output_dir: str | Path = "results/asr/raw"):
    """Instantiate a local recognizer from either pipeline or ASR config shape."""
    backend = str(config.get("backend", "")).casefold().replace("-", "_").replace(".", "_")
    decoding = dict(config.get("decoding") or {})
    if backend in {"openai_whisper", "whisper", "openai_whisper_python"}:
        return OpenAIWhisperRunner(
            str(config.get("model_name", config.get("model", "base"))),
            device=config.get("device"), **decoding,
        )
    if backend in {"whisper_cpp", "whisper_cpp_cli"}:
        model_path = config.get("model_path", config.get("model"))
        if not model_path:
            raise ValueError("whisper.cpp config requires model_path (or model)")
        return WhisperCppRunner(
            model_path,
            executable=str(config.get("executable") or "whisper-cli"),
            output_dir=output_dir,
            language=str(config.get("language", decoding.get("language", "en"))),
            threads=int(config.get("threads", decoding.get("threads", 4))),
            beam_size=int(config.get("beam_size", decoding.get("beam_size", 5))),
            best_of=int(config.get("best_of", decoding.get("best_of", 5))),
            temperature=float(config.get("temperature", decoding.get("temperature", 0.0))),
            no_gpu=bool(config.get("no_gpu", decoding.get("no_gpu", False))),
            model_version=str(config.get("version") or ""),
        )
    if backend in {"wav2vec2_ctc", "transformers_ctc", "wav2vec2"}:
        return Wav2Vec2CTCRunner(
            str(config.get("model_id", config.get("model", "facebook/wav2vec2-base-960h"))),
            device=config.get("device"),
            save_frame_confidence=bool(config.get("save_frame_confidence", True)),
            revision=config.get("revision"),
        )
    raise ValueError(f"Unsupported ASR backend {config.get('backend')!r}")


def transcribe_files(
    audio_paths: Iterable[str | Path],
    config: Mapping[str, Any],
    output_jsonl: str | Path = "results/asr/asr_runs_primary.jsonl",
    *,
    raw_output_dir: str | Path = "results/asr/raw",
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """Transcribe files and write one provenance-rich JSONL row per input."""
    runner = build_asr_runner(config, output_dir=raw_output_dir)
    return run_batch(runner, audio_paths, output_jsonl, continue_on_error=continue_on_error)


__all__ = ["build_asr_runner", "transcribe_files"]
