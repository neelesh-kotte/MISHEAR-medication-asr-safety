"""Local ASR backends and immutable transcription-run records.

Supported backends:
* ``WhisperCppRunner``: local whisper.cpp executable and ggml model file.
* ``OpenAIWhisperRunner``: optional ``openai-whisper`` Python package.
* ``Wav2Vec2CTCRunner``: optional Transformers/PyTorch CTC model.

No reference transcript or intended medication label is accepted by these
interfaces. Evaluation labels are created later by ``error_alignment.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _whisper_output_label(model_name: str) -> str:
    """Return a safe short filename label for a Whisper model ID or checkpoint path."""
    candidate = Path(model_name)
    label = candidate.stem if candidate.is_file() else candidate.name
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in label)
    return safe or "whisper"


def _audio_stack():
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("ASR audio preparation requires numpy, scipy, and soundfile.") from exc
    return np, sf, resample_poly


def _prepare_whisper_cpp_wav(input_path: str | Path, output_path: str | Path, target_rate: int = 16_000) -> dict[str, Any]:
    """Convert model input to mono 16-bit PCM WAV, recording any gain applied."""
    np, sf, resample_poly = _audio_stack()
    from math import gcd

    audio, source_rate = sf.read(str(input_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    source_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if source_rate != target_rate:
        divisor = gcd(int(source_rate), int(target_rate))
        audio = resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)
    # whisper.cpp's documented CLI currently expects 16-bit WAV. A single
    # scalar gain preserves the signal/noise ratio if float DSP audio exceeds FS.
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    gain = min(1.0, 0.999 / peak) if peak > 0.999 else 1.0
    audio = audio * gain
    sf.write(str(output_path), audio, target_rate, subtype="PCM_16", format="WAV")
    return {
        "input_sample_rate_hz": int(source_rate),
        "asr_input_sample_rate_hz": int(target_rate),
        "asr_input_channels": 1,
        "input_peak": source_peak,
        "asr_input_gain": float(gain),
        "asr_input_conversion": "mono 16 kHz PCM-16 WAV",
    }


@dataclass(frozen=True)
class ASRRunRecord:
    recording_id: str
    audio_path: str
    audio_sha256: str
    recognizer: str
    model_id: str
    model_version: str
    decoding_config: Mapping[str, Any]
    transcript: str
    raw_output_path: str
    duration_s: float
    runtime_s: float
    error: str = ""


class WhisperCppRunner:
    """Run the official whisper.cpp CLI locally, preserving its full JSON."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        executable: str = "whisper-cli",
        output_dir: str | Path = "results/asr/whisper_cpp",
        language: str = "en",
        threads: int = 4,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        no_gpu: bool = False,
        extra_args: Iterable[str] = (),
        model_version: str = "",
    ):
        resolved = shutil.which(executable) if not Path(executable).is_file() else str(Path(executable))
        if resolved is None:
            raise FileNotFoundError(f"Could not find whisper.cpp executable {executable!r}; install/build whisper.cpp first.")
        self.executable = str(Path(resolved).resolve())
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Whisper model file not found: {self.model_path}")
        self.output_dir = Path(output_dir)
        self.language = language
        self.threads = int(threads)
        self.beam_size = int(beam_size)
        self.best_of = int(best_of)
        self.temperature = float(temperature)
        self.no_gpu = bool(no_gpu)
        self.extra_args = tuple(str(arg) for arg in extra_args)
        self.model_version = model_version
        self._model_hash = _sha256(self.model_path)
        if not self.model_version:
            version = subprocess.run([self.executable, "--version"], capture_output=True, text=True, check=False)
            self.model_version = (version.stdout or version.stderr).strip()[:200] or "version-unreported"

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        recording_id = audio_path.stem
        raw_path = self.output_dir / f"{recording_id}.json"
        log_path = self.output_dir / f"{recording_id}.log.txt"
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="mishear_asr_") as temp_dir:
            cli_wav = Path(temp_dir) / "input.wav"
            conversion = _prepare_whisper_cpp_wav(audio_path, cli_wav)
            output_prefix = self.output_dir / recording_id
            command = [
                self.executable,
                "-m", str(self.model_path),
                "-f", str(cli_wav),
                "-l", self.language,
                "-t", str(self.threads),
                "-bs", str(self.beam_size),
                "-bo", str(self.best_of),
                "-tp", str(self.temperature),
                "-ojf", "-of", str(output_prefix),
            ]
            if self.no_gpu:
                command.append("-ng")
            command.extend(self.extra_args)
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            log_path.write_text(
                "COMMAND: " + json.dumps(command) + "\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr,
                encoding="utf-8",
            )
            generated_json = Path(f"{output_prefix}.json")
            if completed.returncode != 0:
                raise RuntimeError(f"whisper.cpp failed with exit {completed.returncode}; see {log_path}")
            if not generated_json.is_file():
                raise RuntimeError(f"whisper.cpp did not produce JSON output; see {log_path}")
            payload = json.loads(generated_json.read_text(encoding="utf-8"))
            if generated_json.resolve() != raw_path.resolve():
                shutil.move(str(generated_json), raw_path)

        segments = payload.get("transcription", payload.get("segments", []))
        transcript = "".join(
            item.get("text", "") for item in segments if isinstance(item, dict)
        ).strip()
        duration = _audio_duration(audio_path)
        elapsed = time.perf_counter() - start
        config = {
            "language": self.language,
            "threads": self.threads,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "no_gpu": self.no_gpu,
            "extra_args": list(self.extra_args),
            **conversion,
        }
        return asdict(ASRRunRecord(
            recording_id=recording_id,
            audio_path=str(audio_path),
            audio_sha256=_sha256(audio_path),
            recognizer="whisper.cpp",
            model_id=str(self.model_path),
            model_version=f"{self.model_version}; sha256={self._model_hash}",
            decoding_config=config,
            transcript=transcript,
            raw_output_path=str(raw_path),
            duration_s=duration,
            runtime_s=round(elapsed, 6),
        ))


class OpenAIWhisperRunner:
    """Optional Python Whisper backend for quick local comparisons."""

    def __init__(self, model_name: str = "base", *, device: str | None = None, **decoding_config: Any):
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI package with `python -m pip install openai-whisper`.") from exc
        self.model_name = model_name
        self.decoding_config = dict(decoding_config)
        self.device = device
        self._whisper = whisper
        self._model = whisper.load_model(model_name, device=device)
        self.model_version = getattr(whisper, "__version__", "version-unreported")

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        audio_path = Path(audio_path).resolve()
        start = time.perf_counter()
        result = self._model.transcribe(str(audio_path), **self.decoding_config)
        elapsed = time.perf_counter() - start
        raw_path = audio_path.with_suffix(f".{_whisper_output_label(self.model_name)}.whisper.json")
        raw_path.write_text(json.dumps(result, ensure_ascii=False, default=_json_default, indent=2), encoding="utf-8")
        return asdict(ASRRunRecord(
            recording_id=audio_path.stem,
            audio_path=str(audio_path),
            audio_sha256=_sha256(audio_path),
            recognizer="openai-whisper-python",
            model_id=self.model_name,
            model_version=self.model_version,
            decoding_config={"device": self.device, **self.decoding_config},
            transcript=str(result.get("text", "")).strip(),
            raw_output_path=str(raw_path),
            duration_s=_audio_duration(audio_path),
            runtime_s=round(elapsed, 6),
        ))


class Wav2Vec2CTCRunner:
    """Optional local Hugging Face Wav2Vec2 CTC backend."""

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base-960h",
        *,
        device: str | None = None,
        save_frame_confidence: bool = True,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install Wav2Vec2 dependencies with `python -m pip install torch transformers`.") from exc
        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        load_options: dict[str, Any] = {}
        if revision:
            load_options["revision"] = revision
        if cache_dir is not None:
            load_options["cache_dir"] = str(cache_dir)
        load_options["local_files_only"] = bool(local_files_only)
        self.processor = AutoProcessor.from_pretrained(model_id, **load_options)
        self.model = AutoModelForCTC.from_pretrained(model_id, **load_options)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model.to(device).eval()
        self.save_frame_confidence = save_frame_confidence
        self.model_version = getattr(self.model.config, "_commit_hash", None) or revision or "model-revision-unpinned"
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        np, sf, resample_poly = _audio_stack()
        from math import gcd

        audio_path = Path(audio_path).resolve()
        audio, source_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if source_rate != self.sample_rate:
            divisor = gcd(int(source_rate), self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // divisor, source_rate // divisor).astype(np.float32)
        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        start = time.perf_counter()
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)
            predicted_ids = self.torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids)[0].strip()
            frame_confidence = probs.max(dim=-1).values[0].detach().cpu().tolist() if self.save_frame_confidence else []
        elapsed = time.perf_counter() - start
        raw_path = audio_path.with_suffix(".wav2vec2.raw.json")
        raw_path.write_text(json.dumps({"frame_max_probability": frame_confidence}, ensure_ascii=False), encoding="utf-8")
        return asdict(ASRRunRecord(
            recording_id=audio_path.stem,
            audio_path=str(audio_path),
            audio_sha256=_sha256(audio_path),
            recognizer="wav2vec2-ctc",
            model_id=self.model_id,
            model_version=self.model_version,
            decoding_config={"device": self.device, "sample_rate_hz": self.sample_rate,
                             "decoder": "greedy CTC", "revision": self.revision},
            transcript=transcript,
            raw_output_path=str(raw_path),
            duration_s=len(audio) / self.sample_rate,
            runtime_s=round(elapsed, 6),
        ))


def _audio_duration(path: str | Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return 0.0


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object {type(value).__name__} is not JSON serializable")


def run_batch(
    runner: Any,
    audio_paths: Iterable[str | Path],
    output_jsonl: str | Path,
    *,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """Run a backend over audio files and append one result record per file.

    Reference text/labels are intentionally not accepted or stored here.
    Raw backend outputs are kept at each record's ``raw_output_path``.
    """
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as stream:
        for path in audio_paths:
            try:
                record = runner.transcribe(path)
            except Exception as exc:
                if not continue_on_error:
                    raise
                path = Path(path)
                record = asdict(ASRRunRecord(
                    recording_id=path.stem,
                    audio_path=str(path),
                    audio_sha256=_sha256(path) if path.is_file() else "",
                    recognizer=runner.__class__.__name__,
                    model_id=str(getattr(runner, "model_path", getattr(runner, "model_id", ""))),
                    model_version=str(getattr(runner, "model_version", "")),
                    decoding_config={},
                    transcript="",
                    raw_output_path="",
                    duration_s=_audio_duration(path) if path.is_file() else 0.0,
                    runtime_s=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                ))
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            records.append(record)
    return records
