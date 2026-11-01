"""Create provenance-bound controlled context stimuli for exploratory ASR tests.

The primary matched-target condition synthesizes a target once and reuses the
same PCM samples across context conditions. It does not make the splice sound
natural or verify that the TTS voice pronounced a drug as intended; those
remain explicit human-review requirements. No labels or ASR output are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthesize(text: str, *, executable: str, voice: str, speed_wpm: int,
               sample_rate_hz: int, work_dir: Path) -> np.ndarray:
    if not text.strip():
        return np.zeros(0, dtype=np.float32)
    wav_path = work_dir / f"segment-{len(list(work_dir.iterdir()))}.wav"
    subprocess.run([executable, "-v", voice, "-s", str(speed_wpm), "-w", str(wav_path), text],
                   check=True, capture_output=True, text=True)
    audio, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("TTS produced empty or non-finite audio")
    if rate != sample_rate_hz:
        from math import gcd
        divisor = gcd(int(rate), sample_rate_hz)
        audio = resample_poly(audio, sample_rate_hz // divisor, int(rate) // divisor).astype(np.float32)
    return audio.astype(np.float32)


def noisy_target(target: np.ndarray, snr_db: float | None, *, seed: int) -> tuple[np.ndarray, float | None]:
    if snr_db is None:
        return target.copy(), None
    signal_power = float(np.mean(target.astype(np.float64) ** 2))
    if signal_power <= 0:
        raise ValueError("Target audio has zero power")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(target.shape)
    noise_power = float(np.mean(noise ** 2))
    noise *= np.sqrt(signal_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    output = target.astype(np.float64) + noise
    output = np.clip(output, -1.0, 1.0).astype(np.float32)
    realized = 10.0 * np.log10(signal_power / float(np.mean((output - target) ** 2)))
    return output, float(realized)


def reverberate_target(target: np.ndarray, sample_rate_hz: int, rt60_seconds: float | None,
                       *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a deterministic synthetic exponentially decaying impulse response.

    This is a controlled synthetic perturbation, not a measured room response.
    The returned target RMS is matched to the input to isolate reverberation
    from a gross loudness change; peak clipping is still measured downstream.
    """
    if rt60_seconds is None:
        return target.copy(), {"requested_rt60_seconds": None, "impulse_response_sha256": None,
                               "measured_drr_db": None, "normalization_gain": 1.0,
                               "reverberation_type": "dry"}
    if not np.isfinite(rt60_seconds) or rt60_seconds <= 0:
        raise ValueError("RT60 must be a finite positive number")
    tail_duration = min(1.5, max(0.25, 3.0 * rt60_seconds))
    frame_count = max(2, round(sample_rate_hz * tail_duration))
    time_axis = np.arange(frame_count, dtype=np.float64) / sample_rate_hz
    rng = np.random.default_rng(seed)
    impulse = rng.standard_normal(frame_count).astype(np.float64)
    impulse *= np.exp(-6.907755278982137 * time_axis / rt60_seconds)
    impulse[0] = 1.0
    tail_power = float(np.sum(impulse[1:] ** 2))
    drr_db = 10.0 * np.log10(1.0 / tail_power) if tail_power > 0 else None
    ir_hash = hashlib.sha256(impulse.astype("<f8").tobytes()).hexdigest()
    filtered = fftconvolve(target.astype(np.float64), impulse, mode="full")
    input_rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
    output_rms = float(np.sqrt(np.mean(filtered ** 2)))
    if output_rms <= 0:
        raise ValueError("Reverberation produced zero-power audio")
    gain = input_rms / output_rms
    filtered = (filtered * gain).astype(np.float32)
    return filtered, {"requested_rt60_seconds": float(rt60_seconds),
                      "impulse_response_sha256": ir_hash, "measured_drr_db": float(drr_db),
                      "normalization_gain": float(gain),
                      "reverberation_type": "synthetic_exponential_decay_noise_ir_not_measured_room"}


def _verify_target_binding(audio: np.ndarray, target: np.ndarray, start: int, end: int) -> None:
    if start < 0 or end - start != target.size or end > audio.size:
        raise ValueError("Target sample boundaries are invalid")
    if not np.array_equal(audio[start:end], target):
        raise ValueError("Embedded target waveform differs from synthesized target")


def generate(spec_path: Path, *, output_dir: Path, executable: str,
             voices: list[str] | None = None, sample_rate_hz: int = 16000,
             speed_wpm: int = 150, gap_ms: int = 100,
             snr_conditions: list[float | None] | None = None,
             reverb_conditions: list[float | None] | None = None, seed: int = 20261031) -> dict[str, Any]:
    output_resolved = output_dir.resolve()
    if "results" not in output_resolved.parts or output_resolved.exists():
        raise ValueError("Output must be a new run-specific directory under results/")
    binary = shutil.which(executable) if not Path(executable).is_file() else str(Path(executable).resolve())
    if not binary:
        raise FileNotFoundError(f"TTS executable not found: {executable}")
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Stimulus spec must contain a non-empty cases list")
    selected_voices = voices or spec.get("voices") or ["en-gb"]
    snr_values = snr_conditions if snr_conditions is not None else [None]
    reverb_values = reverb_conditions if reverb_conditions is not None else [None]
    if gap_ms < 0 or sample_rate_hz <= 0 or speed_wpm <= 0:
        raise ValueError("Audio settings must be positive and gap_ms non-negative")

    output_resolved.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    gap = np.zeros(round(sample_rate_hz * gap_ms / 1000), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="mishear-tts-") as temp:
        work = Path(temp)
        version_proc = subprocess.run([binary, "--version"], check=True, capture_output=True, text=True)
        version = (version_proc.stdout or version_proc.stderr).strip()
        binary_hash = sha256_file(Path(binary))
        for case in cases:
            case_id = str(case.get("case_id") or "").strip()
            target_text = str(case.get("target") or "").strip()
            conditions = case.get("conditions")
            if not case_id or not target_text or not isinstance(conditions, dict) or not conditions:
                raise ValueError("Each case requires case_id, target, and non-empty conditions")
            for voice in selected_voices:
                target = synthesize(target_text, executable=binary, voice=voice,
                                    speed_wpm=speed_wpm, sample_rate_hz=sample_rate_hz, work_dir=work)
                target_raw = target.astype("<f4").tobytes()
                for rt60 in reverb_values:
                    reverbed, reverb_meta = reverberate_target(
                        target, sample_rate_hz, rt60,
                        seed=seed + int((rt60 or 0) * 1000))
                    for snr in snr_values:
                        variant, realized_snr = noisy_target(reverbed, snr, seed=seed)
                        variant_hash = sha256_bytes(variant.astype("<f4").tobytes())
                        for condition_name, context in conditions.items():
                            prefix = synthesize(str(context.get("prefix", "")), executable=binary, voice=voice,
                                                speed_wpm=speed_wpm, sample_rate_hz=sample_rate_hz, work_dir=work)
                            suffix = synthesize(str(context.get("suffix", "")), executable=binary, voice=voice,
                                                speed_wpm=speed_wpm, sample_rate_hz=sample_rate_hz, work_dir=work)
                            pieces = [piece for piece in (prefix, gap, variant, gap, suffix) if piece.size]
                            if not pieces:
                                raise ValueError("Stimulus contains no audio")
                            audio = np.concatenate(pieces).astype(np.float32)
                            start = int(prefix.size + gap.size)
                            end = start + variant.size
                            _verify_target_binding(audio, variant, start, end)
                            relative = (Path("audio") / case_id / voice /
                                        f"{condition_name}-rt60-{rt60 if rt60 is not None else 'dry'}-snr-{snr if snr is not None else 'clean'}.wav")
                            target_path = output_resolved / relative
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            sf.write(str(target_path), audio, sample_rate_hz, subtype="PCM_16")
                            readback, read_rate = sf.read(str(target_path), dtype="float32", always_2d=True)
                            readback = readback.mean(axis=1)
                            readback_int16, int16_rate = sf.read(str(target_path), dtype="int16", always_2d=True)
                            actual_segment = readback[start:end]
                            if read_rate != sample_rate_hz or not np.allclose(
                                    actual_segment, variant, atol=(1.0 / 32768.0 + 1e-7), rtol=0.0):
                                raise ValueError("Written WAV failed exact target-segment sample validation")
                            if int16_rate != sample_rate_hz or readback_int16.shape[1] != 1:
                                raise ValueError("Written WAV PCM16 channel/rate validation failed")
                            actual_pcm16 = np.asarray(readback_int16[start:end, 0], dtype="<i2")
                            rows.append({
                                "case_id": case_id, "condition": str(condition_name), "voice": voice,
                                "target_text": target_text, "prefix_text": str(context.get("prefix", "")),
                                "suffix_text": str(context.get("suffix", "")),
                                "audio_path": relative.as_posix(), "audio_sha256": sha256_file(target_path),
                                "sample_rate_hz": sample_rate_hz, "duration_seconds": len(audio) / sample_rate_hz,
                                "target_start_sample": start, "target_end_sample": end,
                                "target_start_seconds": start / sample_rate_hz, "target_end_seconds": end / sample_rate_hz,
                                "target_pcm_float32_sha256": variant_hash,
                                "target_pcm16_sha256": sha256_bytes(actual_pcm16.tobytes()),
                                "clean_target_pcm_float32_sha256": sha256_bytes(target_raw),
                                "target_sample_count": int(variant.size), "requested_snr_db": snr,
                                "realized_snr_db": realized_snr, **reverb_meta,
                                "target_rms": float(np.sqrt(np.mean(actual_segment.astype(np.float64) ** 2))),
                                "prefix_rms": (float(np.sqrt(np.mean(readback[:prefix.size].astype(np.float64) ** 2)))
                                               if prefix.size else None),
                                "suffix_rms": (float(np.sqrt(np.mean(readback[end + gap.size:].astype(np.float64) ** 2)))
                                               if suffix.size else None),
                                "full_audio_rms": float(np.sqrt(np.mean(readback.astype(np.float64) ** 2))),
                                "peak_absolute_amplitude": float(np.max(np.abs(readback))),
                                "clipping_fraction_at_pcm16_ceiling": float(np.mean(np.abs(readback) >= (32767.0 / 32768.0))),
                                "splice_policy": "independent prefix + fixed target + independent suffix; fixed zero gaps; no crossfade",
                                "tts": {"engine": "eSpeak NG", "version_output": version,
                                        "binary_sha256": binary_hash, "binary_path": binary,
                                        "voice": voice, "speed_wpm": speed_wpm},
                                "pronunciation_status": "requires human verification; synthesizer output is not pronunciation validation",
                                "scope": "exploratory_synthetic_stimulus_not_human_speech_or_clinical_evidence",
                            })
    manifest = output_resolved / "stimuli.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    provenance = {
        "status": "SYNTHETIC_STIMULI_GENERATED_REQUIRES_PRONUNCIATION_REVIEW",
        "spec_path": str(spec_path.resolve()), "spec_sha256": sha256_bytes(spec_bytes),
        "stimulus_manifest": str(manifest.resolve()), "stimulus_manifest_sha256": sha256_file(manifest),
        "stimulus_rows": len(rows), "cases": len(cases), "voices": selected_voices,
        "snr_conditions_db": snr_values, "reverb_conditions_rt60_seconds": reverb_values,
        "seed": seed, "generation_systems": ["eSpeak NG"],
        "note": "This does not satisfy the multi-generator benchmark requirement and is not ASR evaluation.",
    }
    (output_resolved / "run_summary.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tts", default="espeak-ng")
    parser.add_argument("--voice", action="append", dest="voices")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--speed", type=int, default=150)
    parser.add_argument("--gap-ms", type=int, default=100)
    parser.add_argument("--snr-db", type=float, action="append", dest="snr_db")
    parser.add_argument("--include-clean", action="store_true",
                        help="Include clean target alongside requested --snr-db conditions")
    parser.add_argument("--rt60-s", type=float, action="append", dest="rt60_s")
    parser.add_argument("--include-dry", action="store_true", help="Include dry target alongside requested RT60 conditions")
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    print(json.dumps(generate(args.spec, output_dir=args.output_dir, executable=args.tts,
                             voices=args.voices, sample_rate_hz=args.sample_rate,
                             speed_wpm=args.speed, gap_ms=args.gap_ms,
                             snr_conditions=(([None] if args.include_clean else []) + (args.snr_db or [None])),
                             reverb_conditions=(([None] if args.include_dry else []) + (args.rt60_s or [None])),
                             seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
