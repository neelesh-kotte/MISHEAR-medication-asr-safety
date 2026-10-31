"""Deterministic DSP augmentation with measured acoustic metadata.

Babble should be built from several *other synthetic utterances* from this
project (never the target recording). ``snr_db`` is measured from the separately
tracked speech and noise components after optional shared room filtering and
before clipping. ``output_snr_db`` additionally counts clipping distortion as
error relative to the dry source; these are deliberately distinct measures.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _audio_dependencies():
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import fftconvolve, resample_poly
    except ImportError as exc:
        raise RuntimeError("DSP requires numpy, scipy, and soundfile; install with `python -m pip install -r requirements.txt`.") from exc
    return np, sf, fftconvolve, resample_poly


def _mono_float(audio):
    np, *_ = _audio_dependencies()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Audio must be a non-empty mono vector or multichannel matrix.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Audio contains NaN or infinite samples.")
    return arr


def read_audio(path: str | Path, target_sample_rate: int | None = None):
    """Read audio as mono float32 and optionally resample to a target rate."""
    np, sf, _, resample_poly = _audio_dependencies()
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if target_sample_rate and sample_rate != target_sample_rate:
        from math import gcd
        divisor = gcd(int(sample_rate), int(target_sample_rate))
        mono = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = target_sample_rate
    return mono, int(sample_rate)


def write_audio(path: str | Path, audio, sample_rate: int) -> None:
    """Write mono 32-bit float WAV, preserving headroom without normalization."""
    _, sf, *_ = _audio_dependencies()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), _mono_float(audio), int(sample_rate), subtype="FLOAT", format="WAV")


def _fit_length(signal, length: int):
    np, *_ = _audio_dependencies()
    if len(signal) >= length:
        return np.asarray(signal[:length], dtype=np.float32)
    if len(signal) == 0:
        return np.zeros(length, dtype=np.float32)
    repeats = (length + len(signal) - 1) // len(signal)
    return np.tile(signal, repeats)[:length].astype(np.float32)


def _active_mask(signal):
    np, *_ = _audio_dependencies()
    peak = float(np.max(np.abs(signal)))
    if peak <= 1e-12:
        return np.ones(len(signal), dtype=bool)
    mask = np.abs(signal) >= peak * 0.01
    return mask if np.any(mask) else np.ones(len(signal), dtype=bool)


def _power(signal, mask=None) -> float:
    np, *_ = _audio_dependencies()
    values = signal[mask] if mask is not None else signal
    return float(np.mean(np.square(values, dtype=np.float64))) if len(values) else 0.0


def measure_snr_db(signal, noise, active_only: bool = True) -> float:
    """Measure component SNR in dB; active-only uses a 1%-peak speech gate."""
    np, *_ = _audio_dependencies()
    signal = _mono_float(signal)
    noise = _fit_length(_mono_float(noise), len(signal))
    mask = _active_mask(signal) if active_only else None
    signal_power = _power(signal, mask)
    noise_power = _power(noise, mask)
    if noise_power <= 1e-20:
        return float("inf")
    if signal_power <= 1e-20:
        return float("-inf")
    return 10.0 * __import__("math").log10(signal_power / noise_power)


def mix_babble_sources(
    sources: Sequence,
    length: int,
    *,
    seed: int = 0,
    sample_rate: int,
    speakers_min: int = 2,
):
    """Mix random segments from at least two non-target synthetic utterances."""
    np, *_ = _audio_dependencies()
    if len(sources) < speakers_min:
        raise ValueError(f"Babble needs at least {speakers_min} distinct source utterances.")
    if length <= 0:
        raise ValueError("length must be positive")
    rng = np.random.default_rng(seed)
    babble = np.zeros(length, dtype=np.float64)
    for source in sources:
        signal = _mono_float(source)
        if len(signal) == 0:
            continue
        # Randomly wrap/offset so the mix is not a single aligned repetition.
        offset = int(rng.integers(0, len(signal)))
        rotated = np.concatenate((signal[offset:], signal[:offset]))
        segment = _fit_length(rotated, length).astype(np.float64)
        rms = float(np.sqrt(np.mean(segment * segment)))
        if rms > 1e-10:
            babble += segment / rms * float(rng.uniform(0.6, 1.0))
    peak = float(np.max(np.abs(babble)))
    if peak <= 1e-12:
        raise ValueError("Babble sources contain no non-zero samples.")
    babble /= peak
    return babble.astype(np.float32)


def make_synthetic_rir(rt60_s: float, sample_rate: int, *, seed: int = 0, tail_rt60: float = 1.5):
    """Create a seeded diffuse exponential-decay room impulse response.

    This lightweight synthetic RIR provides a controllable DSP approximation;
    it is not a measured hospital room response. RT60 is measured from the
    resulting impulse response rather than copied from the requested setting.
    """
    np, *_ = _audio_dependencies()
    if rt60_s <= 0 or sample_rate <= 0 or tail_rt60 < 1.0:
        raise ValueError("rt60_s/sample_rate must be positive and tail_rt60 >= 1")
    length = max(2, int(round(rt60_s * tail_rt60 * sample_rate)))
    t = np.arange(length, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    decay = np.power(10.0, -3.0 * t / rt60_s)
    rir = rng.normal(0.0, 1.0, length) * decay
    rir[0] = 1.0
    # Normalize total energy to a stable unit-energy response.
    rir /= np.sqrt(np.sum(rir * rir))
    return rir.astype(np.float32)


def measure_rt60_s(rir, sample_rate: int) -> float:
    """Estimate RT60 from a Schroeder energy decay curve (-5 to -35 dB fit)."""
    np, *_ = _audio_dependencies()
    rir = _mono_float(rir).astype(np.float64)
    energy = np.square(rir)
    decay = np.cumsum(energy[::-1])[::-1]
    if decay[0] <= 0:
        return 0.0
    edc_db = 10.0 * np.log10(np.maximum(decay / decay[0], 1e-15))
    times = np.arange(len(rir)) / float(sample_rate)
    mask = (edc_db <= -5.0) & (edc_db >= -35.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope, _ = np.polyfit(times[mask], edc_db[mask], 1)
    return float(-60.0 / slope) if slope < 0 else float("nan")


def _scale_noise_to_snr(signal, noise, target_snr_db: float):
    np, *_ = _audio_dependencies()
    signal = _mono_float(signal)
    noise = _fit_length(_mono_float(noise), len(signal))
    mask = _active_mask(signal)
    signal_power = _power(signal, mask)
    noise_power = _power(noise, mask)
    if signal_power <= 1e-20 or noise_power <= 1e-20:
        raise ValueError("Cannot set SNR for silent speech or noise.")
    scale = np.sqrt(signal_power / (noise_power * (10.0 ** (target_snr_db / 10.0))))
    return (noise * scale).astype(np.float32)


@dataclass(frozen=True)
class AugmentationConfig:
    noise_kind: str | None = None  # None, "babble", or "white"
    target_snr_db: float | None = None
    rt60_target_s: float | None = None
    clipping_threshold: float | None = None  # normalized full scale, e.g. 0.75
    seed: int = 0


def augment_audio(
    clean_audio,
    sample_rate: int,
    config: AugmentationConfig,
    *,
    babble_sources: Sequence | None = None,
):
    """Apply babble/white noise, synthetic reverberation and clipping.

    Returns ``(augmented_float32, metadata)``. For babble, supply multiple
    arrays loaded from different synthetic utterances at ``sample_rate``.
    Conditions are applied with reverberation before final noise scaling so the
    reported component SNR is measured after the room filter.
    """
    np, _, fftconvolve, _ = _audio_dependencies()
    clean = _mono_float(clean_audio)
    speech_component = clean.copy()
    noise_component = np.zeros_like(clean)
    measured_rt60 = None

    if config.noise_kind not in (None, "babble", "white"):
        raise ValueError("noise_kind must be None, 'babble', or 'white'")
    if config.noise_kind is None and config.target_snr_db is not None:
        raise ValueError("target_snr_db requires a noise_kind")
    if config.noise_kind is not None and config.target_snr_db is None:
        raise ValueError("target_snr_db is required when noise is enabled")

    if config.noise_kind == "babble":
        if not babble_sources:
            raise ValueError("babble_sources must contain at least two non-target utterances")
        noise_component = mix_babble_sources(
            babble_sources, len(clean), seed=config.seed, sample_rate=sample_rate
        )
    elif config.noise_kind == "white":
        rng = np.random.default_rng(config.seed)
        noise_component = rng.normal(0, 1, len(clean)).astype(np.float32)

    rir = None
    if config.rt60_target_s is not None:
        rir = make_synthetic_rir(config.rt60_target_s, sample_rate, seed=config.seed + 17)
        measured_rt60 = measure_rt60_s(rir, sample_rate)
        speech_component = fftconvolve(speech_component, rir, mode="full").astype(np.float32)
        if config.noise_kind is not None:
            noise_component = fftconvolve(noise_component, rir, mode="full").astype(np.float32)
        else:
            noise_component = np.zeros_like(speech_component)

    if config.noise_kind is not None:
        noise_component = _fit_length(noise_component, len(speech_component))
        noise_component = _scale_noise_to_snr(speech_component, noise_component, float(config.target_snr_db))

    mixture = speech_component + noise_component
    snr_measured = measure_snr_db(speech_component, noise_component) if config.noise_kind else None
    output = mixture.copy()

    if config.clipping_threshold is not None:
        threshold = float(config.clipping_threshold)
        if not 0 < threshold <= 1:
            raise ValueError("clipping_threshold must be in (0, 1]")
        output = np.clip(output, -threshold, threshold)

    # This second number includes both additive noise and clipping distortion;
    # it should not be interpreted as a clean component SNR.
    dry_extended = _fit_length(clean, len(output))
    output_error = output - dry_extended
    output_snr = measure_snr_db(dry_extended, output_error, active_only=True)
    peak_before_clip = float(np.max(np.abs(mixture)))
    clipping_fraction = (
        float(np.mean(np.abs(mixture) >= config.clipping_threshold))
        if config.clipping_threshold is not None else 0.0
    )

    metadata = {
        **asdict(config),
        "sample_rate_hz": int(sample_rate),
        "duration_s": round(len(output) / sample_rate, 6),
        "measured_snr_db": round(float(snr_measured), 5) if snr_measured is not None else None,
        "measured_rt60_s": round(float(measured_rt60), 5) if measured_rt60 is not None else None,
        "output_snr_including_distortion_db": round(float(output_snr), 5),
        "peak_before_clipping": round(peak_before_clip, 7),
        "clipped_sample_fraction": round(clipping_fraction, 8),
        "rt60_method": "Schroeder reverse-integrated energy decay; linear fit -5 to -35 dB",
        "snr_method": "active speech samples (>=1% of component peak); component powers",
        "synthetic": True,
    }
    return output.astype(np.float32), metadata


def augment_wav(
    input_path: str | Path,
    output_path: str | Path,
    config: AugmentationConfig,
    *,
    babble_paths: Sequence[str | Path] | None = None,
    metadata_csv: str | Path | None = None,
):
    """File-oriented wrapper; writes augmented WAV and optional one-row CSV."""
    clean, sample_rate = read_audio(input_path)
    source_paths = [Path(path) for path in (babble_paths or [])]
    target_resolved = Path(input_path).resolve()
    if any(path.resolve() == target_resolved for path in source_paths):
        raise ValueError("The target utterance must not be included in its own babble sources.")
    sources = [read_audio(path, sample_rate)[0] for path in source_paths]
    augmented, metadata = augment_audio(clean, sample_rate, config, babble_sources=sources)
    write_audio(output_path, augmented, sample_rate)
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    metadata.update({
        "input_path": str(input_path),
        "input_sha256": sha256(Path(input_path)),
        "output_path": str(output_path),
        "babble_source_paths": ";".join(str(path) for path in source_paths),
        "babble_source_sha256": ";".join(sha256(path) for path in source_paths),
    })
    if metadata_csv:
        target = Path(metadata_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metadata))
            writer.writeheader()
            writer.writerow(metadata)
    return metadata
