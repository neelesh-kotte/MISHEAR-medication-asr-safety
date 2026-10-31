"""Public audio-augmentation API backed by the measured DSP implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .dsp_augmentation import (
    AugmentationConfig,
    augment_audio,
    augment_wav,
    make_synthetic_rir,
    measure_rt60_s,
    measure_snr_db,
    mix_babble_sources,
    read_audio,
    write_audio,
)


def augment_file(
    source_path: str | Path,
    output_path: str | Path,
    config: AugmentationConfig,
    *,
    babble_paths: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    """Augment a WAV and return metadata including measured SNR and RT60."""
    return augment_wav(source_path, output_path, config, babble_paths=babble_paths)


__all__ = [
    "AugmentationConfig", "augment_audio", "augment_file", "augment_wav",
    "make_synthetic_rir", "measure_rt60_s", "measure_snr_db",
    "mix_babble_sources", "read_audio", "write_audio",
]
