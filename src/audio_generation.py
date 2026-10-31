"""Synthetic utterance generation through the local macOS ``say`` command.

Example:
    templates = [{"template_id": "neutral_01", "text": "Please say {drug} now."}]
    jobs = build_synthesis_jobs(vocabulary, pair_rows, templates, ["Samantha"], [175])
    generate_jobs(jobs, "data/audio_metadata", sample_rate=16_000)

Only fictional carrier text and selected medication names should be passed in.
The local ``say`` backend is a feasibility option; check current system-voice
terms before redistributing any generated recordings.
"""

from __future__ import annotations

import csv
import hashlib
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SynthesisJob:
    utterance_id: str
    drug_name: str
    ingredient_id: str
    pair_id: str
    template_id: str
    sentence: str
    context_condition: str
    voice: str
    rate_wpm: int


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return value[:64] or "item"


def build_synthesis_jobs(
    vocabulary: Iterable[Mapping[str, str]],
    pair_rows: Iterable[Mapping[str, str]],
    templates: Sequence[Mapping[str, str]],
    voices: Sequence[str],
    rates_wpm: Sequence[int],
) -> list[SynthesisJob]:
    """Expand selected, ingredient-resolved pairs into clean TTS jobs.

    Each pair endpoint is synthesized with every supplied template, voice and
    rate. This is a full grid for the provided pilot subset, not a prescription
    to generate a full cross-product for the eventual benchmark.
    Vocabulary rows accept ``name`` or ``drug_name`` and ``ingredient_rxcui``
    or ``ingredient_id``. Pair rows accept ``pair_id``/``id`` and
    ``name_a``/``drug_a``, ``name_b``/``drug_b`` plus optional ingredient IDs.
    Template text contains ``{drug}`` and may provide ``template_id`` and
    ``context_condition`` metadata.
    """
    name_index: dict[str, str] = {}
    for row in vocabulary:
        name = (row.get("name") or row.get("drug_name") or "").strip()
        ingredient = (row.get("ingredient_rxcui") or row.get("ingredient_id") or "").strip()
        if name:
            name_index[name.casefold()] = ingredient

    jobs: list[SynthesisJob] = []
    for pair_index, pair in enumerate(pair_rows):
        pair_id = (pair.get("pair_id") or pair.get("id") or f"pair_{pair_index:04d}").strip()
        endpoints = (
            (pair.get("name_a") or pair.get("drug_a") or "", pair.get("ingredient_a") or ""),
            (pair.get("name_b") or pair.get("drug_b") or "", pair.get("ingredient_b") or ""),
        )
        for drug_name, explicit_ingredient in endpoints:
            drug_name = drug_name.strip()
            if not drug_name:
                continue
            ingredient = explicit_ingredient.strip() or name_index.get(drug_name.casefold(), "")
            for template_index, template in enumerate(templates):
                template_id = template.get("template_id", f"template_{template_index:02d}")
                template_text = template.get("text", "")
                if "{drug}" not in template_text:
                    raise ValueError(f"Template {template_id!r} must include '{{drug}}'")
                sentence = template_text.format(drug=drug_name)
                context = template.get("context_condition", "unspecified")
                for voice in voices:
                    for rate in rates_wpm:
                        stable = f"{pair_id}|{drug_name}|{template_id}|{voice}|{rate}"
                        digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:8]
                        uid = "_".join((_slug(pair_id), _slug(drug_name), _slug(template_id), _slug(voice), str(rate), digest))
                        jobs.append(SynthesisJob(
                            utterance_id=uid,
                            drug_name=drug_name,
                            ingredient_id=ingredient,
                            pair_id=pair_id,
                            template_id=template_id,
                            sentence=sentence,
                            context_condition=context,
                            voice=voice,
                            rate_wpm=int(rate),
                        ))
    return jobs


def synthesize_say(job: SynthesisJob, output_wav: str | Path, sample_rate: int = 16_000) -> dict[str, object]:
    """Synthesize one utterance with macOS ``say`` and write mono PCM WAV.

    Requires ``soundfile`` and ``scipy`` for reliable AIFF decode/resampling.
    The subprocess receives arguments directly (no shell interpolation).
    """
    if platform.system() != "Darwin" or shutil.which("say") is None:
        raise RuntimeError("The macOS 'say' executable is unavailable; select another licensed TTS backend.")
    try:
        import soundfile as sf
        from math import gcd
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("Install the reviewed audio dependencies: python -m pip install soundfile scipy") from exc

    target = Path(output_wav)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mishear_tts_") as temp_dir:
        aiff_path = Path(temp_dir) / "say_output.aiff"
        command = ["say", "-v", job.voice, "-r", str(job.rate_wpm), "-o", str(aiff_path), job.sentence]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"say failed ({completed.returncode}): {completed.stderr.strip()}")
        audio, source_rate = sf.read(aiff_path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if source_rate != sample_rate:
            divisor = gcd(int(source_rate), int(sample_rate))
            mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor).astype("float32")
        sf.write(target, mono, sample_rate, subtype="PCM_16", format="WAV")

    return {
        **asdict(job),
        "audio_path": str(target),
        "sample_rate_hz": sample_rate,
        "duration_s": round(len(mono) / sample_rate, 6),
        "tts_engine": "macOS say",
        "tts_engine_version": platform.mac_ver()[0],
        "tts_voice": job.voice,
        "tts_rate_wpm": job.rate_wpm,
        "synthetic": True,
    }


def generate_jobs(
    jobs: Iterable[SynthesisJob], output_dir: str | Path, sample_rate: int = 16_000
) -> list[dict[str, object]]:
    """Generate a sequence of jobs and save a metadata CSV next to the audio."""
    root = Path(output_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = [
        synthesize_say(job, audio_dir / f"{job.utterance_id}.wav", sample_rate)
        for job in jobs
    ]
    if records:
        metadata_path = root / "synthetic_audio_metadata.csv"
        fieldnames = list(records[0])
        with metadata_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    return records
