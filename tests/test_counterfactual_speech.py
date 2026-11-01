import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.counterfactual_speech import generate, noisy_target
from src.run_counterfactual_asr import observed_segment_scores, run
from src.validate_counterfactual_stimuli import validate


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="local eSpeak NG executable unavailable")
def test_matched_target_samples_are_identical_across_contexts(tmp_path: Path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"voices": ["en-gb"], "cases": [{
        "case_id": "unit-case-1", "target": "metformin", "conditions": {
            "neutral": {"prefix": "The medicine is", "suffix": "today."},
            "misleading": {"prefix": "The record says metoprolol, but the patient takes", "suffix": "daily."},
        }}]}))
    out = tmp_path / "results" / "counterfactual-test-run"
    result = generate(spec, output_dir=out, executable=shutil.which("espeak-ng"),
                      voices=["en-gb"], snr_conditions=[None, 10.0],
                      reverb_conditions=[None, 0.1], seed=33)
    assert result["stimulus_rows"] == 8
    rows = [json.loads(line) for line in (out / "stimuli.jsonl").read_text().splitlines()]
    validation = validate(out / "stimuli.jsonl", tmp_path / "validation.json")
    assert validation["status"] == "MATCHED_TARGET_PCM_VALIDATED"
    for rt60 in (None, 0.1):
        for snr in (None, 10.0):
            group = [row for row in rows if row["requested_rt60_seconds"] == rt60
                     and row["requested_snr_db"] == snr]
            assert len({row["target_pcm_float32_sha256"] for row in group}) == 1
            assert len({row["target_pcm16_sha256"] for row in group}) == 1
    for condition in ("neutral", "misleading"):
        clean = next(row for row in rows if row["condition"] == condition
                     and row["requested_snr_db"] is None and row["requested_rt60_seconds"] is None)
        degraded = next(row for row in rows if row["condition"] == condition
                        and row["requested_snr_db"] == 10.0 and row["requested_rt60_seconds"] is None)
        assert clean["clean_target_pcm_float32_sha256"] == degraded["clean_target_pcm_float32_sha256"]
    for row in rows:
        wav, rate = sf.read(out / row["audio_path"], dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        a, b = row["target_start_sample"], row["target_end_sample"]
        assert rate == 16000
        assert b - a == row["target_sample_count"]
        # PCM_16 round trip is allowed; the generator independently checks
        # exact quantized samples before writing its manifest row.
        assert np.isfinite(wav[a:b]).all()
        assert row["pronunciation_status"].startswith("requires human")
        assert row["target_rms"] > 0
        assert 0 <= row["clipping_fraction_at_pcm16_ceiling"] <= 1
        if row["requested_rt60_seconds"] is not None:
            assert row["impulse_response_sha256"]
            assert row["measured_drr_db"] is not None


def test_seeded_degradation_is_deterministic_and_reports_snr():
    target = np.linspace(-0.2, 0.2, 4000, dtype=np.float32)
    first, first_snr = noisy_target(target, 15.0, seed=123)
    second, second_snr = noisy_target(target, 15.0, seed=123)
    assert np.array_equal(first, second)
    assert first_snr == second_snr
    assert abs(first_snr - 15.0) < 0.1


def test_asr_runner_passes_only_bound_audio_and_excludes_gold_fields(tmp_path: Path):
    audio = tmp_path / "stimulus.wav"
    sf.write(audio, np.zeros(8000, dtype=np.float32), 16000, subtype="PCM_16")
    import hashlib
    audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest = tmp_path / "stimuli.jsonl"
    manifest.write_text(json.dumps({
        "case_id": "case-1", "voice": "v", "condition": "neutral", "requested_snr_db": None,
        "audio_path": "stimulus.wav", "audio_sha256": audio_hash,
        "target_text": "metformin", "prefix_text": "diabetes", "target_start_sample": 2,
    }) + "\n")

    class FakeRunner:
        model_path = "/pinned/model"
        _model_hash = "model-hash"
        model_version = "v1"
        decoding_config = {"temperature": 0}

        def transcribe(self, path):
            assert Path(path).read_bytes() == audio.read_bytes()
            assert Path(path).parent.name == "inputs"
            return {"audio_sha256": audio_hash, "recognizer": "fixture", "model_id": "id",
                    "model_version": "v1", "decoding_config": {}, "transcript": "opaque output",
                    "raw_output_path": "raw.json", "runtime_s": 0.01}

    out = tmp_path / "results" / "asr-run"
    result = run(manifest, out, runner=FakeRunner())
    assert result["successful_predictions"] == 1
    prediction = json.loads((out / "predictions.jsonl").read_text())
    assert prediction["stimulus_audio_sha256"] == audio_hash
    assert "target_text" not in prediction
    assert "prefix_text" not in prediction
    assert prediction["excluded_inputs"]


def test_asr_runner_rejects_changed_stimulus_hash(tmp_path: Path):
    audio = tmp_path / "stimulus.wav"
    sf.write(audio, np.zeros(4000, dtype=np.float32), 16000, subtype="PCM_16")
    manifest = tmp_path / "stimuli.jsonl"
    manifest.write_text(json.dumps({"case_id": "x", "voice": "v", "condition": "c",
                                    "audio_path": "stimulus.wav", "audio_sha256": "wrong"}) + "\n")

    class FakeRunner:
        def transcribe(self, path):
            raise AssertionError("invalid audio must be rejected before inference")

    result = run(manifest, tmp_path / "results" / "failed-run", runner=FakeRunner())
    assert result["failed_predictions"] == 1
    assert "hash" in result["errors"][0]["error"].lower()


def test_independent_counterfactual_validator_detects_target_tampering(tmp_path: Path):
    audio = tmp_path / "stimulus.wav"
    samples = np.zeros(8000, dtype=np.float32)
    samples[1000:2000] = 0.1
    sf.write(audio, samples, 16000, subtype="PCM_16")
    import hashlib
    raw = audio.read_bytes()
    wav, _ = sf.read(audio, dtype="int16")
    target_hash = hashlib.sha256(np.asarray(wav[1000:2000], dtype="<i2").tobytes()).hexdigest()
    manifest = tmp_path / "stimuli.jsonl"
    row = {"case_id": "x", "voice": "v", "condition": "a", "requested_snr_db": None,
           "audio_path": "stimulus.wav", "audio_sha256": hashlib.sha256(raw).hexdigest(),
           "sample_rate_hz": 16000, "target_start_sample": 1000, "target_end_sample": 2000,
           "target_sample_count": 1000, "target_pcm16_sha256": "tampered:" + target_hash}
    manifest.write_text(json.dumps(row) + "\n")
    result = validate(manifest, tmp_path / "validation.json")
    assert result["status"] == "FAILED"
    assert result["errors"][0]["issue"] == "target_pcm16_hash_mismatch"


def test_observed_whisper_scores_are_read_only_and_hash_bound(tmp_path: Path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"segments": [{"start": 0.1, "end": 0.9, "text": "drug",
                                             "avg_logprob": -0.7, "no_speech_prob": 0.02,
                                             "compression_ratio": 1.1, "private": "ignore"}]}))
    scores, digest = observed_segment_scores(str(raw))
    assert scores == [{"start": 0.1, "end": 0.9, "text": "drug", "avg_logprob": -0.7,
                       "no_speech_prob": 0.02, "compression_ratio": 1.1}]
    import hashlib
    assert digest == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert observed_segment_scores(str(tmp_path / "absent.json")) == ([], None)
