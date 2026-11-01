"""ASR wrapper regressions that do not load external model weights."""

import json
from pathlib import Path
import wave

from src.asr_evaluation import OpenAIWhisperRunner, _whisper_output_label


def test_whisper_checkpoint_path_has_safe_output_label(tmp_path):
    checkpoint = tmp_path / "local base model.pt"
    checkpoint.touch()
    assert _whisper_output_label(str(checkpoint)) == "local_base_model"
    assert _whisper_output_label("org/model:revision") == "model_revision"


def test_openai_whisper_runner_accepts_local_checkpoint_path(tmp_path):
    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 160)
    checkpoint = tmp_path / "base.pt"
    checkpoint.touch()

    class FakeModel:
        def transcribe(self, _audio_path, **_kwargs):
            return {"text": "development smoke"}

    runner = OpenAIWhisperRunner.__new__(OpenAIWhisperRunner)
    runner.model_name = str(checkpoint)
    runner.decoding_config = {}
    runner.device = "cpu"
    runner._model = FakeModel()
    runner.model_version = "unit-test"
    result = runner.transcribe(audio)

    assert result["transcript"] == "development smoke"
    assert result["raw_output_path"].endswith(".base.whisper.json")
    assert json.loads(Path(result["raw_output_path"]).read_text(encoding="utf-8"))["text"] == "development smoke"
