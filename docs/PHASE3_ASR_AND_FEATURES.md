# Phase 3: local ASR evaluation and pair features

## Python environment

From the project directory, create/use the project's virtual environment and install `requirements.txt`. The file now includes core DSP, optional Python Whisper and Wav2Vec2 dependencies, and CMUdict pronunciation data:

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Whisper's Python package needs FFmpeg on the system path. Whisper.cpp also needs CMake to build:

```bash
brew install ffmpeg cmake
```

The repository-prespecified ASR is local `whisper.cpp`; the Python Whisper wrapper and Hugging Face Wav2Vec2 runner are also implemented as optional comparison backends. Do not treat a library install as a fixed model: download and pin exact model revisions separately in `config/asr_config.yaml`. Current official whisper.cpp instructions show a local CMake build and ggml model download; on Apple Silicon, Core ML support is a separate optional model-conversion/build path. See the official project instructions before choosing that path.

Example whisper.cpp setup from the project root (keeps the external checkout/model outside tracked project source):

```bash
mkdir -p external
git clone https://github.com/ggml-org/whisper.cpp.git external/whisper.cpp
cd external/whisper.cpp
sh ./models/download-ggml-model.sh base.en
cmake -B build
cmake --build build -j --config Release
```

For Core ML encoder support on Apple Silicon, the official project documents generating the Core ML model before building with `-DWHISPER_COREML=1`; pin the whisper.cpp commit, Core ML model, Python version, and decoding config in the run record. The CLI currently expects 16-bit WAV. `WhisperCppRunner` converts mono/resampled or float DSP WAV inputs to PCM-16 and records the sample-rate conversion and applied gain.

## Run ASR on augmented WAV files

The runner accepts audio paths only. It cannot receive reference sentences, intended medication names, or labels. It stores original transcripts, backend raw JSON, audio SHA-256, model/version, decoding configuration, duration, and runtime.

```python
from pathlib import Path
from src.asr_evaluation import WhisperCppRunner, run_batch

runner = WhisperCppRunner(
    model_path="external/whisper.cpp/models/ggml-base.en.bin",
    executable="external/whisper.cpp/build/bin/whisper-cli",
    output_dir="results/asr/whisper_cpp",
    language="en",
    threads=4,
    beam_size=5,
    best_of=5,
    temperature=0.0,
)
audio_files = sorted(Path("data/audio_metadata/augmented").rglob("*.wav"))
records = run_batch(runner, audio_files, "results/asr/whisper_cpp/transcriptions.jsonl")
```

Optional Python Whisper comparison:

```python
from src.asr_evaluation import OpenAIWhisperRunner, run_batch

runner = OpenAIWhisperRunner(
    model_name="base",
    device="cpu",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,
)
run_batch(runner, audio_files, "results/asr/openai_whisper/transcriptions.jsonl")
```

Optional Wav2Vec2 CTC comparison:

```python
from src.asr_evaluation import Wav2Vec2CTCRunner, run_batch

runner = Wav2Vec2CTCRunner("facebook/wav2vec2-base-960h", device=None)
run_batch(runner, audio_files, "results/asr/wav2vec2/transcriptions.jsonl")
```

`device=None` chooses MPS when PyTorch reports it available, otherwise CPU. This runner uses greedy CTC decoding and saves framewise maximum token probabilities as raw output. Model-specific decoding is not interchangeable; record the model ID/revision and settings and freeze them before test evaluation. Whisper confidence-like scores and Wav2Vec2 frame probabilities are not assumed calibrated.

`run_batch(..., continue_on_error=True)` records per-file errors and continues; the default stops at the first error. Review failures before interpreting model performance. Keep raw outputs unchanged.

## Extract pair-level features

`src/context_features.py` computes lexical edit/Jaro-Winkler features, CMUdict ARPAbet edit distance, syllable/stress differences and broad articulatory feature mismatches where pronunciation coverage exists. If CMUdict has no entry, it leaves phoneme-level fields missing and labels Soundex similarity as a rough proxy. The proxy must not be mislabeled as a phoneme distance.

```python
from src.context_features import extract_pair_features

features = extract_pair_features(
    "hydroxyzine",
    "hydralazine",
    context_template="Please say {drug} now.",
)
```

Context model features are opt-in. To use them, first resolve the provenance question recorded in `DATA_AND_LICENSING.md`, then load a pinned masked language model and pass only fictional matched sentence templates:

```python
from src.context_features import MaskedLMContextScorer, extract_pair_features

scorer = MaskedLMContextScorer("<reviewed-model-id>", revision="<immutable-revision>")
features = extract_pair_features(
    "hydroxyzine", "hydralazine",
    context_template="For the fictional example, use {drug}.",
    contextual_scorer=scorer,
)
```

The masked-LM values are pseudo-likelihood features, not calibrated probabilities. Compare candidate embeddings only within the same pinned model and matched context. No test labels or test performance should be used to tune feature definitions.
