# Phase 2 audio generation and DSP workflow

## Scope

`src/audio_generation.py` generates clean synthetic utterances from caller-supplied fictional carrier templates and already-filtered medication pairs. `src/dsp_augmentation.py` generates condition variants. No human or patient audio is accepted by project policy. The TTS backend currently implemented is local macOS `say`; its use is suitable for local feasibility work while current System Voice terms are reviewed. Do not redistribute generated `say` audio until rights are confirmed.

## Install the audio dependencies

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This installs the NumPy/SciPy/soundfile stack. If installed versions are later frozen, record their exact resolved versions in the run log. `say` is an OS component, not a pip dependency.

## Generate pilot speech

Use a small pair CSV/list, a few fictional sentence templates, and only voices reported by `say -v '?'`. Provide an ingredient identifier for every pair endpoint so variants do not become false independent samples. The generation helper currently expands every selected endpoint across the provided templates, voices, and rates; provide only the intended pilot grid.

```python
from src.audio_generation import build_synthesis_jobs, generate_jobs

vocabulary = [
    {"name": "hydroxyzine", "ingredient_rxcui": "INGREDIENT_A"},
    {"name": "hydralazine", "ingredient_rxcui": "INGREDIENT_B"},
]
pairs = [{"pair_id": "pair_0001", "name_a": "hydroxyzine", "name_b": "hydralazine"}]
templates = [
    {"template_id": "neutral_01", "context_condition": "neutral", "text": "Please say {drug} now."},
]
jobs = build_synthesis_jobs(vocabulary, pairs, templates, voices=["<installed voice>"], rates_wpm=[160, 180])
records = generate_jobs(jobs, "data/audio_metadata", sample_rate=16_000)
```

The output WAV files are mono PCM-16 at the requested rate; `synthetic_audio_metadata.csv` records text, ingredient/pair/template, voice, rate, platform version, duration, and synthetic status. The code does not determine the correct pronunciation of a name. Verify name pronunciations and record any pronunciation overrides before the benchmark.

## Apply conditions and measure them

For babble, supply at least two *other* synthetic utterances as source files. Do not include the target clip in its own babble. A 5 dB request is scaled to 5 dB based on the measured speech and babble component powers over the active-speech mask after optional shared room filtering; the achieved value is stored as `measured_snr_db`. White noise is an optional DSP control and is not a substitute for the blueprint’s babble condition.

```python
from src.dsp_augmentation import AugmentationConfig, augment_wav

metadata = augment_wav(
    "data/audio_metadata/audio/example.wav",
    "data/audio_metadata/augmented/example_babble5_rt06.wav",
    AugmentationConfig(noise_kind="babble", target_snr_db=5, rt60_target_s=0.6, seed=20261030),
    babble_paths=["data/audio_metadata/audio/other_1.wav", "data/audio_metadata/audio/other_2.wav"],
    metadata_csv="data/audio_metadata/augmented/example.csv",
)
```

Prespecified Phase 2 targets from the project plan:

- Babble SNR: 5, 10, and 15 dB.
- Simulated room RT60: 0.3, 0.5, and 0.8 seconds.
- Clipping thresholds: define three full-scale amplitudes in the experiment config before generation (the blueprint specifies three thresholds but does not give numeric values).
- Harsh held-out combination: 5 dB babble plus RT60 0.8 seconds; hold it out from model development and calibration. The full clipping family is also held out.

The reverberation implementation uses a seeded exponentially decaying diffuse-noise room impulse response, not a measured hospital impulse response. `measured_rt60_s` is estimated from the generated RIR with a Schroeder reverse-integrated energy curve and a linear fit from -5 to -35 dB. This gives a reproducible simulated condition, not evidence about any real room. `measured_snr_db` is component SNR before clipping; `output_snr_including_distortion_db` folds reverberation/clipping residuals into the difference from the dry source and must not be reported as additive-noise SNR. `clipped_sample_fraction` quantifies the applied clipping. DSP WAV outputs are float WAV to avoid unreported PCM saturation.

## Run provenance

Fix and record the seed, source utterance paths/hashes, input audio hash, requested and measured SNR/RT60, clipping threshold, sample rate, TTS engine/version/voice/rate, sentence/template ID, and downstream ASR configuration. Keep each transform as a derived output; preserve the original clean TTS file unchanged. The current code stores the audio hashes and condition measurements in the metadata CSV. Build and freeze the dataset split before model work; condition labels are metadata, while measured SNR/RT60 are separate measured features.
