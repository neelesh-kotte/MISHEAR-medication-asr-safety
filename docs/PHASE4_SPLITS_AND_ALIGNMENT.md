# Phase 4 utilities: split manifests and ASR error scoring

The implementations are in `src/test_data_splits.py` and `src/test_error_alignment.py`; the established `src/error_alignment.py` import path re-exports the alignment API.

## Environment setup commands

This workspace currently reports Python 3.9.6, FFmpeg present, CMake missing, and `scipy`, `soundfile`, Whisper, Transformers, PyTorch, and CMUdict absent. The earlier numerical requirements are in `requirements.txt`; install them in a project venv from the user's Terminal:

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
brew install cmake
command -v ffmpeg
command -v cmake
python - <<'PY'
import importlib.util
for package in ("numpy", "scipy", "soundfile", "whisper", "transformers", "torch", "cmudict"):
    print(f"{package}: {bool(importlib.util.find_spec(package))}")
PY
```

The ASR dependencies download sizeable packages; this install was not run during code generation. Model weights are separate from pip dependencies and must be pinned in the ASR config.

## Ingredient-disjoint splits for H1

Every recording row needs `ingredient_id` (or equivalent normalized ID); pair edges need `ingredient_a` and `ingredient_b`. Brand/generic aliases must already resolve to the same active ingredient. The pair graph's connected components are assigned whole to one partition, so related pairs cannot bridge train and test.

```python
from src.test_data_splits import create_ingredient_disjoint_splits, write_split_manifest

splits = create_ingredient_disjoint_splits(
    records,                    # recording/audio metadata rows
    pair_edges=medication_pairs, # ingredient_a, ingredient_b, pair_id
    fractions={"train": 0.60, "validation": 0.20, "test": 0.20},
    seed=20261030,
)
write_split_manifest(splits, "data/dataset_splits/ingredient_split_v1.json")
```

The function raises if a record lacks ingredient identity, spans multiple components, or an ingredient/component leaks across partitions. If graph connectivity creates too few independent components for useful partitions, report that feasibility limitation rather than breaking connected components.

## Held-out acoustic conditions for H2

Each row must retain an explicit `acoustic_condition_id` and `condition_family`. Reserve the full clipping family and the ID for the harshest 5 dB babble plus RT60 0.8 s combination. Set those IDs from the frozen DSP manifest; do not infer conditions from measured SNR/RT60 after seeing evaluation performance.

```python
from src.test_data_splits import create_unseen_acoustic_condition_split, write_split_manifest

condition_splits = create_unseen_acoustic_condition_split(
    records,
    held_out_condition_ids={"babble_5db_rt0608"},
    held_out_families={"clipping"},
    pair_edges=medication_pairs,
    validation_fraction=0.20,
    seed=20261030,
)
write_split_manifest(condition_splits, "data/dataset_splits/acoustic_split_v1.json")
```

The acoustic split intentionally permits the same medication component in development and held-out condition test rows; the test is condition generalization. Its train/validation rows remain ingredient/pair-component grouped. The ingredient split and acoustic split answer separate questions and must be evaluated separately.

## Pair augmented files with ASR records and score

Run `src/asr_evaluation.py` first on audio paths alone. Then join its output to generated-audio metadata, which contains the fictional reference sentence and intended drug label. Longest-match vocabulary recognition identifies other medication names in ASR output. Provide normalized generic/brand/salt aliases with an active-ingredient ID when they should count as the same ingredient.

```python
from src.test_error_alignment import (
    attach_reference_metadata,
    load_records,
    pair_audio_metadata_with_asr,
    align_evaluation_records,
    save_alignment_outputs,
)

clean_tts = load_records("data/audio_metadata/synthetic_audio_metadata.csv")
dsp_rows = load_records("data/audio_metadata/augmented/dsp_metadata.csv")
asr_rows = load_records("results/asr/whisper_cpp/transcriptions.jsonl")
augmented = attach_reference_metadata(clean_tts, dsp_rows)
joined = pair_audio_metadata_with_asr(augmented, asr_rows)
aligned = align_evaluation_records(
    joined,
    drug_vocabulary={"hydroxyzine": "ingredient_a", "hydralazine": "ingredient_b"},
    reference_field="sentence",
    hypothesis_field="transcript",
    intended_drug_field="drug_name",
)
save_alignment_outputs(
    aligned,
    records_jsonl="results/tables/aligned_asr_records.jsonl",
    metrics_csv="results/tables/error_metrics_by_condition.csv",
)
```

Outputs include token-level reference/hypothesis alignment, WER, medication substitution/deletion/insertion labels, per-mention rates, and condition-stratified summaries. The word-level drug event calls any nonempty incorrect rendering of the intended medication a substitution; a missing drug with no replacement is a deletion; additional vocabulary-recognized medications are insertions. Keep this definition fixed before final evaluation. The calculated error table is an evaluation label product and must not be passed into the predictor or auditor inputs.
