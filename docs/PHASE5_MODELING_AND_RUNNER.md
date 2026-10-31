# Phase 5: Pair Risk Models and Validation Runner

## Model and target

`src/pair_risk_model.py` fits regularized logistic regression using explicit
phonetic, lexical, contextual, acoustic, and recognizer feature groups. Median
imputation and categorical levels are learned from training rows only. The
feature allowlist excludes transcript/reference strings, labels, recording and
pair identifiers, ingredient IDs, and requested DSP settings. `None` acoustic
measurements remain missing until train-fitted imputation. A phonetic-only
logistic baseline and a Soundex similarity ranking baseline are reported.

The pair-confusion target is positive when ASR replaces the intended name with
the other endpoint of that candidate pair. Correct names, deletions, and
substitutions to a third name are negative for this pair-specific target; the
full alignment artifact retains their separate event labels. The unit is an
utterance-condition record and uncertainty resampling groups by the
ingredient-pair connected component. This estimates synthetic benchmark
performance, not clinical error risk. Regenerate pair features and reports
after upgrading to the Phase 6 label definition; do not mix old output files.

`src/selective_prediction.py` selects a risk threshold using validation scores
only. It uses a Bonferroni-adjusted, one-sided Clopper–Pearson bound over
observed score thresholds and converts the negative-class FPR bound to false
alerts per 100 total mentions using validation prevalence. If the requested
budget cannot be certified, it reports `guarantee_met: false` and an
always-negative threshold. Freeze a numeric budget in the config before any
test evaluation. The threshold is then applied once to test scores.

`src/statistical_analysis.py` reports average precision (AUPRC), AUROC, Brier
score and threshold metrics, exact two-sided McNemar tests for paired
correctness, ingredient-component bootstrap intervals, and paired component
bootstrap AUPRC deltas. McNemar in the runner compares full and phonetic models
at thresholds each selected on validation data. Groups, rather than clips, are
resampled; the bootstrap interval is a percentile interval and can be unstable
when the benchmark has few independent components.

## End-to-end run

Prepare the reviewed medication vocabulary CSV with `name` and resolved
`ingredient_rxcui` (or `ingredient_id`) columns. Configure paths, approved TTS
voice, fixed DSP conditions, local ASR model, and a numeric false-alert budget
in `config/pipeline_config.yaml`. The checked-in YAML intentionally leaves the
budget null, so the runner refuses to make a confirmatory test report until it
is set.

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
command -v ffmpeg
command -v cmake || true
```

The default primary backend is whisper.cpp. Install/build whisper.cpp and place
the reviewed model at `models/ggml-base.en.bin`, or change its executable and
model paths in the YAML. On macOS, the configured `say` voice must be available.
For a small pipeline feasibility run, set `max_pairs` to a small number, use a
short set of DSP conditions, and provide a budget chosen before inspecting any
test output.

```bash
python -m src.run_validation_pipeline --config config/pipeline_config.yaml --stage all
```

Artifacts are written under the configured output directory: candidate pairs,
TTS metadata/audio, augmented audio and measured metadata, raw ASR JSON/logs,
alignment records, pair features/labels, grouped split manifest, test
predictions, and a JSON report. `--stage model` resumes from existing
candidate-pair, synthesis, augmentation, and ASR artifacts. `--stage asr`
reruns ASR/alignment from saved candidate and synthesis metadata.

## Analysis boundary

Phase 6 extends the runner with H2, validation-fitted calibration, and H4;
see `PHASE6_EXTENDED_VALIDATION.md`. Contextual language-model features are
supported by `context_features.py` when a scorer is supplied, but the default
end-to-end config does not select or download a contextual model.
