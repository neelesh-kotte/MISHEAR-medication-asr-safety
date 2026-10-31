# Repository map: cross-corpus human-speech protocol

## Active layout

- `config/corpus_config.yaml`: local corpus roots/manifests, field mapping, licenses, and pretraining stop rules.
- `config/asr_config.yaml`: recognizer settings; pin model revisions before inference.
- `data/vocabulary/`: vocabulary metadata, drug aliases, and resolved ingredient IDs. The checked-in pilot CSV is insufficient for training.
- `data/external/`: ignored local AfriSpeech-200 and PriMock57 data. Never commit these files.
- `src/data_inventory.py`: mention/ingredient/speaker inventory and feasibility report.
- `src/data_splits.py`: AfriSpeech official speaker split validation, ingredient holdout, and consultation-grouped PriMock57 splits.
- `src/asr_inference.py`: local ASR inference. It accepts no gold reference fields.
- `src/baselines.py`: confidence, N-best uncertainty, recognizer disagreement, lexical/phonetic anomaly, and combined text-side baseline.
- `src/auditor.py`: ASR-side span retrieval, acoustic support, and review-priority scoring without transcript correction.
- `src/alignment.py`, `src/audit_utils.py`: evaluation-only reference alignment and mention error typing.
- `src/evaluation.py`: top-budget error capture, correctly-recognized-mention false-alert rate, flagged duration, oracle ceiling, and cluster bootstrap.
- `src/ablations.py`: feature-family ablations after partitions are frozen.
- `tests/`: unit tests for the data boundaries, blinding, and offline metrics.

## Execution order

1. Verify licenses and local paths.
2. Run corpus/vocabulary inventory and honor all stop reasons.
3. Verify official AfriSpeech speaker separation; freeze ingredient holdouts and consultation groups.
4. Run ASR and extract blinded baseline/auditor scores.
5. Join reference labels only for alignment and evaluation.
6. Report error capture against fixed review budgets with speaker/ingredient-aware uncertainty.

Historical `src/run_validation_pipeline.py`, `src/dsp_augmentation.py`, and Phase 2–6 synthetic reports are retained as provenance. They are not the active human-corpus protocol and must not be used to substantiate its claims.
