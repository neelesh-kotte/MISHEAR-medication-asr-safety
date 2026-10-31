# MISHEAR: Cross-Corpus Medication Transcription Auditing on Existing Human-Speech Datasets

MISHEAR evaluates whether an audio-aware auditor can prioritize medication transcription errors beyond ASR confidence and recognizer disagreement, on held-out speakers and medication ingredients, and whether performance transfers to PriMock57. The planned corpora are AfriSpeech-200 and PriMock57. **This workspace contains only synthetic pipeline fixtures, not corpus audio/transcripts; no cross-corpus empirical results are claimed.**

AfriSpeech-200 is a read-sentence clinical/general speech dataset and is distributed under CC BY-NC-SA 4.0. PriMock57 contains mock primary-care conversations with human utterance transcripts and is distributed under CC BY 4.0 in its upstream repository. Their speech styles differ: read sentence material is not equivalent to mock conversational speech. See [the data and licensing audit](docs/DATA_AND_LICENSING.md) before downloading, preprocessing, or sharing either corpus.

## Start here

1. Review the frozen pivot protocol in [`prespecification.md`](prespecification.md).
2. [`config/corpus_config.yaml`](config/corpus_config.yaml) points at the expected local real-corpus paths and intentionally fails closed until files, access acceptance, and immutable revisions are verified. The fixtures have a separate [`config/corpus_config.stub.yaml`](config/corpus_config.stub.yaml). Follow [`docs/ACCESS_AND_REPRODUCIBILITY.md`](docs/ACCESS_AND_REPRODUCIBILITY.md); keep licensed audio and transcripts outside Git.
3. Run the deterministic metadata/split/artifact smoke test:

   ```bash
   python run_pipeline.py --stage all --config config/corpus_config.stub.yaml
   ```

   It reports `FEASIBLE_FOR_PIPELINE_SMOKE_TEST` while explicitly setting `training_authorized: false`.
4. For real data, run the inventory gate on the private pinned config. It must pass before ASR-side training/evaluation.
6. Generate paper tables and error-capture curves from frozen outputs by copying [`config/paper_artifacts.example.yaml`](config/paper_artifacts.example.yaml) to a local config, filling verified paths, then running `python -m src.generate_paper_artifacts --config config/paper_artifacts.yaml`.

## Blinding and evaluation

ASR inference, baseline scoring, and auditor scoring may use audio, ASR text, ASR confidence/N-best output, other recognizer output, and the versioned authorized vocabulary. They must not receive the reference drug, reference timestamps, recording-specific gold candidates, or correctness labels. Gold annotations are joined only in evaluation code. Primary outputs are error capture at top 1%, 5%, 10%, and 20% review budgets, false-alert rate among correctly recognized medication mentions, precision among flagged utterances, missed errors, duration reviewed, and an explicitly labeled oracle ceiling. These quantify error concentration and prioritization; they do not establish workload reduction or harm prevention.

## Repository map

- `config/`: corpus, recognizer, and experiment settings.
- `data/vocabulary/`: v1.0 development vocabulary with RxNorm ingredient IDs and explicit brand/combination mappings; recheck pinned source releases before extending it.
- `src/data_inventory.py`, `src/data_splits.py`: inventory, stop rules, and corpus-aware partitioning.
- `src/baselines.py`, `src/auditor.py`: blinded transcript-only baselines and audio-aware review scoring.
- `src/evaluation.py`: offline review-budget metrics and cluster-aware intervals.
- `src/generate_paper_artifacts.py`: ablation, subgroup, inventory table, and error-capture paper artifacts.
- `docs/VOCABULARY_SCHEMA.md`: vocabulary format, validation, and release requirements.
- `docs/ACCESS_AND_REPRODUCIBILITY.md`: exact access, revision-pinning, and real-data setup protocol.
- `tests/`: unit tests for boundary matching, splits, blinding interfaces, and budget metrics.
- `docs/DAVIDSON_TECHNICAL_PAPER.md`: paper draft; currently a protocol, not an empirical report.

Never commit corpus audio, full licensed annotation files, model weights, credentials, or local absolute paths. Preserve each dataset’s attribution and license conditions.
