# Phase 6: Shift Validation, Cross-Recognizer Transfer, and Reporting

## H2 acoustic-condition holdout

`src/run_validation_pipeline.py` reads `h2.held_out_condition_ids` and
`h2.held_out_families` in the pipeline YAML, reserves those complete acoustic
conditions as test data, and groups the remaining development records by
ingredient-pair connected component for train/validation. The model is fitted
only on development training rows. Platt and isotonic calibrators are fitted
only to validation labels/scores and compared on the held-out conditions using
AUPRC, AUROC, Brier score, precision, recall, and equal-width reliability bins.
The report includes prevalence as the AUPRC reference and a grouped bootstrap
comparison. H2 outputs include `h2_split_manifest.json` and
`h2_calibration_predictions.csv`.

The default config holds out the harsh babble-plus-reverberation condition and
the full clipping family. Make sure at least one development condition remains
and that train and validation each contain both outcome classes. Change these
IDs only before accessing H2 test results.

## H4 cross-recognizer transfer

An optional `secondary_asr` configuration runs a second architecture over the
same augmented recordings. The example uses `Wav2Vec2CTCRunner` with a local
Hugging Face CTC model. The transfer model trains on primary-recognizer
development utterances and evaluates the same ingredient-held-out recording
IDs transcribed by the secondary recognizer. H4 labels whether the transcript
confused the intended medication with the other candidate-pair endpoint; other
outcomes are negatives for this pair-specific confusion task. Model fitting
does not use secondary-recognizer test labels. H4 outputs are
`h4_transfer_report.json` and `h4_transfer_predictions.csv`.

Model identifiers/revisions and the configured device are recorded in ASR run
artifacts. Pin a specific model revision for a final reproducible run, and
review its license and model card before use. Loading the Hugging Face model may
download weights the first time; no model is downloaded while editing these
files. The configured `secondary_asr.revision` is passed to both Hugging Face
loaders; set it to a reviewed immutable model commit before freezing a run.

## Results summary

After the pipeline finishes, generate tables, calibration-curve data and PNG,
and a write-up draft with:

```bash
python -m src.generate_report --results-dir results/phase6
```

Outputs are written under `results/phase6/submission/`:

- `evaluation_metrics.csv`
- `h2_calibration_curve_data.csv`
- `h2_calibration_curves.png`
- `DAVIDSON_FELLOWS_RESULTS_DRAFT.md`

The draft is a structured starting point, not a claim that the text meets any
current submission prompt or word limit. Review the program's current rules,
complete exact sample/group counts and model versions, and keep claims limited
to this controlled synthetic benchmark.
