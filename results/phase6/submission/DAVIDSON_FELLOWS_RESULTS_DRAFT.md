# MISHEAR Results Summary Draft

This is a results-writing aid for the Davidson Fellows submission. Check the current program prompt, eligibility rules, and word limits before using it. Replace bracketed fields with project-specific details and retain negative or inconclusive findings.

## Research question and design

MISHEAR evaluates whether interpretable models can identify medication-name substitutions made by ASR on a controlled synthetic-speech benchmark. Candidate medication pairs were generated from the configured vocabulary; fictional utterances were synthesized, acoustically augmented, transcribed, and aligned to intended names. Models use phonetic, lexical, contextual, measured acoustic, and recognizer features where available. Ingredient-linked groups are kept together for H1; H2 reserves configured acoustic conditions; H4 evaluates the primary-recognizer model on matched examples from a second recognizer.

## Results to report

- **Descriptive primary-recognizer WER:** 1.2563 overall across 40 recordings (160 reference words); by condition: babble_10db_rt60_04=1.3438; babble_15db=1.1875; babble_5db_rt60_08=1.8750; clean=0.9375; clipping_075=0.9375. This is a synthetic pilot summary, not a held-out generalization estimate.
- **H1, ingredient-disjoint test:** full-model AUPRC unavailable (n=unavailable, ingredient components=unavailable, prevalence unavailable); AUROC unavailable. The grouped-bootstrap AUPRC difference versus the phonetic-only model was unavailable (95% interval unavailable to unavailable); versus the Soundex ranking baseline it was unavailable (unavailable to unavailable). Pair-confusion positives: 0/40; no model result was estimable without both classes.
- **H2, unseen acoustic conditions:** status not run (classifier stage did not complete); raw-model AUPRC unavailable (n=not reported, ingredient components=not reported), compared with held-out prevalence unavailable. Calibration methods evaluated: none.
- **H4, cross-recognizer transfer:** not run (no secondary recognizer report found); secondary recognizer not reported; AUPRC unavailable, AUROC unavailable (n=not reported, ingredient components=not reported).
- **Selective alerting:** H3 selective auditing was not evaluated: the pair-risk classifier did not produce validation scores or a frozen threshold. Do not describe a validation bound as a guarantee under distribution shift.

## Interpretation and limits

These results characterize the configured synthetic benchmark. Synthetic voices, generated carrier sentences, selected medication terms, the chosen ASR checkpoints, and simulated room/noise conditions do not establish performance on real clinical speech or clinical benefit. Report the number of independent ingredient-pair components and class prevalence alongside clip counts; repeated clips are not independent observations. If split construction leaves too few independent components or a class absent, label that analysis infeasible rather than relaxing the split after viewing test results.

## Suggested figures and tables

- `evaluation_metrics.csv`: model-by-hypothesis metrics table.
- `wer_by_condition.csv` and `wer_by_condition.png`: descriptive primary-recognizer WER by acoustic condition.
- `h2_calibration_curves.png` and `h2_calibration_curve_data.csv`: generated only when held-out H2 probability predictions exist.
- `report_status.md`: states when a calibration artifact could not be produced from available outputs.
- `../test_predictions.csv`, `../h2_calibration_predictions.csv`, and `../h4_transfer_predictions.csv`: per-example predictions for audit and supplementary analysis.

## Items to complete before submission

Insert the final sample/group counts, exact ASR and TTS versions, prespecified thresholds, confidence intervals, error counts, limitations, and the project's novel contribution after checking the saved artifacts. Keep synthetic benchmark claims separate from patient-safety claims.
