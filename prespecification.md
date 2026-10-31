# MISHEAR v2 prespecification

**Status:** Timestamped project prespecification; not a public registry registration.
**Frozen at:** 2026-10-30 (America/Los_Angeles; repository commit timestamp records the exact time).
**Source protocol:** `MISHEAR v2 Complete Execution Plan.pdf`, updated October 30, 2026.
**Research type:** Fully computational; synthetic fictional speech, public terminology, and pretrained open-source ASR. No human participants, patient recordings, medical records, or real clinical notes are planned.

## Research question

Can a calibrated machine-learning system predict medication-name substitutions made by automatic speech-recognition (ASR) systems on previously unseen active ingredients and previously unseen acoustic conditions, using phonetic, lexical, contextual, and measured acoustic features, and can a selective auditing system identify these errors at a controlled false-alert rate?

## Confirmatory hypotheses

- **H1 — Unseen-medication predictive generalization:** A model combining phonetic, lexical, contextual, and measured acoustic features will predict medication-name transcription substitutions involving held-out active ingredients better than a phonetic-similarity-only baseline. **Primary endpoint:** AUPRC on the ingredient-disjoint test set, with paired AUPRC difference versus the phonetic baseline.
- **H2 — Unseen-acoustic-condition generalization:** Predictive performance will remain measurably above the positive-class prevalence baseline on conditions excluded from model development. **Primary endpoint:** AUPRC on the held-out acoustic-condition set, compared with positive-class prevalence.
- **H3 — Selective auditing:** A calibrated medication-specific auditor will identify more actual medication-name transcription errors than a similarity-only alert system at the same prespecified false-alert rate. **Primary endpoint:** error detection rate (percentage of actual errors detected) at a fixed false-alert budget, selected on validation data only using a finite-sample risk-control method. The numerical budget must be fixed before any test run; the plan gives 5 false alerts per 100 medication mentions as an example, not a final value.
- **H4 — Cross-recognizer transfer:** The predictive approach will retain measurable value when evaluated using a second independently selected pretrained ASR recognizer. **Primary endpoint:** AUPRC on held-out data from the second recognizer.

All hypotheses are falsifiable. Failed hypotheses and negative results will be reported. Apply Holm correction across H1–H4.

## Exploratory analyses (not used to rescue confirmatory results; no multiplicity correction)

- **E1 — Clinical context:** Compare ASR error rates and auditor performance for matched context-informative and context-neutral fictional sentences.
- **E2 — Human-risk overlap:** Compare ASR-vulnerable medication pairs with public human look-alike/sound-alike risk lists; identify vulnerable pairs absent from those lists as candidates for verification, not clinical findings.
- **E3 — Recognizer disagreement:** Assess whether disagreement between the two recognizers is a useful auditor feature.
- **E4 — False-alert control under shift:** Assess whether the validation-selected false-alert budget holds on held-out acoustic conditions.

## Separate prediction tasks

1. **Pair-level confusion prediction:** Estimate whether a recognizer confuses a medication-name pair under specified conditions. The target is pair vulnerability.
2. **Recording-level error auditing:** Given audio, an ASR transcript, candidate alternatives, and available context, flag a specific transcript as potentially erroneous. The reference transcript and true medication label are withheld from the auditor and used only after prediction to calculate outcomes. The prototype will never automatically replace a medication name.

## Evaluation and analysis commitments

- Main model: interpretable logistic regression using prespecified feature groups; LightGBM is optional only if data support it. Comparators: positive prevalence, phonetic similarity ranking, and phonetic-plus-lexical logistic regression.
- Calibration: compare Platt scaling and isotonic regression using validation data only; report reliability and Brier score.
- H1 split: ingredient-disjoint partitions based on connected components of the medication-pair graph, with approximate 60/20/20 train/validation/test targets and no shared active ingredient across partitions.
- H2 split: reserve the full clipping family and the harshest 5 dB babble plus RT60 0.8 s condition from training, feature selection, tuning, calibration, and threshold selection.
- H3 threshold: choose on validation only with a prespecified finite-sample method and fixed false-alert budget. Evaluate the frozen threshold once on test data; E4 checks behavior under shift.
- Report positive-class prevalence, AUPRC, AUROC, precision, recall, F1, Brier score/calibration, confidence intervals, errors missed, unnecessary alerts, abstentions, and risk-coverage as applicable. Do not rely on accuracy alone.
- Use grouped bootstrap resampling over independent ingredient groups, paired resamples for model differences, and clustered/grouped analyses when appropriate. Do not treat clips as independent sample units. Apply Holm correction across confirmatory H1–H4; E1–E4 remain exploratory.
- Preserve original ASR outputs and record recognizer/model version and decoding configuration. Experimental labels are derived from reference comparisons and must not leak into predictor inputs.
- No test-set inspection for tuning. Freeze feature definitions, model configuration, calibration and thresholds before final test evaluation.

## Scope and interpretation

Results will characterize this controlled synthetic benchmark only. Synthetic voices do not represent the full variability of real clinical speech. No result will be described as proving reduced patient harm, real-hospital safety, or clinical effectiveness. Novelty is provisional pending literature review. Feasibility is pending the pilot; if errors or independent groups are insufficient, revise the question or narrow claims rather than manufacture errors or weaken splits without explanation.

## Change control

This file is frozen by its initial Git commit. Any later change must be made in a new, dated amendment that explains the reason and whether test data have been accessed; retain the original text and commit history.
