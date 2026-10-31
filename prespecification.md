# MISHEAR cross-corpus study prespecification

**Protocol status:** pivoted protocol; confirmatory thresholds, corpus manifests, and medication adjudication remain to be frozen before outcome evaluation.

## Research question

Can an audio-aware auditor identify medication transcription errors better than ASR token/span confidence and recognizer-disagreement baselines on held-out speakers and ingredients, and does the advantage transfer to PriMock57?

## Data sources and scope

- **AfriSpeech-200:** existing human speech with official speaker-disjoint train/dev/test partitions; corpus license CC BY-NC-SA 4.0. It contains read sentence distributions and is not mock conversation.
- **PriMock57:** 57 mock primary-care consultations with human utterance-level transcripts; upstream repository license CC BY 4.0. It is the external conversational transfer corpus. The entire consultation is the grouping unit.
- Medication vocabulary must be versioned, alias-aware, and resolve brand/generic and combination products to ingredient identifiers. The checked-in schema-v1.0 development vocabulary is mapped to RxNorm IDs, but corpus coverage and reference labels still require validation before confirmatory use.
- Do not place corpus audio, restricted labels, or identifying metadata in Git.

## Outcome and blinding

Gold reference-to-ASR alignment labels each medication mention/utterance as correct or erroneous and records error subtype. Gold identity, reference-derived timestamps, gold candidate lists, and correctness labels are evaluation-only. Auditor and baseline prediction interfaces accept audio, ASR transcript, ASR confidence/N-best, independent ASR output, and the generally authorized vocabulary subset only. Candidate retrieval must be generated from ASR-side text and the general vocabulary, never from the known reference.

## Partitions

1. **Evaluation A — unseen speakers:** preserve the official AfriSpeech-200 train/dev/test partitions. Verify speaker IDs do not overlap; train/tune on official train/dev and evaluate once on official test.
2. **Evaluation B — unseen ingredients:** freeze selected test ingredients before model fitting. No utterance containing a selected ingredient may enter supervised fitting, feature selection, or tuning. For multi-ingredient utterances, any mention of a held ingredient sends the whole utterance to test (or it is excluded under a separately frozen strict-pure policy); never split its labels across partitions.
3. **External transfer — PriMock57:** train and freeze the auditor using AfriSpeech development data, then evaluate on the separately preprocessed PriMock57 corpus. If PriMock57 is partitioned for any development analysis, all utterances from one consultation remain together. Do not tune on external test outcomes.

## Systems and baselines

- Token/span confidence from the primary ASR.
- N-best entropy or top-two score margin.
- Transcript disagreement between independent recognizers/configurations.
- Lexical/phonetic anomaly relative to the authorized medication vocabulary.
- Combined text-side baseline classifier.
- Audio-aware auditor: ASR-side mention spans, general vocabulary retrieval, audio support comparison for retrieved candidates, confidence and disagreement features, and a review-priority score. It never produces an automatic correction.

## Primary evaluation

Rank utterances by risk and report error capture at review budgets 1%, 5%, 10%, and 20%: medication errors in selected utterances divided by all medication errors. Report the review subset size, precision among flagged utterances, errors missed, flagged audio duration, false-alert rate, and the error-capture curve. The false-alert denominator is correctly recognized medication mentions; the numerator is correct medication mentions in flagged utterances. Provide a labeled oracle ceiling at the same review budget, computed using gold labels only for an upper bound.

Use speaker- and ingredient-cluster-aware uncertainty estimates. Never treat repeated utterance segments or augmentations as independent observations. Threshold or budget definitions and ingredient holdouts are frozen before test evaluation. Do not claim workload reduction, harm prevention, clinical effectiveness, or patient safety improvement from these offline prioritization measures.

## Feasibility stop rules

Run `src.data_inventory` first. Stop model training if corpus paths/licenses are not verified, medication annotations cannot be resolved to a versioned vocabulary, official speaker separation cannot be validated, the development set lacks both outcome classes, or held-out test ingredients are not represented. Report insufficient sample size rather than altering partitions after test inspection.

## Deviations and prior work

The previous synthetic TTS/DSP protocol, its results, and pair-level H1–H4 analyses are superseded for the primary study. Historical artifacts may remain for provenance, but must not be presented as evidence for the cross-corpus human-speech question.
