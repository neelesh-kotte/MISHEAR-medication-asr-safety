# MISHEAR: Cross-Corpus Medication Transcription Auditing on Existing Human-Speech Datasets

**Manuscript status (October 30, 2026):** protocol and implementation report; empirical sections are pending real-corpus access, annotation, and execution. This manuscript does not report synthetic smoke-test metrics as human-speech results.

## Abstract

Medication-name transcription errors are difficult to detect from aggregate word error rate alone, while an automatic post-corrector can alter a transcript without reliable evidence. MISHEAR specifies an audio-aware auditor that ranks medication-containing utterances for offline review and compares it with ASR confidence, N-best uncertainty, recognizer disagreement, lexical/phonetic anomaly, and a combined text-side classifier. The proposed evaluation uses AfriSpeech-200 for speaker-disjoint and ingredient-held-out tests and PriMock57 as a consultation-grouped external transfer corpus. The auditor is blinded to reference medication identities, reference timestamps, and correctness labels at inference. Outcomes are error capture at fixed review budgets and a mention-based false-alert rate, with speaker- and consultation-clustered uncertainty estimates. The vocabulary has been mapped to a versioned RxNorm-based schema, and local synthetic fixtures verify inventory, splitting, evaluation, and figure generation. **No authorized real-corpus manifests or audio are present in the workspace, and no human-speech model evaluation has been run.** Consequently, no comparative performance or clinical-effect estimate is reported.

## 1. Introduction

Speech recognition errors involving medication names can be consequential because medication strings are specialized, acoustically confusable, and often uncommon in general ASR training text. Aggregate WER can obscure this subset: an utterance may be mostly correct while its medication mention is wrong. Confidence scores also need not identify the particular medication errors that warrant attention.

MISHEAR asks whether audio-aware evidence can rank medication transcription errors more effectively than strong ASR-side baselines on held-out speakers and ingredients, and whether such ranking transfers to a separate mock-conversation corpus. The system is designed to produce a review-priority score; it does not automatically replace medication text. Evaluation is offline and measures error concentration in selected review subsets. It does not directly measure clinical safety, workflow effects, or patient outcomes.

The two planned corpora provide different distribution tests. AfriSpeech-200 contains prompted read sentences, while PriMock57 contains mock primary-care conversations. Neither is a substitute for a prospective study of actual clinical deployment. The distinction between read and conversational speech is part of the transfer question, not a claim that the corpora are interchangeable.

## 2. Related Work

Medication-specific ASR research has used synthetic speech and recognizer adaptation to improve recognition. Work on accented clinical speech has also shown the need to evaluate medical entities separately from general transcription. These approaches motivate the medication-specific outcome but differ from an auditor that ranks errors while leaving the ASR transcript unchanged. See the primary studies summarized in the [literature review](LITERATURE_REVIEW.md), including SynthASR and AfriSpeech-200.

Entity correction and retrieval methods use lexical catalogs, phonetic features, or language models to repair ASR hypotheses. MISHEAR shares the motivation to use medication vocabulary and phonetics, but the target is risk prioritization under a review budget, not automatic correction. FDA POCA screens drug names for orthographic and phonetic similarity; it is not an ASR error probability model and is not reproduced here.

Confidence estimation and recognizer disagreement provide important baselines because they can identify uncertain ASR output without medication-specific ground truth at inference. The evaluation compares those signals against lexical/phonetic retrieval and audio support under the same held-out conditions. The provisional research gap and closest-work matrix are documented in [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md); no first-of-its-kind claim is made.

## 3. Methodology

### 3.1 Blinded auditor

At inference, the auditor may receive an audio segment, ASR transcript, ASR token probabilities or log-probabilities, N-best alternatives, a second recognizer's output, and a general authorized vocabulary. It must not receive reference medication identity, reference-derived timestamps, recording-specific gold candidate lists, or correctness labels. Gold alignment and error labels are joined only after risk scores are frozen for evaluation.

The implemented feature interface represents four evidence families:

1. **ASR uncertainty:** token/span confidence and entropy or margin over N-best hypotheses.
2. **Recognizer disagreement:** normalized word and character distance; optional semantic distance when a frozen embedding interface is supplied.
3. **Lexical/phonetic retrieval:** similarity between ASR-side spans/tokens and entries in the general vocabulary.
4. **Audio support:** candidate CTC sequence support or a compatible acoustic scorer over the supplied waveform segment.

The combined risk model is a regularized logistic classifier by default, with a gradient-boosted option. It is fit on training data only and calibrated on a development split. The test and external sets are not fitting inputs. The score ranks utterances for review and does not propose or apply transcript corrections. These interfaces and modules exist in the repository; they have not yet been run on authorized corpus audio.

### 3.2 Vocabulary

The vocabulary CSV follows schema v1.0. Single-ingredient rows use a source-mapped INN and RxNorm ingredient RxCUI; the combination example stores each ingredient component separately. Brand names and aliases map to the row's ingredient identifier, while combination-product aliases map to all listed components. Metaphone and Soundex fields are deterministic retrieval keys, not evidence of clinical interchangeability. The release and source snapshot are recorded in `data/vocabulary/medication_vocabulary.metadata.yaml`; see [VOCABULARY_SCHEMA.md](VOCABULARY_SCHEMA.md).

Before a production study, the vocabulary release should be rechecked against the [WHO INN lists](https://www.who.int/teams/health-product-and-policy-standards/inn/inn-lists) and [NLM RxNorm releases](https://www.nlm.nih.gov/research/umls/licensedcontent/rxnormfiles.html), and its redistribution terms should be reviewed. Vocabulary resolution does not itself label a corpus transcript as a correct or incorrect medication mention.

### 3.3 Error labels and blinding audit

Reference transcripts are used after inference to locate medication mentions, align the ASR hypothesis, and label correct mentions versus transcription errors. Ambiguous, fragmented, or multi-word renderings require a documented adjudication policy. Reference-derived identity and timing information must not be copied into feature manifests. Every prediction artifact should preserve split IDs and a provenance record sufficient to audit the separation between feature creation and outcome joining.

## 4. Experimental Setup

### 4.1 Corpora and partitions

| Corpus | Planned role | Partition rule | License and speech-style limitation | Local execution state |
|---|---|---|---|---|
| AfriSpeech-200 | Primary held-out-speaker and held-out-ingredient evaluation | Preserve upstream train/dev/test; verify speaker disjointness. Freeze ingredient holdouts before fitting; any multi-ingredient utterance containing a held-out ingredient is kept out of supervised training/tuning. | CC BY-NC-SA 4.0 per upstream card. Prompted read sentences are not mock conversation. | No local manifest/audio found; access acceptance and immutable commit are not confirmed. |
| PriMock57 | External transfer to mock-conversation speech | Group by complete consultation; never distribute utterances from one consultation across partitions. | CC BY 4.0 per upstream repository. Mock consultation speech is not actual patient-care audio. | No local manifest/audio found; exact upstream commit is not pinned. |

The operational access and provenance steps are in [ACCESS_AND_REPRODUCIBILITY.md](ACCESS_AND_REPRODUCIBILITY.md). Corpus downloads, source revisions, and local annotations are not checked into this workspace.

### 4.2 Systems and ablations

The planned comparisons are ASR confidence, N-best uncertainty, recognizer disagreement, lexical/phonetic anomaly, a combined text-only classifier, and the audio-aware auditor. Ablations remove audio evidence, disagreement, or phonetic retrieval context in turn. Training and calibration use development partitions only; test outcomes are reserved for final scoring. PriMock57 is external transfer and cannot be used for tuning.

### 4.3 Review-budget outcomes

At budget \(b\), the system selects the top-ranked \(b\) fraction of utterances. Primary error capture is medication errors within the selected subset divided by all medication errors. The mention-based false-alert rate is

\[
\mathrm{FAR}=\frac{\text{correct medication mentions in flagged utterances}}
{\text{all correctly recognized medication mentions}}.
\]

Report precision among flagged utterances, missed errors, flagged audio duration, and an oracle upper bound at the same utterance budget. The oracle uses gold outcomes to rank items and is explicitly not a realizable model. Confidence intervals should resample at speaker and consultation levels; the bootstrap procedure and seed must be reported. These quantities describe error concentration and prioritization only.

## 5. Results

### 5.1 Feasibility and data provenance

The real-data inventory was run against `config/corpus_config.yaml`. It returned `STOP_BEFORE_TRAINING`, `training_authorized: false`, and zero manifest records scanned for both planned corpora. The reasons include missing local manifests, missing a verified PriMock57 40-character source commit, unconfirmed AfriSpeech access terms, and minimum sample counts not met. The current vocabulary validates as schema v1.0 with 10 distinct ingredient IDs and no unresolved rows, but vocabulary readiness does not substitute for corpus readiness. The machine-readable inventory is [results/human_speech/inventory.json](../results/human_speech/inventory.json).

The `config/stubs/` manifests are synthetic fixtures. They verify software mechanics and are excluded from every empirical claim and table below. The stub run's `training_authorized` value is false by design.

### 5.2 Table 1 — real corpus inventory

| Corpus | Local records scanned | Medication utterances | Medication mentions | Distinct resolved ingredients | Speakers | Consultations | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| AfriSpeech-200 | 0 (manifest absent) | Not measured | Not measured | Not measured | Not measured | Not measured | STOP_BEFORE_TRAINING |
| PriMock57 | 0 (manifest absent) | Not measured | Not measured | Not measured | Not measured | Not measured | STOP_BEFORE_TRAINING |

These zeros describe the number of local manifest rows available to this run, not corpus sizes or corpus-level medication prevalence.

### 5.3 Table 2 — ablation matrix

| System / ablation | AfriSpeech unseen speakers | AfriSpeech unseen ingredients | PriMock57 transfer |
|---|---|---|---|
| Full audio-aware auditor | Not run | Not run | Not run |
| Remove audio evidence | Not run | Not run | Not run |
| Remove recognizer disagreement | Not run | Not run | Not run |
| Remove phonetic retrieval | Not run | Not run | Not run |

No real-data scores or cluster-aware intervals are available. The generated ablation matrix under `results/stub_dry_run/` uses only synthetic fixture labels and is not this table's evidence.

### 5.4 Figure 1 — error capture by review budget

**Not available for real human speech.** No real prediction scores or labels exist in the workspace, so real AfriSpeech or PriMock57 curves cannot be plotted. A synthetic smoke-test figure exists under `results/stub_dry_run/paper_artifacts/figure1_error_capture.png`; it is labeled synthetic and excluded from this manuscript.

### 5.5 Leakage and significance audit

Real-data speaker/ingredient leakage checks, model fitting, held-out evaluation, clustered bootstrap intervals, and ablation comparisons have not been executed because the corpora and adjudicated medication-error labels are absent. Therefore this manuscript makes no claim that leakage was excluded in an empirical run and reports no p-values, confidence intervals, AUPRC, AUROC, or error-capture estimates.

## 6. Discussion

The current evidence supports a reproducible software protocol and a passing synthetic code-path smoke test, not an empirical answer to the research question. The central unresolved dependency is data availability and annotation: the public corpus transcripts do not automatically provide adjudicated medication mention identities or ASR correctness labels. A verified manifest and a blinded annotation/adjudication process are needed before any risk model can be trained or evaluated.

AfriSpeech-200 and PriMock57 also differ in speech style, recording conditions, participant roles, and conversational structure. A transfer result would therefore be specific to these two datasets and would not establish generalization to clinical encounters. ASR checkpoint provenance, training-data overlap, license compatibility, and recognizer independence must be documented for each model. If the recognizers share pretraining data or architecture, disagreement may be less independent than intended.

The review budget is an offline prioritization proxy. Flagged audio duration and error capture quantify how errors are concentrated in selected utterances; they do not directly quantify reviewer effort, workload reduction, harm prevention, or clinical benefit. Those claims would require separate prospective workflow and outcome studies.

## 7. Limitations and required completion before submission

1. No real AfriSpeech-200 or PriMock57 manifest/audio is present; access terms and corpus commits are not confirmed.
2. No reference medication annotation/adjudication set or error-labeled ASR predictions have been generated.
3. No production ASR feature extraction, model fitting, held-out scoring, or real-corpus ablation was run by `run_pipeline.py --stage all`; that command is a synthetic metadata/artifact smoke test only.
4. The current RxNorm-mapped development vocabulary is small and cannot establish medication coverage in either corpus.
5. The literature review is preliminary rather than systematic, and the nearest-work comparison must be updated before a novelty claim.
6. Existing human-speech corpora do not establish actual clinical deployment safety or patient outcomes.

## 8. Conclusion and future work

MISHEAR defines a blinded, audio-aware medication-error prioritization study with confidence, disagreement, vocabulary, and acoustic evidence; ingredient- and speaker-aware evaluation; and an external mock-conversation transfer test. The repository now provides schema validation, inventory and split utilities, scoring/evaluation modules, and artifact generation with synthetic fixtures. The empirical study remains blocked. Next steps are to accept and record AfriSpeech access, obtain and pin both corpora, build and adjudicate medication-reference labels, freeze splits and recognizer checkpoints, run the prespecified comparisons, and replace each “Not run” entry only with traceable real-data outputs.

## References and data documentation

- AfriSpeech-200 dataset card and official split description: [Hugging Face dataset card](https://huggingface.co/datasets/intronhealth/afrispeech-200); [Olatunji et al., TACL 2023](https://aclanthology.org/2023.tacl-1.93/).
- PriMock57 repository, data description, and LFS instructions: [Babylon Health repository](https://github.com/babylonhealth/primock57); license: [LICENSE.md](https://github.com/babylonhealth/primock57/blob/main/LICENSE.md).
- WHO International Nonproprietary Names: [WHO INN lists](https://www.who.int/teams/health-product-and-policy-standards/inn/inn-lists).
- RxNorm identifiers and release provenance: [NLM RxNorm release files](https://www.nlm.nih.gov/research/umls/licensedcontent/rxnormfiles.html); [NLM RxNorm FAQ](https://www.nlm.nih.gov/research/umls/rxnorm/faq.html).
- Related-work references and provisional gap: [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md).
