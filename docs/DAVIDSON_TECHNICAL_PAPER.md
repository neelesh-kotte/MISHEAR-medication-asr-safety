# MISHEAR: Cross-Corpus Medication Transcription Auditing on Existing Human-Speech Datasets

**Research protocol draft — empirical cross-corpus results not yet available**

## Abstract

Medication-name errors in automatic speech recognition can be difficult to prioritize from general ASR confidence alone. MISHEAR will test whether an audio-aware review-priority auditor improves medication-error capture over confidence, recognizer disagreement, lexical/phonetic anomaly, N-best uncertainty, and a combined text-side baseline on held-out speakers and medication ingredients, and whether performance transfers to PriMock57. The planned corpora are AfriSpeech-200 and PriMock57. The previous synthetic-speech pilot is superseded and is not evidence for this study. Corpus files have not been inventoried in the present workspace; therefore this document specifies the protocol and reports no new corpus counts or empirical model scores.

## 1. Research question and contribution

The central question is whether audio evidence helps rank existing human-speech medication transcription errors above strong transcript-side baselines under speaker/ingredient shift and external-corpus transfer. The system is an auditor that prioritizes utterances for review. It does not correct transcripts or make clinical recommendations. The evaluation measures error concentration within offline review budgets; it does not estimate workflow savings or patient outcomes.

## 2. Corpora and data limitations

### AfriSpeech-200

AfriSpeech-200 provides accented English read-sentence speech with official train/dev/test partitions. It is distributed under **CC BY-NC-SA 4.0**. Preserve attribution and share-alike/noncommercial conditions when applicable. Its read sentence distributions differ from spontaneous or mock-conversation speech, and medication-name coverage may be sparse. The official partitions will be used for the unseen-speaker evaluation after metadata disjointness checks.

### PriMock57

PriMock57 contains 57 mock primary-care consultations, human utterance-level transcripts, and associated consultation material. The upstream repository supplies the corpus under **CC BY 4.0**. It provides a conversational external-transfer setting that differs from AfriSpeech read sentences. Consultation integrity is mandatory: all utterances from one consultation stay in a single partition. Only necessary audio/transcript fields will be processed; notes and unrelated human evaluation annotations are out of scope.

Dataset terms and upstream descriptions are tracked in [`DATA_AND_LICENSING.md`](DATA_AND_LICENSING.md). Exact release/commit, download date, checksums, corpus manifests, and applicable licenses must be recorded before analysis. Audio and full transcripts are not committed to this repository.

## 3. Medication vocabulary and inventory

The medication inventory stage scans transcripts against a versioned vocabulary that maps spelling variants and brand/generic names to resolved ingredient identifiers. Combination products retain all component ingredient IDs. Ambiguous names and unresolved mention annotations are explicitly reported. The inventory report is a feasibility check, not an ASR performance result. Training stops if labels or ingredient identities cannot be resolved reliably, if speaker/corpus partition metadata is absent, or if the development sample lacks both correct and erroneous mentions.

## 4. Blinded architecture

At inference, all systems receive only audio, ASR transcript, ASR token/span confidence, N-best outputs, outputs from independent recognizers, and the general authorized vocabulary. They do **not** receive the reference medication identity, reference-derived timestamps, recording-specific gold candidates, or correctness labels. Reference annotations are joined after the risk scores are frozen.

Baselines are (1) ASR token/span confidence; (2) N-best entropy or score margin; (3) recognizer disagreement; (4) lexical/phonetic anomaly to the closest vocabulary term; and (5) a combined classifier over these transcript-side signals. The audio-aware auditor identifies ASR-side candidate spans, retrieves plausible alternatives from the general vocabulary, compares candidate acoustic support (for example, forced-alignment or acoustic-phonetic scores), and combines the evidence into a review-priority score. Candidate alternatives are not replacements, and the interface has no automatic correction output.

## 5. Evaluation design

**Evaluation A (unseen speakers):** use AfriSpeech-200 official speaker-disjoint train/dev/test sets, verify no speaker overlap, fit and select settings on train/dev, and evaluate once on test.

**Evaluation B (unseen ingredients):** freeze a list of held-out ingredients before model development. Exclude all utterances mentioning those ingredients from supervised training and tuning. Any utterance containing both held-out and non-held-out medications is assigned wholly to test (or excluded under a separately prespecified pure-mention policy).

**External transfer:** fit and freeze on AfriSpeech development data, then score PriMock57 without tuning on PriMock57 outcomes. Consultation groups are never divided across partitions.

The error label and alignment subtype are computed only after inference. Alignment distinguishes correct mentions, paired/other medication substitutions, non-drug renderings, deletions, and insertions/hallucinations. Adjudication rules and normalization are frozen before test scoring.

## 6. Primary endpoints and uncertainty

For budgets (b in \{1\%,5\%,10\%,20\%\}\), select the top-risk utterances and calculate:

\[
\text{Error Capture}(b)=\frac{\text{medication errors in selected utterances}}{\text{all medication errors}}.
\]

Also report the flagged utterance precision, medication false-alert rate (incorrectly flagged correct mentions / all correctly recognized medication mentions), number of missed errors, number of flagged utterances, and flagged audio duration. Plot error capture against review budget for each baseline and auditor. Include a clearly labeled oracle upper bound that selects utterances using gold error labels at the same budget; it is a theoretical ceiling, not a system result.

Uncertainty estimates resample at speaker and ingredient levels. Repeated utterances, medication mentions, or segments from the same group are not treated as independent. Comparisons are paired on the same evaluation examples and partitions. Report prevalence and denominators beside each metric; do not use accuracy alone.

## 7. Feasibility and current status

The inventory/feasibility report must precede heavy model execution. At this draft revision, the repository contains no AfriSpeech-200 or PriMock57 corpus manifest or audio, and the existing medication CSV is a small unversioned pilot list. Accordingly, no mention counts, WER, error-capture values, baseline comparisons, confidence intervals, or transfer results are available. Existing synthetic-pilot numbers are not carried forward as evidence for this human-speech protocol.

## 8. Limitations and interpretation

AfriSpeech read sentences and PriMock57 mock consultations represent different speech distributions. PriMock57 has a finite number of consultations, while medication mentions may be rare and concentrated among speakers or ingredients. Dataset annotations can miss spoken medication mentions or normalize names inconsistently. A vocabulary-only exact scan is insufficient as a definitive gold-label process. Results, if obtained, support only offline error prioritization on these licensed corpora. They do not demonstrate reduced workload, prevented harm, clinical effectiveness, or readiness for clinical use.

## References

1. Olatunji et al. “AfriSpeech-200: Pan-African Accented Speech Dataset for Clinical and General Domain ASR.” TACL 2023. [Paper](https://aclanthology.org/2023.tacl-1.93/); [dataset card and license](https://huggingface.co/datasets/intronhealth/afrispeech-200).
2. Papadopoulos Korfiatis et al. “PriMock57: A Dataset of Primary Care Mock Consultations.” ACL 2022. [Paper](https://arxiv.org/abs/2204.00333); [upstream repository and license](https://github.com/babylonhealth/primock57).
3. U.S. Food and Drug Administration. [Phonetic and Orthographic Computer Analysis (POCA) Program](https://www.fda.gov/drugs/resources-drugs/phonetic-and-orthographic-computer-analysis-poca-program).
