# MISHEAR: Predicting and Selectively Detecting Medication-Name Errors in Synthetic Speech Recognition

**Technical research draft — descriptive pilot completed; confirmatory models not estimable**

**Project:** MISHEAR v2

**Prespecification:** [`prespecification.md`](../prespecification.md), frozen October 30, 2026

**Current artifact audit:** October 30, 2026

> **Results integrity note.** The initial `say` attempt produced zero-frame audio and was discarded. A later pilot using eSpeak NG generated 40 nonempty utterance-condition records and completed primary Whisper alignment. The pair-risk classifier could not fit because there were zero positive pair-confusion labels. Consequently, the reported WER values below are descriptive results over the 40 generated recordings, not held-out model results. H1/H2/H4 AUPRC, Brier scores, and McNemar p-values remain unavailable. The small vocabulary, TTS change, and inspection of pilot labels prevent treating this run as a confirmatory validation.

## Abstract

Medication-name recognition errors can be obscured by aggregate speech-recognition metrics, while similarity screening alone does not estimate whether a particular recognizer will confuse a particular pair under a particular acoustic condition. MISHEAR separates pair-level confusion prediction from transcript-level selective auditing. In a small synthetic pilot, eight medication names formed four candidate pairs; eSpeak NG generated fictional utterances, five DSP conditions were applied, and OpenAI Whisper transcribed 40 utterance-condition recordings. Descriptive WER ranged from 0.9375 to 1.8750, with overall WER 1.2563. No pair-confusion positives occurred, so H1 classifier fitting stopped and H2 calibration/H4 transfer did not run. The pilot therefore provides acoustic-condition error measurements but no evidence for model predictive performance or statistical significance. It does not establish performance on real clinical speech or a reduction in patient harm.

**Keywords:** medication names; automatic speech recognition; synthetic speech; selective prediction; phonetic similarity; calibration; robustness

## 1. Introduction

Medication names are specialized lexical items. A recognizer can produce a fluent sentence while substituting, deleting, or inserting a medication term. Aggregate word error rate (WER) is useful for general transcription quality, but a medication-specific outcome is needed to characterize errors involving drug names. Prior work has explored synthetic speech for medication recognition (Fazel et al., [SynthASR](https://www.isca-archive.org/interspeech_2021/fazel21_interspeech.pdf)), medical-entity recognition across accents (Afonja et al., [Interspeech 2024](https://www.isca-archive.org/interspeech_2024/afonja24_interspeech.pdf)), and phoneme-aware named-entity correction (Wang et al., [Interspeech 2020](https://www.isca-archive.org/interspeech_2020/wang20p_interspeech.pdf)). The FDA's [POCA](https://www.fda.gov/drugs/resources-drugs/phonetic-and-orthographic-computer-analysis-poca-program) screens drug names for orthographic and phonetic similarity, but such similarity is not a measured probability that a particular ASR system will substitute one name for another.

MISHEAR asks whether a model can predict medication-pair substitutions under controlled synthetic conditions and whether an auditor can selectively flag suspected errors without automatically rewriting a transcript. The distinction matters: pair-level vulnerability estimates whether two names are confused under a condition; recording-level auditing evaluates one recognizer output for a specific utterance. Neither task is a clinical decision system.

The research gap is provisional. The initial comparison and its limitations are documented in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md). Novelty should not be claimed as established until the search is expanded and the empirical study has valid outputs.

## 2. Prespecified questions and hypotheses

The project prespecification defines four confirmatory hypotheses:

| Hypothesis | Question | Prespecified primary endpoint |
|---|---|---|
| H1 | Does the full phonetic, lexical, contextual, and acoustic model generalize to held-out active ingredients better than a phonetic-only baseline? | Test AUPRC and paired grouped-bootstrap AUPRC difference on ingredient-disjoint test data. |
| H2 | Does predictive performance remain above the positive-class prevalence baseline on unseen acoustic conditions? | Held-out-condition AUPRC compared with test prevalence. |
| H3 | Does selective auditing detect more medication errors than a similarity-only alert at the same false-alert budget? | Error detection rate at the validation-selected budget, evaluated once on test data. |
| H4 | Does the predictive approach retain value under a second recognizer? | AUPRC on ingredient-held-out data transcribed by the second recognizer. |

The prespecification calls for Holm correction across H1–H4, grouped bootstrapping over ingredient components, calibration comparison using validation data only, and no test-set tuning. The numerical H3 false-alert budget is explicitly not finalized in the prespecification; five alerts per 100 mentions is an example, not a selected value. The current `config/pipeline_config.yaml` still sets the budget to `null`, so H3 is not authorized for confirmatory test evaluation until this is resolved and recorded before test inspection.

Exploratory questions E1–E4 cover contextual sentence effects, overlap with public human look-alike/sound-alike references, recognizer disagreement, and false-alert behavior under shift. They do not replace failed confirmatory results.

## 3. Dataset creation protocol

### 3.1 Medication vocabulary and candidate pairs

The planned vocabulary uses public medication terminology with names mapped to active-ingredient identifiers. Pair generation filters within-ingredient variants, then ranks distinct-ingredient candidates using Jaro–Winkler and a transparent Soundex-based string proxy. The proxy is not a pronunciation dictionary and does not reproduce FDA POCA. Pronunciation-based ARPAbet features are intended to use CMUdict where names are covered; unavailable pronunciations remain missing or are separately marked if a G2P provider is used.

The data file currently present at `data/vocabulary/medication_vocabulary.csv` has eight rows and columns `drug_name, ingredient, phonetic_key, category`. It does not provide resolved RxNorm/ingredient IDs, and its `drug_name` header is not among the current CSV loader's accepted name aliases. It therefore does not establish a valid candidate vocabulary for the configured pipeline. No production vocabulary size or candidate-pair count can be reported from the current artifacts.

### 3.2 Fictional utterances and synthesis

Each selected pair endpoint was placed in the fictional carrier “Please say {drug} now.” at a configured speaking rate of 175 words per minute. macOS `say` returned successfully but generated zero-sample output on this machine. The run therefore switched to the locally installed eSpeak NG 1.52.0 voice `en-us`, which produced nonempty audio resampled to 16 kHz. This differs from the initial Samantha configuration and must be treated as a disclosed pilot protocol change. eSpeak NG is distributed under GPL-3.0-or-later; review the [upstream license](https://github.com/espeak-ng/espeak-ng) before redistributing software or generated assets. The run generated eight medication endpoints from four candidate pairs.

### 3.3 DSP conditions and measurement

The configured grid comprises clean audio, babble at target SNR 15 dB, babble at 10 dB with target RT60 0.4 s, babble at 5 dB with target RT60 0.8 s, and clipping at normalized threshold 0.75. Babble is assembled from at least two other generated utterances, not from the target utterance. Synthetic reverberation uses a seeded exponential-decay impulse response; it is a controllable approximation, not a measured hospital room response.

The implementation records component SNR using active speech samples and measures RT60 from the generated impulse response using a Schroeder energy-decay fit. Across the eight recordings per condition, measured mean SNR was 15.00 dB for the 15 dB babble setting, 10.00 dB for the 10 dB plus 0.4 s RT60 setting, and 5.00 dB for the 5 dB plus 0.8 s RT60 setting. Mean measured RT60 was 0.3973 s and 0.7993 s for those reverberant conditions, respectively. Clean and clipping conditions have no additive-noise SNR or RIR RT60 measurement. The clipping threshold was configured to 0.75; the run metadata contains measured clipping fractions.

### 3.4 ASR and label construction

The primary recognizer was OpenAI Whisper Python `base` on CPU. Forty augmented recordings were transcribed and aligned. The secondary recognizer was configured as Wav2Vec2 CTC (`facebook/wav2vec2-base-960h`) but did not run because the pipeline stopped at classifier fitting. Model revision should be pinned before a new run.

Reference text and intended medication labels are attached only after inference. Word alignment produces WER and medication event labels (`correct`, `substitution`, `deletion`, and insertion counts). The pair-confusion target is positive when the hypothesis replaces the intended name with the other endpoint of that candidate pair; all other outcomes are negative for this pair-specific task. The complete medication event labels remain available for broader error-rate analysis.

## 4. Model architecture and evaluation

The prespecified main classifier is regularized logistic regression. Its feature allowlist groups phonetic features (including phoneme similarity where available and the Soundex proxy), lexical similarity, optional contextual compatibility, measured acoustic quantities, and recognizer metadata. Numeric imputation/standardization and categorical levels are fit on training data only. Transcript strings, reference strings, target labels, recording IDs, ingredient IDs, and requested DSP settings are excluded from predictor inputs. Baselines include phonetic-only logistic regression, similarity ranking, and prevalence.

H1 partitions the medication-pair graph into ingredient-connected components and assigns whole components to train, validation, and test. H2 reserves the configured harsh babble-plus-reverberation condition and the full clipping family, then groups development data for train/validation. Platt and isotonic calibration are fitted on validation predictions only. H4 trains on primary-recognizer development data and evaluates matched ingredient-held-out recordings transcribed by Wav2Vec2. H3 threshold selection uses a finite-sample one-sided false-positive bound on validation data only; the threshold remains unset until its false-alert budget is fixed.

Statistical summaries include AUPRC, AUROC, Brier score, precision, recall, F1, grouped bootstrap intervals, paired bootstrap deltas, and exact two-sided McNemar comparisons of paired correctness. The prespecification requires Holm adjustment across H1–H4. The current implemented report path has no reported Holm-adjusted p-values; any paper claiming confirmatory significance must calculate and present the prespecified correction. Clip-level independence is not assumed; uncertainty should be resampled over independent ingredient-pair components.

## 5. Empirical results: current artifact audit

The following table reflects the files currently found in `results/phase6/`, not an intended or simulated outcome.

| Outcome | Current evidence | Reportable empirical estimate? |
|---|---|---|
| Condition | n | Mean measured SNR (dB) | Mean measured RT60 (s) | S / D / I | WER | Medication substitutions / deletions |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 8 | — | — | 16 / 2 / 12 | 0.9375 | 8 / 0 |
| Babble, target 15 dB | 8 | 15.00 | — | 17 / 0 / 21 | 1.1875 | 8 / 0 |
| Babble + reverb, target 10 dB / 0.4 s | 8 | 10.00 | 0.3973 | 17 / 1 / 25 | 1.3438 | 7 / 1 |
| Babble + reverb, target 5 dB / 0.8 s | 8 | 5.00 | 0.7993 | 27 / 3 / 30 | 1.8750 | 7 / 1 |
| Clipping threshold 0.75 | 8 | — | — | 18 / 0 / 12 | 0.9375 | 8 / 0 |
| **Overall descriptive pilot** | **40** | — | — | **95 / 6 / 100** | **1.2563** | **38 / 2** |

The per-condition rows are aggregated from the aligned utterance records by acoustic condition. WER exceeds 1.0 in three conditions because insertions are included in the WER numerator. These scores reflect this small eSpeak NG/Whisper synthetic pilot and should not be generalized to other voices or clinical speech.

| Model outcome | Current result |
|---|---|
| Pair-confusion positive labels | 0 of 40; 38 alignments were substitutions, but none was an exact substitution to the other endpoint of its generated pair. Two were deletions. |
| H1 AUPRC / AUROC / Brier | Not estimable; training labels lacked positive examples, so logistic fitting stopped. |
| H2 AUPRC / calibration / Brier | Not run; pipeline stopped before H2 evaluation. |
| H4 AUPRC / AUROC / Brier | Not run; secondary Wav2Vec2 inference was not reached. |
| McNemar p-values | Not estimable; no paired classifier predictions were produced. |
| H3 selective alerting | Not evaluated; no classifier scores or validation threshold were produced. |

No statistical significance conclusion can be drawn. The output set does not support accepting or rejecting H1–H4. The pair-confusion training target had no positive cases, even though medication-level alignment recorded substitutions; these substitutions were not exact confusions with the other candidate name. This is a feasibility result and motivates a larger, better-covered vocabulary and/or a separately prespecified medication-error endpoint. The grouped test labels were inspected during diagnosis of the single-class problem; no thresholds or models were tuned from them, but this pilot should not be described as an untouched confirmatory test.

### 5.1 Reproducibility blockers identified in the current workspace

The pilot completed synthesis, DSP augmentation, primary ASR, and transcript alignment, but not pair-risk training, H2 calibration, or H4 transfer. The following corrections were required:

1. The pipeline now accepts the `drug_name` CSV column.
2. The runner now imports and dispatches to `OpenAIWhisperRunner` correctly.
3. The false-alert fraction was set to 0.05 (five alerts per 100 mentions), but selective evaluation still requires valid train/validation labels.
4. macOS `say` produced zero samples; this pilot used eSpeak NG instead. This synthesis change is recorded as a protocol amendment.

The main remaining limitation is insufficient positive pair-confusion data. The vocabulary has only eight names and four generated pairs, without validated RxNorm identifiers. A larger vocabulary and independent ingredient groups are needed before training or inferential comparison is feasible. The result package contains descriptive WER and measured DSP summaries; it does not contain classification or calibration outputs.

## 6. Discussion and safety implications

If future valid results show that pair-risk estimates generalize across ingredients, acoustic conditions, or recognizers, the immediate implication would be a candidate mechanism for prioritizing human review of uncertain medication mentions. Such a system should surface uncertainty and alternatives for a qualified reviewer; it should not automatically replace medication names. False alerts, missed errors, calibration drift, and shifts in speaker, accent, vocabulary, devices, and clinical context would require separate evaluation.

No such performance has been demonstrated by the current artifact set. Synthetic speech and algorithmically generated noise cannot establish real-world clinical validity, lower medication harm, or clinical effectiveness. This study has no patient recordings, clinical notes, or human-subject validation. A later clinical study would need appropriately governed data, expert adjudication, workflow analysis, and independent safety evaluation before any clinical deployment claim.

## 7. Limitations

The current classification evaluation is incomplete. The local vocabulary is small and lacks validated RxNorm identifiers; the model could not train because no positive pair-confusion cases occurred; H2 and H4 were not executed; and no calibration/McNemar results exist. The pilot also changed TTS backend after the configured macOS synthesizer emitted empty audio. Even after a larger dataset is assembled, a single synthetic voice, fictional carrier phrases, synthetic reverberation, and selected pretrained ASR checkpoints limit external validity. Ingredient-connected components can collapse into too few groups for stable test estimates or bootstrap intervals. Contextual language-model features remain optional and are not populated by the default pipeline. Literature novelty remains provisional pending an expanded search.

## 8. Conclusion

MISHEAR defines an experimental separation between medication-pair confusion prediction and selective transcript auditing, with ingredient-grouped, acoustic-shift, and cross-recognizer validation planned. This pilot provides descriptive synthetic ASR evidence: WER was 0.9375–1.8750 across five configured conditions, with overall WER 1.2563. It did not produce estimable H1/H2/H4 classifier results or significance tests because the pair-confusion target had no positive examples in this small vocabulary. A larger, ingredient-resolved dataset and a new prespecified run are required before claims about risk prediction, calibration, or cross-recognizer transfer can be evaluated.

## References

1. Fazel et al. “SynthASR: Unlocking Synthetic Data for Speech Recognition.” Interspeech 2021. [ISCA proceedings](https://www.isca-archive.org/interspeech_2021/fazel21_interspeech.pdf).
2. Afonja et al. “Performant ASR Models for Medical Entities in Accented Speech.” Interspeech 2024. [ISCA proceedings](https://www.isca-archive.org/interspeech_2024/afonja24_interspeech.pdf).
3. Olatunji et al. “AfriSpeech-200: Pan-African Accented Speech Dataset for Clinical and General Domain ASR.” TACL 2023. [ACL Anthology](https://aclanthology.org/2023.tacl-1.93/).
4. Wang et al. “ASR Error Correction with Augmented Transformer for Entity Retrieval.” Interspeech 2020. [ISCA proceedings](https://www.isca-archive.org/interspeech_2020/wang20p_interspeech.pdf).
5. Pusateri et al. “Retrieval-Augmented Correction of Named Entity Speech Recognition Errors.” ICASSP 2025. [Apple ML Research](https://machinelearning.apple.com/research/retrieval-asr).
6. U.S. Food and Drug Administration. [Phonetic and Orthographic Computer Analysis (POCA) Program](https://www.fda.gov/drugs/resources-drugs/phonetic-and-orthographic-computer-analysis-poca-program).
