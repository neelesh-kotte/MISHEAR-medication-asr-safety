# ASR and feature extraction: human-corpus protocol

The prior Phase 3 pair-feature design is retained only as historical project material. The active problem is utterance/mention-level medication error prioritization on human speech.

ASR inference must be label-blind. The recognizer receives audio and frozen decoding settings. Its transcript, token/span confidence, N-best hypotheses, and independently generated recognizer output may enter the auditor. Ground-truth medication identity, timestamps, correction labels, and reference-specific candidate lists may enter evaluation only after risk scoring.

Use `src/asr_inference.py` for local model wrappers, `src/baselines.py` for confidence, alternative uncertainty, disagreement, and vocabulary anomaly scores, and `src/auditor.py` for audio-aware support comparisons. Record corpus release, exact checkpoint/revision, decoding configuration, sample rate, and output hashes. See [`DATA_AND_LICENSING.md`](DATA_AND_LICENSING.md) before downloading or using a checkpoint.
