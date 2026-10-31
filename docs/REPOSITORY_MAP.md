# Repository map and build order

## Blueprint directories

- `config/`: dataset, ASR, and experiment configuration.
- `data/`: public vocabulary, selected pairs, licensed human-risk references, fictional transcripts, audio metadata, and frozen splits. Keep large audio/model artifacts out of Git unless redistribution terms permit it.
- `src/`: pipeline modules below.
- `experiments/`: pilot, characterization, generalization, ablation/SHAP, context, auditing, cross-recognizer, and overlap analyses.
- `results/`: generated tables, figures, and run logs.
- `paper/`: literature, research gap, methods, results, limitations, and manuscript.
- `tests/`: data leakage, feature, alignment, DSP measurement, and metric integrity checks (write these before analysis code, as the plan directs).

## Module dependency order

1. `vocabulary.py` → RxNorm terms/ingredient identity and variants.
2. `pair_generation.py` → documented candidate/control pair sampling, graph, seed.
3. `pronunciation_features.py`, `lexical_features.py` → pair features.
4. `audio_generation.py` → fictional matched sentences and licensed TTS metadata.
5. `dsp_augmentation.py`, `acoustic_features.py` → transformations and measured SNR/RT60/quality.
6. `asr_runner.py` → Whisper and Wav2Vec2, frozen decoding configs and raw outputs.
7. `error_alignment.py` → medication substitution/deletion/insertion labels and WER.
8. `context_features.py` → context compatibility features.
9. `pair_risk_model.py` → baselines and MACP pair-level model.
10. `calibration.py`, `recording_auditor.py`, `selective_prediction.py` → recording-level score, validation calibration and threshold, abstention/flag output.
11. `explainability.py`, `human_risk_overlap.py`, `statistical_analysis.py` → prespecified analyses.

The blueprint’s initial source list also names `acoustic_features.py`; this repository map includes it even though the user’s examples did not mention it. Build stubs only after Phase 1 confirms data sources, licenses, TTS/ASR feasibility, and pilot design.
