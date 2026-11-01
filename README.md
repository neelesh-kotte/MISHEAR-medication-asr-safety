# MISHEAR v3 counterfactual engineering candidate

This source-only package is not a validated medication-error benchmark or trained auditor.
No corpus text, audio, annotations, model weights, or empirical outcome tables are included.
This preliminary source release is MIT-licensed within the scope in RELEASE_AUDIT.md.

Python 3.9+; create a local environment and install requirements-engineering.txt.
Run `python -m pytest -q`. An optional local eSpeak NG executable enables the
real synthesis integrity test; otherwise that test skips. Tests mock ASR and do
not establish model accuracy. For actual inference additionally install openai-whisper
and PyTorch, and obtain a permitted local Whisper checkpoint. No download is automatic.

Generate an exploratory pilot:

    python -m src.counterfactual_speech --spec config/counterfactual_pilot.example.json --output-dir results/pilot --voice en-gb --voice en-us --snr-db 10 --include-clean --rt60-s 0.1 --include-dry
    python -m src.validate_counterfactual_stimuli --manifest results/pilot/stimuli.jsonl --output results/pilot/validation.json
    python -m src.synthetic_quality_review --manifest results/pilot/stimuli.jsonl --events results/review/events.jsonl --port 8766

Open http://127.0.0.1:8766/ and follow docs/SYNTHETIC_REVIEW_GUIDE.md.
The general human-speech review module is included as a utility dependency only;
its neighboring-text functionality requires the full MISHEAR checkout.

After obtaining a local checkpoint:

    python -m src.run_counterfactual_asr --manifest results/pilot/stimuli.jsonl --output-dir results/inference --checkpoint /path/to/base.pt
    python -m src.counterfactual_token_evaluation --manifest results/pilot/stimuli.jsonl --predictions results/inference/predictions.jsonl --output results/token-review.json

Token alignment is a post-inference review aid. It never assigns a correctness label.
The target waveform is held identical across contexts within a voice/condition group;
pronunciation, plausibility and splice validity still require human review. One target
and one generator cannot support population/generalization claims. No model fitting,
calibration, final evaluation or clinical performance is claimed.

Source implementation and documentation are AI-assisted; owner listening observations
are separate and excluded. No independent external reproduction is claimed.
