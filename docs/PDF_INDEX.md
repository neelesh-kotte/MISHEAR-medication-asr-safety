# Blueprint index: MISHEAR v2 Complete Execution Plan

**Parsed source:** `~/Downloads/MISHEAR v2 Complete Execution Plan.pdf` (43 pages; updated October 30, 2026). This index summarizes the entire document for navigation; consult the PDF for full protocol detail.

## Research core

The blueprint proposes a synthetic, fully computational medication-ASR benchmark and MACP (Medication Acoustic Confusion Predictor), comprising two separate tasks: pair-level confusion prediction and recording-level selective auditing. It proposes H1–H4 confirmatory tests and E1–E4 exploratory analyses; details and endpoints are frozen in [`../prespecification.md`](../prespecification.md). No patient data or human participants; no clinical effectiveness claim.

## Page index

| PDF pages | Topics |
|---|---|
| 1–3 | Project framing, research question, contributions, H1–H4, E1–E4, distinction between pair and recording prediction |
| 4–8 | RxNorm vocabulary, pair selection and human-risk references, synthetic audio design, staged scale, ingredient and acoustic splits, leakage prevention |
| 9–12 | MACP architecture, feature groups, baselines, calibration, risk control, proposed contribution |
| 13–18 | Experiments 1–7: error characterization, unseen ingredient/condition tests, ablation and SHAP, context, auditing, cross-recognizer validation, human-risk overlap |
| 19–20 | Primary endpoints, metrics, grouped uncertainty, clustered analysis, multiplicity, feasibility, negative findings |
| 21–22 | Software stack, avoid list, AI-assisted research practices |
| 23–25 | Repository tree and reproducibility requirements |
| 26–28 | Human-participant scope, final deliverables and figures |
| 29–30 | Phase 1 (Oct 30–Nov 7 2026): foundation, literature review, source/license verification, feasibility pilot and go/no-go |
| 31–33 | Phases 2–6: benchmark, MACP, auditor, independent validation/paper, Davidson application |
| 34–37 | Paper outline, application framing and dates to verify |
| 38–39 | Risks and mitigations |
| 39–41 | Completion checklist |
| 41–43 | Final strategic recommendation, scope, priority ranking and cautions |

## Pipeline and repository module map

The proposed modules are listed in [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md). Pipeline order: public vocabulary and pair graph → fictional text/audio generation → DSP and measured acoustics → two ASR runs → error alignment/labels → feature extraction → pair-level model and recording auditor → calibration/selective decision → grouped statistical analysis and explainability. Keep reference labels out of model inputs.

## Key experimental constraints

- Pilot 20–30 medication pairs before a 100–200 pair main benchmark; these are planning targets, not guaranteed power. Count independent ingredients and actual error events, not clips as sample size.
- Ingredient-disjoint split via connected components of the pair graph (initial approximate 60/20/20).
- Held-out acoustic evaluation reserves all clipping and the harshest babble/reverb combination.
- Quantify actual SNR and RT60; condition labels are metadata, not measured features.
- Run Whisper and Wav2Vec2; record versions, decoding settings, raw transcripts, and probabilities/n-best where available.
- Finite-sample threshold is selected on validation only. Distribution shift may invalidate the false-alert budget; E4 measures that.
- Hypotheses can fail; report negative results and limitations.
