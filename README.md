# MISHEAR: medication ASR safety research

Research project: predicting, explaining, and selectively detecting medication-name errors in automatic speech recognition under unseen-drug and unseen-condition generalization. This is a fully computational study using public medication terminology and synthetic fictional speech. No patient records, real clinical notes, or human recordings are in scope. This is not a clinically validated product and makes no claim of clinical effectiveness.

## First steps

1. Read [`docs/PDF_INDEX.md`](docs/PDF_INDEX.md) for the blueprint index.
2. Read [`prespecification.md`](prespecification.md) before working with evaluation data.
3. Complete Phase 1 literature, source/license checks, feasibility, and required school review.
4. Start with a small pilot; do not scale until observed error counts and independent ingredient groups support the planned tests.

## Repository status

Phase 1 foundation only. Source modules are planned in `docs/REPOSITORY_MAP.md`; implement them in dependency order after the prespecification and feasibility decisions. Data directories contain no data yet.

## Environment

See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md). The dependency file is intentionally minimal until the TTS, ASR, and DSP stack is verified on the target machine.

## Reproducibility and scope

Use fixed seeds, save configurations and exact splits, preserve original ASR output, and keep all test-set labels out of model inputs. Track AI-assisted code in [`AI_ASSISTANCE_LOG.md`](AI_ASSISTANCE_LOG.md).
