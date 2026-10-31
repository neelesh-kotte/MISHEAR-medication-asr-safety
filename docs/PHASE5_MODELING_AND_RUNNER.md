# Auditor models and offline review prioritization

The former synthetic pair-level classifier plan is superseded. The active systems rank existing human-speech utterances by medication-error risk: token/span confidence, N-best uncertainty, recognizer disagreement, lexical/phonetic anomaly, a combined transcript-only classifier, and the audio-aware auditor.

Model fitting and feature selection use only training-side partitions. Select the final pipeline on development data, freeze it, then evaluate once on AfriSpeech official test, the ingredient-held-out test, or PriMock57 external transfer. Keep reference-derived features out of training inputs. The `src/ablations.py` runner compares full, text-only/no-audio, no-disagreement, and no-phonetic-retrieval variants.

Primary outputs are error capture at 1%, 5%, 10%, and 20% ranked review budgets, mention-denominator false-alert rate, flagged-utterance precision, missed errors, flagged duration, and oracle ceiling. Describe these as error concentration and prioritization only.
