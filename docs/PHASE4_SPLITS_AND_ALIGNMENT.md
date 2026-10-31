# Human-speech partitions and reference alignment

The synthetic Phase 4 protocol is superseded. The active partition utilities are in `src/data_splits.py`.

## Evaluation A: AfriSpeech-200 unseen speakers

Use official train/dev/test metadata without reassigning recordings. `verify_official_speaker_partitions` requires all three partitions and raises if any speaker appears in more than one. Preserve source split labels in the frozen manifest.

## Evaluation B: held-out medication ingredients

Freeze ingredient IDs before fitting or tuning. `create_ingredient_holdout_split` assigns every multi-ingredient utterance containing any held-out ID wholly to test; its optional `exclude` policy removes mixed utterances under a separately prespecified pure-mention evaluation. A held-out ID absent from the manifest is an error.

## External transfer: PriMock57

Use consultation IDs as the grouping unit. `create_primock_consultation_splits` assigns all utterances from a consultation to one partition. The main external-transfer test should remain separate from model selection.

Reference/hypothesis alignment belongs after ASR inference. Keep reference drug identities and timestamps out of all auditor and baseline inputs. Save normalization rules, unresolved terms, split manifests, corpus revision, and checksums. Do not treat utterance rows from one speaker or one consultation as independent bootstrap units.
