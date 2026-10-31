# Existing human-speech audio and blinded inference

The earlier synthetic TTS/DSP protocol is archived and superseded for the primary MISHEAR question. The active plan uses licensed AfriSpeech-200 and PriMock57 human speech; no artificial babble or room simulation is part of the primary evidence.

## Corpus handling

Configure only local paths in `config/corpus_config.yaml`. Run `python -m src.data_inventory` before inference. Preserve original audio and exact corpus metadata. Resampling must record source and target sample rates. Do not add corpus audio or full transcripts to Git.

## Blinded ASR-side processing

`src.asr_inference.transcribe_files` writes recognizer output and provenance before gold labels are joined. `src.auditor.audit_segment` accepts the waveform, sample rate, transcript, N-best alternatives, token confidence, alternate recognizer text, and general vocabulary subset. It emits a review score and never emits a corrected drug name. Reference medication names, reference-derived timestamps, and correctness labels are prohibited inputs.

## Corpus limits

AfriSpeech-200 consists of read-sentence distributions and is not representative of mock conversation. PriMock57 provides mock primary-care conversational speech, but it is not a real clinical deployment corpus. Cross-corpus differences must be described as domain transfer, not proof of clinical generalization.
