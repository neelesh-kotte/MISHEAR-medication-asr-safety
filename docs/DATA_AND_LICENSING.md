# Human-speech corpus, vocabulary, and tool licensing audit

**Protocol status (2026-10-30):** no real corpus files are present in this workspace. The active default config points to synthetic smoke fixtures; `config/corpus_config.real.example.yaml` is a template with intentionally unresolved commit placeholders. Verify exact releases, commits, license files, access acceptance, and downstream conditions before real-data use or redistribution.

## Planned corpora

| Corpus | Role and speech style | License | Data handling and limitations |
|---|---|---|---|
| [AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200) | Primary corpus for official speaker-held-out and medication-ingredient-held-out evaluation. English accented speech across prompted read sentences; medication mentions may be sparse. | **CC BY-NC-SA 4.0**, as shown on the dataset card. | The upstream Hub requires account information/terms acceptance before file access. Accept those terms yourself and set the local config acknowledgment only afterward. Retain attribution; observe noncommercial and share-alike terms for applicable redistributed/adapted materials. Preserve the official train/dev/test partitions, verify speaker IDs are disjoint, and keep audio and full annotations outside Git. Read-sentence distributions do not represent mock conversational speech. |
| [PriMock57](https://github.com/babylonhealth/primock57) | External transfer corpus: 57 mock primary-care consultations with audio and manual utterance-level transcripts. More conversational than read-sentence material. | Upstream [`LICENSE.md`](https://raw.githubusercontent.com/babylonhealth/primock57/main/LICENSE.md) states **CC BY 4.0**. | Attribute Babylon Health and the dataset authors, link the license, and mark modifications. Keep every consultation intact in any split; no utterance-level random split. Mock consultations are not real-world clinical recordings and are not equivalent to AfriSpeech read sentences. Audio is stored with Git LFS upstream. |

The license statements are based on the live upstream [AfriSpeech-200 dataset card](https://huggingface.co/datasets/intronhealth/afrispeech-200) and [PriMock57 license file](https://raw.githubusercontent.com/babylonhealth/primock57/main/LICENSE.md), reviewed October 30, 2026. The [AfriSpeech card](https://huggingface.co/datasets/intronhealth/afrispeech-200) describes its read-sentence task, dataset fields, official speaker splits, and gated access. The [AfriSpeech paper](https://aclanthology.org/2023.tacl-1.93/) describes the corpus. This table does not replace review of the exact downloaded revision or advice about a particular use.

## Medication vocabulary and annotation

The checked-in `data/vocabulary/medication_vocabulary.csv` now follows schema v1.0, includes RxNorm ingredient IDs, a release identifier, brand aliases, and a combination-product example. The vocabulary validator passes. This does not clear corpus feasibility: medication coverage in both datasets is unknown, local manifests are absent, and references still need medication-mention resolution/adjudication. Keep ambiguity and unresolved-term reports with inventory output. Do not infer a confirmed medication identity from a fuzzy string match alone; uncertain references require adjudication.

## Blinded inference and data protection

The auditor and baseline runtime inputs are limited to audio, ASR transcript, ASR token/span confidences, N-best alternatives, other ASR output, and the general authorized vocabulary. Reference drug names, timestamps, recording-specific reference candidates, and correctness labels are evaluation-only and join after scores are frozen. Do not use PriMock57 consultation notes or unrelated human-evaluation materials as model input. Do not commit raw audio, transcript dumps, annotations, speaker identities, or local corpus paths.

## ASR/model tools

- OpenAI Whisper code and weights are published under MIT; pin the exact package/model revision and preserve notices. See [upstream license](https://github.com/openai/whisper#license).
- Wav2Vec2 and Hugging Face checkpoint terms vary by checkpoint. Audit the exact model card, revision, training data, and license before use; do not assume an architecture-wide license.
- Local inference does not itself grant rights to redistribute corpora, model weights, or outputs.
- RxNorm/NLM terms depend on the exact product/release and source vocabularies. Record the source and release, use only terms permitted for the planned purpose, and resolve ingredient identifiers reproducibly.

## Required audit before modeling

1. Record corpus repository/release commit, retrieval date, license-file checksum, and permitted use.
2. Verify AfriSpeech official partition/speaker metadata and save an immutable split manifest.
3. Record PriMock57 consultation IDs locally and verify no consultation crosses partitions.
4. Version the medication vocabulary; resolve every analyzed mention and combination ingredient or report it unresolved.
5. Run `python -m src.data_inventory --config config/corpus_config.yaml`; stop if its report says `STOP_BEFORE_TRAINING`.
6. Freeze held-out ingredients and evaluation budgets before inspecting model outcomes.
