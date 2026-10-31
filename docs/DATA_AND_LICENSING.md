# Data, licensing, and tool availability audit

**Checked:** 2026-10-30. Availability was checked in the current local macOS workspace; license statements below were checked against the linked publisher, official project, or model pages. This is a project audit, not legal advice. Recheck exact versions, voice/model cards, and terms before acquisition, redistribution, or publication.

## Local availability snapshot

| Component | Local check | Status |
|---|---|---|
| macOS `say` TTS | `/usr/bin/say` exists; voice listing works and includes multiple English voices. | Available for a local feasibility trial. Voice output redistribution/use rights need current macOS license review; do not publish generated audio until cleared. |
| Xcode command-line tools | `xcode-select -p` returned `/Library/Developer/CommandLineTools`. | Available; useful for native builds. |
| OpenAI Whisper Python package | `whisper` Python module not installed; `whisper` executable not found. | Not available in this environment yet. |
| whisper.cpp | `whisper-cli`/`whisper.cpp` executable not found. | Not available yet; build/install after pinning a release and recording configuration. |
| Transformers / PyTorch | `transformers` and `torch` Python modules not installed. | Wav2Vec2 inference not available yet. |
| Phonetic packages | `phonemizer`, `pronouncing`, and `cmudict` Python modules not installed. | The pilot implementation currently uses a transparent built-in Soundex fallback; install/use CMUdict or a G2P tool only after licensing and pronunciation coverage review. |
| Similarity packages | `jellyfish` and `textdistance` Python modules not installed. | No external similarity dependency is needed for the starter pilot: `src/vocabulary.py` implements Jaro-Winkler, edit distance and Soundex directly. |

This is a point-in-time environment check. The project has not downloaded RxNorm, model weights, voices, or external datasets.

## Source and license register

| Source/tool | Publicly documented terms and access | Project decision / outstanding check |
|---|---|---|
| **Whisper code and model weights** | OpenAI documents Whisper code and weights under the [MIT License](https://github.com/openai/whisper#license). The [`whisper.cpp` project](https://github.com/ggml-org/whisper.cpp) also identifies MIT licensing. | Suitable candidate for local research subject to preserving notices and pinning the selected model/checkpoint. Record exact version, model size, checksum, decoding settings and Core ML/CPU path. “MIT” does not mean every third-party dependency or derived asset is automatically cleared. |
| **Wav2Vec2 model** | Model licenses vary. The official [facebook/wav2vec2-base-960h model card](https://huggingface.co/facebook/wav2vec2-base-960h) lists Apache-2.0. The [Transformers Wav2Vec2 documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2) documents the CTC architecture/API. | Pin a specific model card and license; do not treat “Wav2Vec2” as one license. Check its training/evaluation dataset terms and suitability for the planned task. Local packages are not installed yet. |
| **Apple macOS `say` and system voices** | `say` and multiple voices are available locally. Apple’s [macOS software license archive](https://www.apple.com/legal/sla/) is the governing source; an older published macOS SLA explicitly restricts recording, publishing or redistribution of System Voices. | Availability is verified; audio rights under the current installed macOS SLA are **not yet cleared**. Use only for private feasibility experiments unless current terms are checked and the intended synthetic audio use is permitted. Prefer a voice with explicit redistribution terms if publishing audio. |
| **Piper TTS** | Current [`OHF-Voice/piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl) is GPL-3.0. Its release history notes a GPLv3 license change. Voices are separate model files and each voice’s model card/license must be checked independently. | Technically plausible local alternative, but not installed. Pin engine release and a specific voice; inspect voice provenance, license, and any dataset obligations. The program license does not itself grant rights to every voice. |
| **Kokoro TTS** | The [`hexgrad/Kokoro-82M` model card](https://huggingface.co/hexgrad/Kokoro-82M) lists Apache-2.0 for the weights; project code and dependencies should be separately checked. | Candidate with comparatively clear model-weight licensing. Not installed or rendered locally. Pin model and voice assets and verify any external phonemizer/runtime terms. |
| **RxNorm / NLM medication vocabulary** | NLM says RxNorm normalized names/codes created by NLM are public domain; downloading the **full** RxNorm release requires a free UMLS license, and the full release can contain proprietary source-vocabulary data with source-specific restrictions. The [NLM terms of service](https://www.nlm.nih.gov/research/umls/rxnorm/docs/termsofservice.html) and [RxNorm FAQ](https://www.nlm.nih.gov/research/umls/rxnorm/faq.html) distinguish the full release from selected/public-domain content. Current Prescribable Content files are listed by NLM as no-license-required. | Before download, choose the exact release/product and record its source vocabularies and restrictions. Use a UMLS account/license for full releases if needed; do not redistribute restricted rows. Preserve NLM attribution. Ingredient grouping must resolve variants to active-ingredient concepts. No RxNorm files downloaded yet. |
| **CMU Pronouncing Dictionary (CMUdict)** | CMU’s [license file](https://github.com/cmusphinx/cmudict/blob/master/LICENSE) permits redistribution under BSD-style conditions with copyright notice and disclaimer retention. | Candidate pronunciation lexicon. Not installed locally. Pin a distribution and preserve notices. Verify coverage for medication names; fallback G2P outputs must be identified as generated pronunciations, not dictionary attestations. |
| **g2p-en** | The upstream [project packaging metadata](https://github.com/Kyubyong/g2p/blob/master/setup.py) identifies Apache Software License; inspect the exact release’s bundled code/dependency notices. | Optional grapheme-to-phoneme fallback; not installed. Confirm the specific package version and dependency licenses before inclusion. |
| **ClinicalBERT** | The [`emilyalsentzer/Bio_ClinicalBERT` model card](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) lists MIT and states it was trained on MIMIC-III clinical notes. | The blueprint prohibits using MIMIC patient records but proposes a ClinicalBERT-derived context feature. No MIMIC files will be used. Before adopting this model, document that the project uses only published pretrained weights and review applicable MIMIC/model-card terms and competition rules; alternatively choose a context model whose provenance fits the protocol. The model is not installed. |
| **FDA POCA** | FDA describes POCA as a tool for orthographic/phonetic similarity screening against drug-name data sources and offers [POCA Online](https://poca-public.fda.gov/). FDA cautions that a score does not guarantee a safety determination. | POCA is an optional external comparator, not a dependency. The scoring implementation is not claimed to be open or recreated. If used, retain actual returned scores, query date, input set and method; do not scrape or automate beyond permitted access. Terms/API reproducibility have not yet been verified. |
| **ISMP confused-drug-name list** | ISMP publishes a [List of Confused Drug Names](https://www.ismp.org/node/102). Public availability does not by itself establish permission to redistribute or republish the full list. | Use only the minimum data needed for pair overlap, record source/version/retrieval date, and verify terms/permission before redistribution. No copy downloaded. |

## Audit actions before the main benchmark

1. Choose and record the exact RxNorm release and whether the no-license-required Prescribable Content is sufficient; if using the full release, complete the UMLS license process and track source restrictions.
2. Select one TTS engine and 3–5 voice assets. Keep engine, model, voice, version, and license separately recorded. Do not redistribute `say` outputs unless current Apple terms permit it.
3. Install/pin Whisper (`whisper.cpp`) and Wav2Vec2 model packages/checkpoints, then capture model IDs, license files, hashes, versions, and decoding parameters in `config/asr_config.yaml`.
4. Check terms for human-risk lists, POCA access, dependency licenses, and any audio redistribution before making the benchmark public.
5. Resolve the ClinicalBERT/MIMIC provenance issue above before using it in the planned context feature.

## Reproducible environment commands

The local environment is not fully installed. Start with a clean venv, then add only audited/pinned packages:

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# After creating a reviewed, pinned requirements file:
python -m pip install -r requirements.txt
```

The minimal candidate test currently runs with the Python standard library only:

```bash
python3 -m unittest tests/test_vocabulary_pilot.py -v
```
