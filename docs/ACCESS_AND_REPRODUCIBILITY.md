# Authorized corpus access and reproducible setup

`config/corpus_config.yaml` points to the expected local real-corpus paths and
is deliberately gated until access is accepted, files are present, and exact
upstream commits are pinned. `config/corpus_config.stub.yaml` points only to
`config/stubs/`. Those synthetic records and audio URIs validate parsing,
splits, ablations, and plotting; never report their results as AfriSpeech-200 or
PriMock57 results.

## AfriSpeech-200

1. Sign into Hugging Face and request/accept the dataset card's access terms at
   [`intronhealth/afrispeech-200`](https://huggingface.co/datasets/intronhealth/afrispeech-200).
   Record who accepted the terms and the date in the private experiment log.
2. Authenticate the local CLI after approval:

   ```bash
   hf auth login
   hf auth whoami
   ```

   Keep the token in the Hugging Face credential store; do not place it in YAML,
   shell scripts, logs, or Git.
3. Resolve and record a full 40-character dataset repository commit. Use that
   commit in `datasets.afrispeech200.revision`, set `access_terms_accepted: true`
   only in the private local config after acceptance, and preserve the dataset's
   official `train`, `dev`, and `test` labels. Do not split or merge speakers
   across those partitions.
4. Load metadata first with audio decoding disabled. The upstream card states
   the transcript is a prompted sentence and that the official split is
   speaker-disjoint; keep this read-sentence limitation visible in reports.
5. Run the inventory gate and inspect `official_partition_audit`, source commit,
   mention/ingredient counts, unresolved vocabulary terms, and missing IDs.

## PriMock57

1. Install Git LFS, clone the upstream repository, and record the exact commit
   before any preprocessing:

   ```bash
   brew install git-lfs       # macOS, if Git LFS is not installed
   git lfs install
   git clone https://github.com/babylonhealth/primock57.git data/external/primock57
   git -C data/external/primock57 rev-parse HEAD
   git -C data/external/primock57 lfs pull
   git -C data/external/primock57 lfs ls-files
   ```

2. Inspect the cloned `LICENSE.md` and upstream README. Preserve attribution and
   license terms. PriMock57 is a mock primary-care conversation set; do not
   characterize it as actual patient-care audio.
3. Convert utterance annotations to the normalized JSONL manifest without
   breaking consultation IDs. Include `consultation_id`, `speaker_id`,
   `utterance_id`, `transcript`, `audio_path`, `duration_s`, and any reviewed
   medication annotation fields. Keep audio and full transcripts out of Git.
4. Set the same pinned commit in `revision` and `source_revision` in the private
   corpus config. Split at the consultation level using a recorded random seed.

## Freeze vocabulary and execution provenance

The checked-in development vocabulary maps source-verified ingredient IDs to
RxNorm RxCUIs and records an RxNorm release in each row. Check its version and
terms against the [WHO INN lists](https://www.who.int/teams/health-product-and-policy-standards/inn/inn-lists)
and [NLM RxNorm monthly releases](https://www.nlm.nih.gov/research/umls/licensedcontent/rxnormfiles.html)
before expanding or changing the active vocabulary. Preserve the upstream
release files or their checksums, the vocabulary CSV checksum, every brand alias
decision, and combination component IDs. The metadata mapping rule is exact:
case-folded `brand_names` and `aliases` resolve to that row's ingredient ID;
combination product aliases resolve to every component ID. Phonetic keys are
retrieval signals only.

After installing both real corpora and accepting the access terms, update
`config/corpus_config.yaml`: set `access_terms_accepted: true`, replace each
`PIN_VERIFIED_*` placeholder with the full immutable upstream commit, and verify
the configured local manifest/audio paths. Then run:

```bash
python -m src.data_inventory --config config/corpus_config.local.yaml \
  --json results/inventory/feasibility.json \
  --markdown results/inventory/feasibility.md
python run_pipeline.py --stage verify-afrispeech \
  --config config/corpus_config.local.yaml \
  --manifest data/external/afrispeech-200/metadata.csv
pytest tests/
```

The smoke config intentionally reports
`FEASIBLE_FOR_PIPELINE_SMOKE_TEST` with `training_authorized: false`. Only an
inventory built from authorized, pinned human-speech corpora and a source-audited
vocabulary can pass the real feasibility gate. Keep checksums, exact commits,
config copies, software versions, seeds, and any manual annotation decisions in
the experiment record so an external reviewer can reconstruct the run.

## Source references

- [AfriSpeech-200 dataset card](https://huggingface.co/datasets/intronhealth/afrispeech-200)
- [PriMock57 upstream repository and LFS instructions](https://github.com/babylonhealth/primock57)
- [PriMock57 upstream license file](https://github.com/babylonhealth/primock57/blob/main/LICENSE.md)
- [WHO INN lists](https://www.who.int/teams/health-product-and-policy-standards/inn/inn-lists)
- [NLM RxNorm release files and terms](https://www.nlm.nih.gov/research/umls/licensedcontent/rxnormfiles.html)
