# Environment for human-speech corpus auditing

The active pipeline uses local corpus files and local ASR inference. Check the exact corpus license and model revision before download. Keep corpus audio and annotations out of the repository.

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.data_inventory --config config/corpus_config.yaml
```

The inventory should complete before any expensive ASR/model run and will return `STOP_BEFORE_TRAINING` when corpora, vocabulary resolution, sample counts, or required grouping metadata are inadequate. Download AfriSpeech-200 and PriMock57 from their upstream sources according to [`DATA_AND_LICENSING.md`](DATA_AND_LICENSING.md); do not put absolute user-specific corpus paths in tracked configs.
