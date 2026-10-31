# Environment setup (macOS / Apple Silicon)

The plan specifies Python/conda, Xcode command-line tools, whisper.cpp with Core ML, and Hugging Face Transformers/Wav2Vec2 (MPS or CPU). Do not install the full ASR/DSP stack until model, voice, data licenses and target versions are recorded.

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
python3 --version
xcode-select --install  # only if command-line tools are missing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current `requirements.txt` covers the numerical/audio base, optional openai-whisper and Transformers/PyTorch backends, and CMUdict. Whisper's Python wrapper needs FFmpeg, and whisper.cpp needs a native build plus a separately downloaded ggml model. See [`PHASE3_ASR_AND_FEATURES.md`](PHASE3_ASR_AND_FEATURES.md) for install and run commands. Other blueprint packages (scikit-learn, statsmodels, SHAP, plotting, etc.) should be added when implementing their corresponding model/analysis phase, with resolved versions recorded before the experiment freeze.

## Initialize or inspect Git

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
git status --short
git log -1 --format='%cI %h %s'
```
