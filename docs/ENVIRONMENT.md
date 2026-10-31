# Environment setup (macOS / Apple Silicon)

The plan specifies Python/conda, Xcode command-line tools, whisper.cpp with Core ML, and Hugging Face Transformers/Wav2Vec2 (MPS or CPU). Do not install the full ASR/DSP stack until model, voice, data licenses and target versions are recorded.

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
python3 --version
xcode-select --install  # only if command-line tools are missing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Add reviewed, pinned packages to requirements.txt before running:
# python -m pip install -r requirements.txt
```

When selected, package groups from the blueprint include NumPy, pandas, SciPy, scikit-learn, statsmodels, matplotlib/seaborn, librosa or pydub, pyroomacoustics, phonemizer or g2p-en, SHAP, Transformers, and the selected TTS and ASR dependencies. Some packages have system or model-download requirements; record those after the pilot setup succeeds.

## Initialize or inspect Git

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
git status --short
git log -1 --format='%cI %h %s'
```
