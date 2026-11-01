"""Build an explicit source-only candidate archive; no Git operations or uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
FILES = [
    'src/__init__.py', 'src/asr_evaluation.py', 'src/local_annotation_review.py',
    'src/counterfactual_speech.py', 'src/run_counterfactual_asr.py',
    'src/validate_counterfactual_stimuli.py', 'src/synthetic_quality_review.py',
    'src/counterfactual_token_evaluation.py',
    'tests/test_counterfactual_speech.py', 'tests/test_synthetic_quality_review.py',
    'tests/test_asr_evaluation.py', 'config/counterfactual_pilot.example.json',
    'docs/SYNTHETIC_REVIEW_GUIDE.md', 'src/validate_synthetic_quality.py',
    'release/LICENSE', 'release/CITATION.cff', 'release/RELEASE_AUDIT.md',
    'scripts/build_v3_engineering_package.py',
]
README = '''# MISHEAR v3 counterfactual engineering candidate

This source-only package is not a validated medication-error benchmark or trained auditor.
No corpus text, audio, annotations, model weights, or empirical outcome tables are included.
This preliminary source release is MIT-licensed within the scope in RELEASE_AUDIT.md.

Python 3.9+; create a local environment and install requirements-engineering.txt.
Run `python -m pytest -q`. An optional local eSpeak NG executable enables the
real synthesis integrity test; otherwise that test skips. Tests mock ASR and do
not establish model accuracy. For actual inference additionally install openai-whisper
and PyTorch, and obtain a permitted local Whisper checkpoint. No download is automatic.

Generate an exploratory pilot:

    python -m src.counterfactual_speech --spec config/counterfactual_pilot.example.json --output-dir results/pilot --voice en-gb --voice en-us --snr-db 10 --include-clean --rt60-s 0.1 --include-dry
    python -m src.validate_counterfactual_stimuli --manifest results/pilot/stimuli.jsonl --output results/pilot/validation.json
    python -m src.synthetic_quality_review --manifest results/pilot/stimuli.jsonl --events results/review/events.jsonl --port 8766

Open http://127.0.0.1:8766/ and follow docs/SYNTHETIC_REVIEW_GUIDE.md.
The general human-speech review module is included as a utility dependency only;
its neighboring-text functionality requires the full MISHEAR checkout.

After obtaining a local checkpoint:

    python -m src.run_counterfactual_asr --manifest results/pilot/stimuli.jsonl --output-dir results/inference --checkpoint /path/to/base.pt
    python -m src.counterfactual_token_evaluation --manifest results/pilot/stimuli.jsonl --predictions results/inference/predictions.jsonl --output results/token-review.json

Token alignment is a post-inference review aid. It never assigns a correctness label.
The target waveform is held identical across contexts within a voice/condition group;
pronunciation, plausibility and splice validity still require human review. One target
and one generator cannot support population/generalization claims. No model fitting,
calibration, final evaluation or clinical performance is claimed.

Source implementation and documentation are AI-assisted; owner listening observations
are separate and excluded. No independent external reproduction is claimed.
'''
def build(destination):
    root=Path(__file__).resolve().parents[1]
    payload={name:(root/name).read_bytes() for name in FILES}
    payload['README.md']=README.encode()
    for name in ('LICENSE', 'CITATION.cff', 'RELEASE_AUDIT.md'):
        payload[name]=payload['release/'+name]
    payload['requirements-engineering.txt']=b'numpy==2.0.2\nscipy==1.13.1\nsoundfile==0.13.1\npytest==8.4.2\n'
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in sorted(payload.items())}
    payload['FILE_SHA256.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):archive.writestr(name,data)
    print(json.dumps({'archive':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'files':len(payload),'status':'PRELIMINARY_SOURCE_ONLY_MIT_SCOPED'}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);build(p.parse_args().output)
