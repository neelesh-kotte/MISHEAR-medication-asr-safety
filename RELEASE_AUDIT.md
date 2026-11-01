# v0.1.0-pilot source release scope

This is preliminary engineering software, not a complete MISHEAR v3 study.
The MIT license applies only to the original project code and documentation
included in this release tree. It does not relicense previous datasets, research
artifacts, dependencies, models, or historical commits.

The owner authorized MIT for material they own/have authority to license.
Git records for included existing source identify Neelesh Kotte and the earlier
project identity MISHEAR Research. The development assistance log records
AI-assisted implementation of the new modules. These records are provenance
evidence, not proof of sole legal authorship. No sole-copyright claim is made.
No copied third-party implementation or conflicting license notice was found
in the allowlisted modules; they call dependency APIs. Any undisclosed ownership
agreement remains outside what repository inspection can establish.

External dependencies are not vendored: NumPy/SciPy/SoundFile use BSD-family
licenses; pytest and optional Whisper use MIT; PyTorch BSD-3-Clause;
Transformers Apache-2.0. SoundFile's native libsndfile has separate LGPL terms.
eSpeak NG is an external GPL program, invoked as a subprocess. This archive
contains no dependency binaries, model weights, corpus text, synthetic/real
audio, annotation events, or outcome-result claims. Users must observe each
installed dependency, model, dataset and generated-output license independently.

History is preserved through a branch based on published main. Prior public
history contains eSpeak synthetic audio, local author path metadata, and
unsupported historical research claims. They are not in this release tree or
its source archive and are not validated or relicensed by this release. No
private consultation corpus was found in the scanned locally reachable history.
A 242-blob audit found no matches for narrow known GitHub/private-key/AWS
credential signatures; this is not a guarantee that every possible secret or
ownership issue has been identified. Original evidence is preserved locally.

The prior descriptive context comparison is invalidated: an omitted RT60 field
made prediction IDs collide. Current IDs and joins use waveform SHA-256. Raw
outputs are isolated by run, and the exact hashed checkpoint path is loaded.
Token alignment is a post-inference review aid; it emits no correctness labels.
Human quality and outcome review remain required. No baseline superiority,
calibration, unseen-medication/generalization, clinical effect or novelty result
is asserted. No independent external reproduction or peer review occurred.
