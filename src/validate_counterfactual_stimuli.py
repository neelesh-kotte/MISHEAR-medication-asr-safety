"""Independently validate saved matched-target counterfactual stimuli."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(manifest: Path, output: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Stimulus manifest is empty")
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    checked = []
    errors = []
    for row in rows:
        audio = (manifest.parent / row["audio_path"]).resolve()
        if not audio.is_file() or sha256(audio) != row.get("audio_sha256"):
            errors.append({"audio_path": row.get("audio_path"), "issue": "audio_hash_mismatch_or_missing"})
            continue
        waveform, rate = sf.read(str(audio), dtype="int16", always_2d=True)
        if rate != int(row["sample_rate_hz"]):
            errors.append({"audio_path": row["audio_path"], "issue": "sample_rate_mismatch"})
            continue
        if waveform.shape[1] != 1:
            errors.append({"audio_path": row["audio_path"], "issue": "not_mono"})
            continue
        start, end = int(row["target_start_sample"]), int(row["target_end_sample"])
        segment = np.asarray(waveform[start:end, 0], dtype="<i2")
        if start < 0 or end <= start or end > len(waveform) or len(segment) != int(row["target_sample_count"]):
            errors.append({"audio_path": row["audio_path"], "issue": "invalid_target_boundaries"})
            continue
        observed_hash = hashlib.sha256(segment.tobytes()).hexdigest()
        if observed_hash != row.get("target_pcm16_sha256"):
            errors.append({"audio_path": row["audio_path"], "issue": "target_pcm16_hash_mismatch"})
            continue
        groups[(str(row["case_id"]), str(row["voice"]), str(row["requested_rt60_seconds"]),
                str(row["requested_snr_db"]))].append(row)
        checked.append(row["audio_path"])
    group_results = []
    for (case_id, voice, rt60, snr), group_rows in sorted(groups.items()):
        hashes = {row["target_pcm16_sha256"] for row in group_rows}
        conditions = [row["condition"] for row in group_rows]
        exact = len(hashes) == 1 and len(conditions) == len(set(conditions))
        group_results.append({"case_id": case_id, "voice": voice, "requested_rt60_seconds": rt60,
                              "requested_snr_db": snr,
                              "condition_count": len(conditions), "conditions": sorted(conditions),
                              "unique_target_pcm16_hashes": len(hashes), "exact_target_match": exact})
        if not exact:
            errors.append({"case_id": case_id, "voice": voice, "requested_rt60_seconds": rt60,
                           "requested_snr_db": snr,
                           "issue": "target_audio_not_identical_across_contexts_or_duplicate_condition"})
    result = {
        "status": "MATCHED_TARGET_PCM_VALIDATED" if not errors and len(checked) == len(rows) else "FAILED",
        "scope": "sample identity and file integrity only; not pronunciation, naturalness, context validity, or ASR performance",
        "manifest_path": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
        "stimuli_expected": len(rows), "stimuli_checked": len(checked),
        "condition_groups": group_results, "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.manifest, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "MATCHED_TARGET_PCM_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
