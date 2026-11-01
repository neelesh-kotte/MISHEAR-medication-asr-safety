# Synthetic stimulus review

Launch from the repository root:

```sh
./.venv/bin/python -m src.synthetic_quality_review --manifest results/human_speech/runs/v3-counterfactual-pilot-final-20261031/stimuli.jsonl --events results/human_speech/runs/v3-quality-review/events.jsonl --port 8766
```

Open http://127.0.0.1:8766/ and use a non-identifying ID such as `reviewer-01`.

1. Select S001–S024. Listen to the isolated medication audio; record what you hear or uncertainty. Save Stage A before intended text is revealed.
2. Listen to the complete clip and judge pronunciation agreement, linguistic context plausibility, and splice naturalness separately. Use uncertain instead of guessing. These questions do not require clinical expertise.
3. Save Stage B. Revisions append and preserve initial responses. Select the next case manually. Reloading requires a fresh Stage A; prior exposure is not erased. Keep the same reviewer ID.

No recognizer hypothesis or risk score is shown. Neutral IDs and hidden text reduce priming but repeated targets and prior project exposure prevent full blinding. Events record server save times and source/crop hashes, not proof that listening occurred. These are owner single-reviewer observations, not independent expert adjudication. The separate C001–C006 PriMock workflow and event log are unchanged.

Audio-quality approval does not itself assign an ASR correctness label. Post-inference word alignment produces candidate renderings for later outcome review. Fragmented names, alignment ties, deletions and boundary insertions require review; exact string absence is not a confirmed error.
