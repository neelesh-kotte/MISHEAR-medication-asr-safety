# Cross-corpus experiment sequence

Run in this order; do not begin model training until the inventory report passes.

1. `01_corpus_feasibility`: inventory medication mentions, speakers, consultations, resolved ingredients, ambiguities, and unresolved terms.
2. `02_official_speaker_test`: preserve AfriSpeech-200 official speaker-disjoint train/dev/test partitions.
3. `03_ingredient_holdout`: freeze held-out ingredient IDs and enforce whole-utterance handling for combination mentions.
4. `04_blinded_asr_scoring`: generate ASR outputs, confidence, N-best, disagreement, and audio-aware scores without reference labels.
5. `05_review_budget_evaluation`: join adjudicated references and compute error capture at fixed review budgets.
6. `06_primock57_transfer`: evaluate the frozen AfriSpeech-trained auditor on PriMock57 with consultations intact.
7. `07_ablations`: compare audio, disagreement, and phonetic-retrieval ablations on development data.

All reported scores must include denominators and speaker/ingredient or consultation-aware uncertainty. The evaluation describes error concentration and prioritization only.
