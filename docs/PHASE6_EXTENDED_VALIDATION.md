# Cross-corpus validation and reporting

The old H2 synthetic acoustic-condition and H4 synthetic recognizer-transfer plan is no longer the primary protocol. The current external validation target is PriMock57, with training/development on AfriSpeech-200 and a frozen, separate mock-consultation evaluation.

The report must identify corpus, official/saved split, speaker and ingredient counts, consultation counts, medication mention/error denominators, ASR checkpoint, review budget, error capture, false-alert rate, flagged precision, missed errors, flagged audio duration, and cluster-aware confidence intervals. An oracle ranking must be labeled theoretical. No output may describe these offline scores as workload reduction, harm prevention, clinical effectiveness, or deployment safety.

At the current workspace state, no corpora are present and the medication vocabulary is unversioned. Therefore no cross-corpus result can yet be generated. Run the inventory gate and resolve its stop reasons before fitting or reporting models.
