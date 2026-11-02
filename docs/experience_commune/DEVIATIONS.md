# EXPERIENCE_COMMUNE_V1: deviations and implementation notes (append-only)

## Before the freeze (implementation choices, each checked to leave the computation unchanged or declared)

- **D0 — exact mixture score in both modes.** Study S's signal (onehot - softmax)(r - b) is replaced in N and U by
  the exact score of its own epsilon-mixture policy (mission §4, §8; AUDIT.md item 7). N is a corrected reference.
- **D1 — row-major plastic brain.** `learn.fast_plastic_class()` holds the activity row-major and drops the
  block-diagonal matrix that only ROCm uses; bit-identical to Study S's PlasticLIF (magnitudes, eligibility means,
  actions after 25 decisions; `manifests/sim_checks.json`), ~18x faster evaluation.
- **D2 — evaluation noise per fixed chunk of 64 bank images** (keyed by block evaluation seed, bank and chunk index)
  rather than per image: the same noise for an image in every learner, mode, checkpoint and graph; independent of the
  training streams.
- **D3 — memory:** training stimuli on the CPU, noise indexed per tick (after an out-of-memory incident with four
  concurrent processes in the pilot; PILOT.md). No change of computation.
- **D4 — one graph process at a time:** concurrency gave no throughput (PILOT.md, benchmarks).
- **D5 — family b reconstructed** from its seeds on this server (no historical hashes survive); invariants checked
  exactly for every graph (`manifests/nulls_b.json`).
- **D6 — training noise:** one generator stream per (block, task), shared by N and U and by every graph; the mission's
  "graines de bruit neuronal" per block are realised this way.

## After the freeze

None affecting the campaign or its frozen analysis.

**Exploratory complements, decided after the result (2026-10-21, owner's authorisation of complements):**
- **X1 — representation probe** (`scripts/experience_commune/probe_representation.py`): untrained readout of the 41
  graphs on block b0's final banks, few-example ridge decoding from the 2,227 counts and from the 8 fixed scores.
- **X2 — learning at matched activity** (`scripts/experience_commune/matched_activity_learning.py`): mode N, blocks
  b0-b2, the P2/P3 nulls at ACTIVITY_MATCH's calibrated gains. `learn.GraphRun` and `learn.run` gained an optional
  `gamma` (default 1, the campaign's value: the campaign's computation is unchanged).

- **Finding (04:30-04:40Z): reproducibility depends on the learner set.** A run is bit-identical when repeated with the
  same learners in the batch (checked again), but a different set changes floating-point rounding in the batched
  products, which the spiking dynamics amplify: the connectome rerun with 2 learners instead of 24 has identical
  weights at 0 decisions, then 86-94 % agreement of greedy predictions at 500 decisions. Exact reproduction of a
  campaign run therefore needs its 24-learner batch (as `scripts/experience_commune/campaign.py` does). For X2 (6
  learners per batch) this acts like another noise realisation, not a bias.
- **X3 — X2 with the campaign's full design** (`scripts/experience_commune/matched_activity_full.py`, launched
  04:41Z): the 20 nulls at their matched gains, the campaign's 24 learners in the campaign's order (N and U, b0-b5),
  final bank only, so the only difference from the campaign's own null runs is the gain. No new graph after 10:15Z.
  Analysis (`matched_activity_analysis.py --x3 U` and `--x3 N`, tested before any X3 result by feeding the campaign's
  runs as the "matched" runs: G_matched = G_common exactly). Reading rule declared now: in each mode, task and family,
  the advantage is "maintained at matched activity" if the 95 % descriptive bootstrap lower bound of G_matched is
  above +3 points (the campaign's margin), "reduced but present" if above 0 only, "not shown" otherwise. Mode U is the
  analogue of the campaign's primary contrast.