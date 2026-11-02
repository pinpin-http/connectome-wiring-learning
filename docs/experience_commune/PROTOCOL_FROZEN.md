# EXPERIENCE_COMMUNE_V1: frozen protocol

**Frozen 2026-10-20T12:55Z**, before any main-campaign run (only the pilot and benchmarks on development graphs existed). The
sha256 of this file and of `configs/experience_commune_v1.yaml`, and the freeze commit, are recorded in RUN_LOG.md. A
local, dated freeze in git, not a public preregistration. Authority: `docs/experience_commune/MISSION.md`. Nothing here depends on
ACTIVITY_MATCH_V2's results. After the freeze, a change is dated in DEVIATIONS.md; if it changes the scientific
computation, the affected runs are requalified or rerun within the budget.

## 1. Question and hypothesis

Study S found the connectome learning the two visual tasks from reward faster than every null (E1, its own reward
rate during training). E1 depends on the actions each graph chose. **Hypothesis:** the connectome keeps a performance
advantage during learning over its nulls when the history of images, actions and rewards is identical across graphs.
An advantage that persists under U cannot be attributed only to differently collected experience. It would not show
a better visual representation, a biological mechanism or independence from every property of the policy or the
optimisation (target probabilities, local gradients, their variance, eligibilities and optimiser states still differ).

## 2. Graphs (plan selected in §10)

Connectome; family b `nulllat:<s>` (out-strength kept; rebuilt and validated here, AUDIT.md item 2); family c
`nullin:<s>` (in-strength kept). Selection before any run, independent of scores
(`manifests/graph_selection.json`): per family a seeded permutation of 0-39; P1 takes the first 20, P2/P3 the first
10 (nested). Development graphs `nulllat:100`, `nullin:100` (pilot only).

## 3. Blocks and shared data

Six blocks b0-b5 (P3: b0-b2), each fixing a readout projection, 6,000 training stimuli per task, the U journal, N's
sampling uniforms, the training noise stream and three evaluation banks (512 progress, 2,048 final, 512 variants),
shared by every graph and both modes; independent draws between blocks (`manifests/data.json`, sha256 of every file;
U journals in `journals/`, identical bytes for every graph, verified in every run record). Each task and mode starts
from the graph's anatomical magnitudes; no weight passes between tasks.

## 4. Modes and the learning rule

Identical in N and U: initial weights, signs and bounds, gain 0.18452, input gain, LIF parameters, 16 ticks from
rest, interface, projection, plastic edges (all connections onto the 2,227 readout neurons), eligibility (kappa 0.2,
mean rate 0.05), eta 0.001, RMS normalisation (0.95, bias-corrected), the class-indexed baseline (rate 0.1, read before
its update; it uses the stimulus class), stimuli, decisions and evaluations. Different, as required:

- **behaviour:** N samples its own pi = 0.9 softmax(z / 2) + 0.1 / 8 by inverse CDF at the block's shared uniform;
  U plays the journal's action (uniform with replacement, independent of graph, scores and labels; reward rate about
  1/8 whatever the learner);
- **sampling correction:** v = rho (r - b) (1 - eps)(u_a / pi_a)(onehot(a) - u), rho = pi_a / q_a (N: 1; U: 8 pi_a,
  in [0.1, 7.3]); L = R^T v / tau, applied to the raw signal before the RMS normalisation; no clipping or
  self-normalisation of rho. The exact mixture score function is used in both modes (Study S used the softmax score
  alone; AUDIT.md item 7): N is a corrected reference, not an exact reproduction of Study S.

Verified before the freeze: analytic bandit (expected signal = finite-difference gradient in N and U, several
baselines and score shapes; rho bounds; tau once); simulator (N and U identical when given the same action with
rho = 1; evaluation read-only and repeatable; resume equals a continuous run bit for bit; initial magnitudes; the
row-major brain equals Study S's PlasticLIF bit for bit) — `tests/unit/test_experience_commune.py`,
`manifests/sim_checks.json`.

## 5. Evaluation

Checkpoints 0, 250, 500, 1000, 2000, 4000, 6000: greedy argmax of the 8 scores on the block's 512 progress images
(64 per class), never U's behaviour, on a separate evaluation brain with copies of the plastic magnitudes; noise keyed
by (block evaluation seed, bank, fixed chunk of 64 images): the same for an image in every learner, mode, checkpoint
and graph, independent of training. After 6,000 decisions: the final bank (2,048, 256 per class) and the variant bank
(512, the report's generalisation variants), if included (§10). Predictions and image ids saved; weight digests at
every checkpoint.

## 6. Measures

- **Primary:** L = trapezoidal area of greedy accuracy over the checkpoints / 6,000 (a mean accuracy during
  learning; not Study S's E1).
- **Contrasts (primary, 4):** G_U(t, f) = mean over blocks of L(connectome, t, U) - mean over the family's graphs and
  blocks of L(null, t, U), t in {where, loom}, f in {b, c}.
- **Reference and interaction:** G_N(t, f) likewise in N; I(t, f) = G_U - G_N (negative: the relative advantage
  shrinks under common experience). Not a "share explained by exploration".
- **Secondary, declared:** A(0), A(6000) - A(0), L - A(0), final IID accuracy; descriptive: N's E1 (reward rate during
  training), U's journal reward rate, variant accuracy, and the diagnostics (target-policy entropy, max probability,
  action frequencies, baseline, rho, raw and RMS-normalised update norms, clipped fraction, active outputs).

## 7. Uncertainty

10,000 bootstrap replicates (seed 20261021): blocks resampled with replacement (one draw for every graph, mode and
task); the nulls of each family resampled as whole graphs; for each selected block and task, the evaluation images
resampled within their 8 classes, identically for every checkpoint, graph and mode (final images by their own ids).
**98.75 % percentile intervals for the four G_U** (Bonferroni over 4; coverage depends on the bootstrap
approximation); 95 % for G_N, I and the rest. Per-block values shown. With 3 or 6 blocks the training variability is
estimated coarsely; a narrow interval is not over-read.

## 8. Decision rules

**Practical margin delta = 0.03 on L** (3 points of mean accuracy during learning; not on E1, not on final accuracy).
Motivation: Study S's connectome-null gaps were 12-19 points of E1 with null SDs of 1.6-1.9 points; 3 points is small
against that effect and about two null SDs.

| 98.75 % interval of G_U(t, f) | reading |
|---|---|
| entirely above +delta | practical advantage under common experience |
| entirely inside [-delta, +delta] | compatible with practical equivalence at this margin (subject to sensitivity) |
| entirely below -delta | practical disadvantage under common experience |
| otherwise | practical importance not settled at this precision |

A general conclusion over both tasks and both families requires the four readings to agree; an inversion is never
hidden by an average. **Instrument validity before speaking of equivalence or absence of advantage:** identical U
journals; validated correction and controls; complete runs; an N reference coherent with the phenomenon; informative
U: for the task considered, the lower bounds of the 95 % intervals of the final U accuracy exceed 0.25 for the
connectome and for each family mean (chance 0.125).

| situation | consequence for the report |
|---|---|
| advantage in N and in U, informative learning | differences in collected experience do not, alone, explain the advantage |
| advantage in N, practical equivalence in U, no floor effect | the advantage depends on the interaction protocol; exploration and its effects on optimisation remain a compatible explanation |
| U degrades everything near chance | the control lacks the sensitivity to decide the mechanism |
| advantage differs between tasks or families | the conclusion is restricted to the supported comparisons |
| N's advantage not reproduced | analyse the non-replication first; a disappearance is not attributed to U |

## 9. Execution

One GPU process per graph (all its tasks, modes and blocks), several at once (§10); a failed process restarts from
its last checkpoint (at most two attempts, technical reasons only); no new graph starts if it could not end before
H20 (2026-10-21T07:32Z); no run is shortened or stopped for its performance; no comparative score is computed or
looked at before every planned run is complete. Technical follow-up only (runs done, time, memory, backups,
incidents).

## 10. Plan, cost and pilot results

- **Pilot (PILOT.md, development graphs, block d0): every declared criterion passes.** U final accuracy 0.635
  (where) and 0.480 (loom) >= 0.25; supervised 1.000; label-independent reward 0.130 / 0.125 <= 0.20; frozen learners'
  predictions identical at all checkpoints.
- **Cost:** one graph (24 learners) takes about 1,032 s on its own; concurrent processes give no throughput.
- **Selected: plan P1 with the variant bank**: 41 graphs (connectome + the 20 b and 20 c of the selection), 6 blocks,
  2 tasks, 2 modes = 984 complete learnings; projection 11.75 h, 14.7 h with the 1.25 factor (<= 15 h).
- **Graph order:** connectome, then b and c alternating in their selection order (a complete unit per graph).
- **Seeds, banks and journals:** `manifests/data.json` (sha256 of every block file), `journals/*.csv`.
- **Implementation notes before the freeze:** DEVIATIONS.md D0-D6.
