# Study S — does the measured fly wiring help a three-factor learner on controlled visual tasks?

*Secondary protocol. Declared 2026-10-11 10:59 UTC (12:59 CEST), before any code or run of this study. Owner's
mandate of 2026-10-11 10:57 UTC: "find another angle, leave Mario if it brings nothing, design a secondary
protocol that determines an advantage or not, without bending the numbers; stay neutral and try anyway".
Governed by PROTOCOL.md §13.2 (Level A, controlled diagnostic tasks), §15 (Study II, architecture and learning),
§9.5 (shared-parameter comparison), §22 (statistics); amendments are appended at the end, never edited in place.*

## 1. Why a second protocol

The 48-hour campaign (`report/MISSION_REPORT.md`) compared the connectome with structural nulls on Super Mario
Bros. It found no advantage, but on the play measures it compared **failures**: no wiring produced a competent
controller, so sample efficiency, robustness and generalisation in RL could not be measured
(`report/PROTOCOL_ASSESSMENT.md` §2). The one measure on which every graph learns, supervised learning speed on
game frames, showed the connectome inside the null distribution (RL_STUDY §24, §26, §27).

Study S keeps everything that is validated and replaces the task:
- **Kept:** the network, the sweep regime, the eye interface, the fixed readout, the local three-factor rule and
  the 80 validated null realisations.
- **Replaced:** Mario becomes controlled visual tasks, learnt **from reward**, that every graph can learn, with
  room between chance and ceiling.

It asks the RL question where it can be answered: does the measured wiring learn faster from reward, generalise
better, or degrade less under damage?

## 2. Questions (two-sided; no direction is expected)

- **Q1, sample efficiency.** Does the connectome learn a reward-based visual discrimination faster than its nulls?
- **Q2, generalisation.** After learning, does it handle unseen stimulus variants better or worse?
- **Q3, robustness.** After learning, does it lose more or less when 10% of its synapses are deleted, or when its
  noise doubles?

## 3. System (unchanged unless stated)

- **Network.**
  - MaleCNS LIF network: lin_raw weights, gain 0.18452, noise 0.02.
  - Sweep regime: reset to rest, 16 ticks per decision.
  - One eye, through `RetinotopicVision`: each trial is the previous frame, then the stimulus frame (the encoder's
    ON/OFF transient plus static terms).
  - Fixed random readout of the 2,227 readout neurons to the 8 outputs (`FixedRandomProjection`, block_seeds(3)).
  - Plasticity only on synapses onto readout neurons, bounded in [0, 2 m0].
- **Learning rule.** `eprop_readout` in its reward form (RL_STUDY / sweep.py, as `common_stream.py` but with a
  policy-gradient factor):
  - centred eligibility (κ 0.2, ebar rate 0.05);
  - learning signal L = R^T g / τ with g = (onehot(a) − π)(r − b_c);
  - RMS-normalised, η 0.001, τ 2;
  - r = 1 for the correct action and 0 otherwise;
  - b_c is a per-class running mean of r (rate 0.1).
- **Action.** Sampled from softmax(scores / τ), with an exploration floor ε = 0.1 (uniform action).
- **Learners.** Per task and graph: 3 readouts × 2 noise seed sets = 6 learners. Every graph sees the same trial
  streams (class sequence, jitters, sampling uniforms, exploration draws): common random numbers.

## 4. Tasks (8 classes each; chance = 0.125)

Frames are 240 × 256 RGB, with the playfield below the 32 HUD rows that the interface crops. The background is
mid-grey (0.5); disks are dark (0.1).

- **T-where (retinotopy).**
  - Stimulus: the background, then a disk of radius 16 px.
  - Classes: 8 centres on a 4 × 2 grid, x ∈ {40, 100, 160, 220} and y ∈ {90, 180}.
  - Jitter: ±6 px uniform per trial.
- **T-loom (a fly-relevant motion cue: looming vs receding).**
  - Classes: 4 quadrant centres (x ∈ {70, 190}, y ∈ {95, 185}) × {looming: radius 10 → 18; receding: 18 → 10}.
    The previous frame holds the first radius, the stimulus frame the second.
  - Jitter: ±6 px.
  - Class i's correct action is output i.
- **Training.** T trials, classes balanced in blocks of 8. T = 3,000 by default; fixed by the pilot (§7).

## 5. Evaluation (frozen weights, greedy argmax over the 8 outputs, noise on, fixed seeds shared by all graphs)

- **A_test.** 8 classes × 25 fresh jitter draws of the training distribution.
- **A_gen.** The same classes under unseen transformations, mean over three transformation sets:
  - T-where: radius 10; radius 24; disk luminance 0.3 (lower contrast).
  - T-loom: radii 8 ↔ 14; 14 ↔ 24; luminance 0.3.
- **A_rob.** A_test with 10% of **all** edges of the graph deleted at random (deletion seed shared across graphs by
  index; plastic edges keep their learnt magnitudes if not deleted).
- **A_noise.** A_test at noise_std 0.04 (×2).

## 6. Endpoints and inference

For each graph, each value is the mean over its 6 learners.

| endpoint | definition | role |
|---|---|---|
| E1 | AULC: mean over all training trials of the correct-action indicator (sampled actions) | **primary** (Q1) |
| E2 | A_test | secondary |
| E3 | A_gen | secondary (Q2) |
| E4 | A_test − A_rob: the drop under 10% synapse deletion | secondary (Q3) |
| E5 | A_test − A_noise: the drop under doubled noise | secondary (Q3) |

All five endpoints are measured for both tasks.

- **Null distributions.** Rung b is `nulllat:0-39`: nodes, degrees, signs, weight multisets, depth strata and
  sides kept. Rung c is `nullin:0-39`: rung b plus every neuron's in-strength. All 80 are built and validated
  (RL_STUDY §24.1, §26.1).
- **Per endpoint and rung.** Rank of the connectome among 41 graphs; z = (connectome − null mean) / null SD;
  two-sided empirical p = min(1, 2 · min(#≤ + 1, #≥ + 1) / (K + 1)).
- **Multiplicity.** Westfall–Young max-|z| per rung, computed within each family: the 2 primary endpoints, then
  the 8 secondary. The adjusted p of endpoint j is (1 + #{nulls whose max |z| over the family ≥ |z_c,j|}) / (K + 1).
  Each null's z is computed against the other 39 (leave-one-out).
- **Decision rule.**
  - **An advantage (or disadvantage) is claimed for an endpoint only if its adjusted p < 0.05 against both rungs,
    in the same direction.**
  - "Typical" means the connectome lies within the central 80% of both null distributions.
  - Anything else is reported as "no difference shown at this power". With K = 40 and the leave-one-out
    max-statistic, the smallest attainable adjusted p is 1/41.

## 7. Instrument checks and pilot (on dedicated pilot nulls, never on the connectome or the analysed nulls)

- **Pilot graphs.** `nulllat:100` and `nulllat:101`, built for this purpose and excluded from every analysis.
- **Default.** T = 3,000. The pilot accepts the first T in {3,000, 6,000, 1,500} (tried in that order) for which,
  on both tasks and averaged over the two pilot graphs:
  1. **learnable:** mean A_test ≥ 0.50;
  2. **not at ceiling:** mean E1 ≤ 0.85.
  If none works, η = 0.003 is tried with the same sequence. The choice is recorded and frozen before the
  connectome or any analysed null is run.
- **References** (pilot graph `nulllat:100`, the frozen T and η):
  - **positive:** a supervised learner (g = onehot(target) − π) reaches A_test ≥ 0.80;
  - **negative:** shuffled-reward learners (r drawn independently of the action, at the same rate as the true
    learners' running reward) stay at A_test ≤ 0.20;
  - a frozen network (η = 0) stays at A_test ≤ 0.20.
  If a reference fails, the instrument is not valid: the failure is reported and no reading is made.

## 8. Order, budget and stop rule

1. Pilot nulls built (CPU).
2. Pilot and references.
3. Freeze.
4. Connectome, then rung b, then rung c: one GPU job per graph with both tasks in one 12-brain batch,
   ≈ 4-5 min per graph, ≈ 6-7 h for 81 graphs.
5. Readings by `scripts/diagnostics/study_s_reading.py`, written before the main runs.

If time runs short, rung c is reduced to its first 20 nulls (seeds 0-19). Rung b is never reduced. This is
recorded.

## 9. What would count, fixed now

- **An advantage of the measured wiring:** E1 higher, E3 higher, or E4 / E5 lower, with an adjusted p < 0.05
  against both rungs.
- **A disadvantage:** the same, in the opposite direction.
- **Neither:** "no difference shown". This is not equivalence; the percentile of the connectome is reported.
- **Interpretation limits:**
  - one connectome;
  - tasks chosen by the lead (two task families, to avoid resting on one);
  - rung b differs from the connectome in activity (rung c much less);
  - the eye interface and the sweep regime are modelling choices that hold for every graph.

## Amendments

*(appended below with their UTC time; none yet)*

### A1 (2026-10-11 11:47 UTC): the declared stimuli fail the instrument check; large stimuli replace them — before any analysed graph is run

**What the pilot showed (pilot graph `nulllat:100` only).**
- A smoke run (T = 1,500) left the true learners at chance (A_test 0.125-0.126).
- The supervised positive reference also failed (0.138 / 0.138): **the instrument is not valid as declared.**
- **Cause, measured untrained on `nulllat:100`:** the 16-px disks are barely decodable from the 2,227 readout
  neurons' spike counts (ridge classifier, 5-fold): 0.171 for T-where and 0.225 for T-loom (chance 0.125). A
  dark block of 64 × 104 px is decoded at 1.000.
- **A mistake of mine, recorded:** a first decodability check used a logistic regression on features divided by
  their SD (zero for constant neurons). It returned exactly 0.125 everywhere and I briefly concluded the network
  was "blind" to every stimulus. A ridge decoder showed that conclusion was an artefact; it is withdrawn.

**New tasks, used from here on.** Background white (1.0), stimuli dark (0.0), playfield below the 32 HUD rows.
- **T-field** (replaces T-where).
  - 8 classes: dark blocks of 64 × 104 px on a 4 × 2 grid, centres x = 32 + 64 i and y = 84 + 104 j.
  - Jitter ±8 px; the previous frame is all white.
  - A_gen sets: blocks of 48 × 80; luminance 0.5; jitter ±16.
- **T-loom** (large).
  - 8 classes: 4 centres (x ∈ {64, 192}, y ∈ {85, 185}) × {looming: radius 30 → 90; receding: 90 → 30}.
  - Jitter ±8.
  - A_gen sets: radii 24 ↔ 72; luminance 0.5; radii 40 ↔ 110.
- **Untrained `nulllat:100`, ridge decodability.**
  - From the readout neurons: 1.000 (field), 1.000 (loom); generalisation variants 0.99-1.00.
  - From the 8 fixed readout scores: 0.69 / 0.69, and 0.51-0.68 for the variants.

**Unchanged:** everything else in §3-§9, including the §7 pilot rule (T in {3,000, 6,000, 1,500}, then η 0.003)
and the references. Since the endpoint names E1-E5 do not depend on the task, they are reported as "field" and
"loom".

### Pilot step 1 (2026-10-11 12:07 UTC): T = 3,000 fails the learnability check by a small margin; T = 6,000 is tried as declared

Pilot graphs only (`study_s/nulllat10{0,1}-pilotT3000.json`).

| task | A_test (nulllat:100 / nulllat:101) | mean | needed |
|---|---|---|---|
| field | 0.586 / 0.409 | 0.497 | ≥ 0.50 |
| loom | 0.476 / 0.500 | 0.488 | ≥ 0.50 |

E1 is 0.21-0.28 (≤ 0.85, not at ceiling). By the §7 rule, T = 6,000 is tried next; the threshold is not moved.

### A2 (same time): budget

At T = 6,000 a graph costs ≈ 15 min (12 brains), so 81 graphs would take ≈ 20 h. Using the §8 provision, declared
now (before the T = 6,000 pilot result is seen): **rung c is analysed on nullin:0-19 (K = 20)**. Rung b keeps all
40. With K = 20 the smallest attainable adjusted p is 1/21 = 0.048. Run order: pilot → references on nulllat:100
→ connectome → nulllat:0-39 → nullin:0-19.

### Pilot step 2 (2026-10-11 12:37 UTC): T = 6,000 passes; frozen

`study_s/nulllat10{0,1}-pilotT6000.json`.

| task | A_test (nulllat:100 / nulllat:101) | mean | E1 range |
|---|---|---|---|
| field | 0.648 / 0.485 | **0.566** ≥ 0.50 | 0.23-0.28 |
| loom | 0.595 / 0.510 | **0.553** ≥ 0.50 | 0.29-0.31 |

Not at ceiling (E1 ≤ 0.85). **Frozen: T = 6,000, η = 0.001.** Nothing has been run on the connectome or on any analysed
null.

### A3 (same time): the shuffled-reward reference's reward rate

§7 asks the shuffled learners to be rewarded "at the same rate as the true learners' running reward". The reference
run has no true learners, so the rate is set to the pilot's true-learner mean reward at T = 6,000: **0.256 (field)
and 0.298 (loom)**, drawn independently of the action (`--shuffle-rate-field/--shuffle-rate-loom`).

### References (2026-10-11 15:00 UTC, read before any analysed graph): the instrument is valid

`study_s/nulllat100-refs.json` (T 6,000, η 0.001; 2 h 12 min for 36 brains).

| reference | field A_test | loom A_test | needed | verdict |
|---|---|---|---|---|
| supervised (positive) | **1.000** | **1.000** | ≥ 0.80 | pass |
| shuffled reward, rates 0.256 / 0.298 (negative) | 0.123 | 0.125 | ≤ 0.20 | pass |
| frozen, η = 0 (negative) | 0.096 | 0.103 | ≤ 0.20 | pass |

Descriptive: supervised learners generalise (A_gen 0.98 / 0.96) and resist doubled noise (A_noise 1.00 / 0.99), but
lose most of their accuracy when 10% of all synapses are deleted (A_rob 0.31 / 0.34): the deletion test is severe,
which leaves room to see differences in E4. The main runs started at 14:48 UTC (connectome, then nulllat:0-39, then
nullin:0-19).

## Result (2026-10-12 06:06 UTC reading; recorded 06:09 UTC): the connectome learns from reward faster than every null, at both rungs

`artifacts/runs/study_s/reading.json` (`study_s_reading.py`, unchanged since the declaration). 61 graphs ran, none
failed: the connectome, nulllat:0-39 (K = 40), nullin:0-19 (K = 20, A2). T = 6,000, η = 0.001.

**Primary endpoint E1 (area under the learning curve, true reward), declared verdict ADVANTAGE on both tasks.**

| task | connectome | rung b: null mean ± SD (range) | rank, z, adj. p | rung c: null mean ± SD (range) | rank, z, adj. p |
|---|---|---|---|---|---|
| field | **0.403** | 0.249 ± 0.016 (0.219-0.284) | 41/41, +9.5, 0.024 | 0.284 ± 0.019 (0.256-0.321) | 21/21, +6.4, 0.048 |
| loom | **0.473** | 0.282 ± 0.018 (0.248-0.313) | 41/41, +10.6, 0.024 | 0.316 ± 0.019 (0.276-0.357) | 21/21, +8.5, 0.048 |

The adjusted p-values are at their floors (1/41 and 1/21): the connectome lies above every null, by 6-11 null SDs.
The rung-c p of 0.048 passes the declared threshold only because K = 20 allows a floor of 0.048 (A2); the effect
size, not the p-value, carries the weight here.

**Secondary endpoints (declared family of 8, both rungs required).**
- Final accuracy (A_test) and generalisation (A_gen): the connectome is again first of all graphs at both rungs.
  - Field: A_test 0.755 against 0.505 ± 0.050 (b) and 0.549 ± 0.046 (c); A_gen 0.698 against 0.464 / 0.523.
  - Loom: A_test 0.797 against 0.548 / 0.609; A_gen 0.732 against 0.484 / 0.557.
  - Adjusted p is 0.024 at rung b but 0.095-0.143 at rung c (one leave-one-out null's max-|z| exceeded these z
    values in the family of 8). **Declared verdict: no difference shown.** This is the declared rule, applied
    as written; at K = 20 it cannot be met.
- Robustness, connectome typical at both rungs:
  - E4 (accuracy loss when 10% of synapses are deleted): rung b percentiles 95 / 77.5, adjusted p 0.44 / 0.98;
    rung c 85 / 15.
  - E5 (doubled noise): percentiles 45 / 17.5 (b) and 85 / 40 (c), adjusted p ≥ 0.88.
  - **No difference shown on robustness.**

**Per-learner check (descriptive).** All 12 connectome learners clear the nulls' mean, and every connectome E1
(0.337-0.525) exceeds every null graph's mean E1. Connectome learners' A_test: field 0.615-0.875, loom
0.625-0.875. Best single null learner of 240 at rung b: 0.805 and 0.750.

**What the declared reading supports.**
- Under identical parameters, on two controlled 8-way visual tasks learned from reward by a local three-factor
  rule acting on the readout neurons' inputs, the connectome's wiring learns faster than nulls that keep
  neurons, degrees, signs, weights, depth from the eye, sides, and (rung c) every neuron's in-strength.
- It is not more robust to synapse deletion or to doubled noise.
- This is a statement about these two tasks and this model. It is not a statement about Mario, where no
  learning advantage was shown (§24, §26).

## Post hoc analyses (labelled exploratory; decided after seeing the result)

**PH1: does lower activity explain it?** (`artifacts/runs/study_s/posthoc_activity.txt`)
- Untrained motor spikes per decision on game frames (`mission/graph_activity.json`): connectome 4,565; rung b
  7,778-8,608; rung c 5,771-6,243.
- Across the 60 nulls, E1 falls with activity (r = −0.70 field, −0.67 loom). That correlation is the between-rung
  shift; within each rung, activity does not predict E1 (|r| ≤ 0.31, p ≥ 0.18).
- Extrapolating the 60-null line to the connectome's activity predicts E1 = 0.308 (95% PI 0.271-0.345, field) and
  0.339 (0.300-0.378, loom). Observed: 0.403 and 0.473.
- **Reading:** activity level, as the nulls vary it, accounts for at most part of the gap. This is an
  extrapolation outside the nulls' activity range, so it is weak evidence either way. A gain change cannot match
  activity (§25.1/§25.3).

## Follow-ups (declared 2026-10-12 06:10 UTC, after the result and before running any of them)

**F1: replication with fresh randomness (confirmatory for the connectome's E1).**
- **Fresh draws** (`study_s.py --seed-block 1`, tag `rep1`): new readout projections (`block_seeds(3,
  master_seed=1)`), new noise seeds (9300+), new exploration draws and new training-stream orders and jitters.
  The test and generalisation sets and every other setting are unchanged.
- **Graphs:** the connectome, then nulllat:0-3 and nullin:0-3, in that order.
- **Declared reading.** E1 replicates on a task if both hold:
  - the replicated connectome's E1 is above every replicated null;
  - it lies more than 3 SD above the mean of the original null distributions at both rungs.
- Also reported: how far each replicated null moves from its original E1 (seed sensitivity).

**F2: where does it come from? (exploratory, descriptive)** (`study_s_mechanism.py`)
- **Measures** on the untrained network and the 200 held-out test stimuli per task:
  - class-related share of the readout layer's variance (η²);
  - ridge decodability from the 8 fixed readout scores (the policy's starting view);
  - ridge decodability from 50 random readout neurons;
  - the readout layer's spike rate.
- **Graphs:** the connectome, nulllat:0-9 and nullin:0-9.
- **Reported:** the connectome's rank on each measure, and the Spearman correlation of each measure with E1
  across these 21 graphs.
- **Question:** is the connectome's readout layer more class-specific before any learning? That would point
  to the wiring between the eye and the readout neurons (e.g. retinotopic order, which the nulls destroy)
  rather than to the learning process. No verdict is attached.

**Order and budget (GPU, one at a time).** F1 connectome (≈ 15 min) → F2 (≈ 30 min) → F1 nulls (8 × 15 min).
Ends ≈ 09:00 UTC, before the 10:57 UTC deadline.

### F2 result (2026-10-12 06:31 UTC, exploratory): before any learning, the connectome's readout layer is the most stimulus-specific of 21 graphs

`study_s/mechanism.json`, `study_s/f2_reading.txt`. Untrained network, 200 held-out test stimuli per task.

| measure (field / loom) | connectome | rung b (10) mean (range) | rung c (10) mean (range) | connectome rank of 21 |
|---|---|---|---|---|
| class share of readout variance η² | 0.470 / 0.575 | 0.412 (0.378-0.457) / 0.453 (0.419-0.491) | 0.310 (0.288-0.348) / 0.346 (0.324-0.378) | 21 / 21 |
| decodability, 50 random readout neurons | 0.965 / 0.966 | 0.931 / 0.938 | 0.881 / 0.900 | 21 / 21 |
| decodability, 8 fixed readout scores | 0.720 / 0.800 | 0.669 / 0.693 | 0.563 / 0.577 | 18 / 21 |
| readout spikes per decision | 4,576 / 4,575 | 8,193 / 8,189 | 6,089 / 6,088 | lowest |

**Reading (exploratory, no verdict).**
- Before any learning, the connectome's readout neurons carry more stimulus-specific signal than every one of
  these 20 nulls, and they do so at the lowest activity.
- The class signal does not explain the rungs' order on its own. Rung c nulls have less class signal than
  rung b but learn faster (E1 0.284 vs 0.249). Within a rung, more class signal tends to go with faster learning
  (Spearman with E1: within c, +0.82 field / +0.48 loom; within b, +0.38 / +0.39; n = 10 each).
- A consistent but untested two-factor description: lower activity (rung c vs rung b) and more class-specific
  output (within rungs) both go with faster learning, and the connectome is extreme on both.
- Which wiring property produces that output is not identified. Retinotopic order from the eye to the
  descending neurons, destroyed by every null here, is the leading candidate. It needs a null that keeps spatial
  order (next rung) to test.

### F1 partial (2026-10-12 06:31 UTC): the connectome's replicate, fresh seeds

`study_s/connectome-rep1.json`.
- Field: E1 0.403 (original 0.403; the per-learner values all differ, 0.352-0.423), A_test 0.759, A_gen 0.698.
- Loom: E1 0.450 (original 0.473), A_test 0.807, A_gen 0.718.
- Against the original null distributions: z = +9.5 (b) and +6.4 (c) for field; +9.4 (b) and +7.3 (c) for loom.
  Every value is above the original nulls' maxima.
- The declared criterion also needs the replicated nulls (running, ≈ 08:30 UTC).

### F3 (declared 2026-10-12 06:32 UTC, before running): complete rung c to K = 40

- **Runs:** nullin:20-39, exactly as the main runs (seed block 0, T 6,000, η 0.001), queued after F1. This uses
  otherwise idle GPU time and reverses A2's budget cut.
- **Reading:** `study_s_reading.py` unchanged (it reads every nullin file present), run **only when all 40 exist**.
  Partial sets are not read. It is reported alongside the K = 20 reading, which remains the declared primary
  reading.
- **Purpose:** at K = 40 the adjusted-p floor for rung c falls to 0.024, so the secondary endpoints (A_test,
  A_gen) can meet the two-rung rule if the effect holds.
- **Timing:** ≈ 5 h, so it runs past the 2026-10-12 10:57 UTC deadline. It completes only if the owner extends
  the mission; otherwise the queue is paused at the deadline and the partial runs stay unread.

### F1 result (2026-10-12 08:27 UTC): the learning-speed advantage replicates with fresh seeds (declared criterion met on both tasks)

`study_s/f1_reading.json` (`study_s_f1.py`). Fresh readouts, noise, exploration and stream orders.

| task | connectome replicate (original) | replicated nulls (8): range | above all? | z vs original nulls (b / c) |
|---|---|---|---|---|
| field | 0.403 (0.403) | 0.229-0.313 | yes | +9.5 / +6.4 |
| loom | 0.450 (0.473) | 0.262-0.343 | yes | +9.4 / +7.3 |

- **Declared verdict: E1 replicates on both tasks.**
- Seed sensitivity: a null's E1 moves by 0.017-0.018 on average between seed blocks (about one null SD). The
  connectome's gap to the null means, 0.12-0.19, is 7-10 times larger.

### F3 result (2026-10-12 14:43 UTC): with rung c at K = 40, final accuracy and generalisation also meet the two-rung rule

The queue paused at the 10:57 UTC deadline with nullin:20-29 done and unread. The owner extended the mission at
12:12 UTC (deadline 16:00 UTC). The queue resumed for nullin:30-39 (12:13-14:43 UTC) and then ran the declared
reading (`study_s_reading.py`, unchanged) on all 40 rung-c files. Nothing from nullin:20-39 was opened before that.
Files: `study_s/reading-K40c.json` (`reading-K40c.log`), `study_s/f3_compare.txt` (`study_s_f3_compare.py`), figure
`videos/daily/2026-10-12b_study_s_K40.png`. Rung b is unchanged (K = 40).

| endpoint | rung b (K 40): rank, z, adj. p | rung c K 20 (declared primary) | rung c K 40 (F3) | verdict K 20 → K 40 |
|---|---|---|---|---|
| E1 field | 41/41, +9.46, 0.024 | 21/21, +6.42, 0.048 | 41/41, +7.12, 0.024 | ADVANTAGE → ADVANTAGE |
| E1 loom | 41/41, +10.59, 0.024 | 21/21, +8.48, 0.048 | 41/41, +9.51, 0.024 | ADVANTAGE → ADVANTAGE |
| A_test field | 41/41, +4.97, 0.024 | 21/21, +4.45, 0.095 | 41/41, +5.02, 0.024 | no difference shown → ADVANTAGE |
| A_gen field | 41/41, +4.62, 0.024 | 21/21, +4.36, 0.095 | 41/41, +4.74, 0.024 | no difference shown → ADVANTAGE |
| A_test loom | 41/41, +4.32, 0.024 | 21/21, +3.49, 0.143 | 41/41, +4.28, **0.049** | no difference shown → ADVANTAGE |
| A_gen loom | 41/41, +4.80, 0.024 | 21/21, +3.91, 0.095 | 41/41, +4.66, 0.024 | no difference shown → ADVANTAGE |
| E4 field | 39/41, +1.64, 0.439 | 18/21, +0.85, 0.857 | 32/41, +0.72, 0.927 | no difference shown (both) |
| E5 field | 19/41, −0.22, 1.000 | 18/21, +0.72, 1.000 | 34/41, +0.72, 0.927 | no difference shown, typical (both) |
| E4 loom | 32/41, +0.82, 0.976 | 4/21, −0.67, 1.000 | 13/41, −0.55, 0.976 | no difference shown, typical (both) |
| E5 loom | 8/41, −0.95, 0.878 | 9/21, −0.14, 1.000 | 19/41, −0.07, 1.000 | no difference shown, typical (both) |

Values at K = 40 (connectome; rung b null mean; rung c null mean): A_test field 0.755; 0.505; 0.552. A_gen field
0.698; 0.464; 0.526. A_test loom 0.797; 0.548; 0.601. A_gen loom 0.732; 0.484; 0.549.

**Reading (as declared in F3, alongside the primary K = 20 reading, which stays the declared primary).**
- The connectome is above all 80 nulls on E1, A_test and A_gen, on both tasks.
- With K = 40 at both rungs, the six learning endpoints meet the two-rung rule. The declared follow-up verdict is
  **ADVANTAGE** on E1, A_test and A_gen for both tasks. Under the declared primary reading (K = 20), A_test and A_gen
  remain "no difference shown". Both statements hold; the difference between them is the number of nulls, which
  F3 was declared to fix.
- One margin is thin: A_test loom at rung c has adjusted p = 0.049 (2/41), because one null's maximum |z| across
  the eight secondary endpoints exceeds the connectome's z of 4.28.
- Robustness (E4, E5): no difference shown at either K. E4 field is high against rung b (the connectome loses more
  accuracy than 38 of 40 copies when 10% of edges are deleted, adjusted p 0.44, not significant). It also starts
  from a higher accuracy (not tested, no claim).
- The 20 added nulls match the first 20 (E1 field mean 0.2841 vs 0.2845; loom 0.3159 vs 0.3154). The rung-c
  distribution did not move; the change in verdict comes from the lower p floor, as F3 anticipated.
