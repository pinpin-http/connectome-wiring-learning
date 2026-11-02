# REVISION48: frozen protocol

**Frozen 2026-10-24 (time and commit in RUN_LOG.md)**, before any run of an analysed graph: only references, checks,
calibration, benchmarks and a tuning pilot on development graphs exist. A local, dated freeze in git, not a public
preregistration. Authority: `docs/revision48/DOSSIER_FR.md` (the owner's dossier); configuration
`configs/revision48/campaign.yaml` (`protocol_status: frozen`); base campaign `docs/experience_commune/PROTOCOL_FROZEN.md`
(EC), whose U mode, learning rule, evaluation and measure L are reused unchanged unless stated here. After the freeze,
any change is a new dated version in RUN_LOG.md with its reason and the list of affected results. Every compared arm
runs on this server (Vast 57548409, RTX 3090, torch 2.9.1+cu128) with the same batch composition; historical results
are context only.

## 1. Questions

- **R1, primary:** does the connectome's learning advantage over its nulls persist when every graph receives the same
  tuning budget (§4)? Measured directly as the gap after tuning and its change D = G_tuned − G_reference.
- **R4:** the same comparison when the target class replaces the reward signal (supervised arm), nothing else changed.
- **R2:** the same comparison under log-normalised weights (§5).
- **R3:** the same comparison against nulls that keep the connectome's 101 autapses (§6).
- R8 (reward-trained linear readout): not implemented; not run.

No favourable result is required; a non-settled interval is a valid result.

## 2. Graphs

`configs/revision48/graph_manifest.json` (sha256 of every file; verified on this server, all 24 identical to the
archived ones): the connectome; family **S** = `nulllat:<s>` (out-strength kept), s in 39, 21, 29, 30, 26, 20, 14, 5, 9,
34; family **E** = `nullin:<s>` (in-strength kept), s in 6, 3, 32, 28, 24, 7, 35, 39, 12, 22 (X3's selection, made
before any new result). 21 analysed graphs, a historical subset, not a new sample. Development graphs (never analysed):
`nulllat:100`, `nullin:100`, `nullin:101`. No graph is removed for learning badly.

Autapse condition: `nulllatL:<s>`, `nullinL:<s>`, the same seeds (`randomize.depth_stratified(..., keep_loops=True)`):
the connectome's 101 autapses kept with node, weight and sign, no new loop, no multi-edge. Validation
(`artifacts/revision48/nulls/validate.json`): 22/22 pass (degrees, sides, strata, sign and weight multisets, the family's
strength constraint, autapses identical, 0 duplicate pairs); edge overlap with the connectome 0.053-0.054, with the
historical same-seed null 0.828-0.829 (descriptive). The autapse condition's connectome is the reference run.

## 3. Blocks, tasks, data

Tasks `where` and `loom` (8 classes, chance 0.125), artificial-eye input from pixels, mode **U** (EC: uniform imposed
actions and the corrected signal; the action journal is identical for every graph and condition). Blocks:
main r0-r5 (evaluation of every condition), tuning t0, t1 (tuning only, never in the main evaluation), development rd0
(references, calibration, pilot). Each block fixes the readout projection, stimuli, noise, U journal and banks.
Seeds and sha256 of the 18 files: `artifacts/revision48/manifests/data.json` (re-hashed on this server: identical).
U journal reward rates 0.119-0.134.

## 4. Conditions and the tuning procedure

Reference setting (EC): gain 0.18452 × gamma 1, eta 0.001, tau 2, raw synapse counts, 16 ticks, plasticity on the
inputs of the 2,227 readout neurons, fixed final projection; every condition restarts from the graph's anatomical
weights. Horizon 6,000 decisions; checkpoints 0, 250, 500, 1,000, 2,000, 4,000, 6,000; final and ood banks after 6,000.
One GPU process per (graph, condition) with 12 learners = 6 main blocks × 2 tasks, block-then-task order, for every
graph and condition.

| condition | graphs | change from the reference |
|---|---|---|
| reference | 21 analysed | none |
| tuned | 21 analysed | each graph's sealed selected trial (trial 1 selected: the reference run is reused) |
| supervised | 21 analysed | the target class replaces the reward signal |
| lognorm | 21 analysed | log weights (§5) |
| autapse | 20 autapse-preserving nulls | the nulls only; connectome = reference |

**Equal-budget tuning:** for every analysed graph the same 9 trials (gamma, eta, tau): 1 (1, 0.001, 2), 2 (0.063,
0.001, 2), 3 (0.25, 0.001, 2), 4 (1, 0.0003, 1), 5 (1, 0.003, 4), 6 (0.063, 0.0003, 4), 7 (0.063, 0.003, 1),
8 (0.25, 0.0003, 4), 9 (0.25, 0.003, 1), on the same blocks t0, t1, horizon 1,500, checkpoints 0, 250, 500, 1,000,
1,500; one process per (graph, gamma) with 3 trials × 2 tasks × 2 blocks. Score: mean over both tasks and both tuning
blocks of the validation area (trapezoid of greedy accuracy on the tuning block's progress bank / horizon); one choice
per graph for both tasks. Ties: lowest trial. A trial with any non-finite score or weight is invalid and not replaced;
a graph without a valid trial is reported as not judgeable. The selection (`artifacts/revision48/tuning/selection.json`)
is sealed with its sha256 by the runner when all 63 tuning batches are complete, before any tuned run; no main or final
score is consulted. Bound check on the development graph (dossier §4): development areas 0.18-0.49, neither all at floor
nor all at ceiling; the grid is kept.

## 5. Log weights (R2)

Operator: log1p of each summed signed count, then every neuron's incoming absolute mass normalised to 1
(`learn.lognorm_operator`). **Base gain 149.97884186649117**, by the declared rule (`calibrate_gain.py`, development
graphs nulllat:100 and nullin:100 only, block rd0): the gain at which their mean untrained readout activity with log
weights equals that with raw weights at 0.18452 (3.2389 spikes per image and readout neuron), bisection on log gain in
[2, 200], tolerance 2 % (reached 0.51 %). The historical log_norm gain 20 gives 1.045 (recorded only).

**Frozen product: exact64.** With fractional weights the float32 CUDA CSR product is not bit-repeatable on this GPU
(every repeated call differs; two continuous 500-decision runs diverge; `torch.use_deterministic_algorithms` does not
help), so resume was not exact (checks.py item 3). The rule (csr if repeatable, else a fixed-order product) was revised
before the freeze, from cost only, to "the fastest bit-repeatable product that passes repeatability and resume":
segment passes but costs 3.7× raw per decision (the module would not fit); **exact64** (`learn.Exact64Operator`:
weights on the 2^-40 grid — a no-op for these weights — float64 CSR product, exact for binary inputs whatever the
order, rounded once to float32) passes repeatability and resume at the frozen gain and agrees with segment to 1.3e-7.
Raw-weight conditions keep csr (integer counts: exact in float32 whatever the order; repeatability and resume pass).
The plastic product is the fixed-order segment reduction in every condition (EC).

## 6. Measures and analysis

L = trapezoidal area of greedy accuracy over the main checkpoints / 6,000, in points. Per condition c, task t, family
f in {S, E}: **G_c(t, f) = mean over main blocks of L(connectome) − mean over the family's graphs and blocks of L(null)**.
**Primary: the four G_tuned**, 98.75 % percentile intervals (Bonferroni over 4). **Change after tuning:**
D(t, f) = G_tuned − G_reference, bootstrapped directly and paired, 95 %. **Sensitivity:** G in supervised, lognorm and
autapse, four contrasts each, 95 %, read separately; no global claim over all conditions is made (it would need a
correction declared here; none is). Mechanism comparisons are secondary or exploratory. Also reported: A(0), final
accuracy, failures, per-graph distributions, per-block values.

Bootstrap: 10,000 replicates, seed 20261024; main blocks resampled with replacement; the nulls of each family as whole
graphs (autapse nulls paired with the analysed nulls by seed); evaluation images within their 8 classes; the same
indices for every graph and condition. Replicate count checked on synthetic data (`validation/bootstrap_stability.json`):
bounds vary by at most 0.10 point across bootstrap seeds. Analysis code checked on synthetic curves with a known
advantage, a known equivalence and graph spread (`tests/unit/test_revision48.py`).

**Reading, margin delta = 3 points** (EC's, for continuity; not a universal measure of biological importance):

| interval | reading |
|---|---|
| entirely above +3 | practical advantage in this plan after tuning |
| entirely inside ±3 | practical equivalence at this margin |
| entirely below −3 | practical disadvantage in this plan |
| otherwise | not settled |

A change after tuning is read from D's interval, never from a significant gap in one arm and a non-significant one in
the other. Limits: with 10 nulls the smallest one-sided rank p is 1/11; six blocks are not six connectomes; intervals
are conditional on the plan. `scripts/revision48/analyse.py` produces the tables mechanically (`--partial` labels
missing runs); the orchestrator does not interpret them.

## 7. References and checks (all on development graphs; outcomes in RUN_LOG.md)

| item | outcome |
|---|---|
| unit tests (test_revision48, test_experience_commune) | pass (57) |
| default = explicit hyperparameters | pass |
| repeatability raw csr; resume raw | pass |
| evaluation read-only, repeatable; frozen learners identical | pass |
| lognorm csr repeatability over 500 / resume | **fail** → exact64 (§5) |
| lognorm exact64 at the frozen gain: repeatability, resume | pass |
| supervised must learn (final ≥ 0.80, both tasks), nulllat:100 and nullin:100 | pass, pass |
| frozen must not learn (identical predictions) | pass, pass |
| label-independent reward near chance (final ≤ 0.20) | pass, pass |
| restore probe (backup, download from the Drive, sha256, resume = continuous) | pass |
| graph rebuild (24 identical), builder regression, autapse nulls (22 pass), data (18 identical) | pass |

A failed reference stops the module concerned.

## 8. Execution, deadline and reductions

T0 = 2026-10-24T18:22:14Z (Vast start_date); limit T0 + 48 h = 2026-10-26T18:22:14Z; no new run starts after
2026-10-26T14:22:14Z (4 h reserve for analysis, archive, backups). Runner `scripts/revision48/campaign.py`, detached
from SSH and Claude, **1 GPU process** (throughput 3.53 / 3.34 / 3.33 main runs per hour with 1 / 2 / 3 concurrent
processes; VRAM 3.4 GB each), in the order tuning → selection sealed → reference → tuned → supervised → lognorm →
autapse (dossier §8 priority). Benchmarks (development graph): tuning batch 317 s, raw main run 1,009 s, lognorm main
run 1,685 s; projected 38.6 GPU-hours (tuned counted in full) for about 40.4 h available at the launch.
**Reduction rule:** no module starts unless all of its remaining runs fit before the last-start time (wall per run =
the module's benchmark, raised by the observed slowdown of this campaign's raw main runs); otherwise the module is not
started, recorded and notified. By the priority order this can only remove the last modules (autapse, then lognorm).
No graph, block or condition is ever selected or dropped from a score. A failed run is retried once from its last
checkpoint, for technical reasons only. STOP: `artifacts/revision48/STOP`. Activity reports (counts and times, never a
score) after every job state change, pushed to the Drive with each finished run; full backup after each phase and every
30 min.
