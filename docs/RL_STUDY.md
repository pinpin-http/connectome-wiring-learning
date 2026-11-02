# RL study — can connectome-constrained architectures improve sample efficiency, robustness and generalisation in embodied RL?

*Opened 2026-10-07 at the owner's request: re-centre the next experiments on this
question. Chain: reliable instrument → useful behaviour → comparison of learning
trajectories → generalisation and robustness. A negative result is acceptable;
nothing here is designed to make the connectome win. Every test is declared in
this file before its data exist; results are appended under their declaration.*

---

## 1. Scientific state on 2026-10-07 (short)

**Established, within their scope**
- The full MaleCNS graph runs as a closed sensorimotor loop (gates A-F); in the
  sweep regime a local three-factor rule learns a two-cue task (96.3%, controls 50%).
- The surrogate eligibility equals the autograd gradient of the motor spike counts
  **only when motor neurons feed nothing back** (`tests/unit/test_sweep.py`); in the
  connectome they do, so the rule is a truncated (e-prop-like) gradient estimate.
- Reward-trained learners (actor-critic v2) end with a "run right" habit: 81/90
  greedy episodes = an always-run script (`docs/GENERALISATION_STUDY.md`).
- Taught by a RAM label, the same rule installs a transferable "object ahead →
  jump" contingency (criterion E, against matched coin twins) — a diagnostic of
  what the system can learn with an explicit signal, not RL.
- Under identical parameters and input the measured wiring's motor response is
  ~2x sparser than six structural nulls (PERCEPTION_CREDIT_STUDY §19).

**Exploratory observations**
- The sparseness arises at depth 2 from the excitatory wiring and is 55-76%
  carried by the in-strength distribution; the connectome's learnt change is more
  frame-specific (post hoc); learning contingency 1.7-2x larger than the nulls'
  but within the null spread.
- Taught learners rank take-off moments well (AUC 0.75-0.80) but never jump in
  greedy play.

**Withdrawn**
- "Enemy presence readable from the untrained brain (AUC 0.92)" (polluted label).
- "The fly ignores the enemy" (occlusion without an untrained baseline).
- Gate D's first pass; the 2026-09-21 chaos diagnosis.

**Declared, not run**
- Calibrated-decision play (PERCEPTION_CREDIT_STUDY §20).
- Stage D against the laterality-preserving nulls (§21).
- Null 1's rate-matched play (§18 measure 2).

**Priority blocker for an RL comparison.** No architecture has yet produced a
*useful behaviour* from the game's reward: the RL learners run and die at the
first enemy. A comparison of learning trajectories is uninterpretable until (i)
the learning procedure is verified to do what we think it does, (ii) the task is
shown to be learnable from these observations, actions and reward by *some*
learner, and (iii) the evaluation of play is reliable.

## 2. Plan

| step | question | output | gate to the next step |
|---|---|---|---|
| 1. Audit the RL procedure | does the implemented update match the algorithm we believe we run? | classified findings (error / assumed approximation / model limit); fixes behind flags with tests; old runs kept as old version | fixes verified on controlled cases |
| 2. Reliable evaluation | can the play and contingency measures carry a conclusion? | per-level, per-learner, per-architecture reporting with effect sizes and cluster-aware uncertainty; temporal dependence of frames; harness checked on references | a declared procedure that passes its positive and fails its negative references at a stated error rate |
| 3. Calibrated decision on existing checkpoints | can the rule already learnt produce useful closed-loop behaviour with an explicit, identical decision procedure? | §20 applied to Stage D connectome and nulls | decision recorded either way |
| 4. Task accessibility | with these observations, actions and budget, can a simple trainable learner acquire the behaviour (supervised; RL)? | simple reference, both controls | tells whether a failure is the task/procedure or the architecture |
| 5. Pilot RL (declared before launch) | connectome vs laterality-preserving null (+ reference) under the corrected procedure | learning curves vs interactions, competence threshold, variance, entropy, activity, weight change | decision rules of §6 below |

Development data: levels 1-1, 2-1, 4-1, 7-1 (training) and 5-1, 8-1, 3-1, 6-1
(inspected while changing labels, criteria and methods) are **development levels**.
Levels never inspected so far (from 1-2 onward, sub-levels and castles excepted
where not playable from savestates) are reserved for the confirmation; their list
is fixed in §7 before any of them is opened.

## 3. Step 1 — audit of the RL procedure (2026-10-07)

Scope: `src/flymario/training/mario_sweep.py`, `algo="actor_critic"` (v2, used by
`mario_gen`, `mario_oracle`, pilot v2), and the teacher path (Stages B-D).
Tool: `scripts/diagnostics/rl_procedure_audit.py` (`rl_procedure_audit-*-20261007.json`).

| # | item | finding | class |
|---|---|---|---|
| 1 | action selection | a ~ mu = (1 − eps) pi + eps/A, pi = softmax(R c / tau), tau 2, eps 0.1; twins share the uniforms (common random numbers) | as declared |
| 2 | policy factor | `onehot(a) − pi`, i.e. the score of pi, with a drawn from mu and no importance weight (declared in MARIO_PILOT_PLAN 2026-10-03: "the gradient factor stays onehot(a) − pi") | **assumed approximation, with an undocumented consequence** (below) |
| 3 | TD error and bootstrap | delta_k = r_k + 0.98 V(s_{k+1}) − V(s_k), applied when s_{k+1} is seen; traces include G_k before delta_k is applied (correct TD(lambda) order) | correct |
| 4 | episode ends | death, time limit (6,000 frames) and stall (150 decisions without progress) are all treated as terminal: delta = r − V(s), no bootstrap | **assumed approximation** for death; **error in kind** for the two truncations (a time limit is not a terminal state; Pardo et al. 2018) |
| 5 | traces | actor E and critic z decay by gamma·lambda = 0.882; zeroed at every restart; no TD error across a restart | correct |
| 6 | eligibility | surrogate eligibility of the motor spike counts, centred (kappa 0.2: 80% of the running-mean eligibility removed) | assumed approximation (Gate G variant B); exact gradient only without motor feedback |
| 7 | step | delta / running RMS(delta); U = E·adv, divided by a running RMS of U; multiplicative on m0; clipped to [0, 2 m0] | as declared; weights at the lower bound 0.00-0.04% in `mario_gen`/`mario_oracle` (no bound saturation) |
| 8 | saturation | learner entropy (8 actions, max 2.08 nats) fluctuates 0.2-1.8 during training; one learner collapses (pixels_ro1 → 0.001); shuffled-reward twins drift too (0.35-1.9) | observation |
| 9 | temporal information | the network is reset before each decision; the eye encodes ON/OFF transients from the previous frame (one step of motion); traces carry credit across decisions; nothing else carries memory | model limit (declared) |
| 10 | teacher path | supervised factor onehot(target) − pi, independent of the action taken: correct whatever the behaviour policy | correct |

**Item 2, quantified** (part A: one-state bandit, 8 actions, known values Q, exact
expectations, eps 0.1). With b = mu·Q the implemented expected update is
mu ⊙ (Q − b) = (1 − eps) pi ⊙ (Q − b) + (eps/A)(Q − b): the gradient of J(pi) plus a
term that does not shrink as pi saturates. Cosine with grad J(pi): 0.999 at entropy
1.95 nats, 0.95 at 1.0, 0.83 at 0.24, 0.66 when saturated; the eps/A term is
14-53% of the update. Plain ascent from the same start reaches the best action
either way when Q is known, but the implemented rule keeps growing the logits
(range 141 vs 10 after 4,000 steps) — no saturation brake — and, with noisy
advantages, keeps pushing every rarely taken action by its *estimated* advantage.
The behaviour-policy factor (1 − eps) pi(a)/mu(a) · (onehot(a) − pi) — the score of
mu — gives exactly the gradient of J(mu) (cosine 1.000). In the game, random jumps
at random moments mostly have negative estimated advantage, so the uncorrected
term keeps lowering "jump" even after it has become rare: a candidate contributor
to the "run" habit (hypothesis, not shown).

**Item 4, quantified** (part B): truncations are 22-28% of the episodes and 38-50%
of the decisions of the RL runs (`mario_gen`, `mario_oracle`), 13-15% / 30-32% in
Stage D. Treating them as terminal teaches the critic that a long episode's last
state is worth only its last reward.

**Decisions (2026-10-07).** Both corrections are added behind settings that
default to the old behaviour, so every earlier run stays reproducible and is
reported as "v2 procedure":
- `pg_factor="behaviour"`: multiply the actor factor by (1 − eps) pi(a)/mu(a).
- `bootstrap_truncation=True`: on a time-limit or stall end, delta = r +
  gamma V(s_T) − V(s_T) (the last state's value stands in for the unseen next
  state's; one extra sweep would be exact), deaths and flags stay terminal.
Version 14 ("v3 procedure") = v2 + both corrections. Tests:
`tests/unit/test_rl_factor.py`.

Item 11 (added after the table): **button hold across decisions.** An action is
held for the 4 frames of a decision; two consecutive jump decisions keep A pressed
for 8 frames, which in Super Mario Bros. is one (higher) jump, not two. Whether a
jump decision starts a jump therefore depends on the previous action, which the
network sees only through one frame of motion. Class: model/interface limit
(the hold-A mechanism test of PERCEPTION_CREDIT_STUDY §13 measured its cost for
a scripted policy). Not changed.

Verification of the corrections (2026-10-07): `tests/unit/test_rl_factor.py` (7 pass:
the behaviour factor equals the gradient of J(mu) at three saturation levels; the
v2 factor equals mu ⊙ (Q − b) and does not vanish at saturation; truncations alone
bootstrap); with the defaults the trainer is unchanged (the new weight is exactly
1.0 and the end error is the same expression; the v2 resume check is bit-identical,
`mario_sweep_resume_check_v2.json`); old checkpoints still load (settings added
since a checkpoint are accepted only at their default).

## 4. Step 2 — reliable evaluation (declared 2026-10-07 18:18, before the runs below)

### 4.1 Play: the reporting and decision rule

Unit of evidence = **level**, not start. The 8 starts of a level (delays 0-21
NOOP decisions) replay nearly the same trajectory; they are averaged within the
level. For each brain and level: d_level = mean over starts of (brain's furthest
x − mean furthest x of its 5 rate-matched random jumpers). Reported per brain:
the four d_level, how many are > 0, the mean over levels, and the number of
distinct trajectories per level (distinct furthest-x values among the 8 starts) as
an index of dependence. **PASS** (useful behaviour vs rate-matched random): d > 0 on
at least 3 of the 4 levels **and** mean d > 0. With four development levels this
is descriptive (a sign test on 4 levels cannot reach p < .05); it guards against a
pass carried by one level. The previous rule (one-sided Wilcoxon over 32 starts)
is reported alongside, never instead.

**Validation before use** (existing data only, `scripts/diagnostics/play_report.py`):
positive references must pass — the scripted experts following the take-off label
v1 and v2 and the full label with press-release (`oracle_criterion_check-*-k5`);
negative references must fail — a second random draw and always-run against the
random reference, and the 9 coin-taught twins of the Stage D arms (connectome,
nulls 0 and 2) whose greedy play was evaluated. The rule is adopted if every
positive passes and at most 1 of the 11 negatives passes; the observed error rates
are reported with the rule.

### 4.2 Contingency frames: temporal dependence

The relabel frames are taken every second emulator step (8 game frames apart) from
scripted play, then capped per category; neighbours are near-duplicates. The
per-learner difference q(object) − q(clear) is re-estimated with a block bootstrap
over runs of consecutive frames (blocks of 10 and 20 frames within a level), and
the lag-1 autocorrelation of q reported. The §18 comparison used means over
learners, so this matters for per-learner claims only; it is reported either way.

### 4.3 Calibrated decision on the existing checkpoints (PERCEPTION_CREDIT_STUDY §20)

Applied now to `mario_stageD` (connectome) and `mario_stageD_null0/1/2`, learners
and coin twins alike, each on its own graph (`threshold_play_check.py`). The
controller is "network + calibrated decision procedure": the threshold is chosen
per brain on labelled frames of the training levels only (take-off label v2,
Youden), which costs labels (counted in the output) and one sweep per calibration
frame. **Reading:** an arm shows useful closed-loop behaviour if >= 2 of 3
learners pass the §4.1 rule and its twins pass at most 1 of 3. If learners fail,
the play logs are examined for (i) timing (jumps too early or late), (ii) holding
(consecutive jump decisions), (iii) observation distribution (calibration frames
unlike the frames met in play), (iv) representation (AUC on play frames), before
any longer training.

### 4.1 result (18:21): the per-level rule is NOT adopted; play on four levels is descriptive only

`play_report.py --validate` (`play_report-validation-20261007.json`):

| reference | expected | levels > 0 | mean d (px) | per level 3-1 / 5-1 / 6-1 / 8-1 | new rule | old rule (Wilcoxon, 32 starts) |
|---|---|---|---|---|---|---|
| expert, take-off v1 | pass | 3/4 | +72 | +100 / +282 / −231 / +136 | pass | pass (22/10) |
| expert, take-off v2 | pass | 2/4 | +82 | 0 / +313 / −231 / +244 | **fail** | pass (15/9) |
| expert, full label | pass | 2/4 | +134 | +327 / −82 / −85 / +376 | **fail** | pass (19/13) |
| random draw b (v2 file) | fail | 2/4 | +16 | | fail | fail |
| always-run | fail | 0/4 | −114 | | fail | fail |
| 9 coin twins (Stage D arms) | fail | 0-2/4 | −97 to +219 | | 0/9 pass | **1/9 pass** (null 0 ro1, 19/11, carried by 5-1 and 8-1) |
| extra: random b, v1 and full files | fail | 3/4, 3/4 | +22, +38 | | **2/2 pass** | 0/2 |

Declared acceptance (every positive passes, ≤ 1 of 11 negatives passes): **the new
rule fails** (positives 1/3), and it also passes two of four extra random-vs-random
negatives. The old rule meets the declared acceptance (3/3, 1/11) but its one false
positive is exactly the dependence problem it ignores.
**Observation.** Even the RAM experts lose heavily on 6-1 (−231 px in all three
files): the benefit of a correct jump rule is level-specific, and four levels are
too few units for a general play verdict. Only eight level savestates exist
(1-1 … 8-1, plus 1-4 and 2-1 variants), all already development data.
**Decision.** Play on these four levels is reported per level and treated as
descriptive; it is not the primary endpoint of the RL pilot (§6). The old rule is
reported for continuity, with its dependence caveat. A fair play endpoint needs
more independent units (more levels or segments) and held-out ones (§7).

### 4.2 result (18:21): frame dependence widens the intervals 2-3x; every rise survives

`frame_dependence-20261007.json`. Lag-1 autocorrelation of q between consecutive
frames of a category: 0.24-0.53. Block-bootstrap 95% intervals (blocks of 20
consecutive frames) are 2-3x wider than the naive ones (e.g. connectome ro1 threat
rise 0.104: naive [0.089, 0.120], block [0.064, 0.143]). All 24 rises (12 learners
x threat, obstacle) still exclude zero. The per-learner contingency claims stand;
the "p < 1e-4" stated in §18 overstated the evidence, which is now given by the
block intervals.

## 5. Step 4 — is the task accessible? (declared 2026-10-07 18:24, before any run)

**Question.** With these observations, these 8 actions, this reward and this budget
(20,000 decisions per learner, training levels 1-1, 2-1, 4-1, 7-1, start after 0-22
NOOP decisions), can a *simple trainable* learner acquire a useful behaviour? And if
not, is the obstacle perception or the reward/procedure?

**References** (`scripts/train_reference.py`; linear softmax policy with a
*trainable* readout — the main architectural difference from the connectome, whose
readout is fixed; no spiking network):
- **R1 supervised, pixels.** Features: the ON/OFF lamina currents the connectome's
  eye receives (3,575, standardised online), palette jitter as the generalists.
  Target: the take-off label v2 (RIGHT+A+B, else RIGHT+B), cross-entropy SGD,
  on-policy data. Asks: is the observation→action relation learnable linearly?
- **R2 RL, pixels.** Same features; actor-critic with the v3 procedure (§3: behaviour
  factor, truncation bootstrap), gamma 0.98, lambda 0.9, eps 0.1, learning signal
  (reward − 75·died)/10, linear critic (normalised LMS 0.05). Asks: can the reward
  and the procedure teach a useful behaviour from these pixels?
- **R3 RL, privileged features.** Same algorithm and reward; features from RAM (a
  reference only, never the fly's input): bias, take-off label v2, its wall/pit part,
  ground threat 16-40 px, ground threat 8-64 px, airborne. Asks: if perception were
  solved, would the reward + procedure + action interface teach jumping?

Three seeds each; the actor step c/d (d = number of features) with c in {0.01, 0.03,
0.1}, all reported, the best c per reference (on training-level performance only)
used for the frozen evaluation.

**Measures.** Training: per episode furthest x minus the always-run script's furthest
x from the same level and start (Δx, computed per start once); learning curve of
Δx (rolling 50 episodes) against decisions; jump share; entropy. **Competence
(declared):** rolling mean Δx ≥ +100 px. **Sample efficiency:** decisions to first
reach competence (censored at 20,000) and the area under the Δx curve. Frozen
evaluation on the development levels 3-1, 5-1, 6-1, 8-1 (8 delays): greedy and
calibrated threshold (§4.3 procedure), vs 5 rate-matched random jumpers, reported
per level (descriptive, §4.1).

**Reading (prompt §6 decision rules).** R3 fails → the task / reward / procedure /
interface is the blocker: fix before any architecture comparison. R3 succeeds, R2
fails → learning from these pixels is the blocker (perception or representation).
R2 succeeds and the connectome RL learners fail → localise in dynamics, readout or
plasticity before comparing nulls. R1 succeeds alone → the mapping is learnable
supervised but not from the reward: examine the transition.

**Amendment 18:27 (before any reference result was read).** The declared competence
criterion fails its negative references on existing runs: on the training levels
every learner *and every non-contingent twin* beats the always-run script by
+100 to +550 px per episode within 4,000-12,000 decisions (`learning_curves-20261007.json`,
Stage D coin twins AUC 400-560, shuffled-reward RL twins 70-520), because random
jumps clear many obstacles and the twins jump more (46% vs 22%). Δx against
always-run measures jumping at all, not useful behaviour. **Replaced** by Δx_rand =
furthest x − mean furthest x of random jumpers at the episode's own jump share,
same level and start (table: `random_jumper_table.py`, K = 3 per level x 12 delays
x 11 rates, interpolated in rate). **Competence:** rolling 50-episode mean Δx_rand
≥ +50 px. It must be validated before use: the non-contingent twins of the
existing runs must not reach it, and a scripted policy following the take-off
label v2 on the training levels (positive reference) must.

**Amendment 2, 18:31 (validation of the new metric on existing runs only; the
reference runs not yet read).** `learning_curves.py` with the rate-matched table
(`reference/random_jumper_train.json`: random jumpers gain a lot from jumping on
these levels, e.g. 1-1: 312 px at rate 0, 1,014 at 0.7):
- positive reference, the scripted take-off v2 policy: mean Δx_rand **+108.5**
  (per level 1-1 +278, 2-1 −79, 4-1 +288, 7-1 −53);
- Stage D taught learners (sampled training episodes): whole-run mean Δx_rand
  +43 to +115 in all 12 (connectome 90 / 51 / 94; nulls 50 / 109 / 102, 115 / 96 / 43,
  78 / 95 / 110); their coin twins −44 to +38;
- RL learners (v2, game reward): generalists −125 / −18 / −75, pixel arm −24 / −101
  / +6, oracle arm −47 / −52 / −27; their shuffled-reward twins −105 to −10;
  1-1-only specialists +39 / +86 / +86 (one level, memorisation possible).
"First crossing of a rolling-50 mean ≥ +50" is passed by 9 of 12 coin twins:
**invalid**. Last-10,000-decision mean ≥ +50: twins 2/12 pass (false positives),
learners 11/12. **Whole-run mean ≥ +50: twins 0/12 and RL twins 0/6 pass, the
scripted reference passes, taught learners 10/12.** Adopted: competence at budget =
whole-run (decision-weighted episode) mean Δx_rand ≥ +50 px; sample efficiency =
the Δx_rand learning curve in 5,000-decision windows and its whole-run mean (the
area under the curve over a fixed budget). Single learners stay noisy: claims
about an arm rest on its learners together and on the comparison with matched
non-contingent twins.

**Observation worth stating on its own.** Taught by the clean take-off label, the
connectome learners and the null learners alike place their *sampled* jumps better
than random jumping at the same rate on the training levels (+51 to +115 px). What
failed earlier was greedy play (argmax never jumps), not the learnt timing. Game
reward alone has produced no such learner (v2 procedure).

*Correction (18:33): the clock times written in §4-§5 were wrong (ahead of the real
time); they are replaced by the times of the commits that recorded each
declaration (`git log`), which are the authoritative record of the order of
declarations and results.*

### 5 results (18:41; `reference/train-20261007.json`, `learning_curves-20261007.json`)

Whole-run mean Δx_rand per learner (seeds s0 / s1 / s2), and the last 5,000-decision
window, 20,000 decisions each:

| reference | c = 0.01 | c = 0.03 | c = 0.1 | last window, best c | competence (≥ +50) |
|---|---|---|---|---|---|
| R1 supervised, pixels (linear) | −64 / −79 / −71 | −75 / −34 / −38 | +10 / −36 / +11 | +13 / −20 / +43 | 0 / 9 |
| R2 RL, pixels (linear) | −104 / −101 / −101 | −66 / −99 / −77 | −84 / −64 / −84 | −101 / −72 / −84 | 0 / 9 |
| R3 RL, privileged RAM features | −84 / −26 / −83 | −181 / +3 / −43 | **+28 / +73 / +47** | **+81 / +172 / +91** | 1 / 9 (c 0.1, s1) |
| *for comparison:* connectome Stage D (spiking network, fixed readout, supervised) | | | | | 3 / 3 (+90 / +51 / +94) |
| *for comparison:* connectome RL v2, pixel arm (`mario_oracle`) | | | | | 0 / 3 (−24 / −101 / +6) |

**Observations.** (1) R3: with the decisive information given as a feature, the
game reward, the v3 procedure and the 8-action interface do teach jumping better
than random at the same rate, but slowly: early windows are negative, the last ones
+81 to +172 px for all three seeds of the best step; only one seed passes the
whole-run criterion. (2) R2 and R1 fail with a linear trainable readout on the raw
lamina currents, even under supervision, although the same currents linearly
decode the take-off label offline (AUC 0.89, regularised logistic, §19 S4 of the
perception study): online learning of a raw high-dimensional linear readout did not
produce timed play within 20,000 decisions at these steps. (3) The spiking networks
taught by the same label (connectome and nulls, plastic motor synapses, fixed
readout) do produce it (+43 to +115).
**Hypotheses.** The network's expansion of the eye's input provides a representation
on which the local rule can learn timing faster than a raw linear readout; nothing
here makes this specific to the measured wiring (the nulls do as well). Learning
from reward is slow even with perfect features (~10,000 decisions before the curve
turns positive), so 20,000 decisions is a small budget for RL from pixels.
**Decision (decision rules of §5).** R3 succeeds partially → the task, reward,
procedure and interface are not the hard blocker; R2 fails → RL from pixels with a
linear readout is not a usable positive reference; R1 fails while the networks
succeed under supervision → the simple reference does not show the observation-
action relation is learnable *by a linear readout* online. The pilot (§6) therefore
compares the spiking architectures with each other under the corrected procedure,
keeps R3 as the evidence that the task is learnable from this reward, and reports
the historical v2 runs alongside. Limits: one step-size grid, linear references
only; a better-tuned or nonlinear reference could change (2).

## 6. Pilot RL (declared 2026-10-07, before launch; commit time is the record)

**Question.** Under the corrected procedure (v3: behaviour-policy factor, truncation
bootstrap), does the measured wiring learn *from the game's reward* a behaviour
better than random jumping at its own rate, and does it learn it with a different
trajectory from laterality-preserving nulls? A pilot: it sizes a confirmatory
experiment; it does not confirm.

**Arms** (`scripts/train_mario_sweep.py --version 14 --graph <g>`, runs
`mario_rl3_<g>`): connectome; `nulllat:0`; `nulllat:1` (two realisations: the spread
between them is part of what the pilot measures). Per arm, per readout
(`block_seeds(3)`): an RL learner `rl_ro{i}` (game reward) and its shuffled-reward
twin `rlshuf_ro{i}` (the reward's sign flipped at random per decision: same size,
no dependence on the action), same noise (8100 + i) and random draws. Identical
across arms: eye, observations, 8 actions, fixed readouts, reward (progress, time
−0.02, death −25; learning signal with death −100, /10), rule (eta 0.001, kappa 0.2,
tau 2, eps 0.1, gamma 0.98, lambda 0.9, critic 0.05), v3 corrections, generalist
conditions (1-1, 2-1, 4-1, 7-1; 0-22 NOOP start; palette jitter), sample seed
20261016, **budget 20,000 decisions per learner** (the budget of every earlier
run; R3 needed ~10,000 to turn positive). Remaining differences: the wiring, and
through it the activity (the lateral nulls fire ~1.8x, §19 of the perception study).
Checkpoints every 1,000 decisions; copies kept at 5,000, 10,000 and 20,000.

**Measures.** Interactions are counted in decisions (one sweep and 4 game frames
each; episodes and weight updates are reported, never substituted). Primary, per
learner: whole-run mean Δx_rand of its training episodes (§5, validated) and its
curve in 5,000-decision windows; learner minus its own twin. Competence: whole-run
mean ≥ +50 px. Secondary: entropy and jump share per window; relative weight change;
motor activity on fixed frames at 20,000; frozen play on the development levels
(greedy and calibrated threshold, per level, descriptive §4.1).

**Decision rules.** (a) No arm's learners beat their twins (learner − twin ≤ 0 in at
least 2 of 3 readouts) → nothing progresses: examine budget, reward and procedure
before more nulls (R3 shows the reward can teach with perfect features).
(b) Connectome and nulls both progress → declare a confirmatory comparison of
trajectories (more realisations, held-out conditions, §7). (c) One arm progresses
and not the others → replicate on fresh seeds before any claim. Arm comparisons
are descriptive (three learners, two null realisations); the §18 rule is not used
as a test. **Historical comparison:** the v2 pixel arm of `mario_oracle` (same
conditions, older procedure, other seeds) is reported beside the connectome arm.

## 7. Protecting the confirmation (plan, 2026-10-07)

**Development data, all eight available level savestates.** Training levels 1-1,
2-1, 4-1, 7-1 and evaluation levels 3-1, 5-1, 6-1, 8-1 have all been inspected while
labels, criteria and methods were changed (the label audit used 4-1, 5-1, 8-1; the
play rule was validated on 3-1, 5-1, 6-1, 8-1). The game integration ships no other
level start (only 1-4 and 2-1 variants). **There is no untouched level today**:
any generalisation result so far is a development result.
**Plan.** Before the confirmation, create start states for levels never opened
(e.g. the x-2 and x-3 levels) by a documented, scripted procedure (writing the world
and level numbers in RAM before a level load, then saving the state), verified only
for "the intended level loads and Mario stands at its start", without running any
agent on them; list them here, then do not open them until the frozen confirmatory
protocol is committed. If that procedure cannot be made reliable, the limitation is
stated in every generalisation claim.
**Frozen for the confirmation** (after the pilot): architecture and dynamics
(MaleCNS, lin_raw, gain, sweep regime), interface and decision procedure, plasticity
rule and RL algorithm (v3 or its successor), controls (twins, null class and number
of realisations), budget, measures, number of learners per arm, analysis rules.
Robustness is tested only on behaviours that pass competence; a policy that always
runs is not a starting point for robustness.

### 6.1 Common-stream diagnostic (declared before running; `scripts/diagnostics/common_stream.py`)

In the game, every learner's data depend on its own actions, so architectures (and
twins) see different frames once their policies differ. This diagnostic gives every
brain the *same* stream: frames of scripted play on the training levels with the
take-off label v2 as target (one fixed order, 4,000 frames), the Stage D supervised
rule unchanged, three readouts per graph, graphs connectome, `nulllat:0`, `nulllat:1`.
Every 500 frames the brains are scored on 300 fixed held-out frames of the
development levels 5-1 and 8-1: AUC of P(jump) for the take-off label. The curves
measure *supervised* learning capacity on identical data (sample efficiency of the
network + rule, not RL). Descriptive (three readouts, two null realisations); run
after the RL pilot, when the GPU is free.

### 4.3 partial results (19:10) — stopped early, a documented deviation

The four calibrated-decision evaluations ran far slower than planned (calibrated
brains jump and survive, so episodes reach the 600-decision cap; four processes
saturated the GPU): 10, 6, 13 and 11 of 32 starts after 50 minutes, i.e. 2-5 hours
to finish while the RL pilot waited. They were stopped at 19:10 to give the pilot
the machine; they are to be rerun after it. **Partial, descriptive only** (mostly
5-1; old rule per start, `play_from_log.py`):

| arm | learners (wins/losses, mean px vs random) | twins |
|---|---|---|
| connectome (10 starts) | ro1 3/6 −21; **ro2 6/2 +128**; ro3 4/6 +24 | ro1 7/3 +211; ro2 1/9 −107; ro3 2/8 −17 |
| null 0 (6) | ro1 1/2 +36; ro2 5/1 +60; ro3 4/0 +27 | ro1 5/1 +216; ro2 2/4 +56; ro3 1/5 −80 |
| null 1 (13) | ro1 3/2 +26; ro2 2/0 +14; ro3 8/4 +145 | ro1 3/0 +25; ro2 2/11 −119; ro3 8/3 +133 |
| null 2 (11) | **never jump** (0/0, 1/0, 1/0) | ro1 3/0 +12; ro2 3/7 +40; ro3 8/3 +145 |

**Observations.** Calibrated thresholds make the learners jump (training-level jump
rates 51-65%), but in closed loop on 5-1 the learners are not clearly better than
their twins, whose calibrated jumping (often 80-98% of decisions) sometimes carries
them far. Null 2's learners never jump at all on these levels: their P(jump) is
compressed (0.07-0.10, SD 0.02 on the relabel frames) and the threshold chosen on
training-level frames (0.07-0.11) sits above it on the new levels — the calibration
transfers poorly when the learnt modulation is small (failure mode iii,
observation distribution); the connectome's larger modulation (post hoc X, §18 of
the perception study) is consistent with it transferring better. **Not a result:**
partial, one level mostly, dependent starts.

### 7.1 Held-out level starts created (19:12) — not to be opened before the frozen confirmation

`scripts/make_heldout_states.py` → `integrations/SuperMarioBros-Nes/Held{W}-{L}.state`
for **14 levels never opened by any agent**: 1-2, 1-3, 2-3, 3-2, 3-3, 4-2, 4-3, 5-2,
5-3, 6-2, 6-3, 7-3, 8-2, 8-3 (water levels 2-2, 7-2 and castles x-4 left out).
Procedure: from the 1-1 state, world, level and area numbers written in RAM, timer
set to 001, Mario dies of time-up, the game reloads the target level, the state is
saved at the first controllable frame. Verification by RAM only after reloading
each saved state (`heldout_states-20261007.json`): world, level, area number, area
type (1-2 and 4-2 underground, the rest ground), player state normal, x = 40, timer
reset to 300 or 400 — 14/14 as expected. No agent was run and no screen was looked
at. **Rule from now on:** these starts are used only by the frozen confirmatory
protocol; any earlier use turns them into development data and must be recorded.
They also answer §4.1's power problem for the confirmation: 14 independent levels
instead of 4.

### 6 results — pilot stopped by the owner at 20:30 (budget not reached; resumable)

Stopped cleanly at the owner's request ("arrête tout avant 20h45"): connectome
**17,619** decisions, `nulllat:0` **11,124**, `nulllat:1` **12,358** of the declared 20,000
(checkpoints; copies at 5k and 10k; relaunch `scripts/gates/rl_pilot.sh` to resume).
The common-stream diagnostic (§6.1) was not run. `pilot_report-20261007.json`:

| arm | learners, whole-run mean Δx_rand (ro1 / ro2 / ro3) | last window | shuffled twins | learner − twin | relative weight change (learners / twins) |
|---|---|---|---|---|---|
| connectome (17.6k) | −54 / −46 / −92 | −128 / −26 / −158 (15-20k) | −307 / −266 / −341 | +253 / +220 / +249 | 0.22-0.24 / 0.09-0.12 |
| lateral null 0 (11.1k) | −39 / −69 / +15 | +66 / −204 / +7 (10-15k) | −138 / −14 / −111 | +99 / −55 / +126 | 0.19-0.20 / 0.10-0.12 |
| lateral null 1 (12.4k) | −64 / +35 / −47 | −1 / +12 / +198 (10-15k) | −40 / −291 / −670 | −24 / +326 / +623 | 0.20 / 0.11-0.13 |

**Observations.** (1) No learner of any arm reaches competence (whole-run mean ≥ +50;
best +35). (2) Under the v3 procedure the learners no longer collapse onto "run":
they converge on *jumping almost all the time* (RIGHT+A+B 72-87% of decisions in
the last window; entropy 0.3-1.7 nats) — on these levels random jumping at a high
rate is already worth +400-700 px over running (§5 table), so the reward taught the
*rate*, not the timing (Δx_rand, which removes the rate, stays ≤ 0 on average).
(3) Learners beat their shuffled-reward twins by a wide margin in most readouts
(7 of 9), because the twins degrade (−14 to −670): the reward signal does carry
information, but what it buys is captured by the rate. (4) Windows fluctuate by
±150 px between 5,000-decision windows: single learners are noisy. (5) No arm
difference can be read: unequal budgets (17.6k vs 11-12k), three learners each,
and curves inside each other's spread.
**Decision (rule (a) of §6).** No architecture learnt useful behaviour (timing beyond
rate) from the game's reward within ~11-18k decisions: comparing more nulls now
would compare failures. The next step examines the task/reward/procedure, which R3
shows *can* teach timing with perfect features after ~10k decisions:
(i) finish the declared budget (resume to 20k) and read the last windows;
(ii) separate rate from timing in the learning signal (e.g. a reward baseline per
action rate, or rewarding progress relative to a rate-matched random rollout),
declared before use; (iii) run the common-stream diagnostic (§6.1); (iv) consider a
declared supervised + RL pipeline (Stage D-style teacher then game reward), applied
identically to all arms, reported as such. **Limit:** a pilot cut short; descriptive.

## 8. Separating rate from timing: a jump cost in the learning signal (declared 2026-10-08, before running)

**Why.** The v3 pilot's learners learnt the *rate* of jumping (72-87% of decisions)
and not the timing: on these levels a random jumper at a 60-70% rate gains 400-700 px
over running (§5 table), so the game's reward pays for jumping often. A cost per
jump decision, subtracted from the learning signal only (the score and the
evaluation metric Δx_rand are unchanged), makes indiscriminate jumping expensive
and leaves well-placed jumps profitable. It changes the task the learner optimises
and is reported as such.

**Test on the references first** (`scripts/train_reference.py --jump-costs 0,3,6`,
tag `jumpcost-20261008`): R3 (RL, privileged RAM features) and R2 (RL, pixels),
actor step c = 0.1 (R3's best on 2026-10-07), three seeds, 20,000 decisions, costs
0 (replicates 2026-10-07: the same seeds and code must give the same numbers), 3 and
6 px-equivalents per jump decision; learning signal (reward − cost·jumped − 75·died)/10.
**Measures:** whole-run mean Δx_rand and its 5,000-decision windows, competence
(whole-run mean ≥ +50), jump share per window. **Decision rule:** a cost is adopted
for the spiking pilot (v4) if R3 with that cost reaches competence in ≥ 2 of 3 seeds
*and* its last-window Δx_rand is not lower than at cost 0 in ≥ 2 of 3 seeds; the
smallest such cost is chosen. If no cost qualifies, the jump cost does not separate
rate from timing here and the next candidate (reward relative to a rate-matched
random rollout) is examined instead. R2 is reported, not used for the decision
(the linear pixel learner failed at cost 0).

### 8 result (12:51) — the jump cost is NOT adopted

`reference/train-jumpcost-20261008.json` (whole-run mean Δx_rand, seeds s0 / s1 / s2;
jump share in the last 5,000 decisions):

| reference | cost 0 | cost 3 | cost 6 |
|---|---|---|---|
| R3 RL, privileged RAM | **+100 / +98 / +35** (jump 0.25 / 0.88 / 0.85) | +162 / +6 / −9 (0.23 / 0.10 / 0.10) | +30 / +8 / −1 (0.08 / 0.05 / 0.05) |
| R2 RL, pixels | −109 / −68 / −47 (0.50 / 0.65 / 0.40) | −68 / −18 / −38 (0.19 / 0.12 / 0.19) | −47 / −44 / −44 (0.21 / 0.13 / 0.17) |

**Decision rule applied:** cost 3 reaches competence in 1 of 3 seeds, cost 6 in 0 of 3 →
neither qualifies; the jump cost is not adopted. **Observations.** (1) The cost does
what it says to the rate (R3 jump share 0.25-0.88 → 0.05-0.23) but not to the timing:
learners mostly stop jumping. (2) Without cost, R3 learns timing *despite* a high rate
(s1, s2 jump on 85-88% of decisions and still beat random jumpers at that rate by
+98 / +35 px): a high rate is not what prevents timing. **This revises the reading of
§6:** the spiking learners' failure is not explained by the reward paying for rate;
with the decisive information given as a feature the same reward teaches timing,
so the blocker is learning *which visual features* predict a good jump (credit
assignment to perception), as found earlier (perception-or-credit study).
*[Amended §11: one hypothesis among several, not the diagnosis.]*
(3) **The cost-0 control did not replicate 2026-10-07's numbers** (R3 +28 / +73 / +47
then): the declaration's expectation was wrong — each learner's episode draws are
seeded by its *position* in the batch, which changed (18 learners instead of 27), so
the same seeds met different level/start sequences. The difference measures the
run-to-run variance of a single learner (±60 px on the whole-run mean), which is
large; claims rest on several learners, never one.
**Next (decided now, declared in §9 before running):** the supervised + RL pipeline
the brief allows: start each architecture's RL from its own Stage D (supervised)
weights and ask whether the game's reward keeps, improves or loses the timing.

## 9. Supervised + RL pipeline (declared 2026-10-08, before running)

**Question.** Each spiking architecture has already learnt, from a label, to time its
sampled jumps better than random (Stage D: Δx_rand +43 to +115 on the training levels,
§5). Starting from those weights, does the game's reward (v3 procedure) keep,
improve or lose that timing — and does the answer differ between the measured
wiring and its nulls? Conclusions concern a *supervised + RL pipeline*, not RL from
scratch; the supervised stage cost 20,000 labelled decisions per learner (take-off
label v2 from RAM), the same for every arm.

**Arms** (`--version 16 --init-from <Stage D checkpoint> --graph <g>`, runs `mario_srl_<g>`):
connectome (`mario_stageD.pt`) and the three depth-stratified nulls
(`mario_stageD_null{0,1,2}.pt`, `--graph null:<s>`) — the only arms with Stage D weights;
these nulls mix the two eyes and fire ~2x (PERCEPTION_CREDIT_STUDY §19): every
difference is labelled activity/laterality-confounded. Per readout: `srl_ro{i}`
(game reward) and `srlshuf_ro{i}` (shuffled reward, the drift control), **both**
starting from that readout's taught learner `tko2_ro{i}`; fresh noise (8200 + i),
sample seed 20261017, critic from zero, generalist conditions, v3 procedure
(behaviour factor, truncation bootstrap, eps 0.1). **Budget 10,000 decisions**
(today's time), checkpoint copies at 5,000 and 10,000.

**Measures.** Δx_rand of the training episodes (§5, validated) in 2,500-decision
windows and over the run; jump share per window; learner minus twin; relative weight
change. Reference points: the same learners' last 10,000 supervised decisions
(Stage D, §5 amendment 2: connectome +82 / +83 / +126; nulls 0: +89 / +149 / +66,
1: +131 / +109 / +38, 2: +86 / +171 / +118).

**Reading.** Per arm: RL *keeps* the timing if the learners' Δx_rand over the run stays
≥ +50 (competence) in ≥ 2 of 3 readouts; *improves* it if their last window exceeds
their Stage D reference in ≥ 2 of 3; *loses* it if the last window falls to ≤ 0 in
≥ 2 of 3. The twins show what drift alone does. Arm differences are descriptive
(three learners per arm, three nulls, eye-mixing confound).

## 10. Readout ablation: the network as a fixed reservoir with a trainable readout (declared 2026-10-08, before running)

**Why.** The fixed random readout is a declared constraint of the main system
(PROTOCOL §11.2). The brief asks for a separate ablation if that constraint might
dominate behaviour. This is also the classic comparison of the connectome-reservoir
literature (Suárez et al. 2021; Damicelli et al. 2022), here in embodied RL.

**Design** (`scripts/train_reservoir_rl.py`): the *untrained* spiking network is a
frozen feature map (one sweep per decision, 16 ticks from rest, no plasticity): its
2,227 motor spike counts, standardised online, feed a linear softmax actor and a
linear critic trained exactly as the linear references R2/R3 (v3 procedure, actor
step c = 0.1, eps 0.1, learning signal (reward − 75·died)/10). Same levels, starts,
palette jitter, actions, reward. Arms: connectome, `nulllat:0`, `nulllat:1`; three
seeds each; **10,000 decisions** (time); run one arm at a time. Reference: R2 (the
same learner on the raw lamina currents; −47 to −109 whole-run Δx_rand).

**Measures and reading.** Whole-run mean Δx_rand and 2,500-decision windows; jump
share; motor spikes per decision. Readings: if reservoir learners beat R2 (raw
pixels) the network's expansion helps RL with a trainable readout; if the connectome
reservoir beats both null reservoirs on the whole-run mean in ≥ 2 of 3 seeds each
and outside their range, that is a wiring difference *in the reservoir setting*
(descriptive: three seeds, two realisations, one step size). If no reservoir
learner reaches competence, the fixed readout is not the dominant constraint for
RL from reward at this budget.

### 6 results at the declared budget (2026-10-08 13:45; resumed 12:41 from the stops of 2026-10-07)

All three runs reached 20,000 decisions (copies at 5k / 10k / 20k).
`pilot_report-20261007.json` (regenerated), whole-run mean Δx_rand per learner:

| arm | learners ro1 / ro2 / ro3 | last 5,000 decisions | twins | learner − twin | last-window dominant action |
|---|---|---|---|---|---|
| connectome | −56 / −47 / −61 | −99 / −39 / +2 | −307 / −281 / −338 | +252 / +235 / +277 | RIGHT+A+B 86%; RIGHT+B 62%; RIGHT+A 78% |
| lateral null 0 | −11 / −87 / −24 | −31 / −102 / −83 | −139 / −6 / −161 | +129 / −81 / +137 | RIGHT+A+B 87%; RIGHT+A 51%; RIGHT+A+B 87% |
| lateral null 1 | −55 / −17 / −27 | −61 / −65 / +109 | −81 / −309 / −683 | +26 / +292 / +656 | RIGHT+A+B 91%; 85%; 73% |

**Reading (rule (a) of §6, now at the full budget).** No learner of any arm reaches
competence (best whole-run mean −11 px; one last window above +50: lateral null 1
ro3 +109); learners beat their shuffled-reward twins in 8 of 9 readouts; most settle
on one jump action 73-91% of the time. The connectome is not distinguishable from
the lateral nulls (its learners lie inside the nulls' spread on every quantity).
Relative weight change: learners 0.21-0.25, twins 0.09-0.16. This confirms the
partial reading of 2026-10-07: RL from scratch on the game's reward, at 20,000
decisions, gives no useful behaviour to any of these architectures.

### 10 results (14:52) — readout ablation: no reservoir learner reaches competence

`reference/train-reservoir-{connectome,nulllat0,nulllat1}-20261008.json`,
`reservoir_rl_summary-20261008.json`; 10,000 decisions, three seeds each:

| reservoir | whole-run mean Δx_rand (s0 / s1 / s2) | last 2,500 decisions | jump share, last 2,500 | motor spikes per decision |
|---|---|---|---|---|
| connectome | −128 / −107 / −89 | −38 / −100 / −165 | 0.40-0.47 | 4,582 |
| lateral null 0 | −76 / −47 / −121 | −58 / −74 / −46 | 0.32-0.39 | 8,154 |
| lateral null 1 | −132 / −76 / −103 | −106 / −117 / +28 | 0.40-0.58 | 8,251 |
| R2 raw pixels (§8, cost 0, 20k) | −109 / −68 / −47 | | | |

**Reading (§10 rules).** No reservoir learner reaches competence; the reservoirs do
not beat the raw-pixel linear learner (R2); the connectome reservoir does not beat
the null reservoirs (it is, if anything, at the low end: whole-run −108 on average
vs −81 and −104). With a trainable readout and RL from the game's reward, at
10,000 decisions, the fixed readout is not the dominant constraint: replacing it
does not produce useful behaviour from any network. Limits: one step size, 10,000
decisions (R3 needed ~10,000 to turn positive even with perfect features), three
seeds; activity differs ~1.8x between arms (standardised features remove scale
but not sparseness).

### 9 results (15:10) — the game's reward does not keep the supervised timing; one exploratory wiring signal

`srl_report-20261008.json`, `srl_reading-20261008.json`; 10,000 RL decisions per run.

| arm (from its Stage D) | learners whole-run (ro1/ro2/ro3) | last 2,500 | Stage D reference | declared verdict | *post hoc:* second half, learners / twins |
|---|---|---|---|---|---|
| connectome | −62 / +19 / +21 | +243 / +31 / +170 | +82 / +83 / +126 | keeps 0/3, **improves 2/3**, loses 0/3 | −69 / +30 / +46 (mean +2) / twins +22 / +70 / −9 (mean +27) |
| null 0 | −71 / −84 / +14 | −127 / −95 / +59 | +89 / +149 / +66 | **loses 2/3** | −84 / −90 / +32 (−47) / twins (+18) |
| null 1 | −8 / −45 / −18 | −172 / −17 / −71 | +131 / +109 / +38 | **loses 3/3** | −39 / −48 / −67 (−51) / twins (−27) |
| null 2 | +10 / +10 / −115 | −58 / +1 / −82 | +86 / +171 / +118 | **loses 2/3** | −20 / +2 / −161 (−60) / twins (−7) |

**Declared reading.** No arm *keeps* competence (0/3 everywhere). By the last-window
rule the connectome *improves* (2 of 3 learners above their Stage D reference) and
every null *loses* (2/3, 3/3, 2/3 at or below 0).
**Post hoc checks (labelled as such, they qualify the declared verdict).** (1) The
last window is noisy: the same learner moves by up to 490 px between consecutive
2,500-decision windows (connectome ro1: −248 → +243). Over the second half of the
run the connectome learners average +2 px — no improvement over Stage D, a loss of
about 95 px. (2) **In every arm the RL learners end below their own shuffled-reward
twins** over the second half (connectome +2 vs +27; nulls −47 vs +18, −51 vs −27,
−60 vs −7): the game's reward degrades the taught timing *more* than random-sign
updates do — it pulls the policy towards the jump-rate solution (§6, §8). (3) The
connectome arm sits higher than the three null arms under both signals (learners +2
vs −47 / −51 / −60; twins +27 vs +18 / −27 / −7).
**Interpretation.** The supervised + RL pipeline does not give an RL improvement in
any architecture: the reward erodes the timing the label installed. The one
architecture-related pattern — the connectome's taught timing erodes less, under the
reward and under random-sign drift alike — is consistent with its more frame-specific
learnt change (PERCEPTION_CREDIT_STUDY §18, post hoc X), but it is post hoc, rests on
three learners per arm, compares against eye-mixing nulls that fire ~2x, and is not
claimed. **Decision.** Worth one declared replication before any claim: fresh seeds,
the laterality-preserving nulls (which first need their own Stage D run), and a
pre-registered robustness measure (second-half mean, not the last window).

### 9.1 Reference check of the §9 reading rule (declared 2026-10-08 15:16, commit 0131dd9, before running)

**Why.** §9 compares the RL learners with the same learners' last 10,000 *supervised*
decisions (Stage D). That rule was not tested on a reference that must pass, and the
harness differs from Stage D's: fresh noise seeds, a new sample seed (other level and
start draws), no teacher. If the *unchanged* Stage D policy scores well below its
Stage D reference in this harness, §9's keeps / loses verdicts measured the change of
harness, not RL.

**Run.** Version 17 = version 16 with the actor never updated (`actor_warmup` = 10^9;
the critic learns, which does not affect actions), connectome only,
`--init-from mario_stageD.pt`, run `mario_srl_frozen`, **5,000 decisions** (time).
Same seeds as `mario_srl`, so the episode draws are paired with it. The twin rows
carry the same weights, noise and draws as the learner rows and should repeat them
exactly (a determinism check).

**Must pass (decided now).** The frozen learners' whole-run mean Δx_rand ≥ +50 in ≥ 2
of 3 readouts (the §9 "keeps" rule). *Pass:* the §9 drops are attributed to the RL
updates. *Fail:* §9's verdicts against the Stage D reference are withdrawn, and the
comparison becomes learner vs frozen policy in the same harness (post hoc, labelled).
Descriptive addition: the frozen windows next to `mario_srl`'s first 5,000 decisions,
paired by readout.

### 9 addendum (15:24) — how strong is "learners end below their twins"? Weak.

Per readout (second half, learner minus its twin): connectome −91 / −40 / +56;
null 0 −107 / −116 / +27; null 1 −63 / +8 / −18; null 2 −8 / +32 / −181. **8 of 12
negative, mean −42 px, two-sided sign test p = 0.39.** The statement of §9 results
(2) holds for the four *arm means* but not readout by readout: the evidence that the
game's reward degrades the taught timing *more than* random-sign updates is weak.
What is robust is that *both* move the policy away from its Stage D level (§9.1
tests whether the harness alone accounts for part of that drop).

**Related work for the replication design (looked up 15:23).** Fine-tuning a
pre-trained policy with RL is known to lose pre-trained capabilities, especially on
states the fine-tuned policy stops visiting (Wołczyk et al., ICML 2024,
arXiv:2402.02868); fine-tuning a behaviour-cloned actor with a randomly initialised
critic gives a rapid drop, avoided by a critic-only phase with the policy frozen, then
a gradual actor warm-up (Ramrakhya et al., PIRLNav, CVPR 2023, arXiv:2301.07302); an
auxiliary loss keeping the student close to the teacher on the student's own
trajectories ("kickstarting", Schmitt et al. 2018, arXiv:1803.03835) lets the student
improve without forgetting. Our §9 run had a critic from zero and no such term, so
the drop has a known, mundane candidate cause *[amended §11: §9.1 then found no
RL-caused drop, so this cause is not supported for §9]*. Available now: the setting
`actor_warmup` (critic-only phase); the teacher label stays available as a training
signal (RAM for labels only), so a kickstarting term is the label kept on during RL.

### 9.1 result (16:07) — the reference check FAILS: §9's verdicts are withdrawn

`srl_frozen_check-20261008.json`, 5,000 decisions. The frozen Stage D policy in the §9
harness: whole-run mean Δx_rand **−46 / +36 / +7** (ro1 / ro2 / ro3; 52 / 45 / 34
episodes), against Stage D references of +82 / +83 / +126. 0 of 3 readouts ≥ +50:
**the declared test fails**, so §9's keeps / improves / loses verdicts measured, in
part, the change from Stage D's conditions to the §9 harness — not RL. They are
withdrawn, as declared. **A declared expectation was wrong:** the twin rows do not
repeat the learner rows, because episode draws are seeded by batch position (the
same cause as §8); they are paired *across runs* at equal position instead
(`mario_srl` and `mario_srl_frozen` draw identical level/start sequences, checked).
The weights of the frozen run stayed constant (checked on the live series), and
`mario_srl` started from the same weights (checks `--init-from`).

**What this raises.** The same weights that scored +82 to +126 during the last 10,000
Stage D decisions score about +0 frozen. During Stage D the label updates the
weights *every decision*, inside each episode; the training-time measure may include
that online steering rather than a timing stored in the weights. This bears on §5
amendment 2 ("the Stage D learners time their sampled jumps") and on the §20 / §4.3
evaluations of frozen checkpoints (partial results also near random).

### 9.2 Is the Stage D timing stored in the weights? (declared 2026-10-08 16:09, before running)

**Design.** The connectome Stage D checkpoint (`mario_stageD.pt`, 20,000 decisions) is
resumed twice for **3,000 decisions**, in its own harness (version 13: same settings,
learners, noise, draws continuing from the checkpoint's random state):
`mario_stageD_cont` with learning on (Stage D unchanged) and `mario_stageD_frozen`
(version 18 = version 13 with `frozen`: no weight update of any kind). Both continue
from the same state, so their episodes are paired (same position, same episode index
→ same level and start; to be checked). Learners: the three taught `tko2_ro<i>` and
their three shuffled-label twins.

**Reference that must pass first.** `mario_stageD_cont` (learning on) reproduces Stage
D: taught learners' mean Δx_rand ≥ +50 in ≥ 2 of 3 readouts. If it fails, the test is
inconclusive (Stage D itself does not reproduce at this budget).
**Reading (taught learners, whole 3,000 decisions).** *Stored timing:* frozen ≥ +50
in ≥ 2 of 3. *Online steering:* frozen < +50 in ≥ 2 of 3 **and** the paired
difference on − off, pooled over the three readouts, has a 95% bootstrap interval
(over episodes) above 0. Otherwise inconclusive. Twins reported alongside. Small
budget (time: stop at 17:20), descriptive power (~30 episodes per readout).

### 9.1 post hoc (16:11, labelled) — RL vs the frozen policy, paired on identical starts

`mario_srl` against `mario_srl_frozen`, first 5,000 decisions, episodes paired by
position and episode index (all pairs had the same level and start: 41/41, 45/45,
34/34, 43/43, 33/33, 45/45). Mean Δx_rand difference RL − frozen:

| | ro1 | ro2 | ro3 | pooled (95% bootstrap over episodes) |
|---|---|---|---|---|
| learners (game reward) | −9 | −21 | −12 | **−14 [−72, +42]**, 120 pairs |
| twins (shuffled reward) | −29 | +71 | +78 | +38 [−25, +105], 121 pairs |

**Reading.** Over its first 5,000 decisions, RL from the game's reward neither
improved nor degraded the Stage D policy detectably; the low level of §9 (≈ 0 px) is
already the frozen policy's level in this harness. The statement of §9 results that
"the reward erodes the taught timing" is **not supported** and is replaced by this
one. What remains to explain is why the frozen Stage D weights score ≈ 0 when the
same learners scored +82 to +126 during Stage D (§9.2).

### 9.2 result (16:33) — online steering: the Stage D timing is mostly produced by the label during the episode

`stageD_frozen_check-20261008.json`; connectome Stage D resumed from decision 20,000
for 3,000 decisions, learning on vs frozen; every compared episode pair had the same
level and start (32/32, 32/32, 25/25 taught; 23, 27, 19 twins).

| | ro1 | ro2 | ro3 | pooled, 95% bootstrap |
|---|---|---|---|---|
| taught, learning on (reference) | +34 | +63 | +278 | |
| taught, frozen | −31 | −4 | +98 | |
| taught, paired on − off | +59 | +66 | +180 | **+96 [+9, +180]**, 89 pairs |
| twins (shuffled label), paired on − off | −11 | −186 | +97 | −50 [−163, +62], 69 pairs |

**Declared reading.** The reference passes (learning on ≥ +50 in 2 of 3: Stage D
reproduces, barely). Frozen ≥ +50 in only 1 of 3, and the paired difference is above 0
in all three readouts with a pooled interval above 0: **verdict "online steering"**.
**What it means.** During Stage D the label (from RAM) updates the weights at every
decision; consecutive frames before an obstacle are similar, so an update at one
decision raises P(jump) at the next ones. Training-time play therefore shows the
*label's* timing, carried through fast weight changes; the stored policy keeps only
part of it (one readout of three). Random labels give no such benefit (twins).
**Consequences.** (1) §5 amendment 2 ("the Stage D learners time their sampled jumps,
+43 to +115") measured training-time behaviour including this steering; it is
**withdrawn as a statement about learnt policies**. (2) The §9.1 failure is explained:
the frozen policy never had that level. (3) PERCEPTION_CREDIT_STUDY §18 quantities 1a
/ 1b (and the §21 plan) are computed on fixed frames from frozen checkpoints
(`criterion_e_relabel.py`), so they are *not* affected: the stored contingency is real
but small (P(jump) rises of 0.05-0.10), consistent with the frozen play here.
(4) Every future play or Δx_rand measure of a taught learner must be taken with
learning off (frozen evaluation), as here. Limits: one graph, 3,000 decisions, the
pooled interval rests heavily on ro3.

### 6.1 result (16:34) — on identical data, the connectome learns the take-off label less well than the lateral nulls

`common_stream-20261007.json` (file named by the script's date; run 15:08-16:33 on
2026-10-08), `common_stream_summary-20261008.json`. Supervised rule of Stage D on one
fixed stream of 4,000 labelled frames; AUC of P(jump) for the take-off label on 300
fixed held-out frames of 5-1 / 8-1 (frozen weights at each scoring point).

| graph | AUC after 500 frames | after 4,000 | mean over the curve (500-4,000), per readout |
|---|---|---|---|
| connectome | 0.82 / 0.87 / 0.86 | 0.94 / 0.90 / 0.88 | 0.876 / 0.867 / 0.837 (mean **0.860**) |
| lateral null 0 | 0.90 / 0.90 / 0.94 | 0.96 / 0.94 / 0.95 | 0.940 / 0.914 / 0.946 (**0.933**) |
| lateral null 1 | 0.84 / 0.82 / 0.86 | 0.91 / 0.95 / 0.96 | 0.912 / 0.874 / 0.927 (**0.904**) |

**Reading (descriptive, as declared).** With the data made identical, all three
graphs learn the label quickly (AUC > 0.8 after 500 frames); the connectome's curve
lies below both lateral nulls' (its readouts are below every null-0 readout and below
two of three null-1 readouts). This matches the S4 decodability before learning
(PERCEPTION_CREDIT_STUDY §19: connectome lower for take-off and obstacles) and is
confounded the same way: the lateral nulls fire ~1.8x. No advantage of the measured
wiring for supervised sample efficiency is seen; if anything a disadvantage, which
an activity-matched comparison would have to confirm.

### 9.3 Replication of §9.2 on a null graph (declared 2026-10-08 16:36, before running)

The same design and reading rule as §9.2, on the depth-stratified null 0
(`mario_stageD_null0.pt`, `--graph null:0`): runs `mario_stageD_null0_cont` (learning
on) and `mario_stageD_null0_frozen` (version 18), 3,000 decisions from decision 20,000,
paired starts. Question: is online steering a property of the teaching procedure
(expected in every graph) or specific to the connectome? Same reference check (on ≥
+50 in ≥ 2 of 3), same verdicts. Must end by 17:20 (stopped with a checkpoint
otherwise; then reported as partial).

### 9.3 result (17:10) — replicated on null 0: online steering is a property of the teaching, not of the wiring

`stageD_frozen_check-null0-20261008.json`; null 0 Stage D resumed from decision 20,000 for
3,000 decisions, learning on vs frozen; all compared pairs had the same start.

| | ro1 | ro2 | ro3 | pooled, 95% bootstrap |
|---|---|---|---|---|
| taught, learning on (reference) | +83 | +160 | +77 | |
| taught, frozen | −32 | −64 | +41 | |
| taught, paired on − off | +111 | +225 | +39 | **+122 [+19, +225]**, 79 pairs |
| twins, paired on − off | −3 | −192 | +9 | −57 [−181, +57], 62 pairs |

**Declared reading.** Reference passes (3 of 3); frozen ≥ +50 in 0 of 3; pooled
interval above 0 → **online steering**, as in the connectome (§9.2). The teaching
procedure, not the wiring, produces the training-time timing; in both graphs the
stored policy keeps little of it (frozen: connectome −31 / −4 / +98; null 0
−32 / −64 / +41).

## 11. Amendments after the owner's review (2026-10-08 17:34)

The review checked the day's entries against the data. Statements corrected here
(the original text stays above, marked where it is in this file):

1. **"No architecture stores the timing"** (JOURNAL 2026-10-08 / 08b, OPEN_PROBLEMS P5,
   PROJECT_OVERVIEW §9c) is too strong. Information learnt from the label **persists in
   frozen weights**: the frozen-checkpoint contingency of the wiring comparison is
   positive (PERCEPTION_CREDIT_STUDY §18, 1a / 1b); in the common stream, brains scored
   *without learning* reach AUC 0.88-0.94 (connectome) and 0.91-0.96 (lateral nulls)
   after 4,000 frames (§6.1); one of three frozen connectome Stage D readouts exceeds
   +50 px in play (§9.2), which is not a reliable success of the set. What §9.2 shows is
   that training-time play overstated the *autonomous* competence of the checkpoints;
   what is missing is the **conversion of the stored information into useful actions
   in closed loop** (OPEN_PROBLEMS P1).
2. **Credit assignment to visual features** (§8) is a hypothesis, not the diagnosis.
   The RAM reference learns and the visual references fail, which locates the
   difficulty in learning from observations; still compatible: (a) the frames met in
   play differ from those on which the score was evaluated; (b) the score separates
   situations without being precise at the moment of the jump; (c) the action choice
   and button holding prevent the score from being used; (d) the optimisation uses
   visual features poorly.
3. **"A fresh critic destroys the pre-trained policy"** (§9 addendum; the day's summary
   to the owner) is not supported: on identical starts RL vs frozen = −14 px [−72, +42]
   (§9.1 post hoc). It does not guide the next steps.
4. **"Connectome = nulls" / "not distinguishable"** is read as **no difference shown at
   this power** (three readouts, two or three realisations).
5. **Two method limits, to be fixed before further runs (2026-10-09):** (a) episode draws
   are seeded by batch position, so adding a learner changes the others' starts — this
   already invalidated replication expectations (§8, §9.1); each learner needs random
   streams identified by a stable key; (b) competence as the *whole-run* training mean
   (§5) summarises a trajectory but penalises a late learner and mixes learning with
   competence; **acquired competence is established by evaluating the frozen
   checkpoint**, separately from training-time scores.
6. **Long wiring comparisons are suspended** until §12 is answered. The §21 runs stay
   paused (13,651 / 11,641 / 9,879 of 20,000 decisions) and are resumed only if a
   measure they give is needed.

**Current statement of the RL question:** *In the configurations and budgets tested,
no visual controller has shown reliable competence acquired from the game's reward.
Supervised learning persists in the weights, but its exploitation in closed loop
remains insufficient. No advantage of the connectome for learning or control is
established.*

## 12. From frame discrimination to frozen play (declared 2026-10-08 17:35, before running; to run 2026-10-09)

**Question (narrow, single).** Why does a discrimination learnt in the weights (frozen
AUC 0.88-0.96 on held-out frames, §6.1) translate into a poor frozen policy in play?
Not a wiring comparison: graphs are included to see whether the gap is general; any
arm difference is descriptive ("no difference shown at this power" otherwise).

**Models.** The common-stream brains (§6.1), re-trained with weights saved
(`common_stream.py` + a save option; same seeds, same stream): connectome,
`nulllat:0`, `nulllat:1`, three readouts each, after 4,000 frames.
*Reproduction check:* AUC at 4,000 frames within ±0.02 of the 2026-10-08 values per
readout; otherwise reported, and the new models are used, labelled.
*Negative models:* the same brains untrained (each graph), and connectome brains
trained on the same stream with the labels permuted (fixed seed), three readouts.
No teacher and no weight update anywhere below (frozen evaluation).

**Step 1 — frozen score on fresh development sequences.** New scripted-play sequences
(run, jump with p 0.25, as the stream) on the four play levels 5-1, 8-1, 3-1, 6-1, each
from the level's start after a fixed number of NOOPs (0, 3, ..., 21), network reset
before every decision (sweep regime), fixed noise seed, new sampling seed: S_fixed =
AUC of P(jump) for the take-off label v2 (RAM, label only), 95% interval by bootstrap
over sequences (frames within a sequence are dependent).

**Step 2 — decision procedure, fixed on calibration data only.** Exactly
PERCEPTION_CREDIT_STUDY §20: per brain, P(jump) = softmax(scores / 2) summed over the
jump buttons; tau = the threshold maximising TPR − FPR for the label on frames of the
*training* levels (1-1, 2-1, 4-1, 7-1; relabel procedure, seed 20261017); press
RIGHT+A+B when P(jump) ≥ tau, else the best non-jump action. Nothing from the play
levels is used to choose it.

**Step 3 — frozen play, no teacher.** `threshold_play_check.py`'s harness (4 levels × 8
starts, ≤ 600 decisions, stall 150, K = 5 random jumpers at the brain's own jump rate,
PASS at one-sided Wilcoxon p < .05; per-level breakdown; a pass carried by one level
does not count, P4). Logged at every decision: summed P(jump), tau, chosen action,
whether a jump decision follows a jump decision (button held), the RAM label on the
agent's own frame, x; for each death or stall, the preceding 20 decisions.

**Measures.** S_own = AUC of P(jump) for the label on the frames the frozen agent
itself visits (bootstrap over episodes); play pass / fail and mean Δx against the
rate-matched random jumpers, per level; timing: for each label onset, lag of the first
press (missed if none while the label is on), length of presses (consecutive jump
decisions), and the label / score / action context before each failure.

**References, checked before reading any trained brain.**
- *Logging alignment:* the scripted expert that presses exactly when the label is on
  must give S_own = 1 from the logged label and actions (catches frame / label offsets).
- *Positive (play):* that expert must pass the play test on the same start set and
  random-jumper draws. It passed once (p 0.020) and failed once (p 0.25) on this harness
  (PERCEPTION_CREDIT_STUDY §19). If it fails here, the play test is underpowered: play is
  reported descriptively (Δx with intervals) and scenario C cannot be declared.
- *Negative:* always-run, the untrained brains and the permuted-label brains must fail
  the play test and must not be "informative" (below). If one of them passes or is
  informative, the criterion is revised before any trained brain is read, and the
  revision is recorded.

**Classification of each trained brain (decided now).** *Informative* score: AUC ≥ 0.70
with the lower 95% bound above the highest point estimate among the negative brains.
- **C — useful frozen controller:** passes the play test (not carried by one level).
- **B — observation shift:** S_fixed informative and S_own lower by ≥ 0.10, the interval
  of the difference excluding 0 → the score works on fixed frames but not on the frames
  the agent meets.
- **A — decision / timing / interface:** S_own informative (and not B) but the play
  test fails; the timing measures then say which: missed presses (threshold), early or
  late onsets (precision), short presses (button holding).
- Otherwise: unclassified, reported.
Per graph: counts over three readouts, descriptive.

**Order and budget.** Step 0 (re-training, ~45 min on a free machine) → references
(scripted, fast) → step 1 → steps 2-3 for the connectome brains (trained, untrained,
permuted) → the two lateral nulls if time allows. Play is the expensive part (earlier
2-5 h per arm for 32 starts): one arm at a time, checkpointed at a stop time,
partial results labelled. Held-out levels stay untouched.

### 12.1 Operational details fixed in code before any result (2026-10-09 08:59)

`scripts/gates/frozen_play_probe.py` (merge / fixed / play / report; reading functions
tested on hand-built episodes, `tests/unit/test_frozen_probe.py`). Details the
declaration left open, fixed now:
- *Scripted references* are played inside the same harness as the brains (same
  environment pool, same reset to level and delay), as two extra slots of the
  connectome batch: `ref_expert` presses RIGHT+A+B while the take-off label v2 of the
  frame it sees is on, RIGHT+B otherwise; `ref_alwaysrun` presses RIGHT+B.
- *Alignment check*: for `ref_expert`, the logged action is RIGHT+A+B exactly when the
  logged label is on, and the AUC of its logged jump decisions for the logged label is
  1. Descriptive addition: its furthest x per start compared with the earlier
  oracle-path run of the same policy (`oracle_criterion_check-takeoff2-k5`).
- *"Carried by one level"*: removing that level leaves a mean difference to the random
  jumpers ≤ 0 over the other three levels.
- *Negatives for "informative"*: all negative brains measured (the untrained brains of
  every graph and the permuted-label connectome brains); the highest point estimate
  among them is the bar, per measure (S_own, S_fixed).
- *Step 1 sequences*: one per (level, delay), up to 400 decisions of scripted play
  (seed 20261090 + 100 × level index + delay), a frame every 2nd decision while Mario is
  in play; bootstrap over the 32 sequences (2,000 resamples).
- *Scenario B's interval*: difference of independent bootstrap draws of S_fixed (over
  sequences) and S_own (over episodes).
- *Failure context*: for episodes ending in death or stall, the last 20 decisions:
  "missed" (label on, never pressed while on), "pressed" (pressed while on), "unlabelled"
  (label never on).
- *Random jumpers*: K = 5 per brain and start, seeds 20261091 + 1000 × level index +
  10 × delay + 100 × slot + 100,000 × draw.

### 11.1 Fixes for the two method limits (2026-10-09 09:03)

**(a) Stable random streams — done.** Trainer setting `stable_streams` (default False
reproduces every earlier run): episode draws (level, delay, palette) come from a
generator keyed by the learner's *group* (a learner and its twin share one), and the
per-group uniforms (action sampling, exploration, reward-sign flip, coin labels) from
one torch generator per group. Adding a learner changes no one else's draws; twins
now share their starts. Tested (`tests/unit/test_rl_factor.py`): a group's draws are
identical with and without an extra group; the default reproduces the old episode
draws; the state survives a checkpoint round trip.

**(b) Acquired competence on frozen checkpoints — declared, to be validated before
use.** `train_mario_sweep.py --version <v> --eval-from <checkpoint>`: every learner takes
the weights of its same-named learner, no weight update of any kind, stable streams
(so every arm's brains of the same group play the same starts). *Definition:* a
checkpoint's learner is competent if its frozen mean Δx_rand over **≥ 100 training
episodes** is ≥ +50 px. The training-time whole-run mean stays a summary of the
learning trajectory (sample efficiency), not a competence verdict.
*Sizing, from the frozen runs of 2026-10-08* (`mario_stageD_frozen`,
`mario_stageD_null0_frozen`, `mario_srl_frozen`; 19-52 episodes per brain): the
per-episode SD of Δx_rand is ~255 px (median of 18 brains), so a zero-mean brain stays
below +50 with 97.5% probability only from ~100 episodes on; with 19 episodes one
frozen shuffled-label twin reached +94.5. The §9.2-§9.3 frozen means (25-33 episodes)
are therefore not competence verdicts; their paired on − off comparison stands.
*References before use:* positive — the scripted take-off v2 policy on the same starts
(+108.5 on the earlier draws) must pass; negatives — the frozen shuffled-label twins of
the Stage D checkpoints (connectome and null 0) must fail. Run when the GPU is free.

### 12.2 Step 0 result (09:27): the common-stream brains are reproduced exactly and saved

Re-training with `--save` (08:56-09:27, four processes in parallel) gives the same AUC
curves as 2026-10-08 to the fourth decimal at every scoring point (maximum absolute
difference 0.0 on all three graphs; at 4,000 frames connectome 0.937 / 0.899 / 0.879,
lateral null 0 0.955 / 0.939 / 0.947, lateral null 1 0.909 / 0.946 / 0.958): the
declared reproduction check (±0.02) passes. The permuted-label connectome brains end
at 0.380 / 0.321 / 0.144 on the same held-out frames (0.62-0.67 after 500 frames: a
label-free update can raise P(jump) where activity is high, which correlates with
take-off moments). Saved: `checkpoints/common_stream_{connectome,nulllat0,nulllat1}.pt`,
`common_stream_perm_connectome.pt`; merged for play: `frozen_probe_<graph>.pt`.

### 12.3 References read first (10:47, before any trained brain)

`frozen_probe/report-20261009.json` (three graphs complete, 32 starts each).
- **Logging alignment: passes.** `ref_expert`'s logged action is RIGHT+A+B exactly when
  its logged label is on (AUC of jump decisions for the label = 1), and it reproduces
  the earlier oracle-path run of the same policy on **32 / 32** starts (same furthest
  x): the harness, the reset and the label timing are as intended.
- **Positive play reference: FAILS.** The scripted expert against random jumpers at
  its own rate on these draws: 14 wins / 10 losses, +25 px, p = 0.115 (on other draws
  2026-10-06: 15 / 9, +82 px, p = 0.020). **As declared, the play test is underpowered
  on this start set: play is reported descriptively and scenario C cannot be
  declared.**
- **Negatives in play: all fail** (always-run, nine untrained brains, three
  permuted-label brains).
- **Negatives and "informative": a finding about the measure.** Some untrained brains
  rank take-off moments well through the fixed random readout alone: S_own 0.811
  [0.735, 0.876] (connectome untrained ro1, which jumps on 6% of decisions) and 0.792
  (null 0 untrained ro2); S_fixed 0.812 (null 0 untrained ro2) and 0.700 (null 0
  untrained ro3). An AUC ≥ 0.70 alone is therefore not specific to learning. The
  declared criterion already has the relative bar for this (lower bound above the
  best negative: **0.811 for S_own, 0.812 for S_fixed**); read literally, no negative
  is "informative" (the bar includes its own estimate), so the declared revision
  trigger is not met and the criterion is kept unchanged. It is strict by
  construction (a maximum over 12 negatives). *Descriptive addition, labelled:* each
  trained brain against the untrained brain with the same readout.

### 12.4 Declared result (10:51): no trained brain can be classified at this power

Scenario C was unavailable (§12.3). Under the declared criteria **all nine trained
brains are "unclassified"**: none is *informative* on its own frames (S_own 0.50-0.71;
lower 95% bounds 0.42-0.57, below the bar 0.811) or on fixed frames (S_fixed 0.88-0.91,
but lower bounds 0.40-0.66 because the pooled AUC changes a lot with the mix of
sequences resampled; bar 0.812), so neither B nor A can be declared.

| graph | readout | S_fixed (fixed frames) | S_own (own frames) | jump rate in play | label onsets missed | play Δx vs random (descriptive) |
|---|---|---|---|---|---|---|
| connectome | ro1 / ro2 / ro3 | 0.913 / 0.910 / 0.912 | 0.561 / 0.619 / 0.594 | 0.008 / 0.048 / 0.000 | 96% / 85% / 100% | +7 / +27 / 0 px |
| lateral null 0 | ro1 / ro2 / ro3 | 0.899 / 0.881 / 0.880 | 0.634 / 0.567 / 0.585 | 0.041 / 0.054 / 0.024 | 75% / 73% / 86% | +48 / +58 / +27 px |
| lateral null 1 | ro1 / ro2 / ro3 | 0.901 / 0.904 / 0.894 | 0.617 / 0.712 / 0.503 | 0.025 / 0.052 / 0.034 | 92% / 68% / 75% | +8 / +3 / +26 px |
| scripted expert | | | 1 (by construction) | 0.382 | 0% | +25 px (p 0.115) |

**What the declared measures do show.** In every trained brain the score ranks
take-off moments much better on fixed scripted frames than on the frames the frozen
agent meets (drop 0.19-0.39), and in play the calibrated threshold is almost never
crossed: the brains jump on 0-5% of decisions where the calibration predicted 52-58%,
and miss 68-100% of the moments the label marks. The intervals are too wide to call
the drop "observation shift" under the declared rule.

### 12.5 Post hoc (10:51, labelled): two properties of the play harness itself

1. **The delayed starts are not independent.** The 8 NOOP delays (0, 3, ..., 21) of each
   level rarely desynchronise the game: the scripted expert produces 5 / 2 / 1 / 2
   distinct episodes on 5-1 / 8-1 / 3-1 / 6-1 (10 of 32), the earlier oracle-path run of
   the same policy 2 / 2 / 1 / 1 (6 of 32); brains that never jump on a level repeat one
   episode there. The paired test over "32 starts" thus rests on a handful of
   independent situations, which explains why the same expert passes on one set of
   random-jumper draws (p 0.020) and fails on another (p 0.115). Every play verdict of
   the harness since 2026-10-04 (PERCEPTION_CREDIT_STUDY §14, §18, §20; RL_STUDY §4) shares
   this limit.
2. **On 3-1 the first killer is outside the label.** Every agent, the label-following
   expert included, dies after 39 decisions at x = 372 with the take-off label v2 never
   on: whatever kills Mario there is not a ground threat, wall or pit as the label
   defines them. 3-1 contributes no information about timing.

### 12.6 Post hoc (11:27, labelled): per level, the score fails in play in two different ways

`frozen_probe/per_level-20261009.json`; step 1 re-run with a per-level output (pooled
values identical to the first run); 3-1 excluded (label never on in play, §12.5).
Values per readout (ro1 / ro2 / ro3):

| graph | level | AUC, fixed frames | AUC, own frames | P(jump) on label-on frames − tau, fixed | same, own |
|---|---|---|---|---|---|
| connectome | 5-1 | 0.94 / 0.95 / 0.93 | 0.67 / 0.61 / 0.72 | +0.02 / +0.07 / −0.01 | −0.26 / −0.10 / −0.26 |
| connectome | 8-1 | 0.97 / 0.94 / 0.89 | 0.96 / 0.96 / 0.95 | +0.05 / +0.08 / 0.00 | −0.40 / −0.26 / −0.33 |
| connectome | 6-1 (night) | 0.49 / 0.44 / 0.59 | 0.46 / 0.41 / 0.47 | −0.50 / −0.39 / −0.30 | −0.51 / −0.36 / −0.32 |
| lateral null 0 | 5-1 | 0.88 / 0.89 / 0.86 | 0.68 / 0.59 / 0.77 | 0.00 / +0.03 / −0.03 | −0.07 / −0.09 / −0.07 |
| lateral null 0 | 8-1 | 0.96 / 0.95 / 0.93 | 0.86 / 0.80 / 0.84 | +0.03 / +0.03 / 0.00 | −0.05 / −0.02 / −0.04 |
| lateral null 0 | 6-1 (night) | 0.55 / 0.28 / 0.50 | 0.55 / 0.49 / 0.51 | −0.31 / −0.43 / −0.43 | −0.30 / −0.45 / −0.44 |
| lateral null 1 | 5-1 | 0.91 / 0.93 / 0.92 | 0.59 / 0.64 / 0.67 | +0.02 / +0.02 / +0.01 | −0.10 / −0.07 / −0.04 |
| lateral null 1 | 8-1 | 0.98 / 0.99 / 0.99 | 0.96 / 0.94 / 0.93 | +0.04 / +0.08 / +0.10 | −0.12 / +0.04 / −0.12 |
| lateral null 1 | 6-1 (night) | 0.43 / 0.47 / 0.30 | 0.44 / 0.48 / 0.36 | −0.38 / −0.43 / −0.44 | −0.35 / −0.43 / −0.44 |

The label is on for 57-58% of the fixed scripted frames of the day levels but for 8-11%
of the frames the agents meet.

**Observations.** (1) *Night level:* no ranking even on fixed frames, scores far below
the threshold everywhere: what was learnt on the day-palette stream does not transfer
to 6-1. (2) *8-1:* the ranking survives on the agent's own frames (0.80-0.96), but at the
moments the label marks, P(jump) stays below the threshold calibrated on training-level
frames — the information is there and the decision does not fire (scenario A in
substance). (3) *5-1:* the ranking itself drops on the agent's own frames (0.59-0.77 vs
0.86-0.95): scenario B in substance. (4) The connectome's score falls furthest below its
threshold on its own frames (8-1: −0.26 to −0.40; lateral nulls −0.12 to +0.04), which
matches its lower jump rate in play; descriptive, three readouts, activity-confounded.
These per-level readings are post hoc and do not replace §12.4.

**Hypotheses (not tested).** (a) The "take-off moments" of scripted play are mostly a
different kind of frame (the scripted Mario, jumping at random with p 0.25, spends long
stretches blocked against a pipe; 57% of its frames are labelled) from the approach
frames an agent meets (9%): calibration and part of the learning are fitted to the
former. (b) The stream had no palette variation; night levels are out of distribution.

**Proposed next steps (to be declared before running, owner's choice).** (1) Calibrate
the threshold on frames from the agent's own frozen play on the training levels
(on-policy calibration), then replay the dev levels. (2) Train on the frames the frozen
agent visits, labelled by RAM (dataset aggregation, DAgger: Ross, Gordon & Bagnell,
AISTATS 2011 — the standard remedy when a policy trained on expert-distribution frames
meets its own distribution). (3) Palette augmentation in the stream. (4) Fix the play
harness first (P4: starts that differ, distinct episodes counted, a positive reference
that passes on several random draws).

### 12.7 Post hoc (11:29, labelled): what the brains were taught is mostly "blocked against an obstacle"

Hypothesis (a) of §12.6, checked descriptively ("blocked" = Mario's x advanced by ≤ 2 px
over the last 4 decisions):

| frames | label on | blocked, among label-on frames | blocked, among label-off |
|---|---|---|---|
| training stream (common stream, 1-1 / 2-1 / 4-1 / 7-1) | 0.92 / 0.50 / 0.54 / 0.42 (all 0.60) | 0.98 / 0.69 / 0.88 / 0.80 (all **0.86**) | 0.04-0.08 |
| step-1 fixed sequences, 5-1 / 8-1 | 0.57 / 0.58 | **0.94 / 0.94** | 0.02 / 0.00 |
| the frozen agents' own play, 5-1 / 8-1 | 0.08 / 0.11 | **0.09 / 0.28** | 0.02 / 0.01 |

The scripted generator (run, jump at random with p 0.25) stays blocked against the
first obstacle for long stretches (on 1-1, 92% of its frames are labelled and 98% of
those are blocked): the "take-off" examples the brains learnt from, and on which their
threshold was calibrated, are mostly frames of Mario pressed against a pipe or wall.
The agents in play meet the label mostly while *approaching* an obstacle, a kind of
frame the stream contains little of — so their score rises there but not to the
threshold (8-1), or ranks them poorly (5-1). This is a concrete, checkable form of
observation shift: **the training distribution, not the network, sets the gap.** It
also qualifies earlier frame-level results built on the same scripted generator (the
S4 probe, PERCEPTION_CREDIT_STUDY §19, which already noted "obstacle is mostly pipe right
ahead"; the common stream, §6.1).

**Consequence for the next experiment (to declare):** a stream whose take-off examples
are approach frames — frames from the frozen agent's own play labelled by RAM
(DAgger), or from a label-following expert with exploration — before any further
comparison of architectures on supervised capacity or frozen play.

### 11.2 Frozen competence, connectome Stage D (11:45; `frozen_competence-20261009.json`)

`mario_stageD_eval_frozeneval`: 15,000 decisions, frozen, stable streams (each readout's
learner, twin and scripted reference share their starts).
- **References:** the three scripted take-off v2 references are competent (+89 / +80 /
  +111 px over 157 / 163 / 164 episodes; intervals [55, 121], [45, 113], [79, 142]); the
  three shuffled-label twins are not (−30 / −34 / +36 px over 119 / 116 / 97 episodes).
  **The criterion passes its references.** (The reader's first version also required
  100 episodes for *every* brain, references included — stricter than declared; one twin
  has 97. It was aligned with the declaration before any learner was read: a brain with
  fewer than 100 episodes is "not judgeable", and a reference twin is in any case not
  competent.)
- **Learners:** ro1 +28 px (133 episodes, [−17, +78]) not competent; ro2 +25 px (126,
  [−3, +56]) not competent; **ro3 +55 px (108, [+1, +113]) competent**, narrowly. One of
  three connectome Stage D checkpoints holds a competent frozen policy on the training
  levels; it is the same readout that stood out in §9.2. The null-0 Stage D checkpoints
  follow (running).

### 11.3 Frozen competence, null 0 Stage D (12:44)

`mario_stageD_eval_null0_frozeneval`, same procedure. **References:** the scripted
references give *exactly* the connectome arm's numbers (157 / 163 / 164 episodes, +89 /
+80 / +111 px): under stable streams both arms played the same starts, as intended. The
twins are not competent (−31 / −31 / 0 px). **Learners:** −7 / +9 / +30 px (124 / 112 /
126 episodes): **0 of 3 competent** (connectome: 1 of 3). The intervals overlap widely:
no difference shown at this power.

**Summary of point 4 (method fixes).** Random streams are now stable across batch
layouts (and across arms: identical starts, checked). Competence is now judged on frozen
checkpoints with ≥ 100 episodes, and the criterion passes its references. By that
criterion, Stage D's supervised learning stored a competent frozen policy in 1 of 6
checkpoints tested (connectome ro3, narrowly) — far from the 10 of 12 that training-time
play suggested (§5 amendment 2, withdrawn on 2026-10-08).

## 13. Play harness v2: starts that differ (declared 2026-10-09 12:56, before any code or run)

**Why.** §12.5: the harness's 8 NOOP-delayed starts per level are mostly one episode
(scripted expert: 10 distinct of 32), so its paired test has few independent situations
and its positive reference passes or fails with the random draws; on 3-1 the first
killer is a hopping Koopa Paratroopa (enemy type 0x0E), which the ground-threat label v2
does not see by construction.

**Starts.** Per level, 16 starts. Each start is the level's savestate followed by a
random *prefix* of 5-40 decisions (uniform), each decision RIGHT+B with p 0.6 or NOOP
with p 0.4 (no jump); a prefix that ends in death or the end of the episode is redrawn.
Seeded per level and start (seed 20261100 + 1000 × level index + start index), stored with
the start's furthest x and a hash of its RAM. Control is handed to the policy after the
prefix. Two suites, reported separately: **training levels** (1-1, 2-1, 4-1, 7-1; play
on the levels the brains were taught on) and **unseen development levels** (5-1, 8-1,
3-1, 6-1). The held-out levels stay untouched.

**Test (unchanged in form).** Per policy and suite: furthest x against the mean of K = 5
random jumpers at the policy's own jump rate on the same start; one-sided Wilcoxon over
starts, PASS at p < 0.05, and not carried by one level (removing any one level leaves a
mean difference > 0).

**Validation, scripted policies only (no brain), decided now.** Three independent sets of
random-jumper draws (seeds 20261200 + 100,000 × set).
- *Distinctness:* the 16 start states of every level are distinct (RAM hash), and the
  scripted expert's episodes are distinct on ≥ 12 of 16 starts in at least 3 of the 4
  levels of a suite.
- *Positive:* the scripted take-off v2 expert (RIGHT+A+B while the label is on, RIGHT+B
  otherwise) PASSES in **3 of 3** draw sets.
- *Negatives:* a second random jumper at the expert's rate (K = 5, independent seeds)
  and always-run FAIL in 3 of 3 draw sets.
A suite whose checks all hold is a valid play measure; otherwise play on it stays
descriptive. Expected from §12.5: the unseen suite may fail (the label is blind on 3-1;
on 6-1 the expert did worse than running); that would be reported, not repaired by
choosing levels after the fact.

### 13 result (13:00): the training-level suite is a valid play measure; the unseen suite is not

`harness_v2/validation-20261009.json` (scripted only; 2 min on 8 cores). All 16 start
states of every level are distinct.

| suite | distinct expert episodes (of 16) | expert vs random, draws 1 / 2 / 3 | second random jumper | always-run | verdict |
|---|---|---|---|---|---|
| training (1-1, 2-1, 4-1, 7-1) | 16 / 13 / 15 / 14 | +71 px p 0.005; +79 p 0.001; +80 p 0.001 (not carried by one level) | p 0.77 / 0.22 / 0.32 | p 1.0 ×3 | **valid** |
| unseen (5-1, 8-1, 3-1, 6-1) | 15 / 15 / 15 / 14 | +35 p 0.080 (carried); +66 p 0.003; +80 p 0.003 | p 0.76 / 0.12 / **0.025** | p 1.0 ×3 | not valid |

Per level, the expert gains +210 to +265 px on 1-1 and 4-1 and loses 59-112 px on 2-1 and
7-1 (training), gains +155 to +292 px on 5-1 and 8-1, never jumps on 3-1 (the label does
not see the Paratroopa) and loses 135-240 px on 6-1 (unseen). **From now on, frozen play
is measured on the training-level suite of harness v2; the unseen suite is reported
descriptively** (a generalisation test needs a label that sees every killer, or other
unseen levels).

### 13.1 Frozen play of the §12 brains on harness v2 (declared 13:00, before running)

The §12 checkpoints (`frozen_probe_<graph>.pt`: cs_ro1-3; for the connectome also the
permuted-label brains and the two scripted slots; untrained brains added by the
evaluator), the §20 threshold rule fixed on training-level calibration frames (as in
§12), frozen play on the **training suite of harness v2** (64 starts), K = 5 random
jumpers per brain and start (seeds as §13, draw set 1), and the per-decision log of §12.
S_fixed on fresh scripted sequences of the training levels (one per start, step 1 of
§12 with these starts). **Reading:** exactly §12's classification (C / B / A /
unclassified; same bars, recomputed on these negatives), with C available only if the
in-batch scripted expert passes in this run. Prediction written now (§12.7): brains that
learnt "blocked against an obstacle → jump" may clear pipes on the training levels and
pass where the expert passes (1-1, 4-1).

### 13.1 references, read first (14:48; connectome batch complete, null batches running)

On the valid training suite of harness v2: the in-batch scripted expert presses exactly
when its label is on and reproduces the harness validation on **64 / 64** starts (same
random-jumper seeds, hence the same numbers: 36 / 27, +71 px, p 0.005, not carried by
one level). **Positive reference passes.** Negatives of the connectome batch all fail:
always-run (p 1.0), the three untrained brains (−214 / −57 / −140 px) and the three
permuted-label brains (−91 / +7 / 0 px). **Scenario C is available** on this suite,
subject to the null batches' untrained brains also failing.

### 13.1 result (15:53): on the valid suite no brain plays usefully; the connectome's brains keep their score on their own frames (A), the lateral nulls' do not (B)

`frozen_probe/reportv2-training-20261009.json`. References passed (above); scenario C
available; negatives' best estimates: S_own 0.754, S_fixed 0.590 (the bars).

| graph | readout | S_fixed | S_own [95%] | drop fixed − own [95%] | play vs random (p) | **declared class** |
|---|---|---|---|---|---|---|
| connectome | ro1 | 0.878 | 0.823 [0.739, 0.881] | 0.055 [−0.007, 0.144] | −31 px (0.83) | unclassified |
| connectome | ro2 | 0.885 | 0.840 [0.769, 0.885] | 0.045 [−0.007, 0.119] | −48 px (1.00) | **A** |
| connectome | ro3 | 0.881 | 0.837 [0.778, 0.885] | 0.044 [−0.008, 0.106] | −4 px (0.33) | **A** |
| lateral null 0 | ro1 / ro2 / ro3 | 0.902 / 0.897 / 0.906 | 0.771 / 0.789 / 0.735 | 0.131 / 0.108 / 0.171 (all > 0) | −44 / −74 / +11 px | **B / B / B** |
| lateral null 1 | ro1 / ro2 / ro3 | 0.907 / 0.919 / 0.921 | 0.659 / 0.636 / 0.737 | 0.248 / 0.283 / 0.184 (all > 0) | +11 / +72 / +28 px (p ≥ 0.085) | **B / B / B** |

**Declared reading.** No brain is a useful frozen controller on the training levels
(none passes; the scripted expert does on the same starts). Two connectome brains are
class A — their score still ranks take-off moments on the frames they meet, and the
failure is in the decision; all six lateral-null brains are class B — their score's
ranking drops on their own frames.

**What fails in class A (the declared timing measures):** the connectome A brains miss
32-41% of label onsets (expert 0%), press with little lag when they do (median 0-1
decision), but **press for one decision only** in 58-66% of presses (expert 30%; median
1 vs 2) — in Super Mario Bros a short press is a low jump — and jump on 16-27% of
label-off decisions (expert 0%). Most of their failures follow a press while the label
was on (40 and 44 of 58-62 failures): they jumped at the right moment, too low. A
per-decision threshold makes P(jump) flicker around tau, so the policy taps instead of
holding A.

**Exploratory, not declared (arm difference):** the drop from fixed to own frames is
smaller in all three connectome brains (0.044-0.055) than in all six lateral-null brains
(0.108-0.283); were the nine drops exchangeable, the chance that the three connectome
ones are the three smallest is 1/84 (0.012). Three readouts per arm, two null
realisations, and the nulls fire ~1.8x the connectome: an activity-confounded,
post hoc pattern, worth one declared replication (more realisations, activity-matched
comparison), not a claim. On the unseen levels (§12.6) the pattern was mixed.

**Next (to declare):** for the A brains, a decision rule that holds A once triggered
(e.g. press-and-hold for up to 6 decisions, as the full RAM expert does) — an
interface change, declared as such; for the B brains and for all, the expert stream
(§14 draft: press-release expert with random prefixes, take-off examples mostly
approaches, blocked share 23% vs 86%).

## 13.2 Hold once triggered: the decision interface for class-A brains (declared 2026-10-09 15:54, before any code or run)

**Why.** §13.1: the connectome's class-A brains rank take-off moments on their own frames
but press A for one decision in most presses (low jumps). In Super Mario Bros jump
height grows with how long A is held; a held A does not jump again once landed.

**Rule ("hold6").** A *trigger* is a decision where the policy wants to jump (brains:
P(jump) ≥ tau, tau fixed exactly as in §12-§13.1) while not holding. After a trigger,
RIGHT+A+B is pressed for 6 decisions whatever the score, then RIGHT+B for at least one
decision before the next trigger. Otherwise the brain's best non-jump action (as §20).
This is an interface change, declared as such; 6 is the hold of the full RAM expert
(§14 of PERCEPTION_CREDIT_STUDY), not tuned.

**Fair control.** The random reference must hold the same way: a *random hold6 jumper*
triggers with probability q on each eligible decision (not holding, not in the release
decision), q matched to the candidate's trigger rate per eligible decision, then follows
the same hold6 rule. K = 5 per start, harness v2 training suite, seeds as §13.

**Validation first (scripted, as §13), three draw sets.** Positive: the take-off v2 expert
under hold6 (trigger = label on) must pass 3 / 3. Negatives: a second random hold6 jumper
at the expert's trigger rate and always-run must fail 3 / 3. Only if all hold is the rule
applied to brains.

**Brains (if validated).** The §12 common-stream brains (connectome, lateral nulls 0 and 1;
untrained brains as negatives), frozen, hold6, training suite. **Reading:** C (passes the
hold6 test, not carried by one level) per brain; descriptive comparison with §13.1 on the
same starts (same brains, threshold rule). Prediction written now: the class-A
connectome brains are the likeliest to pass.

### 13.2 validation result (15:57): hold6 is valid on the unseen suite, not on the training suite

`harness_v2/validation-hold6-20261009.json` (scripted, three draw sets, random hold6 jumpers at
the expert's trigger rate).

| suite | expert under hold6, draws 1 / 2 / 3 | second random hold6 jumper | always-run | verdict |
|---|---|---|---|---|
| training | +309 px p 0.054 (carried by 4-1: +1,238); +334 p 0.020; +348 p 0.016 | p 0.37 / 0.96 / 0.43 | p 1.0 | **not valid** (2 of 3) |
| unseen | +257 p < 0.001; +238 p < 0.001; +246 p < 0.001 (not carried) | p 0.44 / 0.53 / 0.58 | p 1.0 | **valid** |

On the unseen levels, pressing and releasing turns the label into a strong controller:
+377 to +440 px on 6-1 (where holding A as long as the label is on lost 135-240 px), +390
to +442 on 5-1; 3-1 stays at 0 (the label does not see the Paratroopa). **As declared, hold6
is not applied to brains on the training suite.**

## 13.3 Hold6 on the unseen suite (declared 15:57, before running)

The §12 common-stream brains (connectome, lateral nulls 0 and 1; untrained brains as
negatives; scripted expert and always-run in the connectome batch), frozen, tau as in
§12-§13.1, **hold6 rule, unseen suite of harness v2** (64 starts, random hold6 jumpers
at each brain's trigger rate, K = 5, seeds as §13 draw set 1). In-batch references
first (the expert must pass again, negatives must fail). **Reading:** C per brain (passes
the hold6 test, not carried by one level); per-level description. Prediction written now
(from §12.6): few triggers on these levels — the brains' P(jump) at take-off moments
stayed 0.1-0.4 below tau there — so most brains are expected to fail; 8-1 is where a pass
is most likely.

## 14. Training on approach frames: the expert stream (declared 15:57, before running)

**Question.** Does teaching on frames where the take-off label marks an *approach* to an
obstacle (rather than Mario blocked against it, §12.7) turn the stored discrimination into
useful frozen play?
**Stream.** `common_stream.py --stream expert`: per life, a random run/NOOP prefix (5-40
decisions), then a press-release take-off v2 expert (RIGHT+A+B while the label is on, A held
≤ 6 decisions then released one); training levels; same frame sampling, seeds, rule
(Stage D supervised form, eta 0.001, tau 2), 4,000 frames, three readouts per graph
(connectome, lateral nulls 0 and 1), weights saved (`common_stream_expert_<graph>.pt`).
**Stream check (declared, before training is read):** the share of blocked frames among
label-on frames must be ≤ 0.35 (pilot measure: 0.23; scripted stream 0.86); if not, the
stream does not test the question and the run is reported as such.
**Evaluation.** Frozen, per-decision log: (i) threshold rule (§20) on the **training
suite** of harness v2 (valid for that rule), (ii) hold6 on the **unseen suite** (valid for
that rule); untrained brains as negatives; in-batch scripted references.
**Reading.** Classes as §12-§13.1 (same bars recomputed on this run's negatives). Primary,
per graph and suite: the number of brains that are useful frozen controllers (C), against
the scripted-stream brains on identical starts (§13.1 for (i), §13.3 for (ii)). The stream
*helps* in a graph if at least 2 of its 3 expert-stream brains are C where its
scripted-stream brains are not. Arm differences descriptive (three readouts, two null
realisations, activity-confounded).

### 14 stream check (16:04): passes

The expert stream (identical for the three graphs): 4,384 frames, label on 0.205, **blocked
among label-on frames 0.136** (≤ 0.35 declared; scripted stream 0.86). The stream tests the
question.

## 15. State at the owner's stop (2026-10-09 16:10) and what remains

**Stopped at 16:10 (nothing running).** The three §14 expert-stream re-trainings were killed at
~500 of 4,000 frames (no checkpoint inside `common_stream.py`: they restart from zero,
~30 min); the chain `probe_x_chain.sh` (merge, then nine frozen plays) had not started.
§14 and §13.3 are declared and their stream check passed; **no result of either exists.**

**Established today.** Reproduction of the common stream (exact); frozen competence
criterion validated on references (scripted 3/3, twins 0/6); stable random streams
(identical starts across arms, checked); play harness v2 with a **valid training-level
suite**; the take-off examples of the scripted stream are mostly Mario blocked against an
obstacle (86%); no frozen brain is a useful controller on the valid suite; the
connectome's class-A brains fail by short presses; hold6 makes the label expert strong
on unseen levels.

**To do, in order (each declared already unless marked).**
1. *§14* — `for g in connectome nulllat:0 nulllat:1: common_stream.py --graphs $g --save --stream expert`
   (three in parallel, ~30 min), then `scripts/gates/probe_x_chain.sh` (~3-4 h, GPU-bound),
   then `frozen_play_probe.py report --suite training --stream expert` (primary),
   `report --suite unseen --rule hold6 --stream expert` and `report --suite unseen --rule hold6`
   (§13.3); read references first, then the declared classes.
2. *Not declared yet:* if class-A brains remain, hold6 cannot be tested on the training
   suite (not valid there); a press-release rule validated on the training suite, or a
   larger training-suite start set, would be needed.
3. *Not declared yet:* a declared replication of the exploratory arm difference (§13.1:
   connectome brains keep their score on own frames, lateral nulls do not), with the third
   lateral-null realisation and an activity-matched comparison.
4. *Not declared yet:* a take-off label that sees hopping enemies (3-1), or other unseen
   levels, so that the unseen suite becomes valid under the threshold rule.
5. Only once some frozen controller is useful (C): return to the RL question (supervised
   then RL with critic warm-up and the label kept as an auxiliary signal; held-out levels
   for the confirmation).
6. Paused: §21 Stage D on the lateral nulls (13,651 / 11,641 / 9,879 of 20,000 decisions;
   resumable; suspended per the review).
7. Housekeeping (owner's decision): `gpucore.87935` (9.9 GB core dump, ignored by git);
   untracked `artifacts/video_assets/` (107 MB) and `media/` (11 MB) not committed.

## 16. Remote NVIDIA server: procedure for the declared experiments (2026-10-09 18:16, before any run there)

The project can now run on a rented Linux/NVIDIA server (`deployment/README.md`,
`deployment/HANDOFF.md`); it has **not yet been run on NVIDIA**. Procedural rules for the
experiments already declared, fixed now:
- **Same platform for every arm.** If §14 runs on NVIDIA, its comparator is the §13.1
  procedure *re-run on NVIDIA* (same code, seeds and starts: `cs-scripted` for the three
  graphs, `cs-permuted`, merge, frozen play on the training suite, S_fixed, report), not
  the ROCm numbers of §13.1. The same holds for §13.3 (its brains are the NVIDIA-trained
  scripted-stream brains) and for any resumption of §21 (every graph rerun on NVIDIA).
- **ROCm results are coherence checks only.** Before any declared experiment on NVIDIA:
  the same-procedure rerun of the common stream must match the ROCm AUC curve within ±0.02
  at every scoring point (the reproduction tolerance of §12.2), and the harness-v2
  validation rerun (emulator only) must be identical. If not, nothing is run and the owner
  decides.
- **No exact continuation across platforms.** A ROCm checkpoint may initialise an NVIDIA
  run (weights), but a ROCm run continued on NVIDIA is a new run, labelled as such.
- `common_stream.py` now saves and resumes its training state (scoring points, SIGTERM);
  an interrupted and resumed run equals an uninterrupted one bit for bit on the workstation
  (`tests/unit/test_common_stream_resume.py`). The learning rule is unchanged.

### 16.1 First runs on NVIDIA (Vast RTX 3090, 2026-10-09 19:59; results of the §16 checks)

Server: RTX 3090, driver 590.48, torch 2.9.1+cu128, Python 3.11.16 (`deployment/VAST_GUIDE.md`).

**Functional: passes.** Transferred files verified by sha256; doctor (emulator, graph and
null caches, sparse product against the CPU ≤ 1.9e-6); smoke (an episode; the 8 harness
starts identical to their RAM fingerprints; one frozen-play start from the ROCm checkpoint).

**A determinism defect found and fixed (not a protocol change).** The first smoke failed its
resume test. Two identical *uninterrupted* runs also differed (max |Δm| 13.9 synapses), so
the cause was not the resume: PlasticLIF's block-diagonal CSR product, bit-repeatable on
ROCm, is not bit-repeatable on CUDA once the plastic magnitudes are fractional (the same
product differed in 99 of 99 repeats; integer magnitudes hide it because integer sums are
exact in float32). On CUDA the motor input is now a segment reduction over each motor
row's connections: 0 of 99 repeats differ, +10-13 % time per tick; ROCm keeps the block
product, so every earlier ROCm result stays bit-reproducible. Same equation, summation
order only (relative ~1e-4 on a motor input). After the fix: two identical runs are equal
bit for bit, the resume test passes, smoke passes (`src/flymario/training/plastic.py`,
`tests/unit/test_plastic.py`).

**Numerical.** GPU random streams differ between ROCm and CUDA for the same seed (doctor);
without noise, the motor spike counts of the reference sweeps are identical on both
(27,465 = 27,465). A noisy run on NVIDIA is therefore a different noise realization of the
same procedure, reproducible on its own platform.

**Scientific coherence (the §16 checks, applied as declared).**
- Harness-v2 validation rerun (emulator only): **identical** to the ROCm file. Passes.
- Common-stream rerun, connectome, scripted stream (`cs-scripted`, 6.5 min on the 3090):
  test AUC per readout at 0 / 500 / 1,000 / 2,000 / 4,000 frames
  NVIDIA 0.19,0.18,0.35 / 0.87,0.85,0.85 / 0.84,0.88,0.71 / 0.87,0.79,0.82 / 0.94,0.92,0.92;
  ROCm 0.24,0.15,0.35 / 0.82,0.87,0.85 / 0.86,0.88,0.79 / 0.86,0.83,0.75 / 0.94,0.90,0.88.
  Largest difference 0.081 (1,000 frames), 0.045 already at 0 frames (untrained brains).
  **The declared tolerance (±0.02 at every point) fails.** As declared, no §13-§14
  experiment is run on NVIDIA until the owner decides.

**Reading (descriptive, post hoc).** The tolerance came from a same-platform rerun, where
the procedure reproduced to the fourth decimal; it was never tested against a reference
that must pass across noise realizations, and the difference already exists before any
learning, where only the noise draws differ. Both curves have the same shape (≈ 0.85 by
500 frames, 0.88-0.94 at 4,000). This suggests, without establishing it, that the check
measures the noise-realization spread rather than a platform effect on learning. Options
for the owner (none taken): (a) declare a cross-platform criterion calibrated on
references first, e.g. the spread of the same procedure over several noise seeds on one
platform must pass and the permuted-label run must fail; (b) accept that every arm runs on
NVIDIA (already required by §16) and treat the ROCm curve as context only; (c) a
coherence run with platform-identical noise drawn on the CPU (the simulator accepts
explicit noise), slower.

## 17. NVIDIA coherence, decomposed (declared 2026-10-09 18:22 UTC, before any of these runs)

*Owner's mandate of 2026-10-09 18:16 UTC (`report/MISSION_48H.md`): resolve the §16.1 block
before the campaign. The §16.1 check failed as declared and stays failed; nothing below
revises it after the fact. The tests below decompose the failure into what the platform
computes (N1-N3) and how much the measure moves between noise realisations (N4); the
recalibrated criterion of N4 is calibrated on some noise seeds and validated on others before
it is applied to ROCm vs NVIDIA.*

**N1 — the CUDA segment reduction computes the motor input.** Weights: the ROCm-trained
connectome brains (`rocm_reference/checkpoints/common_stream_connectome.pt`, fractional
magnitudes). Presynaptic spike vectors: 20 ticks recorded from real sweeps of the scripted
stream (3 brains). Motor input by (a) CUDA segment reduction (production path on NVIDIA), (b)
CUDA block SpMM (the ROCm code path), (c) float64 on the CPU (scipy). **Pass** if, for (a) and
(b), max |x − x64| / max |x64| ≤ 1e-5 on every tick.
**N2 — reproducibility.** (a) repeated 50 times on the same inputs: **pass** if 50 / 50 are
bit-identical; (b) reported (known not to be repeatable on CUDA).

**N3 — identical noise, three implementations.** The common stream exactly as `cs-scripted`
(same frames, order, readouts, rule) but the membrane noise of every tick drawn from one CPU
generator (seed 20261201, (n, 3) standard normals × noise_std, copied to the device), so every
implementation sees the same noise; first 500 training frames, AUC on the same 300 held-out
frames at 0, 250 and 500 frames. Implementations: CUDA-segment, CUDA-SpMM, CPU (float32).
*References:* must pass — CUDA-segment against a second CUDA-segment run: identical AUCs and
weights; must fail — CUDA-segment with permuted labels (seed 20261077) against CUDA-segment:
some |ΔAUC| > 0.02 at 250 or 500 frames. **Pass** (implementation equivalence) if, for both
CUDA paths against the CPU, |ΔAUC| ≤ 0.02 at every scoring point and readout (the §12.2
tolerance, now with the noise held identical). Reported: first-sweep motor spike counts,
relative difference of the learnt weight change.

**N4 — the spread between noise realisations on NVIDIA.** Statistic, exactly that of §16:
D(a, b) = max over the 3 readouts and 9 scoring points of |AUC_a − AUC_b|. One batch on
NVIDIA (`scripts/diagnostics/cs_noise_spread.py`): the `cs-scripted` procedure for noise sets
s = 0..8 (noise seed of readout i: 8300 + 100 s + i; s = 0 is the production set), plus, on
noise set 0, three brains with eta halved and three with permuted labels.
- *Batch check (must pass):* the s = 0 brains reproduce the existing NVIDIA `cs-scripted`
  curve exactly, and the permuted brains reproduce `cs-permuted` when it exists (otherwise
  reported); if not, the batch is not the same procedure and N4 is reported, not applied.
- *Calibration:* T = the largest D over the 6 pairs of sets 1-4.
- *Validation (on sets 5-8, distinct from the calibration):* the criterion is valid if at most
  1 of the 6 pairs of sets 5-8 has D > T (it does not flag the same procedure) **and** the
  permuted brains differ from each of sets 5-8 by D > T (4 / 4: it flags a different
  procedure). Reported, not a gate: how many of sets 5-8 the eta/2 brains are flagged against
  (a subtle defect the criterion may or may not see).
- *Application:* ROCm vs NVIDIA (set 0): **coherent** if D ≤ T. Also reported: D between the
  ROCm curve and each NVIDIA set.

**Decision rule.** N1, N2(a) and N3 pass and N4 is valid with ROCm vs NVIDIA ≤ T → platform
coherence accepted at the measured level; the campaign runs on NVIDIA with every arm on NVIDIA
(§16), ROCm results kept as a separate series. N3 fails → the implementation differs: at most
two bounded investigations, then the block is reported. N4 invalid → reported; the decision
then rests on N1-N3 and on §16's same-platform rule, stated as such.

## 18. NVIDIA baseline, a third null realisation, and a declared replication (declared 2026-10-09 18:28 UTC, before any of these runs)

*Runs only if §17 accepts the platform (otherwise this section waits).*

**Baseline (§16, unchanged).** The §13.1 procedure re-run on NVIDIA for every arm: `cs-scripted`
for the connectome (done 2026-10-09 17:51, the §16.1 run) and `nulllat:0`, `nulllat:1`;
`cs-permuted`; merge; frozen play on the training suite of harness v2 (threshold rule); S_fixed;
report. Reading exactly as §13.1 (classes C / B / A / unclassified, bars recomputed on this run's
negatives; C available only if the in-batch scripted expert passes).

**Addition: `nulllat:2`.** The third laterality-preserving realisation (cached and validated
2026-10-06) goes through the same pipeline, and through §14 and §13.3 below, and is read as a
separate graph. Why: the arm comparisons so far rest on two null realisations (P3); a third is the
cheapest way to see the realisation spread. It changes no criterion; the §14 primary verdict
("helps in a graph") is read per graph as declared.

**Declared replication of the §13.1 exploratory pattern.** §13.1 observed (post hoc) that the
drop S_fixed − S_own was smaller in all three connectome brains than in all six lateral-null
brains (1/84 if exchangeable). Here the brains are re-trained and re-played on NVIDIA, i.e. a new
noise realisation of the same procedure. *Test:* one-sided exact Mann-Whitney on the drops,
connectome (3) smaller than lateral nulls 0-1 (6); **replicated if p < 0.05** (at most two
connectome-null pair inversions, exact tail 0.048). Secondary, reported: the same with nulls 0-2
(3 vs 9). *Interpretation fixed now:* replicated → a reproducible difference in how the stored
discrimination transfers to the agent's own frames, still activity-confounded (the nulls' motor
layer fires ~1.8x the connectome's), not an advantage in play unless play also differs;
not replicated → the §13.1 pattern is not distinguished from noise-realisation variation.

### 17.1 Results (2026-10-09 18:58 UTC): the platform computes the rule correctly; the learning trajectory amplifies rounding; decision: the campaign runs on NVIDIA

Files: `artifacts/runs/platform/{operator-N1N2,samenoise-*,lockstep-N3b,cs_noise_spread-nvidia,report-17}.json`
(`scripts/diagnostics/platform_equivalence.py`, `cs_noise_spread.py`, `platform_report.py`).

**As declared.**
- **N1 passes.** Motor input of the ROCm-trained fractional weights (77% of magnitudes
  fractional), 20 recorded ticks: CUDA segment reduction max relative error 1.4e-6, CUDA block
  SpMM 2.9e-7, against float64 (tolerance 1e-5).
- **N2(a) passes.** The segment reduction is bit-identical in 50 / 50 repeats on every tick; the
  block SpMM is not repeatable on CUDA (1-7 of 50 identical), as known.
- **N3 fails.** Identical CPU-drawn noise, same frames: the first sweep's motor spike counts are
  identical in all five runs (4,429 / 4,432 / 4,431) and the untrained AUCs agree to the fourth
  decimal; after learning, CUDA-segment vs CPU differs by up to **0.079** (250 frames: 0.778 /
  0.885 / 0.790 vs 0.791 / 0.922 / 0.869), CUDA-SpMM vs CPU by up to 0.061 (tolerance 0.02). The
  references behave: the CUDA-segment rerun is identical (AUCs and weights), the permuted-label run
  is flagged (|ΔAUC| up to 0.44 at 250 frames). By the declared rule the failure would mean "the
  implementation differs", to be examined in at most two bounded investigations.
- **N4: the batch check fails, so N4 is reported, not applied.** The 33-brain batch reproduces the
  production NVIDIA curve exactly before learning (0.1925 / 0.1796 / 0.3536) but not after 500
  frames (0.906 / 0.902 / 0.870 vs 0.871 / 0.854 / 0.848): on the same GPU, with the same code and
  noise seeds, changing only the batch size changes the trajectory. The run was stopped at frame
  749 (state saved, resumable) to free the GPU; only the 0- and 500-frame points exist.

**The two bounded investigations.**
1. *Lock-step CPU vs CUDA* (`lockstep-N3b.json`; same frames, the same noise tensor every tick, the
   rule applied to both): motor spikes are identical for the first 29 training steps; the
   eligibility's relative difference grows from 4e-8 (step 1) to 2e-5 (step 10), 2e-4 (19), 3e-2
   (25), then spikes start to differ (76 motor cells at step 30, 358 at step 40; weights 20%
   apart). Per step the computation agrees to float32 precision; the difference is not a wrong
   operation but a rounding-level difference that the learning loop amplifies by about five orders
   of magnitude in ~30 updates (weights feed the surrogate derivative, which feeds the update,
   which is RMS-normalised).
2. *Summation order only, same GPU* (CUDA-SpMM vs CUDA-segment, identical noise): up to **0.063**
   apart at 250 frames, the same size as CPU vs CUDA (0.079) and ROCm vs NVIDIA (0.081, §16.1).

**Reading.** The premise of the declared N3 rule ("a difference means the implementation
differs") is contradicted by the investigations: the rule's computation is the same to float32
precision on both devices, and any two executions that are not bit-identical (another device,
another summation order, another batch size) end up as different trajectories after a few dozen
updates. After 500 frames the learnt weight changes of CPU, CUDA-segment and CUDA-SpMM are
essentially decorrelated (‖Δm difference‖ ≈ 1.0-1.05 × ‖Δm‖) while their AUCs stay in the same
range. A tolerance of ±0.02 between single runs therefore cannot be met by any two non-identical
implementations, ROCm and NVIDIA included; the §16.1 failure says this, not that NVIDIA computes
something else. *Descriptive, post hoc (partial N4):* across nine noise realisations on NVIDIA the
AUC at 0 frames spans 0.15-0.25 / 0.14-0.20 / 0.34-0.38 per readout and at 500 frames 0.80-0.92 /
0.83-0.92 / 0.76-0.91 (SD 0.03-0.05); the ROCm values (0.238 / 0.152 / 0.346 and 0.816 / 0.872 /
0.855) lie inside these ranges. This is not the declared N4 criterion and is not used as one.
Not claimed: chaos in the dynamical-systems sense (no Lyapunov exponent was measured), or that the
sensitivity comes from the spiking nonlinearity rather than from the closed learning loop.

**Decision (lead, under the owner's mandate of 2026-10-09).** The campaign runs **entirely on
NVIDIA**: every arm of every comparison on this GPU with the declared pipelines (one graph per
process, three brains per batch, as on ROCm), ROCm results kept as a separate series used for
context only, never mixed into a comparison. Justification: N1-N2 and investigation 1 show the
CUDA path computes the declared rule; the declared cross-platform tolerances (§16, N3) fail and
stay failed, and investigation 2 shows they cannot be met even on one GPU. **Consequence for every
later reading:** a training run is one draw from a distribution that rounding alone spreads by
~0.03-0.05 AUC; differences between single runs of that size carry no information, and
comparisons must rest on several readouts / noise realisations (as the declared ones do), with
batch layouts identical across arms.

### 18.1 Supervised sample efficiency on identical data (declared 2026-10-09 19:04 UTC; see the note on what had been seen)

**Question.** On the same frames, labels, order, readouts and rule, does the connectome learn the
take-off label faster than its laterality-preserving nulls? (The supervised part of "sample
efficiency"; not RL.) On ROCm (§6.1) the connectome's curve was lower (curve-mean AUC 0.86 vs 0.93 /
0.90), a descriptive result.
**Measure.** Per brain, the curve-mean AUC = the mean of the held-out AUC over the 8 scoring points
after 0 (500 ... 4,000 frames), from the runs of §18 / §14 on NVIDIA (3 brains per graph).
**Test.** Connectome (3 brains) vs lateral nulls 0-2 (9 brains), exact two-sided Mann-Whitney
(no direction assumed), p < 0.05. Effect: difference of means with the range of the per-null means.
**Primary instance: the expert stream** (`cs-expert`, no data existed at declaration). Secondary:
the scripted stream (`cs-scripted`). *What had been seen:* while checking the queue at 19:04 UTC the
log of the running `cs-scripted nulllat:0` displayed its AUCs at 2,000 and 2,500 frames (0.93-0.96);
no other point of the null runs of either stream had been seen; the connectome's scripted curve on
NVIDIA was known (§16.1, curve-mean 0.871).
**Interpretation fixed now.** A difference is reported with its direction and is activity-confounded
(the nulls' motor layer fires ~1.8x the connectome's); "no difference shown at this power" otherwise.
With 3 vs 9 brains the smallest attainable two-sided p is 0.009; rounding alone moves a brain's AUC
by 0.03-0.05 at single points (§17.1), less on the curve mean.

### 18.1 result (2026-10-09 19:53 UTC): on the expert stream the connectome learns more slowly than all nine lateral-null brains; on the scripted stream no difference is shown

`artifacts/runs/mission/readings-18.json` (`scripts/diagnostics/mission_readings.py`), NVIDIA runs of
19:06-19:52 UTC. Curve-mean held-out AUC (8 points, 500-4,000 frames), per readout:

| stream | connectome | nulllat:0 | nulllat:1 | nulllat:2 | Mann-Whitney (3 vs 9, two-sided) | declared verdict |
|---|---|---|---|---|---|---|
| **expert (primary)** | 0.579 / 0.660 / 0.545 | 0.914 / 0.893 / 0.904 | 0.863 / 0.768 / 0.787 | 0.669 / 0.788 / 0.724 | U = 0, p = 0.009 | **difference: connectome lower** (mean −0.217; per-null means 0.904 / 0.806 / 0.727) |
| scripted (secondary) | 0.889 / 0.872 / 0.865 | 0.941 / 0.935 / 0.941 | 0.908 / 0.864 / 0.933 | 0.863 / 0.859 / 0.880 | U = 10, p = 0.60 | no difference shown at this power |

**Reading, as fixed in the declaration:** a difference in supervised learning speed on identical
data, connectome slower, activity-confounded (the nulls' motor layer fires ~1.8x the connectome's).
*Context, not part of the verdict:* the held-out frames are the scripted test set of 5-1 / 8-1 for
every stream (label on 39%, mostly Mario blocked, §12.7), so the expert-stream number measures how
well training on approach frames transfers to those frames; the null realisations themselves spread
widely (per-null means 0.73-0.90 on the expert stream), and the connectome lies below the lowest. On
the scripted stream, where training and test frames are of the same kind, the ROCm difference (§6.1:
0.86 vs 0.93 / 0.90) is not reproduced as a declared difference (connectome 0.88 vs 0.87-0.94).
Single-run AUC points move by 0.03-0.05 under rounding alone (§17.1); a −0.22 curve-mean gap is far
larger.

## 19. G-DAgger-v1: training on the agent's own frames (declared 2026-10-09 21:25 UTC, before any DAgger run; a change inside G only)

**Why.** §14's expert stream did not give useful frozen controllers so far (connectome 0 / 3, null 0
0 / 3 pass; declared classes pending S_fixed). Seen before declaring (from the play logs; labelled):
the expert-stream brains jump at the expert's overall rate (0.27-0.51 of decisions vs 0.32) but
21-41% of their label-off decisions are jumps, they miss 27-38% of label onsets and 48-62% of their
presses last one decision; their score's ranking on their own frames (S_own 0.64-0.80) is not above
the scripted-stream brains'. Imitation from expert states compounds errors in the states the
learner itself reaches (Ross, Gordon & Bagnell, AISTATS 2011; report/LITERATURE_G.md §A); the RAM
label can be queried on any frame the agent visits, which is what DAgger needs.
**Variant (G, learning only; `scripts/diagnostics/dagger_stream.py`, smoke-tested 19:13 UTC).**
Start from each graph's §14 expert-stream brains (round 1 = the expert stream). Two further rounds:
calibrate each brain's threshold exactly as the evaluation does (§20); the frozen brains play the
training levels with that rule from *collection* starts (own prefix seeds 20261500+, never the
harness-v2 evaluation starts); every 2nd decision in play, the current the brain saw and the RAM
take-off label v2 of that frame are stored (1,500 per brain per round); each brain then takes 1,500
updates of the unchanged Stage D supervised rule on frames drawn uniformly from its aggregate (the
expert stream + all its own frames). Pixels remain the only input; RAM is the label.
**Exposure-matched control (`--control`).** Identical, except that each round adds 1,500 frames of a
fresh expert stream instead of the agent's own frames: same number of updates and aggregate size.
**Arms.** connectome, nulllat:0, nulllat:1 (core), then nulllat:2; three readouts each; DAgger and
control brains of a graph share readouts and noise seeds.
**Evaluation (unchanged, outside G).** Frozen play on the training suite of harness v2 with the
threshold rule, S_fixed, the §13.1 / §14 report and classes (C / B / A, bars recomputed on the run's
negatives; C only if the in-batch expert passes).
**Readings, fixed now.**
1. *Primary, per graph:* DAgger "helps" in a graph if at least 2 of its 3 DAgger brains are C and at
   most 1 of its 3 control brains is C.
2. *Secondary, pooled over the core graphs (9 readout pairs, 12 with nulllat:2 reported apart):*
   paired DAgger − control difference of the play statistic (mean furthest-x gain over the rate-
   matched random jumpers, 64 starts); one-sided Wilcoxon signed-rank, DAgger > control at p < 0.05.
3. *Descriptive:* S_own, jump rate while the label is off, missed onsets, one-decision presses,
   DAgger vs control per brain; arm (wiring) differences descriptive only (three readouts, three
   realisations, activity-confounded).
**Prediction written now.** If the failure is compounding error in the agent's own states, DAgger
lowers off-label jumps and raises S_own relative to the control; whether that turns into C depends
on the one-decision presses, which DAgger does not address (no memory across decisions, P8).
**Budget and stop rule.** Training ~10-15 min per arm and graph; evaluation ~25 min play + S_fixed
per arm and graph; ≈ 6 h for the four graphs. Started only after §14 / §18's queue, one GPU job at
a time; a run that fails twice is reported, not retried.

### 18.2 Results (2026-10-09 22:40 UTC): NVIDIA baseline and §14 on the core graphs — no useful frozen controller from either stream; the §13.1 pattern does not replicate

Reports `frozen_probe/reportv2-training-20261009.json` (scripted stream) and
`reportv2x-training-20261009.json` (expert stream), NVIDIA runs 19:52-22:40 UTC; readings
`mission/readings-18.json`. nulllat:2 follows later in the queue (read apart).
**References, read first (both reports):** logging alignment passes; the in-batch scripted expert
passes (+70.7 px, 64 / 64 starts equal to the oracle path); always-run, every untrained brain and the
permuted-label brains fail. Scenario C is available. Bars (best negative): S_own 0.768 / 0.740,
S_fixed 0.588 (scripted / expert report).

| graph | scripted stream: class (play Δx px) | expert stream: class (play Δx px) |
|---|---|---|
| connectome | unclassified (+12), **B** (−14), **B** (+13) | **B** (+34), unclassified (−13), **B** (−4) |
| nulllat:0 | **B** (−113), unclassified (−81), **B** (+2) | unclassified (+45), unclassified (+43), unclassified (+34) |
| nulllat:1 | **B** (−28), **B** (−28), unclassified (−108) | unclassified (−20), unclassified (+24), **A** (−121) |

No brain passes the play test (0 / 9 per stream; the expert passes on the same starts).
**§14 declared verdict:** the expert stream helps in no graph (0 of 3 expert-stream brains C in each,
where ≥ 2 were required). **§18 declared replication:** the drop S_fixed − S_own is 0.069 / 0.163 /
0.103 (connectome) vs 0.114 / 0.083 / 0.231 and 0.254 / 0.140 / 0.066 (nulls 0-1); Mann-Whitney U = 7,
p = 0.36 → **not replicated**: the §13.1 pattern (connectome class A, nulls class B) is not
distinguished from noise-realisation variation. The class assignments themselves changed between
the ROCm and NVIDIA realisations of the same procedure (ROCm connectome A / A / unclassified, here
unclassified / B / B).
**Descriptive (timing, expert stream):** jump rates 0.27-0.56 (expert 0.32), jumps on 20-41% of
label-off decisions, 22-38% of label onsets missed, 48-62% of presses one decision long. Training on
approach frames raised the jump rate to the expert's level but not its timing.

## 20. Exploratory proposal, outside G: how long the jump button is held (declared 2026-10-10 00:23 UTC, before any hold-k run)

*Status: exploratory. It changes the decision interface (PROTOCOL §11), not learning; it does not
replace any declared comparison and is read separately. Motivated by data already seen (§13.1,
§18.2, the §19 collection logs): trained brains press A for one decision in 48-66% of presses, and
in their own play the take-off label is on mostly while Mario is blocked against an obstacle
(70-85% in the DAgger collections) — consistent with jumps too low to clear it. In Super Mario Bros
the jump height grows with how long A is held; the sweep regime has no memory across decisions (P8),
so a per-decision policy cannot hold by itself.*

**Rule hold-k** (generalises §13.2's hold6): a trigger (P(jump) ≥ tau, tau exactly as §20 / §13.1)
presses RIGHT+A+B for k decisions, then RIGHT+B for one; the random reference holds the same way at
the candidate's trigger rate (K = 5, harness v2 training suite, §13 seeds).
**Choice of k, on scripted references only (no brain data):** validate k = 2, 3, 4 exactly as §13.2
(three draw sets: the take-off v2 expert under hold-k must pass 3 / 3, not carried by one level; a
second random hold-k jumper and always-run must fail 3 / 3). k* = the smallest validated k. If none
validates, the proposal stops and is reported. Code check first: re-running the hold6 validation
with the generalised code must reproduce `validation-hold6-20261009.json` exactly.
**Application (if k* exists):** frozen play under hold-k* on the training suite, for the core graphs'
brains of the scripted stream, the expert stream and §19 (DAgger and control), untrained brains as
negatives, in-batch expert reference. **Reading:** C per brain (passes the hold-k* test, not carried);
counts per stream and graph, descriptive; untrained brains must fail (else the reading stops).
*Interpretation fixed now:* brains that become C under hold-k* but not under the threshold rule →
the stored discrimination is usable and the per-decision interface was the blocker; none → the
blocker is not (only) the hold. Any wiring difference here is exploratory (three readouts).

### 20.1 Validation (2026-10-10 00:31 UTC, scripted references only): k* = 2

`harness_v2/holdk/validation-hold{2,3,4,6}-check.json`. Code check: the hold6 rerun with the generalised
code reproduces `validation-hold6-20261009.json` exactly. Expert gain over its rate-matched random
hold-k jumpers (draws 1 / 2 / 3; all not carried by one level), second random jumper p:

| k | training suite | valid | unseen suite | valid |
|---|---|---|---|---|
| 2 | +319 / +324 / +321 px; random p 0.76 / 0.62 / 0.94 | **yes** | +189 / +180 / +177; p 0.63 / 0.29 / 0.96 | yes |
| 3 | +275 / +276 / +271; p 0.85 / 0.61 / 0.91 | yes | +178 / +190 / +168; p 0.53 / **0.005** / 0.81 | no |
| 4 | +358 / +377 / +362; p 0.67 / 0.38 / 0.63 | yes | +287 / +279 / +281; p 0.59 / 0.48 / 0.49 | yes |

(Always-run fails everywhere.) By the declared rule **k* = 2**. Note: under hold-2 the label-following
expert beats rate-matched random play by ~320 px on the training levels, against +71 px with the
per-decision rule (§13): holding A for two decisions is worth far more to the expert than its
per-decision timing. Brains are played under hold2 next (queued after §19).

### 19.1 Result (2026-10-10 02:24 UTC): DAgger does not produce useful frozen controllers and is not better than its exposure-matched control

Reports `frozen_probe/reportv2{d,c}-training-20261009.json` (references first: alignment passes, the
in-batch expert passes, every negative fails; C available in both). Collection logs
`artifacts/runs/dagger/`.

| graph | DAgger: class (play Δx px) | control: class (play Δx px) |
|---|---|---|
| connectome | unclassified (+41), **A** (−59), unclassified (+30) | unclassified (−30), unclassified (+11), unclassified (−3) |
| nulllat:0 | **A** (−54), **B** (−51), **B** (−36) | unclassified (+20), unclassified (−73), unclassified (+25) |
| nulllat:1 | unclassified (−33), **B** (−8), **A** (−48) | unclassified (−58), **B** (+14), **B** (+46) |

**Declared readings.** (1) DAgger helps in no graph (0 / 3 C in each; the controls 0 / 3 too).
(2) Paired DAgger − control play difference over the 9 core readout pairs: +71, −70, +33, −75, +22,
−61, +25, −23, −94 px; Wilcoxon signed-rank one-sided W = 15, **p = 0.82** → not better than the
control. (3) Descriptive: S_own DAgger − control −0.13 to +0.17 (5 of 9 up), off-label jump rate
−0.32 to +0.19 (6 of 9 down): the prediction (fewer off-label jumps, higher S_own) holds in some
brains and reverses in others; no consistent effect. In the collections the agents' own frames were
mostly Mario blocked against an obstacle with the label on (label on 45-64%, blocked 68-85% of those),
so aggregation added more "blocked → jump" examples, the kind the scripted stream already over-taught.

## 21. Exploratory analysis: how good must the score be? A noisy label expert (declared 2026-10-10 02:47 UTC, before running; references only, no brain)

*Status: exploratory, a calibration of the play measure; no brain is involved and no criterion
changes.* Seen before declaring: under hold2 the expert-stream connectome brains do not pass (+15 /
−12 / +20 px; expert +319), with trigger rates 0.27-0.39 (expert 0.20).
**Question.** Given hold2 on the training suite, how many missed take-off moments and how many
off-label triggers can a label-following policy afford and still pass the play test? Where do the
trained brains sit on that map?
**Policy.** On each eligible decision (hold2 rule), trigger with probability h if the RAM take-off
label v2 is on and f if it is off (seeded per start). Grid h ∈ {1, 0.7, 0.5, 0.3}, f ∈ {0, 0.05, 0.1,
0.2, 0.3}. Measured per configuration: the onset miss rate, the off-label trigger rate, the play test
against K = 5 random hold2 jumpers at its own trigger rate (64 starts, draw set 1), pass / not carried.
**Reading.** The region of (onset miss rate, off-label trigger rate) where the noisy expert passes,
against the brains' measured values under the same rule (descriptive). If the brains sit outside the
passing region, the blocker is the quality of the stored score at decision time (what is learnt and
seen), not the hold; if inside, something else (e.g. timing within the label window).

### 21.1 Result (2026-10-10 02:55 UTC, exploratory): a useful controller needs ≤ ~5% off-label triggers; the brains make 15-50%

`artifacts/runs/mission/noisy_expert-hold2.json` (`scripts/diagnostics/noisy_expert.py`). Check: h = 1,
f = 0 reproduces the hold2 expert exactly (+318.9 px). Play gain over rate-matched random hold2
jumpers (training suite, 64 starts); passes marked ✔ (all passes are not carried by one level):

| per-decision hit prob. h \ off-label trigger prob. f | 0 | 0.05 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| 1.0 (onset miss 0%) | +319 ✔ | +184 ✔ | +38 | +51 | +61 |
| 0.7 (onset miss ~5%) | +304 ✔ | +121 ✔ | +20 | +28 | +62 |
| 0.5 (onset miss ~9%) | +171 ✔ | +28 | +14 | +55 | +39 |
| 0.3 (onset miss ~11-14%) | +51 | +10 | −10 | +72 | +3 |

**Reading (descriptive).** The play test is passed only with off-label triggers at ≤ ~4-5% of
eligible decisions (and most onsets caught); at ~9% off-label triggers even a policy that never
misses a take-off moment does not beat random jumping at its own rate — false jumps cost as much as
right jumps gain. The trained brains of every wiring, stream and variant jump on 15-50% of label-off
decisions under the threshold rule (§18.2, §19.1), far outside the passing region. **The blocker is
the specificity of the stored score at decision time** (a ranking with AUC ~0.65-0.9 on the agent's
own frames cannot reach a ≤ 5% false-trigger rate while catching take-off moments), not the hold and
not the wiring. Hypothesis for why (not tested here): the take-off label marks a short window per
obstacle, and the visual frames just before and after it are near-identical in the eye's current, so
a per-frame score cannot separate them sharply (P6: the interface is mostly a luminance detector).

### 20.2 Procedure change (2026-10-10 02:57 UTC): hold2 plays batched per graph

The 11 hold2 plays not yet started are replaced by one play per graph with every stream's brains in
one batch (`merge --stream all`: scripted, expert, DAgger, control; for the connectome also the
permuted-label brains and the two scripted references; untrained brains added by the evaluator):
same brains, rule, tau procedure, starts and random-jumper seeds, ~2.5 h of GPU saved. Frozen play
has no learning, so batching can change P(jump) only at rounding level; the two plays already run
per stream (connectome and null 0, expert stream) are compared with the batched ones as a check
(reported). Reading unchanged (§20).

## 22. Exploratory: a specificity-first threshold (declared 2026-10-10 02:57 UTC, before any run)

*Status: exploratory, decision interface (outside G), motivated by data already seen:* §21.1 (a
useful controller needs ≤ ~5% off-label triggers) and `mission/own_frame_roc.json` (on their own
frames, several brains reach a true-positive rate of 0.5-0.69 at a 5% false-positive rate, while
the Youden threshold puts them at 15-50% off-label jumps).
**Rule hold2f05.** As hold2 (§20), but each brain's tau is the lowest threshold whose false-positive
rate on the *same calibration frames* (training levels, relabel procedure, seed 20261017 — never the
play data) is ≤ 0.05. Random references hold the same way at the candidate's trigger rate.
**Validation first:** the scripted references do not depend on tau, so the hold2 validation (§20.1)
covers the harness; the in-batch expert must pass and every untrained / permuted brain must fail,
otherwise the reading stops.
**Application:** the batched brains of §20.2 (connectome, nulls 0-1). **Reading:** C per brain,
counts per stream and graph, descriptive. *Fixed now:* C brains here but not under Youden + hold2 →
the stored scores contain a usable controller and the threshold criterion (sensitivity-specificity
balance on scripted frames) was the blocker; none → the stored scores are not specific enough on the
agent's own frames at any threshold that keeps triggers.

## 23. An independent block for the first useful controllers (declared 2026-10-10 06:24 UTC, before any block-1 run)

*Status: exploratory (the hold2 interface is §20's exploratory proposal); it tests a finding seen in
the data, so block 0 is development and block 1 is the test.* **Seen (block 0, §20.2):** under hold2
on the training suite, two of the three null-0 expert-stream brains are useful frozen controllers
(+128 / +170 px in the batched play, p 0.005 / 0.001; +101 / +210 px, p 0.022 / 0.001, in the
separate play — not carried by one level in either), the third is not (+36 / +45); no connectome
brain of any stream is (0 / 12); null 1 and null 2 not yet seen.
**Block 1 (protocol §11.2: new readout seeds between independent blocks).** The expert stream of §14
re-trained for connectome, nulllat:0, nulllat:1, nulllat:2 with three new fixed readouts
(`block_seeds(6)[3:6]`) and new noise seeds (9300 + i); same frames, order, rule and budget
(`common_stream.py --stream expert --block 1`). Frozen play under hold2 on the harness-v2 training
suite (same starts and random-jumper seeds; tau as §20 / §13.1; untrained negatives; in-batch expert
for the connectome).
**Readings, fixed now.**
1. *Does hold2 + expert-stream competence replicate?* In block 1, count C brains per graph. The
   null-0 finding replicates if ≥ 2 of 3 null-0 block-1 brains are C; "competence from this procedure"
   is claimed for a wiring only if it holds in both blocks.
2. *Wiring (exploratory):* blocks 0 + 1 pooled, 6 brains per wiring: connectome vs the 18 null brains,
   two-sided Fisher exact test on C counts and two-sided Mann-Whitney on the play gain (Δx over
   rate-matched random hold2 jumping), p < 0.05; per-null counts reported (realisation spread).
   Interpretation: activity-confounded (the nulls fire ~1.8x), an interface chosen after seeing data,
   one connectome; "no difference shown at this power" otherwise.
3. Untrained brains must fail and the in-batch expert must pass, or the block is not read.
**Budget.** ≈ 30 min training + ≈ 2-2.5 h play; queued before §22.

### 21.2 Correction (2026-10-10 06:52 UTC): §21.1's "≤ ~5% off-label triggers" is too strong for brains

Two null-0 expert-stream brains pass the hold2 play test (§23, block 0) with off-label trigger rates
of 0.14 and 0.30. The noisy expert's false triggers are independent of the frame (uniform in time);
a brain's are not — they can fall where a jump is harmless or useful although the label is off (the
label v2 marks a narrow take-off window). §21.1's map therefore bounds *random* false triggers only;
the sentence "the blocker is the specificity of the stored score at decision time" is withdrawn as a
general statement and kept as "random-like false triggers above ~5% cancel the gain of correct ones".

### 20.3 Result (2026-10-10 07:27 UTC, exploratory): under hold2, 3 of 36 trained brains are useful controllers — none of the connectome's

`frozen_probe/reportv2a-training-hold2-20261009.json` (batched plays of §20.2). References first:
the in-batch expert passes (+319 px), every untrained and permuted-label brain fails; C available.
Play gain over rate-matched random hold2 jumping (px), C marked ✔:

| graph | scripted | expert | DAgger | DAgger control |
|---|---|---|---|---|
| connectome | −35 / −47 / +15 | +18 / −12 / +21 | −1 / +19 / −16 | +80 / −56 / +33 |
| nulllat:0 | −39 / +6 / −93 | **+128 ✔ / +170 ✔** / +36 | −14 / −18 / −54 | +72 / +108 / +54 |
| nulllat:1 | −73 / −53 / −34 | +46 / −39 / −27 | **+78 ✔** / +27 / −55 | +10 / −5 / +3 |

**Reading (as fixed in §20).** Holding A for two decisions turns 3 of 36 trained brains into useful
frozen controllers (null 0 expert ro1 / ro2 — both again C in their separate play, +101 / +210; null 1
DAgger ro1); 0 of 12 connectome brains. The stored discrimination is usable for control in a few
brains once the interface holds the button; for most it is not. Wiring differences: exploratory and
underpowered here (3 readouts, one block); tested with an independent block in §23.

### 23.1 Result (2026-10-10 09:52 UTC, exploratory): the null-0 competence does not replicate by the declared rule; no wiring difference shown

`mission/readings-23.json` (`scripts/diagnostics/block_readings.py`); block-1 report
`frozen_probe/reportv2y-training-hold2-20261009.json` (references first: the in-batch expert passes,
every untrained brain fails). Play gain over rate-matched random hold2 jumping, expert stream (px):

| graph | block 0 (readouts 1-3) | block 1 (new readouts and noise) | C (b0 / b1) | mean gain |
|---|---|---|---|---|
| connectome | +18 / −12 / +21 | +12 / −42 / −44 | 0 / 0 | −8 |
| nulllat:0 | **+128 ✔** / **+170 ✔** / +36 | **+151 ✔** / +61 / +118 (p 0.050) | 2 / 1 | +111 |
| nulllat:1 | +46 / −39 / −27 | +8 / −10 / −32 | 0 / 0 | −9 |
| nulllat:2 | −3 / +114 / +52 | −17 / +23 / +49 | 0 / 0 | +36 |

**Declared readings.** (1) Null 0 block 1: 1 of 3 C (≥ 2 required) → **the competence is not
replicated** by the declared rule, although all six null-0 brains have positive gains (+36 to +170).
(2) Pooled wiring comparison: C 0 / 6 (connectome) vs 3 / 18 (nulls), Fisher p = 0.55; gain −8 vs +46
px, Mann-Whitney U = 25, p = 0.056 → **no difference shown at this power.** (3) References pass.
**Descriptive.** The between-null spread is as large as any connectome-null gap: one null realisation
(0) plays consistently better than the connectome, the two others (1, 2) do not. This is the P3
situation: with one connectome and three null realisations, a single realisation can drive a
difference; it is a property of that realisation (or of chance), not of nulls as a class, unless more
realisations say otherwise.

## 24. The connectome against a distribution of 40 nulls (declared 2026-10-10 11:12 UTC; null realisations 3-39 not built yet, no run of this section exists)

**Why (methodology, from the literature check of 2026-10-10, `report/LITERATURE_NULLS_STATS.md`).**
With one connectome and K null realisations, the question "is the connectome an outlier of the null
distribution?" has an empirical p-value of at least 1 / (K + 1) (Phipson & Smyth 2010): with K = 3 no
test can reach 0.05. The readout-level tests of §18.1, §18.2 and §23 treated brains as independent
although they are nested in three graphs; their verdicts stand as comparisons *of these brains*
(connectome vs these three realisations), not as inference about the null class (correction recorded
in §24.0 below). The one large difference found (§18.1, expert stream: connectome curve-mean AUC
0.55-0.66 vs 0.67-0.91) is the right candidate for a graph-level test: cheap (one training run per
graph), measured on identical data.

### 24.0 Correction of scope (same time)
§18.1's "difference: connectome lower" (p = 0.009) and §23's "no difference shown" are statements
about 3 connectome brains vs 9 (or 18) null brains from 3 realisations; pooling brains across
realisations as independent inflates false positives when realisations differ (they do: per-null
means 0.73-0.90). They are not evidence that the connectome differs from (or matches) laterality-
preserving nulls as a class. Unchanged as reported; scope narrowed.

**Design.** Null realisations `nulllat:s`, s = 0..39 (40; 0-2 exist, 3-39 built by
`scripts/diagnostics/build_nulls.py` from their seeds, same constructor, round-trip checked, and
validated with `null_validate.py`: degrees, signs, weight multisets, depth strata and side pairs exact).
Per graph: the §14 expert-stream run exactly as `cs-expert` (same frames, order, held-out frames,
readouts block 0, noise seeds 8300 + i, three brains per batch, 4,000 frames). The existing runs of the
connectome and nulls 0-2 (2026-10-09 19:19-19:52 UTC) are used as they are.
**Statistic (per graph).** S = mean over the three readouts of the curve-mean held-out AUC (8 points,
500-4,000 frames).
**Primary test.** Two-sided empirical p of the connectome's S within the 40 nulls' S:
p = min(1, 2 · min[(1 + #{null ≤ S_c}), (1 + #{null ≥ S_c})] / 41); difference if p < 0.05 (only if
the connectome lies beyond every null, min p = 0.049). Reported: rank, z = (S_c − mean) / SD of the
nulls.
**Secondary (descriptive).** (a) Activity: untrained motor spikes per decision on the 300 held-out frames
for every graph (`scripts/diagnostics/graph_activity.py`); across the 40 nulls, the regression of S on
log activity, and whether the connectome lies inside the 95% prediction interval — inside means its S
is what a null with its activity would be expected to have (no evidence beyond activity). (b) Seed
variability: the connectome's block-1 run (§23, new readouts and noise) placed in the same null
distribution (block-1 statistics for nulls exist only for 0-2; reported, not tested).
**Interpretation fixed now.** p < 0.05 → the connectome learns the take-off label at a different speed
than laterality-preserving nulls (direction reported), on this stream and interface; then (a) says
whether its activity level accounts for it. p ≥ 0.05 → no difference shown against the null
distribution (not equivalence: no smallest effect of interest was declared, so no TOST).
**Budget.** Builds ≈ CPU only; training ≈ 37 × 6.5 min ≈ 4 h GPU; activity ≈ 20 min. Placed before the
remaining exploratory §22 plays.

### 24.1 Null validation (2026-10-10 13:52 UTC): the 37 new realisations pass

`artifacts/runs/mission/nulls/null_validate-24.json`: for nulllat:3-39, out- and in-degrees exact, edge
count, sign and weight multisets equal, depth-stratum edge classes and side-pair counts equal; sampled
edge overlap with the connectome 0.053-0.054 (nulllat:0-2: 0.054). Built in 238-260 s each.

## 25. The same test at matched activity (declared 2026-10-10 14:53 UTC, before §24's reading and before any §25 run)

*A matched-operating-regime comparison (PROTOCOL §9.5), distinct from §24's shared-parameter one; it
does not replace it. Motivation: under identical parameters the connectome's motor layer fires about
half as much as every null; the literature finds that verdicts flip with the dynamical regime
(Suárez 2021, 2024; Damicelli 2022; `report/PROTOCOL_ASSESSMENT.md`).*
**Matching.** For each null, one global gain (the dynamics' single scalar) chosen by bisection so that its
untrained motor spikes per decision on the 300 held-out frames equal the connectome's at the frozen
operating point (relative tolerance 1%; `scripts/diagnostics/match_gain.py`; search in [0.3, 1] × the
frozen gain). The connectome keeps its frozen gain and its existing run. Nulls whose target cannot be
reached in that range are reported and excluded.
**Runs.** For each matched null, the §24 run (`cs-expert`) at its matched gain
(`common_stream.py --gain-file`), everything else identical.
**Statistic and test.** As §24 (S per graph; two-sided empirical p among the K matched nulls; rank, z).
**Joint reading, fixed now.** §24 p < 0.05 and §25 p ≥ 0.05 → the connectome's learning-speed difference
is accounted for by its activity level (operating regime); both < 0.05 → a difference beyond gross
activity; §24 ≥ 0.05 and §25 < 0.05 → a difference masked by activity; neither → no difference shown in
either regime. "Accounted for" is a statement about this measure only.
**Budget and order.** ≈ 1.5-2 h gain search + ≈ 4.3 h training (40 nulls), GPU, queued right after
§24's reading and before the remaining nulllat:2 declared runs and §22 (exploratory), whose value is
lower now that §24 / §25 use 40 realisations; the order change is recorded here.

### 24.2 Deviation recorded (2026-10-10 14:54 UTC): an interim look at §24

While testing the generalised reading script (`null_distribution.py --matched` for §25), the §24 reading
was executed on the 34 nulls then finished and its output seen: connectome S = 0.595, 5th from the bottom
of 35 graphs, null mean 0.740, SD 0.112, z = −1.30. No criterion, threshold, design or order is changed by
this (the §24 and §25 declarations stand as written, §25 was declared before it); the declared reading is
the one at K = 40, reported below when complete.

### 24.3 Result (2026-10-10 15:52 UTC): against 40 nulls the connectome's learning speed is not unusual

`mission/readings-24.json` (`null_distribution.py`), `mission/graph_activity.json`. All 40 null runs complete.
- **Primary (declared):** connectome S = 0.595 (readouts 0.579 / 0.660 / 0.545); nulls: mean 0.731, SD 0.121,
  range **0.455-0.950**; the connectome is **7th from the bottom of 41** graphs, z = −1.13, two-sided empirical
  **p = 0.34 → no difference shown against the null distribution.**
- The three nulls of §18.1 (S 0.904 / 0.806 / 0.727) were the 39th, 29th and 19th of 40: §18.1's
  "connectome lower than all nine null brains" reflected those three realisations, not nulls as a class
  (as §24.0 anticipated).
- Seed check: the connectome's block-1 run (new readouts and noise, §23) gives S = 0.591, also 7th from the
  bottom.
- **Activity (secondary, descriptive):** untrained motor spikes per decision: connectome 4,565, nulls
  7,778-8,608 (all ≈ 1.8x). Across nulls, S is not predicted by activity (slope on log activity n.s., r = 0.21,
  p = 0.18); the connectome lies far outside the nulls' activity range, so the regression extrapolates and its
  prediction interval (−0.85 to 1.07) is uninformative: this measure cannot say whether activity accounts for
  anything. §25 (matched activity) addresses it directly.
**Reading.** On the one measure where three realisations suggested a large wiring effect, a proper null
distribution shows the connectome inside it. The null realisations themselves differ enormously (0.46-0.95)
in how fast they learn the same label from the same frames: realisation variance, not wiring, dominates.

### 25.1 Result (2026-10-10 16:23 UTC): the matched-activity comparison cannot be run as declared

`mission/matched_gains.json`. The connectome's untrained motor activity at the frozen gain is 4,565 spikes
per decision. For every one of the 40 nulls, activity at the frozen gain is 7,99x-8,6xx and at 0.3x the
gain still ~7,450-7,500 (e.g. null 0: 8,142 → 7,499): **no null reaches the target in [0.3, 1] × gain**, so
by the declared rule all are excluded and §25 is not run. Observation: the motor activity of the nulls is
nearly insensitive to the recurrent gain over a 3.3-fold range (as Gate E noted for the connectome:
"the raw operator's rate is flat across a threefold range of coupling"); the connectome's lower activity
is therefore not a matter of where the common gain sits but of the wiring itself, at least over that
range. A wider exploration is §25.2 (exploratory).

### 25.2 Exploratory: activity against gain over a wide range (declared 2026-10-10 16:25 UTC, before running)
Untrained motor spikes per decision on the same 300 held-out frames, connectome and nulllat:0-2, at gain ×
{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3}. Descriptive only: is there any gain at which nulls match the
connectome's 4,565, and is the connectome's own curve shifted or differently shaped?

### 25.3 Result (2026-10-10 16:40 UTC, exploratory): the connectome's activity ceiling is structural

`mission/regime_map.json`. Untrained motor spikes per decision vs gain multiplier:

| gain × | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| connectome | 0 | 221 | 2,588 | 3,702 | 4,565 | 5,082 | 4,884 |
| nulllat:0 | 2 | 216 | 6,294 | 7,499 | 8,142 | 8,198 | 8,342 |
| nulllat:1 | 2 | 142 | 5,898 | 7,487 | 8,252 | 8,423 | 8,567 |
| nulllat:2 | 2 | 470 | 5,996 | 7,342 | 7,851 | 8,040 | 8,178 |

All graphs ignite between 0.03x and 0.1x the gain and then saturate; the connectome saturates at ~4.5-5.1k,
the nulls at ~7.8-8.6k (≈ 0.6x over the whole saturated range). A null reaches the connectome's activity
only on its ignition slope (~0.05x gain), where the connectome itself is near 1-2.5k: matching gross
activity with one global gain would compare a saturated connectome with near-threshold nulls, not the same
regime. The activity difference is a structural ceiling (earlier analyses attribute 55-76% of it to the
in-strength distribution; PERCEPTION_CREDIT_STUDY §19), not an operating-point offset; a matched-regime
comparison needs a null that preserves in-strength (Milisav et al. 2025), not a gain change.

## 26. A stricter null rung: in-strength kept (declared 2026-10-10 16:41 UTC, before any of these nulls exist)

**Why.** The literature recommends a ladder of nulls (Váša & Mišić 2022) and strength-preserving nulls for
weighted networks (Milisav et al. 2025); §25.3 shows the activity gap is a structural ceiling, of which
55-76% followed the in-strength distribution on 2026-10-06. §24 used the laterality-preserving null (rung
b); this is rung c.
**Nulls.** `depth_stratified(g, seed, lateral=True, weights_by="dst")` (spec `nullin:<s>`, s = 0..39): rung b,
and each target neuron receives back its own incoming weights, so every neuron's (signed) in-strength is
exact; out-strength is not kept. Validated like §24.1 (degrees, signs, weight multiset, strata, sides) plus
in-strength equality.
**Runs and test.** Exactly §24 (expert stream, readouts block 0, noise 8300 + i; S per graph; two-sided
empirical p of the connectome among the 40; rank, z) and the untrained motor activity of each null.
**Interpretation fixed now.** p < 0.05 → the connectome's learning speed differs from nulls that keep
degrees, depth, sides and in-strength (direction reported); p ≥ 0.05 → no difference shown at this rung.
Activity: reported, and whether these nulls close the activity gap (descriptive).
**Budget.** Builds ~30 min CPU; training ≈ 4.3 h GPU, queued after the current items (overnight).

### 26.1 Null validation (2026-10-10 17:24 UTC): the 40 in-strength nulls pass

`mission/nulls/instrength_validate.json` (all 40): signed in-strength of every neuron exact, in- and
out-degrees exact, weight and sign multisets equal; out-strength changed (max |Δ| 84,506), as designed.
`mission/nulls/null_validate-26.json` (nullin:0, 17, 39): depth-stratum edge classes and side pairs equal,
edge overlap with the connectome 0.054. Built in ~4-5 min each.

## 27. §24 on the scripted stream (declared 2026-10-10 17:52 UTC, before any of these runs)

The §24 design applied to the scripted stream (`cs-scripted`, RL_STUDY §12): the 40 laterality-preserving
nulls (nulllat:0-39; 0-2 exist from 2026-10-09), S per graph (mean over three readouts of the curve-mean
held-out AUC), two-sided empirical p of the connectome (its 2026-10-09 17:51 run) among the 40, rank, z.
Interpretation as §24. Purpose: whether §24's reading holds on the second training stream (where §18.1 saw
no difference with three nulls). Queued last (overnight, ≈ 4 h); order change recorded: §26 now runs before
the remaining exploratory §22 plays.

## 28. Speedrun feasibility (owner's request 2026-10-10 18:10 UTC; declared 18:12 UTC, exploratory)

*The speedrun is the project's stated finality (WORKING_AGREEMENT §0). The owner asks to study the
feasibility of the any% warp route (1-1, 1-2 → 4-1, 4-2 → 8-1, 8-2, 8-3, 8-4) in under 10 minutes of game
time, with TAS / human records as comparators. Exploratory: no wiring comparison, no criterion of an
earlier section changes.*
**Reserved levels are not opened.** 1-2, 4-2, 8-2 and 8-3 are held-out confirmation levels (§7.1): nothing
is run on them; they are analysed from the literature and TAS knowledge only. 8-4 has no start state.
Measurements use the development levels of the route: 1-1, 4-1, 8-1.
**(a) Ceiling of the label / interface paradigm (scripted, no brain).** From each level's start state,
until flag, death or 2,400 decisions (160 s of game time): always-run; the take-off v2 label expert
(per-decision); the label expert under hold2, hold4 and hold6. Measured: cleared or not, furthest x, game
time to the flag (decisions × 4 frames / 60.1 fps). These bound what a brain that perfectly imitated the
label could do with these interfaces.
**(b) Where the brains are.** From the existing frozen-play logs (harness v2 training suite, all streams,
rules and graphs): how often any brain reached a flag, and its furthest x relative to the level's flag x.
**(c) Requirements** of a sub-10-minute route (literature: `report/LITERATURE_SPEEDRUN.md`) against (a)
and (b): which capabilities are missing, which need memory or planning across decisions (impossible in
the sweep regime, P8).

### 28.1 Measurements (2026-10-10 18:12 UTC): the paradigm's ceiling is below clearing a single level

`mission/speedrun_ceiling.json` (`speedrun_ceiling.py`, scripted, level start states, ≤ 2,400 decisions):

| level | always-run | label expert | label + hold2 | label + hold4 | label + hold6 |
|---|---|---|---|---|---|
| 1-1 | death, x 312 | death, x 694 | death, x 690 | death, x 694 | death, x 696 |
| 4-1 | death, x 323 | time-out, x 657 | death, x 1,842 | death, x 2,111 | death, x 2,112 |
| 8-1 | death, x 227 | death, x 559 | stall, x 546 | death, x 559 | death, x 342 |

**No scripted policy built on the take-off label v2 and the RIGHT+B / RIGHT+A+B action pair clears 1-1, 4-1
or 8-1 from the start.** A brain that imitated that label perfectly, under any of these decision rules, would
not clear a level either. (b) In all frozen-play logs of the mission (harness v2 training suite, ≤ 600
decisions per episode): 0 flags in 8,064 trained-brain episodes; furthest x 1,675 (1-1), 1,864 (2-1), 2,960
(4-1), 2,242 (7-1) — close to the untrained brains' luckiest episodes (1,559 / 1,649 / 2,535 / 1,938).
**Reading (exploratory).** Before any question of wiring, the teaching signal and action set used so far
cap the achievable play far below a level clear; a speedrun requires a different teacher (one that clears the
route: e.g. a planner or recorded expert inputs) and a richer action / timing interface.

### 18.3 nulllat:2 added to §14, §18, §19 (2026-10-10 18:24 UTC; declared readings rerun with --force)

References pass in all four reports. nulllat:2 brains, play Δx (class): scripted −6 (B), −7 (B), −41 (B); expert
−56, +54, −102 (all unclassified); DAgger +2 (B), +8, −56; control −40, −34, +9. **No nulllat:2 brain is C on any
stream**: §14 helps in none of the four graphs; DAgger in none. §18 replication, secondary (connectome 3 vs
nulls 0-2, 9 drops; nulllat:2 drops 0.35 / 0.29 / 0.29): Mann-Whitney U = 7, p = 0.14 → not replicated
(primary p = 0.36, §18.2). These brain-level comparisons are limited by the three-realisation scope of §24.0.

### 26.2 Result (2026-10-10 23:08 UTC): no difference shown at the in-strength rung either

`mission/readings-26.json`, `mission/graph_activity.json`. 40 in-strength-preserving nulls, all runs complete.
- **Primary (declared):** connectome S = 0.595; nulls mean 0.739, SD 0.105, range 0.544-0.903; the connectome is
  **4th from the bottom of 41**, z = −1.38, two-sided empirical **p = 0.20 → no difference shown** against
  nulls that keep degrees, depth, sides and every neuron's in-strength. Block-1 check: S = 0.591, also 4th.
- **Activity (descriptive):** untrained motor spikes per decision 5,771-6,377 for these nulls (rung b:
  7,778-8,608; connectome 4,565): keeping in-strength closes ≈ 60% of the activity gap, consistent with the
  55-76% of 2026-10-06; every null still fires more than the connectome. Across these nulls S vs log activity:
  r = −0.27, p = 0.09; the connectome lies outside the nulls' activity range (extrapolation), so its prediction
  interval is not interpreted.
**Reading.** At both rungs (b: rank 7 of 41, p = 0.34; c: rank 4 of 41, p = 0.20) the connectome's supervised
learning speed on identical data lies inside the null distribution. The trend is towards the lower part at
both rungs; with K = 40, a two-sided difference would have required the connectome to lie beyond every null.

### 22.1 Result (2026-10-11 01:52 UTC, exploratory): a specificity-first threshold makes play worse

`frozen_probe/reportv2a-training-hold2f05-20261009.json` (references first: the in-batch expert passes,
negatives fail; C available). Under hold2 with tau set for ≤ 5% false positives on the calibration frames,
1 of 36 trained brains is C (null 0, DAgger control ro1, +85 px); connectome 0 / 12 (all −23 to −90 px);
null 0 gains −79 to +85, null 1 −52 to +11. The two null-0 expert-stream brains that were C under hold2 with the
Youden threshold (§20.3) are not (+22, −38). **Reading (as fixed in §22):** the stored scores are not specific
enough on the agent's own frames at a threshold that keeps triggers rare: cutting false jumps to 2-5% leaves
the remaining jumps mistimed, and play falls below rate-matched random jumping in most brains of every wiring.

### 24.4 Descriptive addition (declared 2026-10-11 02:52 UTC, before running): how much of the spread is the graph?

Resume N4 (`cs-noise-spread`, §17; stopped at frame 749, state saved) to completion after §27: nine noise
realisations × three readouts of the connectome on the scripted stream (`cs_noise_spread.py` uses
`cs-scripted`'s stream). Reported, not tested: the within-graph (seed) SD of S for the
connectome on the scripted stream vs the between-graph SD of the 40 nulls on the same stream (§27), and the
intraclass correlation ICC = σ²_graph / (σ²_graph + σ²_seed) (Aarts et al. 2014; `LITERATURE_NULLS_STATS.md`).
Caveat fixed now: N4's 33-brain batch is a different batch layout from the production runs (§17.1), which
changes trajectories at rounding level; it estimates seed spread, not production values.

### 27.1 Result (2026-10-11 05:22 UTC): on the scripted stream too, the connectome sits inside the null distribution

`mission/readings-27.json`. Connectome S = 0.875 (readouts 0.889 / 0.872 / 0.865); 40 laterality-preserving
nulls: mean 0.873, SD 0.115, range 0.477-0.969 (skewed: most 0.85-0.97, a tail of slow realisations); the
connectome is **11th from the bottom of 41**, z = 0.02, two-sided empirical **p = 0.54 → no difference shown.**
Activity does not predict S across nulls (r = 0.15, p = 0.37; the connectome is outside the nulls' activity
range).
**Summary of the graph-level tests (§24, §26, §27).** Supervised learning speed on identical data, connectome vs
40 nulls: expert stream, rung b — rank 7 / 41 (p = 0.34); expert stream, rung c — rank 4 / 41 (p = 0.20);
scripted stream, rung b — rank 11 / 41 (p = 0.54). No difference shown on either stream or rung.

### 24.5 Result of §24.4 (2026-10-11 06:53 UTC): the graph, not the seed, carries the spread

`mission/readings-24-4.json` (N4 complete, `platform/cs_noise_spread-nvidia.json`). Scripted stream, connectome:
S over nine noise realisations = 0.869-0.887 (SD **0.0057**; production run 0.875); the 40 rung-b nulls (§27):
SD 0.115 → graph SD ≈ 0.115, **ICC ≈ 0.998**. Across its nine seeds the connectome's rank among the 40 nulls
stays 10th-15th of 41. Single AUC points move by 0.03-0.08 between executions (§17.1), but the curve mean S
is stable: one run per graph was enough for S, and the §24 / §26 / §27 readings do not hinge on the seed.
Also: S = 0.497 for the permuted-label brains and 0.860 for eta/2 (one noise set).
**N4, §17's criterion, now complete (descriptive only: its batch check failed, §17.1):** T (largest D among
calibration pairs) = 0.176; validation pairs 0 / 6 above T; permuted brains flagged 4 / 4 (D ≈ 0.72); eta/2
flagged 0 / 4 (D 0.10-0.13): the max-|ΔAUC| criterion would not detect a halved learning rate. ROCm vs NVIDIA
(set 0): D = 0.090 ≤ T. Consistent with §17.1's decision; not used as a criterion.

## 29. Speedrun stage S0 probe: can a savestate lookahead planner clear the route's development levels? (declared 2026-10-11 06:53 UTC, before any run; exploratory, no brain)

*Roadmap stage S0 (`report/SPEEDRUN_FEASIBILITY.md` §5): a teacher that clears the route. Reserved levels are not
opened; development levels 1-1, 4-1, 8-1 only.*
**Planner (random-shooting model-predictive control on the emulator, RAM used by the planner only).** At each
decision, from a saved emulator state, K = 24 random action sequences of H = 24 decisions are simulated; each
decision is one of RIGHT+B, RIGHT+A+B, NOOP, LEFT (the existing action set), drawn in runs (hold lengths 1-8
decisions) so jumps can be held; a sequence scores its furthest x, minus a large penalty if Mario dies; the first
4 decisions of the best sequence are executed, then the planner replans. Seeds 20261800 + level index.
**Measured.** Per level from the level start state: cleared (flag) or not, furthest x, game time to the flag
(decisions × 4 frames / 60.1 fps), deaths (the run stops at the first death).
**Reading.** Cleared → a teacher exists for that level (S0 met there), and its game time is compared with the
human / TAS splits (`report/LITERATURE_SPEEDRUN.md`); not cleared → S0 needs a stronger planner. Exploratory; no
criterion of any section changes.

### 29.1 First planner run and a variant (2026-10-11 07:09 UTC, exploratory)

`mission/planner_teacher-k24-h24.json` (as declared, K 24, H 24): 1-1 death at x 1,558 after 14.4 s of game time;
4-1 death at x 1,426 (77.5 s); 8-1 death at x 3,750 after 176.8 s (beyond 8-1's 300-unit timer, ≈ 120 s: a
time-out). Further than the label teacher (694 / 2,112 / 559) but no level cleared. **Variant v2 (decided after
seeing this, labelled):** K 32, H 40 (2.7 s of lookahead), RIGHT+B / RIGHT+A+B only (no NOOP / LEFT), to remove
deaths just beyond the horizon and time-outs. Running.

### 29.2 Planner v2 (2026-10-11 07:11 UTC, exploratory): a teacher clears 1-1 and 4-1

`mission/planner_teacher-k32-h40.json`: **1-1 cleared** in 318 decisions = **21.2 s** of game time (flag at x 3,161);
**4-1 cleared** in 361 decisions = **24.0 s** (x 3,593); 8-1 not cleared (x 3,980, time-out after 165 s). The first
flags reached in the project by any agent that does not replay a recording. Pace: a sub-10-minute route allows
≈ 75 s per level on average (8 levels, transitions included); 21-24 s is well inside it. Stage S0 is met on 2 of 3
development levels; 8-1 needs a stronger search (a v3 with K 64, H 60 is tried on 8-1 only). The recorded action
sequences (`actions`) are candidate teaching signals for stage S2 (imitation), not used here.

### 29.3 8-1 resists (2026-10-11 07:53 UTC, exploratory)
v3 (K 64, H 60, RIGHT+B / RIGHT+A+B only) on 8-1: time-out at x 3,913 after 163 s, like v2 (x 3,980): the search
stalls before x ≈ 3,900-4,000 for > 100 s. Hypothesis: a gap that needs a run-up (moving left first), which v2/v3
cannot do. Last bounded attempt: v4 = K 64, H 60 with NOOP / LEFT kept (v1's action mix). If it fails, 8-1 is
recorded as needing a stronger search (beam search with backtracking over savestates).
v4 result (08:53 UTC): K 64, H 60 with NOOP / LEFT, 8-1: time-out at x 4,056 after 170 s. **8-1 is not cleared by
random-shooting lookahead in four variants** (x 3,750-4,056); it is recorded as needing a stronger search (beam
search with backtracking over savestates). Stage S0: met on 1-1 and 4-1, not on 8-1.

## 30. Speedrun stage S2 probe: can the connectome imitate a level-clearing teacher? (declared 2026-10-11 08:53 UTC, before any run; exploratory, G variant "G-planner-imitation-v1")

*Time box: results by 14:30 UTC or the probe stops (consolidation and the 18:16 UTC deadline). One wiring (the
connectome), three readouts: a feasibility probe for the speedrun roadmap, not a wiring comparison.*
**Teacher.** The §29 v2 planner (K 32, H 40, RIGHT+B / RIGHT+A+B): from any emulator state, its first action
after one planning step is the label (RAM and savestates used by the teacher only; pixels into the brain).
**Data (development levels 1-1 and 4-1, which the planner clears).** Round 0: planner trajectories from the
level start and from 12 random run/NOOP-prefix starts per level (seeds 20261900+, never the harness-v2
evaluation starts), every decision a (previous frame, frame) pair encoded as in the common stream, labelled with
the planner's action. Rounds 1-2 (DAgger, as §19): the frozen brains play from collection starts, the planner
labels every visited state, frames are aggregated.
**Learning (G).** The Stage D supervised rule (common_stream.py: eta 0.001, tau 2, centred eligibility,
RMS-normalised), target RIGHT+A+B when the planner jumps, RIGHT+B otherwise; 3,000 updates per round from the
aggregate; starting from untrained weights.
**Decision rule (interface, fixed now):** press RIGHT+A+B when the brain's jump probability (sum over the jump
buttons) exceeds its run probability (RIGHT+B + RIGHT), else RIGHT+B — a greedy two-class rule, since the
teacher jumps on ~45% of decisions.
**Evaluation (frozen).** From each level's start state and from the 16 harness-v2 training-suite starts of 1-1
and 4-1: flag or not, furthest x; references: the planner (must clear), always-run and the take-off-label
teacher (x 694 on 1-1, 2,112 on 4-1 at best).
**Reading.** S2 "met" on a level if a frozen brain clears it from the level start; "progress" if brains go beyond
the label teacher's furthest x from the start; otherwise reported as not met.

### 30.1 Result (2026-10-11 09:51 UTC, exploratory): stage S2 not met — the connectome does not imitate the level-clearing teacher

`mission/s2/s2_probe.json`, `mission/s2_probe.log`. Round 0: the planner cleared 24 of 26 starts (12 / 13 on each level),
9,906 labelled frames (44% jumps); two DAgger rounds added 800 planner-labelled own states per brain (aggregate
11,506). Frozen evaluation, greedy two-class rule, three readouts: **0 flags in 17 starts per level for every
brain**; from the level start, 1-1: x 670 / 312 / 595, 4-1: x 658 / 415 / 406; mean furthest x over the 17 starts
407-670 (1-1), 389-721 (4-1). References: the planner clears both; the take-off-label teacher reaches 694 (1-1) and
2,112 (4-1) at best; always-run 312 / 323. **Reading (as fixed in §30):** S2 not met and no progress beyond the
label teacher: a teacher that clears the level is not enough; the per-decision, memoryless brain does not reproduce
a policy made of held jumps and timed runs from the planner's labels at this budget (3 × 3,000 updates). Next steps
on the roadmap remain S1 (memory across decisions) and an interface that can hold, before more imitation.
