# Perception or credit assignment? — study declaration (DRAFT, launch blocked)

Follows `docs/GENERALISATION_STUDY.md`. Owner's go: 2026-10-03, after asking
whether game RAM should be used, and how, for the study to stay sound.
**Status: criterion E validated 2026-10-03 23:20 (§5); study run `mario_oracle`
launched right after, in the background (§4b).**

## 1. Question

The reinforcement-learning fly (actor-critic, three-factor rule on the
motor-incoming synapses, fixed random readout) ended with a habit: under greedy
play 81 of 90 episodes ended within 20 px of a vision-free always-run script
(GENERALISATION_STUDY §8). Two explanations remain:

- **(a) perception**: the visual code the rule can use does not separate the
  moments to jump well enough;
- **(b) credit assignment**: the information is there (a supervised readout of
  the frozen motor code, fitted on 1-1, transfers to unseen levels, §7 of the
  generalisation study) but a scalar TD error does not route credit to it.

The experiment: same brain, same eye, same rule, same budget, but an input that
makes perception trivial. If the rule then learns to jump at enemies, (a); if it
still learns a habit, (b).

## 2. Place in the protocol

- PROTOCOL §1.2: the study is about what connectome wiring constrains; Mario is
  the instrument (§13.3). This study does not test wiring; it locates a
  bottleneck in the learning instrument before the wiring comparisons (§8 null
  models) are run with it.
- PROTOCOL §10.4: RAM may enter the policy input only when declared. Here it
  does, in one arm, **declared as a control condition**, never the primary
  condition, never reported as "the fly plays".
- PROTOCOL §10.3: the schematic view is an alternative sensory mapping, i.e. a
  sensitivity analysis of the interface.

## 3. The instrument: a schematic view drawn from RAM

`src/flymario/env/schematic.py`. The frame keeps the emulator's shape (224 x 240)
and goes through the **same eye** (`RetinotopicVision`, ON/OFF lamina of the
right eye): only what is drawn changes. White = empty; grey = every non-empty
metatile of the level buffer; black = every active enemy on screen; dark =
Mario. No palette, no scenery, no day/night.

Checked before any use (2026-10-03): the schematic overlaid on real frames of
1-1, 4-1 (Lakitu), 5-1 (Koopa, Goombas), 3-1 (Paratroopa, night) and 6-1 (night)
matches tiles, enemies and Mario after correcting an 8 px crop at the left and
top. A Piranha plant hidden in its pipe is drawn (privileged information,
accepted for an oracle).

Found while checking: `MarioEnv.enemy_ahead` reads the enemy's screen x from
0x3AE+i, which is wrong for enemies off screen (a Lakitu 277 px to the right
read as x 28), and it counts enemies off screen vertically. Its *distance* (from
level x) is right, so the generalisation study's occlusion boxes (enemies 24-72
px ahead, on screen) were right; its "enemy ahead" labels include off-screen
Lakitus (GENERALISATION_STUDY §11). The new `danger_ahead` counts on-screen
enemies only.

## 4. Design (fixed now; launch only after §5)

One run, `mario_oracle`, v2 rule exactly as `mario_gen` (actor-critic, TD(lambda)
.98/.9, floor .1, death -100 in the learning signal, eta .001), six learners, per
fixed readout `block_seeds(3)` i, noise seed 7400 + i (fresh, not mario_gen's):

| arm | eye sees | levels | start | palette |
|---|---|---|---|---|
| `oracle_ro1-3` | the schematic (RAM) | 1-1, 2-1, 4-1, 7-1 | 0-22 NOOP decisions | n/a |
| `pixels_ro1-3` | the game (as mario_gen generalists) | same | same | jitter as mario_gen |

Budget 20,000 decisions per learner (matches mario_gen's 20,505), stopped with a
checkpoint. The pixel arm is a fresh replication of the habit finding.

Evaluation, frozen, each learner in its own view, plus the **untrained brain
with the same readout and seeds as baseline** (§5 shows why it is required) and
the vision-free always-run script: unseen 5-1, 8-1, 3-1, 6-1, starts after 0,
3, ..., 21 NOOP decisions; greedy and sampled.

## 5. Primary criterion: three candidates tested, none validated yet

Project rule: a criterion is used only after it passes a
known positive and fails known negatives.

| candidate | positive reference | negatives | verdict |
|---|---|---|---|
| A. furthest x beats random jumping at the learner's own rate (paired Wilcoxon) | scripted RAM jumper (jumps when an on-screen enemy is 8-64 px ahead): **fails** (17/12, p 0.58) | random vs random fails; always-run fails | invalid: in SMB random jumping also clears pipes and pits, so the test measures overall competence, not timing (`oracle_criterion_check-20261003.json`) |
| B. occlusion, enemy vs two textured scenery boxes, Wilcoxon p < .05 and ratio >= 2 | supervised danger readout: **fails** (ratio 1.52, p .022) | mario_gen learners on their own frames: 6/6 fail (p alone would pass one: generalist_ro1 p .020) | invalid as set (`occlusion_criterion_check-20261003.json`) |
| C. occlusion specificity S = (d_enemy - d_scen) / d_swap >= 0.12 and p < .05 (threshold fixed from B's numbers, then tested on fresh frames, seed 20261008, readout refitted on other 1-1 frames) | supervised readout: **fails** (S 0.105) | **untrained brains pass** (S 0.19-0.24); mario_gen learners pass 4/6 | invalid (`occlusion_criterion_check-fresh-20261003.json`) |

**What C teaches.** Erasing an enemy changes the decision of an *untrained* brain
about twice as much as erasing an equally textured scenery box: the enemy is
simply a salient stimulus for this eye at that position (dark, compact, near
Mario, often moving). Occlusion specificity measured on one brain therefore says
nothing about learning. On the same fresh frames, every trained learner's S is
*below* its untrained twin's (ro1: 0.22 untrained vs 0.08 specialist, 0.14
generalist; ro2: 0.24 vs 0.16, 0.13; ro3: 0.19 vs 0.15, 0.05).

**Next candidate (D), to validate before launch:** a *learning-relative*
measure. For each learner and its untrained twin (same readout, same frames,
same noise): Delta = [P(jump | danger) - P(jump | clear)]_trained -
[same]_untrained, on fresh frames of unseen levels (danger: on-screen enemy
8-64 px ahead; clear: none within 160 px), with occlusion of the enemy as a
secondary, also twin-relative, measure. Positive reference needed: a learner
known to have learnt danger-triggered jumping *through the fixed readout* -- e.g.
a supervised three-factor learner on the RAM danger label (the Gate G
supervised mode) -- which must pass; mario_gen learners and the untrained twins
against themselves (different noise) must fail.

### Criterion D, fixed before its references exist (2026-10-03, evening)

`scripts/gates/criterion_d_check.py`. Frames from scripted play on 5-1 and 8-1
(seed 20261009), DANGER = on-screen enemy 8-64 px ahead, CLEAR = none within
160 px, up to 80 of each per level, each encoded after its true previous frame.
P(jump) = softmax(scores / 2) over RIGHT+A, RIGHT+A+B, A, mean of 2 noise seeds
shared by every brain. For a learner, dp = P(jump)_trained - P(jump)_untrained
twin (same readout, same noise); **D = mean dp(DANGER) - mean dp(CLEAR)**.
**PASS: one-sided Mann-Whitney p < 0.05 and D >= 0.10.**

Validation: valid if >= 2 of 3 positive references pass, 0 of 6 `mario_gen`
learners pass, and 0 of 3 noise twins (untrained brain, other noise seeds,
scored as if trained) pass. If the positives fail, criterion D is not validated
and that failure is itself reported (the supervised rule did not learn
danger-triggered jumping in its budget).

**Positive reference run, `mario_teacher`** (`scripts/train_mario_sweep.py
--version 5 --max-decisions 10000`): three learners `teacher_ro1-3`, readouts
`block_seeds(3)`, noise 7500 + i, sample seed 20261007, generalist conditions
(1-1, 2-1, 4-1, 7-1; start after 0-22 NOOP decisions; palette jitter), eye on
the game. Rule: the Gate G supervised form, every decision, G = e_centred *
L(onehot(target) - pi) through the fixed readout, RMS-normalised, eta 0.001; the
target comes from RAM: RIGHT+A+B if an on-screen enemy is 8-64 px ahead, else
RIGHT+B. Actions are the learner's own (sampled, floor 0.1). 10,000 decisions per
learner. Smoke run before declaration (400 decisions, discarded): P(jump) 0.24
in danger vs 0.13 clear on training levels, so the mode learns in the right
direction; nothing about unseen levels was looked at.

This run is a *reference for the criterion*, not Stage B: its result on unseen
levels is not read as "the teacher works" until Stage B is declared.

### Criterion D, first validation (21:59): NOT validated, as declared

`criterion_d_check-20261003.json`, 285 fresh frames (160 danger, 125 clear):

| brain | D | p | pass |
|---|---|---|---|
| teacher_ro1 / ro2 / ro3 (positive) | 0.106 / 0.325 / 0.201 | < 1e-6 | 3/3 |
| noise twins ro1-3 (negative) | -0.001 / 0.007 / 0.000 | 0.75 / 0.07 / 0.45 | 0/3 |
| mario_gen specialist ro1 / ro2 / ro3 | 0.044 / **0.110** / -0.022 | 2e-6 / < 1e-6 / 0.95 | 1/3 |
| mario_gen generalist ro1 / ro2 / ro3 | 0.026 / 0.034 / 0.004 | 7e-5 / < 1e-6 / 0.001 | 0/3 |

One "negative" passed (specialist_ro2), so by the rule the criterion is not
validated. The fault is in the choice of negatives, not in the measure: the
mario_gen learners were assumed to be pure habit, which was the hypothesis under
test, not a known fact. The data contradict it in a graded way: 5 of 6 show a
small but significant learnt rise of jumping on danger frames of unseen levels
(D 0.004-0.11), and specialist_ro2 -- the learner that passed the first Goomba at
10,000 decisions -- reaches teacher_ro1's level. These are measurements, to be
read only once a criterion is validated.

### Criterion D, revision 2 (declared 22:03, before its run)

Same measure, same frames, same thresholds. Negatives = brains *known* not to
have learnt danger-contingent jumping: the noise twins (already measured, 0/3)
and a new run **`mario_teacher_shuffled`** (`--version 6 --max-decisions
10000`): exactly `mario_teacher` (same readouts, noise 7500 + i, sample seed,
levels, rule, budget), but the teacher's target is a coin -- jump with p 0.25 --
independent of the frame (the counterpart of Gate G's shuffled twin). Valid if
>= 2 of 3 teachers pass (known: 3/3) and 0 of 6 negatives pass. mario_gen
learners are measurements, not references.

### Criterion D, revision 2 result (22:27): NOT validated, as declared

`criterion_d_check-v2-20261003.json`. The shuffled-target teachers, which never
saw any link between danger and the target, also show D > 0: tshuffled_ro1 /
ro2 / ro3 = 0.027 / **0.118** / 0.030 (p < 0.001), and ro2 passes the threshold.
Negatives passing: 1 of 6. **Criterion D is invalid**: any learning that raises
the propensity to jump raises it more on danger frames than on clear ones, by
an amount that depends on the readout (largest for ro2). The same holds in
log-odds (shuffled teachers 0.15-0.55), so a change of scale does not remove it.

**What this changes for the measurements already made** (exploratory, per-frame
data in `criterion_d_frames-20261003.npz`): against the shuffled-target teacher
with the same readout, on the same frames --

| learner | D minus matched shuffled teacher | p |
|---|---|---|
| teacher_ro1 / ro2 / ro3 | +0.079 / +0.207 / +0.171 | < 1e-6 each |
| specialist_ro1 / ro2 / ro3 (mario_gen) | +0.016 / -0.008 / -0.052 | < 1e-6 / 1.0 / 1.0 |
| generalist_ro1 / ro2 / ro3 (mario_gen) | -0.002 / -0.085 / -0.025 | 0.45 / 1.0 / 1.0 |

- The RAM-taught learners jump on danger frames of **unseen** levels more than
  non-contingent learning explains, on all three readouts: the local rule,
  through the fixed readout, can learn a visual danger-to-jump mapping that
  transfers (exploratory: the criterion is not yet validated).
- The reinforcement learners do not exceed non-contingent learning (one +0.016):
  specialist_ro2's D of 0.110 is what ro2 gives to non-contingent learning
  (0.118). The habit reading of the generalisation study now holds against a
  proper reference (the shuffled teacher is not their exact twin: different
  training, same readout).

**Consequence for the design (§4), to amend before launch:** every learner needs
a matched non-contingent twin -- same readout, noise, seeds and training, only
the contingency removed (shuffled reward for the RL arms, as in pilot v1's
trios) -- and the criterion must compare the learner with that twin.

**Candidate E, proposed (not yet declared):** per readout, D(learner) -
D(matched shuffled twin) >= 0.05, one-sided Mann-Whitney on per-frame
differences p < 0.05. Its positive reference passes on existing data (teachers
vs shuffled teachers, +0.08 to +0.21), but the margin was chosen after seeing
those numbers, so validation needs a fresh negative pair: a second
shuffled-target teacher run (other coin and sample seeds) compared with the
first must fail on all three readouts.

### Criterion E, declared 2026-10-03 22:39, before its validation run

For a learner L and its matched non-contingent twin T (same readout, noise
seeds, random draws, training; only the contingency removed), on the criterion
frames (5-1 and 8-1, seed 20261009, as D), per frame q = P(jump)_L - P(jump)_T
(the untrained brain cancels). **E = mean q(DANGER) - mean q(CLEAR). PASS:
one-sided Mann-Whitney (q DANGER > q CLEAR) p < 0.05 and E >= 0.05.**

Validation run **`mario_ref2`** (`--version 7 --max-decisions 10000`), fresh: per
readout a trio in one group (shared noise 7500 + i and draws, sample seed
20261008): `teacher2` (RAM teacher), `tshufA` and `tshufB` (coin targets, p 0.25,
different coins). Positive pairs: teacher2 vs tshufA (3) -- valid only if >= 2
pass. Negative pairs: tshufB vs tshufA and tshufA vs tshufB (6) -- valid only if
0 pass. The 0.05 margin was chosen after seeing the first teacher-vs-shuffled
numbers (+0.08 to +0.21), which is why every reference here is new.

### Criterion E validation result (23:20): VALIDATED

`criterion_e_validate-20261003.json`; `mario_ref2` trained 10,000 decisions
(9 brains, ~4.4 decisions/s); 285 frames (160 danger, 125 clear).

| pair | ro1 | ro2 | ro3 |
|---|---|---|---|
| positive: teacher2 vs tshufA | E 0.110, p < 1e-6, PASS | E 0.192, p < 1e-6, PASS | E 0.013, p 0.98, fail |
| negative: tshufB vs tshufA | -0.012, p 0.996 | 0.027, p < 1e-6 | -0.010, p 0.999 |
| negative: tshufA vs tshufB | 0.012, p 0.004 | -0.027, p 1.0 | 0.010, p 0.001 |

Positives 2 of 3, negatives 0 of 6: **criterion E is valid**. Two remarks: twin
vs twin differences reach p < 0.01 in one direction on every readout, so p alone
would not discriminate -- the 0.05 margin does (largest negative 0.027); and the
fresh teacher on ro3 did not learn enough in 10,000 decisions (the first
teacher_ro3 did, +0.171), so one readout in three can fail even when learning
works -- hence the ">= 2 of 3" rule.

## 4b. Design as amended before launch (2026-10-03, 22:45)

Replaces §4's arms (the lesson of criterion D: every learner needs a matched
non-contingent twin). Run **`mario_oracle`** (`scripts/train_mario_sweep.py
--version 9 --max-decisions 20000`), 12 learners. Per readout `block_seeds(3)` i,
noise seed 7400 + i, two pairs, each pair one group (shared noise and draws):

| learner | twin (reward sign shuffled per decision) | eye sees | episodes |
|---|---|---|---|
| `oracle_ro{i}` | `oracleshuf_ro{i}` | the RAM schematic | 1-1, 2-1, 4-1, 7-1, start after 0-22 NOOP decisions |
| `pixels_ro{i}` | `pixelsshuf_ro{i}` | the game, palette jitter | same |

Rule v2 as mario_gen (actor-critic, TD(lambda) .98/.9, floor .1, death -100 in
the learning signal, eta .001), sample seed 20261010, 20,000 decisions per
learner, checkpoint every 1,000. **Launched only if criterion E is validated on
`mario_ref2`.** Smoke check before declaration (60 decisions, discarded): the
oracle eye receives the schematic (values 0/60/128/255), the pixel eye the
jittered game; both learn; old checkpoints still load.

**Primary measure:** criterion E for each learner against its twin
(`scripts/gates/criterion_e_check.py study`), each on the criterion frames in its
own view. An arm "learnt danger-triggered jumping" if >= 2 of 3 readouts pass.
**Secondary (descriptive):** frozen greedy and sampled play on 5-1, 8-1, 3-1, 6-1
against the twin and the always-run script.

## 6. Reading the result (declared now)

| oracle arm | pixel arm | reading |
|---|---|---|
| learns (criterion D) | habit | bottleneck = perception: improve the sensory interface (augmentations, temporal filtering, LC/LPLC pathways) |
| habit | habit | bottleneck = credit assignment: Stage B (teacher) |
| learns | learns | the earlier habit was budget or seed variance: extend |
| habit | learns | an interface artefact of the schematic: report, inspect |

## 7. Stage B (only if the reading is "credit assignment"), outline

A privileged teacher during training only (Chen et al. 2019, CoRL): the RAM
danger label sets the target action (jump when an on-screen enemy is 8-64 px
ahead, run otherwise) and the local rule learns it supervised (L = onehot(target)
- pi through the fixed readout), pixels only at test; or an asymmetric critic
that reads the RAM (Pinto et al. 2018, RSS). Details frozen before launch,
applied identically to the connectome and to the null graphs when those are run.

## 8. Not claimed

The oracle arm is not "the fly playing": it is the same network fed a drawing
made from game memory. No claim about fly vision is drawn from it.

---

## 9. Results (2026-10-04)

**Run.** `mario_oracle` launched 2026-10-03 23:21; the PC went down at 00:54,
two minutes after the 18,000-decision checkpoint; resumed 2026-10-04 14:38 from
that checkpoint (the trainer's resume restores weights, critic, generators and
emulator states) and finished 20,000 decisions per learner at 14:49. The
evaluation chain first waited forever: its "wait for training" pattern matched
the shell that launched it; fixed (anchored on the python process) and rerun.

**Primary: criterion E** (`criterion_e_study-20261003.json`; 285 frames of 5-1
and 8-1, 160 danger / 125 clear; each learner against its shuffled-reward twin,
in its own view):

| arm | ro1 | ro2 | ro3 | arm verdict (>= 2 of 3) |
|---|---|---|---|---|
| oracle (schematic eye) | E 0.006, p 0.006 | 0.011, p 3e-4 | 0.008, p 4e-5 | **0 / 3: did not learn** |
| pixels (game eye) | -0.042, p 1.0 | **0.066**, p < 1e-6, PASS | 0.007, p 0.001 | **1 / 3: did not learn** |

Reading table §6: "habit / habit". Neither reinforcement-learning arm learnt
danger-triggered jumping beyond its matched twin; the one pixel learner that
passes (ro2) does so modestly (0.066, against 0.11-0.19 for the RAM-taught
learners).

**Instrument check, post hoc** (`oracle_premise_check-20261004.json`; should
have been run before the launch, and was not): is "danger" more decodable from
the untrained brain's motor code with the schematic than with the game? Linear
decoder trained on one level, tested on the other:

| decoded from | game | schematic |
|---|---|---|
| motor code (AUC, mean of both directions) | **0.920** | 0.872 |
| lamina input | 0.829 | 0.791 |

**The oracle premise fails**: the schematic is cleaner to a human eye but not to
this eye and this brain; it carries the danger about as well as the game, not
better. So the oracle arm cannot, by itself, separate perception from credit
assignment -- the declared reading of the "habit / habit" row rested on a
premise that was not tested beforehand.

**What does separate them** is a contrast already measured with the same eye
(pixels), the same rule, the same fixed readout and the same training levels,
where only the learning signal differs:

| learning signal | learnt danger-triggered jumping on unseen levels? | source |
|---|---|---|
| RAM teacher (supervised label), 10,000 decisions | yes: E 0.110 / 0.192 / 0.013 vs matched shuffled twins (2 of 3, fresh validation run) | `criterion_e_validate-20261003.json` |
| RAM teacher, first run, 10,000 decisions | yes on 3 of 3 vs shuffled teachers (+0.079 / +0.207 / +0.171, exploratory) | `criterion_d_check-v2-20261003.json` |
| game reward (actor-critic), 20,000 decisions | no: 1 of 3 (0.066) | this run |
| game reward, `mario_gen`, 20,505 decisions | no: <= 0.016 vs shuffled teachers (exploratory) | `criterion_d_check-v2-20261003.json` |

And the information is there before any learning: danger is linearly decodable
from the untrained motor code on a level never used for fitting (AUC 0.92).

## 10. Conclusion

1. **The bottleneck is credit assignment, not perception.** The untrained
   connectome's motor code already carries an "enemy just ahead" signal that
   transfers across levels (AUC 0.92). The local three-factor rule, through the
   fixed random readout, can turn that signal into danger-triggered jumps that
   work on levels never seen -- when the learning signal says which moments
   matter (RAM teacher). Driven by the game's reward through a TD error, the same
   rule, with the same eye and twice the budget, learns no such contingency in
   either view.
2. **The oracle arm did not do the job it was designed for** (its premise failed,
   found after the run). It still adds one fact: a palette-free, scenery-free view
   does not rescue reinforcement learning either.
3. **Method.** Four criteria failed their references before one held (A, B, C, D);
   each failure removed a confound: random jumping is a strong policy in SMB;
   the enemy is salient to an untrained eye; any rise in jumping inflates
   danger-conditional measures; only a matched non-contingent twin isolates
   learnt contingency. Validating criteria first prevented at least two false
   positives (the RL learners would have "passed" D).

**Next (Stage B, §7), to declare before running:** train with the RAM teacher as
the learning signal (pixels only at test), with matched shuffled-teacher twins,
on the four training levels; test with criterion E and frozen play on unseen
levels; then the same comparison on the null graphs (PROTOCOL §8), which is where
the study's actual question -- what the wiring contributes -- is asked.

## 11. Exploratory follow-up, declared 2026-10-04 15:04 before running: do the RAM-taught learners *play* better?

Criterion E shows that RAM-taught learners jump more on danger frames than their
twins. Whether that makes them go further is a separate question: a scripted
jumper that jumps exactly at enemies does *not* beat random jumping (§5,
candidate A), because pits and pipes stop Mario too. Frozen greedy play of the
`mario_ref2` brains (teacher2, tshufA, tshufB per readout, plus untrained
brains), same evaluator and suites as the other studies (1-1, desynchronised
1-1, unseen day 5-1/8-1, unseen night 3-1/6-1; 3 episodes each;
`mario_generalisation_eval-ref2-greedy-20261004.json`). Descriptive, no
criterion. Expectation written now: teacher2_ro1/ro2 (which passed E) get past
the first enemy more often than their twins on unseen day levels, but their
furthest x need not differ much, because the obstacles after the first enemy
are mostly pipes and pits the teacher never labelled.

### §11 result (16:02) and the study's secondary play results

**RAM-taught learners, frozen greedy play** (`mario_generalisation_eval-ref2-greedy-20261004.json`;
learner minus its twin tshufA, mean furthest x, wins/losses over episodes):

| readout (E) | 1-1 and desynchronised 1-1 | unseen day 5-1, 8-1 | unseen night |
|---|---|---|---|
| ro1 (0.110) | +0 (all ties) | +0 | +0 |
| ro2 (0.192) | **+964 (6/0)**: x 827-1,557 vs 312 | +67 (2/0, rest ties; 8-1: 559 vs 227) | +0 |
| ro3 (0.013) | +4 (6/0) | +38 (2/4) | +142 (3/3) |
| twin B vs twin A | identical everywhere | identical | identical |

As expected in writing beforehand: a learnt contingency changes greedy play only
when it is strong enough to flip the preferred button (ro2; not ro1, whose E of
0.110 leaves "run" on top), and on new levels the gain is small because the
obstacles after the first enemy are pipes and pits, which the teacher never
labelled. The one learner that plays 1-1 far better is also the only brain in
three studies to clear the first Goombas reliably under greedy play.

**Study arms, frozen play (secondary, descriptive)**
(`mario_generalisation_eval-oracle-{greedy,sampled}-20261003.json`): under greedy
play both RL arms are the always-run script again (oracle 307 / 202 / 441,
pixels 308 / 200 / 442 on 1-1 / unseen day / unseen night). Pixel learners beat
their twins (+151 to +382 on ro2/ro3) only because those shuffled-reward twins
often stand still (x 40): what the reward taught is to run, not to avoid. Sampled
play shows no consistent pattern (learner-minus-twin from -309 to +224).

### Conclusion, completed

The RL learners learn the reward-contingent part of the task that is common to
all frames (run) and nothing frame-specific; a supervised signal from RAM makes
the same rule, eye and readout learn a frame-specific contingency that transfers
to unseen levels, and, when strong, plays the training level far better. For a
teacher to improve play on new levels it must label every jump-worthy situation
(enemies, walls and pipes, pits -- all readable from the RAM tile buffer), not
enemies only.

**Stage B, to declare next session:** a RAM teacher labelling enemies, walls and
pits; matched shuffled-teacher twins; 20,000 decisions; criterion E plus frozen
play against the twin on unseen levels; the always-run script as the null for
play; then the null graphs (PROTOCOL §8).

## 12. Stage B, declared 2026-10-04 16:05 before running

**Behavioural reference, checked first** (`oracle_criterion_check-full-20261004.json`):
the full RAM label `schematic.jump_label` -- jump if an on-screen enemy is 8-64 px
ahead, a non-empty metatile is at Mario's body height 4-28 px ahead of his front
(wall, pipe), or there is no metatile under his feet 8-40 px ahead (pit);
labels checked by eye on 1-1 and 5-1 -- drives a scripted expert (press-release:
A held at most 6 decisions) that **beats random jumping at its own rate** on 5-1,
8-1, 3-1, 6-1 (22 / 10, mean +104 px, p 0.023), while random vs random (p 0.74)
and always-run (p 0.999) fail. Candidate A, invalid with the enemy-only label,
has a valid positive reference with the full label.

**Run `mario_stageB`** (`scripts/train_mario_sweep.py --version 10
--max-decisions 20000`): per readout `block_seeds(3)` i, noise 7600 + i, one group:
`tfull_ro{i}` taught by the full RAM label (Gate G supervised form, eta 0.001)
and its twin `tfullshuf_ro{i}` taught by a coin (jump with p 0.35, close to the
label's rate: 0.33 in a 150-decision smoke run, discarded). Eye on the game,
generalist conditions (1-1, 2-1, 4-1, 7-1; start after 0-22 NOOP decisions;
palette jitter); sample seed 20261011; 20,000 decisions per learner.

**Primary:** criterion E (`criterion_e_check.py stageB`), learner vs twin;
"learnt" if >= 2 of 3 pass. **Secondary (descriptive):** frozen greedy play
(`mario_generalisation_eval.py`, standard suites) against the twin, the
always-run script and the untrained brain. Expectation written now: E passes
on >= 2 readouts (as the enemy-only teacher did); in play, the gain over the
twin is larger than the enemy-only teacher's on unseen day levels, because walls
and pits are now taught. A rate-matched closed-loop test of the learners
(candidate A applied to learners) needs a harness not yet written.

### Stage B results (2026-10-04, 17:06)

`mario_stageB` trained 20,000 decisions per learner (16:05-16:52, ~7
decisions/s).

**Primary, criterion E** (`criterion_e_stageB-20261003.json`): **3 of 3 pass** --
tfull vs its coin twin E = 0.275 / 0.222 / 0.199 (p < 1e-6 each), against 0.110 /
0.192 / 0.013 for the enemy-only teacher and <= 0.066 for every RL learner.

**Secondary, frozen greedy play** (`mario_generalisation_eval-stageB-greedy-20261004.json`;
mean furthest x; learner minus twin, wins/losses; the pooled Wilcoxon tests are
exploratory, the play comparison was declared descriptive):

| suite | taught (tfull) | twin (coin) | untrained | per readout ro1 / ro2 / ro3 | pooled W/L, p |
|---|---|---|---|---|---|
| 1-1 + desynchronised | 724 / 628 | 312 | 99 / 102 | +380 (6/0) / +343 (5/1) / +368 (5/1) | 16/2, 0.0004 |
| unseen day 5-1, 8-1 | **330** | 207 | 60 | +54 (2/0) / 0 / **+313 (5/0)** | 7/0, 0.008 |
| unseen night 3-1, 6-1 | 447 | 417 | 132 | +79 (4/2) / 0 / +11 (2/0) | 6/2, 0.043 |
| all unseen | | | | | **13/2, 0.0009** |

Against the expectation written at 16:05: E passes on >= 2 readouts -- yes, 3/3;
the gain over the twin on unseen day levels exceeds the enemy-only teacher's --
yes (mean over readouts +122 px vs +35). tfull_ro3 reaches x 933 on 5-1 and 546
on 8-1, levels it never saw, where its twin dies at 184-227; tfull_ro1 clears
the first obstacle of 3-1 at night (498-594 vs 372). ro2's learner learnt the
contingency (E 0.222) but its greedy choice on unseen levels stays "run".

**Reading.** With a learning signal that says which frames call for a jump --
enemies, walls, pits, read from RAM during training only -- the connectome-based
learner, seeing only pixels at test, learns a visual rule that makes it play
better on levels and palettes it never saw. This is the generalisation the owner
asked for (2026-10-03), obtained by fixing credit assignment, not perception.
The behavioural result is from one run (three readouts) and post hoc pooled
tests; a confirmation on fresh seeds, with the closed-loop rate-matched test
(candidate A, now validated) applied to learners, is the next step, then the
null graphs (PROTOCOL §8).

## 13. Stage B confirmation, declared 2026-10-04 17:11 before running

Run **`mario_stageB2`** (`--version 11 --max-decisions 20000`): Stage B exactly
(same learners, label, coin p 0.35, levels, rule, budget) with fresh noise seeds
7700 + i and sample seed 20261012. Resumable if the PC is switched off.

**Primary 1, criterion E** (`criterion_e_check.py stageB2`): confirmed if >= 2 of 3
tfull learners pass against their twins.

**Primary 2, play, candidate A applied to learners**
(`scripts/gates/closed_loop_rate_check.py`): each frozen learner plays greedily
on 5-1, 8-1, 3-1, 6-1, starts after 0, 3, ..., 21 NOOP decisions (32 episodes);
on each (level, start), a random jumper at that episode's own jump rate plays
too; PASS per learner: furthest x greater than the rate-matched random jumper,
one-sided Wilcoxon p < 0.05. Validated today on references (full RAM expert
passes, 22/10, p 0.023; random vs random and always-run fail). Within the run,
the coin-taught twins are the negatives: confirmed if >= 2 of 3 tfull learners
pass and at most 0 of 3 twins pass. The harness is applied first to the first
Stage B run as an exploratory check (results labelled as such).

### §13, exploratory application to the first Stage B run (17:44)

`closed_loop_rate_check-stageB-20261004.json` (32 paired episodes per brain on
5-1, 8-1, 3-1, 6-1; random jumper at the brain's own greedy jump rate):

| brain | W / L | mean diff | p | jump rate | pass |
|---|---|---|---|---|---|
| tfull_ro1 | 12 / 8 | +19 | 0.084 | 0.49 | no (trend) |
| tfull_ro2 | 1 / 0 | +6 | 0.5 | 0.003 | no: almost never jumps greedily |
| **tfull_ro3** | **10 / 4** | **+54** | **0.032** | 0.11 | **yes** |
| coin twins ro1-3 | <= 1 / <= 6 | ~0 | >= 0.5 | ~0 | 0 / 3 |
| untrained brains | 0-1 / 31-32 | -233 to -261 | 1.0 | | 0 / 3 |

**This qualifies the Stage B play result.** The "13 / 2 against the twins" on
unseen levels mixes two things: jumping at all (the coin twins, greedy, never
jump) and jumping at the right time. Against random jumping at its own rate --
which isolates timing -- one learner of three (ro3) shows better-timed jumps on
unseen levels, ro1 a trend, ro2 none. By the rule declared for the confirmation
(>= 2 of 3), the first run would not pass this test. What stands: the taught
contingency (criterion E, 3/3) and, for one learner, timing-specific play gains
on unseen levels. The confirmation run (`mario_stageB2`, resumable; evaluation
chained by `scripts/gates/stageb2_pipeline.sh`) decides.

### §13 confirmation result (2026-10-04, 18:41)

`mario_stageB2`, 20,000 decisions (finished 18:17).

**Primary 1, criterion E** (`criterion_e_stageB2-20261003.json`): tfull vs twin
E = 0.302 (pass) / 0.0498 (fail, just under the margin) / 0.339 (pass):
**2 of 3, confirmed.** The RAM-taught contingency is learnt and transfers to
unseen levels on fresh seeds.

**Primary 2, rate-matched play** (`closed_loop_rate_check-stageB2-20261004.json`):

| brain | W / L | mean diff | p | jump rate |
|---|---|---|---|---|
| tfull_ro1 | 5 / 10 | -62 | 0.98 | 0.30 |
| tfull_ro2 | 0 / 0 | 0 | 1.0 | 0.00 |
| tfull_ro3 | 7 / 6 | +27 | 0.32 | 0.15 |
| twins, untrained | | | | 0 / 6 pass |

**0 of 3, not confirmed.** The exploratory pass of tfull_ro3 in the first run
(10/4, p 0.032) did not replicate.

**Reading.** On fresh seeds the taught fly reliably *wants* to jump more when an
enemy, a wall or a pit is ahead on levels it never saw (E 0.30-0.34), but under
greedy play its jumps are not better placed than random jumps at the same rate.
A likely mechanism, to test, not established: the label says "jump" on every
frame of a danger window and the sweep regime has no memory across decisions,
so the learnt policy can only *hold* A through the window; in SMB a held A jumps
once, at whichever frame the window opens, and cannot re-jump, whereas the
scripted reference released A after 6 decisions and a random jumper toggles A
by construction. Candidates for the next step: a label that asks for a jump only
on the frame where the jump should start (the right distance), or a label that
also teaches the release.

**Overall, for the report:** fixing credit assignment with a privileged
teacher makes the connectome-based learner acquire a visual contingency that
generalises to new layouts (confirmed); turning it into better play on those
layouts is not yet achieved (not confirmed; one exploratory pass, not
replicated).

### Mechanism test, declared 2026-10-04 19:00 before running (exploratory, no brain)

Hypothesis (§13): a memoryless policy taught "jump" on every frame of a danger
window can only hold A, so it jumps once when the window opens. Test with the
validated harness (`oracle_criterion_check.py --hold`): the same full RAM label,
but A held for as long as the label says jump (no release), against random
jumping at its own rate, same levels and starts as the press-release expert
(which passed, 22/10, p 0.023). Supports the hypothesis if the holding expert
does **not** pass (p >= 0.05) while random vs random and always-run still fail;
refutes it if the holding expert passes.

**Result (19:05, `oracle_criterion_check-hold-20261004.json`): the hypothesis is
supported.** Same full RAM label, same levels, starts and rate-matched random
jumpers:

| scripted expert | vs random jumping at its own rate |
|---|---|
| releases A after 6 decisions (§12) | 22 / 10, +104 px, p 0.023 -- passes |
| **holds A through the danger window** | **11 / 20, -177 px, p 0.90 -- fails** |
| random vs random; always-run | 18 / 14, p 0.65; 6 / 22, p 1.0 -- fail |

A policy that knows exactly when danger is ahead plays *worse* than random
jumping when it can only hold the button: the label is right, the execution is
wrong. This is what the taught flies can do -- no memory across decisions, one
action per frame -- and matches their failure on the rate-matched test. It
points to the fix: teach the take-off (a label on the frame where a jump should
start, run otherwise), so that a memoryless policy presses and releases by
itself as the danger window moves. Not yet tested with learners.

### Take-off label test, declared 2026-10-04 19:03 before running (exploratory, no brain)

`schematic.takeoff_label`: jump only at take-off distance (enemy 16-40 px ahead,
wall 4-12 px ahead of Mario's front, pit edge 8-16 px ahead), run otherwise. A
*memoryless* scripted policy following it (`oracle_criterion_check.py
--takeoff`) against random jumping at its own rate, same levels and starts. If it
passes (p < 0.05, with random vs random and always-run failing), this label is
the candidate teacher for the next learner run; if not, the label needs work
before any training. Window sizes chosen by reasoning about SMB jump arcs, not
tuned on these levels.

**Take-off test result (19:04, `oracle_criterion_check-takeoff-20261004.json`):
inconclusive.** The memoryless take-off expert beats its rate-matched random
jumper (19 / 12, +128 px, p 0.0015), but random vs random *also* passed (22 / 7,
+104, p 0.031): with one random draw per start, two random jumpers differ by
~100 px by chance, so the declared rule (negatives must fail) is not met.

**Harness revision, declared 19:06 before rerunning:** the rate-matched baseline
is the mean of K = 5 random jumpers per start (`--k 5`), for the reference and
the candidate alike. Rerun both the press-release expert (the positive
reference; must still pass) and the take-off expert, in parallel. The earlier
single-draw results (full 22/10, hold 11/20) are kept as measured.

**K = 5 results (19:04; `oracle_criterion_check-{full,takeoff}-k5-20261004.json`):**

| scripted expert vs mean of 5 rate-matched random jumpers | W / L | diff | p | negatives (random vs random; always-run) |
|---|---|---|---|---|
| press-release, full label (positive reference) | 19 / 13 | +134 | 0.020, pass | p 0.105, p 1.0 -- fail |
| **take-off label, memoryless** | **22 / 10** | **+72** | **0.030, pass** | p 0.18, p 1.0 -- fail |

The harness with K = 5 passes its positive and fails its negatives (random vs
random at p 0.105 is closer to the threshold than one would like: more starts
would help). **A memoryless policy following the take-off label jumps better
than chance on unseen levels** -- unlike one holding A through the danger window
(11/20, -177, single-draw harness). The take-off label is the teacher for the
next learner run (Stage C, to declare): learners taught `takeoff_label`, coin
twins at its rate, evaluated with criterion E and this K = 5 rate-matched test.

## 14. Stage C, declared 2026-10-04 19:25 before running

**Run `mario_stageC`** (`scripts/train_mario_sweep.py --version 12 --max-decisions
20000`): per readout `block_seeds(3)` i, noise 7800 + i, one group: `tko_ro{i}`
taught by `schematic.takeoff_label` (jump only at take-off distance: enemy
16-40 px, wall 4-12 px, pit edge 8-16 px; Gate G supervised form, eta 0.001) and
its twin `tkoshuf_ro{i}` taught by a coin with p 0.45 (the label's rate in a
300-decision smoke run, 0.463, discarded). Eye on the game, generalist conditions,
sample seed 20261013, 20,000 decisions per learner. Evaluation chained by
`scripts/gates/stagec_pipeline.sh` (headless).

**Primary 1, criterion E** (`criterion_e_check.py stageC`): >= 2 of 3 tko pass.
**Primary 2, rate-matched play** (`closed_loop_rate_check.py --k 5`: greedy
learner vs the mean of 5 random jumpers at its own rate, 4 unseen levels x 8
starts; the K = 5 harness passed its scripted positive and failed its negatives
at 19:04): >= 2 of 3 tko pass and 0 of 3 twins pass.

Expectation written now: E passes (the contingency is taught as before); play
passes on at least 2 readouts if the take-off label fixes the held-button
problem as it did for the scripted policy. A learner whose greedy policy never
jumps (as stage B's ro2) fails play by ties -- a known risk.

### Stage C results (2026-10-04, 20:39)

`mario_stageC`, 20,000 decisions (19:25-20:13).

**Primary 1, criterion E** (`criterion_e_stageC-20261003.json`): tko vs coin twin
E = 0.178 / 0.209 / 0.0507, p < 1e-6 each: **3 of 3, passed.**

**Primary 2, rate-matched play, K = 5** (`closed_loop_rate_check-stageC-20261004.json`):

| brain | W / L | mean diff | p | greedy jump rate | pass |
|---|---|---|---|---|---|
| tko_ro1 | 4 / 14 | -36 | 0.99 | 0.16 | no |
| **tko_ro2** | **7 / 2** | **+120** | **0.014** | 0.08 | **yes** |
| tko_ro3 | 3 / 0 | +0.5 | 0.13 | 0.002 | no (hardly jumps) |
| twins ro1 / ro2 / ro3 | 14/17, 9/12, 6/10 | | 0.45, 0.22, 0.89 | 0.22, 0.03, 0.10 | 0 / 3 |

**1 of 3 tko pass: not met** (rule: >= 2 of 3; twins 0/3 as required).

**Reading.** The take-off label is taught as well as the earlier labels (E 3/3),
but it did not turn into better-timed greedy play on unseen levels for most
learners -- the same 1-of-3 pattern as Stage B's first run (not replicated in
its confirmation). The scripted test showed the label itself works for a
memoryless policy; what differs is that a scripted policy *is* the label,
whereas the learners only raise their jump probability by 0.05-0.21 on danger
frames. Under greedy play that rise flips the preferred button rarely (ro3,
jump rate 0.002) or at the wrong frames (ro1, worse than random). The remaining
gap is between a learnt contingency and a policy precise enough to time jumps:
more training, a stronger update on the label, or a readout able to express a
sharp decision are the candidates; none tested yet.

**State of the Mario studies after four stages.** Robust across five runs: the
game reward alone teaches a habit; a RAM teacher (enemy-only, full, take-off)
makes the connectome-based learner acquire a frame-specific contingency that
transfers to unseen levels (criterion E passes in every teacher run). Not yet
achieved: turning that contingency into play reliably better than
rate-matched random jumping (1/3, 0/3, 1/3 across three teacher runs).

## 15. Stage C extension, declared 2026-10-05 17:05 before resuming

Question: is the gap between the taught contingency (E 3/3) and better-timed play
(1/3) a matter of training budget? `mario_stageC` resumes from its 20,000-decision
checkpoint, same learners and settings, up to 100,000 decisions
(`--version 12 --max-decisions 100000`). At 40k, 60k, 80k and 100k decisions the
checkpoint is copied and evaluated while training continues
(`scripts/gates/stagec_ext_pipeline.sh`): criterion E and the rate-matched play
test with 5 random jumpers, as in §14.

**Stopping and reading rules (declared now).** Several looks inflate chance
passes, so a milestone counts as "play achieved" only if the play criterion
(>= 2 of 3 tko pass, 0 of 3 twins pass) holds at **two consecutive milestones**;
training stops then. It also stops if the number of passing tko learners and
their mean difference both fail to improve over two consecutive milestones
(budget is not the answer), or when the owner stops it; in that case the last
evaluated milestone is reported as is. The 20k result (§14) is the baseline.

### Exploratory diagnostic at 40k (2026-10-05 18:05; no training, not a criterion)

`scripts/diagnostics/stagec_extrapolate.py` on the 40k copy
(`stagec_extrapolate-mario_stageC_40k.json`): the learnt weight change scaled
by k (clipped to the rule's bounds), greedy choice on the criterion frames
(5-1, 8-1; game view). Share of frames where the greedy button is a jump,
danger / clear:

| learner | k = 0 (untrained) | **k = 1 (as learnt)** | k = 2 | k = 4 | k = 8 |
|---|---|---|---|---|---|
| tko_ro1 | 0.39 / 0.36 | **0.78 / 0.25** | 0.98 / 0.30 | 0.98 / 0.28 | 0.99 / 0.66 |
| tko_ro2 | 0.91 / 1.00 | **0.89 / 0.36** | 1.00 / 0.41 | 0.22 / 0.00 | 0.74 / 0.06 |
| tko_ro3 | 0.04 / 0.03 | **0.00 / 0.00** | 0.14 / 0.00 | 0.29 / 0.02 | 0.58 / 0.03 |

At 40k two learners already make a *selective* greedy decision on frames of
unseen levels (jump far more often when an enemy is ahead than when clear); the
third moves the right way only with a much larger change. This points to budget
and update strength as levers -- to be read against the closed-loop 40k
evaluation, which is the measure that counts. Frames are isolated (scripted
play), not the learners' own trajectories; the scaling is a thought experiment,
not a trained network.

### §15 milestone 40k (2026-10-05 18:40)

`criterion_e_stageC-mario_stageC_40k.json`, `closed_loop_rate_check-stageC-40k-20261004.json`:

| learner | E (20k → 40k) | play vs 5 random at own rate (40k) | greedy jump rate |
|---|---|---|---|
| tko_ro1 | 0.178 → 0.172 | **17 / 11, +120 px, p 0.019, pass** | 0.40 |
| tko_ro2 | 0.209 → 0.207 | 7 / 22, -18, p 0.99 | 0.28 |
| tko_ro3 | 0.051 → 0.025 (now fails) | never jumps | 0.00 |
| twins ro1-3 | | 0 / 3 pass | 0.27, 0.13, 0.00 |

Play criterion: 1 of 3 (as at 20k, but a different learner: ro1 now, ro2 then).
Stopping rule: passing count unchanged (1 -> 1) but the tko mean difference rose
(+28 -> +34 px), so "no improvement over two milestones" is not met; training
continues to 60k. Note: tko_ro2's greedy choice is selective on isolated frames
(diagnostic above) yet loses to random in closed loop -- its own trajectories and
timing differ from scripted frames.

## 16. The RAM "danger" label was polluted (found 2026-10-05, 19:00-19:20)

**How it was found.** Closed-loop traces of the 40k take-off learners
(`jump_timing_trace-mario_stageC_40k.json`) showed jumps around enemies but at
scattered distances (onset median 42 and 55 px, inter-quartile 20-80 px), 39-42 %
of onsets with no enemy within 64 px and no take-off label, and every death
against an enemy. A probe of whether *distance* is readable then gave chance --
and, unexpectedly, so did *presence* (fit on the four training levels, test on
5-1/8-1: motor code AUC 0.51, raw input 0.34), contradicting the earlier 0.92.

**What is wrong.** `danger_ahead` (and the enemy part of every teacher label and
of the criterion frames) counts any active on-screen enemy ahead, at any height
and of any type. Per level, under scripted play (`schematic` RAM types):

| level | frames labelled danger (8-64 px) | of which enemy clearly above Mario | dominant types |
|---|---|---|---|
| 1-1 | 0.20 | 0.00 | Goomba |
| 2-1 | 0.18 | 0.52 | Goomba, Koopa |
| 4-1 (training) | **0.70** | 0.64 | **Piranha Plant**, Lakitu, Spiny |
| 7-1 (training) | 0.22 | 0.23 | Bullet Bill, Paratroopa |
| 5-1 | 0.35 | 0.16 | Koopa, Goomba, Piranha Plant |
| 8-1 | **0.97** | 0.66 | **Piranha Plant** (1,449 of 1,461) |

Piranha Plants sit in pipes (mostly hidden) and Lakitu flies: the label largely
meant "a pipe ahead" or "Lakitu around", not "an enemy to jump over".

**Fix.** `schematic.threat_ahead`: on-screen enemies ahead whose position lies in
Mario's body band, excluding Piranha Plants and Lakitu; `takeoff_label_v2` uses
it. Ground-threat rates per level: 0.07-0.14 (8-1: 0.01).

**What it changes (exploratory, labels now clean).**
- *Presence of a ground threat* is **not** linearly readable across levels from
  the untrained motor code: fit 1-1 -> test 5-1 AUC 0.45 (raw input 0.41); fit
  four training levels -> test 5-1/8-1 AUC 0.56 (raw 0.51)
  (`distance_probe-threat-*.json`). With the old label, the same 1-1 -> 5-1 split
  gives 0.81 (raw 0.22): the earlier 0.92-0.955 "the information is there"
  reflected label correlates (pipes, level composition), not enemy recognition.
  **The statement "perception is not the bottleneck" (§10) is withdrawn as
  stated**; with a linear readout, ground-enemy information is not shown to be
  available at the motor layer in a level-invariant form.
- Criterion E's "danger" frames were mostly pipe frames on 8-1, and the full and
  take-off teachers also label walls: the taught contingency may be "jump at
  obstacles" more than "jump at enemies". Re-analysis with threat / obstacle /
  clear frames: `criterion_e_relabel-20261005.json` (below).

**Re-analysis with clean frames (19:35, `criterion_e_relabel-20261005.json`;
unseen 5-1/8-1; 168 ground-threat, 240 obstacle, 240 clear frames; learner minus
matched twin; one-sided Mann-Whitney vs clear):**

| learners | threat - clear | obstacle - clear |
|---|---|---|
| Stage B (full label) ro1/ro2/ro3 | +0.16 / +0.12 / +0.13 (p < 1e-4) | +0.21 / +0.20 / +0.16 (p < 1e-4) |
| Stage C 40k (take-off) ro1/ro2/ro3 | +0.17 / +0.17 / 0.00 | +0.25 / +0.27 / +0.04 |
| ref2 enemy-only teacher ro1/ro2/ro3 | +0.05 / +0.16 / +0.04 | +0.06 / +0.21 / +0.09 |
| RL, game reward (mario_oracle pixels) | -0.02 / +0.04 / +0.01 | -0.01 / +0.02 / +0.00 |

The taught learners jump more when *something* stands ahead -- a ground enemy or
an obstacle, obstacles more -- than on clear ground, on levels never seen; the
reward-trained ones do not. Together with the probe (ground threats not linearly
separable from other frames across levels), the supportable reading is: **the
teacher installs a transferable "object ahead -> jump" rule; nothing shows enemy
recognition as such.** The reward alone installs neither. Stage C's play gap is
also explained in part: its teacher's enemy part fired on hidden Piranha Plants
and Lakitus, i.e. "jump" with nothing visible to jump over.

### Stage C extension stopped after the 60k copy (19:37) -- a declared-rule deviation

The extension's stop rule (§15) was not met; it was stopped by me, on the new
fact of §16: its teacher's enemy part fires on hidden Piranha Plants and flying
Lakitus, so more budget would mostly train a polluted label. The 60k copy is
still evaluated (`stagec_ext_pipeline.sh`); the budget question is re-asked with
the clean label (§17).

## 17. Stage D, declared 2026-10-05 19:37 before running

**Run `mario_stageD`** (`--version 13 --max-decisions 20000`): Stage C exactly
(take-off teacher, Gate G supervised form, eta 0.001, generalist conditions,
pixels), with the label `schematic.takeoff_label_v2` -- enemy part restricted
to ground threats (`threat_ahead`), wall and pit parts unchanged -- and coin
twins at p 0.48 (the v2 label's rate in a 300-decision smoke run, discarded).
Per readout `block_seeds(3)` i: `tko2_ro{i}` and `tko2shuf_ro{i}`, noise 7900 + i,
sample seed 20261015. Evaluation chained by `scripts/gates/staged_pipeline.sh`.

**Primary: rate-matched play** (`closed_loop_rate_check.py --k 5`, unseen 5-1,
8-1, 3-1, 6-1, 8 starts): >= 2 of 3 tko2 pass and 0 of 3 twins pass.
**Secondary: what is learnt** (`criterion_e_relabel.py --stageD`): learner-minus-
twin rise on ground-threat and on obstacle frames vs clear frames.
Expectation written now: fewer jumps without a visible reason than Stage C at
40k (39-42 % of onsets), hence better timing; play passing on >= 2 readouts is
possible but not expected at 20,000 decisions (Stage C passed 1 of 3 at 20k and
40k).

### §15 milestone 60k and the extension's reading (2026-10-05 20:12)

`criterion_e_stageC-mario_stageC_60k.json`, `closed_loop_rate_check-stageC-60k-20261004.json`,
`criterion_e_relabel-mario_stageC_60k.json`:

| learner | E (20k / 40k / 60k) | play vs 5 random (20k / 40k / 60k) | threat-clear, obstacle-clear (40k -> 60k) |
|---|---|---|---|
| tko_ro1 | 0.178 / 0.172 / 0.276 | -36 / **+120 pass** / -28 (9/20) | +0.17 -> +0.23, +0.25 -> +0.26 |
| tko_ro2 | 0.209 / 0.207 / 0.099 | **+120 pass** / -18 / never jumps | +0.17 -> +0.04, +0.27 -> +0.07 |
| tko_ro3 | 0.051 / 0.025 / 0.233 | never jumps / never jumps / +15 (10/19) | 0.00 -> +0.15, +0.04 -> +0.23 |
| twins | | 0/3 at every milestone | |

**Reading.** With the polluted take-off label, more training changes *which*
learner has a usable contingency (ro3 learns late, ro2 loses its jumping) but
not play: 1/3, 1/3, 0/3 against rate-matched random jumping. Budget alone is not
the lever with this teacher. Stage D tests the clean label.

## 18. First wiring comparison (pilot), declared 2026-10-05 20:37 before running

**Question (PROTOCOL §1.2, Axis II).** With the validated instrument -- the
clean take-off teacher, learners against matched coin-taught twins, frames of
unseen levels -- does the measured wiring learn the "object ahead -> jump" rule
differently from its depth-stratified nulls?

**Arms.** Connectome: `mario_stageD` (§17), reused as is. Nulls: three runs
`mario_stageD_null0/1/2` (`--version 13 --graph null:<s> --max-decisions 20000`),
the cached `depth_stratified` realisations s = 0, 1, 2 (Gate B's primary null:
edges rewired only within sensorimotor depth strata and sign; same nodes, same
edge count 24,539,704, same 769,342 motor-incoming plastic synapses). Everything
else identical to Stage D: learners and twins, readouts `block_seeds(3)`, noise
7900 + i, sample seed 20261015, label, rule, budget.

**Shared parameters, not rate-matched** (PROTOCOL §9.5). Smoke run (60
decisions, discarded): null s0 motor layer fires ~2x the connectome's (9,101 vs
4,553 spikes per decision). Activity is reported per arm; an effect that could be
an activity effect will be labelled as such; a rate-matched comparison is a
separate, later analysis.

**Measures (each run, frozen at 20,000 decisions, its own graph):**
1. *Contingency* (`criterion_e_relabel.py --graph`): learner-minus-twin rise of
   P(jump) on ground-threat and obstacle frames vs clear frames, unseen 5-1/8-1.
2. *Play* (`closed_loop_rate_check.py --k 5 --graph`): passes vs rate-matched random.

**Reading rule (pilot; three readouts per graph, three null realisations).** A
wiring difference is claimed for a measure only if the connectome's mean over its
three learners lies outside the range of the three nulls' means *and* differs
from the nulls' grand mean by more than twice the spread (SD) of the null means.
Otherwise: no difference shown at this power. Labelled PILOT (PROTOCOL §6.2,
§20), never pooled with a confirmatory run.

**Schedule (owner, 20:40):** the three null runs are trained tonight and the PC is
powered off after training (`scripts/gates/nullcmp_train_then_poweroff.sh`); their
evaluations (§18 measures) run next session, with
`closed_loop_rate_check.py --graph null:<s>` and `criterion_e_relabel.py --graph null:<s>`.

### Stage D results (2026-10-05 21:58)

`mario_stageD` (clean take-off label), 20,000 decisions.

**Primary, rate-matched play (K = 5)** (`closed_loop_rate_check-stageD-20261004.json`):
**0 of 3** -- and for a telling reason: under greedy play the taught learners
almost never press jump (jump rate 0.014 / 0.000 / 0.006), so they play like
the always-run script; the twins 0/3 as well (twin ro1 jumps 70% of decisions and
loses, 7/25).

**Secondary, what is learnt** (`criterion_e_relabel-mario_stageD.json`; unseen
5-1/8-1; learner minus twin vs clear frames): ground threat +0.10 / +0.05 / +0.14,
obstacle +0.15 / +0.08 / +0.15 (p < 1e-4 each) -- the "object ahead -> jump"
contingency is learnt by all three with the clean label too.

**Reading.** With the clean label the contingency is still installed and still
transfers, but the jump probability stays below the run probability almost
everywhere, so the greedy choice never flips: the gap is now plainly a decision
problem (the level at which "jump" wins), not a perception or a credit problem
for this rule. Expectation written at 19:37 (fewer reasonless jumps, better
timing) cannot be judged on greedy play: the learners hardly jump at all. Next
candidates, to declare: evaluate the same brains with a calibrated decision
(e.g. sampled play, or jump when P(jump) exceeds its rate on clear frames by a
margin learnt on training levels), or train with a stronger update on the label.

## 19. Wiring comparison: clarification, secondary analyses and a matched-regime arm (declared 2026-10-06 19:10, before any null evaluation)

Written while the three null runs were finishing their training (resumed 18:55 from
7,766 / 7,001 / 7,467 decisions); no null learner had been evaluated, and their
training logs had not been inspected.

**Literature check (owner's request: look for solutions, stay evidence-based).**
Published connectome-vs-random comparisons point at two confounds that our pilot
does not yet control. (i) *Operating regime.* Human-connectome reservoirs beat
rewired nulls mainly near the critical regime, not in stable or chaotic ones
(Suárez, Richards, Lajoie & Mišić 2021, *Nat. Mach. Intell.*); with a matched
spectral radius, connectome-derived echo-state networks performed as well as, not
better than, random ones on memory tasks (Damicelli, Hilgetag & Goulas 2022, *PLoS
Comput. Biol.* 18:e1010639). (ii) *Weak nulls and activity.* Fly-connectome
reservoirs beat Erdős–Rényi graphs (Morra & Daley 2022, arXiv 2201.09359), but a
2026 preprint (single author, not peer-reviewed; Dhiman, arXiv 2604.04033) reports
that apparent advantages of connectome-constrained networks vanished under
from-scratch initialisation for both graphs and degree-preserving nulls, with mean
activity differing between arms. Our design already trains both arms from the
same start under the same rule, and uses a path-matched null stronger than
degree-preserving rewiring (Gate B); what it does not control is activity: the null
motor layer fires ~2x the connectome's under the shared parameters (§18).

**Clarification of the §18 reading rule (before any null number).** "The spread
(SD) of the null means" is the sample SD (ddof = 1) of the three null-run means.
The rule is applied separately to each §18 quantity: (1a) rise on ground-threat
frames, (1b) rise on obstacle frames (both learner minus twin, vs clear frames),
(2a) mean px difference to the 5 averaged rate-matched random jumpers, (2b)
number of learners passing. Analysis script: `scripts/diagnostics/wiring_compare.py`.

**Secondary analyses (exploratory; reported, never used to override §18).** From
the per-frame data that `criterion_e_relabel.py --pairs` now saves (`.npz`; same
frames for every arm, the connectome arm re-run to produce them):
- *S1, activity:* mean motor spikes per decision, learners and twins, per arm.
- *S2, scale-free contingency:* AUC of the per-frame learner-minus-twin
  difference q for "object ahead" (threat or obstacle) vs clear frames. A rank
  measure is insensitive to the size of P(jump) changes, which depends on how
  strongly the motor layer fires -- the activity confound.
- *S3, decision quality without a threshold:* AUC of the learner's own P(jump) for
  the take-off label v2 of each frame, minus the same AUC for its twin. It asks
  whether a calibrated threshold could time jumps, the gap left open in §18.

**Matched-operating-regime arm (PROTOCOL §9.5; next session).** Each null gets its
own global gain such that its untrained motor layer fires, on 200 frames of the
training levels (1-1, 4-1, scripted play), as many spikes per decision as the
connectome at the frozen gain (`dynamics.gain` = 0.18452; `scripts/diagnostics/null_rate_match.py`,
bisection on log gain, 2% tolerance; the task is never consulted, §8.3). Then
three rate-matched null runs, identical to §18 except the gain, with the same
measures and reading rule. The shared-parameter comparison (§18) and the matched
one answer different questions and are reported side by side, neither replacing
the other (PROTOCOL §9.5).

**Addenda before any null evaluation (19:15).** (a) Calibration procedure: a first
bisection run showed null 0 still firing 8,664 motor spikes per decision at half the
gain (target 4,582), outside the declared bracket; the calibration now scans a
coarse log grid (gain / 2^k, k = 0-4) for every graph, the connectome's own curve
included, then bisects inside the bracket. It runs on the CPU (the GPU is kept for
the null trainings). (b) *S4, wiring and the code before learning*
(`scripts/diagnostics/wiring_probe.py`): decodability (the project's C = .05
balanced logistic decoder on untrained motor counts; fit on 1-1 and 4-1, AUC on
unseen 5-1 and 8-1) of the take-off label v2, of obstacles and of ground threats,
for the connectome and the three nulls at the shared gain (and, once calibrated,
at the matched gains), with the eye's lamina input as reference. Same reading
rule, one value per graph. Exploratory.

### Found while calibrating (19:15-19:25): the activity difference is a wiring difference

- *Gain curves* (untrained, 200 training-level frames, `null_rate_match.py`): the
  connectome's motor layer gives 4,582 / 4,279 / 3,451 / 2,946 / 1,684 spikes per
  decision at the frozen gain and at 1/2, 1/4, 1/8, 1/16 of it; null 0 gives 9,034 /
  8,664 / 8,145 / 6,993 / 4,148. Matching the null by the global gain needs about
  1/13 of the frozen gain, a different dynamical regime for the whole network, not
  a correction of the motor rate. The global-gain arm declared above is therefore
  not a fair matched-regime comparison; it is kept only as a recorded calibration
  (matched gains 0.0126 and 0.0118 for nulls 0 and 1, i.e. 1/15 and 1/16; stopped
  before null 2 to free the CPU, `null_rate_match-20261006.json`).
- *Where the extra drive comes from* (`sensory_fanout.py`, `sensory_fanout-20261006.json`):
  the stimulated eye's 3,575 lamina cells reach 30,948 neurons in one synapse in the
  connectome, 52,344 / 52,444 / 52,494 in nulls 0 / 1 / 2; two synapses from the
  eye, the motor layer receives a summed weight of 12,524 vs 23,683 / 23,773 /
  23,596 (1.89x), and 390 vs 566-572 motor neurons are reached. Direct eye-to-motor
  weight (39) and motor in-strength are identical, as built.
- *Mechanism, checked:* with the lamina of **both** eyes as the source, the null
  reaches exactly the connectome's 60,703 neurons and 24,834 two-hop motor weight.
  The strata do not distinguish the two eyes, so rewiring swaps targets between
  left- and right-eye lamina cells: in the null one eye's signal spreads into both
  eyes' pathways. The game drives one eye only, so the null's motor layer receives
  about twice the drive. The ~2x activity difference of §18 is a consequence of a
  real wiring property (laterality: 78.9% of the connectome's edges join neurons
  annotated with the same side, L-L, R-R or central-central; 46.8% in null 0,
  `null_validate.py`) that the primary null does not preserve. Strictly, it is an
  interaction: the interface drives one eye (PROTOCOL §10), and the null mixes the
  eyes; with both eyes driven, the two graphs would receive the same two-hop drive.
- *Consequence for the design (proposal, not yet declared as a run):* a stronger
  null that also keeps the side (C / L / M / R) of both endpoints in the stratum key
  (`randomize.depth_stratified_lateral`, spec `nulllat:<s>`; three realisations
  being built tonight). If it removes the activity difference by construction, it
  gives the matched-regime comparison without touching any parameter, and it
  sharpens the contrast to: given the layers *and* the sides, does the specific
  wiring matter? It needs its own null validation (§8.3) before use.

## 20. Calibrated-decision play (declared 2026-10-06 19:40 for the next session; not run)

**Why.** Stage D learners rank take-off frames of unseen levels well by their own
P(jump) (AUC 0.80 / 0.76 / 0.75; twins 0.43 / 0.68 / 0.58; frames of
`criterion_e_relabel-mario_stageD.npz`), yet greedy play never jumps (§18): their
P(jump), split over three jump buttons, stays below "run". Threshold moving is the
standard remedy for a calibrated posterior under class imbalance (Elkan 2001;
Saerens et al. 2002): the learnt model is kept, the decision threshold is chosen
on training data.

**Test** (`scripts/gates/threshold_play_check.py`). Per brain, tau = the threshold
on summed P(jump) that maximises TPR - FPR for the take-off label v2 on frames of
the training levels only (relabel procedure, seed 20261017). Play on the four
unseen levels x 8 delays: RIGHT+A+B when P(jump) >= tau, else the best non-jump
action; against 5 random jumpers at the brain's own jump rate; PASS per brain at
Wilcoxon p < .05, as §14.

**Reading.** Learners 3 / 3 pass and twins (same calibration) at most 1 / 3: the
decision level was the gap, and the learnt rule times jumps. Learners <= 1 / 3:
the ranking (AUC ~0.75) is not precise enough to time jumps, even with a
calibrated threshold. The same test then becomes the play measure of the wiring
comparison (all arms, own graph). Before use, the harness is checked on the
scripted references of §14 (positive: take-off expert; negative: coin twin).

### The laterality-preserving null, built and validated (19:30-19:45)

`randomize.depth_stratified_lateral` (stratum key + side of both endpoints; 6,984
edge classes instead of 1,368). The builder was first sped up (one stable sort
instead of a mask per class); rebuilding `depth_stratified` s0 with the new code
reproduces the cached realisation bit for bit (src, dst, weight, sign).
Validation (`null_validate.py`, no task involved), realisation 0: in- and
out-degrees exact, edge count, sign and weight multisets equal, depth-stratum
edge classes equal, side-pair edge counts equal (same-side share 0.7887, as the
connectome), sampled edge overlap with the connectome 5.4% (the primary null: 2.7%).
Fan-out (`sensory_fanout.py`): the stimulated eye reaches exactly the connectome's
30,948 neurons and gives exactly its two-hop motor weight (12,524); 373 motor
neurons reached in two hops (connectome 390). The aggregate two-hop drive is thus
matched by construction, while 94.6% of the edges are rewired. **Correction
(20:00): the motor firing is not matched** -- see the S4 results below (lateral
nulls 7,870-8,237 spikes per decision vs 4,574).

## 21. Wiring comparison against the laterality-preserving null (declared 2026-10-06 19:45; to run next session)

Declared before any result of the lateral nulls other than their validation and
fan-out above (the S4 probe of the lateral nulls was still running).

**Why.** The primary null mixes the two eyes, which doubles the motor drive under
a one-eye interface (§19); a difference against it may be a laterality/activity
effect. The lateral null keeps depth strata *and* sides, matches the eye's
aggregate two-hop drive by construction (but not the motor firing, found 20:00),
and rewires 94.6% of the edges: the contrast becomes "given the
layers and the sides, does the specific wiring matter?".

**Runs.** `mario_stageD_nulllat0/1/2` = Stage D (version 13) with
`--graph nulllat:<s>`, s = 0, 1, 2, 20,000 decisions, everything else identical to
§17-§18 (learners, twins, readouts, noise, sample seed, label, rule). Shared
parameters; no gain change.

**Measures and reading rule.** Exactly §18 (1a, 1b, 2a, 2b) with the §19
clarification (sample SD of the null means), plus S1-S3 from the per-frame data,
and the §20 calibrated-decision play once its harness passes its references.
Primary for the wiring question: 1a and 1b. PILOT (three readouts, three
realisations); a confirmatory run would follow with more realisations, sized from
the spread measured here (`wiring_compare.py` reports the detectable difference
for 3 / 5 / 10 / 20 nulls).

**Activity check, reported first.** The lateral nulls' motor spikes per decision
on the evaluation frames (S1) next to the connectome's; if they differ by more
than 20%, the comparison is labelled activity-confounded like §18.

### §19 S4 results at the shared gain (19:15-19:38; `wiring_probe-20261006.json`)

Decodability from the untrained motor code (AUC, fit 1-1 + 4-1, test unseen 5-1 +
8-1; 600 test frames: take-off 292 positive, obstacle 275, ground threat 48 vs 483):

| label | eye input | connectome | null 0 / 1 / 2 | pilot rule |
|---|---|---|---|---|
| take-off v2 | 0.89 | **0.79** | 0.95 / 0.88 / 0.98 | difference shown, connectome lower (-0.15; 2 SD = 0.11) |
| obstacle | 0.93 | **0.83** | 0.97 / 0.91 / 1.00 | difference shown, connectome lower (-0.13; 2 SD = 0.09) |
| ground threat | 0.75 | **0.62** | 0.48 / 0.40 / 0.46 | difference shown, connectome higher (+0.17; 2 SD = 0.08) |

Motor spikes per decision on these frames: connectome 4,574, nulls 9,038 / 9,361 /
9,264. Reading: before any learning, the eye-mixing nulls -- which fire twice as
much -- carry obstacles and take-off moments *more* linearly in their motor
layer than the connectome, and ground threats less (their threat decoders fall
below chance on new levels: what they learn on 1-1/4-1 does not transfer). Both
directions are confounded with the 2x activity and the eye mixing; the
laterality-preserving nulls (same drive) are the test of whether either survives.
Limitation: the scripted frames are dominated by moments stuck against a pipe
(1-1: 89% labelled), so "obstacle" is mostly "pipe right ahead".

**Lateral nulls 1 and 2 (19:45):** same fan-out as the connectome exactly (30,948
neurons; two-hop motor weight 12,524; 379 motor neurons reached, connectome 390).
**Play-harness reference for the v2 label** (`oracle_criterion_check.py --takeoff2 --k 5`,
scripted, no brain; needed by §20-§21 and missing since §17): the expert following
`takeoff_label_v2` beats 5 averaged random jumpers at its own rate (0.38) on the
unseen levels, 15 / 9, +82 px, p = 0.020; a second random draw does not pass
(p = 0.25), always-run does not (1 / 23). Harness validated for the clean label.

### S4 against the laterality-preserving nulls (19:45-20:00; `wiring_probe-nulllat-20261006.json`)

| | connectome | lateral null 0 / 1 / 2 | pilot rule |
|---|---|---|---|
| motor spikes per decision | **4,574** | 8,142 / 8,237 / 7,870 | difference shown: connectome sparser (-3,509; 2 SD = 382) |
| take-off v2 (AUC) | 0.79 | 0.79 / 0.96 / 0.75 | no difference (inside the range) |
| obstacle | 0.83 | 0.75 / 0.99 / 0.77 | no difference (inside the range) |
| ground threat | 0.62 | 0.59 / 0.37 / 0.42 | no difference (above the range, but within 2 SD = 0.23) |

**Reading.** (1) The eye mixing explained only part of the activity gap: with the
eye's two-hop drive matched exactly, the lateral nulls still fire 1.7-1.8x the
connectome's motor spikes. Across all six nulls of both classes (7,870-9,361) the
connectome's motor layer is about half as active under the same parameters and
input: a robust, wiring-specific property of the dynamics -- the measured wiring
makes the motor response *sparser* (PROTOCOL H1: the nulls are not functionally
neutral). Where it arises beyond two synapses is not yet known. (2) What the eye-
mixing nulls seemed to do better (obstacles, take-off moments) does not survive the
lateral nulls; decodability varies widely between realisations of the same null
class (take-off 0.75-0.96), so single realisations would mislead. (3) The
ground-threat signal is the one place the connectome sits above every null
(0.62 vs 0.37-0.59), not significant at this power (48 positive test frames).
Hence the §21 activity check will label the lateral comparison activity-confounded
too: in this model, activity is not a nuisance to match away but one of the
things the wiring determines.

**Where the sparseness arises** (`activity_by_depth.py`, `activity_by_depth-20261006.json`;
untrained, 64 training-level frames, spikes per decision summed per depth stratum
from the lamina -- the strata every null keeps):

| stratum | connectome | lateral null 0 | primary null 0 |
|---|---|---|---|
| depth 0 (lamina) | 11,832 | 9,908 (0.84x) | 11,183 (0.95x) |
| depth 1 | 62,414 | 67,946 (1.09x) | 73,775 (1.18x) |
| depth 2 | 80,820 | 135,906 (1.68x) | 153,409 (1.90x) |
| depth 3 | 75,799 | 123,897 (1.63x) | 146,767 (1.94x) |
| depth 4+ | 10,550 | 19,669 (1.86x) | 24,334 (2.31x) |
| motor layer | 4,552 | 8,157 (1.79x) | 9,011 (1.98x) |

The arms agree up to one synapse from the lamina and part at the second: from
depth 1 to depth 2 activity grows 1.3x in the connectome, 2.0x in the lateral
null. The measured wiring between the first and second relay contains the
spread of activity; one rewiring within the same layers, sides, degrees, signs
and weights does not. Candidate mechanism, untested: inhibition targeted at the
co-activated neurons (the nulls keep the sign multiset per class, not who
inhibits whom).

**Which synapses keep it sparse** (`sign_hybrids.py`, 20:03; same frames and seeds):
mixing the connectome with lateral null 0 by sign --

| graph | depth 1 | depth 2 | depth 3 | motor |
|---|---|---|---|---|
| connectome | 62,414 | 80,820 | 75,799 | 4,552 |
| excitatory from the connectome, inhibitory from the null | 64,281 | 80,817 | 70,846 | **4,502** |
| excitatory from the null, inhibitory from the connectome | 72,036 | 145,323 | 138,755 | **10,356** |
| lateral null 0 | 67,946 | 135,906 | 123,897 | 8,157 |

The hypothesis written above (targeted inhibition) is **refuted**: with the
connectome's excitatory wiring the response stays sparse whatever the inhibitory
wiring, and with the null's excitatory wiring it spreads even with the
connectome's inhibition. The measured *excitatory* wiring contains the spread --
consistent with the eye fan-out finding (convergent excitation: co-active inputs
share targets, so fewer downstream neurons are recruited). Next test, not run:
target overlap of co-active (retinotopically neighbouring) neurons, connectome vs
nulls.

**Convergence is not it; in-strength heterogeneity is the lead** (`convergence.py`,
20:04; scratch check of the depth 1 -> 2 weights, 20:05). For the connectome's own
co-active sets, the number of distinct excitatory targets is the same in every
graph (depth 1 -> 2: 18.40 edges per target in the connectome, 18.29 in lateral null
0, 18.29 in null 0). What differs is how the input is *distributed* over the
targets: same number of excitatory inputs per depth-2 neuron (median 17) and same
mean summed weight (122.7, by construction), but in the connectome the weight is
concentrated on few neurons (coefficient of variation 3.24 vs 2.28; 90th
percentile 201 vs 271; 99th 1,309 vs 1,068), and the median depth-2 neuron's
total in-strength is 392 vs 530. Our nulls keep each source's weights (out-strength)
but hand them to random targets, homogenising in-strength: more neurons near
threshold, more spikes. Proposal for the control ladder (PROTOCOL §8.2): a null
that also keeps in- and out-strength, e.g. by simulated annealing of the weights
(Milisav, Bazinet, Betzel & Mišić 2024, *Nat. Comput. Sci.*); if it fires like the
connectome, the sparseness is a strength-sequence property rather than a
who-connects-to-whom property.

**In-strength test** (`instrength_null_check.py`, 20:09): lateral null 0 rebuilt with
each target's own incoming weights handed back (`weights_by="dst"`: in-strength
exact, out-strength not; same seed, same frames). Spikes per decision: depth 1
50,383, depth 2 95,631, depth 3 111,863, motor **6,190** -- between the connectome
(80,820 at depth 2; motor 4,552) and the lateral null (135,906; 8,157). Keeping the
in-strength sequence removes about three quarters of the excess at depth 2 and
about half at the motor layer: the connectome's sparser response is largely a
property of how synaptic strength is distributed over target neurons, and partly
of something the in-strength sequence alone does not carry (out-strength was lost
here; joint in/out strength, or the specific pairing). A null keeping both
strengths (simulated annealing, Milisav et al. 2024) is the next rung.

## 18 (results). First wiring comparison, contingency (2026-10-06 20:47; `wiring_compare-20261006.json`)

Learner minus coin twin, rise of P(jump) vs clear frames, unseen 5-1 / 8-1, each
brain on its own graph:

| quantity | connectome (ro1 / ro2 / ro3) | null 0 / 1 / 2 (mean of 3 readouts) | pilot rule (§18, sample SD) |
|---|---|---|---|
| **1a ground threat** | 0.104 / 0.050 / 0.142 (mean **0.099**) | 0.051 / 0.079 / 0.017 (0.049, SD 0.031) | outside the range, but +0.050 < 2 SD (0.062): **no difference shown** |
| **1b obstacle** | 0.154 / 0.081 / 0.145 (mean **0.126**) | 0.080 / 0.105 / 0.033 (0.073, SD 0.037) | outside the range, but +0.054 < 2 SD (0.073): **no difference shown** |
| S1 motor spikes, learners | 4,925 | 9,023 / 9,017 / 9,100 | difference: connectome ~2x sparser |
| S2 rank of the rise (AUC, object vs clear) | 0.828 | 0.830 / 0.876 / 0.799 | no difference |
| S3 take-off AUC, learner minus twin | 0.210 | 0.250 / 0.443 / 0.306 | no difference (connectome lower, within 2 SD) |
| X (post hoc) frame-specific SD of the logit change | 0.76 | 0.40 / 0.41 / 0.34 | larger in the connectome (exploratory only) |

Every learner of every arm acquired the contingency (p < 1e-4 for threat and
obstacle vs clear, 12 of 12). **Reading.** The connectome's learners moved P(jump)
1.7-2x as much before objects (1a, 1b, connectome above all three nulls), but
by the declared rule this is not a shown difference at this power; the *ranking*
of frames is the same in both arms (S2), and the nulls' learnt change is mostly a
global "jump less" bias with half the frame-specific part (X, post hoc). The arms
also differ twofold in motor activity, so even a larger confirmed effect on 1a/1b
would be labelled activity-confounded (§18), until the lateral- and
strength-preserving nulls are run. **Planning:** the spread of the null means
(SD 0.031-0.037) limits detection: against 3 / 5 / 10 / 20 nulls, the detectable
connectome-null difference is 0.19 / 0.13 / 0.10 / 0.09 for 1a. **Corrected at
20:55** (the first version said more readouts per graph would fix this): splitting
the spread, the readout noise within a graph (SD 0.026 for 1a, 0.032 for 1b) is
smaller than the genuine variation between null realisations (SD 0.027 / 0.032
once readout noise is removed; from three nulls, so itself imprecise). With 12
readouts per graph and 20 nulls the detectable difference is still ~0.085 (1a) /
0.10 (1b). Against one connectome the test asks whether it is an outlier of the
null distribution; at the observed +0.05 (about 1.7-1.9 realisation SDs) it is
not separable, whatever the budget. A confirmatory test of 1a/1b is only worth
running if the expected difference is larger, e.g. against the lateral nulls.

**Caution on the play measure, seen in the partial logs (20:50).** The 32 starts are
8 delays on each of 4 levels, and a frozen greedy policy replays almost the same
trajectory from neighbouring delays: null 0's coin twin ro1 reaches x ~930 on 5-1
from 6 of its first 8 starts (random jumpers at its rate: ~240-620), so 8 "wins"
are close to one. The Wilcoxon over 32 starts treats them as independent. For
§20-§21 the play result should also be reported per level (4 levels, sign of the
mean difference), and a pass that comes from one level flagged as such. This is
a reading caution, not a change to the declared rule.

**Replication on every realisation (20:54-20:58; `activity_by_depth-20261006.json` now
holds all seven graphs, same frames and seeds; `instrength_null_check-s1/-s2`):**

| graph | depth 1 | depth 2 | depth 3 | motor |
|---|---|---|---|---|
| connectome | 62,414 | 80,820 | 75,799 | 4,552 |
| lateral nulls 0 / 1 / 2 | 67,946 / 68,649 / 68,230 | 135,906 / 135,370 / 134,808 | 123,897 / 125,103 / 118,944 | 8,157 / 8,269 / 7,835 |
| primary nulls 0 / 1 / 2 | 73,775 / 74,219 / 74,740 | 153,409 / 155,251 / 154,952 | 146,767 / 150,978 / 150,197 | 9,011 / 9,377 / 9,270 |
| lateral, in-strength kept, 0 / 1 / 2 | 50,383 / 50,665 / 51,130 | 95,631 / 93,925 / 96,003 | 111,863 / 109,444 / 111,102 | 6,190 / 6,026 / 6,027 |

Realisations of a class agree within a few percent; the connectome lies far
outside every class. Keeping the in-strength sequence closes 72-76% of the excess
at depth 2 and 55-60% at the motor layer in the three realisations.

## 18 (results, play). Rate-matched greedy play (21:00)

| arm | learners: passes, jump rate | twins: passes |
|---|---|---|
| connectome (§17) | 0 / 3; 0.014 / 0.000 / 0.006 | 0 / 3 |
| null 2 | 0 / 3; 0.000 / 0.000 / 0.000 (never jump: identical to always-run) | 0 / 3 (8/20, 2/13, 1/0) |
| null 0 | 0 / 3; 0.000 / 0.876 / 0.000 (ro2 jumps 88% and loses, 1/7) | **1 / 3**: coin twin ro1 passes (19/11, +219 px, p 0.007) |
| null 1 | stopped at 3 of 32 starts (deviation, JOURNAL 20:53); rerun next session | |

Under greedy play the learners of both arms do not jump at all, so measure 2 cannot
separate the wirings: the decision-level gap of §18 is shared by the nulls. The
calibrated-decision test (§20) is the play measure that can.

**Same amount of plasticity in every arm** (run records, 20,000 decisions): relative
RMS weight change of the learners 0.127-0.131 (connectome) vs 0.120-0.128 (nulls);
twins 0.134-0.138 vs 0.113-0.121. The connectome's learners did not change their
synapses more; the same amount of change produced a larger, more frame-specific
change of the output (post hoc X above) -- consistent with, not proof of, the
sparse-code reading of THEORY_NOTES §12.

**A negative control passed (21:11).** Null 0's coin twin ro1 (jump rate 0.66, not
taught any contingency) beats its rate-matched random jumpers -- mostly by
replaying one far-reaching trajectory on 5-1 (x ~930 from most delays). This is
the clustering caution above made real: the play harness can be passed without a
learnt contingency, by a fixed policy that happens to suit one level. Before §20
-§21 use it, the play verdict must be reported per level and a pass carried by
one level must not count.

## 22. Note (2026-10-08 16:37): taught learners' training-time play includes online steering

From `docs/RL_STUDY.md` §9.1-§9.2. Resuming the connectome Stage D from its checkpoint
with learning on vs frozen, on paired starts, gives +96 px Δx_rand [+9, +180] for
learning on; the frozen weights reach ≥ +50 in one readout of three. The label, applied
at every decision, raises P(jump) at the following decisions through fast weight
changes: **training-time play of a taught learner shows the label's timing, not only a
stored policy.** Consequences for this study:
- §18 quantities 1a / 1b (and S1-S4) are computed on fixed frames from frozen
  checkpoints (`criterion_e_relabel.py`, `wiring_probe.py`): **not affected**.
- §18 play (rate-matched greedy play) and §20 calibrated-decision play evaluate frozen
  checkpoints: not affected by steering; their weak results agree with the frozen
  Stage D policy timing little.
- Statements about Stage D learners' *training-time* play (RL_STUDY §5 amendment 2)
  are withdrawn as statements about learnt policies.
- **§21** (Stage D on the lateral nulls, running since 2026-10-08 15:13, resumable):
  its primary measures (1a / 1b) remain valid; any play reading must use frozen
  checkpoints.
