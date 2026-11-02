# Does the fly learn Mario, or memorise 1-1? — generalisation study

Owner's question (2026-10-03): does the fly memorise the level by sight, or does
it learn to recognise enemies and obstacles — and would it cope with a similar
level laid out differently? This document declares the study before it runs
(sections 1-5), then records what was done and found (sections 6+).

## 1. What the architecture allows, before any data

- **No memory of time.** Each decision is one sweep from rest; the network keeps
  no state across decisions and has no clock. It cannot learn "at decision 37,
  jump". The only inputs are the current frame and its difference from the
  previous one (the ON/OFF transient terms).
- **Two ways to fit 1-1 remain.** (a) *Visual memorisation*: map specific
  screens (this hill, this block, this cloud) to actions — 1-1 replays nearly
  identically because the emulator is deterministic. (b) *Recognition*: map a
  local configuration ("something dark just ahead of Mario") to an action. The
  camera follows Mario, who stays near a fixed screen column while running, so a
  danger at a given distance ahead falls on nearly the same eye position
  wherever it occurs: (b) is cheap for a retinotopic system.
- **Only the last layer learns.** The 769,342 motor-incoming synapses read
  features the frozen connectome computes. Recognition is possible only if
  those features already carry a danger signal that is stable across scenes.
  Evidence so far: the frozen motor code identifies scenes across unseen frames
  at 82.6% versus 66.7% for the raw input (more invariant than the input), while
  the two-cue learners reached only ~60% on unseen frames.

## 2. Literature that shapes the design

Agents trained on one level overfit it; more distinct training levels reduce
the generalisation gap (Cobbe et al. 2019, CoinRun, ICML; Zhang et al. 2018, "A
study on overfitting in deep RL"). Randomising nuisance factors during training
forces invariance (domain randomisation, Tobin et al. 2017, IROS); so does
augmenting the observations (RAD, Laskin et al. 2020, NeurIPS). Occlusion
sensitivity shows which part of an input drives a decision (Zeiler & Fergus
2014, ECCV). Survey: Kirk et al. 2023 (JAIR), zero-shot generalisation in RL.

## 3. Design

Same model, regime, fixed readout and rule as pilot v2 (actor-critic, TD(lambda)
.98/.9, exploration floor .1, death -100 in the learning signal, eta .001). One
run, `mario_gen` (`scripts/train_mario_sweep.py --version 3`), six learners: per
fixed readout (`block_seeds(3)`), a pair sharing noise seed (7300 + i) and
random draws (action sampling, exploration):

| learner | trains on | start | palette |
|---|---|---|---|
| specialist_ro1-3 | 1-1 only | at once | real |
| generalist_ro1-3 | 1-1, 2-1, 4-1, 7-1 (one drawn per episode) | after 0-22 NOOP decisions (0-1.5 s) | RGB channels permuted, gain 0.75-1.25, per episode |

Training time is fixed by the owner's two-hour budget (stopped ~15:00 with a
checkpoint); both conditions get exactly the same number of decisions.

## 4. Tests (trained weights frozen, greedy actions)

Each trained learner and the *untrained* brain with the same readout and noise
seed (the baseline) play, on the real palette:

| suite | levels | what it separates |
|---|---|---|
| trained | 1-1 | competence on the training level |
| desynchronised | 1-1 after 6-25 NOOP decisions | enemies meet Mario elsewhere in the scenery: memorisation breaks, recognition does not |
| unseen, day | 5-1, 8-1 | new layouts, familiar tiles and palette |
| unseen, night | 3-1, 6-1 | new palette (dark sky) |

Three episodes per learner and level (start delays drawn from a fixed
evaluation seed). Measures: furthest x per episode and the share of episodes
that get past the first enemy met. **Occlusion test** on frames where a trained
learner meets an enemy ahead (RAM labels, analysis only): erase the enemy's
screen box (paint it with the surrounding background) versus an equal box of
background elsewhere; record the change in the probability of the chosen action
and of jumping.

*Refinement before any result (13:50, smoke run only):* a box of plain sky is
not a control -- erasing sky with sky changes nothing (the smoke run gave
exactly 0), so it could never come out disruptive. The two scenery boxes are
therefore the most textured candidates (pixel variance) at the enemy's height
or above, overlapping neither the enemy nor Mario: hills, bushes, trees,
fences, blocks (`scripts/gates/mario_generalisation_eval.py`, `boxes`).

*Addition at 14:28, after seeing the 10,000-decision greedy episodes (labelled
post hoc):* under greedy play most learners of both conditions only ran and died
at the first enemy whatever the start delay, so greedy play hides graded
preferences -- a learner with 40% "jump" in front of a Goomba never jumps. The
final checkpoint is therefore also evaluated with the learnt policy *sampled*
(softmax over the scores at tau 2, without the exploration floor), same suites
and episodes. The occlusion test, which measures the change in the action's
probability, is unaffected by this choice.

*Addition at 14:52, after the 10,000-decision occlusion (post hoc):* changes in
probability were tiny for enemy and scenery alike (|delta| < 0.004), but a saturated
policy cannot move its probabilities whatever the input, so that measure could not
come out positive. The final occlusion also reports the change in the log-odds of
jumping against the chosen action (unaffected by saturation) and a positive
reference -- the whole frame swapped for another sample's -- that gives the scale of
a large change.

## 5. Predictions, stated before the results

- **Memorisation** predicts: good on 1-1, a drop with desynchronised starts,
  unseen levels no better than the untrained brain, background occlusion as
  disruptive as enemy occlusion.
- **Recognition** predicts: held-out day levels above the untrained brain,
  desynchronisation harmless, enemy occlusion far more disruptive than
  background occlusion.
- **Generalist vs specialist**: if training diversity matters here as in deep
  RL, the generalist loses a little on 1-1 and gains on held-out levels.
- **Night levels**: expected to fail for everyone (the eye sees different
  contrasts); palette jitter may soften it.
- A run too short to learn anything on 1-1 makes every generalisation test
  uninformative; that outcome is reported as such.

---

## 6. What was done (2026-10-03, 13:38 - ~15:40)

1. **Level switching, delayed starts, RAM labels** (`src/flymario/env/retro_env.py`,
   `training/mario.py`): the emulator pool can start any of the nine savestates
   after a number of NOOP decisions; the nearest enemy ahead (distance, type,
   screen position) is read from RAM for metrics and analysis only. Addresses
   checked against a Goomba approaching in 1-1 (type 6; Mario died at x ~300 when
   it reached him). 6-1 turned out to be a night level (mean frame luminance 33
   against 125-150 for day levels), so it joined 3-1 in the night suite.
2. **Per-episode randomisation in the trainer** (`training/mario_sweep.py`):
   per-learner level pool, start delay and palette jitter (RGB channels permuted,
   gain 0.75-1.25), drawn from a per-learner seeded generator saved in the
   checkpoint; the eye receives the jittered frame, the screen and the score are
   untouched.
3. **The run** `mario_gen` (`scripts/train_mario_sweep.py --version 3`), launched
   13:41: six learners (three specialist/generalist pairs), v2 rule, ~7.5
   decisions per second per learner, followed in Fly Command.
4. **Evaluator** (`scripts/gates/mario_generalisation_eval.py`): frozen greedy
   play of the six learners and three untrained brains on the four suites, plus
   paired occlusion. A smoke run showed that erasing plain sky changes nothing (an
   occlusion control that cannot come out disruptive); before any result, the
   scenery boxes were redefined as the most textured boxes at the enemy's height
   (hills, bushes, trees, fences), checked visually on 1-1, 5-1 and 8-1.
5. **Representation probe** (`scripts/diagnostics/danger_probe.py`, exploratory):
   can a linear readout of the *untrained* connectome's motor code recognise
   "enemy just ahead" on levels it was never fitted on?
6. **Theory** for the manuscript: `report/THEORY_NOTES.md` §7.

## 7. Closed-loop danger readout: exploratory run and declared confirmation

**Exploratory run (14:53, `danger_controller-20261003.json`).** A logistic readout
fitted on the *frozen, untrained* connectome's motor spike counts, on 1-1 frames
only (584 frames, RAM labels "enemy 8-64 px ahead" / "none within 160 px"), drives
Mario: jump if P(danger) > 0.5, else run (`scripts/diagnostics/danger_controller.py`).
Supervised and therefore *not* the study's reinforcement learner: it asks whether
the connectome's code supports a policy that transfers. On unseen day levels 5-1
and 8-1 (6 episodes): furthest x 656 ± 270 against 207 ± 7 always running, 245 ±
13 jumping at random at the same rate, 251 ± 40 for the same readout on the raw
lamina input; on 8-1 it reached x 1826 and 1075. Night levels: no controller got
past the first obstacle (all equal). Six episodes with two large successes: too
few to rely on.

**Confirmation, declared at 14:58 before running.** Same script and threshold,
fresh seeds (collection 20261005, noise 881000, so a newly fitted readout), five
day levels the readout never saw (2-1, 4-1, 5-1, 7-1, 8-1), starts after
0, 3, 6, ..., 21 NOOP decisions: 40 paired episodes per controller.
Criterion: the motor readout beats both `random_jump` and `raw_input` in furthest
x, paired one-sided Wilcoxon p < 0.05 each. Reported whatever the outcome.

**Confirmation result (15:12, `danger_controller-confirm-20261003.json`).** Readout
refitted on 1,759 fresh 1-1 frames (272 danger). Furthest x on the five unseen day
levels, 40 paired episodes per controller (mean ± SE): connectome readout
515 ± 42, raw-input readout 656 ± 61, random jumps at the same rate 407 ± 33,
always running 263 ± 9. Connectome readout against always running 37 wins / 1
loss (Wilcoxon p < 1e-4), against random jumping 27 / 12 (p = 0.002), against the
raw-input readout 17 / 22 (p = 0.93). **Criterion: FAIL** -- it beats random
jumping but not the raw-input readout. The exploratory advantage of the
connectome over its own input (6 episodes) did not replicate in closed loop. What
did replicate: a danger detector fitted on one level only makes jumps *timed by
vision* that pay off on five levels it never saw -- shown for the connectome
readout against random jumps at its own rate (0.34-0.36 of decisions).

*Caveat found on re-reading (15:36):* the raw-input readout jumps far more often
(0.58 of decisions against 0.36), and the random control was matched to the
connectome readout's rate only. Its win over random jumping (28 / 12, p = 0.002)
is therefore not rate-matched, and its lead over the connectome readout may come
partly from jumping more rather than from better timing. A rate-matched control
per readout is needed before comparing the two.

## 8. Results of the reinforcement-learning run (`mario_gen`)

Stopped at 15:00 by the owner's budget, checkpoint at **20,505 decisions per
learner** (~23 min of game time each, 125-150 episodes); evaluated frozen
(`mario_generalisation_eval-20261003.json`, final; `-at10k-20261003.json` at
10,000 decisions).

**Greedy play, final checkpoint** (furthest x, mean ± SE, 9 episodes per cell
on 1-1, 18 on the unseen pairs):

| suite | untrained | specialist | generalist |
|---|---|---|---|
| 1-1 (trained) | 87 ± 25 | 302 ± 3 | 310 ± 3 |
| 1-1 desynchronised | 113 ± 37 | 302 ± 3 | 310 ± 3 |
| 5-1, 8-1 unseen day | 72 ± 20 | 196 ± 4 | 198 ± 4 |
| 3-1, 6-1 unseen night | 113 ± 19 | 465 ± 11 | 442 ± 12 |

Every trained episode on a day level ended in a death at the first enemy (54/54).
The trained learners beat the untrained brain everywhere, but only because the
untrained brain's greedy action rarely moves Mario. **81 of 90 trained greedy
episodes end within 20 px of where a vision-free script that always runs ends**
on the same level and start (mean difference 8 px; `danger_controller-20261003.json`,
`always_run`). The night "advantage" is the same thing: the first obstacle of 3-1
and 6-1 lies further on. Generalist and specialist do not differ.

At 10,000 decisions one specialist (ro2) did get past the first Goomba under
greedy play (x 687-723 on 1-1, 594 desynchronised, 691 on 5-1, 546 on 8-1); by
20,000 it had lost it. The training curves (figure, panel A) show the share of
training episodes past the first enemy -- with sampled actions and the
exploration floor -- rising to 0.8-0.95 for two specialists, then falling back to
0.4-0.55 for all learners from episode ~80.

**Occlusion, final checkpoint** (162 frames with an enemy 24-72 px ahead,
paired noise; |change| in the log-odds of jumping against the chosen action):

| erased | specialist (82) | generalist (80) |
|---|---|---|
| the enemy | 0.009 | 0.006 |
| scenery box 1 | 0.011 | 0.007 |
| scenery box 2 | 0.006 | 0.004 |
| whole frame swapped (positive reference) | 0.154 | 0.129 |

The test can detect a change (the swap moves the decision 15-20 times more than
any box), and erasing the enemy moves it no more than erasing a bush. Even the
swap is small in absolute terms: the policy hardly depends on the picture.

*Correction (2026-10-03, evening; `docs/PERCEPTION_CREDIT_STUDY.md` §5).* This
occlusion test had no untrained baseline, and it needs one. On fresh frames
(scripted play on 5-1 and 8-1, `occlusion_criterion_check-fresh-20261003.json`)
erasing the enemy changes the decision of the **untrained** brains about twice as
much as erasing scenery (specificity S 0.19-0.24): the enemy is a salient
stimulus for this eye whatever was learnt. On those frames 4 of the 6 trained
learners also show enemy > scenery, but every one of them less than its
untrained twin (S 0.05-0.16). The occlusion result therefore does not show that
the learners "ignore the enemy"; it shows that learning did not *increase* the
weight of the enemy in their decisions (it lowered it, relative to the twins).
The table above is kept as measured, on the learners' own frames.

**Sampled policy** (post hoc, §4 addition; `mario_generalisation_eval-sampled-20261003.json`;
softmax at tau 2, no floor; furthest x mean ± SE, paired by readout and episode):

| suite | untrained | specialist | generalist |
|---|---|---|---|
| 1-1 (trained) | 519 ± 61 | 412 ± 54 | 644 ± 75 |
| 1-1 desynchronised | 504 ± 74 | 413 ± 57 | 731 ± 176 |
| 5-1, 8-1 unseen day | 307 ± 34 | 257 ± 22 | 397 ± 81 |
| 3-1, 6-1 unseen night | 437 ± 29 | 531 ± 42 | 513 ± 38 |

Sampled, the *untrained* brain gets past the first enemy often: its near-random
choices include jumps. No trained condition beats it on unseen levels
(day: specialist 8 wins / 10 losses, generalist 8 / 10, Wilcoxon p > 0.25). The
only nominal difference, generalist over specialist on 1-1 (6 / 2, p = 0.039
two-sided), does not survive the twelve comparisons made. On unseen day levels two
generalists averaged x 472 and 504 and the third 214: with three learners per
condition, learner-to-learner variation dominates. Sampling confirms the greedy
picture: training sharpened a bias, it did not add a visual skill.

## 9. Conclusion

**The owner's question -- by heart, or by recognising enemies?** Neither, at this
stage. After 20,505 decisions the reinforcement learners of both conditions have
learnt a *habit*: a policy that hardly depends on what the eye sees (whole-frame
swap barely moves it; occlusion: see the correction in §8 -- learning did not
raise the enemy's weight above the untrained twin's), which under greedy play is
behaviourally the same as a script that always runs. They cannot
have learnt the level by heart in time (the architecture keeps no memory across
decisions, §1), and they did not learn it by sight either: nothing is tied to the
enemy. Their better scores on unseen levels than the untrained brain are not
generalisation; they are the "run right" habit, which transfers trivially.
Training on four levels with palette and timing randomisation changed nothing
within this budget (generalist = specialist on every suite), as expected when
there is no visual mapping for diversity to act on (`THEORY_NOTES.md` §7).

**Is recognition possible with this brain?** Yes. The untrained connectome's motor
code carries an "enemy just ahead" signal that a linear readout fitted on 1-1
recognises on unseen levels (open-loop AUC 0.955, exploratory), and in closed loop
a readout fitted on 1-1 only, driving jumps, beats untimed jumping on five unseen
day levels (27/12 paired wins, p = 0.002, confirmation). This is the
generalisation the owner asked about, with a caveat added after a later check
(PERCEPTION_CREDIT_STUDY §5): the readout's logit changes 1.1-1.3 when the enemy
is erased but also 0.76-0.85 when a scenery box is, so its jumps are driven by
the picture and transfer to new layouts, but they are not shown to be triggered
by enemies specifically; and a scripted jumper that jumps exactly at enemies
(from RAM) does *not* beat random jumping at its rate (17/12, p 0.58), because
random jumps also clear pipes and pits. Further limits: the
same readout on the raw lamina input goes as far or further (it also jumps more
often, so the comparison is not settled; the connectome is not shown to help
beyond its input), and nothing works at night (all controllers
die at the first night obstacle) -- recognition fitted on day colours does not
survive a new palette: the same open-loop readout fitted on 1-1 is at chance on
3-1 and 6-1 (AUC 0.50 from motor counts, 0.49 from the raw input;
`danger_probe_night-20261003.json`, exploratory).

**Why the reinforcement learner misses what a supervised readout finds.** Its only
teacher is a scalar TD error, broadcast to all 769,342 plastic synapses through
fixed random readout columns. Raising the "run" bias pays on every one of the
~15 decisions per second; tying a jump to the enemy pays only in the few
decisions per encounter where a jump saves Mario (enemy-ahead frames are 11% of
decisions in scripted 1-1 play, the decisive window a fraction of those), and
only if a jump happens to be sampled then. The input-independent direction
collects most of the credit (an estimate; THEORY_NOTES §6, §8). The information is in the brain; the credit
assignment does not reach it in 20,000 decisions.

## 10. Proposed solution (next study, to declare before running)

1. **Teach the danger signal with privileged information during training only**
   -- "learning by cheating" (Chen et al. 2019, CoRL): a teacher that reads the RAM
   (enemy distance) says *when* a jump is right; the fly's readout learns it from
   vision with the local rule (supervised three-factor learning passed Gate G at
   96%); at test the fly sees only pixels. §7 shows this transfers to unseen
   levels.
2. **Or keep reinforcement learning but give the critic the privileged state** --
   asymmetric actor-critic (Pinto et al. 2018, RSS): a critic that knows the enemy
   distance produces a TD error sharply peaked at encounters, so the actor's
   credit lands on the frames that matter. The actor still sees only vision.
3. **Concentrate the experience on encounters** -- start episodes from save
   states a few decisions before enemies (on the four training levels), so that
   most decisions carry danger-related credit instead of "keep running".
4. **Keep the randomisation** (levels, timing, palette) once a visual mapping
   exists; add darker palettes (night) to the jitter so that day-trained
   recognition survives 3-1 and 6-1.
5. **Evaluate exactly as here**: unseen levels, desynchronised starts, occlusion
   with a textured control and the frame-swap positive reference, and a
   vision-free "always run" baseline, which this study showed is the right null.

## 11. Limits

- One run, three readouts per condition, 20,505 decisions per learner: a longer
  run might find the mapping (one specialist briefly did). The negative result is
  about this budget.
- Occlusion frames come from where the learners actually went (mostly the first
  enemy of each level), so they sample few enemy types.
- The danger readouts are supervised by RAM labels: a proof that the visual code
  supports transfer, not a learning rule.
- The open-loop probe (§6.5) is exploratory and its frames are consecutive (the
  effective N is smaller than 767); its shuffled-label references range from AUC
  0.15 to 0.63 across runs, a measure of how noisy these open-loop numbers are.
  The closed-loop confirmation is the result to cite.
- Post hoc additions (sampled evaluation, log-odds and frame swap) are labelled
  as such in §4.
- "Enemy ahead" labels (`MarioEnv.enemy_ahead`) also count enemies off screen
  vertically (e.g. a Lakitu out of view, present in 4-1 and 6-1); found on
  2026-10-03 evening (PERCEPTION_CREDIT_STUDY §3). Distances and the occlusion
  boxes are unaffected; danger labels on those levels are noisier.
- The occlusion comparison lacked an untrained baseline (correction in §8).

References: Cobbe et al. 2019, ICML (CoinRun); Zhang et al. 2018 (overfitting in
deep RL); Tobin et al. 2017, IROS (domain randomisation); Laskin et al. 2020,
NeurIPS (RAD); Zeiler & Fergus 2014, ECCV (occlusion); Kirk et al. 2023, JAIR
(zero-shot generalisation survey, doi 10.1613/jair.1.14174); Pinto et al. 2018,
RSS (asymmetric actor-critic); Chen et al. 2019, CoRL (learning by cheating).
