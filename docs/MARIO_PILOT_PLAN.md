# Learning in the game — pilot plan, declared before the first run

2026-09-23. **Status: PILOT** (protocol §20). Exploratory by definition: it
decides whether learning happens at all in the benchmark, how fast, how much it
varies between learners, and what a sensible budget is. Pilot results are
labelled as such in every file and plot and are never pooled with confirmatory
estimates (§20.2). Nothing here is a Study II comparison: there is one arm.

It starts only if Gate G passes (docs/GATE_G_PLAN.md §8), and uses the rule
Gate G chose. Code: `src/flymario/training/mario.py`, `scripts/train_mario.py`.

## 1. The model — unchanged from the frozen configuration

MaleCNS v1.0, lin_raw, gain 0.18452, noise σ 0.02, 16 ticks per decision,
warm-up 150 ticks (once, at the start), the eye (`RetinotopicVision`), the
fixed readout (`FixedRandomProjection`, 8 actions, argmax, read-only). Pixels
only reach the policy; RAM only gives reward, termination and metrics.

## 2. What learns

As at Gate G: the 769,342 connections onto the 2,227 motor neurons, sign fixed,
magnitude in [0, 2 × initial count]; one private copy per learner.

## 3. Learners and their twins

Three learners, one per fixed-readout seed (`block_seeds(3)`, the Gate G
readouts). Each has a **frozen twin**: same readout, same noise seed, no
learning. Twins start from the same state and make the same decisions until the
learner's first update, so inside a pair the only difference is learning.
Noise seeds 7000, 7001, 7002. Six brains, six emulators.

## 4. The rule in the game

- Reward: the environment's own, per decision (retro_env.py): forward progress
  past the best point reached, −0.02 per decision, −25 on death, +100 at the flag.
- Advantage: per learner, (r − r̄) / running RMS of (r − r̄), both exponential
  averages over 1,000 decisions — so the advantage has unit size, as ±1 had at
  Gate G.
- Credit across decisions: E ← γ E + e_decision, γ = 0.9 (credit reaches about
  10 decisions, 0.7 s of game, back); Δm = η · advantage · E. A decision's
  consequences in the game arrive later than its reward at Gate G did; this is
  the smallest change that lets a jump be credited with what follows it.
- η = η_GateG × √(1 − γ): the trace sums ~1/(1 − γ) decisions of eligibility
  whose noise adds in quadrature, so one update keeps the size it had at Gate G.

## 5. Episodes

World 1-1, 4 frames per decision (15 decisions per second of game). An episode
ends on death, at the flag, after 6,000 frames (100 s; a run to the flag takes
about 30 s), or after 150 decisions (10 s) without a new furthest point. The
emulator restarts; the brain keeps running (its state is not reset); the eye,
the readout trace and E restart, so no credit crosses an episode boundary.

## 6. Budget and control

Up to 300,000 decisions per learner, stoppable at any time (`make stop`);
checkpoint every 1,000 decisions and on request. A checkpoint holds the
emulators' own save states, so a resumed run continues the same games
bit-identically (verified before launch on the connectome: weights, eligibility
trace, screens and positions identical after a save and a resume).

## 7. What is looked at

Per pair, learner against twin: the furthest x of every episode and its
rolling mean over 20 episodes; flags reached; how episodes end; the action mix.
No pass/fail: a pilot reports what it saw. The questions it has to answer
before any Study II design: does any learner separate from its twin; after how
many decisions; how different are the three pairs; is the budget sensible.

## 8. What the dashboard shows

`make monitor` → http://localhost:8765/runs.html, run `mario_pilot`: every
screen refreshed every 2 s (slow motion — the brain sets the pace: a decision
costs 16 ticks of 170,000 neurons), each learner's latest finished episode
replayed at game speed, and the curves of §7.

---

## Amendment 2026-10-02 — the pilot in the sweep regime (supersedes §1-§4 above)

Declared before the first run. Sections 1-4 above described the continuous
regime and the node-perturbation rule; Gate G passed instead in the sweep
regime with `eprop_readout` (docs/GATE_G_PLAN.md, amendments 2026-09-30 to
2026-10-02), so the pilot uses that. §5-§8 (episodes, budget, what is looked
at, dashboard) stand, with the changes below.

**Model.** One sweep per decision: the frame -> `RetinotopicVision` (ON/OFF
transients from the learner's previous frame) -> reset to rest -> 16 ticks ->
motor spike counts -> `FixedRandomProjection`, 8 actions, read-only -> softmax
over the 8 scores / tau, sampled. No warm-up, no state across decisions.

**Rule** (`src/flymario/training/mario_sweep.py`). Gate G variant B: eta .001,
centred eligibility kappa .2, RMS-normalised multiplicative step, signs and
[0, 2 m0] bounds. Policy factor over the 8 actions,
L_i = sum_a (onehot(a) - pi)_a R_a,i / tau. Credit: E <- .8 E + G_k (stage 1:
gamma .8 learns a two-decision delay; .5 did not), the decision's reward applied
to E at once. Advantage: (r - r_bar) / running RMS over 1,000 decisions. E
restarts with every episode.

**Temperature, decided by a rule fixed now.** If `reversal-adaptive`
(seed sets 281/282) passes its criterion and its phase-1 mean accuracy is at
least .90, the pilot uses tau_adapt .5 (floor 2); otherwise tau stays 2.

**Learners.** Three trios, one per Gate G readout (`block_seeds(3)`): a learner,
a frozen twin (eta 0) and a shuffled-reward twin (the reward's sign flipped at
random, same size). A trio shares its noise seed (7100, 7101, 7102) and its
random draws, so its members act alike until their weights differ. Nine
brains, nine emulators.

**Verified before launch.** `scripts/train_mario_sweep.py --resume-check`:
40 decisions straight vs 15 + checkpoint + fresh emulators + 25 -- weights,
eligibility trace, screens, positions and episodes identical
(`artifacts/runs/mario_sweep_resume_check.json`).

**Reading, declared now (a pilot reports, it does not pass).** Per trio, the
mean furthest x over the learner's last 50 episodes against each twin's last 50.
"Learning seen" = the learner exceeds both twins by more than twice the standard
error of the difference (episodes as samples) in at least 2 of 3 trios. Also
reported: flags, how episodes end, action mix, policy entropy, weight change,
decisions per second. Budget as §6: up to 300,000 decisions per learner,
stoppable and resumable at any point.

---

## Amendment 2026-10-03 — pilot v2 (actor-critic, exploration floor)

Declared before the run. Pilot v1 (amendment 2026-10-02) was stopped by the
owner after 6,681 decisions; its audit (`mario_pilot_audit-20261003.json`,
JOURNAL 2026-10-03) found the learners collapsed onto one action and died at
the first Goomba (4% of episodes past it, frozen twins 69%). v2 changes the
learner, not the brain, the regime, the readout or the benchmark score:

- **Actor-critic** (`algo="actor_critic"`). A critic V(s) = w . phi(s) + b reads
  the motor spike counts of the look (phi = counts / 4), learnt by TD(lambda)
  with a normalised LMS step (0.05). The actor's three-factor update uses the TD
  error delta = r + 0.98 V(s') - V(s) (V(s') = 0 at an episode's end) in place of
  the immediate reward; actor and critic traces decay by gamma lambda = 0.98 x
  0.9 per decision. The critic never chooses actions and sees only the brain's
  response to pixels.
- **Exploration floor** 0.1: one decision in ten is a uniformly drawn button
  (twins share the draws). The gradient factor stays onehot(a) - pi.
- **Learning signal** = (reward + death_extra x died) / 10 with death_extra = -75
  (dying weighs -100 in the learning signal; the reported score and return keep
  the environment's -25).
- Unchanged: eta .001, centred eligibility (kappa .2), tau 2, multiplicative
  RMS-normalised step, bounds [0, 2 m0], sweep regime, fixed readout, 3 trios
  (learner, frozen twin, shuffled-reward twin; noise seeds 7200-7202, action
  draws seeded 20261003), up to 300,000 decisions, stoppable from Fly Command.
- Verified before launch: bit-identical resume with the critic
  (`mario_sweep_resume_check_v2.json`).
- Reading, as for v1: learner above both twins by > 2 SE on the furthest x of
  the last 50 episodes in >= 2 of 3 trios; also the share of episodes past the
  first Goomba (v1: learners 4%, frozen 69%, shuffled 44%).
