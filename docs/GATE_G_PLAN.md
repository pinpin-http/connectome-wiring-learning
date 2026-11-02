# Gate G — learning sanity: the plan, declared before any learning code runs

2026-09-23. Written and committed before a single learning update exists, so
that every criterion below can be checked against what was actually run.
Reconciles the two proposals on the table: node perturbation (this session)
and a precisely specified three-factor rule (docs/GATE_G_OPTIONS.md, which
also supplies the staged task design, the controls and the evaluation rules
adopted here). Neither is chosen in advance: the comparison chooses, by the
rule in §8.

**The owner's condition, 2026-09-23:** more rigour on pre-declared criteria.
Hence §6 — every criterion is run on references that must pass and references
that must fail *before* it is applied to a learning result. A criterion that
does not separate them is fixed and re-declared before any real run.

**Current reading, 2026-09-27:** Gate G remains unresolved. Ordinary node
perturbation passed the v3 held-out direction screen; no connectome learning
has thereby been established. Read the dated amendments below as corrections
to the historical sections, particularly §4's estimator claims and §6's
calibration test. The latest amendment replaces a successful BPTT witness as
a mandatory prerequisite for an exploratory local-rule pilot with measured
exploration, direction and control checks. It does not lower the learning
criterion, change an old verdict, pass Gate G or authorize Mario.

## 1. What Gate G asks

Protocol §29: *pass only if the selected learning mechanism solves a simple
task under controlled conditions.* §12.4: *failure at this stage is a
learning-rule feasibility result and must not be attributed to connectome
architecture.* Mario comes after this gate, not before.

## 2. The task (stage 1 is the gate; stages 2-3 are diagnostics)

- **Cues**: two game scenes from the calibration bank, through the real visual
  interface (RetinotopicVision → the right eye's 3,575 cells) — the pipeline
  Mario will use. The two scenes are fixed in advance: the pair with the
  largest decodability under the Gate E instrument, chosen on seeds 1-12,
  never on evaluation seeds.
- **Actions**: the fixed readout's 8 outputs, of which the task permits two
  (RIGHT, LEFT); the choice is the higher of those two scores. Declared
  restriction of the choice set, not a change to the readout.
- **Reward**: +1 if the action matches the cue's assigned action, −1 otherwise.
  Cues drawn at random, balanced 50/50.
- **Time**: continuous, as in play — one warm-up (150 ticks) at the start, then
  one decision every 16 ticks with the current cue held; the cue changes
  between trials. (Gate D: a change reaches the readout within 8 ticks.)
- Stage 2 (diagnostic): reward delivered two decisions late.
  Stage 3 (diagnostic): the cue→action mapping reverses after learning.

## 3. What is plastic

The 769,342 connections onto the 2,227 motor neurons the readout reads (3.1 %
of all connections); everything else frozen. Transmitter sign fixed (Dale);
magnitude bounded to [0, 2 × its initial synapse count]. No connection is
created or removed. The readout stays fixed and untrainable (§11.2 `fixed`,
enforced).

## 4. The two candidate rules

Both are score-function (REINFORCE-type) estimators of the gradient of
expected reward, with a running per-cue reward baseline R̄ (τ = 50 trials):
`Δw = η (R − R̄) e`, eligibility `e` accumulated over the decision.

- **(A) Node perturbation** (Fiete & Seung 2006). The LIF already adds Gaussian
  noise ξ ~ N(0, σ²) to every membrane each tick. By Stein's lemma the
  expected reward's gradient with respect to motor neuron i's input at tick t
  is E[R ξ_i(t)]/σ², and connection j→i contributes pre_j(t) to that input, so
  `e_ij = Σ_t pre_j(t) ξ_i(t) / σ²`.
- **(B) Spike REINFORCE** — the exact three-factor rule for this stochastic LIF.
  Given the pre-noise membrane v̂, neuron i spikes with probability
  p = 1 − Φ((θ − v̂)/σ); the score is (s − p) φ((θ − v̂)/σ) / (σ p (1 − p)),
  times ∂v̂/∂w_ij = the leak-filtered presynaptic trace. The biased "pre × post
  × reward" form (demonstrated biased in scripts/gates/learning_rule_audit.py)
  is not used.

Known caveat for both: presynaptic activity can itself depend on the noise
through recurrence, which biases the estimate. §6 measures the bias instead
of assuming it away.

Exploration: the frozen model's own noise, σ = 0.02. If the pilot (§7) shows
no usable gradient signal at σ = 0.02, the declared fallback raises the
exploration noise on the 2,227 motor neurons only, recorded as a deviation.

## 5. Evaluation

Learning frozen; 200 test trials per learner on noise seeds never used in
training; 3 independent fixed-readout seeds (`policy.readout.block_seeds`);
2 independent training seed sets. Uncertainty u = max(binomial SE, half the
gap between the two seed sets) — the binomial SE alone understated Gate E's
uncertainty by a factor of five.

Controls, run exactly like the learners:
- **η = 0** — no learning;
- **shuffled reward** — ±1 at random, independent of the action: must not learn;
- **positive reference** — the same pipeline with the two task rows of the
  readout replaced by a supervised logistic regression on the motor features:
  deliberately cheating, it shows the task is learnable from these neurons at
  all (a ceiling, never a result).

## 6. Test the tests first (before any learning run)

1. **Estimator check** (amended 2026-09-23, see §9), small random LIF network
   (a few hundred neurons), two versions: motor neurons feeding back into the
   network (as in the connectome) and not. For 20 random directions d in the
   space of plastic weights, plus the estimator's own direction, compare the
   estimator's projection d·ĝ with a central finite difference of expected
   reward along d, (J(w+δd) − J(w−δd)) / 2δ, both averaged over the same noise
   draws. Required: regression slope of the finite differences on the
   estimates with a 95 % interval containing 1, and correlation ≥ 0.8.
   **References for this check, run first:** finite difference against an
   independent finite difference (must pass: the noise ceiling); a random
   vector and the biased pre × post × reward form (must fail). A rule that
   fails is not run on the connectome.
2. **Criterion check**: the pass criterion of §8 is evaluated on the positive
   reference (must PASS), on η = 0 and on shuffled reward (must both FAIL). If
   it does not separate them, it is corrected and re-declared here, with the
   reason, before §7.

## 7. Pilot (exploratory, seed set 1 and readout seed 1 only)

η for each rule from {1e-4, 1e-3, 1e-2} × a scale set by the estimator check,
≤ 500 trials each. Chosen by learning-curve slope on training trials only.
Never touches the evaluation seeds.

## 8. Pass criterion and choice

A rule passes stage 1 if, on both training seed sets and all 3 readout seeds,
its frozen-learning test accuracy is **≥ 0.75**, exceeds both η = 0 and
shuffled reward by **more than 2u**, and both controls stay **within 2u of
0.5**. Budget: ≤ 2,000 training trials and ≤ 2 GPU-hours per rule.

If both pass: the one reaching 0.75 in fewer trials (median over learners);
both reported. If one passes: that one. If neither: Gate G fails as a
learning-rule feasibility result (§12.4), and the next candidates are those of
GATE_G_OPTIONS.md (e-prop, then a small BPTT reference) — owner's decision.

## 9. What would change this plan

Only a written amendment in this file, dated, with its reason, made before
the step it affects runs.

### Amendments

- **2026-09-23, §6.1, before any code for it existed.** The first version
  compared estimator and finite difference weight by weight on randomly chosen
  connections. Most of those have a near-zero true gradient, so the
  correlation would be dominated by noise and could fail for a perfect
  estimator — a criterion able to fail for the wrong reason. Replaced by
  directional derivatives (every plastic weight moved at once, far more signal)
  and given its own pass/fail references. Correlation threshold 0.9 → 0.8,
  because the noise-ceiling reference now measures what is attainable instead
  of the threshold assuming it.
- **2026-09-23, §4 and §6.1, before the check ran.** Two things, both stated
  before any number exists:
  1. *A prediction, which the check can refute.* In this model the noise is
     added to the membrane and carried forward by the leak. Then node
     perturbation is the exact likelihood-ratio gradient of the whole membrane
     trajectory -- presynaptic spikes are functions of past membranes, which the
     likelihood conditions on -- so it should be unbiased *even with
     recurrence*; §4's caveat was wrong for (A). Rule (B) treats the membrane
     as deterministic between spikes (an escape-noise model), which this model
     is not: it is predicted to be biased.
  2. *The check's noise level.* At σ = 0.02 expected reward is nearly
     piecewise-constant in the weights and a finite difference has almost no
     signal. The pass/fail check runs at σ = 0.1, where it has power;
     unbiasedness is a property of the estimator's mathematics, not of σ.
     σ = 0.02 is run and reported alongside.
- **2026-09-23, §6.1 instrument settings, fixed on the references only (no rule
  estimate had been computed).** A calibration pass measured only the finite
  difference, never a rule. Settings, now frozen:
  - *Network*: 300 LIF neurons (40 sensory, 40 motor), connection probability
    0.10, 70 % excitatory presynaptic neurons (Dale), synapse counts
    1 + ⌊lognormal(0, 1)⌋ (median 2, heavy tail), coupling calibrated to the
    project's 7 % target rate. Its simulator matches `LIFBrain` spike for
    spike (selftest: 0 disagreeing ticks out of 78, 7,845 spikes).
  - *Task*: as §2 in miniature -- a 30-tick warm-up with an unrewarded cue,
    then 3 rewarded decisions of 16 ticks, cues random, the readout trace
    continuous; fixed random readout (the project's class); reward ±1.
  - *Draws and step*: δ = 4 synapse counts along unit directions, 400,000
    episodes per finite difference and per estimate, in chunks of 50,000.
    Calibration: δ = 2 and δ = 8 gave the same slope within error (linear
    regime); standard error ≈ 1e-4 against slopes of order 3e-3.
  - *"The same noise draws"* is read as: J(m+δd) and J(m−δd) share their
    draws (common random numbers); the estimate uses independent draws, so
    shared noise cannot make a biased rule look right. The noise-ceiling
    reference uses independent draws likewise.
  - *Power*: a configuration whose finite-difference reference fails, or where
    a rule's estimate has a median standard error above one third of the
    spread of true slopes, is **inconclusive**, not failed, and is rerun with
    more draws.
  - *Diagnostics, reported, not judged*: the slope at δ/2 on the same draws
    (linearity); node perturbation accumulated over the whole episode instead
    of the decision window (the size of the truncation the rule as used makes).
  - *Correction of the previous amendment's premise*: at σ = 0.02 the finite
    difference had the same standard error as at σ = 0.1 on this network --
    the premise that σ = 0.02 lacks signal was wrong here. So the criterion is
    applied at **both** σ = 0.1 and σ = 0.02, in both network versions: a rule
    passes §6.1 only if it passes all four. Stricter than before, fixed before
    any rule estimate exists.
- **2026-09-23, §2, §5, §7, §8 formalised before §6.2, §7 or §8 ran.** The
  steps below exist in code (scripts/gates/gate_g_task.py, the function
  `criterion()` is shared by the check and the gate) and none has run.
  1. *Cues*: `1-1_jumping` vs `1-1_open`, chosen by the declared rule
     (scripts/gates/gate_g_pair.py, seeds 1-12): decode 1.000, held-out
     p(true scene) 0.937; runner-up `1-1_jumping` vs `4-1` at 0.936; 5 of 45
     pairs decode perfectly. The cue current is each scene's first frame
     encoded from a reset of the eye -- the stimulus of the instrument that
     chose the pair -- held for the whole trial.
  2. *Streams*: for seed set s ∈ {1, 2}, SeedSequence([20260923, s, purpose])
     gives the noise seed, the cue order (5/5 in every block of 10 trials) and
     the shuffled rewards; the test streams (purpose 2) are never used in
     training. Readout seeds: `block_seeds(3)` = 1610069010, 673228719,
     1093961226. The action is RIGHT or LEFT, whichever of those two rows of
     the fixed readout scores higher on the continuous motor trace (τ 5 ms,
     agent.py); a tie goes to RIGHT.
  3. *u*: for each of the candidate and the two controls, the larger of its
     largest binomial SE over blocks and half the gap between its two seed
     sets' means; u is the largest of the three. A block is (seed set,
     readout seed). Pass iff in every block: candidate ≥ 0.75, candidate − η0
     > 2u, candidate − shuffled > 2u, |η0 − 0.5| ≤ 2u, |shuffled − 0.5| ≤ 2u.
  4. *§6.2 check*: η = 0 (one brain per seed set, traces recorded) and
     shuffled reward for each rule that passed §6.1, at the pilot's LARGEST η
     (the most drift), 2,000 training and 200 test trials. Positive reference:
     the Gate D/E logistic decoder (C 0.05, standardised) fit on η = 0's 2,000
     training traces, tested on its test traces. Evaluated as declared: cue 0
     (`1-1_jumping`) → RIGHT, cue 1 → LEFT.
     **Correction fixed now, applied only if that does not separate:**
     counterbalancing. Each block trains one learner per mapping and a block's
     accuracy is the mean over the two. Reason, known in advance: the fixed
     readout already sends each cue to some action, so a non-learning brain
     can sit anywhere between 0 and 1 under one mapping -- but its accuracies
     under a mapping and its reverse sum to exactly 1, so counterbalanced η = 0
     and shuffled controls are exactly 0.5 by construction.
  5. *η scale (§7)*: the η at which one update's RMS change over the plastic
     connections equals 10 × the median initial synapse count, measured with
     η = 0 over 50 trials (seed set 1, readout 1, both mappings). The grid
     {1e-4, 1e-3, 1e-2} × scale then moves a typical connection by 0.1 %, 1 %
     and 10 % of the median count per update: from cautious to aggressive.
     The scale is set on the connectome, not on the §6.1 network: the two
     differ in size and in-degree, so its magnitudes do not transfer.
  6. *Pilot choice (§7)*: per rule, the η with the largest OLS slope of
     training reward on trial index over 500 trials (mean of both mappings).
     §4's fallback triggers if no η of any surviving rule has a slope whose
     95 % interval lies above 0.
  7. *Trials to 0.75 (§8 choice)*: per learner, the first trial at which the
     previous 50 trials' fraction correct reaches 0.75; median over learners;
     a learner that never reaches it counts as budget + 1.
  8. *Engine checks, done before any of this*: the plastic brain equals
     TorchLIF spike for spike at initial and at modified weights
     (tests/unit/test_plastic.py); a resumed run is bit-identical to an
     uninterrupted one, on a small net (tests/unit/test_two_cue.py) and on the
     connectome (artifacts/runs/gate_g_resume_check.json: weights, baselines,
     membranes, rewards, features all identical).
- **2026-09-23, §6.1 power rerun, declared after the first configuration and
  before any rerun.** In `no_feedback, σ 0.1` node perturbation's estimate
  had a median standard error of 5.8e-4 against a true-slope spread of 1.31e-3
  (ratio 0.44 > 1/3): inconclusive by the declared power rule. The rerun of an
  inconclusive configuration: the estimates (the rule's and its own direction)
  are redrawn with 4 × the draws (1,600,000) on fresh seeds; the finite
  differences along the 20 random directions are reused, since they were
  powered (SE 1e-4, reference slope 0.998, r 1.000) and are independent of the
  new draws; the own-direction finite differences are remeasured along the new
  own directions. Same criterion, same thresholds. Also recorded now, from the
  first configuration: the pre × post × reward reference failed on its slope
  (0.048) while its correlation was 0.895, above 0.8 -- the check rejects it
  through its scale, not its direction. This is reported with the result and
  does not change the criterion.
- **2026-09-24, Gate G estimand and next-run protocol, after the estimator
  verdict and before any §6.2, §7, §8 or Mario run.** The existing run remains
  recorded exactly as declared: its finite-difference reference and candidate
  outcomes are valid under the original §6.1 exact-calibration criterion.
  Nothing below changes that verdict or retroactively makes Gate G pass. The
  next estimator artifacts use the versioned `*_v2.json` paths; the earlier
  check, power and verdict files remain untouched.

  Gate G now reports two separate questions:

  1. **Estimator fidelity:** does the rule estimate the mathematical gradient
     at the correct scale? Keep the original exact-unbiasedness test as a
     fidelity diagnostic. For a future practical-equivalence claim, replace
     the “95% CI contains exactly 1” decision with an equivalence test whose
     tolerance is justified and frozen before collecting that run's rule
     estimates. The tolerance is deliberately not selected from the existing
     result. Report slope, interval, correlation, power and noise ceiling even
     when the rule is not calibrated.
  2. **Learning feasibility:** can updates restricted to the declared plastic
     weights improve the task with the fixed readout, against no-learning and
     shuffled-reward controls? A positive slope scale can be absorbed into the
     learning-rate pilot. For eligibility to enter a learning-feasibility
     pilot, the finite-difference reference must be powered, correlation must
     be at least 0.8, and the candidate's powered 95% slope interval must lie
     above zero in every feedback × noise configuration. This is a directional
     screen, not a claim that the estimator is unbiased. The §8 learning and
     control criterion remains the actual feasibility decision.

  The former logistic-regression reference is retained and renamed the
  **observability control**: it asks whether the motor features contain cue
  information. Add a separate **controllability positive control** using
  surrogate-gradient BPTT on the same task, the same plastic edges, the same
  weight bounds and signs, and the unchanged fixed readout. Its forward pass
  uses the hard-spike LIF dynamics; only the backward threshold derivative is
  surrogate. It is trained from the cue/action target, then evaluated with the
  same fixed-readout argmax and held-out streams. A successful BPTT control
  shows that the selected plastic subspace can control the readout. Logistic
  success alone does not. The BPTT optimizer is projected Adam; its training-
  only pilot uses 500 trials on seed set 1, readout 1 and mapping 0, with
  learning rates `{1e-4, 1e-3, 1e-2}`. Choose the rate by highest last-100
  training accuracy, breaking ties by lower last-100 cross-entropy and then
  the smaller rate. Evaluation uses 2,000 training trials per learner and
  200 held-out trials per mapping, over both seed sets, all three readouts and
  both mappings. The control must meet the existing §8 criterion against
  frozen-weight and shuffled-target controls.

  Promote `node_perturbation_full_trajectory` from estimator diagnostic to the
  primary node-perturbation candidate. For each ten-decision rollout, freeze
  weights and accumulate the tick-level score into `S_k = Σ_{t≤k} e_t`. At
  each decision accumulate `(R_k − b_k) S_k`; apply the summed update once at
  the rollout boundary, then clear the score. The network state and readout
  trace continue across boundaries. Ten decisions match the already-declared
  balanced cue block. Re-run estimator checks in all four feedback × noise
  configurations before a learning pilot. If the directional estimate is
  useful but noisy, first test antithetic draws and motor-only exploration,
  declared and checked in the estimator bench before the connectome run.

  The next sequence is: complete the powered four-configuration estimator
  rerun; run the BPTT controllability control; run the full-trajectory node
  perturbation feasibility pilot and its four controlled evaluations; only
  then consider reward-based e-prop and a Mario pilot. Spike REINFORCE remains
  a historical candidate result; it is not promoted for repair because its
  score model does not match the simulator's noise-driven membrane dynamics.
  No Mario pilot is authorized by this amendment. Before any confirmatory
  connectome result, use the §24.1 manifest now implemented for the Gate G
  estimator, BPTT control and two-cue entry points. It records the start and
  final status, git state, runtime/backend, configuration, seed registry and
  input/output hashes under `artifacts/runs/manifests/`. This addition does
  not reconstruct manifests for historical runs or cover unrelated run
  scripts; see `docs/INVENTORY.md`.

- **2026-09-25, BPTT control validity and local GPU safety, before any further
  BPTT or global Gate G run.** The first 2,000-trial BPTT model in the 18-model
  evaluation completed, but its output is not a valid control result: the
  saved vector contains NaNs in 585,992 of 769,342 weights. The three earlier
  pilot checkpoints have the same NaN mask, so their chance-level accuracies
  and learning-rate choice are not interpretable. The run was stopped at
  1/18; no BPTT verdict, Gate G pass/fail, or Mario result follows from it.
  The original §6.1 estimator verdict is unaffected.

  A focused gradient diagnostic locates the non-finite values in
  backpropagation through the 150-tick unscored warm-up: differentiating
  through that burn-in produced the same NaN count, while detaching the warmed
  state before one scored decision produced finite gradients for all weights.
  That single-decision observation does not yet establish that a full
  ten-decision training block stays finite. The BPTT implementation now treats
  burn-in as forward-only and checks gradients and weights for finiteness
  before a checkpoint can be saved. This amendment supersedes the BPTT pilot
  and evaluation claims above; do not reuse their checkpoints as evidence.

  A short GPU smoke run also coincided with a desktop failure. `rocm-smi`
  showed 14.8 GB of 17.1 GB VRAM in use; the kernel logged “Not enough memory
  for command submission” and framebuffer pin failures (`-12`). Hyprland then
  aborted in its OpenGL render path; its safe-mode restart segfaulted while
  tearing down Aquamarine's asynchronous DRM output. System RAM remained
  plentiful and there was no host OOM event. The timing and resource logs make
  VRAM pressure from the BPTT work the likely trigger, though they do not prove
  every link in the chain. The BPTT entry point now caps its PyTorch GPU
  allocator at 50% by default. Do not run another GPU pilot until the detached
  warm-up path has been checked on a short, memory-capped block and the desktop
  has ample VRAM headroom. No global Gate G test or Mario run is authorized by
  this amendment.

  **Follow-up, same date:** the first retry under that allocator cap attempted
  ten decisions and stopped with a PyTorch allocator OOM while requesting an
  additional 940 MiB above its 7.96 GiB process limit. The process exited; GPU
  use returned to its prior ~3.8 GB level, and Hyprland remained running. This
  capped failure did not reproduce the desktop crash, but it shows that a
  ten-decision block does not fit the current cap. The smoke entry point is now
  reduced to one trial in a one-decision block. The longer pilot and evaluation
  runs remain paused pending a memory redesign.

  **One-decision check, same date:** with the 50% PyTorch allocator cap, the
  one-trial/one-decision smoke completed in 4.0 seconds and all 769,342 updated
  weights were finite. A `rocm-smi` sample during the short run showed 86% GPU
  busy and 45% VRAM in use; after it exited, readings returned to 4% busy and
  22% VRAM, and Hyprland remained running. This validates only one update after
  detached burn-in. It does not validate a ten-decision block, learning, or the
  full control. The sampled busy percentage was below 90%, but no hard GPU-busy
  ceiling is configured; the enforceable safety guard remains the 50% memory
  allocator cap. Keep all longer runs paused until block memory is redesigned.

- **2026-09-25, §6.1 and §8 analysis corrections, before any v3 estimator,
  criterion-check, learning-pilot or global Gate G run.** The completed v2
  estimator outcomes remain the outcomes of the protocol declared for v2; no
  v2 file or verdict is edited, and this amendment does not declare Gate G
  passed. The v2 implementation appended one rule-selected direction to the
  20 random directions before fitting each calibration regression. That
  selected direction is a useful, independently validated ascent diagnostic,
  but it is not sampled from the same direction distribution as the other 20
  and can have much greater leverage. Mixing it into ordinary unweighted OLS
  changed reference and candidate conclusions in saved data:

  - For the feedback, σ = 0.02 finite-difference reference, the random-only
    sensitivity fit is slope 1.0054, 95% CI [0.9769, 1.0340], r = 0.9984;
    with the selected point, it is 1.0065 [1.0013, 1.0117], r = 0.99994.
    The reference moves from passing to inconclusive only because the selected
    point was included.
  - For node perturbation without feedback at σ = 0.02, the random-only fit is
    0.6179 [0.2671, 0.9688], r = 0.6573; including its selected direction
    gives 0.8722 [0.8202, 0.9241], r = 0.9924. Full-trajectory node
    perturbation moves from r = 0.7161 to r = 0.9812 in the same comparison.

  These are exploratory re-fits of stored v2 points, not replacement verdicts:
  there are only 20 random directions, and ordinary OLS treats estimated
  projections as fixed even though both axes have Monte Carlo error. For v3,
  the finite-difference noise-ceiling and estimator-calibration regressions use
  only the prespecified random directions. Their slopes, intervals and
  correlations remain descriptive fidelity diagnostics. A rule-selected
  direction is recorded separately: choose it on one Monte Carlo batch, check
  its estimate on an independent batch, and check ascent with a third batch of
  common-random-number finite differences. The estimate and held-out finite
  difference must each have a 95% interval above zero in all four declared
  feedback × noise conditions to enter the learning-feasibility pilot. Do not
  require its scale to be exactly one or use the mixed-direction correlation
  as a learning gate; §7 tunes η and §8 tests actual controlled learning.
  The random-direction finite-difference slope interval is not a prerequisite
  for the pilot; its fit and paired Monte Carlo noise ceiling are reported as
  fidelity diagnostics. Pilot eligibility depends on the separately held-out
  candidate direction and the expected negative controls.
  Treat the OLS interval as descriptive until a future version accounts for
  Monte Carlo error in both axes and the shared estimate batch, using a
  resampling unit that preserves the complete simulation batches.
  The reproducible CPU-only sensitivity table is
  `artifacts/runs/gate_g_estimator_sensitivity_v2.json`, generated by
  `scripts/gates/analyze_gate_g_estimator_sensitivity.py`.

  Any future practical-equivalence claim about scale must state a scientifically
  justified equivalence interval before collecting that run's rule estimates,
  then use an equivalence test whose confidence interval lies wholly inside
  that interval. “The interval includes 1” remains only a test of failure to
  detect a difference from 1, not evidence of equivalence. The interval is not
  to be chosen from v2 or v3 results.

  The script now writes v3 estimator artifacts to separate paths, and the
  connectome pilot refuses the v2 verdict. This only changes future runs.
  To avoid repeating unchanged work, v3 also offers
  `--reuse-v2-random-fd`: it verifies the frozen network seed, draw count,
  direction count and step size,
  reuses v2's paired finite differences on the same 20 random directions, and
  reruns the estimator batches plus fresh finite differences for selected
  directions. The chunk size may differ because those stored finite differences
  are already complete; the original source chunk and the new chunk are both
  recorded. New estimate and selected-direction batches use the new chunk index
  seed schedule, with the plus/minus finite differences still paired by their
  common seeds. This keeps the estimand and total draw count fixed while saving
  the much larger set of repeated random-direction simulations.

  A second issue is the counterbalanced outcome in §8. The two mappings were
  averaged, so a high score for one mapping could mask chance performance for
  its reverse. Future candidate and BPTT-control verdicts must retain and show
  both mapping scores and require each tested mapping to reach 0.75, in
  addition to the existing counterbalanced comparison with controls. The
  no-learning and shuffled controls remain counterbalanced, which neutralizes
  fixed-readout bias. Both mapping evaluations reuse the same held-out cue/noise
  stream; use N_TEST, not 2 × N_TEST, for the binomial uncertainty calculation.
  This is a conservative interim correction. Before confirmatory publication,
  use a paired resampling analysis over held-out trials and learner/readout
  blocks to estimate uncertainty under the shared streams.

  Finally, the full-trajectory node-perturbation implementation accumulates the
  exact score only within each ten-decision, fixed-weight update block and then
  clears it, while membrane state and readout trace continue. That is a
  well-defined short-block test, but it still truncates cross-block credit. It
  is not the same rule as Mario's persistent E ← 0.9E + e. Keep the ten-decision
  fixed-weight result labelled as such. Before claiming that Gate G validates
  Mario's delayed-credit rule, add a distinct persistent/decaying-trace variant
  to TwoCue, freeze its decay and update schedule before its run, and test it
  there first. Do not silently change the current node-perturbation estimator.

  Primary literature consulted for this amendment: Fiete & Seung (2006),
  which accumulates perturbation eligibility over the whole trajectory before
  reward modulation and derives the stochastic Markov case; Miconi (2017),
  which accumulates node-perturbation eligibility through an episode and
  discusses how truncating post-perturbation effects can lose recurrent credit;
  and Schuirmann (1987), which defines equivalence through a predeclared
  equivalence interval. Hodges & Moore (1972) describe the no-measurement-error
  assumption behind ordinary least squares. Links: https://fietelab.mit.edu/wp-content/uploads/2018/12/spike_learning_theory-1.pdf ;
  https://elifesciences.org/articles/20899 ;
  https://doi.org/10.1007/BF01068419 ; https://doi.org/10.2307/2346491 .

- **2026-09-25, BPTT control launch budget, before any global control run.**
  The detached-burn-in check is finite for one scored decision, but a
  ten-decision block exceeded the 50% VRAM allocator cap. The planned
  evaluation contains 18 trained models; the first model took about 55 minutes,
  implying roughly 16.6 hours if that rate holds, far above the original
  two-GPU-hour-per-rule budget. Do not start the global evaluation. First
  redesign the retained state/gradient memory or declare and validate a staged
  compute budget; a one-decision finite smoke is not evidence that the control
  learns.

- **2026-09-25, BPTT memory redesign, before the checkpoint-granularity smoke.**
  Replace checkpointing one complete 16-tick decision with non-reentrant
  checkpointing of each simulation tick. This preserves the hard-spike forward
  equations, surrogate backward, frozen weights within a block, optimizer,
  allowed plastic edges and fixed readout, while recomputing a single recurrent
  step during backward to lower the activation peak. The smoke may use an
  explicitly requested trial count/block length, reports peak PyTorch allocated
  and reserved VRAM, and makes no learning claim. GPU training blocks are paced
  toward 80% mean duty with the existing 50% allocator cap. This engineering
  change does not authorize the 18-model evaluation or alter its criteria; those
  remain paused until a finite ten-decision smoke fits and a prospectively
  budgeted control design is recorded.

- **2026-09-25, BPTT ten-decision memory result and staged feasibility witness,
  before any learning-rate pilot.** The one-decision tick-checkpoint smoke was
  finite, with 2,062 MiB peak PyTorch allocation. A ten-decision block still
  failed in backward at the 7.96 GiB process allocator limit while requesting
  another 940 MiB. During that block ROCm reported 100% GPU busy and total VRAM
  rose to 89%; after process exit usage returned to 29%, temperatures were
  40/42/42 °C, and the Hyprland process remained running. Do not retry a
  ten-decision block on this workstation. Ten separate one-decision smoke
  updates did complete with finite gradients/weights in 30.6 s and a 2,068 MiB
  peak PyTorch allocation. One-second ROCm samples during that and the next
  one-decision smoke reached 100%; a Steam/Wine `GameThread` was also active in
  the process list during the later samples, so their source is confounded.
  The pilot therefore paces every forward simulation tick and every backward
  checkpoint step, rather than relying on pauses after an entire decision.

  To answer the narrower engineering question without an 18-model, multi-hour
  control, add a **staged BPTT feasibility witness**. It trains exactly the
  declared incoming motor weights with the same hard-spike forward model,
  surrogate backward, projected bounds and fixed readout. Truncate gradients
  at each decision boundary (16 ticks remain differentiable); forward state
  still carries between decisions. Use seed set 1/readout 1, 500 training
  trials for each of three rates on mapping 0, choose the rate by the existing
  training-only rule, then train mapping 1 for 500 trials at that rate. Test
  both selected models on the untouched 200-trial streams and compare them with
  frozen weights. The witness passes only if each mapping reaches 0.75 and
  beats its frozen baseline by more than twice the largest binomial standard
  error across those measurements. This is explicitly a one-seed, one-readout
  staged witness: it has no shuffled-target control and does not satisfy the
  full §8 BPTT criterion. Passing permits only the exploratory §7 TwoCue pilot;
  the full Gate G evaluation remains guarded by the full BPTT control. A failure
  is inconclusive about longer-horizon controllability, and this witness says
  nothing about delayed reward or Mario. Pace each simulation tick and each
  backward tick toward 60% mean GPU duty and retain the 50% PyTorch allocator
  cap. The exploratory TwoCue pilot uses the same per-tick pacing. The ROCm
  interface cannot enforce a hard GPU-busy percentage ceiling.

- **2026-09-25, staged-control launch deferred before training.** A
  non-project Steam/Wine `GameThread` was using the GPU during the supervision
  window: sampled busy values were mostly 61–91% and VRAM 63–65%. A 15-minute
  watcher required busy ≤30% and VRAM ≤45% for 30 continuous seconds before
  launch; that condition never held. No BPTT pilot or TwoCue learning run was
  started, and the external process was left untouched. This is an execution
  deferral, not a Gate G result.

- **2026-09-25, Gate G v3 execution pacing, before any complete v3 estimator
  configuration.** The first GPU attempt and a CPU attempt were stopped before
  producing any completed configuration; their manifests remain marked
  `stopped` and are not scientific results. The CPU selftest matched all 624
  reference ticks exactly. The AMD card has a supported power cap but no control
  that directly caps its reported busy percentage. For the next v3 run, cap
  batches at 10,000 draws and synchronize between batches, then leave enough idle
  time for a target 80% average GPU duty. The machine power cap is set to 157 W
  (about 90% of the former 175 W cap). These are execution safeguards; the
  selftest must still show zero spike disagreements on the selected device, and
  every result retains the unchanged total of 400,000 draws per estimate and
  the same four feedback × noise conditions. No learning criterion, random
  direction, or estimator definition changes here.

- **2026-09-25, §6.1 v3 result and next TwoCue candidate, before any §6.2
  criterion or learning pilot.** The complete v3 run re-used the frozen v2
  random-direction finite differences and finished all four feedback × noise
  configurations. Its simulator selftest had zero spike disagreements; the
  finite-difference reference passed, both negative controls failed as
  expected, and the noise ceiling was powered in every configuration. Exact
  scale fidelity remains inconclusive for node perturbation and
  full-trajectory node perturbation; this is not the learning-direction
  decision.

  The separately held-out direction screen passed for ordinary
  `node_perturbation` in all four conditions. It failed for
  `node_perturbation_full_trajectory` only at no-feedback, σ = 0.02: the
  independent estimate interval was [−0.0277, 0.0250], while its held-out
  finite-difference interval was positive [0.0208, 0.0213]. For the next
  immediate-reward TwoCue feasibility stages, select ordinary
  `node_perturbation` as the sole candidate. Its eligibility is reset at each
  decision, so this selection does not test delayed credit or authorize a Mario
  pilot. Keep full-trajectory node perturbation out of the candidate list until
  its low-noise direction screen is repaired with a prospectively declared
  variance-reduction or exploration variant and revalidated. Spike REINFORCE's
  positive direction screen does not promote it: its score model still does
  not match the simulator's additive membrane-noise dynamics.

  This amendment selects the next exploratory TwoCue candidate from the v3
  results; it does not rewrite the v2 verdict or declare Gate G passed. The
  same-readout BPTT controllability control remains required before the §7
  learning pilot. The future TwoCue test of ordinary node perturbation must be
  reported as a one-decision eligibility rule and cannot support claims about
  Mario's delayed-reward credit.

- **2026-09-26, full-duty restart of the staged BPTT witness, before any
  scored trial.** After the prior run was stopped mid-fit without a resumable
  optimizer checkpoint, restart the registered one-decision witness from its
  first fit using gpu-duty=1.0, block-len=1, and the existing
  gpu-memory-fraction=0.50. The owner requested maximum GPU throughput.
  This removes the per-tick pacing sleeps; it does not change the forward
  dynamics, trainable synapses, readout, seed, trial counts, pass rule, or
  scope of the witness. The allocator cap remains in place for VRAM headroom.
  Record this execution configuration in the run manifest. This witness is
  still exploratory and cannot by itself pass the full Gate G evaluation or
  authorize Mario.

- **2026-09-26, block-1 BPTT witness result and block-4 horizon diagnostic,
  before any new BPTT fit.** The registered block-1 staged witness completed
  all eight tasks. The three mapping-0 learning-rate fits ended at 55%, 52%,
  and 58% accuracy over their final 100 training trials; the training-only
  choice was 0.01. On the untouched 200-trial streams, candidate accuracy was
  49.5% for mapping 0 and 50.5% for mapping 1, versus 50% frozen for both.
  Both staged-witness checks failed. This does not pass the BPTT control and
  does not authorize the node-perturbation pilot or Mario. It is inconclusive
  about longer-horizon controllability because gradients were detached after
  every 16-tick decision.

  Before spending another full pilot budget, run a four-trial, one-update
  finite-gradient and memory smoke with block-len=4. Keep the hard-spike
  forward model, surrogate derivative, allowed incoming motor edges, fixed
  readout, seed, and 50% PyTorch allocator cap. Use full GPU duty, record the
  peak allocator use, and stop if gradients or weights are non-finite, the
  allocator reaches its cap, total VRAM reaches 85%, or Hyprland exits. This
  smoke makes no learning claim. If it completes within those limits, run a
  separately tagged block-len=4 staged witness with the same three training-
  only rates, 500 training trials per mapping, untouched 200-trial evaluations,
  and the same pre-existing pass rule. Preserve the block-1 report, checkpoints,
  and manifest under their existing names; the block-4 artifacts must use a
  distinct run tag. Neither staged witness alone constitutes the full Gate G
  control.

- **2026-09-26, block-4 BPTT smoke failure and block-2 diagnostic, before any
  additional temporal BPTT fit.** The four-trial, one-update block-4 smoke
  stopped at backward with `FloatingPointError`: 585,992 of 769,342 weight
  gradients were non-finite. It stopped before an optimizer update and produced
  no model or learning result. The run manifest is marked failed. This points
  to numerical instability in the longer unroll, but the source of the NaNs is
  not yet identified. It was not a VRAM-limit failure; after exit, GPU memory
  returned to 25%, temperatures were 39/40 °C at junction/memory, and Hyprland
  remained alive.

  Do not run the block-4 learning pilot. To bound the shortest tractable credit
  horizon, run one separate two-trial, one-update block-2 finite-gradient and
  memory smoke with the same 50% allocator cap and full GPU duty. It makes no
  learning claim. If gradients and weights are finite and the desktop/resource
  guards remain satisfied, a separately tagged block-2 staged witness may use
  the already registered three training-only rates, 500 trials per mapping,
  untouched 200-trial evaluations, and the unchanged staged pass rule. If the
  block-2 smoke also has non-finite gradients, stop further BPTT pilots until
  the gradient implementation is reviewed; do not advance to node perturbation
  or Mario on the basis of these BPTT results.

- **2026-09-26, block-2 witness outcome; before any further Gate G learning
  run.** The two-trial block-2 smoke completed with finite gradients and
  weights (2,306.8 MiB peak allocated, 3,404 MiB reserved). The separately
  tagged 500-trial-per-mapping witness also completed all eight tasks. Mapping
  0's three training-rate fits ended at 50%, 51%, and 42% on their last 100
  trials; the training-only choice was 0.001. Mapping 1 ended at 49% on its
  last 100 trials. Held-out candidate accuracy was 49.5% for mapping 0 and 52%
  for mapping 1, versus 50% frozen for each. Both preregistered staged checks
  failed. The block-1 staged witness also failed (49.5% and 50.5% held-out vs
  50% frozen); the block-4 smoke produced non-finite gradients. These staged
  outcomes do not establish controllability, but also do not prove that the
  architecture is uncontrollable or change the original §6.1 estimator
  verdict.

  Do not launch the node-perturbation feasibility pilot, the 18-model BPTT
  evaluation, or Mario from these results. Before another learning run, review
  the BPTT gradient path and optimization signal against the fixed-readout
  objective and plastic-edge ordering. Any corrected implementation or new
  optimization schedule requires a separate prospective amendment and a
  preserved run tag. Gate G learning feasibility remains unresolved.

- **2026-09-26, BPTT gradient-source audit; before any new learning fit.** To
  locate the block-4 NaNs without spending another pilot budget, run one
  no-update, one-block gradient audit at horizons of 1, 2, 3 and 4 decisions.
  Each case starts from the same frozen weights, readout, seed and first cue/noise
  prefix, uses the existing 150-tick warm-up, 16 ticks per decision, hard-spike
  forward pass, surrogate derivative and mean cross-entropy objective, and
  stops immediately after backward: no optimizer step, candidate checkpoint,
  learning-rate selection or held-out evaluation. Record warm-up and scored
  forward membrane/trace finiteness and maxima, decision logits/loss,
  per-tick state-gradient finiteness/maxima, final plastic-edge gradient
  finiteness/maxima and an autograd anomaly trace. Keep
  the 50% PyTorch allocator cap and stop if total VRAM reaches 85%, temperature
  thresholds are crossed, or Hyprland exits. These diagnostics only identify
  numerical behavior; they cannot establish learning or alter any Gate G
  verdict. Do not launch another long fit unless the failure source is isolated
  and any proposed training change is prospectively registered.

- **2026-09-26, clipped-state-gradient BPTT candidate; before any further BPTT
  training.** The no-update audit found finite forward states (peak membrane
  magnitude about 16,510) but recurrent state-gradient L2 maxima of
  1.25×10^8, 2.75×10^21 and 2.39×10^37 at horizons 1, 2 and 3. At horizon 4,
  the gradient first became non-finite at tick 19; PyTorch identified
  `SparseAddmmBackward0`, the sparse recurrent multiplication backward, as
  returning NaNs. Test a per-tick global-norm cap of 1×10^9 on membrane and
  spike-state adjoints only. This is about eight times the largest observed
  one-decision state-gradient norm, so that reference horizon should remain
  untouched. First run no-update smokes at horizons 1 and 4 with the same seed,
  weights, readout and first cue/noise prefix. Keep the 50% allocator cap and
  resource stops above. If both have finite gradients and the one-decision cap
  never activates, run a separately tagged block-4 staged witness using the
  already registered training-only learning-rate grid, 500 training trials per
  mapping, 200 untouched held-out trials per mapping and unchanged staged
  pass rule. Apply the same 1×10^9 state-gradient cap throughout training;
  record that intervention in the manifest/report. The result remains only a
  staged engineering control, not a full Gate G pass. If either smoke fails or
  the one-decision gradient is clipped, do not run the staged witness.

- **2026-09-26, gradient audit and clipping-smoke outcome; before any new
  learning fit.** The no-update, unclipped audit completed at horizons 1–4.
  Peak forward membrane magnitude stayed finite at about 16,510. Maximum
  recurrent state-gradient L2 norms were 1.25×10^8, 2.75×10^21 and 2.39×10^37
  at horizons 1, 2 and 3. At horizon 4, the first non-finite state gradient
  appeared on the spike adjoint at tick 19 (52 entries); PyTorch reported
  `SparseAddmmBackward0`, identifying the sparse recurrent backward as the
  failure site. This is gradient explosion through the recurrent graph, not a
  forward-state NaN or memory-cap failure.

  The prospective no-update smoke with the 1×10^9 per-tick membrane/spike
  adjoint norm cap completed at horizons 1 and 4. The cap never activated for
  the one-decision reference (minimum scale 1.0). At four decisions it
  activated 50 times; all post-cap state and plastic-edge gradients were
  finite, the largest post-cap state component was 7.01×10^8, and the plastic
  edge gradient's maximum/RMS were 2.04×10^4 / 306.3. Peak memory was 3,573 MiB
  allocated and 4,968 MiB reserved. Neither smoke applied an optimizer step.

  The condition for a clipped block-4 staged witness is met, but do not start
  it within the owner's 16:10 Paris stop window: it takes about 50 minutes and
  has no safe in-fit checkpoint. Leave the GPU idle and resume with the already
  registered, separately tagged 500-trial-per-mapping witness; this is not a
  Gate G pass and unlocks neither the full evaluation nor Mario.

- **2026-09-27, clipped block-4 staged witness outcome; before any further
  learning fit.** The separately tagged full-duty rerun completed all eight
  declared tasks with block length 4, the 1e9 per-tick state-gradient cap, a
  50% GPU allocator cap, and the unchanged three-rate / held-out design.
  Mapping-0 final-100 training accuracy was 48%, 52% and 46% at learning rates
  1e-4, 1e-3 and 1e-2; the training-only rule selected 1e-3. Mapping 1 reached
  49% on its final 100 training trials. On 200 untouched trials per mapping,
  candidate accuracy was 52% and 49.5%, versus frozen 50% and 50%. The largest
  binomial standard error was 3.535 percentage points; the pass rule required
  each mapping at least 75% and a candidate-minus-frozen difference greater
  than 7.071 points. Both checks failed.

  This is a completed negative result for the staged witness, not a Gate G
  verdict. Its scope remains one seed set, one readout and two mappings, with
  no shuffled-target control. It neither establishes controllability nor proves
  that the full connectome cannot learn. Keep the original §6.1 estimator
  verdict unchanged. Do not launch the 18-model BPTT control, node-perturbation
  feasibility pilot or Mario from this result. Before another learning fit,
  review the BPTT objective and gradient path, clipping intervention,
  train/evaluation alignment and plastic-edge ordering. Register any proposed
  change prospectively and use a distinct run tag. No new learning run is
  authorized by this outcome.

- **2026-09-27, static BPTT/TwoCue implementation review; before any new BPTT
  fit.** Source inspection found the same RIGHT/LEFT readout order, cue currents,
  trace decay, action tie rule, weight bounds, LIF parameter values and
  cue-to-target mapping in `SurrogateBPTT` and the `TwoCue` evaluation path.
  The BPTT hard-spike forward remains a separate implementation: motor rows use
  a sparse COO operator there, while `PlasticLIF` uses a batched CSR operator.
  No direct spike-for-spike parity check for `SurrogateBPTT` was found in the
  unit-test or Gate G scripts inspected. Algebraic similarity is not enough to
  guarantee parity, because sparse summation order can shift membrane values
  near threshold. Before any further BPTT learning fit, perform a no-update
  forward-parity diagnostic using identical fixed weights, cue currents and
  noise samples; compare spikes at every tick and motor traces. This is a
  simulator-validation diagnostic, not a learning result, and does not unlock
  the full BPTT control, node-perturbation feasibility pilot or Mario.

- **2026-09-27, protocol audit and prospective diagnostic sequence; before
  the diagnostics below or any new learning fit.** The owner requested a
  correction of the protocol and a practical investigation of the blockage.
  Preserve every original estimator and BPTT report, including failed and
  invalid runs. New diagnostics use distinct run tags and manifests; they do
  not replace historical verdicts. This amendment records a design, not the
  outcome of a diagnostic that has yet to run.

  **Separate the scientific questions.** Protocol §12.4 and §29 require the
  selected learning mechanism to learn a controlled task. A successful
  surrogate-BPTT reference would establish one constructive route through the
  plastic subspace; an unsuccessful approximate optimizer cannot establish
  that the subspace is uncontrollable. BPTT remains an engineering diagnostic,
  but its success is no longer a mandatory veto on an **exploratory** local-rule
  pilot. This supersedes that pilot prerequisite in the 24–25 September
  amendments. It does not initiate the full §8 evaluation or alter its success
  thresholds: both mappings must reach 0.75 in every declared block and the
  candidate must beat the prescribed controls. The final evaluation's
  references and analysis must be validated before that evaluation runs.
  No learning fit is launched from this amendment alone: first establish
  actual exploration and a useful update direction in the target pipeline.

  **Repair the interpretation of the control check.** Calling
  `criterion(eta0, eta0, shuffled, n)` or
  `criterion(shuffled, eta0, shuffled, n)` makes the negative candidate fail
  algebraically: its improvement over an identical comparator is zero.
  Likewise, scoring one frozen brain under opposite mappings makes its
  counterbalanced accuracy exactly 0.5, even if either mapping is biased.
  These are algebraic sanity checks, not empirical false-positive validation.
  Keep the mapping-specific scores visible. Before calling §6.2 empirically
  validated, add independently generated no-learning and shuffled
  pseudo-candidates, distinct from their comparators, and register their
  generation and decision rule before collecting their outcomes. Retain the
  positive observability reference; do not rename it controllability. The
  `2u` threshold remains an exploratory engineering criterion, not a calibrated
  confidence interval for paired, temporally dependent outcomes. The estimator
  label `powered` denotes the declared SE-to-spread screen, not a formal power
  calculation. Neither correction converts a failed run into a pass.

  **Scope of the evidence already available.** A read-only audit of the
  completed clipped block-4 witness checkpoints found finite weights and no
  plastic weights at either bound. Relative displacement
  `||m - m0||2 / ||m0||2` was 0.00033235 (0.0332%) for the selected mapping-0
  rate 1e-3, versus 0.00335323 (0.3353%) at the largest rate 1e-2; about
  45–46% of weights were unchanged. These observations describe saved weights,
  not a new training result. They motivate measuring effective step size but
  do not prove that a larger step would learn. Separately, a CPU fixture
  exposed a no-update diagnostic bug: `fit(..., apply_update=False)` failed
  to carry the detached state between blocks because that assignment was
  inside the optimizer-update branch. Correct state carry independently of
  whether weights update. This affects multi-block no-update diagnostics;
  it does not invalidate the earlier single-block gradient probes or explain
  the failed training witnesses, which enabled updates.

  **1. Forward parity, with no weight updates.** Compare `SurrogateBPTT` with
  `PlasticLIF` on training seed sets 1 and 2, each at initial weights and at a
  deterministic bounded modified vector. The latter multiplies consecutive
  magnitudes by the repeating factors `[0, 0.5, 1.5, 2]`, independent of cues,
  reward and readout. Use the frozen first readout, 150 warm-up ticks followed
  by 20 decisions of 16 ticks. Draw each noise tensor once and supply that
  identical tensor to both engines; record the seed, noise hash and per-tick
  discrepancies. Verify plastic-edge ordering, signs, magnitudes and unchanged
  weights. Compare every tick's spikes, refractory state, membranes and motor
  trace, and every decision's scores and action. Require zero spike,
  refractory and motor-trace disagreement and finite states for exact parity;
  report membrane rounding errors separately. No backward pass or optimizer
  is used. On GPU keep the PyTorch allocator cap at 0.50 and the existing
  resource stops. A disagreement is a simulator investigation, not evidence
  about learning. CPU fixtures validate the instrument; only the full-graph
  diagnostic establishes parity on that graph and those trajectories.

  **2. Exploration, with no learning.** On training-purpose streams measure
  cue-conditional action counts, score margins and reward variance under the
  frozen noise level, fixed readout and original weights. Report the sample
  counts and per-seed distributions; zero observed switches bounds what this
  finite probe saw, not the true switching probability. A nearly fixed action
  per cue can collapse the advantage after its per-cue reward baseline adapts.
  A toy-network direction check does not establish adequate exploration in
  the connectome. If the target pipeline lacks a measurable signal, declare
  and validate an exploration variant before a local-rule fit; the previously
  proposed motor-only noise fallback is not an implemented capability merely
  because it appears in §4.

  **3. Direction and scale, with no training trajectory.** After validating
  the relevant forward implementation, compare the historical reset-through-
  gradient surrogate at slope k=1 with a separately labelled k=25 surrogate
  whose reset is detached in the backward pass. Forward hard spikes and all
  anatomical, readout and plasticity constraints remain unchanged. Select
  each direction on training streams, then assess it on independent
  training-purpose noise; never use held-out gate streams for selection.
  For each direction, evaluate both signs and a seeded random direction
  against unchanged weights at relative RMS steps
  `{1e-4, 1e-3, 1e-2, 1e-1}`, where relative RMS is
  `RMS(delta m / m0)` before projection. Report the realized displacement
  and bound fractions after projection, the paired loss/accuracy/margin
  changes, and their uncertainty. For a loss gradient, the negative sign is
  the predicted improvement direction. Freeze sample counts and random seeds
  in the diagnostic manifest before evaluation. This is a prospective
  hypothesis about gradient usefulness and scale, not a guaranteed repair or
  permission to launch a fit. A useful direction must improve the actual
  hard-spike forward objective on independent draws; a finite gradient alone
  is insufficient.

  **Estimator wording and sources.** The historical §4 assertion that
  recurrence itself invalidates both scores is superseded. A complete
  Gaussian transition-likelihood score can include recurrence; resetting its
  eligibility while state persists truncates credit. Spike REINFORCE's
  spike-only score remains mismatched to the additive membrane-noise model,
  as already recorded, and is not promoted. Dynamic perturbation learning
  with recurrent networks is derived by Fiete & Seung (2006),
  https://fietelab.mit.edu/wp-content/uploads/2018/12/spike_learning_theory-1.pdf .
  Surrogate derivatives are approximate backward models; see Neftci, Mostafa
  & Zenke (2019), https://arxiv.org/abs/1901.09948 . These sources motivate
  diagnostics, not a claim that their methods have succeeded here.

- **2026-09-27, direction probe execution detail, before its run.** Use training
  seed set 1, first readout, mapping 0, decision windows starting at trials 0
  and 2, one decision each, both declared surrogate variants and all four
  relative step sizes. Each window has one gradient-noise replay and one
  independent training-noise replay, plus both gradient signs and a seeded
  Rademacher direction. Seeds derive from `[MASTER, 1, 20260927, purpose]`,
  purpose 341 for independent noise and 342 for random direction. Retain the
  1e9 state-adjoint cap and 50% GPU allocator limit. These two windows are
  instrument probes: report individual paired outcomes, not a population
  confidence interval or a learning verdict. No optimizer trajectory is run.

- **2026-09-27, diagnostic outcome and remaining implementation scope.** Full
  forward parity was exact across all four registered cases (319,569,920
  neuron-ticks, 80 decisions). At requested relative RMS .1, the narrow,
  detached-reset negative direction lowered independent-replay loss on both
  probed cues (.5974 and .1349), while the original direction increased it
  (.4063 and .1288). Reverse and random controls increased loss. This is
  exploratory selection from four step sizes on two windows, not learning.
  Projection reduced the candidate's realized relative RMS to .0227/.0134,
  versus .1 for random; future comparisons must match realized displacement.
  Independent replay noise shares the frozen initial state. Replicate on
  fresh training windows before proposing a fit; no new fit was launched.
  The §6.2 runner now labels self-comparisons algebraic and `evaluate` rejects
  reports without independent negative validation. The independent control
  design and amended local-pilot launch route still require implementation;
  the current CLI's legacy BPTT checks must not be mistaken for scientific
  evidence or silently bypassed.

- **2026-09-27, fresh-seed replication design, before its run.** Replicate the
  exploratory direction test on training seed set 2, mappings 0 and 1, and
  decision starts 2, 3, 5 and 6 (two windows of each cue in each mapping).
  Keep the first fixed readout, one decision per window, both historical k=1
  and candidate k=25/detached-reset backward variants, the unchanged forward,
  original and independent training-noise replays, the same four relative step
  sizes, 1e9 state-adjoint cap and 50% GPU allocator cap. The earlier .1 step
  is the candidate of interest; retain all four steps to show sensitivity.
  For each step, match the random direction's *realized* RMS after projection
  to the negative gradient's realized RMS. The positive gradient is projected
  at the same requested step, giving the same realized RMS for symmetric
  clipping from m0. Compare paired loss differences against unchanged weights
  separately for every window; report the fraction improving and the mean
  with its range, with no confirmatory p-value. Eight training windows remain
  a feasibility screen, not a learning outcome. Only a consistent independent
  replay advantage over reverse and matched random justifies designing a
  short training fit; a mixed result calls for diagnosing cue/mapping dependence.

- **2026-09-27, short relative-weight controllability control, before its run.**
  The seed-2 direction replication completed on four starts and both mappings.
  At the previously selected requested relative RMS .1, the k=25 detached-reset
  negative gradient lowered independent-noise loss in 7/8 windows; in all 8 it
  did better than reverse and a random direction matched on realized RMS.
  These windows are exploratory and do not establish learning. The next
  engineering reference tests whether gradients aggregate into weights that
  control the *actual continuous* TwoCue pipeline. It is deliberately
  supervised and is not a local biological learning-rule result.

  Freeze the design now: use training-purpose seed 2, first readout, the first
  80 balanced decisions, one-decision hard-forward k=25/detached-reset
  surrogate gradients from frozen-state prefixes, with the existing 1e9
  per-tick state-adjoint cap. Normalize each gradient in relative coordinates
  `r=m/m0` to RMS 1, average its negative over 80 decisions, then normalize
  the resulting direction once. Mapping 1 uses the opposite direction, since
  binary cross-entropy at the same two logits has opposite output gradients.
  For each mapping separately, choose the projected relative RMS request from
  `{0, .01, .03, .1, .3, 1}` by accuracy on 200 continuous, independent
  training-purpose decisions from seed 3; break ties by lower two-class
  cross-entropy and then smaller step. Freeze the two resulting weight vectors.
  Test them on 200 further continuous training-purpose decisions from seed 4,
  paired against original weights and one Rademacher direction per mapping
  matched on *realized* relative RMS after projection. Require per mapping
  at least .75 correct and improvement over both controls greater than twice
  the largest per-arm binomial SE for a provisional positive engineering
  witness. This threshold is exploratory, not an inferential confidence bound.
  Report cue-specific action counts, margins, loss and realized weight movement
  for each arm, plus selection and test results. Seeds 3/4 are only for this
  engineering control; untouched Gate G test streams are not consulted. Keep
  readout, support, Dale signs, bounds, cue currents and hard-spike forward
  fixed. Report interruption status and source hashes; never alter historical
  witness artifacts. Even a pass permits only a local-rule exploratory pilot,
  not Gate G, full evaluation or Mario.

- **2026-09-27, continuous-interface information diagnostic, before its run.**
  The supervised relative-weight engineering control completed and missed its
  provisional witness: selected steps .03 and 1.0 yielded 54% and 56% on the
  independent continuous probe, versus 48.5%/51.5% frozen and 47.5%/47.0%
  matched random, all below 75%. Its selection stream also stayed near chance.
  This does not prove that the anatomy is uncontrollable: averaged gradients
  may cancel useful cue-specific information. Before changing the plastic
  subset or training rule, test whether frozen motor features still identify
  the cue in the actual *continuous* Gate G task. The pair-selection instrument
  reset the eye and brain for each scene, whereas Gate G never resets between
  decisions; that difference may erase the cue signal or change its statistics.
  Run exactly 200 balanced continuous training-purpose decisions on seeds 3
  and 4, 150 warm-up ticks and 16 ticks per decision, same frozen MaleCNS,
  cue currents and motor trace. Fit the existing StandardScaler plus logistic
  regression (C=0.05) to seed-3 motor traces and cue labels; score seed 4.
  Fit a negative label-permutation reference with RNG seed 20260927999 and
  score against seed-4 true labels. Also report both mappings' fixed-readout
  accuracies per seed. The decoder is a cheating observability ceiling, never
  a controllability or Gate G result. A high held-out decode would locate the
  failure in optimization/readout steering; a near-chance decode would show
  that the existing task interface needs correction before more fits. This
  diagnostic uses no Gate G test-purpose stream or weight update.

- **2026-09-27, positive-reference instrument repair, after its failed
  execution and before a result.** The frozen continuous-trace run completed
  200 decisions on each training-purpose seed, but StandardScaler produced
  NaNs before any decode score. The source traces are finite float32 values;
  23 motor columns on the training seed carry decayed subnormal values near
  1e-45. Scikit-learn computed nonzero scales near 1e-46 in float64, then
  cast them to zero while transforming float32 features. Convert the features
  to float64 before fitting and transforming, in both this diagnostic and
  `gate_g_task.positive_reference`. The decoder architecture, labels, split,
  hyperparameters and fixed brain stay unchanged. Reuse the checkpointed
  no-update traces; the failed attempt has no scientific decode outcome.

- **2026-09-27, continuous cue timescale diagnostic, before new runs.** With
  the original 16-tick continuous task, frozen motor traces permitted 100%
  in-sample logistic decoding on seed 3 but exactly 50% on independent seed 4;
  a shuffled-label decoder scored 50.5%. Splitting either seed into temporal
  halves also yielded about chance (45–54%), and the two seeds' cue-contrast
  vectors had cosine 0.028. A stored-trace lag scan 0–8 decisions was
  exploratory and showed no stable cross-seed decoding (49–53.4%). These
  results disqualify the reset-selected pair as a demonstrated positive
  observability reference in the continuous 16-tick task. Before considering
  another learning fit, repeat the same no-update, seed-3 train/seed-4 test,
  200-decision decoding instrument at 32 and 64 ticks per decision, retaining
  the same 150-tick warm-up, cues, fixed brain, decoder, negative shuffled-label
  reference and first readout. Report both durations separately and retain all
  failed outcomes. A cross-seed decode at least .75 with the shuffled control
  within two binomial SE of .5 would make that duration an exploratory task
  candidate, not a retroactive Gate G pass; any task change still requires a
  new declared criterion and controls before learning.

- **2026-09-27, same-sequence trace versus spike-count diagnostic, before its
  run.** The planned 32- and 64-tick continuous trace tests scored 49.5% and
  55% across training-purpose seeds, with shuffled-label references 48.5%
  and 46%. Longer exposure alone did not restore the near-perfect decoder
  measured after per-scene reset. The pair-selection instrument also used
  the *sum of motor spikes over the decision*, whereas Gate G's fixed readout
  uses a rapidly decaying motor trace. Isolate this measurement difference at
  the original 16 ticks: on the same 200 continuous decisions, seeds 3 and 4,
  collect both the decision's per-motor spike count and the readout trace,
  without changing weights, cues, noise or timing. Train the same C=.05
  standardized logistic decoder on seed 3 and evaluate seed 4 for each feature
  type; include a fixed-seed shuffled-label reference for each. Compare both
  feature types' held-out accuracy with their own negative control. If counts
  decode but traces do not, a temporal readout/interface mismatch explains
  the failed positive reference. If both fail, resetting each scene during
  pair selection is the stronger mismatch. This is diagnostic, not a gate
  pass, and does not touch test-purpose streams.

- **2026-09-27, continuous ten-scene screen, before its run.** The paired
  16-tick feature check scored 50% on independent seed 4 for both motor trace
  and spike count; each fitted decoder scored 100% on seed 3, as did the
  shuffled-label fits. Thus the reset-selected pair's in-sample separation
  does not transfer in the actual continuous task. Test whether any of the
  ten existing calibration scenes gives a stable continuous *motor-trace*
  contrast before attempting another fit. Freeze weights and gain, use 150
  warm-up ticks and 16 ticks per decision. On each of training-purpose seed
  sets 3, 4 and 5, present 400 decisions in blocks containing one of each
  scene in random order, with no reset between decisions. Fit the same
  float64 StandardScaler + C=.05 logistic decoder on seed 3 for each of the
  45 scene pairs; score seed 4 and select the pair with highest accuracy,
  breaking ties by mean true-class probability then alphabetical pair.
  Evaluate only that frozen winner on seed 5, alongside a decoder fitted
  after shuffling seed-3 labels with seed 20260927999. A seed-5 score >=.75
  and a shuffled score within two binomial SE of .5 identifies an exploratory
  candidate. Because ten-scene context differs from the actual two-cue
  stream, even that candidate requires a fresh two-cue validation before
  learning. If no pair meets the screen, report that limitation rather than
  inferring an anatomy-wide impossibility. Use only training-purpose streams;
  preserve all previous pair-selection and Gate G results.

- **2026-09-27, matched assay positive control, before its run.** The frozen
  ten-scene screen selected `1-1_obstacles`/`1-1_open` at 62.5% on seed 4,
  but confirmation on seed 5 was 47.5%, versus 50% for shuffled labels. Its
  strongest selection score might be a 45-pair winner's curse. To establish
  that the *same* selection/confirmation decoder can detect a stable
  continuous cue signal, use the saved frozen features, and inject a synthetic
  cue-locked offset into exactly one existing motor-trace coordinate for every
  `1-1_open` decision across all three seeds. Choose the coordinate solely by
  largest seed-3 pooled standard deviation, and set the offset to four times
  that standard deviation. Rerun all 45 fits, seed-4 selection and seed-5
  confirmation; also score the selected pair with shuffled seed-3 labels.
  Do not tune the injected size or coordinate to the outcome. A seed-5
  accuracy >=.75 against a near-chance shuffled reference validates that this
  assay can detect a sufficiently strong persistent signal; it does not
  establish that such a signal exists in the unmodified brain.

- **2026-09-27, restricted-capacity decoder correction, before its run.**
  The declared four-SD synthetic control failed too: it selected the same pair
  at 62.5% and confirmed at 47.5%, while shuffled confirmation was 50%.
  Therefore the 2,227-feature C=.05 logistic assay is not a validated
  positive reference for a *single stable feature*; its seed-3 100% fit was
  high-dimensional memorization. Retain these results, and run one new,
  fixed low-capacity assay on exactly the saved 400-decision traces. For each
  pair, fit StandardScaler in float64 on seed 3, exclude training coordinates
  with range <=1e-6, choose the one remaining motor coordinate with largest
  absolute between-cue difference in standardized means (ties: lowest index),
  then fit C=.05 logistic regression on that single coordinate. Select the
  pair by seed-4 accuracy, then true-class probability, then alphabetical
  order; confirm on seed 5. Apply *the identical* 45-pair procedure to both
  original traces and the already declared four-SD synthetic injection,
  and assess seed-5 shuffled-label references for each winner. A synthetic
  confirmation >=.75 with shuffle within two binomial SE of .5 validates
  sensitivity to a stable one-coordinate signal. A real-brain result below
  .75 then only says this restricted assay found no qualifying pair, not that
  all distributed cues or all task designs are unreadable. This is still
  exploratory; the original criterion and full Gate G remain untouched.

- **2026-09-27, owner-authorized Gate G task redesign, before its run.** The
  owner explicitly allows a solution for Gate G beyond the original written
  choices. The original continuous 16-tick task and its failed results remain
  unchanged. As a separately named Level-A diagnostic candidate, test an
  *episodic* two-scene task: reset neural state before each cue, hold the real
  encoded scene through the same 150-tick warm-up and 16-tick decision, retain
  the frozen wiring and first fixed readout. This matches the condition under
  which the original pair was selected; it does not claim to model continuous
  Mario play. Use 24 fresh training-purpose noise seeds 101–124; seeds 101–112
  fit two supervised observability decoders and seeds 113–124 test them. On
  each exact simulation, collect both total motor spikes during the decision
  and the five-ms motor trace. Fit the previously used full C=.05 logistic
  decoder and the now positively calibrated one-feature C=.05 decoder,
  separately for each measurement, with a shuffled-label reference. Report
  the unchanged fixed-readout accuracy for both action mappings. An
  independent-seed accuracy >=.75 on trace with a shuffled reference near
  .5 establishes a candidate compatible with the existing motor readout;
  count-only success instead requires a prospectively changed temporal
  readout before any learning test. This checks observable task information,
  not whether weights can learn or whether Gate G passes. Any subsequent
  supervised or local update must get its own design, budget and controls.

- **2026-09-27, episodic supervised motor-weight direction pilot, before its
  run.** The proposed episodic reset condition gave 100% held-out cue decode
  from one motor-trace coordinate on fresh seeds 113–124, versus 37.5% after
  shuffled labels; the unchanged fixed readout scored 45.8%/54.2% for the
  two mappings. This supplies an observable task but not controllability.
  On seeds 101 and 102, compute one hard-forward k=25/detached-reset
  surrogate gradient for each cue, each after a fresh 150-tick no-gradient
  warm-up and over only its 16-tick decision. Cap state adjoints at 1e9,
  normalize each gradient in relative weight coordinates, average their
  negative directions, and normalize once. For mapping 1 use the opposite
  direction. Keep the original motor-edge support, signs, bounds, all other
  weights and random readout fixed. Select one relative RMS request from
  `{0,.03,.1,.3,1}` per mapping by accuracy, then CE, then smaller step on
  episodic seeds 125–136. Evaluate the frozen selections on fresh seeds
  137–160, paired against unchanged weights and one random direction matched
  to each realized relative RMS after projection. Report cue-specific action
  counts, accuracy, CE and movement. A provisional engineering witness needs
  >=.75 in both mappings and >2 largest-arm binomial SE improvement over
  both controls on the 48-decision confirmation set. This is supervised
  internal-weight controllability, not a local biological rule or Gate G
  passage. The test explicitly catches gradients that help from the original
  warm-up state but fail when changed weights alter the next reset/warm-up.

- **2026-09-27, separate cue-direction search, before its run.** The
  episodic one-step pilot selected .03 for both mappings but failed its
  confirmation checks. Mapping 0 improved from 31.25% to 47.92%, yet it
  still chose LEFT on 22/24 cue-0 trials while improving cue-1 LEFT to
  21/24. The average gradient was therefore mostly an action-bias change,
  not a reliable cue-dependent policy. To test whether averaging opposed
  cue directions destroyed a useful combination, recompute the four
  registered reset-episode gradients (seeds 101/102 × two cues), average
  separately within cue after per-gradient relative-RMS normalization, and
  normalize each cue's negative direction. For mapping 0 only, screen the
  nine combinations `alpha_cue0, alpha_cue1 ∈ {0,.03,.1}` using a single
  projected update `m=m0*clip(1+alpha0*d0+alpha1*d1,0,2)` on independent
  reset-episode seeds 125–136. Choose by accuracy, then CE, then smaller
  realized relative RMS, then lexicographic coefficients. Confirm on seeds
  137–160 against unchanged weights and a Rademacher direction matched to
  realized RMS. Report cue-specific actions; >=.75 and >2 largest-arm SE
  advantage over both references is a provisional mapping-0 engineering
  witness only. This is a small causal search in a two-dimensional
  surrogate-derived subspace, not a Gate G result or a trained local rule.

- **2026-09-27, outcome and boundary.** The two cue-specific directions were
  nearly orthogonal (cosine −0.0328). Independent selection chose cue-0
  coefficient .03 and cue-1 coefficient 0. On 48 fresh reset episodes,
  mapping-0 accuracy was 56.25% versus 31.25% frozen and 39.58% for a
  matched-RMS random direction. Both improvements exceed twice the largest
  arm binomial SE (7.16 percentage points), but accuracy remains below the
  75% engineering threshold. The update moved cue-0 RIGHT decisions from
  3/24 to 21/24 while moving cue-1 LEFT decisions from 12/24 to 6/24:
  a global action-bias correction rather than two-cue discrimination. The
  original continuous Gate G remains failed, the episodic version is an
  observable but unsolved diagnostic, and neither the supervised control nor
  either local rule passes Gate G. Do not launch Mario training on this basis.
  The next test must address cue-dependent control under the fixed readout,
  and its criterion/controls must be declared before it runs; simply adding
  more of this same one-step direction is not supported by these data.

- **2026-09-30, owner-authorized model revision (sweep regime) and a new local
  rule; declared and committed before the explore, reference and confirm stages
  run.** The owner authorized bypassing the earlier Gate G protocol and revising
  the model ("tu peux bypasser le protocole", "tu peux revoir entièrement le
  modèle", "l'essentiel est de documenter tout ça et me faire un truc qui
  fonctionne et apprend bien"), while keeping the study's rules. Everything
  below is new and separately named; no earlier result or verdict changes.

  *Diagnosis (exploratory, no weight update;
  `scripts/diagnostics/gate_g_dynamics.py`, reports
  `artifacts/runs/gate_g_dynamics_{switch,decode,smooth}-20260930.json`).* At
  the frozen operating point, a network driven from rest by cue A and switched
  to cue B keeps representing A: in every 16-tick window up to 128 ticks after
  the switch, the switch index f is ~0 at depth >= 2 and at the motor neurons
  (0 = still looks like the old cue, 1 = looks like a fresh brain on the new
  cue), and every motor neuron that separates the switched groups does so with
  the OLD cue's sign. The continuous Gate G task therefore never presents the
  current cue to the motor neurons, which explains its failures independently
  of any learning rule. Lower gains, the in-degree-normalised transforms,
  short-term depression, spike-frequency adaptation and a graded-rate stand-in
  either keep the lock or lose the cue before the motor layer (same report).
  From rest, in contrast, the motor spike counts of the first 8-16 ticks
  identify the scene on held-out noise far better than the Gate E window
  (150-166) of the same brains (decode report), and neighbouring gameplay
  frames give neighbouring responses (smooth report).

  *Model revision, the sweep regime (`src/flymario/training/sweep.py`).* Each
  decision: reset membranes, refractory counters and last spikes to rest; hold
  the encoded frame for 16 ticks (`timing.ticks_per_decision`); the motor
  response is each motor neuron's spike count over those ticks; the unchanged
  `FixedRandomProjection` scores it. Connectome, transform, gain, noise,
  threshold, interface and projection are unchanged; noise generators are not
  reset. No state crosses decisions (a glimpse model): this is a modelling
  choice with consequences for the architecture comparison that must be
  measured on the nulls, and the Mario loop (`agent.py`) is NOT yet switched.

  *Rule `eprop_readout`.* Exactly as specified in the module docstring:
  eligibility e_ij = sum_t psi_i(t) eps_ij(t) (presynaptic trace cleared by a
  postsynaptic spike; fast-sigmoid surrogate of width 0.5 at the post-noise
  membrane), neuron-specific learning signal L_i = g (R_RIGHT,i - R_LEFT,i) /
  tau from the fixed readout, g the softmax policy-gradient factor
  (1[a=RIGHT] - P(RIGHT)) (r - b_cue) with a sampled action, r = +-1, per-cue
  running baseline (rate 0.1), tau = 2 count units; raw update sign_ij L_i e_ij
  divided by its running RMS (decay 0.95); m <- clip(m + eta m0 * that, 0, 2 m0).
  Only the 769,342 motor-incoming magnitudes change; signs, support, bounds and
  readout stay fixed. `tests/unit/test_sweep.py` shows e_ij equals the autograd
  surrogate gradient when motor neurons feed nothing back, that a sweep carries
  no history, and that a resumed run is bit-identical. It is a supervised-
  quality credit signal broadcast from the readout (symmetric e-prop), not a
  claim of biological plausibility for that broadcast.

  *Stages (`scripts/gates/gate_g_sweep.py`), each on its own seed sets
  (`sweep.streams`, MASTER 20260930):* **explore** (201, 202): variants A
  (eta .001), B (eta .001, centred eligibility kappa .2), C (eta .001, tau
  adapted to .5 RMS of the score difference, floor 2), D (eta .002, kappa .2,
  adaptive tau), plus supervised, frozen and shuffled references; 3 readouts
  (`block_seeds(3)`) x 2 mappings x 2 seed sets = 12 learners each; 800
  training trials; held-out evaluation at 200/400/600 (50 trials) and 800 (100
  trials). Selection, fixed now: highest mean held-out accuracy at 800; ties
  within .01 -> higher minimum -> fewer options. **reference** (211, 212): the
  criterion is applied first to supervised (must PASS), frozen, shuffled and
  anti-reward (must FAIL); 800 trials, 200 test trials. **confirm** (221,
  222): the selected variant with matched frozen, shuffled and supervised
  learners under its own rule configuration; 800 trials, 200 test trials; run
  only if the reference stage validated the criterion.

  *Pass criterion (all stages).* Per cell (readout, mapping) and seed set: (i)
  held-out accuracy >= .75 in all 12; (ii) exceeds its frozen and its shuffled
  twin by more than 2u in all 12, u = max(binomial SE of either difference in
  either seed set, half the seed-set gap of the candidate's accuracy); (iii)
  run-level: mean(shuffled - frozen) within 2 SE of 0. Frozen accuracy per
  mapping is not expected at .5 (the fixed projection may prefer one action per
  cue; averaged over mappings it is .5 by construction, so that average is not
  used as evidence). `2u` is a readability threshold, not a calibrated bound.

  *What a pass would and would not mean.* It would establish that a local
  three-factor rule on the motor synapses learns a two-cue visual association
  through the fixed readout, on held-out noise, in the sweep regime. It would
  not establish competence in continuous play, on Mario, or any architecture
  effect; the sweep regime's cost (no memory across decisions) is a declared
  limitation. Stage 2 (delayed reward) and stage 3 (reversal) are not part of
  this pass.

- **2026-10-01, stage-1 follow-up tests before Mario; declared and committed
  before any of them runs.** The owner asked for the three tests that predict
  whether the Gate G learner will survive Mario's difficulties. Same task,
  sweep regime and rule as the pass (variant B: eta .001, kappa .2, tau 2,
  RMS-normalised multiplicative step), three fixed readouts x two mappings x two
  seed sets = 12 learners per condition, held-out greedy evaluation on the
  'test' streams of each seed set, the Gate G criterion (i)-(iii) unchanged.
  Script `scripts/gates/gate_g_stage1.py`, pipeline
  `scripts/gates/gate_g_stage1_pipeline.sh`; extensions in
  `src/flymario/training/sweep.py` are off by default, so the Gate G path is
  unchanged (`tests/unit/test_sweep.py`, 5 tests).

  1. *Delayed reward* (Mario rewards a jump later). The reward of decision k
     arrives at decision k + 2. Reward learners keep an eligibility trace across
     decisions, E <- gamma E + G_k, and update with (r - b) E, b a global running
     mean: they are not told which decision a reward belongs to. The supervised
     reference gets exact delayed credit (the stored G of its own decision).
     Explore on seed sets 231/232 with gamma .5 and .8, 1,200 trials, 100 test
     trials; selection: highest mean, ties within .01 -> higher minimum ->
     smaller gamma. Confirm on 241/242 with the selected gamma, 1,200 trials,
     200 test trials, controls frozen and shuffled reward.
  2. *Reversal* (real plasticity, not a frozen bias). Seed sets 251/252: 800
     trials on mapping m, then 800 on 1 - m, learning state carried over.
     Phase-2 controls: FROZEN2 (stops learning at the switch) and SHUF2
     (shuffled reward after the switch). Criterion on the reversed mapping.
  3. *Generalisation to unseen frames* (no Mario frame ever repeats). Seed sets
     261/262: each trial shows one frame of the scene's 48-decision gameplay
     sequence, encoded in order (ON/OFF transients included); training frames
     0-23, test frames 24-47 (frames 0-23 also scored, 100 trials). 800
     training trials, 200 test trials.

  In every test the supervised reference is read first: if it fails (i), the
  test is uninformative (the task is not learnable in that form), not a
  failure of the rule. These are engineering predictors for the Mario pilot;
  none changes the Gate G verdict.

- **2026-10-02 (prepared 2026-10-01), delayed-reward outcome and a separately
  named longer-budget test; declared before it runs.** `delayed-explore`
  (seed sets 231/232, 1,200 trials) selected gamma .8: 95.9% held-out mean (min
  91%, 12/12 >= .75) versus 74.4% for gamma .5, frozen 50%, shuffled 50%,
  exact-credit supervised 98.9%. `delayed-confirm` (241/242) gave 92.1% (min
  76%, 12/12 >= .75; supervised 98.7%) but **failed (ii)** in 2 of 12 cells,
  both readout 1093961226 on seed set 242: 0.79 against a frozen twin already at
  0.735, and 0.97 against a shuffled twin that drifted to 0.76 (u inflated to
  .105 by the seed-set gap). That verdict stands. Its training curve was still
  rising at 1,200 trials (86% -> 91% greedy in the last blocks), so the next
  question is budget, asked on fresh data rather than by re-scoring:
  `delayed-long`, seed sets 271/272, gamma .8, 2,400 training trials, 200 test
  trials, the same conditions and criterion. A pass would say the delayed-reward
  learner meets the criterion with twice the budget; a fail would say the
  failure is not only a budget effect. Neither changes the delayed-confirm
  verdict.

- **2026-10-02, frame observability and an adaptive-temperature reversal test;
  the test declared before it runs.** Diagnostic (exploratory,
  `scripts/diagnostics/gate_g_frames.py`, `gate_g_frames-20261002.json`): a
  frozen-network decoder of motor counts after one sweep identifies the scene on
  frames 24-47 after training on 0-23 at 82.6% (shuffled 45.6%; neighbouring
  frames 94.5%; same frames, new noise 99.0%), against 66.7% for the same decoder
  on the injected lamina current itself. The information generalises; the
  stage-1 learners (rule 59.8%, supervised 64.5%) did not extract it. For the
  reversal failure, one factor changes from rule B: the softmax temperature
  follows the running RMS of the score difference (tau_adapt .5, floor 2), so a
  confident policy keeps exploring. `reversal-adaptive`, seed sets 281/282,
  otherwise identical to `reversal` (800 + 800 trials, 200 test trials, phase-2
  controls FROZEN2 and SHUF2, criterion on the reversed mapping). The
  phase-1 evaluation also tells whether the change costs anything on the
  original task.

- **2026-10-03, exploration-floor reversal test; declared before it runs.** The
  Mario pilot v1 audit and the reversal failures share one mechanism: a
  saturated softmax zeroes the policy gradient (onehot - pi), so a learner that
  always chooses one action never sees evidence for the other. One change to
  rule B: 10% of training actions are drawn uniformly (`RuleConfig.epsilon`),
  the gradient factor unchanged. `reversal-floor`, seed sets 291/292, otherwise
  identical to `reversal` (800 + 800 trials, 200 test trials, phase-2 controls
  FROZEN2 and SHUF2, criterion on the reversed mapping, greedy evaluation). It
  runs alongside the Mario pilot v2 rather than before it (owner's request to
  proceed directly); its outcome is reported, not used to tune the pilot.
