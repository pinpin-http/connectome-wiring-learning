# ACTIVITY_MATCH_V1: frozen protocol (version 1)

**Frozen 2026-10-20T08:40Z**, before any main-campaign simulation (no main bank had been simulated; only the
label-free calibration and the development trial existed). The sha256 of this file and of
`configs/activity_match/protocol.yaml`, and the freeze commit, are recorded in `RUN_LOG.md` and `JOURNAL.md`.
A local, dated freeze in a git repository; not a public preregistration.

Authority: `docs/activity_match/MISSION.md` (owner's text). This document fixes every choice left open there, before
any main-campaign simulation. After the freeze, any scientific change creates version 2 and, if scores have been
seen, is labelled exploratory. Configuration: `configs/activity_match/protocol.yaml` (the launcher refuses the main
campaign unless it says `protocol_status: frozen` and `calibration.verification_passed: true`).

## 1. Question and the four readings it must separate (mission §1)

Does the connectome's orientation-decoding deficit of RETINO_ALIGN_V1 persist when connectome and nulls run at
comparable mean output activity? The intervention is a positive scalar gamma on the global synaptic gain, per graph.
A negative result or an impossible matching is a valid outcome; no setting is searched to make the connectome win.

## 2. Graphs (mission §5)

Connectome `data/connectome/malecns.npz` (sha256 `6befb67a0278…`) and the 40 family-c nulls `nullin:0-39`
(`depth_stratified_lateral_instrength`, files identical to RETINO_ALIGN_V1's validated ones; per-file sha256 in the
main manifest). Development only, never analysed: `nullin:100`, `nullin:101`. No graph is dropped for being hard to
match or for its score. Results are conditional on this graph set and a single biological connectome.

## 3. Simulator (fixed, verified in AUDIT.md and `artifacts/activity_match/controls/controls.json`)

RETINO_ALIGN_V1's `FrozenNet` unchanged: lin_raw integer operator, LIF decay 0.8466 (tau 20 ms, dt 3.33 ms),
threshold 1, reset 0, one refractory tick, noise 0.02 N(0, 1) per neuron and tick, bias 0, 16 ticks from a zeroed
state per image, static input convention (reset + two identical frames) on the right eye's 3,575 lamina cells, readout
= the 2,227 `datasets.motor_index` neurons, features = their spike counts (uint8, max 8). Gain = gamma x 0.18452; the
gain multiplies `W @ s` only. No plasticity; the operator digest is checked before and after every lot.
Noise key (study seed 20261019, noise seed, example uid): identical across graphs and conditions.

## 4. Conditions (mission §5)

| condition | gamma | role |
|---|---|---|
| C0 | 1 for every graph | the V1 setting, re-measured on new data |
| CL | per graph, calibrated to the low target | sensitivity to the activity level |
| CM | per graph, calibrated to the intermediate target | **primary comparison** |
| CH | per graph, calibrated to the high target | sensitivity to the activity level |

All inputs ordered (no retinotopic permutation). Two main noise seeds, 31001 and 31002. Full plan 41 x 4 x 2 = 328
extraction lots. With per-graph gammas the nulls no longer share the connectome's absolute in-strengths: CM asks a
different question from C0, which is why C0 is always reported beside it.

## 5. Calibration (mission §7) — label-free, completed before this freeze

- Banks from the frozen generator: CAL_FIT `am_cal_fit` (128 images, 8 per orientation x brightness cell, seed 21101),
  CAL_CHECK `am_cal_check` (128, seed 21102). Their labels are dropped when they are encoded and never saved; the
  calibration code receives images and uids only. Calibration noise seeds 21001, 21002.
- Rate r = readout spikes / (images x 2,227 x 0.05328 s), silent neurons included; mean over the two seeds.
- Validity of a point: finite diagnostics, r > 0, mean active fraction >= 0.01, mean fraction of outputs at 8
  spikes (>= 0.9 x the coded maximum 8) <= 0.05 (DECISIONS D3).
- Grid {0.5, 0.75, 1, 1.5, 2}; if it has no common range, the declared extension (D5) to 19 points from 0.02 to 16.
- Common range [max over graphs of their lowest valid rate, min over graphs of their highest valid rate];
  targets at 0.25, 0.50, 0.75 of it (CL, CM, CH).
- Bounded search per graph (D4): at most 9 extra gamma evaluations in total; bracketing only on branches where the
  evaluated rates are monotone; selection = within 2 % of the target and closest to gamma = 1, then smallest gamma.
- Independent check on CAL_CHECK with the selected gammas unchanged: |r - target| / target <= 0.10 on the mean of the
  two seeds, for every graph and every target (per-seed errors shown).

**Calibration result (filled at the freeze):** see §12.

## 6. Main data (mission §9)

New banks from the frozen V1 generator (setting C): train `am_train` 1,024 (64 per cell, seed 31101), validation
`am_val` 512 (32, seed 31102), test `am_test` 2,560 (160, seed 31103). Checked before extraction
(`run_am.py validate`): 16 balanced cells per partition, unique uids, image digests and latent noise seeds, no
overlap with V1's banks or the calibration banks, uid position uncorrelated with either label. The same images enter
every graph and condition; no state is kept between images. The test set never chooses gamma, lambda, the outputs,
the targets or the generator.

## 7. Decoder (mission §11)

RETINO_ALIGN_V1's ridge, unchanged: one-hot targets, normalised loss (1/n)||ZW + b - Y||^2 + lambda ||W||^2, bias not
penalised, standardisation fitted on the training subset only (constant columns: SD 1, zero weight), lambda in
{1e-4, 1e-3, 1e-2, 0.1, 1, 10} chosen on validation accuracy, ties to the largest lambda, no refit on train +
validation, argmax decision. One SVD per subset serves every lambda and both tasks. Budgets 8, 16, 32, 64, 128 per
orientation class (32-512 images), 10 nested draws balanced over the 16 cells (seeds 8201-8210), shared by every
graph and condition. The 512 validation labels are an additional resource beyond each budget. A separate decoder per
graph, condition and noise seed.

## 8. Metrics (mission §13)

Proportions in computation, percentage points in tables. M(g, c) = test accuracy averaged over 5 budgets x 10 draws
x 2 seeds (per-example correctness on the 2,560 test images; chance 0.25).

- **Primary:** G_CM = M(connectome, CM) - mean_k M(null_k, CM), orientation.
- Secondary, declared: G_C0, G_CL, G_CH; T_real = M(connectome, CM) - M(connectome, C0); T_nulls = mean_k
  [M(null_k, CM) - M(null_k, C0)]; R = G_CM - G_C0 = T_real - T_nulls; the same for brightness. Absolute scores of
  both families are always reported. G_C0 is recomputed on these data, never taken from V1.

## 9. Uncertainty and decision rules (mission §14)

- 5,000 paired bootstrap replicates, seed 41001: test images resampled within the 16 label cells, the same multiset
  for every graph, condition, seed, budget, draw and task; the 40 nulls resampled as whole graphs with all their
  conditions; the connectome kept; no decoder or calibration refitted. 95 % percentile intervals. Code checked on
  synthetic data with known contrasts (tests/unit/test_activity_match.py). The intervals are conditional on the
  training and validation data, the calibrated gains, the graphs, the fitted decoders and the two seeds; they do not
  cover a new calibration or a new population of connectomes.
- Practical margin delta = 0.02 (2 points). Precision expected: V1's G interval with this design had a half-width of
  0.0055, so an interval inside +-0.02 is attainable; delta is not widened afterwards.

| interval of G_CM | reading |
|---|---|
| entirely inside [-delta, +delta] | compatible with practical equivalence, conditional on this margin |
| entirely below -delta | persistent deficit of practical importance at the intermediate target |
| entirely above +delta | advantage of practical importance at the intermediate target |
| otherwise | practical importance not settled at this precision |

- Strong interpretation ("the deficit depends strongly on the initial dynamic setting") only if all hold:
  matching valid (CAL_CHECK passed and test-image rates within 10 % of the CM target for every graph); G_C0 interval
  entirely below -delta; G_CM inside +-delta; R interval above 0; T_real interval above 0; lower bounds of the
  absolute CM scores of both families (connectome M, null mean M) above 0.30. T_real and T_nulls reported always.
- CL, CH, brightness and per-budget results characterise; they never replace CM. Their intervals are descriptive
  (no multiplicity family is declared). The rank among 41 graphs is descriptive; no p-value.
- If test-image rates leave the tolerance despite CAL_CHECK, the contrasts are reported with that transport defect
  and the comparison is not called fully matched; nothing is retuned on the test.

## 10. Failure criteria and stops (mission §20)

Stop and report a diagnostic if: graphs or weights unidentifiable; gamma ineffective or applied wrongly; label leak
into calibration or test; CM not validated for every graph (then no campaign announced as matched; if only CL or CH
fails, the reduced C0 + CM plan, declared here before the freeze); persistent numerical invalidity; projected cost
beyond the budget; incomplete results that prevent the matched contrast. A lot is rerun only for a documented
technical reason (at most 2 attempts on CUDA, memory or I/O errors).

## 11. Compute ceiling and reductions (mission §16)

72 h elapsed from T0 2026-10-20T07:37:04Z; no new long job after 2026-10-23T01:37:04Z. Projected: 328 lots x about
11 s plus graph loads, about 1.2 h; decoders about 30 min; analysis minutes. No reduction is needed; the
mission's reductions (C0 + CM only, 5 draws, smaller test) are not used unless a measured overrun forces them, in
that order and before any score is seen.

## 12. Calibration result and selected gains (2026-10-20T07:51-08:39Z, label-free)

Files: `artifacts/activity_match/calibration/calibration.json` (sha256 `3deadc1f…`, every point, search log and check)
and `selection.json` (sha256 `6a9caf71…`, the gamma of every graph and condition; the main manifest records it).

- Mission grid 0.5-2: no common range ("empty intersection": connectome 35.8-42.3 Hz, nulls 45-56 Hz), so the
  declared extension D5 was applied to all 41 graphs (19 points, 0.02-16).
- Invalid points: gamma 0.02 for every graph and 0.03 for 38 of the 40 nulls (active fraction < 1 %); no point
  failed the saturation rule. Rates at gamma = 1: connectome 38.6 Hz, nulls 48.4-53.5 Hz; plateaus: connectome
  45.4 Hz, nulls 55.2-59.9 Hz. The rate rises monotonically with gamma for the nulls; the connectome dips once
  between 2 and 3 (42.2 then 40.7 Hz), outside every bracket used.
- Common range [6.46, 45.42] Hz (lower end: nullin:6 at gamma 0.05; upper end: the connectome's plateau).
  **Targets: CL 16.20 Hz, CM 25.94 Hz, CH 35.68 Hz.**
- Search: 0-6 extra evaluations per graph (mean 4.2 of the 9 allowed); no non-monotone bracket; every selected point
  within the 2 % fit tolerance on CAL_FIT (maximum 1.97 %).
- **Selected gammas:** connectome CL 0.0689, CM 0.150, CH 0.500; nulls (median, range) CL 0.085 (0.078-0.089),
  CM 0.113 (0.104-0.123), CH 0.164 (0.142-0.200). C0 = 1 for all.
- **CAL_CHECK passed for all 41 graphs and all three targets**: maximum relative error 2.5 % (CL), 2.1 % (CM),
  2.1 % (CH) on the mean of the two seeds (largest single-seed error at CM 2.1 %), against the 10 % tolerance.
  The full plan C0, CL, CM, CH is therefore kept (no reduction).

Consequence stated before any score: every matched condition runs all graphs far below the V1 gain (the nulls at
about a tenth of it at CM); at gamma below 0.1 the connectome is the more active graph, above it the less active.
The calibrated conditions therefore compare the graphs in a weakly coupled regime, not near the V1 operating point;
CH (35.7 Hz) is the matched level closest to the connectome's own V1 activity (38.6 Hz).
