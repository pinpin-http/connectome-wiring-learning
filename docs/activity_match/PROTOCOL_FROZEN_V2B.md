# ACTIVITY_MATCH_V2B: frozen protocol (complement, family b)

**Frozen 2026-10-21T01:02Z**, before any main-campaign simulation (only the label-free calibration existed); hashes and commit
recorded in RUN_LOG_V2B.md and JOURNAL.md. A local, dated freeze in git, not a public preregistration. Authority: `MISSION_V2B.md` (owner's
authorisation of complements) and, for everything else, `PROTOCOL_FROZEN_V2.md`, which this version follows
**exactly except for the points listed here**:

- **Nulls:** family b, `nulllat:<s>` (depth_stratified_lateral: degrees, signs, weight multisets, depth strata and
  sides kept, every neuron's **out**-strength kept), the 20 graphs drawn (seeded, score-independent) and validated for
  EXPERIENCE_COMMUNE_V1: seeds 39, 21, 29, 30, 26, 20, 14, 5, 9, 34, 24, 13, 16, 33, 17, 32, 10, 18, 19, 27. With
  K = 20 the null-mean is less precise than with V2's 40.
- **Grid:** gamma {0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0} (family b runs hotter).
- **New banks and seeds:** calibration `am3_cal_fit` / `am3_cal_check` (62201 / 62202, noise 62001 / 62002); main
  `am3_train` / `am3_val` / `am3_test` (63101-63103), noise 61001 / 61002, subset draws 8401-8410, bootstrap 63001.
- **Plan:** 21 graphs x 2 conditions x 2 seeds = 84 lots. No development trial (the V2 pipeline is reused unchanged).
- **Decision rules:** those of V2 §8, unchanged (margin 0.02, informative threshold 0.27, same reading table and
  criteria).

**Sections 1-9 below are V2's text, kept verbatim for reference; the list above overrides them wherever they differ
(the null family and its 20 graphs, the grid, the banks, the seeds and the number of lots).**

## 1. Question

ACTIVITY_MATCH_V1 left one question open: with the connectome **left at its own setting** (gamma = 1), do the 40
nulls lose their orientation advantage when they are brought down to the connectome's own output activity? V1 matched
every graph to common targets far below the V1 gain, which changed the connectome too (its orientation score fell by
2.1 points) and left both families near chance.

## 2. Graphs and simulator

The 41 graphs of V1 (connectome + `nullin:0-39`, same files), development graphs `nullin:100`, `101` only for the
trial. The V1 simulator unchanged; the only intervention is `gain = gamma x 0.18452` on `W @ s`.

## 3. Conditions

| condition | connectome | each null |
|---|---|---|
| C0 | gamma = 1 | gamma = 1 (the V1 setting, re-measured on new images) |
| CN | gamma = 1 (unchanged) | gamma_k calibrated to the connectome's rate |

41 graphs x 2 conditions x 2 noise seeds (51001, 51002) = 164 lots. The connectome's two conditions have identical
parameters; both are simulated anyway and must give identical counts (a determinism check: T_real must be exactly 0).

## 4. Calibration (label-free, before this freeze)

- New banks from the frozen generator: CAL_FIT `am2_cal_fit` (128 images, seed 52201) and CAL_CHECK `am2_cal_check`
  (128, seed 52202), labels dropped at encoding; calibration noise seeds 52001, 52002.
- **Target:** the connectome's readout rate at gamma = 1 on CAL_FIT (mean of the two seeds). The 38.6 Hz of V1 is an
  indication only; the value measured here is the target.
- Each null: grid gamma {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0} on CAL_FIT, then V1's bounded search
  (at most 9 extra evaluations, bracketing only on monotone branches) and selection (within 2 % of the target,
  closest to gamma = 1, then the smaller gamma). Validity rules of V1 (active >= 1 %, 8-spike fraction <= 5 %,
  finite).
- **Check on CAL_CHECK:** each null's rate (selected gamma, unchanged) against the connectome's rate on the same
  CAL_CHECK images: |r_null - r_connectome| / r_connectome <= 0.10, mean of the two seeds, for every null (per-seed
  errors shown). If one null fails, no campaign announced as matched (diagnostic delivered instead).

**Calibration result (filled at the freeze):** §10.

## 5. Main data and decoder

New banks `am2_train` 1,024 (seed 53101), `am2_val` 512 (53102), `am2_test` 2,560 (53103), 16 balanced cells,
disjoint from every earlier bank (checked by `run_am.py validate` against V1, AM V1 and all calibration banks).
Decoder, budgets (8-128 per class), 10 nested draws (seeds 8301-8310), lambda grid and selection: those of V1
unchanged. A separate decoder per graph, condition and noise seed.

## 6. Metrics

M(g, c): test accuracy averaged over 5 budgets x 10 draws x 2 seeds. **Primary:** G_CN = M(connectome, CN) - mean_k
M(null_k, CN), orientation. Secondary, declared: G_C0; T_nulls = mean_k [M(null_k, CN) - M(null_k, C0)]; R = G_CN -
G_C0 (= -T_nulls, since T_real = 0); the same for brightness; absolute scores of both families in both conditions.

## 7. Intervals

5,000 paired bootstrap replicates (seed 53001): test images resampled within the 16 cells, identically for every
graph, condition, seed, budget, draw and task; nulls resampled as whole graphs with both conditions; connectome kept;
nothing refitted. 95 % percentile intervals, conditional on the training data, the calibrated gains, these graphs,
the decoders and the seeds.

## 8. Decision rules (fixed now, before any main score)

- **Margin delta = 0.02** (2 points), as in V1. Expected precision: V1's G intervals with the same design had
  half-widths of 0.0045-0.0055.

| interval of G_CN | reading |
|---|---|
| entirely inside [-0.02, +0.02] | compatible with practical equivalence: at the connectome's own activity the nulls no longer outperform it |
| entirely below -0.02 | the deficit persists when the nulls are brought to the connectome's activity |
| entirely above +0.02 | the connectome is ahead when the nulls are brought to its activity |
| otherwise | practical importance not settled at this precision |

- **Informativeness:** a reading is called informative only if the lower bounds of both families' absolute CN scores
  exceed **0.27 = chance + delta**: an equivalence between two chance-level scores says nothing. V1's threshold of
  0.30 is not reused as the criterion, because the connectome is unchanged here and scored 0.295 [0.290, 0.301] at
  gamma = 1 in V1: that criterion would fail by construction whatever the nulls do. Whether 0.30 is reached is
  reported descriptively.
- **"The deficit at the common gain is accounted for by the activity level, in this model"** only if all hold:
  matching valid (CAL_CHECK, and test-image rates of every null within 10 % of the connectome's), connectome identical
  in both conditions, G_C0 interval below -delta, G_CN interval inside +-delta, R interval above 0, informative.
- Secondary intervals (brightness, per budget, per seed) are descriptive; the rank among 41 is descriptive; no
  p-value.

## 9. Ceiling and stops

Within the ACTIVITY_MATCH 72 h envelope (no new long job after 2026-10-23T01:37:04Z). Projected: calibration
≈ 20 min, 164 lots ≈ 35 min, decoders ≈ 12 min, analysis minutes. Stops as in V1 (§20 of the mission): failed
matching, label leak, numerical invalidity, incomplete results. A lot is rerun only for a documented technical
reason.

## 10. Calibration result and selected gains (2026-10-21T00:48-00:58Z, label-free)

Files: `artifacts/activity_match/calibration_v2b/calibration.json` (sha256 `eea00b6b…`), `selection.json`
(sha256 `3d316356…`).

- **Target: the connectome at gamma = 1 runs at 38.686 Hz on the new CAL_FIT** (38.729 Hz on CAL_CHECK).
- The family-b nulls run at 65.1-71.8 Hz at gamma = 1 on the same images (family c in V2: 48.4-53.4 Hz).
- Every grid point valid; no non-monotone bracket; 0-2 extra evaluations per null (mean 1.5).
- **Selected gammas:** median 0.0645, range 0.0611-0.0698 (family c in V2: 0.16-0.25).
- **CAL_CHECK passed for all 20 nulls:** maximum relative error 1.71 % (single seed 1.79 %), tolerance 10 %.
