# Revision campaign: effect of tuning on measures that do not saturate

**Status:** fixed before this analysis read the campaign's tuned-condition exams. The campaign's declared reading (G, D
on the learning area) was already known, so this is a post hoc sensitivity analysis (§1).

## 1. Why

Question: after equal-budget tuning the connectome reaches 99.8–100 % final accuracy and the
nulls 92–97 % (manuscript §6.4), so the learning area L is bounded above. The reduction of the connectome's
advantage after tuning (D = G_tuned − G_reference, −5.7 to −10.0 points) may then come from the ceiling of L
rather than from the tuning. The abstract says so ("sans que l'on puisse encore séparer l'effet du réglage de
celui du plafond de la mesure"). This analysis tries to separate the two.

**This is post hoc.** The campaign's main reading has been seen. Whatever this analysis gives is a sensitivity
analysis, reported as such, and changes no declared verdict of the campaign.

## 2. What the synthetic references already show (before any data)

A reference in which tuning multiplies every graph's learning rate by the same factor (the connectome stays
exactly twice as fast; only the ceiling differs) gives:

| measure | D on the reference with a pure common speed-up | can it separate tuning from ceiling? |
|---|---|---|
| learning area L | −9.3 points | no: it shrinks although the speed ratio is unchanged |
| accuracy gap at 250 decisions | +10.8 points | **no**: moves with the position on the curve |
| accuracy gap at 1,000 decisions | −10.9 points | **no** |
| log2 decisions to reach 75 % | −0.04 (margin ±0.32) | **yes**: invariant to a common speed-up |
| log2 decisions to reach 50 % | not interpretable | crossings fall before the first exam (250) |

So the fixed-time gaps (at 250 or 1,000 decisions) are reported descriptively only: they do not answer the
question. The decisions-to-criterion measure does.

## 3. Measures

Per learning run (graph × block × task, 6,000 decisions, exams at 0, 250, 500, 1,000, 2,000, 4,000, 6,000 on the
512-image bank):

- **Sensitivity measure (the one read):** `log2_t75`, log2 of the decisions needed to reach 75 % accuracy, by
  linear interpolation in decisions between the two exams that bracket the first crossing.
- Secondary: `log2_t50`, the same at 50 %.
- Descriptive only: exam accuracy at 250 and at 1,000 decisions.

A run that never reaches the threshold counts as 6,000 decisions (right-censored); a run that reaches it within
the first 250 decisions is flagged left-censored. Graph value: mean over its six blocks; family value: mean of
its ten graphs. Gap G (time measures): family mean − connectome, a log2 speed ratio (positive = connectome
faster). D = G_tuned − G_reference.

## 4. Intervals and reading

Paired percentile bootstrap, 10,000 replicates, seed 20261102: the six blocks are resampled with replacement
(same draw for every graph and both conditions) and the ten graphs of the family are resampled with replacement
(same draw for both conditions). 95 % intervals, per task and family (four contrasts), no multiplicity
correction (sensitivity analysis).

Margin for time measures: ±log2(1.25) = ±0.32 (a 25 % difference in the decisions needed). Reading of D:
interval entirely below −0.32 → "the tuning reduces the connectome's speed advantage beyond the ceiling";
entirely within ±0.32 → "the reduction of L is compatible with the ceiling alone"; entirely above +0.32 → "the
tuning increases the speed advantage"; otherwise "not decided at this precision".

A time measure is **not interpreted** for a contrast if more than 20 % of the runs of any of its four cells
(connectome and family, reference and tuned) are censored; the censoring counts are always reported.
Fallback, fixed now: if `log2_t75` is not interpretable for a contrast, `log2_t50` is read for that contrast
instead; if neither is, the conclusion for that contrast is "these exams cannot separate the tuning from the
ceiling" (likely when the historical settings leave the nulls below 75 % at 6,000 decisions, or when the tuned
connectome passes 50 % before the first exam at 250).

## 5. References (must pass / must fail), checked by the script before it reads any data

| reference | construction | expectation |
|---|---|---|
| ceiling | connectome 2× faster than every null; tuning ×4 for every graph | D(log2_t75) within margin; D on the fixed-time gaps **not** within margin; D on L < −3 points |
| shrink | tuning ×8 for the nulls, ×2 for the connectome | D(log2_t75) below margin |
| identical | all graphs identical | every gap and every D within margin |

All three pass (2026-11-02). The script stops before reading any data if one fails.

## 6. Input

The campaign's exam table `artifacts/revision48/analysis/tables/per_run_block.csv` (written by
`scripts/revision48/figures.py`; one row per condition × graph × task × block, exam accuracies `A_0` … `A_6000` as
fractions), read by `--r48`; only the `reference` and `tuned` conditions are used, nothing else is filtered.
`artifacts/revision48/analysis/tables/` is the campaign's own output, imported from the owner's archive (see
`paper/PROVENANCE.md`).

```bash
.venv/bin/python scripts/diagnostics/revision_unsaturated.py \
    --r48 artifacts/revision48/analysis/tables/per_run_block.csv --out artifacts/revision48/unsaturated/unsaturated.json
```
