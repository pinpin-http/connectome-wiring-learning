# Local validation of the Study S reading — declared 2026-10-19

## Purpose and status

A diagnostic audit of an existing analysis, after the historical results are known. This is not a
preregistration of Study S or a replacement of its declared verdict. The owner asked to advance locally.
The following simulation design and reading criteria are fixed before executing this new script.

## Designs

- Seed 2026101902; 20,000 independent simulated experiments per cell; batches of 500.
- K = 20 and 40 reference graphs; m = 2 and 8 endpoints, as in the historical analysis families.
- One observed graph and K null graphs are independent draws from the same joint distribution under
  the complete null. Four distributions: independent standard normal; normal with endpoint correlation
  0.8; independent Student t with 3 degrees of freedom; independent lognormal (log SD 1.5).
- Compare the historical asymmetric construction (observed against K nulls, each null against K−1)
  with a symmetric diagnostic: include all K+1 rows, compute each row's z against the other K rows,
  then rank its absolute z against the row-wise maxima, including the observed row.
- The symmetric diagnostic is exchangeable under these simulated nulls by construction. It is a
  benchmark, not proof that the real rewiring ensemble is exchangeable with the measured connectome.
- Count any endpoint with adjusted p < 0.05 as a false rejection. Report complete-null family-wise
  rejection rate and 95% exact binomial interval, plus Monte Carlo counts. This does not establish
  strong FWER control with some false and some true null hypotheses.

## Fixed references and reading criteria

1. Hand fixture: four scalar nulls [1,2,3,4], observed 10; historical doubled-tail p=.4 and max-|z| p=.2.
   Agreement with the historical implementation on a fixed random small array is required.
2. Symmetric construction: relabeling/reordering rows must relabel/reorder the statistics, and
   adjusted p must never be below the raw p formed from those same absolute z statistics.
3. Ties: identical rows and constant endpoints return p=1, not NaN or artificial significance.
4. False-positive reference: eight independent unadjusted normal z tests at .05 (known population
   mean and SD) must exceed .10 family-wise rejection; this checks that the diagnostic detects inflation.
5. Positive sensitivity reference: independent standard normals, K=40, m=2, observed row shifted by
   +8 on both endpoints. Both adjusted methods must reject in at least 95% of 2,000 experiments.
6. For the 16 complete-null cells × two adjusted methods, flag excessive rejection only if a one-sided
   exact binomial test against .05 has p < .01/32 (Bonferroni). Do not call a nonsignificant deviation
   proof of calibration. No criterion or simulation distribution will be changed after reading results.

## Historical-data sensitivity (explicitly post hoc)

After the fixed references, apply the symmetric benchmark to the same stored graph-level vectors,
for Kc=20 and Kc=40, with unchanged endpoint families and direction checks. Report it beside the
historical reading, never overwrite `reading-K20c.json` or `reading-K40c.json`. Report all endpoints,
including those whose classifications do not change. This is a sensitivity analysis, not a new
confirmatory result or a remedy for F3's data-dependent sample-size extension.

## Execution and constraints

CPU only; no model training or null-graph construction. Write results atomically after every design
cell, with `complete: false` until finished. A STOP file is checked between batches; rerunning resumes
from completed cells with independent design-derived RNG streams. A partially completed cell is
recomputed from its fixed seed rather than reused. No GPU rental, protocol replacement or new scientific
training intervention is authorized by this audit.
