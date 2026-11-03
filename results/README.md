# Results: where the manuscript's numbers come from

Every table and figure of the manuscript is computed from small aggregated files (JSON). They sit
at the paths the manuscript and the dated audits cite (`artifacts/runs/study_s/`,
`artifacts/runs/mission/`), so those citations stay valid. This folder holds only the SHA-256 records
that state which bytes the published numbers were computed from.

```bash
python tools/verify_results.py            # standard library only; no GPU, no venv, no data needed
python tools/verify_results.py --strict   # a listed file that is absent is also a failure
```

| manifest | what it pins | recorded by |
|---|---|---|
| `SHA256SUMS.study_s-audit-20261019` | the 102 Study S result files (connectome, 40 + 40 nulls, F1 replicates, readings, F2) | the report audit of 2026-10-19 (`artifacts/report_audit_20261019/audit.json`) |
| `SHA256SUMS.statistical-audit-design-20261019` | the frozen design of the statistical validation (script and declaration) | `tools/audits/statistical_20261019.py`, before it ran (`artifacts/statistical_audit_20261019/calibration.json`) |
| `SHA256SUMS.malecns-v1.0` | the three raw MaleCNS v1.0 tables and the graph built from them | `deployment/flydeploy.py`, 2026-10-09 (`deployment/manifest/data_manifest.json`) |
| `SHA256SUMS.revision48-freeze-20261024` | the equal-tuning controls' frozen protocol and sealed tuning selection | the campaign, at the freeze and after tuning |
| `SHA256SUMS.revision48-analysis-20261102` | the campaign's aggregated analysis and the ceiling-sensitivity result | recorded on import, 2026-11-02 |
| `SHA256SUMS.imports-20261102` | the mission files, ACTIVITY_MATCH V2/V2B results and protocols, and the complements | recorded on import, 2026-11-02 |
| `SHA256SUMS.null-graphs` | 47 null-graph realisations (families S and E), regenerated from their seeds by `make nulls` | the RETINO_ALIGN frozen protocol and the 2026-10-09 deployment manifest |

A record is never regenerated to make a check pass: a mismatch means the file changed after the
record, and is reported as such. A new record (for example, the equal-tuning controls) is a new
manifest file with its date in the name.

## What is in the repository

| set | path | in git | note |
|---|---|---|---|
| Mario and perception studies up to 2026-10-09 | `artifacts/runs/*.json`, `artifacts/runs/manifests/` | yes | |
| Statistical calibration (figure of the calibration simulations) | `artifacts/statistical_audit_20261019/calibration.json` | yes | |
| Report audit of 2026-10-19 (hashes, recomputed statistics) | `artifacts/report_audit_20261019/` | yes | |
| Study S (visual tasks): per-graph files, readings, F1, F2 | `artifacts/runs/study_s/*.json` | yes | `SHA256SUMS.study_s-audit-20261019` |
| 48-hour mission (Mario): readings, graph activity, nulls validation, regime map, speedrun ceiling | `artifacts/runs/mission/**/*.json` | yes | `SHA256SUMS.imports-20261102` |
| Equal-tuning controls: frozen protocol and sealed tuning selection | `docs/revision48/PROTOCOL_FROZEN.md`, `artifacts/revision48/tuning/selection.json` | yes | `SHA256SUMS.revision48-freeze-20261024` |
| Equal-tuning controls: aggregated analysis | `artifacts/revision48/analysis/` | yes | `SHA256SUMS.revision48-analysis-20261102` |
| Experiments 2-6 (imposed actions, probe, activity match, permutation) | `artifacts/{experience_commune_v1,activity_match,retino_align}/` | yes | ACTIVITY_MATCH V2/V2B: `SHA256SUMS.imports-20261102` |
| Complements (bootstrap coverage simulation; decoding vs activity) | `artifacts/complements/*.json` | yes | `SHA256SUMS.imports-20261102`; scripts in `scripts/diagnostics/` |

Anything large (null realisations, checkpoints, raw logs) goes to a separate Zenodo deposit, not
to git.
