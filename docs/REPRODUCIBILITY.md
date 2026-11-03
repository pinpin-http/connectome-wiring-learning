# Reproducing the preprint: every table and figure, and what produces it

The manuscript is not in this repository. Figures are listed by the file name `make figures` writes to
`artifacts/figures/`, tables by the manuscript's LaTeX label. `ARCHITECTURE.md` explains how the experiments fit
together.

In the status column, ✅ means the script and its input are in this repository. `make figures` rebuilds every
computed figure from these inputs; `python tools/verify_results.py --strict` checks the recorded hashes.

Recomputing a table or figure from its aggregated input takes seconds on a CPU. A "re-run" costs as much as
recomputing the input itself.

## Figures (`make figures` -> `artifacts/figures/`)

| file | content | script | input | status |
|---|---|---|---|---|
| `fig_dispositif.tex`, `fig_revision_G.tex` | set-up diagram; equal-tuning controls, G intervals | TikZ in the manuscript source | G intervals: `artifacts/revision48/analysis/tables/contrasts_G.csv` | ✅ (data) |
| `fig_lobe_optique.png` | central nervous system, optic lobe | external illustration | | |
| `fig_temoins.pdf`, `fig_transformation.pdf`, `fig_synthese.pdf`, `fig_ec_contrastes.pdf` | rewiring effect, weight-conversion exposure, frozen-network synthesis, imposed-action contrasts | `make_figures_campagne.py` (values written in the script) | | ✅ |
| `fig_balayage.pdf` | learning validation in the sweep regime | `make_figures_campagne.py` | `artifacts/runs/gate_g_sweep_confirm-20260930.json` | ✅ |
| `fig_retino.pdf` | eye-map permutation (RETINO_ALIGN_V1) | `make_figures_campagne.py` | `artifacts/retino_align/main_v1/metrics/analysis.json` | ✅ |
| `fig_calibration.pdf` | output activity vs gain factor γ (ACTIVITY_MATCH) | `make_figures_campagne.py` | `artifacts/activity_match/calibration/calibration.json` | ✅ |
| `fig_ec_courbes.pdf` | learning with chosen vs imposed actions (EXPERIENCE_COMMUNE_V1) | `make_figures_campagne.py` | `artifacts/experience_commune_v1/metrics/runs_table.csv` | ✅ |
| `fig_sonde.pdf` | readability before learning vs learning | `make_figures_campagne.py` | `artifacts/experience_commune_v1/complements/probe_representation/probe.json` | ✅ |
| `fig_validation.pdf` | calibration of the statistical tools | `make_figures_campagne.py` | `artifacts/statistical_audit_20261019/calibration.json` + `artifacts/complements/bootstrap_coverage.json` (`scripts/diagnostics/bootstrap_coverage.py`) | ✅ |
| `fig_regime.pdf` | output activity vs global gain | `make_figures_campagne.py` | `artifacts/runs/mission/regime_map.json` | ✅ |
| `fig_appui.pdf` | scripted players, hold extension | `make_figures_campagne.py` | `artifacts/runs/mission/speedrun_ceiling.json` | ✅ |
| `fig_stimuli.pdf` | permutation stimuli | `make_figures_campagne.py` | `artifacts/retino_align/pilot_e2e/figures/fig1_stimulus_permutation.png` | ✅ |
| `fig_activite.pdf` | decoder accuracy vs output activity | `make_figures_campagne.py` | `artifacts/complements/synthesis_activity.json` (`scripts/diagnostics/synthesis_activity.py`, from the ACTIVITY_MATCH V1, V2, V2B analyses) | ✅ |
| `fig_s_taches.pdf` | stimuli of the two tasks | `make_figures_taches.py` | none | ✅ |
| `fig_s_courbes.pdf`, `fig_s_mesures.pdf`, `fig_s_exploratoire.pdf` | Study S curves, distributions, exploratory | `make_figures_taches.py` | `artifacts/runs/study_s/*.json` | ✅ |
| `fig_mario_copies.pdf` | Mario, learning speed vs 40 nulls per family | `make_figures_taches.py` | `artifacts/runs/mission/readings-{24,26,27}.json`, `graph_activity.json` | ✅ |
| `\figrev{F01_courbes_tuned,F01_courbes_supervised,F07_reglage_scores,F10_ecart_par_point_de_controle,F11_gain_du_reglage}.png` | equal-tuning controls: curves after tuning, supervision, tuning scores, gap per exam, tuning gain | `scripts/revision48/figures.py` | `artifacts/revision48/analysis/figures/` (written from `tables/`) | ✅ |

`make_figures_campagne.py` reads its inputs from the repository paths above; with `--campagne <dir> --complements
<dir>` it reads the same files from the owner's unpacked backups (`inputs_from_backups()` lists each path).

## Tables (LaTeX labels of the manuscript)

| label | content | script | input | status |
|---|---|---|---|---|
| `tab-anatomie` | scale of the simulated MaleCNS v1.0 graph | `scripts/build_graph.py`, `scripts/check_connectome.py` | `data/connectome/malecns_stats.json` (`make download build-graphs`) | ✅ |
| `tab-familles` | properties preserved by the two null families | `src/flymario/connectome/randomize.py`, `scripts/diagnostics/{null_validate,instrength_validate}.py` | `artifacts/runs/mission/nulls/*.json` | ✅ |
| `tab-temoins` | validation on a reserved null | `scripts/diagnostics/study_s.py refs` | `artifacts/runs/study_s/nulllat10{0,1}-*.json` | ✅ |
| `tab-experiences`, `tab-bilan`, `tab-volumes` | design, conclusions, compute volume | none (text) | | |
| `tab-vitesse`, `tab-secondaires`, `tab-robustesse` | Study S: speed, accuracy, robustness | `scripts/diagnostics/study_s_reading.py` | `artifacts/runs/study_s/reading-K{20,40}c.json` | ✅ |
| `tab-posthoc` | exploratory values | `scripts/figures/make_figures_taches.py` (prints them) | `artifacts/runs/study_s/*.json` | ✅ |
| `tab-ec` | chosen vs imposed actions | `scripts/experience_commune/analyse.py` | `artifacts/experience_commune_v1/metrics/` | ✅ |
| `tab-f2` | readability before learning | `scripts/diagnostics/study_s_mechanism.py`, `scripts/experience_commune/` (probe) | `artifacts/runs/study_s/mechanism.json`, `artifacts/experience_commune_v1/complements/probe_representation/` | ✅ |
| `tab-revision` | equal-tuning controls: L, G, D | `scripts/revision48/analyse.py` | `artifacts/revision48/analysis/report.json`, `tables/contrasts_G.csv`, `change_after_tuning_D.csv` | ✅ |
| `tab-r48-grille` | tuning grid and scores | `scripts/revision48/figures.py` | `artifacts/revision48/analysis/tables/tuning_selection.csv` | ✅ |
| `tab-mario` | Mario: supervised learning speed vs 40 nulls | `scripts/diagnostics/{mission_readings,null_distribution}.py` | `artifacts/runs/mission/readings-{24,26,27}.json` | ✅ |
| `tab-speedrun` | distance on the route levels | `scripts/diagnostics/speedrun_ceiling.py` | `artifacts/runs/mission/speedrun_ceiling.json` | ✅ |

## Where the methods are implemented

| item | where |
|---|---|
| Null graphs (rewiring) and their seeds | `src/flymario/connectome/randomize.py` (`depth_stratified_lateral` = family S, `depth_stratified_lateral_instrength` = family E), `src/flymario/connectome/nullcache.py`; revision-campaign seeds and autapse nulls: `scripts/revision48/nulls.py`, `configs/revision48/graph_manifest.json` |
| Checks of the preserved properties | `scripts/diagnostics/null_validate.py`, `instrength_validate.py`, `instrength_null_check.py`, `scripts/revision48/checks.py`; tests in `tests/unit/` |
| Rank, z score, empirical p, max-\|z\| adjustment | `scripts/diagnostics/study_s_reading.py` (reproduced field by field on 2026-10-19: `artifacts/report_audit_20261019/audit.json`) |
| Symmetric variant and calibration simulations | `tools/audits/statistical_20261019.py`, declared in `docs/STATISTICAL_AUDIT_20261019.md` (hash-checked) |
| Paired bootstrap of the equal-tuning controls | `src/flymario/revision48/analysis.py`, `scripts/revision48/analyse.py` (10,000 replicates, seed 20261024) |
| Frozen revision protocol | `docs/revision48/PROTOCOL_FROZEN.md` and the sealed tuning selection `artifacts/revision48/tuning/selection.json` (`70adb964…`), both SHA-256 checked by `results/SHA256SUMS.revision48-freeze-20261024` |
| Ceiling sensitivity of the tuning effect (2026-11-02, post hoc) | `docs/REVISION_UNSATURATED.md` (frozen `219868a`), `scripts/diagnostics/revision_unsaturated.py`, result `artifacts/revision48/unsaturated/unsaturated.json` |
| Criteria declared before the data | `docs/STUDY_S.md`, `docs/RL_STUDY.md`, `docs/PERCEPTION_CREDIT_STUDY.md`, `docs/{retino_align,activity_match,experience_commune,revision48}/`, `configs/*/` |
| Tests | `tests/unit/` (`make test`; slow ones: `.venv/bin/python -m pytest tests -m slow`) |

## GPU non-determinism (what "reproduce" means here)

The simulations ran on an AMD RX 9060 XT (ROCm 6.4) and then on an NVIDIA RTX 3090 (CUDA 12.8). Both compute
the same dynamics to about 1e-6, but learning amplifies rounding differences: a 1e-7 difference grows by about
10^5 within some thirty updates, and changing only the order of summation moves the final result. On one card a
run is bit-identical as long as the batch of learners simulated together is unchanged; otherwise trajectories
diverge as with another noise seed. Every training run is therefore treated as one draw, and the unit of
evidence is a distribution over graphs. A re-run will not give the published numbers bit for bit; it gives
an equivalent distribution. All runs compared with each other in the manuscript ran on the same card.
Manuscript §3.11 (*Calcul et reproductibilité*) states the same thing.

Two things are exact: recomputing every table and figure from the committed aggregated files, and the statistical
calibration (`tools/audits/statistical_20261019.py`, fixed master seed 2026101902).

## Equal-activity decoding (ACTIVITY_MATCH V2 and V2B)

The equal-activity results of manuscript §7.2 (family E: V2, 40 nulls; family S: V2B, 20 nulls) are in
`artifacts/activity_match/{calibration_v2,main_am2,calibration_v2b,main_am2b}/`, under the frozen protocols
`docs/activity_match/PROTOCOL_FROZEN_V2.md` and `PROTOCOL_FROZEN_V2B.md`. The share of output neurons active per test
image (`activity_test.<graph>.<condition>.active_frac_mean` in `metrics/analysis.json`) is 71.5 % for the connectome
and 62.1 % on average for the family-E nulls brought to its activity (CN; range 60.7-64.5 %), 61.6 % for family S.
