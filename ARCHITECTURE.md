# Architecture

How the experiments are built, declared, run and read. The preprint gives the scientific account;
`docs/REPRODUCIBILITY.md` maps each of its tables and figures to the script and file that produce it.

## 1. The comparison

One measured wiring diagram, the MaleCNS v1.0 connectome, is turned into a spiking network and compared with **null
graphs**: copies rewired at random under constraints that keep everything the model uses except who connects to whom.
The question is whether the measured organisation changes how the network learns. Every comparison follows the same
recipe:

```
raw tables ──► graph ──► null graphs (seeded, validated)
                 │              │
                 └──► same network, same interface, same rule, same stimuli, same seeds ──► learners
                                                                                            │
                                         measure per learner ──► value per graph ──► connectome vs nulls
```

The graph is the unit of analysis. Several learners per graph (readout projections × noise seeds × blocks) are
averaged into one value; the connectome is then placed in the distribution of its nulls.

## 2. Pipeline, from data to a reading

| stage | what happens | code |
|---|---|---|
| Data | MaleCNS v1.0 tables (Janelia, CC BY), SHA-256-checked on download | `src/flymario/connectome/download.py`, `make download` |
| Graph | neurons, signed synapse counts, sensorimotor depth from the lamina, sides, output neurons | `src/flymario/connectome/{malecns,schema,strata,datasets}.py`, `scripts/build_graph.py` |
| Null graphs | depth-stratified rewiring that keeps degrees, signs, weight multiset, depth classes and laterality; family **S** keeps every neuron's out-strength (`nulllat:<seed>`), family **E** its in-strength (`nullin:<seed>`) | `src/flymario/connectome/randomize.py`, `nullcache.py`, `scripts/diagnostics/build_nulls.py` |
| Null checks | every realisation is validated before use (degrees, signs, strata, sides, strength) | `scripts/diagnostics/{null_validate,instrength_validate}.py`, `scripts/revision48/checks.py` |
| Network | leaky integrate-and-fire neurons on the whole graph, batched on the GPU; the *sweep regime*: one fresh look per decision, from rest | `src/flymario/models/torch_lif.py`, `src/flymario/training/sweep.py` |
| Eye | image → injected current on retinotopically mapped lamina neurons | `src/flymario/env/vision.py`, `src/flymario/retino_align/retina.py` |
| Readout | spike counts of the 2,227 output neurons → fixed random projection → 8 action scores | `src/flymario/policy/readout.py` |
| Learning | local three-factor rule (eligibility × reward prediction error) on the synapses onto output neurons | `src/flymario/training/{plastic,rules}.py`, `src/flymario/experience_commune/learn.py` |
| Tasks | two 8-way visual discriminations (*where?*, *approach or recede?*), and *Super Mario Bros.* | `scripts/diagnostics/study_s.py`, `src/flymario/env/retro_env.py` |
| Statistics | rank, z score, empirical p with max-\|z\| adjustment; paired bootstrap over blocks and graphs with declared margins | `scripts/diagnostics/study_s_reading.py`, `src/flymario/{experience_commune,revision48,activity_match,retino_align}/analysis.py` |

Model parameters (gain, time constants, operating point) and what constrains each one: `docs/CALIBRATION.md`. The
general protocol (null ladder, unit of analysis, decision rules): `docs/PROTOCOL.md`.

## 3. The experiments

Numbering is the manuscript's (`tab-experiences`). Each experiment has one declaration, one code folder, one config
and one output folder.

| # | experiment | question | declaration (frozen before the runs) | code | outputs |
|---|---|---|---|---|---|
| 1 | Reward-learned visual tasks (Study S) | does the wiring speed up learning, generalisation, robustness? connectome vs 40 S + 40 E | `docs/STUDY_S.md` | `scripts/diagnostics/study_s*.py` | `artifacts/runs/study_s/` |
| 2 | Imposed actions | does the advantage persist when every graph gets the same images, actions and rewards? | `docs/experience_commune/PROTOCOL_FROZEN.md`, `DEVIATIONS.md` | `src/flymario/experience_commune/`, `scripts/experience_commune/` | `artifacts/experience_commune_v1/` |
| 3 | Probe and equalized activity | is the advantage explained by the information available before learning, or by the activity level? | same as 2 (complements) | `scripts/experience_commune/{probe_representation,matched_activity*}.py` | `artifacts/experience_commune_v1/complements/` |
| 4 | Equal-tuning controls (REVISION48) | does the advantage survive an equal tuning budget, log weights, nulls with autapses? does it exist under supervision? | `docs/revision48/PROTOCOL_FROZEN.md`, `configs/revision48/` | `src/flymario/revision48/`, `scripts/revision48/` | `artifacts/revision48/analysis/` |
| 4b | Ceiling sensitivity (post hoc) | can the tuning effect be separated from the ceiling of the learning area? | `docs/REVISION_UNSATURATED.md` | `scripts/diagnostics/revision_unsaturated.py` | `artifacts/revision48/unsaturated/` |
| 5 | Eye-map permutation (RETINO_ALIGN) | does the connectome depend more on retinotopic alignment? (frozen network, decoders) | `docs/retino_align/PROTOCOL_FROZEN.md`, `configs/retino_align/` | `src/flymario/retino_align/`, `scripts/retino_align/` | `artifacts/retino_align/` |
| 6 | Equalized activity, frozen network (ACTIVITY_MATCH) | is the visual decoding gap explained by activity? | `docs/activity_match/PROTOCOL_FROZEN{,_V2,_V2B}.md`, `configs/activity_match/` | `src/flymario/activity_match/`, `scripts/activity_match/` | `artifacts/activity_match/` |
| 7 | *Super Mario Bros.* control | does the wiring help learn when to jump, then to play? | `docs/RL_STUDY.md` §17-§30, `docs/PERCEPTION_CREDIT_STUDY.md`, `docs/GENERALISATION_STUDY.md`, `docs/MARIO_PILOT_PLAN.md` | `src/flymario/training/mario*.py`, `src/flymario/env/`, `scripts/diagnostics/{common_stream,mission_readings,frozen_competence,speedrun_ceiling}.py` | `artifacts/runs/`, `artifacts/runs/mission/` |
| - | Statistical calibration | do the rank / max-\|z\| tools hold their error rate? | `docs/STATISTICAL_AUDIT_20261019.md` | `tools/audits/statistical_20261019.py` | `artifacts/statistical_audit_20261019/` |

The early gates (A-G: closed loop, operating point, interface, learning-rule validation) that set up the model are
declared in `docs/PROTOCOL.md` and `docs/GATE_G_PLAN.md`; their scripts are in `scripts/gates/`.

## 4. Declared criteria

1. **Declaration before data.** Each experiment's question, primary measure, null families, number of graphs, seeds,
   intervals, margins and reading rule are written down before any run on an analysed graph (`docs/`).
2. **References before reading.** Every criterion is first tested on references that must pass and references that
   must fail (scripted players, permuted labels, identical graphs, synthetic curves with a known answer). A criterion
   that fails its references is not used.
3. **Development graphs only.** Pilots, calibrations and timing use reserved development nulls (`nulllat:100`,
   `nulllat:101`), never an analysed graph.
4. **Frozen protocols.** The larger campaigns (2, 4, 5, 6) fix their protocol in a `PROTOCOL_FROZEN.md`; their SHA-256,
   and that of REVISION48's tuning selection, are in `results/`.
5. **Exploratory analyses.** An analysis decided after seeing a result is labelled exploratory (or post hoc) and never
   replaces the declared reading.

## 5. Repository layout

| path | content |
|---|---|
| `src/flymario/connectome/` | dataset adapters, graph container, depth strata, null models and their cache |
| `src/flymario/models/` | LIF dynamics (CPU reference and batched GPU), dense baselines |
| `src/flymario/training/` | learning rules, sweep regime, plastic learners, Mario learners |
| `src/flymario/env/` | Gymnasium wrapper of the game, action set, visual interface, stimulus bank |
| `src/flymario/policy/` | fixed motor readout |
| `src/flymario/{experience_commune,revision48,retino_align,activity_match}/` | code specific to experiments 2-6 |
| `src/flymario/{run_manifest,runctl}.py` | run manifests (inputs, hashes, git commit) and checkpoint / stop control |
| `scripts/` | entry points: `build_graph.py`, `diagnostics/` (experiments 1 and 7, analyses), one folder per campaign, `gates/` |
| `configs/` | base configuration and one config per campaign |
| `tests/unit/` | unit tests (`make test`) |
| `docs/` | protocols and study declarations |
| `scripts/figures/` | the preprint's figure scripts (`make figures` -> `artifacts/figures/`) |
| `artifacts/` | run outputs; aggregated results are versioned, checkpoints and caches are not |
| `results/` | SHA-256 records of the result files and frozen protocols; `tools/verify_results.py` checks them |
| `deployment/` | install, transfer and monitoring on a rented GPU server |
| `integrations/` | *Super Mario Bros.* integration for stable-retro (no ROM) |

## 6. Runs and outputs

Long runs are headless, checkpoint atomically and can be stopped and resumed (`make checkpoint`, `make stop`,
`src/flymario/runctl.py`). Every run writes a manifest with its inputs, their hashes, the configuration and the git
commit (`src/flymario/run_manifest.py`). Per-learner and per-graph results are small JSON or CSV files under
`artifacts/`; the declared reading of each experiment is computed from them by its analysis script, and the figures by
`scripts/figures/make_figures_*.py`.

GPU computations are not bit-reproducible across cards or batch compositions: learning amplifies rounding
differences, so each run is one draw and the evidence is a distribution over graphs. All runs compared with each
other ran on the same card. A re-run reproduces the distribution, not the exact numbers; recomputing the tables and
figures from the stored results is exact.
