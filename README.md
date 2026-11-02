# What wiring determines: a fruit-fly connectome as a learning network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23898968.svg)](https://doi.org/10.5281/zenodo.23898968)

Code, dated protocols and aggregated results for the preprint

> S. Bouazzaoui, *L'agencement des connexions d'un système nerveux aide-t-il à apprendre ?*,
> preprint, 2026-11-01, revised 2026-11-02 (French; an English edition exists). The manuscript itself is not in this
> repository; every figure it computes is rebuilt here by `make figures`.

The MaleCNS v1.0 reconstruction (Janelia) is turned into a spiking network of about 170,000 neurons with an
artificial eye, a fixed projection onto eight actions and a local three-factor learning rule. It is compared
with rewired null graphs that keep neurons, degrees, signs, weights, sensorimotor depth, laterality and, by
family, the out-strength (family S) or the in-strength (family E) of every neuron. The graph is the unit of
analysis. Tasks: two reward-learned visual tasks, and a control in *Super Mario Bros.*

- **How the experiments and protocols are organised: [ARCHITECTURE.md](ARCHITECTURE.md).**
- **Which script and file produce each table and figure: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).**

> Every table and figure of the manuscript is computed from files committed here
> ([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)); their SHA-256 records are in [results/](results/README.md).

## Check the numbers without a GPU (minutes)

```bash
python tools/verify_results.py      # result files vs their recorded SHA-256 (standard library only)
make setup                          # Python 3.11 venv with the pinned requirements (CPU is enough)
make figures                        # the computed figures of the manuscript, from the aggregated files
make test                           # unit tests (the slow ones: .venv/bin/python -m pytest tests -m slow)
```

On a fresh clone, two records list files that are built locally rather than committed: the MaleCNS tables and
graph (`make download build-graphs`) and the 47 null graphs (`make nulls`). They are reported as missing until
then; `python tools/verify_results.py --strict` passes once they are built.

## Hardware and duration

| experiment (manuscript table 17) | hardware used | duration |
|---|---|---|
| Graph build from the raw tables | CPU | minutes |
| One null graph (rewiring + checks) | CPU | ~4 min |
| Visual tasks, one graph (2 tasks × 6 learners × 6,000 trials) | 1 × RTX 3090 24 GB | ~15 min (median 876 s) |
| Visual tasks, all 81 graphs | 1 × RTX 3090 | ~20 GPU hours |
| Mario control (48-hour mission) | 1 × RTX 3090, 12 cores, 72 GB RAM | 34.2 GPU hours of queued runs |
| Equal-tuning controls | 1 × RTX 3090, single process | ~40 hours |
| Statistical calibration (16 scenarios × 20,000 synthetic experiments) | CPU, 2 threads | tens of minutes |
| Tables and figures from the aggregated files | CPU | seconds |

Total before the equal-tuning controls: about 55 GPU hours on one rented card (manuscript annex).

## Environment

| | reference workstation | GPU server (all compared runs) |
|---|---|---|
| GPU | AMD Radeon RX 9060 XT (gfx1200) | NVIDIA RTX 3090 24 GB, driver 590.48.01, sm_86 |
| Python | 3.11.16 | 3.11.16 |
| PyTorch | 2.9.1+rocm6.4 | 2.9.1+cu128 |
| Emulator | stable-retro 1.0.1 (FCEUX core), gymnasium 1.0.0 | same |

Every other package is pinned in [`requirements.txt`](requirements.txt) (no torch inside, so it is portable).
The full freeze of the reference workstation is `deployment/manifest/reference_env_freeze.txt`.

```bash
# CPU or reference set-up
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch==2.9.1                     # CPU, or:
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1   # NVIDIA
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/rocm6.4 -r requirements-gpu.txt  # AMD
uv pip install --python .venv/bin/python --no-deps -e .
```

A CPU-only container that reproduces the tables, figures and tests from the aggregated files:
`docker build -t flymario . && docker run --rm flymario` ([`Dockerfile`](Dockerfile)). A remote NVIDIA
server is set up with [`deployment/README.md`](deployment/README.md).

**GPU non-determinism.** A re-run gives an equivalent distribution, not the same numbers bit for bit:
learning amplifies rounding differences between cards and between batch compositions. Every training run is
treated as one draw and the unit of evidence is a distribution over graphs. Details in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md#gpu-non-determinism-what-reproduce-means-here) and §3.11 of
the manuscript.

## Input data

| | source | licence | in this repository |
|---|---|---|---|
| MaleCNS v1.0 (3 tables, ~565 MB) | Janelia FlyEM, `gs://flyem-male-cns` | CC BY 4.0 | no: `make download` fetches it and checks the SHA-256 of each table |
| FlyWire 783 (replication axis, not used by the reported studies) | Zenodo `10676866`, `flyconnectome/flywire_annotations` | CC BY 4.0 | no: `make download-all` |
| *Super Mario Bros.* ROM | supply your own legal copy | copyrighted | **never**: `roms/` and `*.nes` are gitignored; see [docs/ROM.md](docs/ROM.md) |

Expected hashes: [`results/SHA256SUMS.malecns-v1.0`](results/SHA256SUMS.malecns-v1.0). The game integration
(`integrations/SuperMarioBros-Nes/`) holds level start states (RAM snapshots of 13 KB, smaller than the
40 KB ROM, so they do not contain it) and the ROM's SHA-1 only.

```bash
make download        # MaleCNS v1.0, verified against its recorded SHA-256
make build-graphs    # -> data/connectome/ (validated graph cache)
make import-rom      # only for the Mario experiments, with your own ROM in roms/
make check-env
```

## One command per experiment

| experiment | command | output |
|---|---|---|
| Visual tasks, one graph | `make study-s GRAPH=connectome` (or `nulllat:<seed>`, `nullin:<seed>`) | `artifacts/runs/study_s/<graph>.json` |
| Small demonstration (one null, both tasks, 500 trials) | `make demo-small` | `artifacts/runs/study_s/<graph>-demo.json` |
| Declared reading of the visual tasks | `make study-s-reading` | `artifacts/runs/study_s/reading.json` |
| Null graphs | `make nulls` | `artifacts/cache/nulls/` |
| Statistical calibration | `make calibration` | `artifacts/statistical_audit_20261019/calibration.json` |
| Imposed actions, probe, equalized activity | `scripts/experience_commune/` (see `ARCHITECTURE.md`) | `artifacts/experience_commune_v1/` |
| Equal-tuning controls | `scripts/revision48/campaign.py`, then `analyse.py` | `artifacts/revision48/` |
| Eye-map permutation, frozen-network activity match | `scripts/retino_align/`, `scripts/activity_match/run_am.py` | `artifacts/{retino_align,activity_match}/` |
| Mario experiments | `docs/RL_STUDY.md` §17-§30, `scripts/diagnostics/` | `artifacts/runs/mission/` |

`make help` lists the rest. Long runs are headless, checkpointed and stoppable (`make stop`).

## Repository layout

| path | content |
|---|---|
| `src/flymario/` | the library: connectome adapters and null models, spiking dynamics, learning rules, environment, per-campaign code |
| `scripts/` | entry points: graph build, `diagnostics/` (experiments 1 and 7, analyses), one folder per campaign, `gates/` |
| `configs/` | base configuration and one config per campaign |
| `docs/` | protocols and study declarations, frozen before the runs |
| `scripts/figures/` | the scripts that draw the preprint's computed figures (`make figures` -> `artifacts/figures/`) |
| `artifacts/` | run outputs: aggregated results are versioned, checkpoints and caches are not |
| `results/` | SHA-256 records of result files and frozen protocols |
| `tests/unit/` | unit tests |
| `tools/` | result verification, statistical calibration, run control |
| `deployment/` | install, transfer and monitoring on a rented GPU server |
| `integrations/` | *Super Mario Bros.* integration for stable-retro (no ROM) |

Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Scientific caveats

```
connectivity     ≠  physiology
synapse count    ≠  synaptic strength
simulated spike  ≠  measured fly activity
Mario action     ≠  natural fly motor behaviour
```

One model, one individual, two null families, two visual tasks.

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)). Results, figures and manuscript: CC BY 4.0
([`LICENSE-CC-BY-4.0.txt`](LICENSE-CC-BY-4.0.txt)). Connectome data: CC BY 4.0, from their authors (not
redistributed). Game ROM: not distributed. To cite: [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository" button); the Zenodo DOI is added at release.
