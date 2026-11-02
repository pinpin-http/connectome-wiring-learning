# Running SMB_Flywire_Speedrun on a remote NVIDIA GPU server

This folder moves the project from the AMD/ROCm workstation to a rented Linux/NVIDIA
server (target: a Vast.ai instance with an RTX 3090 24 GB, ~24 vCPU, ~40 GB RAM, SSH;
Runpod works the same way), checks that it works there, and runs declared experiments
detached from any SSH session.

**What it does not do.** It never rents, starts, stops or destroys an instance and never
launches a long campaign by itself. **Stopping a run, closing SSH or stopping Python does
not stop the provider's billing: stop or destroy the instance from the provider's
console.** It has not yet been run on an NVIDIA machine: until `doctor`, `smoke`, `bench`
and the comparisons of step 8 have passed there, the deployment is *prepared, not
validated*.

One tool, `deployment/flydeploy` (see `deployment/flydeploy --help`), does everything:

| on your PC | on the server |
|---|---|
| `inventory` `pack` `push` `push-rom` `pull` `remote` `reference` | `install` `verify` `fetch-data` `import-rom` `doctor` `smoke` `bench` `run` `status` `summary` `wait` `stop` `logs` `compare` |

The path: **prepare locally → transfer → install → data/ROM → diagnose → short test →
benchmark → one explicitly chosen experiment → save results → stop the instance.**

---

## 0. Requirements

- **Your PC**: this repository with its data (the reference copies), `git`, `ssh`,
  `rsync`, Python 3.11 venv (`.venv`). The ROM is yours (`roms/`, see `docs/ROM.md`).
- **The server**: Linux x86_64, glibc ≥ 2.28 (Ubuntu ≥ 20.04), an NVIDIA driver ≥ 525
  (≥ 570 selects CUDA 12.8 wheels, else 12.6), ≥ 30 GB free disk (15 GB minimum),
  outbound internet (PyPI, download.pytorch.org, astral.sh). Root is convenient (`apt`
  for rsync/tmux) but not required. Docker-in-Docker and systemd are not used.
- **On Vast.ai**: pick an image with CUDA 12 (for example a PyTorch or CUDA 12 Ubuntu
  22.04 template; the venv brings its own PyTorch), a disk of ≥ 30 GB, and add your SSH
  public key to the account. The instance card shows `ssh -p PORT root@HOST`.
- **On Runpod**: same; use the persistent volume (`/workspace`) as `REMOTE_DIR` and the
  pod's exposed TCP port for SSH.

## 1. Prepare locally (your PC)

```bash
cp deployment/deploy.env.example deployment/deploy.env      # gitignored; no secrets in it
$EDITOR deployment/deploy.env                               # REMOTE_HOST, REMOTE_PORT, REMOTE_DIR, SSH_KEY path
deployment/flydeploy inventory                              # sizes + sha256 -> deployment/manifest/
deployment/flydeploy push --dry-run                         # shows every ssh/rsync command, sends nothing
```

`inventory` rewrites `deployment/manifest/data_manifest.json` and `INVENTORY.md` (the list
of everything the server needs beyond the code, with size, sha256, origin and use).
`pack` (run by `push`) builds `artifacts/deploy/flymario-code-<commit>[-dirty].tar.gz`:
`git archive HEAD` plus tracked files modified but not committed and untracked,
non-ignored files, each listed in `deployment/CODE_VERSION.json` inside the bundle.
Never in the bundle: the ROM, `deploy.env`, keys, `.venv`, core
dumps, `artifacts/`, `data/`, `media/` (checked on the finished bundle).

## 2. Transfer (your PC)

```bash
deployment/flydeploy push              # code bundle + TRANSFER_SETS, then a remote sha256 verify
deployment/flydeploy push-rom          # your ROM alone, chmod 600, never archived
```

Data sets (`TRANSFER_SETS` in `deploy.env`; sizes from the current manifest):

| set | content | size | needed for |
|---|---|---|---|
| `core` | `data/connectome/malecns.npz` (+ ledger, stats), the 3 laterality-preserving nulls | 427 MB | every run |
| `reference` | Δx_rand tables, harness v2 starts and validations, v1 expert file | 0.1 MB | play measures and reports |
| `rocm-reference` | ROCm-trained common-stream brains and their result files | 80 MB | cross-platform checks |
| `nulls-primary` | the 4 depth-stratified nulls (`null:<s>`) | 446 MB | `null:<s>` runs only |
| `rocm-stageD` | ROCm Stage D checkpoints | 481 MB | frozen competence evaluations only |
| `public-raw` | MaleCNS v1.0 raw tables (downloaded on the server, not sent) | 566 MB | graph rebuild check only |

Default: `core,reference,rocm-reference` ≈ **530 MB** with the code (23 MB); everything ≈
1.46 GB. Each file goes with its own `rsync --partial` (an interrupted push resumes; scp is
used, not resumable, when the server has no rsync yet), then `verify` compares sha256 with
the manifest and lists anything missing or different with its fix. ROCm-made files land
under `artifacts/rocm_reference/` on the server, so no NVIDIA output can overwrite them and
no trainer can silently resume from them.

**Two data routes.** (a) Transfer (default): the graph and null caches above are the exact
files every result so far was computed on. (b) Public: `deployment/flydeploy fetch-data`
downloads the raw MaleCNS tables from the public bucket (pinned by sha256, resumable with
`curl -C -`, disk space checked first, valid files never fetched twice);
`fetch-data --rebuild-graph` rebuilds `malecns.npz` only if it is absent and **refuses**
a rebuild whose sha256 differs from the reference (the null caches are keyed to the
graph's hash: a different graph would not find them, and no null is ever rebuilt
silently).

## 3. Install (server)

```bash
ssh -p PORT root@HOST          # or, from your PC: deployment/flydeploy remote install
cd /workspace/SMB_Flywire_Speedrun
deployment/flydeploy install   # relaunchable; log in deployment/state/install.log
deployment/flydeploy verify
deployment/flydeploy import-rom
```

`install` (`deployment/install_remote.sh`): checks glibc and the driver (read only, never
changed), installs only missing `rsync tmux git curl unzip` when apt and root are
available, installs `uv`, a standalone Python 3.11 and a fresh `.venv`, then **torch 2.9.1**
(the reference version) from the CUDA 12.6 or 12.8 index, the pinned `requirements.txt` and
the project (`-e .`). The ROCm installer (`scripts/install_gpu.sh`) is not used; no venv is
copied. Differences from the reference environment, by design: the CUDA build of torch
instead of `2.9.1+rocm6.4`, and none of the workstation's GUI/video packages (manim,
PySide6, …), which no experiment imports (`deployment/requirements-cuda.txt`,
`deployment/manifest/reference_env_freeze.txt`).

## 4. Diagnose, test, measure (server)

```bash
deployment/flydeploy doctor     # ~20 s: GPU, sparse CSR matmul vs CPU, emulator + ROM, graph, nulls,
                                #        and the comparison with the reference platform
deployment/flydeploy smoke      # ~6 min: headless episode, harness starts vs stored RAM hashes,
                                #        common_stream checkpoint/resume test, one frozen-play start
deployment/flydeploy bench      # ~1 min: sweeps/s for 3 and 11 brains, emulator steps/s, RAM, VRAM
```

Reports go to `deployment/state/` (bring them home with `pull --state`). The reference
numbers of the workstation are in `deployment/manifest/platform_reference.json` and
`bench_reference_rocm.json` (RX 9060 XT: ~22 sweeps/s for 3 brains, ~6-7 for 11; ~600
emulator steps/s; 0.3-0.7 GB VRAM and ~6.8 GB RAM per process).

**Three levels of agreement, not to be confused.**
1. *Functional compatibility* — `doctor` and `smoke` pass.
2. *Numerical proximity* — `doctor` / `compare platform`: noise-free sweeps of fixed frames
   (motor spike counts, P(jump) of a ROCm checkpoint) and the GPU generator's first draws,
   against the workstation's values. On the workstation itself they match exactly; on
   NVIDIA, small differences are expected (sparse sums in another order, other random
   kernels). **Trajectories are not expected to be bit-identical across platforms.**
3. *Coherence of the scientific measures* — a same-procedure rerun on NVIDIA against the
   ROCm result: `run cs-scripted` then `compare common-stream` (declared tolerance ±0.02 AUC
   at every scoring point, RL_STUDY §12.2), and `run harness-check` then
   `compare harness --file artifacts/runs/harness_v2/validation-rerun-20261009.json`
   (emulator only: must be identical).

## 5. Run (server)

```bash
deployment/flydeploy run cs-scripted --param graph=connectome     # one declared experiment, detached
deployment/flydeploy status                                       # runs + GPU use
deployment/flydeploy summary                                      # progress digest (frames, starts, AUC, ETA, outputs)
deployment/flydeploy logs <run_id> -f                             # follow (Ctrl-C leaves it running)
deployment/flydeploy stop <run_id>                                # ask it to save and exit
deployment/flydeploy wait <run_id>                                # block until it ends (exit code = the run's)
deployment/flydeploy run cs-scripted --param graph=connectome --resume   # continue a stopped resumable run
```

`run` starts only experiments of `deployment/experiments.json`, each tied to its
declaration in the study documents. It runs under a supervisor in **tmux** (or `setsid`
when tmux is absent), so it survives SSH disconnection. Each run
gets `artifacts/deploy_runs/<experiment>[-graph]-<UTC>-<commit>/` with:
`manifest.json` (commit and uncommitted files, Python/torch/CUDA/GPU/driver, seeds,
`configs/base.yaml` sha256 and a copy, sha256 of the graph, null and checkpoint inputs,
command, max duration), `status.json` (state, pid, heartbeat every 10 s, exit code),
`log.txt`, and `outputs/` (every file the run created or changed under `artifacts/runs/`).
A run refuses to start when its outputs exist (`--resume` for resumable experiments,
`--force` to overwrite). `MAX_HOURS` (or `--max-hours`) asks it to stop cleanly when
reached; `stop` does the same on request. Resumable experiments save and exit cleanly:
`common_stream.py` saves its training state at every scoring point and on SIGTERM (exit 75)
and resumes bit for bit (`tests/unit/test_common_stream_resume.py`); the trainer
(`train_mario_sweep.py`) is stopped through its STOP file and resumes from its checkpoint;
frozen play saves after every start and skips finished starts on resume.

**Parallelism.** `MAX_PARALLEL_GPU=1` by default: start with one GPU experiment, look at
`bench` and at the decisions per second of that run, then try two and compare the total
throughput before going further. Each process needs ~7 GB RAM and spawns one emulator
process per brain; 24 vCPU do not mean 24 experiments.

Registered experiments (`flydeploy run <name> --param key=value`):

| name | what (declared in) | ~hours on the workstation |
|---|---|---|
| `selftest` | supervisor check, no science | 0.01 |
| `cs-scripted` | common stream, scripted, graph=connectome/nulllat:0/nulllat:1 (§6.1, §12.2) | 0.5 |
| `cs-permuted` | permuted-label connectome brains (§12) | 0.5 |
| `cs-expert` | expert stream (§14) | 0.5 |
| `probe-merge`, `probe-merge-expert` | merge brains for frozen play | < 0.1 |
| `probe-play-training[-expert]` | frozen play, threshold rule, training suite (§13.1, §14 i) | 1.5 |
| `probe-play-unseen-hold6[-expert]` | frozen play, hold6, unseen suite (§13.3, §14 ii) | 1.5 |
| `probe-fixed-training[-expert]` | S_fixed on the training levels (§13.1, §14) | 0.5 |
| `probe-report` | declared reading, suite=…, rule=…, stream=… (§12-§14) | < 0.1 |
| `harness-check` | harness v2 validation rerun (§13) | 0.1 |
| `stageD` | Stage D, graph=connectome/nulllat:<s> (§17, §21) | 3 |
| `stageD-eval` | frozen competence of a Stage D checkpoint (§11.1) | 1.5 |

## 6. Checkpoints and random states across platforms

Weights are platform-independent data: a ROCm checkpoint loads on NVIDIA
(`map_location="cpu"`) and can initialise a new run (`--init-from`, `--eval-from`,
`frozen_play_probe.py --smoke`). Random-generator states are not a reliable bridge: their
byte format may load, but the streams drawn from them on another platform are not
guaranteed equal (`compare platform` reports whether the first draws match). **A ROCm run
resumed on NVIDIA is therefore a new run initialised from weights, never an exact
continuation**, which is why ROCm checkpoints are kept under `artifacts/rocm_reference/`,
where the trainer does not look for its own checkpoints. **Every arm of a new comparison
(connectome vs nulls, expert vs scripted stream, …) must run on the same platform**: on
NVIDIA, rerun the comparison arms there (e.g. `cs-scripted` and `cs-expert` for §14; all
graphs of `stageD` for §21) and use the ROCm results only for the coherence checks.

## 7. Bring results home and stop paying (your PC, then the provider)

```bash
deployment/flydeploy pull                 # all runs -> artifacts/remote/<host>/deploy_runs/
deployment/flydeploy pull --run <run_id> --state
```

`pull` never writes into your `artifacts/runs/`. Then **stop or destroy the instance in the
provider's console**: on Vast.ai a stopped instance still bills its disk; destroying it
ends all charges (and deletes its data — pull first). On Runpod, stopping a pod keeps
billing the volume; terminate it to end all charges.

## 8. Disk and time estimates

- Server disk: venv with CUDA torch ≈ 6-7 GB, uv cache ≈ 4-5 GB (`uv cache clean` after
  install), data 0.5-2 GB, outputs of the planned §14 sequence ≈ 1-3 GB (runs also copy
  their outputs into their run folder): **≥ 30 GB recommended, 15 GB minimum.**
- Transfer: ≈ 530 MB by default, ≈ 1.46 GB with every set; the ROM 40 kB.
- Time: install 5-15 min (downloads), `smoke` ~6 min, `bench` ~1 min. Throughput on the
  3090 is unknown until `bench` runs (the sweeps are memory-bound sparse products; a faster
  card is expected but not assumed).

## 9. What has been checked, and what remains on NVIDIA

Checked on the workstation (ROCm, 2026-10-09): shell syntax; lint; unit tests of the tool
(`tests/unit/test_deploy.py`: configuration, sensitive-file refusal, experiment resolution,
inventory destinations, manifest checksums); `inventory`; `pack` (no sensitive file in the
bundle); `push` / `push-rom` in `--dry-run` (rsync and scp variants); `verify`; `fetch-data`
on valid files (pins match; public URLs answer with the pinned sizes); `reference`;
`doctor` (same-platform check exact: identical draws, 100% identical spike counts, P(jump)
difference 0); `smoke` (passed, 5.7 min); `bench`; `run` / `status` / `logs` / `stop` /
max duration / overwrite refusal with `selftest`, with and without tmux; the
`common_stream.py` resume test (interrupted + resumed = uninterrupted, bit for bit).

To do on the NVIDIA server, in this order: `install`; `verify`; `import-rom`; `doctor`
(blocking checks, numerical proximity); `smoke`; `bench`; `run cs-scripted --param
graph=connectome` and `compare common-stream`; `run harness-check` and `compare harness`;
only then the declared experiments.

## 10. Troubleshooting

- `REMOTE_HOST is not set` — fill `deployment/deploy.env`.
- `verify`: MISSING / MISMATCH — rerun `push` (resumes) or `fetch-data`; a mismatch after a
  complete transfer means the local file changed since `inventory`: rerun `inventory`.
- `fetch-data`: interrupted — rerun it (resumes the `.part` file); sha256 differs — the
  public file changed: use the transfer route.
- `doctor`: no CUDA device — wrong image or `CUDA_VISIBLE_DEVICES`; driver < 525 — choose
  another host (drivers are the provider's).
- `run`: outputs exist — `--resume` (resumable) or `--force`; another GPU run active — wait,
  or raise `MAX_PARALLEL_GPU` after `bench`; supervisor did not start — `--no-tmux`.
- A run shows `DEAD (supervisor gone)` — the instance restarted or the process was killed:
  rerun with `--resume` (resumable experiments continue from their last save).
