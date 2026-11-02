#!/usr/bin/env python3
"""flydeploy: run SMB_Flywire_Speedrun on a remote Linux/NVIDIA GPU server.

One tool, two sides. Read deployment/README.md for the full path.

  your PC     inventory  pack  push  push-rom  pull  remote  reference
  the server  install  fetch-data  import-rom  verify  doctor  smoke  bench
              run  status  summary  wait  stop  logs  compare

The local side, `verify` and the run supervisor use the standard library only; `doctor`,
`smoke`, `bench`, `compare` and `reference` need the project venv (`deployment/flydeploy`
picks `.venv/bin/python`). Nothing here rents or stops a cloud instance: stopping a run does
NOT stop the provider's billing.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deployment"
MANIFEST = DEPLOY / "manifest" / "data_manifest.json"
PLATFORM_REF = DEPLOY / "manifest" / "platform_reference.json"
EXPERIMENTS = DEPLOY / "experiments.json"
CODE_VERSION = DEPLOY / "CODE_VERSION.json"
PACK_DIR = ROOT / "artifacts" / "deploy"
RUNS_DIR = ROOT / "artifacts" / "deploy_runs"
STATE_DIR = DEPLOY / "state"                     # server-side reports (gitignored)
ROCM_REF = "artifacts/rocm_reference"            # where ROCm-made reference files land remotely

DEFAULTS = {
    "REMOTE_HOST": "", "REMOTE_PORT": "22", "REMOTE_USER": "root",
    "REMOTE_DIR": "/workspace/SMB_Flywire_Speedrun", "SSH_KEY": "", "SSH_EXTRA": "",
    "LOCAL_ROM": "roms/Super Mario Bros. (World).nes",
    "TRANSFER_SETS": "core,nulls-lateral,reference,rocm-reference",   # core = the graph alone
    "TORCH_VERSION": "2.9.1", "TORCH_CUDA": "auto",
    "MAX_HOURS": "8", "MAX_PARALLEL_GPU": "1", "PULL_DIR": "artifacts/remote",
}

# Sensitive: never in a bundle, an archive, a push or a pull (checked on the finished bundle).
SENSITIVE = ["*.nes", "*.sfc", "roms/*.zip", "*.md5", ".env", "deployment/deploy.env", "*.pem", "*.key",
             "id_rsa*", "id_ed25519*", "*.p12", ".claude/*", ".claude.json", ".anthropic/*", "*credentials*",
             "gpucore.*"]
# Also not added to the bundle as uncommitted extras (environments, caches, data, outputs).
DENY = SENSITIVE + [".venv/*", "__pycache__/*", "*.pyc", "*.part", "*.tmp", "core", "core.*", ".git/*",
                    ".pytest_cache/*", ".ruff_cache/*", ".ipynb_checkpoints/*", "artifacts/*", "data/*",
                    "media/*", "roms/*", "deployment/state/*"]

# What to transfer, download or provide privately: (set, path or glob, action, origin, used by).
# "rocm-reference" files are ROCm-made and land under artifacts/rocm_reference/ on the server,
# so they never collide with new NVIDIA outputs (they are for comparisons only).
SPEC = [
    ("core", "data/connectome/malecns.npz", "transfer",
     "built locally from MaleCNS v1.0 (scripts/build_graph.py); its sha256 keys the null caches",
     "every run (the measured wiring)"),
    ("core", "data/connectome/malecns_ledger.json", "transfer", "written with the graph", "build ledger"),
    ("core", "data/connectome/malecns_stats.json", "transfer", "written with the graph", "graph statistics"),
    ("nulls-lateral", "artifacts/cache/nulls/malecns_*_depth_stratified_lateral_s*.npz", "transfer",
     "built locally (randomize.depth_stratified_lateral, seeds 0-2), keyed to the graph hash",
     "nulllat:<s> runs (RL_STUDY §12-§14, PERCEPTION_CREDIT_STUDY §21)"),
    ("nulls-primary", "artifacts/cache/nulls/malecns_*_depth_stratified_s*.npz", "transfer",
     "built locally (randomize.depth_stratified, seeds 0-3), keyed to the graph hash",
     "null:<s> runs (Stage D nulls, RL_STUDY §9)"),
    ("reference", "artifacts/runs/reference/always_run_train.json", "transfer",
     "computed locally (scripted policy, emulator)", "Δx_rand metric (learning_curves.py)"),
    ("reference", "artifacts/runs/reference/random_jumper_train.json", "transfer",
     "computed locally (random_jumper_table.py)", "Δx_rand metric"),
    ("reference", "artifacts/runs/reference/scripted_takeoff2_train.json", "transfer",
     "computed locally (cached positive reference)", "learning_curves.scripted_reference"),
    ("reference", "artifacts/runs/harness_v2/starts-20261009.json", "transfer",
     "play harness v2 starts (RL_STUDY §13; regenerable, RAM hashes recorded)", "frozen play (§13-§14)"),
    ("reference", "artifacts/runs/harness_v2/validation-20261009.json", "transfer",
     "harness v2 validation (scripted, emulator)", "probe report (§13.1, §14)"),
    ("reference", "artifacts/runs/harness_v2/validation-hold6-20261009.json", "transfer",
     "harness v2 hold6 validation", "probe report (§13.3, §14)"),
    ("reference", "artifacts/runs/oracle_criterion_check-takeoff2-k5-20261004.json", "transfer",
     "scripted expert, harness v1", "probe report v1 (§12)"),
    ("rocm-reference", "artifacts/runs/checkpoints/common_stream_connectome.pt", "transfer",
     "ROCm-trained common-stream brains (§12.2)", "cross-platform checks; §13.3 on ROCm weights"),
    ("rocm-reference", "artifacts/runs/checkpoints/common_stream_nulllat0.pt", "transfer", "ROCm (§12.2)", "same"),
    ("rocm-reference", "artifacts/runs/checkpoints/common_stream_nulllat1.pt", "transfer", "ROCm (§12.2)", "same"),
    ("rocm-reference", "artifacts/runs/checkpoints/common_stream_perm_connectome.pt", "transfer", "ROCm (§12.2)", "same"),
    ("rocm-reference", "artifacts/runs/checkpoints/frozen_probe_connectome.pt", "transfer",
     "ROCm, merged for play (§12)", "smoke: checkpoint load and one-start play"),
    ("rocm-reference", "artifacts/runs/checkpoints/frozen_probe_nulllat0.pt", "transfer", "ROCm (§12)", "comparisons"),
    ("rocm-reference", "artifacts/runs/checkpoints/frozen_probe_nulllat1.pt", "transfer", "ROCm (§12)", "comparisons"),
    ("rocm-reference", "artifacts/runs/common_stream-20261007.json", "transfer",
     "ROCm AUC curves of the common stream (§6.1)", "compare: same-procedure NVIDIA rerun"),
    ("rocm-reference", "artifacts/runs/common_stream-connectome-20261009.json", "transfer", "ROCm (§12.2)", "compare"),
    ("rocm-reference", "artifacts/runs/common_stream-nulllat0-20261009.json", "transfer", "ROCm (§12.2)", "compare"),
    ("rocm-reference", "artifacts/runs/common_stream-nulllat1-20261009.json", "transfer", "ROCm (§12.2)", "compare"),
    ("rocm-reference", "artifacts/runs/common_stream-perm-connectome-20261009.json", "transfer", "ROCm (§12.2)", "compare"),
    ("rocm-reference", "artifacts/runs/frozen_probe/reportv2-training-20261009.json", "transfer",
     "ROCm §13.1 declared result", "compare (§13.1 rerun on NVIDIA)"),
    ("rocm-stageD", "artifacts/runs/checkpoints/mario_stageD.pt", "transfer",
     "ROCm Stage D connectome (20,000 decisions)", "frozen competence evaluations (--eval-from)"),
    ("rocm-stageD", "artifacts/runs/checkpoints/mario_stageD_null?.pt", "transfer",
     "ROCm Stage D depth-stratified nulls 0-2 (20,000 decisions)", "frozen competence evaluations"),
    ("rocm-stageD", "artifacts/runs/checkpoints/mario_stageD_nulllat?.pt", "transfer",
     "ROCm Stage D lateral nulls, paused at 13,651 / 11,641 / 9,879 decisions (§21)",
     "reference only: an NVIDIA continuation would not be exact (README); rerun §21 on NVIDIA"),
    ("public-raw", "data/raw/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather", "download",
     ("https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
     "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"), "graph rebuild check only"),
    ("public-raw", "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather", "download",
     ("https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
     "body-annotations-male-cns-v1.0-minconf-0.5.feather"), "graph rebuild check only"),
    ("public-raw", "data/raw/body-neurotransmitters-male-cns-v1.0.feather", "download",
     ("https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
     "body-neurotransmitters-male-cns-v1.0.feather"), "graph rebuild check only"),
]


# --------------------------------------------------------------------------- utilities
def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def denied(rel: str, patterns=None) -> bool:
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat) for pat in (patterns or DENY))


def load_config(path: Path | None = None) -> dict:
    """deployment/deploy.env (KEY=VALUE, shell syntax without expansion), then the
    environment, over the defaults."""
    cfg = dict(DEFAULTS)
    path = path or DEPLOY / "deploy.env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    for k in DEFAULTS:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def run(cmd: list[str], dry: bool = False, check: bool = True, **kw) -> int:
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return 0
    r = subprocess.run(cmd, **kw, check=False)
    if check and r.returncode != 0:
        die(f"command failed ({r.returncode}): {cmd[0]}")
    return r.returncode


# --------------------------------------------------------------------------- inventory
def expand_spec(root: Path = ROOT) -> list[dict]:
    items = []
    for set_, pat, action, origin, used in SPEC:
        paths = sorted(root.glob(pat)) if any(ch in pat for ch in "*?[") else [root / pat]
        for p in paths:
            rel = p.relative_to(root).as_posix()
            dest = rel
            if set_ in ("rocm-reference", "rocm-stageD"):
                dest = f"{ROCM_REF}/{rel.removeprefix('artifacts/runs/')}"
            items.append({"set": set_, "path": rel, "dest": dest, "action": action, "origin": origin,
                          "used_by": used, "present_locally": p.exists()})
    return items


def git(*args, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        die(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.rstrip("\n")         # not strip(): porcelain lines start with a status column


def cmd_inventory(a):
    """Size and sha256 of every file the server needs beyond the code; written to
    deployment/manifest/data_manifest.json and deployment/manifest/INVENTORY.md."""
    items = expand_spec()
    for it in items:
        p = ROOT / it["path"]
        if p.exists():
            it["size"] = p.stat().st_size
            it["sha256"] = sha256(p)
        else:
            it["size"], it["sha256"] = None, None
    rom = ROOT / load_config()["LOCAL_ROM"]
    totals = {}
    for it in items:
        if it["size"]:
            t = totals.setdefault(it["set"], {"files": 0, "bytes": 0, "action": it["action"]})
            t["files"] += 1
            t["bytes"] += it["size"]
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no").splitlines()
    man = {"what": "files the server needs beyond the code (deployment/README.md)", "generated_utc": now(),
           "commit": commit, "tracked_changes_not_committed": dirty, "items": items, "totals": totals,
           "rom": {"expected_sha1_without_header": "facee9c577a5262dbe33ac4930bb0b58c8c037f7",
                   "local_present": rom.exists(), "action": "private (push-rom), never packed or committed"}}
    atomic_json(MANIFEST, man)
    lines = ["# Deployment inventory (generated by `flydeploy inventory`; do not edit)", "",
             f"Commit `{commit[:12]}`, generated {man['generated_utc']}.", "",
             "| set | action | files | size |", "|---|---|---|---|"]
    for s, t in totals.items():
        lines.append(f"| {s} | {t['action']} | {t['files']} | {t['bytes'] / 1e6:,.1f} MB |")
    lines += ["", "| set | path | size | sha256 (first 16) | origin | used by |", "|---|---|---|---|---|---|"]
    for it in items:
        size = f"{it['size'] / 1e6:,.1f} MB" if it["size"] else "missing locally"
        lines.append(f"| {it['set']} | `{it['path']}` | {size} | {(it['sha256'] or '')[:16]} | "
                     f"{it['origin']} | {it['used_by']} |")
    (DEPLOY / "manifest" / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    for s, t in totals.items():
        print(f"{s:16s} {t['action']:9s} {t['files']:3d} files {t['bytes'] / 1e6:9,.1f} MB")
    missing = [it["path"] for it in items if not it["size"] and it["action"] == "transfer"]
    if missing:
        print("missing locally (cannot be transferred):", *missing, sep="\n  ")
    print(f"-> {MANIFEST.relative_to(ROOT)}")


# --------------------------------------------------------------------------- pack
def cmd_pack(a):
    """Code bundle: `git archive HEAD` + tracked files modified but not committed + untracked,
    non-ignored files (each listed in CODE_VERSION.json), minus the deny list."""
    commit = git("rev-parse", "HEAD")
    modified = [l[3:] for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l]
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    extra = [p for p in modified + untracked if p and not denied(p) and (ROOT / p).is_file()]
    skipped = [p for p in modified + untracked if p and denied(p)]
    dirty = bool(modified)
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    name = f"flymario-code-{commit[:12]}{'-dirty' if dirty or extra else ''}"
    tar_path = PACK_DIR / f"{name}.tar.gz"
    tmp_tar = PACK_DIR / f"{name}.tar"
    with open(tmp_tar, "wb") as f:
        subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, stdout=f, check=True)
    version = {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "packed_utc": now(),
               "uncommitted_files_included": extra, "excluded_by_deny_list": skipped, "dirty": dirty or bool(extra)}
    with tarfile.open(tmp_tar, "a") as t:
        for p in extra:
            t.add(ROOT / p, arcname=p)
        data = json.dumps(version, indent=1).encode()
        info = tarfile.TarInfo("deployment/CODE_VERSION.json")
        info.size, info.mtime = len(data), int(time.time())
        import io
        t.addfile(info, io.BytesIO(data))
    with open(tmp_tar, "rb") as src, __import__("gzip").open(tar_path, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    tmp_tar.unlink()
    with tarfile.open(tar_path) as t:
        bad = [m.name for m in t.getmembers() if denied(m.name, SENSITIVE)]
    if bad:
        tar_path.unlink()
        die(f"deny-listed files reached the bundle: {bad[:5]}")
    version["bundle"] = tar_path.name
    version["bundle_sha256"] = sha256(tar_path)
    version["bundle_bytes"] = tar_path.stat().st_size
    atomic_json(PACK_DIR / f"{name}.json", version)
    print(f"{tar_path.relative_to(ROOT)}  {version['bundle_bytes'] / 1e6:.1f} MB  sha256 {version['bundle_sha256'][:16]}")
    if extra:
        print("uncommitted files included:", *extra, sep="\n  ")
    if skipped:
        tops = {}
        for p in skipped:
            tops[p.split("/")[0]] = tops.get(p.split("/")[0], 0) + 1
        print("not packed (deny list):", ", ".join(f"{k} ({v} files)" for k, v in sorted(tops.items())))
    return tar_path, version


# --------------------------------------------------------------------------- remote helpers
def ssh_base(cfg) -> list[str]:
    if not cfg["REMOTE_HOST"]:
        die("REMOTE_HOST is not set: copy deployment/deploy.env.example to deployment/deploy.env and fill it")
    cmd = ["ssh", "-p", cfg["REMOTE_PORT"], "-o", "ServerAliveInterval=30"]
    if cfg["SSH_KEY"]:
        cmd += ["-i", os.path.expanduser(cfg["SSH_KEY"])]
    cmd += shlex.split(cfg["SSH_EXTRA"])
    return cmd + [f"{cfg['REMOTE_USER']}@{cfg['REMOTE_HOST']}"]


def rsync_e(cfg) -> str:
    parts = ["ssh", "-p", cfg["REMOTE_PORT"], "-o", "ServerAliveInterval=30"]
    if cfg["SSH_KEY"]:
        parts += ["-i", os.path.expanduser(cfg["SSH_KEY"])]
    parts += shlex.split(cfg["SSH_EXTRA"])
    return " ".join(shlex.quote(p) for p in parts)


def remote_target(cfg, path: str = "") -> str:
    return f"{cfg['REMOTE_USER']}@{cfg['REMOTE_HOST']}:{cfg['REMOTE_DIR'].rstrip('/')}/{path}"


def copy_cmd(cfg, src: str, dest: str, has_rsync: bool, extra=()) -> list[str]:
    """rsync (resumable with --partial) when the server has it, else scp (not resumable)."""
    if has_rsync:
        return ["rsync", "-a", "--no-owner", "--no-group", "--partial", *extra, "-e", rsync_e(cfg), src, dest]
    ssh = ssh_base(cfg)
    opts = [x for x in ssh[1:-1] if x != "-p"]
    return ["scp", "-P", cfg["REMOTE_PORT"], *[o for o in opts if o != cfg["REMOTE_PORT"]], src, dest]


def remote_has_rsync(cfg, dry: bool) -> bool:
    if dry:
        return True
    r = subprocess.run(ssh_base(cfg) + ["command -v rsync"], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print("warning: no rsync on the server yet: using scp (not resumable). "
              "`apt-get install -y rsync` there, or run `deployment/flydeploy install` first.")
    return r.returncode == 0


def cmd_push(a):
    cfg = load_config()
    ssh_base(cfg)                                   # fails early if REMOTE_HOST is not set
    sets = [s for s in (a.sets or cfg["TRANSFER_SETS"]).split(",") if s]
    if not MANIFEST.exists():
        die("no data manifest: run `deployment/flydeploy inventory` first")
    man = json.loads(MANIFEST.read_text())
    tar_path, _version = cmd_pack(a) if not a.data_only else (None, None)
    rd = shlex.quote(cfg["REMOTE_DIR"])
    run(ssh_base(cfg) + [f"mkdir -p {rd}/.deploy"], a.dry_run)
    has_rsync = remote_has_rsync(cfg, a.dry_run)
    if tar_path:
        for f in (tar_path, tar_path.with_suffix("").with_suffix(".json"), MANIFEST):
            run(copy_cmd(cfg, str(f), remote_target(cfg, ".deploy/"), has_rsync), a.dry_run)
        run(ssh_base(cfg) + [(f"cd {rd} && tar -xzf .deploy/{shlex.quote(tar_path.name)} && "
                              f"mkdir -p deployment/manifest && cp .deploy/data_manifest.json deployment/manifest/")],
            a.dry_run)
    items = [it for it in man["items"] if it["set"] in sets and it["action"] == "transfer"]
    missing = [it["path"] for it in items if not (ROOT / it["path"]).exists()]
    if missing:
        die("listed for transfer but missing locally: " + ", ".join(missing))
    dirs = sorted({str(Path(it["dest"]).parent) for it in items})
    if dirs:
        run(ssh_base(cfg) + ["mkdir -p " + " ".join(f"{rd}/{shlex.quote(d)}" for d in dirs)], a.dry_run)
    total = sum(it["size"] or 0 for it in items)
    print(f"transferring {len(items)} files, {total / 1e6:,.1f} MB (sets: {', '.join(sets)})")
    for it in items:     # one copy per file: resumable with rsync (--partial), verified below by sha256
        run(copy_cmd(cfg, str(ROOT / it["path"]), remote_target(cfg, it["dest"]), has_rsync, ("--info=progress2",)),
            a.dry_run)
    vbin = f"cd {rd} && python3 deployment/flydeploy.py verify --sets {shlex.quote(','.join(sets))}"
    run(ssh_base(cfg) + [vbin], a.dry_run, check=False)


def cmd_sync_code(a):
    """Incremental code update after the first `push`: rsync sends only the changed bytes of
    the working tree's tracked files (minus the sensitive patterns) and of untracked,
    non-ignored files (minus the deny list), plus CODE_VERSION.json as for a bundle. Data,
    checkpoints and results on the server are untouched (nothing is deleted there)."""
    cfg = load_config()
    commit = git("rev-parse", "HEAD")
    modified = [l[3:] for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l]
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    tracked = [p for p in git("ls-files").splitlines() if p and (ROOT / p).is_file() and not denied(p, SENSITIVE)]
    extra = [p for p in untracked if p and (ROOT / p).is_file() and not denied(p)]
    files = sorted(set(tracked + extra) - {"deployment/CODE_VERSION.json"})
    version = {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "packed_utc": now(),
               "uncommitted_files_included": sorted(set(modified) & set(files)) + extra,
               "dirty": bool(modified or extra), "method": "sync-code (rsync of the working tree)"}
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    lst = PACK_DIR / "sync-code.files"
    lst.write_text("\n".join(files) + "\n")
    ver = PACK_DIR / "CODE_VERSION.sync.json"
    atomic_json(ver, version)
    print(f"syncing {len(files)} files at {commit[:12]}{' (dirty)' if version['dirty'] else ''}")
    run(["rsync", "-a", "--no-owner", "--no-group", "--partial", "--stats", f"--files-from={lst}", "-e", rsync_e(cfg),
         str(ROOT) + "/", remote_target(cfg)], a.dry_run)
    run(["rsync", "-a", "--no-owner", "--no-group", "-e", rsync_e(cfg), str(ver),
         remote_target(cfg, "deployment/CODE_VERSION.json")], a.dry_run)


def cmd_push_rom(a):
    """The ROM goes alone, privately: never into a bundle, an archive or git."""
    cfg = load_config()
    rom = Path(cfg["LOCAL_ROM"])
    rom = rom if rom.is_absolute() else ROOT / rom
    if not rom.is_file():
        die(f"LOCAL_ROM not found: {rom} (set LOCAL_ROM in deployment/deploy.env)")
    data = rom.read_bytes()
    sha1 = hashlib.sha1(data[16:] if data[:4] == b"NES\x1a" else data).hexdigest()
    if sha1 != "facee9c577a5262dbe33ac4930bb0b58c8c037f7":
        die(f"{rom.name}: sha1 {sha1} is not the expected US/World ROM (docs/ROM.md)")
    rd = shlex.quote(cfg["REMOTE_DIR"])
    run(ssh_base(cfg) + [f"mkdir -p {rd}/roms && chmod 700 {rd}/roms"], a.dry_run)
    run(copy_cmd(cfg, str(rom), remote_target(cfg, "roms/"), remote_has_rsync(cfg, a.dry_run), ("--chmod=F600",)),
        a.dry_run)
    run(ssh_base(cfg) + [f"chmod 600 {rd}/roms/*"], a.dry_run)
    print("next, on the server: deployment/flydeploy import-rom")


def cmd_pull(a):
    """Bring server results home without touching local results: into PULL_DIR/<host>/."""
    cfg = load_config()
    dest = ROOT / cfg["PULL_DIR"] / cfg["REMOTE_HOST"]
    dest.mkdir(parents=True, exist_ok=True)
    src = "artifacts/deploy_runs/" + (f"{a.run}/" if a.run else "")
    run(["rsync", "-a", "--partial", "-e", rsync_e(cfg), remote_target(cfg, src),
         str(dest / "deploy_runs" / (a.run or "")) + "/"], a.dry_run)
    if not a.run:     # the monitoring record is small: always
        run(["rsync", "-a", "-e", rsync_e(cfg), remote_target(cfg, "artifacts/monitor/"), str(dest / "monitor") + "/"],
            a.dry_run, check=False)
    if a.state:
        run(["rsync", "-a", "-e", rsync_e(cfg), remote_target(cfg, "deployment/state/"), str(dest / "state") + "/"],
            a.dry_run)
    print(f"-> {dest}")


def cmd_remote(a):
    """Run a flydeploy subcommand (or any shell command with --shell) in REMOTE_DIR."""
    cfg = load_config()
    rd = shlex.quote(cfg["REMOTE_DIR"])
    inner = " ".join(shlex.quote(x) for x in a.args)
    cmd = f"cd {rd} && {inner}" if a.shell else f"cd {rd} && deployment/flydeploy {inner}"
    sys.exit(run(ssh_base(cfg) + ["-t", cmd] if a.tty else ssh_base(cfg) + [cmd], a.dry_run, check=False))


# --------------------------------------------------------------------------- server side
def cmd_install(a):
    cfg = load_config()
    env = dict(os.environ, TORCH_VERSION=cfg["TORCH_VERSION"], TORCH_CUDA=cfg["TORCH_CUDA"])
    sys.exit(subprocess.run(["bash", str(DEPLOY / "install_remote.sh"), *(["--cpu-only"] if a.cpu_only else [])],
                            cwd=ROOT, env=env, check=False).returncode)


def cmd_verify(a):
    """Every manifest file of the chosen sets is present at its destination with its sha256."""
    if not MANIFEST.exists():
        die("deployment/manifest/data_manifest.json missing (push it, or run inventory locally)")
    man = json.loads(MANIFEST.read_text())
    sets = [s for s in (a.sets or load_config()["TRANSFER_SETS"]).split(",") if s]
    bad = 0
    for it in man["items"]:
        if it["set"] not in sets or not it["sha256"]:
            continue
        p = ROOT / it["dest"]
        if not p.exists():
            print(f"  MISSING   {it['dest']}  ({it['action']}: "
                  f"{'push from your PC' if it['action'] == 'transfer' else 'deployment/flydeploy fetch-data'})")
            bad += 1
        elif p.stat().st_size != it["size"] or (not a.quick and sha256(p) != it["sha256"]):
            print(f"  MISMATCH  {it['dest']}  (size/sha256 differ from the manifest: re-transfer it)")
            bad += 1
        else:
            print(f"  ok        {it['dest']}")
    if bad:
        die(f"{bad} file(s) missing or different", 2)
    print("all listed files present and verified" + (" (sizes only)" if a.quick else ""))


def cmd_fetch_data(a):
    """Public raw MaleCNS tables (pinned by sha256 from the reference copies); resumable;
    checks free disk space first. --rebuild-graph rebuilds data/connectome/malecns.npz from
    them only if it is absent, and refuses a result whose sha256 differs from the reference
    (the null caches are keyed to it)."""
    man = json.loads(MANIFEST.read_text())
    items = [it for it in man["items"] if it["set"] == "public-raw"]
    need = sum(it["size"] for it in items if not (ROOT / it["path"]).exists())
    free = shutil.disk_usage(ROOT).free
    if need * 1.2 > free:
        die(f"not enough disk: need ~{need * 1.2 / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    for it in items:
        dest = ROOT / it["path"]
        if dest.exists() and dest.stat().st_size == it["size"] and sha256(dest) == it["sha256"]:
            print(f"[ok  ] {it['path']}")
            continue
        part = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[get ] {it['path']} ({it['size'] / 1e6:.0f} MB)")
        r = subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(part), it["origin"]], check=False)
        if r.returncode != 0:
            die(f"download interrupted ({r.returncode}); rerun `flydeploy fetch-data` to resume from {part.name}")
        got = sha256(part)
        if got != it["sha256"]:
            die(f"{part.name}: sha256 {got[:16]} != pinned {it['sha256'][:16]} — the public file changed or the "
                "download is corrupt; remove the .part file and retry, or transfer the graph instead")
        os.replace(part, dest)
    if a.rebuild_graph:
        graph = next(it for it in man["items"] if it["path"] == "data/connectome/malecns.npz")
        target = ROOT / graph["path"]
        if target.exists():
            die(f"{graph['path']} already exists: not rebuilding over it (remove it first to test a rebuild)")
        py = ROOT / ".venv/bin/python"
        run([str(py), "scripts/build_graph.py"], cwd=ROOT)
        got = sha256(target)
        if got != graph["sha256"]:
            bad = target.with_name("malecns.REBUILT-DIFFERENT.npz")
            os.replace(target, bad)
            die(f"rebuilt graph sha256 {got[:16]} != reference {graph['sha256'][:16]}: moved to {bad.name}; "
                "the null caches would not match it — transfer data/connectome/malecns.npz instead")
        print("rebuilt graph identical to the reference (sha256)")


def venv_python() -> str:
    p = ROOT / ".venv/bin/python"
    if not p.exists():
        die("no .venv yet: run `deployment/flydeploy install` first")
    return str(p)


def cmd_import_rom(a):
    roms = list((ROOT / "roms").glob("*.nes")) + list((ROOT / "roms").glob("*.zip"))
    if not roms:
        die("no ROM in roms/: run `deployment/flydeploy push-rom` from your PC")
    sys.exit(subprocess.run([venv_python(), "scripts/import_rom.py"], cwd=ROOT, check=False).returncode)


# --------------------------------------------------------------------------- checks (project imports)
def _project():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts" / "gates"))
    sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))


def platform_info() -> dict:
    import platform

    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None),
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_available": torch.cuda.is_available(), "cpus": os.cpu_count()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 1e9, 1), gpu_count=torch.cuda.device_count())
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False)
        info["nvidia_smi"] = out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = None
    try:
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        info["ram_total_gb"] = round(int(mem["MemTotal"].split()[0]) / 1e6, 1)
        info["ram_available_gb"] = round(int(mem["MemAvailable"].split()[0]) / 1e6, 1)
    except (OSError, KeyError, ValueError):
        pass
    info["disk_free_gb"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)
    return info


def rng_values(seed: int, n: int = 8) -> list[float]:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    return [round(float(x), 6) for x in torch.randn(n, generator=g, device=dev).cpu()]


def reference_frames(n: int = 6):
    """Deterministic frames: Level1-1, RIGHT+B every decision, every 10th decision kept."""
    from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
    env = MarioEnv(EnvConfig(state="Level1-1"))
    prev, _ = env.reset()
    out = []
    for i in range(10 * n):
        obs, *_ = env.step(ACTION_NAMES.index("RIGHT+B"))
        if (i + 1) % 10 == 0:
            out.append((prev, obs))
        prev = obs
    env.close()
    return out


def sweep_counts(frames, ckpt: Path | None = None):
    """Noise-free sweeps (no random generator involved): motor spike counts of the untrained
    connectome and, with `ckpt`, P(jump) of the first three brains of that checkpoint."""
    import numpy as np
    import torch

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.env.retro_env import ACTION_NAMES
    from flymario.env.vision import RetinotopicVision
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.policy.readout import FixedRandomProjection, block_seeds
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    g, W, motor = graphsel.load("connectome")
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=0.0)
    K = int(config.get("timing.ticks_per_decision"))
    b = PlasticLIF(W, motor, params, seeds=[1, 2, 3], cache_key=("deploy_ref", "connectome"))
    if ckpt is not None:
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        b.set_weights(st["brain"]["m"][:3].to(b.dev))
    ro = block_seeds(3)
    R = torch.as_tensor(np.stack([np.asarray(FixedRandomProjection(b.M, len(ACTION_NAMES), r).W) for r in ro]),
                        device=b.dev)
    vis = RetinotopicVision(g)
    counts, pj = [], []
    jumps = [ACTION_NAMES.index(x) for x in ("RIGHT+A", "RIGHT+A+B", "A")]
    for prev, obs in frames:
        vis.reset()
        vis.encode(prev)
        cur = torch.as_tensor(np.repeat(vis.encode(obs).astype(np.float32)[:, None], 3, 1), device=b.dev)
        c, _ = sweep_once(b, params, cur, K, 0.5, False)
        counts.append(c[:, 0].cpu().numpy().astype(int).tolist())
        z = torch.einsum("bam,mb->ba", R, c) / 2.0
        pj.append([round(float(x), 6) for x in torch.softmax(z, 1)[:, jumps].sum(1).cpu()])
    return counts, pj


def cmd_reference(a):
    """Local (reference platform) values for the cross-platform checks: random draws of the
    GPU generator and noise-free sweeps, untrained and with the ROCm common-stream brains."""
    _project()
    frames = reference_frames()
    counts, _ = sweep_counts(frames)
    _, pj = sweep_counts(frames, ROOT / "artifacts/runs/checkpoints/frozen_probe_connectome.pt")
    ref = {"what": "reference values for flydeploy doctor/smoke (noise-free sweeps, GPU randn)",
           "generated_utc": now(), "platform": platform_info(),
           "randn_seed_8300": rng_values(8300), "randn_seed_7900": rng_values(7900),
           "frames": "Level1-1, RIGHT+B, decisions 10, 20, ..., 60", "motor_counts_untrained": counts,
           "pjump_frozen_probe_connectome": pj}
    atomic_json(PLATFORM_REF, ref)
    print(f"-> {PLATFORM_REF.relative_to(ROOT)} ({ref['platform'].get('gpu')}, torch {ref['platform']['torch']})")


def compare_platform(ref: dict, ckpt: Path) -> dict:
    import numpy as np
    frames = reference_frames()
    counts, _ = sweep_counts(frames)
    out = {"randn_identical": {"8300": rng_values(8300) == ref["randn_seed_8300"],
                               "7900": rng_values(7900) == ref["randn_seed_7900"]}}
    a, b = np.array(counts), np.array(ref["motor_counts_untrained"])
    out["motor_counts_exact_share"] = round(float((a == b).mean()), 4)
    out["motor_counts_total_here_vs_ref"] = [int(a.sum()), int(b.sum())]
    out["motor_counts_corr"] = round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 4) if a.std() and b.std() else None
    if ckpt.exists():
        _, pj = sweep_counts(frames, ckpt)
        d = np.abs(np.array(pj) - np.array(ref["pjump_frozen_probe_connectome"]))
        out["pjump_max_abs_diff"] = round(float(d.max()), 6)
    return out


def cmd_doctor(a):
    """Environment diagnosis; writes deployment/state/doctor-<utc>.json; exit 1 on a blocking problem."""
    _project()
    rep, blocking = {"utc": now()}, []
    try:
        rep["platform"] = platform_info()
    except Exception as e:  # noqa: BLE001
        rep["platform_error"] = repr(e)
        blocking.append("torch import")
    import torch
    if not torch.cuda.is_available():
        (blocking if not a.allow_cpu else []).append("no CUDA device visible to torch")
    else:
        try:      # the sparse operation the simulator relies on: CSR @ dense, against CPU
            import numpy as np
            import scipy.sparse as sp
            rng = np.random.default_rng(0)
            A = sp.random(2000, 2000, density=0.01, random_state=1, format="csr", dtype=np.float32)
            x = rng.standard_normal((2000, 7)).astype(np.float32)
            T = torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int32)),
                                        torch.from_numpy(A.indices.astype(np.int32)),
                                        torch.from_numpy(A.data), size=A.shape, device="cuda")
            y = (T @ torch.from_numpy(x).cuda()).cpu().numpy()
            rep["sparse_csr_max_abs_err"] = float(np.abs(y - A @ x).max())
            if rep["sparse_csr_max_abs_err"] > 1e-3:
                blocking.append("sparse CSR matmul disagrees with the CPU")
        except Exception as e:  # noqa: BLE001
            rep["sparse_error"] = repr(e)
            blocking.append("sparse CSR matmul on GPU")
    if a.no_emulator:          # studies without the game (e.g. RETINO_ALIGN_V1): no ROM on the server
        rep["emulator"] = {"ok": None, "skipped": "--no-emulator (this study does not use the game)"}
    else:
        _doctor_emulator(rep, blocking)
    _doctor_rest(a, rep, blocking)


def _doctor_emulator(rep: dict, blocking: list) -> None:
    try:
        import retro  # noqa: F401

        from flymario.env.retro_env import EnvConfig, MarioEnv
        env = MarioEnv(EnvConfig(state="Level1-1"))
        env.reset()
        for _ in range(30):
            _o, _r, _t, _tr, info = env.step(2)
        env.close()
        rep["emulator"] = {"ok": True, "mario_x_after_30": info.get("mario_x")}
    except Exception as e:  # noqa: BLE001
        rep["emulator"] = {"ok": False, "error": repr(e)}
        blocking.append("emulator / ROM (deployment/flydeploy import-rom)")


def _doctor_rest(a, rep: dict, blocking: list) -> None:
    import torch
    graph = ROOT / "data/connectome/malecns.npz"
    rep["graph_present"] = graph.exists()
    if not graph.exists():
        blocking.append("data/connectome/malecns.npz missing (push or fetch-data --rebuild-graph)")
    nulls = sorted((ROOT / "artifacts/cache/nulls").glob("*.npz"))
    rep["null_caches"] = [p.name for p in nulls]
    ref_ckpt = ROOT / ROCM_REF / "checkpoints/frozen_probe_connectome.pt"
    if PLATFORM_REF.exists() and graph.exists() and torch.cuda.is_available() and ref_ckpt.exists():
        rep["vs_reference_platform"] = compare_platform(json.loads(PLATFORM_REF.read_text()), ref_ckpt)
    elif not ref_ckpt.exists():
        rep["vs_reference_platform"] = "skipped: rocm-reference set not transferred"
    rep["blocking"] = blocking
    atomic_json(STATE_DIR / f"doctor-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    if blocking:
        die("blocking: " + "; ".join(blocking))
    print("doctor: no blocking problem")


def cmd_smoke(a):
    """A few minutes: headless Mario episode, harness-v2 starts regenerated against the stored
    RAM hashes, the common-stream resume test, one frozen-play start from a ROCm checkpoint."""
    _project()
    rep, failed = {"utc": now()}, []
    t0 = time.time()
    from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
    from flymario.env.schematic import takeoff_label_v2
    env = MarioEnv(EnvConfig(state="Level1-1"))
    env.reset()
    labels, end = 0, None
    for n in range(400):
        _o, _r, term, trunc, info = env.step(ACTION_NAMES.index("RIGHT+B"))
        labels += bool(takeoff_label_v2(env.env.unwrapped.get_ram()))
        if term or trunc or info.get("died"):
            end = n
            break
    env.close()
    rep["episode"] = {"decisions": end, "mario_x": info.get("mario_x"), "label_on": labels, "s": round(time.time() - t0, 1)}
    starts = ROOT / "artifacts/runs/harness_v2/starts-20261009.json"
    if starts.exists():
        import play_harness_v2 as h
        st = json.loads(starts.read_text())["starts"]
        same = [h.make_start(s["level"], s["start"])["ram_md5"] == s["ram_md5"] for s in st[:: max(1, len(st) // 8)]]
        rep["harness_starts_identical"] = f"{sum(same)}/{len(same)}"
        if not all(same):
            failed.append("harness v2 starts differ from the stored RAM hashes (emulator not deterministic here?)")
    else:
        failed.append("harness v2 starts file missing (push the reference set)")
    r = subprocess.run([venv_python(), "-m", "pytest", "-m", "slow", "-q", "-p", "no:cacheprovider",
                        "tests/unit/test_common_stream_resume.py"], cwd=ROOT, capture_output=True, text=True, check=False)
    rep["common_stream_resume_test"] = "passed" if r.returncode == 0 else r.stdout[-1500:]
    if r.returncode != 0:
        failed.append("common_stream resume test")
    ck = ROOT / ROCM_REF / "checkpoints/frozen_probe_connectome.pt"
    smoke_out = ROOT / "artifacts/runs/frozen_probe/playv2-training-connectome-smoke-20261009.json"
    smoke_out.unlink(missing_ok=True)              # a stale smoke file would be resumed
    if ck.exists():
        r = subprocess.run([venv_python(), "scripts/gates/frozen_play_probe.py", "play", "--graph", "connectome",
                            "--suite", "training", "--smoke", str(ck), "--k", "1"], cwd=ROOT, capture_output=True, text=True, check=False)
        rep["frozen_play_one_start"] = [l for l in r.stdout.splitlines() if l.startswith("Level")][-1:] or r.stderr[-800:]
        if r.returncode != 0:
            failed.append("frozen play, one start")
        smoke_out.unlink(missing_ok=True)
    else:
        failed.append(f"{ck.relative_to(ROOT)} missing (push the rocm-reference set)")
    rep["seconds"] = round(time.time() - t0, 1)
    rep["failed"] = failed
    atomic_json(STATE_DIR / f"smoke-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    if failed:
        die("smoke failed: " + "; ".join(failed))
    print("smoke: passed")


def cmd_bench(a):
    """Throughput and memory of one process: sweeps per second for 3 and 11 brains on the
    connectome and a lateral null, emulator steps per second, peak RAM and VRAM. Suggests a
    parallelism level; it changes nothing by itself (MAX_PARALLEL_GPU stays as configured)."""
    _project()
    import resource

    import numpy as np
    import torch

    from flymario import config
    from flymario.connectome import graphsel
    from flymario.env.retro_env import EnvConfig, MarioEnv
    from flymario.models.spiking_reservoir import LIFParams
    from flymario.training.plastic import PlasticLIF
    from flymario.training.sweep import sweep_once
    rep = {"utc": now(), "platform": platform_info(), "sweeps": []}
    params = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    K = int(config.get("timing.ticks_per_decision"))
    for spec in ("connectome", "nulllat:0"):
        try:
            _g, W, motor = graphsel.load(spec)
        except FileNotFoundError as e:
            rep["sweeps"].append({"graph": spec, "error": str(e)})
            continue
        for B in (3, 11):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            b = PlasticLIF(W, motor, params, seeds=list(range(B)), cache_key=("bench", spec))
            cur = torch.as_tensor(np.random.default_rng(0).random((W.shape[0], B), dtype=np.float32) * 0.2, device=b.dev)
            for observe in (False, True):
                sweep_once(b, params, cur, K, 0.5, observe)          # warm-up
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t = time.time()
                n = 20
                for _ in range(n):
                    sweep_once(b, params, cur, K, 0.5, observe)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                rep["sweeps"].append({"graph": spec, "brains": B, "learning_trace": observe,
                                      "sweeps_per_s": round(n / (time.time() - t), 2)})
            rep["sweeps"][-1]["vram_peak_gb"] = (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                                                 if torch.cuda.is_available() else None)
            del b
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        del W
    env = MarioEnv(EnvConfig(state="Level1-1"))
    env.reset()
    t = time.time()
    for i in range(600):
        _o, _r, term, trunc, _i = env.step(2 if i % 7 else 4)
        if term or trunc:
            env.reset()
    rep["emulator_steps_per_s"] = round(600 / (time.time() - t), 1)
    env.close()
    rep["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    p = rep["platform"]
    vram_job = max((s.get("vram_peak_gb") or 0) for s in rep["sweeps"]) or None
    if vram_job and p.get("vram_gb"):
        by_vram = int(p["vram_gb"] * 0.8 // (vram_job + 1.0))
        by_ram = int((p.get("ram_available_gb") or 0) * 0.7 // max(rep["peak_rss_gb"] + 3.0, 1))
        by_cpu = int((p.get("cpus") or 1) // 4)
        rep["suggested_parallel_jobs"] = max(1, min(by_vram, by_ram, by_cpu))
        rep["suggestion_note"] = ("upper bound from VRAM, RAM and CPUs of single-process measurements; the GPU is "
                                  "shared, so total throughput grows less than linearly: raise MAX_PARALLEL_GPU "
                                  "step by step and compare decisions per second")
    atomic_json(STATE_DIR / f"bench-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json", rep)
    print(json.dumps(rep, indent=1, default=str))


def cmd_compare(a):
    """NVIDIA vs the ROCm reference. `platform`: noise-free sweeps and GPU random draws.
    `common-stream`: AUC curves of a same-procedure rerun against the ROCm curves (declared
    reproduction tolerance of RL_STUDY §12: ±0.02). `harness`: a rerun of the harness-v2
    validation (emulator only) must be identical."""
    _project()
    out = {}
    if a.what == "platform":
        out = compare_platform(json.loads(PLATFORM_REF.read_text()), ROOT / ROCM_REF / "checkpoints/frozen_probe_connectome.pt")
    elif a.what == "common-stream":
        tag = a.graph.replace(":", "")
        here = json.loads((ROOT / f"artifacts/runs/common_stream-{a.prefix}{tag}-20261009.json").read_text())["results"][a.graph]
        ref = json.loads((ROOT / ROCM_REF / f"common_stream-{a.prefix}{tag}-20261009.json").read_text())["results"][a.graph]
        rows = []
        for x, y in zip(here, ref):
            rows.append({"frames": x["frames"], "nvidia": x["auc"], "rocm": y["auc"],
                         "max_abs_diff": round(max(abs(u - v) for u, v in zip(x["auc"], y["auc"])), 4)})
        worst = max(r["max_abs_diff"] for r in rows)
        out = {"graph": a.graph, "points": rows, "worst_abs_diff": worst, "within_0.02": worst <= 0.02}
    elif a.what == "harness":
        here = json.loads(Path(a.file).read_text())["results"]
        ref = json.loads((ROOT / "artifacts/runs/harness_v2" / Path(a.ref).name).read_text())["results"]
        out = {"identical": here == ref}
    print(json.dumps(out, indent=1))
    atomic_json(STATE_DIR / f"compare-{a.what}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json", out)


# --------------------------------------------------------------------------- runs
def experiments() -> dict:
    return json.loads(EXPERIMENTS.read_text())["experiments"]


def expand(template, params: dict):
    if isinstance(template, list):
        return [x for t in template for x in [expand(t, params)] if x != ""]
    return template.format(**params)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def run_status(run_dir: Path) -> dict:
    p = run_dir / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def snapshot(base: Path) -> dict:
    out = {}
    for p in base.rglob("*"):
        if p.is_file():
            try:
                s = p.stat()
                out[p.relative_to(ROOT).as_posix()] = (s.st_mtime_ns, s.st_size)
            except OSError:
                pass
    return out


def resolve_experiment(e: dict, kv: list[str], name: str = "") -> tuple[dict, list[str], list[str]]:
    """Parameters (defaults, then key=value overrides, checked against the choices), the
    command and the output paths of one registry entry."""
    params = dict(e.get("defaults", {}))
    for item in kv:
        if "=" not in item:
            die(f"--param expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        params[k] = v
    params["gtag"] = params.get("graph", "").replace(":", "")
    for k, choices in e.get("choices", {}).items():
        if params.get(k) not in choices:
            die(f"{name}: {k}={params.get(k)!r} not in {choices}")
    g = params.get("graph", "connectome")          # the trainer's run names (train_mario_sweep.py)
    params["runname"] = "mario_stageD" if g == "connectome" else f"mario_stageD_{params['gtag']}"
    params["evalname"] = ("mario_stageD_eval" if g == "connectome" else f"mario_stageD_eval_{params['gtag']}") + "_frozeneval"
    params["x"] = {"": "", "expert": "x", "dagger": "d", "daggerctl": "c", "all": "a", "expertb1": "y"}.get(params.get("stream", ""), "?")  # probe-report output name
    params["rule_suffix"] = f"-{params['rule']}" if params.get("rule") else ""
    cmd = expand(e["cmd"], params)
    for k, flag in {"rule": "--rule", "stream": "--stream"}.items():     # optional flags (probe-report)
        if params.get(k) and flag not in cmd:
            cmd += [flag, params[k]]
    outputs = [o for o in expand(e.get("outputs", []), params) if o]
    return params, cmd, outputs


def cmd_run(a):
    """Start one declared experiment, detached (tmux when available, else setsid), with a
    unique run id, a manifest, a log, a heartbeat and a maximum duration."""
    cfg = load_config()
    exps = experiments()
    if a.experiment not in exps:
        die(f"unknown experiment {a.experiment!r}; known: {', '.join(exps)}")
    e = exps[a.experiment]
    params, cmd, outputs = resolve_experiment(e, a.param or [], a.experiment)
    g = params.get("graph", "connectome")
    existing = [o for o in outputs if (ROOT / o).exists()]
    if existing and not (a.force or (a.resume and e.get("resumable"))):
        die("outputs already exist (refusing to overwrite): " + ", ".join(existing)
            + (" — pass --resume to continue this resumable experiment" if e.get("resumable") else "")
            + " — or --force")
    for need in expand(e.get("needs", []), params):
        if not (ROOT / need).exists():
            die(f"{a.experiment} needs {need} (see deployment/README.md)")
    if e.get("gpu", True):
        busy = [d.name for d in RUNS_DIR.glob("*") if run_status(d).get("state") == "running"
                and run_status(d).get("gpu") and alive(run_status(d).get("supervisor_pid"))]
        if len(busy) >= int(cfg["MAX_PARALLEL_GPU"]) and not a.allow_parallel:
            die(f"{len(busy)} GPU run(s) already running ({', '.join(busy)}); MAX_PARALLEL_GPU="
                f"{cfg['MAX_PARALLEL_GPU']}. Measure with `bench` first, then raise it or pass --allow-parallel")
    code = json.loads(CODE_VERSION.read_text()) if CODE_VERSION.exists() else {}
    commit = code.get("commit") or subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                  text=True, check=False).stdout.strip() or "unknown"
    run_id = f"{a.experiment}{'-' + params['gtag'] if params.get('gtag') else ''}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{commit[:8]}"
    rd = RUNS_DIR / run_id
    rd.mkdir(parents=True, exist_ok=False)
    data_files = ["data/connectome/malecns.npz"]
    if ":" in g:                                   # the null realisation this run reads
        kind = {"null": "depth_stratified", "nulllat": "depth_stratified_lateral",
                "nullin": "depth_stratified_lateral_instrength"}[g.split(":")[0]]
        data_files += [p.relative_to(ROOT).as_posix()
                       for p in (ROOT / "artifacts/cache/nulls").glob(f"malecns_*_{kind}_s{g.split(':')[1]}.npz")]
    data_files += [x for x in expand(e.get("needs", []), params) if x.endswith((".pt", ".npz"))]
    data_ids = {f: (sha256(ROOT / f) if (ROOT / f).exists() else None) for f in data_files}
    cfg_file = ROOT / "configs/base.yaml"
    manifest = {"run_id": run_id, "experiment": a.experiment, "params": params, "cmd": cmd,
                "declared_in": e.get("declared_in"), "seeds": e.get("seeds"), "outputs": outputs,
                "code": {"commit": commit, "dirty": code.get("dirty"), "uncommitted_files": code.get("uncommitted_files_included")},
                "config": {"configs/base.yaml": sha256(cfg_file) if cfg_file.exists() else None},
                "data_sha256": data_ids, "max_hours": float(a.max_hours or cfg["MAX_HOURS"]),
                "created_utc": now(), "resume": bool(a.resume)}
    # server-local history: the bundle's commit says where the code came from; the server
    # repository says what it is now
    sg = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True,  # noqa: E731
                                   check=False).stdout.strip()
    if (ROOT / ".git").exists():
        manifest["code"]["server_git"] = {"head": sg("rev-parse", "HEAD"),
                                          "dirty": sg("status", "--porcelain", "--untracked-files=no").splitlines()}
    try:
        _project()
        manifest["platform"] = platform_info()
    except Exception as ex:  # noqa: BLE001
        manifest["platform"] = {"error": repr(ex)}
    atomic_json(rd / "manifest.json", manifest)
    if cfg_file.exists():
        shutil.copy2(cfg_file, rd / "base.yaml")
    atomic_json(rd / "status.json", {"state": "starting", "gpu": e.get("gpu", True), "created_utc": now()})
    sup = [sys.executable, str(DEPLOY / "flydeploy.py"), "_supervise", str(rd)]
    if shutil.which("tmux") and not a.no_tmux:
        r = subprocess.run(["tmux", "new-session", "-d", "-s", run_id, " ".join(shlex.quote(x) for x in sup)],
                           cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, check=False, start_new_session=True)
        if r.returncode != 0:
            die(f"tmux failed ({r.stderr.strip()}); retry with --no-tmux")
        how = f"tmux attach -t {run_id}   (or: deployment/flydeploy logs {run_id} -f)"
    else:
        subprocess.Popen(sup, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        how = f"deployment/flydeploy logs {run_id} -f"
    for _ in range(30):                            # the supervisor must take over within seconds
        if run_status(rd).get("state") != "starting":
            break
        time.sleep(0.5)
    else:
        die(f"{run_id}: the supervisor did not start (state still 'starting'); see {rd}, retry with --no-tmux")
    if a.print_id:
        print(run_id)
    else:
        print(f"started {run_id}\n  follow: {how}\n  status: deployment/flydeploy status\n"
              f"  stop:   deployment/flydeploy stop {run_id}\n  wait:   deployment/flydeploy wait {run_id}")


def cmd_supervise(a):
    rd = Path(a.run_dir)
    man = json.loads((rd / "manifest.json").read_text())
    exp = experiments().get(man["experiment"], {})
    st = run_status(rd)
    before = snapshot(ROOT / "artifacts/runs")
    env = dict(os.environ, PYTHONUNBUFFERED="1", SDL_VIDEODRIVER="dummy", MPLBACKEND="Agg")
    # One OpenBLAS thread per process (numpy only; GPU work unaffected): 20 emulator workers
    # x 64 BLAS threads exceed a container pid limit of 3,072.
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    t0 = time.time()
    with open(rd / "log.txt", "ab", buffering=0) as log:
        proc = subprocess.Popen(man["cmd"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    st.update(state="running", pid=proc.pid, supervisor_pid=os.getpid(), started_utc=now())
    atomic_json(rd / "status.json", st)
    reason, stop_t = None, None
    grace = float(exp.get("grace_s", 180))
    max_s = man["max_hours"] * 3600

    def ask_stop(why):
        nonlocal reason, stop_t
        reason, stop_t = why, time.time()
        if exp.get("runctl_name"):               # the trainer saves a checkpoint and exits
            (ROOT / "artifacts/runs/live" / expand(exp["runctl_name"], man["params"])).mkdir(parents=True, exist_ok=True)
            (ROOT / "artifacts/runs/live" / expand(exp["runctl_name"], man["params"]) / "STOP").touch()
        else:                                     # common_stream saves its state on SIGTERM
            os.killpg(proc.pid, signal.SIGTERM)
        st.update(stop_requested=why, stop_requested_utc=now())

    while proc.poll() is None:
        time.sleep(10)
        if reason is None and (rd / "STOP").exists():
            ask_stop("requested")
        if reason is None and time.time() - t0 > max_s:
            ask_stop("max_hours")
        if reason and time.time() - stop_t > grace and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            st["killed_after_grace_s"] = grace
        st["heartbeat_utc"] = now()
        st["elapsed_s"] = round(time.time() - t0)
        atomic_json(rd / "status.json", st)
    rc = proc.returncode
    after = snapshot(ROOT / "artifacts/runs")
    changed = [k for k, v in after.items() if before.get(k) != v and "/live/" not in k]
    outdir = rd / "outputs"
    for k in changed:
        src = ROOT / k
        if src.stat().st_size < 3e9:
            dst = outdir / k
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    st.update(exit_code=rc, ended_utc=now(), elapsed_s=round(time.time() - t0), outputs_copied=len(changed),
              state=("finished" if rc == 0 else "stopped" if (reason or rc == 75) else "failed"),
              resumable=bool(exp.get("resumable") and rc != 0))
    atomic_json(rd / "status.json", st)


def cmd_status(a):
    rows = []
    for d in sorted(RUNS_DIR.glob("*")):
        st = run_status(d)
        if not st:
            continue
        state = st.get("state")
        if state == "running" and not alive(st.get("supervisor_pid")):
            state = "DEAD (supervisor gone)"
        rows.append((d.name, state, st.get("elapsed_s"), st.get("heartbeat_utc") or st.get("ended_utc"), st.get("exit_code")))
    for r in rows:
        print(f"{r[0]:60s} {r[1]!s:24s} {str(r[2]) + 's':>8s}  {r[3]}  exit {r[4]}")
    if not rows:
        print("no runs")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv"], check=False)


def cmd_stop(a):
    targets = [d for d in RUNS_DIR.glob("*") if (a.all and run_status(d).get("state") == "running") or d.name == a.run_id]
    if not targets:
        die("no such running run")
    for d in targets:
        (d / "STOP").touch()
        print(f"stop requested: {d.name} (the supervisor asks it to save and exit; see `status`)")
    print("reminder: stopping runs does not stop the instance's billing — stop it from the provider's console")


def cmd_wait(a):
    """Block until a run ends; exit with its exit code (for sequences: run --print-id, then wait)."""
    rd = max(RUNS_DIR.glob("*"), key=lambda d: d.stat().st_mtime) if a.last else RUNS_DIR / (a.run_id or "")
    if not (rd / "status.json").exists():
        die(f"no such run: {rd.name}")
    while True:
        st = run_status(rd)
        if st.get("state") in ("finished", "failed", "stopped"):
            print(f"{rd.name}: {st['state']} (exit {st.get('exit_code')})")
            sys.exit(int(st.get("exit_code") or 0))
        if st.get("state") == "running" and not alive(st.get("supervisor_pid")):
            die(f"{rd.name}: supervisor gone (instance restarted?); rerun with --resume", 3)
        time.sleep(a.every)


def run_progress(rd: Path) -> dict:
    """What a run has done so far, read from its log and its live status (no project import)."""
    import re
    man = json.loads((rd / "manifest.json").read_text())
    exp, params = man["experiment"], man.get("params", {})
    lines = (rd / "log.txt").read_text(errors="replace").splitlines() if (rd / "log.txt").exists() else []
    out = {}
    if exp.startswith("cs-"):
        pts = [l for l in lines if "'frames':" in l]
        if pts:
            m = re.search(r"'frames': (\d+), 'auc': \[([^\]]*)\]", pts[-1])
            if m:
                out["frames"] = f"{m.group(1)}/4000"
                out["test_auc_per_readout"] = m.group(2)
        out.update({k: v for k, v in (("stream_check", next((l for l in lines if l.startswith("expert stream:")), None)),
                                      ("resumed", next((l for l in lines if "resumed at frame" in l), None))) if v})
    elif exp.startswith("probe-play"):
        starts = [l for l in lines if l.startswith("Level")]
        out["starts"] = f"{len(starts)}/64"
        if starts:
            out["last_start"] = starts[-1][:220]
        if any(l.startswith("resumed:") for l in lines):
            out["resumed"] = True
    elif exp.startswith("probe-fixed"):
        out["brains_scored"] = sum(1 for l in lines if re.match(r"^(cs|csx|csperm|untrained)\w*_ro\d \d", l))
    elif exp in ("stageD", "stageD-eval"):
        name = params.get("evalname" if exp == "stageD-eval" else "runname")
        live = ROOT / "artifacts/runs/live" / str(name) / "status.json"
        if live.exists():
            st = json.loads(live.read_text())
            out["decisions_k"] = f"{st.get('completed')}/{st.get('total')}"
            sps = st.get("seconds_per_step")
            if sps and st.get("total") is not None and st.get("completed") is not None:
                out["eta_h"] = round((st["total"] - st["completed"]) * sps / 3600, 2)
    elif exp == "probe-report":
        out["report_tail"] = lines[-12:]
    out["log_tail"] = lines[-3:]
    produced = [o for o in man.get("outputs", []) if (ROOT / o).exists()]
    if produced:
        out["outputs_present"] = produced
    return out


def cmd_summary(a):
    """A compact digest of every run (newest first): state, elapsed time, progress, last log
    lines, outputs; GPU, disk."""
    rows = []
    for d in sorted(RUNS_DIR.glob("*"), key=lambda d: d.stat().st_mtime, reverse=True)[: a.n]:
        st = run_status(d)
        if not st or not (d / "manifest.json").exists():
            continue
        state = st.get("state")
        if state == "running" and not alive(st.get("supervisor_pid")):
            state = "DEAD (supervisor gone: rerun with --resume)"
        rows.append({"run_id": d.name, "state": state, "elapsed_h": round((st.get("elapsed_s") or 0) / 3600, 2),
                     "exit_code": st.get("exit_code"), "heartbeat_utc": st.get("heartbeat_utc"),
                     "stop_requested": st.get("stop_requested"), "progress": run_progress(d)})
    gpu = None
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, check=False)
        gpu = (r.stdout.strip() or None) if r.returncode == 0 else None
    rep = {"utc": now(), "runs": rows, "gpu": gpu, "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1)}
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
        return
    print(f"{rep['utc']}  disk free {rep['disk_free_gb']} GB  GPU {gpu or 'n/a'}")
    for r in rows:
        print(f"\n{r['run_id']}\n  {r['state']}, {r['elapsed_h']} h, exit {r['exit_code']}"
              + (f", stop requested ({r['stop_requested']})" if r["stop_requested"] else ""))
        for k, v in r["progress"].items():
            if isinstance(v, list):
                print(f"  {k}:")
                for x in v:
                    print(f"    {x}")
            else:
                print(f"  {k}: {v}")
    if not rows:
        print("no runs yet")


def cmd_logs(a):
    p = RUNS_DIR / a.run_id / "log.txt"
    if not p.exists():
        die(f"no log for {a.run_id}")
    if a.follow:
        os.execvp("tail", ["tail", "-n", str(a.n), "-F", str(p)])
    lines = p.read_text(errors="replace").splitlines()
    print("\n".join(lines[-a.n:]))


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(prog="flydeploy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p
    add("inventory", cmd_inventory, "local: sizes and sha256 of the files the server needs")
    add("pack", cmd_pack, "local: code bundle (git HEAD + uncommitted files, deny list applied)")
    p = add("push", cmd_push, "local: send the code bundle and the data sets, then verify remotely")
    p.add_argument("--sets", help="comma list (default TRANSFER_SETS)")
    p.add_argument("--data-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = add("sync-code", cmd_sync_code, "local: incremental code update over rsync (after the first push)")
    p.add_argument("--dry-run", action="store_true")
    p = add("push-rom", cmd_push_rom, "local: send your ROM alone (private), then import it on the server")
    p.add_argument("--dry-run", action="store_true")
    p = add("pull", cmd_pull, "local: bring server runs home into PULL_DIR/<host>/ (never over local results)")
    p.add_argument("--run")
    p.add_argument("--state", action="store_true", help="also deployment/state (doctor, smoke, bench reports)")
    p.add_argument("--dry-run", action="store_true")
    p = add("remote", cmd_remote, "local: run a flydeploy subcommand on the server over SSH")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.add_argument("--shell", action="store_true", help="run a plain shell command instead")
    p.add_argument("--tty", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    add("reference", cmd_reference, "local (reference platform): values for the cross-platform checks")
    p = add("install", cmd_install, "server: venv, PyTorch CUDA, dependencies (relaunchable)")
    p.add_argument("--cpu-only", action="store_true")
    p = add("verify", cmd_verify, "server: manifest files present with their sha256")
    p.add_argument("--sets")
    p.add_argument("--quick", action="store_true", help="sizes only")
    p = add("fetch-data", cmd_fetch_data, "server: public raw MaleCNS tables (pinned, resumable)")
    p.add_argument("--rebuild-graph", action="store_true")
    add("import-rom", cmd_import_rom, "server: import the pushed ROM (scripts/import_rom.py)")
    p = add("doctor", cmd_doctor, "server: environment diagnosis")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--no-emulator", action="store_true", help="skip the game check (no ROM needed by the study)")
    add("smoke", cmd_smoke, "server: functional test, a few minutes")
    add("bench", cmd_bench, "server: throughput and memory of one process")
    p = add("compare", cmd_compare, "server: NVIDIA vs the ROCm reference")
    p.add_argument("what", choices=("platform", "common-stream", "harness"))
    p.add_argument("--graph", default="connectome")
    p.add_argument("--prefix", default="", help="e.g. 'expert-' or 'perm-' (common-stream file prefix)")
    p.add_argument("--file", help="harness: the rerun's output")
    p.add_argument("--ref", default="validation-20261009.json")
    p = add("run", cmd_run, "server: start one declared experiment, detached")
    p.add_argument("experiment")
    p.add_argument("--param", action="append", help="key=value, e.g. graph=nulllat:0")
    p.add_argument("--max-hours", type=float)
    p.add_argument("--resume", action="store_true", help="continue a resumable experiment whose outputs exist")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    p.add_argument("--allow-parallel", action="store_true")
    p.add_argument("--no-tmux", action="store_true")
    p.add_argument("--print-id", action="store_true", help="print only the run id (for scripts)")
    p = add("_supervise", cmd_supervise, argparse.SUPPRESS)
    p.add_argument("run_dir")
    add("status", cmd_status, "server: runs and GPU use")
    p = add("summary", cmd_summary, "server: digest of the runs' progress (for the Claude app)")
    p.add_argument("-n", type=int, default=10, help="newest runs shown")
    p.add_argument("--json", action="store_true")
    p = add("wait", cmd_wait, "server: block until a run ends; exit with its code")
    p.add_argument("run_id", nargs="?")
    p.add_argument("--last", action="store_true")
    p.add_argument("--every", type=float, default=30.0)
    p = add("stop", cmd_stop, "server: ask a run to save and exit")
    p.add_argument("run_id", nargs="?")
    p.add_argument("--all", action="store_true")
    p = add("logs", cmd_logs, "server: a run's log")
    p.add_argument("run_id")
    p.add_argument("-n", type=int, default=40)
    p.add_argument("-f", "--follow", action="store_true")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
