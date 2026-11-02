#!/usr/bin/env python
"""Restore probe (dossier §11): a copy kept only on the server is not a backup.

    restore_probe.py [--graph nulllat:100]   -> artifacts/revision48/probe/restore_probe.json (exit 1 on failure)

1 a technical run on a development graph and block (2 learners) stops at decision 250 with its resume state;
2 SHA256SUMS of its directory; 3 one off-instance backup (deployment/backup_remote.sh --once);
4 the directory downloaded from the Drive into a separate place; 5 every file's sha256 checked against SHA256SUMS;
6 the downloaded copy resumed to 500 decisions; 7 a continuous run to 500 on the same platform: the weight digests at
500 must be equal. No analysed graph and no score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts/revision48"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from flymario.experience_commune import learn
    from flymario.experience_commune.learn import Learner
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="nulllat:100")
    ap.add_argument("--slow", action="store_true")
    a = ap.parse_args()
    c = yaml.safe_load((ROOT / "configs/revision48/campaign.yaml").read_text())
    dev = c["blocks"]["dev"][0]
    ls = [Learner(t, "U", dev) for t in c["tasks"]]
    base = ART / "probe"
    shutil.rmtree(base, ignore_errors=True)
    run_dir, cont_dir = base / "run", base / "continuous"
    kw = {"banks_final": (), "eval_cols": 64, "slow": a.slow}
    learn.run(a.graph, ls, ART / "data", run_dir, stop_at=250, **kw)
    files = sorted(p for p in run_dir.iterdir() if p.is_file())
    (run_dir / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files))
    rec: dict = {"graph": a.graph, "files": [p.name for p in files], "resume_state_present": (run_dir / "resume.pt").exists()}
    b = subprocess.run(["bash", str(ROOT / "deployment/backup_remote.sh"), "--once"], cwd=ROOT, capture_output=True,
                       text=True, check=False)
    rec["backup_exit"] = b.returncode
    remote = os.environ.get("BACKUP_REMOTE", "smbbackup:SMB_Flywire_Speedrun_backups")
    dest = f"{remote}/{os.environ.get('CONTAINER_LABEL', socket.gethostname())}/artifacts/revision48/probe/run"
    restored = Path("/tmp/r48_restore_probe/run")
    shutil.rmtree(restored.parent, ignore_errors=True)
    d = subprocess.run(["rclone", "copy", dest, str(restored)], capture_output=True, text=True, check=False)
    rec["download_exit"] = d.returncode
    sums = {}
    if (restored / "SHA256SUMS").exists():
        for line in (restored / "SHA256SUMS").read_text().splitlines():
            h, name = line.split("  ", 1)
            sums[name] = h
    rec["downloaded_files_verified"] = bool(sums) and all((restored / n).exists() and sha(restored / n) == h
                                                           for n, h in sums.items())
    res = learn.run(a.graph, ls, ART / "data", restored, stop_at=500, **kw) if rec["downloaded_files_verified"] else None
    cont = learn.run(a.graph, ls, ART / "data", cont_dir, stop_at=500, **kw)
    rec["resumed_from"] = res["resumed_from"] if res else None
    rec["digests_equal_at_500"] = bool(res and res["weight_digests"]["500"] == cont["weight_digests"]["500"])
    rec["pass"] = bool(rec["backup_exit"] == 0 and rec["download_exit"] == 0 and rec["downloaded_files_verified"]
                       and rec["resume_state_present"] and rec["resumed_from"] == 250 and rec["digests_equal_at_500"])
    (base / "restore_probe.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
