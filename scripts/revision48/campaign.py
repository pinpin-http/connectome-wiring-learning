#!/usr/bin/env python
"""REVISION48 campaign runner: the frozen jobs, several GPU processes in parallel, resumable, with activity reports.

    campaign.py [--workers N] [--dry-run] [--phases tuning,reference,...]

Refuses unless configs/revision48/campaign.yaml has protocol_status: frozen and execution.rental_start_utc set.
Jobs: 63 tuning batches, then one 12-learner run per graph per phase; only tuned waits for the sealed selection.
After every state change writes artifacts/revision48/ACTIVITY.md and activity.json (never a score), copies them
to the Drive, and notifies via ntfy. artifacts/revision48/STOP stops new starts; no job starts past the deadline;
a failed job is retried once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import queue
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/revision48"
PY = str(ROOT / ".venv/bin/python")
ORDER = ("tuning", "reference", "tuned", "supervised", "lognorm", "autapse")   # configs execution.order


def now() -> datetime:
    return datetime.now(UTC)


def iso(t: datetime | None = None) -> str:
    return (t or now()).isoformat(timespec="seconds")


def tag(spec: str) -> str:
    return spec.replace(":", "")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def jobs(c: dict) -> list[dict]:
    man = json.loads((ROOT / c["graphs"]["manifest"]).read_text())
    analysed = man["order"]
    # the autapse condition's connectome is the reference run (same numerical plan); only its nulls run here
    autapse = [g for pair in zip(c["graphs"]["autapse"]["S"], c["graphs"]["autapse"]["E"]) for g in pair]
    gammas = sorted({t["gamma"] for t in c["tuning"]["trials"]}, reverse=True)
    out = []
    for g in analysed:
        for gm in gammas:
            out.append({"phase": "tuning", "graph": g, "key": f"tuning:{g}:g{gm}",
                        "dir": ART / "runs/tuning" / f"g{gm}" / tag(g), "cmd": ["tune", "--graph", g, "--gamma", str(gm)]})
    for cond in ORDER[1:]:
        for g in (autapse if cond == "autapse" else analysed):
            out.append({"phase": cond, "graph": g, "key": f"{cond}:{g}", "dir": ART / "runs/main" / cond / tag(g),
                        "cmd": ["run", "--graph", g, "--condition", cond]})
    return out


def done(j: dict) -> bool:
    if (j["dir"] / "reuse.json").exists():
        return True
    rj = j["dir"] / "run.json"
    return rj.exists() and bool(json.loads(rj.read_text()).get("complete"))


def checksum_dir(d: Path) -> None:
    files = sorted(p for p in d.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "resume.pt")
                   and not p.name.endswith(".part"))
    tmp = d / "SHA256SUMS.part"
    tmp.write_text("".join(f"{sha(p)}  {p.name}\n" for p in files))
    os.replace(tmp, d / "SHA256SUMS")


def drive_dest() -> str:
    remote = os.environ.get("BACKUP_REMOTE", "smbbackup:SMB_Flywire_Speedrun_backups")
    return f"{remote}/{os.environ.get('CONTAINER_LABEL', socket.gethostname())}"


PUSHQ: queue.Queue = queue.Queue()


BACKUPS: list[tuple[str, dict]] = []                     # (phase, result) from the background thread


def _pusher() -> None:
    while True:
        work = PUSHQ.get()
        try:
            work()
        finally:
            PUSHQ.task_done()


def push_later(paths: list[Path]) -> None:
    """Drive copies run in one background thread, in order, so the GPU never waits for them."""
    PUSHQ.put(lambda: push(paths))


def backup_later(phase: str) -> None:
    def work() -> None:
        b = subprocess.run(["bash", str(ROOT / "deployment/backup_remote.sh"), "--once"], cwd=ROOT,
                           capture_output=True, text=True, check=False)
        BACKUPS.append((phase, {"utc": iso(), "exit": b.returncode}))
    PUSHQ.put(work)


def push(paths: list[Path]) -> bool:
    """Copy files or directories (relative to ROOT) to the same place on the Drive; nothing is deleted there."""
    ok = True
    for p in paths:
        if not p.exists():
            continue
        rel = p.relative_to(ROOT)
        if p.is_dir():
            args = ["rclone", "copy", str(p), f"{drive_dest()}/{rel}", "--exclude", "resume.pt", "--exclude", "*.part"]
        else:
            args = ["rclone", "copyto", str(p), f"{drive_dest()}/{rel}"]
        try:
            r = subprocess.run(args + ["--retries", "3", "--tpslimit", "6", "--log-level", "ERROR"],
                               capture_output=True, text=True, check=False, timeout=900)
            ok = ok and r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
    return ok


def notify(title: str, msg: str, priority: str = "default") -> None:
    try:
        topic = Path(os.environ.get("NOTIFY_TOPIC_FILE", "/root/.config/smb-notify/topic")).read_text().strip()
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode(), method="POST",
                                     headers={"Title": title.encode("utf-8").decode("latin-1", "ignore"),
                                              "Priority": priority, "Tags": "bar_chart"})
        urllib.request.urlopen(req, timeout=20).read()
    except OSError:
        pass


def gpu() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    try:
        u, m, t = (x.strip() for x in r.stdout.splitlines()[0].split(","))
        return f"GPU {u} %, VRAM {float(m) / 1024:.1f}/{float(t) / 1024:.0f} Go"
    except (IndexError, ValueError):
        return "GPU n/a"


def cost() -> str:
    try:
        c = json.loads((ROOT / "artifacts/monitor/state.json").read_text())["cost"]
        return f"{c['cost_so_far']} $ en {c['hours_so_far']} h ({c['rate_per_hour']} $/h)"
    except (OSError, KeyError, json.JSONDecodeError):
        return "n/a"


def activity(J: list[dict], st: dict, running: dict, workers: int, dur: dict, deadline: datetime) -> str:
    """ACTIVITY.md (in French) and activity.json: counts, running jobs, times; never a score."""
    by = {p: {"total": 0, "done": 0, "running": 0, "failed": 0} for p in ORDER}
    failed = {f["job"] for f in st["failed"]}
    for j in J:
        b = by[j["phase"]]
        b["total"] += 1
        b["done"] += done(j)
        b["running"] += j["key"] in running
        b["failed"] += j["key"] in failed
    n_done = sum(b["done"] for b in by.values())
    n_total = sum(b["total"] for b in by.values())
    el_h = (now() - datetime.fromisoformat(st["started_utc"])).total_seconds() / 3600
    done_now = len(st["done_log"])
    rate = done_now / el_h if el_h > 0.05 and done_now else None
    left_s = sum(dur[j["phase"]] for j in J if not done(j)) / max(1, workers)
    proj_end = now() + timedelta(seconds=left_s)
    data = {"utc": iso(), "workers": workers, "done": n_done, "total": n_total, "by_phase": by,
            "running": {k: v["since"] for k, v in running.items()}, "failed": st["failed"], "retried": st["retried"],
            "skipped_deadline": st["skipped_deadline"], "jobs_per_hour": rate, "projected_end_utc": iso(proj_end),
            "deadline_utc": iso(deadline), "gpu": gpu(), "cost": cost(), "stopped": st.get("stopped_utc")}
    lines = [f"# REVISION48 — activité ({iso()} UTC)", "",
             (f"**{n_done} / {n_total} travaux terminés**, {len(running)} en cours ({workers} en parallèle), "
              f"{len(failed)} en échec. Fin projetée {iso(proj_end)} UTC (banc de vitesse divisé par {workers}, à "
              f"confirmer par le débit réel) ; dernier lancement possible {iso(deadline)} UTC."), "",
             f"{gpu()} · coût {cost()}" + (f" · {rate:.1f} travaux/h" if rate else ""), "",
             "| phase | terminés | en cours | échecs | total |", "|---|---|---|---|---|"]
    lines += [f"| {p} | {b['done']} | {b['running']} | {b['failed']} | {b['total']} |" for p, b in by.items()]
    lines += ["", "En cours : " + (", ".join(f"{k} (depuis {v['since'][11:16]})" for k, v in running.items()) or "aucun")]
    if st["failed"] or st["retried"] or st["skipped_deadline"]:
        lines += ["", (f"Relancés : {len(st['retried'])} · échecs définitifs : {[f['job'] for f in st['failed']]} · "
                       f"non lancés (échéance) : {st['skipped_deadline']}")]
    lines += ["", "Derniers terminés : " + (", ".join(x["job"] for x in st["done_log"][-8:]) or "aucun"),
              "", "Aucun score n'est affiché ici : la lecture déclarée se fait une fois tous les travaux terminés."]
    for name, text in (("ACTIVITY.md", "\n".join(lines) + "\n"), ("activity.json", json.dumps(data, indent=1))):
        p = ART / name
        tmp = p.with_name(name + ".part")
        tmp.write_text(text)
        os.replace(tmp, p)
    return f"{n_done}/{n_total} terminés, {len(running)} en cours, {len(failed)} échecs ; fin projetée {iso(proj_end)[11:16]} UTC"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and projected hours; run nothing")
    ap.add_argument("--phases", default="")
    ap.add_argument("--notify-every", type=float, default=3.0, help="hours between activity notifications")
    a = ap.parse_args()
    c = yaml.safe_load((ROOT / "configs/revision48/campaign.yaml").read_text())
    ex = c["execution"]
    gp = ex.get("gpu_processes")
    workers = int(a.workers) if a.workers else (int(gp) if isinstance(gp, int) else 0)
    J = jobs(c)
    if a.phases:
        keep = set(a.phases.split(","))
        J = [j for j in J if j["phase"] in keep]
    J.sort(key=lambda j: ORDER.index(j["phase"]))
    bench_p = ART / "checks/benchmark.json"
    bench = json.loads(bench_p.read_text())["5_benchmark"] if bench_p.exists() else None
    dur = None
    if bench:                                     # seconds per job by phase; lognorm has its own operator's benchmark
        dur = {p: (bench["projected_tuning_batch_s"] if p == "tuning" else bench["projected_main_run_s"]) for p in ORDER}
        lb = ART / "checks/benchmark_lognorm.json"
        if lb.exists():
            dur["lognorm"] = json.loads(lb.read_text())["projected_main_run_s"]
    if a.dry_run:
        todo = [j for j in J if not done(j)]
        by: dict[str, int] = {}
        for j in todo:
            by[j["phase"]] = by.get(j["phase"], 0) + 1
        gpu_h = None if dur is None else sum(dur[j["phase"]] for j in todo) / 3600
        print(json.dumps({"jobs_total": len(J), "jobs_to_do": by, "gpu_hours_one_process": gpu_h,
                          "note": "wall time with N workers is gpu_hours / N only if concurrency scales; measure it",
                          "benchmark": str(bench_p) if bench else "missing (checks.py --bench-only)"}, indent=1))
        return 0
    if c.get("protocol_status") != "frozen":
        print("refused: protocol_status is not frozen (docs/revision48/PROTOCOL_FROZEN.md)")
        return 2
    start = ex.get("rental_start_utc")
    if not start or start == "TO_FREEZE":
        print("refused: execution.rental_start_utc is not set")
        return 2
    if dur is None:
        print("refused: no benchmark (scripts/revision48/checks.py --bench-only)")
        return 2
    if workers < 1:
        print("refused: execution.gpu_processes is not frozen (or pass --workers)")
        return 2
    deadline = (datetime.fromisoformat(str(start)) + timedelta(hours=float(c["max_rental_hours"]))
                - timedelta(hours=float(ex["final_reserve_hours"])))
    st_p = ART / "campaign_status.json"
    prev = json.loads(st_p.read_text()) if st_p.exists() else {}
    st = {"started_utc": iso(), "deadline_utc": iso(deadline), "workers": workers, "done_log": prev.get("done_log", []),
          "failed": [], "retried": prev.get("retried", []), "skipped_deadline": [],
          "phase_backups": prev.get("phase_backups", {}), "running": {}}
    for k in ("selection_sealed", "skipped_modules"):              # kept across a restart of the runner
        if k in prev:
            st[k] = prev[k]
    if prev.get("started_utc"):
        st["restarts"] = prev.get("restarts", []) + [{"previous_start_utc": prev["started_utc"], "utc": iso()}]
    threading.Thread(target=_pusher, daemon=True).start()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    running: dict[str, dict] = {}
    tries: dict[str, int] = {}
    phases_done = {p for p in ORDER if any(j["phase"] == p for j in J) and all(done(j) for j in J if j["phase"] == p)}
    last_note = time.time()

    def save() -> str:
        st["updated_utc"] = iso()
        st["running"] = {k: {"since": v["since"], "try": v["try"]} for k, v in running.items()}
        st["n_done"], st["n_total"] = sum(1 for j in J if done(j)), len(J)
        tmp = st_p.with_name(st_p.name + ".part")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, st_p)
        summary = activity(J, st, running, workers, dur, deadline)
        push_later([ART / "ACTIVITY.md", ART / "activity.json", st_p])
        return summary

    # phases already begun (a run complete or resumable) are not re-checked as a whole on a restart
    started = {p for p in ORDER if any(done(j) or (j["dir"] / "resume.pt").exists() for j in J if j["phase"] == p)}

    def sealed() -> bool:
        return (ART / "tuning/selection.json").exists()

    def ready(j) -> bool:
        if done(j) or j["key"] in running or j["key"] in {f["job"] for f in st["failed"]}:
            return False
        if j["key"] in st["skipped_deadline"]:
            return False
        return j["phase"] != "tuned" or sealed()

    notify("REVISION48 : campagne lancée", save() + f" ; {workers} en parallèle.")
    while True:
        for key, r in list(running.items()):                      # finished children
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["log"].close()
            del running[key]
            j = r["job"]
            if done(j):
                if (j["dir"] / "run.json").exists():
                    checksum_dir(j["dir"])
                st["done_log"].append({"job": key, "utc": iso(), "minutes": round((time.time() - r["t0"]) / 60, 1)})
                push_later([j["dir"]])
            elif (ART / "STOP").exists():
                pass                                              # saved at a checkpoint; resumes on the next start
            elif tries.get(key, 0) < 2:
                st["retried"].append({"job": key, "rc": rc, "utc": iso()})
            else:
                st["failed"].append({"job": key, "rc": rc, "utc": iso()})
                notify("REVISION48 : échec", f"{key} a échoué deux fois (code {rc}) ; les autres travaux continuent.",
                       "high")
            summary = save()
            for p in ORDER:                                       # phase transitions
                pj = [x for x in J if x["phase"] == p]
                if pj and p not in phases_done and all(done(x) for x in pj):
                    phases_done.add(p)
                    if p == "tuning" and not sealed():
                        s = subprocess.run([PY, str(ROOT / "scripts/revision48/r48.py"), "select", "--seal"], cwd=ROOT,
                                           capture_output=True, text=True, check=False)
                        st["selection_sealed"] = {"exit": s.returncode, "utc": iso()}
                        push_later([ART / "tuning"])
                    backup_later(p)                               # recorded and notified when it completes
        while BACKUPS:
            p, res = BACKUPS.pop(0)
            st["phase_backups"][p] = res
            notify(f"REVISION48 : phase {p} terminée", save())
        if (ART / "STOP").exists():
            if not running:
                st["stopped_utc"] = iso()
                notify("REVISION48 : arrêt demandé", save())
                PUSHQ.join()
                return 0
        else:
            for j in J:
                if len(running) >= workers:
                    break
                if not ready(j):
                    continue
                if j["phase"] != "tuning" and j["phase"] not in started:
                    # a module starts only if all of it fits before the deadline (wall time per run: observed mean
                    # of this launch's main runs x the module's benchmark ratio, never below its benchmark)
                    rest = [x for x in J if x["phase"] == j["phase"] and not done(x)]
                    obs = [x["minutes"] * 60 for x in st["done_log"] if x["job"].split(":")[0] in ("reference", "supervised", "autapse")]
                    per = dur[j["phase"]]
                    if obs:
                        per = max(per, sum(obs) / len(obs) * dur[j["phase"]] / dur["reference"])
                    if now() + timedelta(seconds=per * len(rest) / workers) > deadline:
                        st["skipped_deadline"] += [x["key"] for x in rest]
                        st.setdefault("skipped_modules", []).append({"phase": j["phase"], "utc": iso(),
                                                                     "runs": len(rest), "s_per_run": round(per)})
                        notify("REVISION48 : module non démarré", f"{j['phase']} ({len(rest)} runs) ne tient plus "
                               "avant l'échéance ; il n'est pas commencé.", "high")
                        save()
                        continue
                    started.add(j["phase"])
                need = dur[j["phase"]]
                if now() + timedelta(seconds=need) > deadline:
                    st["skipped_deadline"].append(j["key"])
                    save()
                    continue
                tries[j["key"]] = tries.get(j["key"], 0) + 1
                suffix = "_" + j["key"].split(":")[-1] if j["phase"] == "tuning" else ""
                log = open(ART / "logs" / f"{j['phase']}_{tag(j['graph'])}{suffix}.log", "a")  # noqa: SIM115
                proc = subprocess.Popen([PY, str(ROOT / "scripts/revision48/r48.py"), *j["cmd"]], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
                running[j["key"]] = {"proc": proc, "log": log, "job": j, "since": iso(), "t0": time.time(),
                                     "try": tries[j["key"]]}
                save()
        if not running and not any(ready(j) for j in J):
            break
        if time.time() - last_note > a.notify_every * 3600:
            notify("REVISION48 : point d'activité", save(), "low")
            last_note = time.time()
        time.sleep(20)
    st["finished_utc"] = iso()
    summary = save()
    PUSHQ.join()
    for p, res in BACKUPS:
        st["phase_backups"][p] = res
    b = subprocess.run(["bash", str(ROOT / "deployment/backup_remote.sh"), "--once"], cwd=ROOT, capture_output=True,
                       text=True, check=False)
    st["phase_backups"]["end"] = {"utc": iso(), "exit": b.returncode}
    save()
    PUSHQ.join()
    notify("REVISION48 : campagne terminée", summary, "high")
    return 0 if not st["failed"] and not st["skipped_deadline"] else 1


if __name__ == "__main__":
    sys.exit(main())
